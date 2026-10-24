//! Border texels through a swizzled texture, ported from `runtime_lane::border_swizzles`.
//!
//! Metal's answer, as the lane fitted it over 1,056 border values on an **Apple M4**: a border texel
//! is the border colour in the format's channels — a missing red, green or blue read as 0 and a
//! missing alpha as 1 — and then the texture's swizzle, exactly as for any texel. `A8Unorm`'s one
//! channel takes the border colour's *red*, so its opaque-black border reads alpha 0 just as its
//! transparent one does. And a **runtime** sampler of clamp-to-zero reads its own border colour,
//! which is what clamp-to-border does, rather than reading zero; the fixture's two **constexpr**
//! `address::clamp_to_zero` samplers carry no border colour and do read zero, in the format's
//! channels and through the swizzle.
//!
//! **The port re-measures it on this M3 Max**, which matters more here than in most of the batch:
//! the rule was fitted on another chip, and the one thing the lane cannot tell us is whether this
//! chip agrees. Record mode runs the case, each recording holds Metal's own words, and the rule is a
//! committed cross-check (the test `every_recorded_texel_is_the_lanes_rule`).
//!
//! **One case per format**, eight of them, each sweeping the six swizzles against the ten samplers
//! and the three coordinates — one inside at a texel centre, and one a texel past each edge, where
//! nearest and linear filtering read only border texels. A venue that holds no texture of a format,
//! or no sampler of a border colour, answers its typed refusal there and counts as lacking the
//! feature instead of leaving the sweep green with nothing compared, which is what the lane does
//! when it prints `BORDER-REFUSED` and moves on.
//!
//! The owned fixture `a2_clamp_to_zero` samples a float texture through two constexpr
//! clamp-to-zero samplers (nearest and linear) and through the runtime sampler at `[[sampler(0)]]`,
//! and also samples a uint and an int texture, five `uint4`s a lane. The case keeps all five, so
//! the integer textures' reads are recorded beside the float texture's rather than dropped.
//!
//! **One difference from the lane, stated rather than hidden:** the lane sets the runtime sampler's
//! `r` address mode beside `t`. A 2D sample reads no `r` coordinate, so that mode cannot change any
//! texel here, and [`crate::case::SamplerSpec`] — a core type other streams share — states `s` and
//! `t` alone. The case therefore states `s` and `t`, and a case that ever samples a 3D or cube
//! texture through such a sampler is what should grow the field.
//!
//! Nothing here can fault a GPU: one compute pass a sampler, three read-only textures, three
//! threads, no loop.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{
    call, made, Abort, Bindings, Case, Failure, Input, MetalSource, SamplerSpec, Side,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::texture::{TextureRequest, Upload};
use crate::transcript::{ByteLayout, Digest, Element, Node, Observed, Sites, Value};
use metal2vulkan::metal::{
    PixelFormat, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter, TextureSwizzle,
    TextureSwizzleChannels,
};
use metal2vulkan_runtime::Degradation;

/// The fixture the case runs, whose entry point is its own name.
const FIXTURE: &str = "a2_clamp_to_zero";

/// The bound `docs/FAST-PATHS.md` states for `texel_conversion = TexelConversionPolicy::Device`,
/// one `f32` ulp for 8- and 16-bit codes, as these sites hold it: a word of the float's bits, where
/// one ulp of a positive finite float is a difference of one in the word.
const LEDGERS_TEXEL_CONVERSION_ULPS_AS_WORDS: f64 = 1.0;

/// Each texture's side: 4 by 4 texels, every texel the same value, so a sample inside reads the
/// format's texel and a sample outside reads only border texels.
const SIDE: u32 = 4;

/// The coordinates each dispatch samples: a texel centre inside, and points at least one texel
/// past the left and right edges, where linear filtering's footprint holds only border texels.
const COORDS: [[f32; 2]; 3] = [[0.375, 0.375], [-0.5, 0.375], [1.5, 0.375]];

/// The `uint4`s the kernel writes a lane: the two constexpr samplers, the runtime sampler, then the
/// uint and int textures.
const PER_LANE: usize = 5;

/// The byte every output word holds beforehand, so a word the kernel did not write is a recorded
/// value rather than whatever was there.
const BLANK: u8 = 0xab;

/// A float texture format the case sweeps: which of red, green, blue and alpha it has, the channel
/// of the border colour each of them takes on Metal, and the bytes of its one texel value.
///
/// The table is public because it is the family's description, and what reads `present` and
/// `border_source` is the cross-check that holds the recordings against the lane's rule: the case
/// itself needs only the format and its texel.
pub struct Format {
    pub name: &'static str,
    pub format: PixelFormat,
    /// Which of red, green, blue and alpha the format presents.
    pub present: [bool; 4],
    /// Which channel of the border colour each channel reads: its own, but for `A8Unorm`, whose one
    /// channel takes red.
    pub border_source: [usize; 4],
    /// The bytes of the one texel value every texel holds.
    pub texel: &'static [u8],
    /// Whether a sample inside reads the unorm codes `0x40`, `0x80`, `0xc0` and `0x20` over 255;
    /// a half-float format reads 0.25, 0.5, 0.75 and 0.125 instead, which the texel bytes are.
    pub unorm: bool,
}

/// Each channel's border colour channel, its own.
const OWN: [usize; 4] = [0, 1, 2, 3];

/// Half-float bits of 0.25, 0.5, 0.75 and 0.125, little-endian.
const R16: &[u8] = &[0x00, 0x34];
const RG16: &[u8] = &[0x00, 0x34, 0x00, 0x38];
const RGBA16: &[u8] = &[0x00, 0x34, 0x00, 0x38, 0x00, 0x3a, 0x00, 0x30];

/// The lane's eight formats, with its texel bytes.
pub const FORMATS: [Format; 8] = [
    Format {
        name: "border-swizzles-r8unorm",
        format: PixelFormat::R8Unorm,
        present: [true, false, false, false],
        border_source: OWN,
        texel: &[0x40],
        unorm: true,
    },
    Format {
        name: "border-swizzles-rg8unorm",
        format: PixelFormat::RG8Unorm,
        present: [true, true, false, false],
        border_source: OWN,
        texel: &[0x40, 0x80],
        unorm: true,
    },
    Format {
        name: "border-swizzles-rgba8unorm",
        format: PixelFormat::RGBA8Unorm,
        present: [true; 4],
        border_source: OWN,
        texel: &[0x40, 0x80, 0xc0, 0x20],
        unorm: true,
    },
    Format {
        name: "border-swizzles-bgra8unorm",
        format: PixelFormat::BGRA8Unorm,
        present: [true; 4],
        border_source: OWN,
        texel: &[0xc0, 0x80, 0x40, 0x20],
        unorm: true,
    },
    Format {
        name: "border-swizzles-a8unorm",
        format: PixelFormat::A8Unorm,
        present: [false, false, false, true],
        border_source: [0, 1, 2, 0],
        texel: &[0x20],
        unorm: true,
    },
    Format {
        name: "border-swizzles-r16float",
        format: PixelFormat::R16Float,
        present: [true, false, false, false],
        border_source: OWN,
        texel: R16,
        unorm: false,
    },
    Format {
        name: "border-swizzles-rg16float",
        format: PixelFormat::RG16Float,
        present: [true, true, false, false],
        border_source: OWN,
        texel: RG16,
        unorm: false,
    },
    Format {
        name: "border-swizzles-rgba16float",
        format: PixelFormat::RGBA16Float,
        present: [true; 4],
        border_source: OWN,
        texel: RGBA16,
        unorm: false,
    },
];

/// The swizzles, by name: each output channel's source.
fn swizzles() -> [(&'static str, TextureSwizzleChannels); 6] {
    use TextureSwizzle::{Alpha as A, Blue as B, Green as G, One as O, Red as R, Zero as Z};
    [
        ("rgba", TextureSwizzleChannels::DEFAULT),
        ("rrr1", TextureSwizzleChannels::new(R, R, R, O)),
        ("argb", TextureSwizzleChannels::new(A, R, G, B)),
        ("01rg", TextureSwizzleChannels::new(Z, O, R, G)),
        ("bgra", TextureSwizzleChannels::new(B, G, R, A)),
        ("ga0r", TextureSwizzleChannels::new(G, A, Z, R)),
    ]
}

/// The runtime samplers: a name, the address mode of `s` and of `t`, and the border colour. A
/// sample leaves the texture along `s` alone; the mixed ones clamp to zero on one axis and to the
/// border on the other, which plain Metal reads the border colour through on every border axis.
const SAMPLERS: [(&str, [SamplerAddressMode; 2], SamplerBorderColor); 10] = {
    use SamplerAddressMode::{ClampToBorderColor as Border, ClampToZero as Zero};
    use SamplerBorderColor::{OpaqueBlack, OpaqueWhite, TransparentBlack};
    [
        ("transparent", [Border, Border], TransparentBlack),
        ("white", [Border, Border], OpaqueWhite),
        ("zero", [Zero, Zero], TransparentBlack),
        ("zero-white", [Zero, Zero], OpaqueWhite),
        ("zero-black", [Zero, Zero], OpaqueBlack),
        ("black", [Border, Border], OpaqueBlack),
        ("zero-s-border-t-white", [Zero, Border], OpaqueWhite),
        ("border-s-zero-t-white", [Border, Zero], OpaqueWhite),
        ("zero-s-border-t-black", [Zero, Border], OpaqueBlack),
        (
            "zero-s-border-t-transparent",
            [Zero, Border],
            TransparentBlack,
        ),
    ]
};

/// The sampler state the row `(_, [s, t], border)` names: the lane's, which filters nearest and
/// samples in normalized coordinates.
fn sampler_spec(modes: [SamplerAddressMode; 2], border: SamplerBorderColor) -> SamplerSpec {
    let mut spec = SamplerSpec::filtered(SamplerMinMagFilter::Nearest);
    spec.s_address_mode = modes[0];
    spec.t_address_mode = modes[1];
    spec.border_color = border;
    spec
}

/// `SIDE` by `SIDE` texels of `texel`, one upload's bytes.
fn texels(texel: &[u8]) -> Vec<u8> {
    (0..SIDE * SIDE)
        .flat_map(|_| texel.iter().copied())
        .collect()
}

/// The case for one format.
pub struct BorderSwizzles {
    at: usize,
}

/// Every format's case.
pub fn all() -> Vec<BorderSwizzles> {
    (0..FORMATS.len()).map(|at| BorderSwizzles { at }).collect()
}

impl BorderSwizzles {
    /// The format the case sweeps.
    fn format(&self) -> &'static Format {
        &FORMATS[self.at]
    }

    /// The fixture, whose entry point is its own name.
    fn fixture() -> Fixture {
        Fixture::new(FIXTURE)
    }

    /// The uint and int textures every dispatch binds at textures 1 and 2: 4 by 4, every texel 1
    /// and -1, with the default swizzle, as the lane makes them.
    fn integer_texels() -> [(PixelFormat, [u8; 4]); 2] {
        [
            (PixelFormat::R32Uint, 1u32.to_le_bytes()),
            (PixelFormat::R32Sint, (-1i32).to_le_bytes()),
        ]
    }

    /// The bytes the output buffer holds beforehand.
    fn blank() -> Vec<u8> {
        vec![BLANK; COORDS.len() * PER_LANE * 16]
    }

    /// The layout the output words are read under: one row a lane, four `uint` words an element of
    /// its five reads, so a diff names the lane and the word rather than a byte offset.
    fn layout() -> ByteLayout {
        let row = (PER_LANE * 16) as u32;
        ByteLayout {
            region: crate::transcript::Region::Buffer {
                offset: 0,
                length: (COORDS.len() * PER_LANE * 16) as u64,
            },
            element: Element::U32,
            row_offset: 0,
            row_elements: (PER_LANE * 4) as u32,
            row_pitch: row,
            rows: COORDS.len() as u32,
            image_pitch: row * COORDS.len() as u32,
            images: 1,
            gaps: crate::transcript::Gaps::Ignored,
        }
    }

    /// The coordinates the case feeds at buffer 0.
    fn coords() -> Vec<u8> {
        COORDS
            .iter()
            .flatten()
            .flat_map(|axis| axis.to_le_bytes())
            .collect()
    }
}

impl Case for BorderSwizzles {
    fn name(&self) -> &'static str {
        FORMATS[self.at].name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Under the default policy the sample goes to the device's own sampler and the texel's
        // `float` value is the device's conversion of its codes, which the runtime announces as
        // `device-sampling` and `device-texel-conversion`. Metal announces neither, so the set of
        // notices differs, and the two notice allowances cover that site.
        //
        // The sampled words themselves may differ too. These samplers filter nearest, so the
        // ledger's linear-filter row is not what bounds them: the bound is the one
        // `docs/FAST-PATHS.md` states for `texel_conversion = TexelConversionPolicy::Device`, one
        // `f32` ulp for 8- and 16-bit codes, which is what every format here holds. The words are
        // recorded as the `u32` bit patterns of those floats, so one ulp is a difference of one in
        // the word -- `every_recorded_word_is_a_positive_finite_float` pins the range that makes
        // the two the same thing. The bound is the ledger's number: a run that exceeds it is the
        // ledger's claim failing, which is what this allowance exists to catch. Under the exact
        // policy neither notice fires, no justification holds, and every word is compared exactly.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // all 8 rows of the family, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        let mut all = vec![
            crate::replay::shared_memory_notice(),
            crate::replay::notice_allowance(Degradation::DeviceSampling.code()),
            crate::replay::notice_allowance(Degradation::DeviceTexelConversion.code()),
        ];
        for (swizzle, _) in swizzles() {
            for (sampler, _, _) in SAMPLERS {
                all.push(Allowance::new(
                    format!("case/{swizzle}/{sampler}/words"),
                    Loosening::Absolute(LEDGERS_TEXEL_CONVERSION_ULPS_AS_WORDS),
                    Justification::Degradation(
                        Degradation::DeviceTexelConversion.code().to_owned(),
                    ),
                ));
            }
        }
        all
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = Self::fixture();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: FIXTURE.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Self::fixture();
        let format = self.format();
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("buffer:0", &Self::coords()),
            Input::bytes("buffer:1", &Self::blank()),
        ];
        // Every float texture the sweep makes, by its swizzle, and the two integer textures: the
        // whole descriptor and its texels, so a texture whose format, extent, swizzle or value
        // moves is `Unrecorded` rather than compared against Metal's answer for another texture.
        let bytes = texels(format.texel);
        for (name, swizzle) in swizzles() {
            let uploads = [Upload::level_0(
                [SIDE, SIDE],
                SIDE * format.texel.len() as u32,
                SIDE,
                &bytes,
            )];
            let request = TextureRequest::read_2d(format.format.as_raw(), SIDE, SIDE)
                .swizzled(swizzle)
                .loaded(&uploads);
            inputs.push(Input::bytes(format!("texture:{name}"), &request.words()));
        }
        for (at, (format, texel)) in (0..).zip(Self::integer_texels()) {
            let bytes = texels(&texel);
            let uploads = [Upload::level_0([SIDE, SIDE], SIDE * 4, SIDE, &bytes)];
            let request = TextureRequest::read_2d(format.as_raw(), SIDE, SIDE).loaded(&uploads);
            inputs.push(Input::bytes(
                format!("texture:integer/{at}"),
                &request.words(),
            ));
        }
        // Every sampler state the sweep makes, so a sampler that moves is a change to the inputs.
        for (name, modes, border) in SAMPLERS {
            inputs.push(Input::bytes(
                format!("sampler:{name}"),
                &sampler_spec(modes, border).words(),
            ));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded = Self::fixture()
            .loaded()
            .map_err(|why| stop("library", why))?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, loaded.entry),
        )?;
        let coords = made(sites, "buffer/coords", side.buffer(&Self::coords()))?;
        let integer = (0..)
            .zip(Self::integer_texels())
            .map(|(at, (format, texel))| {
                let bytes = texels(&texel);
                let uploads = [Upload::level_0([SIDE, SIDE], SIDE * 4, SIDE, &bytes)];
                let request = TextureRequest::read_2d(format.as_raw(), SIDE, SIDE).loaded(&uploads);
                made(
                    sites,
                    &format!("texture/integer/{at}"),
                    side.texture(&request),
                )
            })
            .collect::<Result<Vec<_>, Abort>>()?;
        let format = self.format();
        let bytes = texels(format.texel);
        for (swizzle_name, swizzle) in swizzles() {
            let uploads = [Upload::level_0(
                [SIDE, SIDE],
                SIDE * format.texel.len() as u32,
                SIDE,
                &bytes,
            )];
            let request = TextureRequest::read_2d(format.format.as_raw(), SIDE, SIDE)
                .swizzled(swizzle)
                .loaded(&uploads);
            let float = made(
                sites,
                &format!("{swizzle_name}/texture"),
                side.texture(&request),
            )?;
            for (sampler_name, modes, border) in SAMPLERS {
                let at = |site: &str| format!("{swizzle_name}/{sampler_name}/{site}");
                let sampler = made(
                    sites,
                    &at("sampler"),
                    side.sampler(&sampler_spec(modes, border)),
                )?;
                let out = made(sites, &at("buffer/out"), side.buffer(&Self::blank()))?;
                let textures = [float, integer[0], integer[1]];
                let command_buffer = made(
                    sites,
                    &at("dispatch"),
                    side.dispatch_threads(
                        pipeline,
                        &Bindings::new(&textures, &[coords, out]).sampled(&[sampler]),
                        &[],
                        [COORDS.len() as u32, 1, 1],
                        [COORDS.len() as u32, 1, 1],
                    ),
                )?;
                call(
                    sites,
                    &at("status"),
                    side.wait(command_buffer),
                    Clone::clone,
                )?;
                let read = call(
                    sites,
                    &at("read-back"),
                    side.read_buffer(out, 0, Self::blank().len()),
                    |_| Value::Absent,
                )?;
                sites.push(
                    Node::branch(format!("{swizzle_name}/{sampler_name}"), Vec::new())
                        .with(Node::leaf("words", Observed::bytes(read, Self::layout()))),
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// What `border` clamps to before the format's channels: the sampler's border colour, under
    /// clamp-to-zero as under clamp-to-border.
    fn border_rgba(border: SamplerBorderColor) -> [f32; 4] {
        match border {
            SamplerBorderColor::OpaqueBlack => [0.0, 0.0, 0.0, 1.0],
            SamplerBorderColor::OpaqueWhite => [1.0; 4],
            _ => [0.0; 4],
        }
    }

    /// Metal's texel of `format` through `swizzle` whose channels are `rgba`'s (picked by
    /// `format.border_source` when `border`), a missing red, green or blue 0 and a missing alpha 1.
    fn rule(
        format: &Format,
        rgba: [f32; 4],
        border: bool,
        swizzle: TextureSwizzleChannels,
    ) -> [f32; 4] {
        let mut value = [0.0, 0.0, 0.0, 1.0];
        for channel in 0..4 {
            if format.present[channel] {
                value[channel] = rgba[if border {
                    format.border_source[channel]
                } else {
                    channel
                }];
            }
        }
        let pick = |source: TextureSwizzle| match source {
            TextureSwizzle::Zero => 0.0,
            TextureSwizzle::One => 1.0,
            TextureSwizzle::Red => value[0],
            TextureSwizzle::Green => value[1],
            TextureSwizzle::Blue => value[2],
            _ => value[3],
        };
        [
            pick(swizzle.red),
            pick(swizzle.green),
            pick(swizzle.blue),
            pick(swizzle.alpha),
        ]
    }

    /// The value a sample inside reads, per channel, of a format's texel.
    fn inside_value(format: &Format) -> [f32; 4] {
        if format.unorm {
            [
                f32::from(0x40u8) / 255.0,
                f32::from(0x80u8) / 255.0,
                f32::from(0xc0u8) / 255.0,
                f32::from(0x20u8) / 255.0,
            ]
        } else {
            [0.25, 0.5, 0.75, 0.125]
        }
    }

    /// The cases name distinct recordings.
    #[test]
    fn the_formats_name_distinct_cases() {
        let named: BTreeSet<&str> = FORMATS.iter().map(|format| format.name).collect();
        assert_eq!(named.len(), FORMATS.len());
        let samplers: BTreeSet<&str> = SAMPLERS.iter().map(|(name, _, _)| *name).collect();
        assert_eq!(samplers.len(), SAMPLERS.len());
        let swizzled: BTreeSet<&str> = swizzles().iter().map(|(name, _)| *name).collect();
        assert_eq!(swizzled.len(), swizzles().len());
    }

    /// The layout covers the whole read-back and no more.
    #[test]
    fn the_layout_covers_the_read_back() {
        assert_eq!(
            BorderSwizzles::layout().extent(),
            BorderSwizzles::blank().len()
        );
    }

    /// Every border texel of every committed recording is the lane's rule's, and so is every texel
    /// a sample inside reads, within one part in a million (a format conversion's rounding, which
    /// this case is not about).
    ///
    /// This is the cross-check the port exists for: the recordings hold what Metal answered on this
    /// M3 Max, and the rule was fitted on an Apple M4. A difference means the chips disagree or the
    /// rule was never Metal's, and either is a finding rather than something to edit away.
    #[test]
    fn every_recorded_texel_is_the_lanes_rule() {
        let mut unrecorded = Vec::new();
        let mut differ = Vec::new();
        let mut checked = 0usize;
        for format in &FORMATS {
            let recording = match crate::recording::read(format.name) {
                Ok(recording) => recording,
                Err(_) => {
                    unrecorded.push(format.name);
                    continue;
                }
            };
            let inside = inside_value(format);
            for answer in &recording.answers {
                for outcome in &answer.outcomes {
                    for (swizzle_name, swizzle) in swizzles() {
                        for (sampler_name, _, border) in SAMPLERS {
                            let site = format!("{swizzle_name}/{sampler_name}");
                            let node = outcome
                                .transcript
                                .root
                                .child(&site)
                                .and_then(|node| node.child("words"));
                            let Some(node) = node else {
                                differ.push(format!("{}: no site {site}", format.name));
                                continue;
                            };
                            let Value::Bytes(bytes) = &node.observed.value else {
                                differ.push(format!("{}/{site}: no bytes", format.name));
                                continue;
                            };
                            let want_inside = rule(format, inside, false, swizzle);
                            // The two constexpr samplers are `address::clamp_to_zero` written in
                            // the shader: they carry no border colour and read zero in the
                            // format's channels. Only the runtime sampler reads the border colour
                            // the case set, and Metal reads it under clamp-to-zero as under
                            // clamp-to-border, which is the lane's finding.
                            let want_zero = rule(format, [0.0; 4], true, swizzle);
                            let want_border = rule(format, border_rgba(border), true, swizzle);
                            for lane in 0..COORDS.len() {
                                // Lane 0 samples inside at a texel centre; the other two sample
                                // past an edge, where nearest and linear read only border texels.
                                for read in 0..3 {
                                    let want = match (lane, read) {
                                        (0, _) => want_inside,
                                        (_, 2) => want_border,
                                        _ => want_zero,
                                    };
                                    for (channel, wanted) in want.iter().enumerate() {
                                        let at = (lane * PER_LANE + read) * 16 + channel * 4;
                                        let word = u32::from_le_bytes(
                                            bytes[at..at + 4].try_into().expect("four bytes"),
                                        );
                                        let got = f32::from_bits(word);
                                        checked += 1;
                                        let same = if lane == 0 {
                                            (got - wanted).abs() <= 1e-6
                                        } else {
                                            word == wanted.to_bits()
                                        };
                                        if !same && differ.len() < 24 {
                                            differ.push(format!(
                                                "{}/{site} lane {lane} read {read} channel \
                                                 {channel}: Metal {got}, the rule's {wanted}",
                                                format.name
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(
            differ.is_empty(),
            "{checked} channels checked; Metal's recorded texels differ from the lane's \
             rule:\n{}",
            differ.join("\n")
        );
        assert!(
            unrecorded.is_empty(),
            "no recording yet for {unrecorded:?}: run `conformance record` on a Mac"
        );
    }

    /// Every word a read of this case writes is the bit pattern of a positive finite `float` no
    /// greater than one.
    ///
    /// That range is what makes this case's bounded allowance mean what it says. The words are
    /// recorded as `u32`, so the allowance bounds `|a - b|` on the words, and over this range a
    /// difference of one in the word is exactly one `f32` ulp -- the bound `docs/FAST-PATHS.md`
    /// states for the device's texel conversion. Outside it (a negative value, an infinity, a NaN)
    /// the two would stop being the same thing, so this test fails rather than letting the
    /// allowance quietly mean something else.
    ///
    /// The lanes' other two words are the blank fill, which neither side writes and both hold
    /// identically, so the allowance's one step over them is a tolerance on a number that does not
    /// move rather than a loosening of a float.
    #[test]
    fn every_recorded_word_is_a_positive_finite_float() {
        let mut unrecorded = Vec::new();
        let mut wrong = Vec::new();
        let mut checked = 0usize;
        for format in &FORMATS {
            let recording = match crate::recording::read(format.name) {
                Ok(recording) => recording,
                Err(_) => {
                    unrecorded.push(format.name);
                    continue;
                }
            };
            for answer in &recording.answers {
                for outcome in &answer.outcomes {
                    for (swizzle_name, _) in swizzles() {
                        for (sampler_name, _, _) in SAMPLERS {
                            let site = format!("{swizzle_name}/{sampler_name}");
                            let node = outcome
                                .transcript
                                .root
                                .child(&site)
                                .and_then(|node| node.child("words"));
                            let Some(node) = node else { continue };
                            let Value::Bytes(bytes) = &node.observed.value else {
                                continue;
                            };
                            for lane in 0..COORDS.len() {
                                for read in 0..3 {
                                    for channel in 0..4 {
                                        let at = (lane * PER_LANE + read) * 16 + channel * 4;
                                        let word = u32::from_le_bytes(
                                            bytes[at..at + 4].try_into().expect("four bytes"),
                                        );
                                        let value = f32::from_bits(word);
                                        checked += 1;
                                        if !(value.is_finite() && (0.0..=1.0).contains(&value))
                                            && wrong.len() < 24
                                        {
                                            wrong.push(format!(
                                                "{}/{site} lane {lane} read {read} channel \
                                                 {channel}: {value} (0x{word:08x})",
                                                format.name
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(
            wrong.is_empty(),
            "{checked} words checked; these are not a positive finite float in [0, 1], so a \
             difference of one in the word is no longer one ulp:\n{}",
            wrong.join("\n")
        );
        assert!(
            unrecorded.is_empty(),
            "no recording yet for {unrecorded:?}: run `conformance record` on a Mac"
        );
    }
}
