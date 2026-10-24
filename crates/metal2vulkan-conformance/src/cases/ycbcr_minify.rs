//! A minified 10-bit two-plane YCbCr texture's samples: where a linear footprint's rounding puts
//! the switch.
//!
//! Ported from `runtime_lane/ycbcr_minify.rs`, which exists because of a consumer bug: a pass that
//! samples a 3840 x 2160 `YCBCR10_420_2P` surface minified into a 1440 x 1080 target wrote rare
//! ±1s in its red, green and blue. The lane sweeps, along each axis, 2,048 places where a
//! footprint's rounding decides which side of a `1 / 256` step a sample reads — 64 places times 32
//! steps, 17 consecutive `f32` coordinates each — and compares the whole band's results with
//! plain Metal's.
//!
//! # Recorded on the oracle, and what the Apple M4 said
//!
//! Metal's answer here used to be **carried** from an Apple M4 under macOS 27.0.1 — a machine the
//! project no longer has, which made these two the weakest evidence in the suite: no living host
//! could re-answer them. It is now recorded on the oracle, because the Metal side lays a texture
//! over a real IOSurface laid out as this case says its stand-in buffer is (`metal_surface.rs`).
//! The runtime still takes the buffer, which is what lets the case run where there is no
//! CoreVideo.
//!
//! The M4's two digests are kept, in the tests' own `APPLE_M4_BANDS`, and
//! `the_recorded_bands_are_the_apple_m4s` holds the recording to them. They are the one
//! cross-chip fact these bands can carry: the band is a sweep of where a linear footprint's
//! rounding puts a switch, and whether two Apple GPU generations put it in the same place is
//! worth an executed check rather than a sentence.
//!
//! # The kernel is not the probe's, and the digests say it does not matter
//!
//! The lane samples through an AIR kernel it generates at run time; a case samples through the
//! committed fixture `fpv_swizzled_reads`' `fpv_swz_sample`. Both do one plain `sample(s, uv)` of
//! the same texture through the same sampler state and write one `float4` a job in job order, so
//! the output bytes are the same program's observable output and the probe's digest is a digest of
//! them. That is an argument, and the cases check it: both bands hash to exactly the probe's
//! numbers, so the substitution is sound rather than assumed. A reader who changes the kernel's
//! output shape breaks both digests at once, which is the right failure.
//!
//! # Why the band is a digest and a window, not 34,816 values
//!
//! A band's 34,816 `float4` results are 544 KiB of values, which is past the suite's 512 KiB
//! ceiling on one recording — base64 would put each of the two files near 750 KiB. So the whole
//! band's site holds the **digest alone**, which is what the probe kept and what the band's
//! question needs, and a second site holds the first and last [`ENDS`] samples whole, with their
//! layout, so a difference at either end of the sweep names the coordinate rather than the band.
//! A difference in the middle still names the band; that is the cost of the ceiling, stated here
//! rather than discovered.
//!
//! # Why it cannot fault a GPU
//!
//! One texture, one sampler, one compute dispatch of 34,816 threads, and no loop. The surface is a
//! 23.7 MiB shared buffer, which is the largest input in the suite and well inside the project's
//! 500 MiB ceiling.

use crate::allowance::Allowance;
use crate::case::{
    made, Abort, Bindings, Case, Failure, Input, MetalSource, PlaneTexture, SamplerSpec, Side,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites};
use metal2vulkan::metal::{PixelFormat, SamplerAddressMode, SamplerMinMagFilter};
use metal2vulkan_runtime::Degradation;

/// The surface's extent in texels: 4K, the size the consumer's pass sampled.
const W: u64 = 3840;
const H: u64 = 2160;

/// Where the surface's planes lie, as CoreVideo lays a 4K `x420` one out: the luma plane at byte
/// 0 with rows as long as its 16-bit codes, the chroma plane right after it.
const LUMA: (u64, u64) = (0, 2 * W);
const CHROMA: (u64, u64) = (2 * W * H, 2 * W);

/// The surface buffer's length: the chroma plane's rows end it.
const LENGTH: usize = (CHROMA.0 + CHROMA.1 * H / 2) as usize;

/// The surface's four-character OSType: an untagged video-range 4:2:0 10-bit surface, which Metal
/// reads as BT.601.
const OSTYPE: [u8; 4] = *b"x420";

/// The format the texture is of.
const FORMAT: PixelFormat = PixelFormat::YCBCR104202P;

/// A band's shape: 64 places along the axis, 32 steps at each, and the consecutive `f32`
/// coordinates taken around each step.
const PLACES: u64 = 64;
const STEPS: u64 = 32;
const GROUP: usize = 17;

/// The samples one band takes.
const SAMPLES: usize = (PLACES * STEPS) as usize * GROUP;

/// The threads in a threadgroup of the sampling dispatch.
const THREADS: u32 = 64;

/// The byte the read-back buffer holds before the dispatch.
const BLANK: u8 = 0xab;

/// The fixture whose kernel samples, and its entry: one `float4` a position, from a buffer of
/// `float2` coordinates, through the sampler state bound at index 0.
const SAMPLE_FIXTURE: &str = "fpv_swizzled_reads";
const SAMPLE_ENTRY: &str = "fpv_swz_sample";

/// The samples recorded whole at each end of the band: enough that a difference where the sweep
/// starts or ends names the coordinate, few enough that the recording stays inside the suite's
/// 512 KiB ceiling.
pub const ENDS: usize = 256;

/// The site a band's digest sits at.
const RESULTS_SITE: &str = "samples";

/// The site the window of whole samples sits at.
const ENDS_SITE: &str = "samples/ends";

/// One case: a band along one axis, and the hash plain Metal answered for it.
pub struct MinifyBand {
    name: &'static str,
    /// 0 along x, 1 along y.
    axis: usize,
}

/// Both bands.
pub fn all() -> Vec<MinifyBand> {
    vec![
        MinifyBand {
            name: "ycbcr-minify-band-x",
            axis: 0,
        },
        MinifyBand {
            name: "ycbcr-minify-band-y",
            axis: 1,
        },
    ]
}

/// `lowbias32`, the generator the probe drew the surface's codes from.
fn mix(value: u32) -> u32 {
    let mut x = value;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// The surface's bytes: luma `(x, y)`'s code `mix(i ^ 0x9e3779b9) & 1023` (`i = y W + x`), and
/// chroma pair `(cx, cy)`'s Cb and Cr from `0x85ebca6b` and `0xc2b2ae35`, each code in the high 10
/// bits of a little-endian 16-bit word; every other byte `0xa5`.
fn surface_bytes() -> Vec<u8> {
    let mut bytes = vec![0xa5; LENGTH];
    let mut put = |at: u64, code: u32| {
        let at = at as usize;
        bytes[at..at + 2].copy_from_slice(&(((code & 1023) as u16) << 6).to_le_bytes());
    };
    for y in 0..H {
        for x in 0..W {
            put(
                LUMA.0 + y * LUMA.1 + 2 * x,
                mix((y * W + x) as u32 ^ 0x9e37_79b9),
            );
        }
    }
    for cy in 0..H / 2 {
        for cx in 0..W / 2 {
            let j = (cy * W / 2 + cx) as u32;
            let at = CHROMA.0 + cy * CHROMA.1 + 4 * cx;
            put(at, mix(j ^ 0x85eb_ca6b));
            put(at + 2, mix(j ^ 0xc2b2_ae35));
        }
    }
    bytes
}

/// The band along `axis`, as the probe drew it: for place `m` of [`PLACES`] (`7 + 59m` along x,
/// `7 + 33m` along y) and step `t` of [`STEPS`], the `f32` nearest the normalized coordinate of
/// the step `p + 1/2 + (4 + 8t + 1/2) / 256` and eight either side of it, the other axis at a
/// fixed coordinate of its own. The coordinates, as the `float2`s the kernel reads.
fn band(axis: usize) -> Vec<u8> {
    let (n, other_n) = if axis == 0 { (W, H) } else { (H, W) };
    let mut bytes = Vec::with_capacity(SAMPLES * 8);
    for m in 0..PLACES {
        let (place, fixed) = if axis == 0 {
            (7 + 59 * m, 13 + 33 * m)
        } else {
            (7 + 33 * m, 13 + 59 * m)
        };
        let other = ((fixed as f64 + 0.5 + 100.0 / 256.0) / other_n as f64) as f32;
        for t in 0..STEPS {
            let step = place as f64 + 0.5 + ((4 + 8 * t) as f64 + 0.5) / 256.0;
            let mut u = (step / n as f64) as f32;
            for _ in 0..8 {
                u = u.next_down();
            }
            for _ in 0..GROUP {
                let pair = if axis == 0 { [u, other] } else { [other, u] };
                bytes.extend(pair.iter().flat_map(|axis| axis.to_le_bytes()));
                u = u.next_up();
            }
        }
    }
    bytes
}

/// The sampler the probe sampled through: linear on both filters, clamp to edge on both axes,
/// normalized coordinates, no mip filter (the texture has one level).
fn sampler_spec() -> SamplerSpec {
    SamplerSpec::filtered(SamplerMinMagFilter::Linear).addressed(SamplerAddressMode::ClampToEdge)
}

/// The layout a band's results are read under: one row a sample of four lanes, so the day the
/// values can be recorded a diff names the coordinate rather than a byte offset.
fn layout() -> ByteLayout {
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: (SAMPLES * 16) as u64,
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: 4,
        row_pitch: 16,
        rows: SAMPLES as u32,
        image_pitch: (SAMPLES * 16) as u32,
        images: 1,
        // A sample's four lanes fill its row, so there is nothing between them.
        gaps: Gaps::Ignored,
    }
}

/// The first and last [`ENDS`] samples of `bytes`, in that order: the window the recording holds
/// whole.
fn ends(bytes: &[u8]) -> Vec<u8> {
    let window = ENDS * 16;
    let mut out = Vec::with_capacity(2 * window);
    out.extend_from_slice(&bytes[..window]);
    out.extend_from_slice(&bytes[bytes.len() - window..]);
    out
}

/// The layout the window is read under: one row a sample of four lanes, as [`layout`] reads the
/// whole band, so a diff names the sample inside the window.
fn ends_layout() -> ByteLayout {
    let mut layout = layout();
    let extent = (2 * ENDS * 16) as u64;
    layout.region = Region::Buffer {
        offset: 0,
        length: extent,
    };
    layout.rows = (2 * ENDS) as u32;
    layout.image_pitch = extent as u32;
    layout
}

/// The degrade notices one of these cases may announce. A list rather than a device query,
/// because a case never branches on the device: it runs and reads what came back, and a venue that
/// announces a kind not here fails the case until the kind is added.
///
/// Declaring a kind states what our side announced at the notices site and loosens no value:
/// every sample of every band still compares exactly.
const PASS_NOTICES: &[Degradation] = &[
    // The buffer each band writes its minified samples into is `Shared`.
    // A device with no unified memory keeps a `Shared` allocation in system memory because that is
    // the only memory both sides can reach, so this notice belongs to the venue's memory model
    // rather than to anything these cases do. A device whose driver is Metal announces nothing
    // for it.
    Degradation::SharedInSystemMemory,
];

impl Case for MinifyBand {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        PASS_NOTICES
            .iter()
            .map(|kind| crate::replay::notice_allowance(kind.code()))
            .collect()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = Fixture {
            name: SAMPLE_FIXTURE,
            entry: SAMPLE_ENTRY,
        };
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{SAMPLE_FIXTURE}{extension}"),
                entry: SAMPLE_ENTRY.to_owned(),
                source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Fixture {
            name: SAMPLE_FIXTURE,
            entry: SAMPLE_ENTRY,
        };
        Ok(vec![
            Input {
                name: "air:sample".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed:sample".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("format", &FORMAT.as_raw().to_le_bytes()),
            Input::bytes("surface:ostype", &OSTYPE),
            Input::bytes("surface:bytes", &surface_bytes()),
            Input::bytes("sampler", &sampler_spec().words()),
            Input::bytes("coordinates", &band(self.axis)),
            Input::bytes("buffer:blank", &[BLANK]),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = Fixture {
            name: SAMPLE_FIXTURE,
            entry: SAMPLE_ENTRY,
        };
        let source = fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&source.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, SAMPLE_ENTRY),
        )?;
        let surface = made(sites, "surface", side.buffer(&surface_bytes()))?;
        let texture = made(
            sites,
            "texture",
            side.plane_texture(
                surface,
                &PlaneTexture {
                    ostype: Some(OSTYPE),
                    format: FORMAT.as_raw(),
                    size: [W as u32, H as u32],
                    plane: 0,
                    planes: [LUMA, CHROMA],
                },
            ),
        )?;
        let sampler = made(sites, "sampler", side.sampler(&sampler_spec()))?;
        let coordinates = made(sites, "coordinates", side.buffer(&band(self.axis)))?;
        let output = made(sites, "output", side.buffer(&vec![BLANK; SAMPLES * 16]))?;
        let textures = [texture];
        let buffers = [output, coordinates];
        let samplers = [sampler];
        let command_buffer = made(
            sites,
            "dispatch",
            side.dispatch_threads(
                pipeline,
                &Bindings::new(&textures, &buffers).sampled(&samplers),
                &[],
                [SAMPLES as u32, 1, 1],
                [THREADS, 1, 1],
            ),
        )?;
        let status = made(sites, "status", side.wait(command_buffer))?;
        if status != crate::transcript::Value::unit_variant("Completed") {
            return Err(Abort {
                at: "status".to_owned(),
                failure: Failure::named("CommandBufferStatus", crate::diff::print(&status)),
            });
        }
        let bytes = made(
            sites,
            "read-back",
            side.read_buffer(output, 0, SAMPLES * 16),
        )?;
        if bytes.len() != SAMPLES * 16 {
            return Err(Abort {
                at: "read-back".to_owned(),
                failure: Failure::named(
                    "ShortReadBack",
                    format!("{} bytes of {}", bytes.len(), SAMPLES * 16),
                ),
            });
        }
        // The whole band is 544 KiB of values, past the suite's 512 KiB ceiling on a recording,
        // so the band's own site holds its digest and the window below holds whole samples.
        sites.push(Node::leaf(
            RESULTS_SITE,
            Observed::absent().digested(Digest::sha256(&bytes)),
        ));
        let ends = ends(&bytes);
        let digest = Digest::sha256(&ends);
        sites.push(Node::leaf(
            ENDS_SITE,
            Observed::bytes(ends, ends_layout()).digested(digest),
        ));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SHA-256 an **Apple M4** answered for each band, in [`all`]'s order, under macOS 27.0.1.
    ///
    /// This is what the two cases carried while no Metal side could make a texture over an
    /// IOSurface, and the M4 is gone. It is kept here, where the one thing that reads it is the
    /// cross-chip check below, rather than in the case, whose answer is its recording.
    const APPLE_M4_BANDS: [&str; 2] = [
        "c3c7fb23a2e0967dd506a088d764b8d33cd622f645566008bebfbe1dee7974c4",
        "536bb62bdfb366a18744a10f34682339739b789ee08d08d2d6d601731baf7533",
    ];

    /// Two cases, one an axis. The M4's two hashes differ, which is what says the two cases ask
    /// different questions.
    #[test]
    fn the_two_bands_are_the_lanes_two_bands() {
        let cases = all();
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].axis, 0);
        assert_eq!(cases[1].axis, 1);
        assert_ne!(APPLE_M4_BANDS[0], APPLE_M4_BANDS[1]);
        for hash in APPLE_M4_BANDS {
            assert_eq!(hash.len(), 64, "{hash}");
        }
    }

    /// **The band this oracle recorded is the band the Apple M4 answered.**
    ///
    /// The cross-chip check, and the reason the M4's digests were kept rather than deleted when
    /// the case stopped carrying them: whether two Apple GPU generations put a linear footprint's
    /// rounding switch in the same place, over 34,816 samples a band, is a fact about Metal and
    /// not about us. If this ever fails, the recording is the oracle and the M4's number is
    /// history — but it must fail loudly rather than be dropped.
    #[test]
    fn the_recorded_bands_are_the_apple_m4s() {
        for (case, apple_m4) in all().iter().zip(APPLE_M4_BANDS) {
            let recording = crate::recording::read(case.name())
                .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
            let answer = recording
                .answers
                .first()
                .unwrap_or_else(|| panic!("{}: the recording holds an answer", case.name()));
            assert!(
                answer.provenance.measured_elsewhere().is_none(),
                "{}: the answer is recorded here now, so it names no other chip: {:?}",
                case.name(),
                answer.provenance
            );
            let outcome = answer
                .outcomes
                .first()
                .unwrap_or_else(|| panic!("{}: the answer holds an outcome", case.name()));
            assert_eq!(
                answer.outcomes.len(),
                1,
                "{}: the band's samples are deterministic",
                case.name()
            );
            let node = outcome
                .transcript
                .root
                .child(RESULTS_SITE)
                .unwrap_or_else(|| panic!("{}: no {RESULTS_SITE}", case.name()));
            let digest = node
                .observed
                .digest
                .as_ref()
                .unwrap_or_else(|| panic!("{}: {RESULTS_SITE} holds no digest", case.name()));
            assert_eq!(
                digest.hex,
                apple_m4,
                "{}: this M3 Max's band and the Apple M4's",
                case.name()
            );
        }
    }

    /// The window is the first and last [`ENDS`] samples, in that order, and its layout names the
    /// sample inside it: a sweep whose ends drifted says which sample did.
    #[test]
    fn the_window_is_both_ends_of_the_band() {
        let mut bytes = vec![0u8; SAMPLES * 16];
        for sample in 0..SAMPLES {
            bytes[sample * 16..sample * 16 + 4].copy_from_slice(&(sample as u32).to_le_bytes());
        }
        let window = ends(&bytes);
        assert_eq!(window.len(), 2 * ENDS * 16);
        let first =
            |at: usize| u32::from_le_bytes(window[at * 16..at * 16 + 4].try_into().unwrap());
        assert_eq!(first(0), 0);
        assert_eq!(first(ENDS - 1), ENDS as u32 - 1);
        assert_eq!(first(ENDS), (SAMPLES - ENDS) as u32);
        assert_eq!(first(2 * ENDS - 1), SAMPLES as u32 - 1);
        let layout = ends_layout();
        assert_eq!(layout.extent(), 2 * ENDS * 16);
        assert_eq!(
            layout.at(0, (2 * ENDS) as u32 - 1, 3),
            (2 * ENDS - 1) * 16 + 12
        );
    }

    /// The surface is the one the probe measured: the lane pins its SHA-256 under CoreVideo's own
    /// layout, so a generator that drifts compares our samples against another surface's answer.
    #[test]
    fn the_surface_is_the_one_metal_sampled() {
        let bytes = surface_bytes();
        assert_eq!(bytes.len(), LENGTH);
        assert_eq!(
            crate::sha256_hex(&bytes),
            "21f3f30e8e4b0e04e50dd27376cfe9a6a8ae2dad7fa0c46db76580fe1b1f317c"
        );
    }

    /// Each band is 2,048 groups of seventeen consecutive `f32` coordinates on its own axis, with
    /// the other axis fixed inside each group. That shape is the whole experiment: a group whose
    /// coordinates were not consecutive, or whose other axis moved, would not say where the
    /// footprint's rounding puts the switch.
    #[test]
    fn each_group_walks_one_axis_by_one_ulp() {
        for axis in [0, 1] {
            let bytes = band(axis);
            assert_eq!(bytes.len(), SAMPLES * 8);
            let lane = |sample: usize, at: usize| {
                let base = sample * 8 + at * 4;
                f32::from_le_bytes(bytes[base..base + 4].try_into().expect("four bytes"))
            };
            assert_eq!(SAMPLES, 2048 * GROUP);
            for group in 0..(SAMPLES / GROUP) {
                let first = group * GROUP;
                let fixed = lane(first, 1 - axis);
                for step in 1..GROUP {
                    let sample = first + step;
                    assert_eq!(lane(sample, 1 - axis), fixed, "group {group} other axis");
                    assert_eq!(
                        lane(sample, axis),
                        lane(sample - 1, axis).next_up(),
                        "group {group} step {step}"
                    );
                }
            }
        }
    }

    /// The layout covers every sample and names the sample as its row.
    #[test]
    fn the_layout_names_the_sample() {
        let layout = layout();
        assert_eq!(layout.extent(), SAMPLES * 16);
        assert_eq!(layout.at(0, SAMPLES as u32 - 1, 3), (SAMPLES - 1) * 16 + 12);
    }
}
