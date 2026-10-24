//! Two-plane YCbCr textures: what Metal says about the formats' descriptors, and what a kernel's
//! `read()` of a `420v` surface's texels answers.
//!
//! Ported from `runtime_lane/ycbcr_textures.rs`. The lane is 4,500 lines over a `YCBCR8_420_2P`
//! texture laid over a CoreVideo surface's two planes, and the two halves of it port very
//! differently, because of one fact about Metal's API:
//!
//! **A texture of a two-plane YCbCr format exists only over an IOSurface.** The lane's own
//! `ycbcr_textures_are_made_only_over_their_surfaces` is that fact: `newTextureWithDescriptor:`
//! answers nil for the format, and so does a buffer's texture of it. The runtime takes a *buffer*
//! standing for the surface, which is what makes the lane run on a host with no CoreVideo, but
//! Metal takes an IOSurface and nothing else. So:
//!
//! - [`FormatDescriptors`] is **recorded on Metal**, because it asks only about descriptors:
//!   `heapTextureSizeAndAlignWithDescriptor:` and whether `newTextureWithDescriptor:` answers a
//!   texture, over every two-plane YCbCr format the translator names. Both are CPU calls over a
//!   descriptor Metal makes no texture of, so the case allocates nothing and reaches no GPU.
//! - [`TexelReads`] is **recorded on Metal too**, since the Metal side lays a texture over a real
//!   IOSurface laid out as the case says its stand-in buffer is (`metal_surface.rs`). It was
//!   carried while no call here could make one.
//!
//! # The probe stays, as the cross-check
//!
//! What the case used to carry is `fixtures/public/ycbcr/metal_m3_max.bin`: 960 texels of red,
//! green and blue steps, written by the committed probe `fixtures/public/ycbcr/probe.swift` over a
//! real IOSurface, on an Apple M3 Max under macOS 27.0.1. That file is no longer the case's answer
//! — the recording is — so it is no longer one of the case's inputs either. It is better used than
//! discarded: `the_recorded_reads_are_the_committed_probes` holds the recorded transcript to it,
//! so two independent Metal paths over the same surface, a Swift probe and this suite's own Metal
//! side, must answer the same 5,760 bytes. That is the strongest check the case can carry, and it
//! is what says the IOSurface this side builds is the surface the probe built.
//!
//! # What the port changes
//!
//! The lane compares its reads with that file **and** with `metal2vulkan::bind::ycbcr_texel`'s
//! model, which is the right thing for a lane to do. What it does not do is keep Metal's values in
//! a shape anything else can read: the comparison is a `Vec<[u16; 3]>` built inside one test
//! function, and the only artifact that survives a failure is the first differing texel. This case
//! records all 960 texels as a region whose rows are the texture's rows, so a diff names the texel
//! by its row and its channel, and the file's own SHA-256 is part of the case's input fingerprint:
//! a change to the measurement makes the case `Unrecorded` rather than quietly comparing against
//! other bytes.
//!
//! # Why neither case can fault a GPU
//!
//! [`FormatDescriptors`] runs no GPU work at all. The multisample descriptors of the lane's own
//! sweep are **left out by construction**: the runtime answers them `Error::Invalid`, which is its
//! word for "Metal's validation stops it, ending the process", and the suite's standing rule is
//! that no case may end Metal's process. [`TexelReads`] is one 40 x 24 texture, one 4 KiB surface
//! and one dispatch of 960 threads, and the one pair it hands Metal — format 500 over a `420v`
//! surface — is the pair the committed probe already ran on this machine.

use crate::allowance::Allowance;
use crate::case::{made, Abort, Bindings, Case, Failure, Input, MetalSource, PlaneTexture, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::texture_query::{HeapTextureQuery as Q, Storage};
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::format::ycbcr_format;
use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan_runtime::Degradation;

/// The case names, as the report and the recordings spell them.
pub const FORMAT_DESCRIPTORS: &str = "ycbcr-two-plane-format-descriptors";
pub const TEXEL_READS: &str = "ycbcr-texel-reads-420v";

/// Every case of the family.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![Box::new(FormatDescriptors), Box::new(TexelReads)]
}

/// Every two-plane YCbCr format the translator names, in `PixelFormat::ALL`'s order.
///
/// The family is read out of the translator rather than written down, because what the case asks
/// is "every two-plane YCbCr format", and a list would answer "the ones somebody typed". A format
/// added to the translator joins the sweep and changes the case's input fingerprint, which is the
/// loud failure we want rather than a silent gap.
fn family() -> Vec<PixelFormat> {
    PixelFormat::ALL
        .iter()
        .copied()
        .filter(|format| ycbcr_format(*format).is_some())
        .collect()
}

/// The texture extent every descriptor of the sweep names, as the lane's does.
const SIZE: [u32; 2] = [40, 24];

/// The slices the array descriptor names, as the lane's does.
const SLICES: u32 = 2;

/// The site a descriptor's answers sit under.
mod site {
    /// The heap size query's answer.
    pub const QUERY: &str = "size-and-align";
    /// Whether `newTextureWithDescriptor:` answered a texture.
    pub const TEXTURE: &str = "makes-texture";
}

/// What Metal says about each two-plane YCbCr format's descriptors, without a surface.
pub struct FormatDescriptors;

impl FormatDescriptors {
    /// The descriptors the case asks about, in order: each format's 2D descriptor in each
    /// storage, and its 2D array descriptor in each storage.
    ///
    /// The 2D multisample and multisample-array descriptors of the lane's sweep are not here:
    /// Metal's own validation stops them, which ends its process.
    fn queries(&self) -> Vec<(Q, bool)> {
        let mut queries = Vec::new();
        for format in family() {
            for storage in [Storage::Shared, Storage::Private] {
                // The 2D descriptor is the one the lane asks `newTextureWithDescriptor:` of, and
                // Metal answers nil for it, so asking allocates nothing.
                queries.push((Q::texture_2d(format, SIZE, storage), true));
            }
            for storage in [Storage::Shared, Storage::Private] {
                queries.push((
                    Q::texture_2d(format, SIZE, storage).of_type(
                        TextureType::Type2DArray,
                        1,
                        SLICES,
                    ),
                    false,
                ));
            }
        }
        queries
    }
}

impl Case for FormatDescriptors {
    fn name(&self) -> &'static str {
        FORMAT_DESCRIPTORS
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut words = Vec::new();
        for (query, asks_texture) in self.queries() {
            words.extend(query.words());
            words.push(u8::from(asks_texture));
        }
        Ok(vec![Input::bytes("descriptors", &words)])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for (query, asks_texture) in self.queries() {
            let mut node = Node::branch(query.site(), Vec::new());
            // Each descriptor's answers are sites of their own, and a call that did not answer is
            // recorded at its site while the sweep goes on: one format the runtime refuses
            // compares as a refusal at that one site, and the other 68 stay readable instead of
            // being lost behind an abort.
            let answered = match side.heap_texture_size_and_align(&query) {
                Ok(answer) => Value::ok(answer.value()),
                Err(failure) => failure.recorded(),
            };
            node = node.with(Node::leaf(site::QUERY, Observed::value(answered)));
            if asks_texture {
                let made_one = match side.makes_texture(&query) {
                    Ok(made_one) => Value::ok(Value::Bool(made_one)),
                    Err(failure) => failure.recorded(),
                };
                node = node.with(Node::leaf(site::TEXTURE, Observed::value(made_one)));
            }
            sites.push(node);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// The texel reads of one measured `420v` surface.
// ---------------------------------------------------------------------------------------------

/// The surface's extent in texels, and its planes' layout, exactly as the lane lays its stand-in
/// buffer out and as the probe laid the IOSurface it measured.
const W: u64 = 40;
const H: u64 = 24;
/// The bytes between rows of either plane.
const ROW: u64 = 64;
/// The chroma plane's first byte.
const CHROMA_AT: u64 = ROW * H;
/// The surface buffer's length.
const LENGTH: usize = 4096;
/// The surface's four-character OSType: a video-range 4:2:0 two-plane surface.
const OSTYPE: [u8; 4] = *b"420v";

/// The format the measured texture is of: `YCBCR8_420_2P`, as the translator names it.
const FORMAT: PixelFormat = PixelFormat::YCBCR84202P;

/// The seed the probe drew the measured surface's codes from.
const SEED: u64 = 5;

/// The texels the texture holds.
const TEXELS: usize = (W * H) as usize;

/// The denominator Metal's reads land on: every channel of every texel is a whole multiple of
/// `1 / 2048`, which is what lets the measurement be kept as steps rather than as floats.
const STEPS: f32 = 2048.0;

/// The committed measurement the recorded answer is cross-checked against, under the public
/// fixtures. Read by the tests alone: the case's answer is its recording.
#[cfg(test)]
const MEASURED_FILE: &str = "ycbcr/metal_m3_max.bin";

/// The fixture whose kernel reads the texels, and its entry.
const READ_FIXTURE: &str = "fpv_swizzled_reads";
const READ_ENTRY: &str = "fpv_swz_read";

/// The threads in a threadgroup of the reading dispatch.
const GROUP: u32 = 64;

/// The byte the read-back buffer holds before the dispatch.
const BLANK: u8 = 0xab;

/// The lane's and the probes' generator of the surface's bytes.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u8
    }
}

/// The surface's bytes at [`SEED`]: the luma plane's codes, then the chroma pairs', each plane's
/// rows [`ROW`] bytes apart, every other byte `0xa5`.
fn surface_bytes() -> Vec<u8> {
    let mut random = Random(SEED);
    let mut bytes = vec![0xa5; LENGTH];
    for y in 0..H {
        for x in 0..W {
            bytes[(y * ROW + x) as usize] = random.next();
        }
    }
    for y in 0..H / 2 {
        for x in 0..W {
            bytes[(CHROMA_AT + y * ROW + x) as usize] = random.next();
        }
    }
    bytes
}

/// The positions the reading kernel reads at: every texel, row by row, as a pair of 32-bit words.
fn positions() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(TEXELS * 8);
    for y in 0..H as u32 {
        for x in 0..W as u32 {
            bytes.extend(x.to_le_bytes());
            bytes.extend(y.to_le_bytes());
        }
    }
    bytes
}

/// The committed measurement's bytes: 960 texels of three little-endian 16-bit steps.
#[cfg(test)]
fn measured() -> Result<Vec<u8>, String> {
    let file = metal2vulkan_devtools::fixtures_dir().join(MEASURED_FILE);
    let bytes = std::fs::read(&file).map_err(|error| format!("{}: {error}", file.display()))?;
    if bytes.len() != TEXELS * 6 {
        return Err(format!(
            "{}: {} bytes, not {TEXELS} texels of six",
            file.display(),
            bytes.len()
        ));
    }
    Ok(bytes)
}

/// The layout the steps are compared under: one row a row of the texture, three steps a texel, so
/// a diff names the texel by its row and column and the channel inside it.
fn step_layout() -> ByteLayout {
    let row_pitch = (W * 3 * 2) as u32;
    ByteLayout {
        // The steps are the read-back buffer's `float4`s projected, not the texture's own bytes,
        // so the region is the buffer they came out of; the rows below are the texture's rows,
        // which is what makes a diff name the texel by its row and column.
        region: Region::Buffer {
            offset: 0,
            length: (TEXELS * 6) as u64,
        },
        element: Element::U16,
        row_offset: 0,
        row_elements: (W * 3) as u32,
        row_pitch,
        rows: H as u32,
        image_pitch: row_pitch * H as u32,
        images: 1,
        // The steps are packed: there is nothing between a row's last step and the next row.
        gaps: Gaps::Ignored,
    }
}

/// The site the steps sit at.
const STEPS_SITE: &str = "texels/steps";

/// The site that says how many texels read an alpha other than one.
const ALPHA_SITE: &str = "texels/alpha-not-one";

/// Metal's `read()` of a measured `420v` surface's every texel.
pub struct TexelReads;

impl TexelReads {
    /// The `float4`s `bytes` holds, as the three steps a channel is, and how many texels read an
    /// alpha other than exactly one.
    ///
    /// A channel off the `1 / 2048` grid, or outside `[0, 1]`, is an error rather than a rounded
    /// step: the measurement is kept as steps precisely because every value Metal answers is one,
    /// and silently rounding would compare a different number from the one Metal gave.
    fn steps(bytes: &[u8]) -> Result<(Vec<u8>, u64), String> {
        let mut steps = Vec::with_capacity(TEXELS * 6);
        let mut alpha_off = 0u64;
        for texel in 0..TEXELS {
            let lane = |at: usize| {
                let base = texel * 16 + at * 4;
                f32::from_le_bytes(
                    bytes[base..base + 4]
                        .try_into()
                        .expect("four bytes of a lane"),
                )
            };
            for channel in 0..3 {
                let value = lane(channel) * STEPS;
                if value != value.round() || !(0.0..=STEPS).contains(&value) {
                    return Err(format!(
                        "texel {texel} channel {channel} reads {}, which is not a step of 1/{}",
                        lane(channel),
                        STEPS as u32
                    ));
                }
                steps.extend((value as u16).to_le_bytes());
            }
            if lane(3) != 1.0 {
                alpha_off += 1;
            }
        }
        Ok((steps, alpha_off))
    }
}

/// The degrade notices one of these cases may announce. A list rather than a device query,
/// because a case never branches on the device: it runs and reads what came back, and a venue that
/// announces a kind not here fails the case until the kind is added.
///
/// Declaring a kind states what our side announced at the notices site and loosens no value:
/// every read channel still compares exactly.
const PASS_NOTICES: &[Degradation] = &[
    // The buffer the texel-read case writes its reads into is `Shared`; the descriptor case runs
    // no pass and announces nothing, which is why it declares nothing.
    // A device with no unified memory keeps a `Shared` allocation in system memory because that is
    // the only memory both sides can reach, so this notice belongs to the venue's memory model
    // rather than to anything these cases do. A device whose driver is Metal announces nothing
    // for it.
    Degradation::SharedInSystemMemory,
];

impl Case for TexelReads {
    fn name(&self) -> &'static str {
        TEXEL_READS
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
            name: READ_FIXTURE,
            entry: READ_ENTRY,
        };
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{READ_FIXTURE}{extension}"),
                entry: READ_ENTRY.to_owned(),
                source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Fixture {
            name: READ_FIXTURE,
            entry: READ_ENTRY,
        };
        Ok(vec![
            Input {
                name: "air:read".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed:read".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("format", &FORMAT.as_raw().to_le_bytes()),
            Input::bytes("surface:ostype", &OSTYPE),
            Input::bytes("surface:bytes", &surface_bytes()),
            Input::bytes("positions", &positions()),
            Input::bytes("buffer:blank", &[BLANK]),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = Fixture {
            name: READ_FIXTURE,
            entry: READ_ENTRY,
        };
        let source = fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&source.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, READ_ENTRY),
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
                    planes: [(0, ROW), (CHROMA_AT, ROW)],
                },
            ),
        )?;
        let positions = made(sites, "positions", side.buffer(&positions()))?;
        let output = made(sites, "output", side.buffer(&vec![BLANK; TEXELS * 16]))?;
        let textures = [texture];
        let buffers = [output, positions];
        let command_buffer = made(
            sites,
            "dispatch",
            side.dispatch_threads(
                pipeline,
                &Bindings::new(&textures, &buffers),
                &[],
                [TEXELS as u32, 1, 1],
                [GROUP, 1, 1],
            ),
        )?;
        let status = made(sites, "status", side.wait(command_buffer))?;
        if status != Value::unit_variant("Completed") {
            return Err(Abort {
                at: "status".to_owned(),
                failure: Failure::named("CommandBufferStatus", crate::diff::print(&status)),
            });
        }
        let bytes = made(sites, "read-back", side.read_buffer(output, 0, TEXELS * 16))?;
        let (steps, alpha_off) = Self::steps(&bytes).map_err(|why| Abort {
            at: STEPS_SITE.to_owned(),
            failure: Failure::named("OffTheGrid", why),
        })?;
        let digest = Digest::sha256(&steps);
        sites.push(Node::leaf(
            STEPS_SITE,
            Observed::bytes(steps, step_layout()).digested(digest),
        ));
        sites.value(ALPHA_SITE, Value::Uint(alpha_off));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sweep is the whole two-plane YCbCr family, which is 69 formats, and it asks each of
    /// them the four descriptors it means to: the 2D and the 2D array, in each storage.
    #[test]
    fn the_sweep_covers_the_whole_family() {
        let formats = family();
        assert_eq!(formats.len(), 69, "{formats:?}");
        let queries = FormatDescriptors.queries();
        assert_eq!(queries.len(), formats.len() * 4);
        // Only the 2D descriptors are asked whether Metal makes a texture: those are the ones the
        // lane establishes Metal answers nil for.
        let asked = queries.iter().filter(|(_, asks)| *asks).count();
        assert_eq!(asked, formats.len() * 2);
        for (query, asks) in &queries {
            assert_eq!(*asks, query.texture_type == TextureType::Type2D);
            // No multisample descriptor is in the sweep: Metal's validation stops those, which
            // ends its process.
            assert_eq!(query.samples, 1, "{}", query.site());
        }
        // Every descriptor has a site of its own, so no two answers land on one name.
        let mut sites: Vec<String> = queries.iter().map(|(query, _)| query.site()).collect();
        sites.sort();
        let before = sites.len();
        sites.dedup();
        assert_eq!(sites.len(), before);
    }

    /// `YCBCR8_420_2P`, the format the measured texture is of, is in the sweep by the raw number
    /// the carried case names, so the two halves of the family cannot drift apart.
    #[test]
    fn the_measured_format_is_in_the_family() {
        assert!(family().contains(&FORMAT));
        assert_eq!(FORMAT, PixelFormat::YCBCR84202P);
    }

    /// The surface the case feeds is the one the probe measured: the lane pins its SHA-256, and a
    /// generator that drifts would compare our reads against another surface's answer.
    #[test]
    fn the_surface_is_the_one_metal_read() {
        let bytes = surface_bytes();
        assert_eq!(bytes.len(), LENGTH);
        assert_eq!(
            crate::sha256_hex(&bytes),
            "bb42f2090b661a1ef27876d6e3ff2404436756016fbf912cafdb9b35bfc6d3b1"
        );
    }

    /// The committed probe's measurement is the one it has always been, by its length and its
    /// SHA-256, which the lane pins too.
    #[test]
    fn the_probes_reads_are_the_committed_measurement() {
        let bytes = measured().expect("the committed measurement");
        assert_eq!(bytes.len(), TEXELS * 6);
        assert_eq!(
            crate::sha256_hex(&bytes),
            "88301e8b9aea601017cfc7466ba830c9255fb1875dcb9d62ffd2d4a8e14d8b8e"
        );
    }

    /// **The recorded answer is the committed probe's answer, byte for byte.**
    ///
    /// Two independent Metal paths over one surface: `fixtures/public/ycbcr/probe.swift`, which
    /// built its own `420v` IOSurface in Swift and read every texel through a runtime-compiled
    /// MSL kernel, and this case, which builds the surface through `metal_surface.rs` and reads it
    /// through the committed `fpv_swizzled_reads` fixture. Nothing but Metal's own `read()` of the
    /// same bytes is common to the two, so an equal 5,760 bytes is what says the surface this
    /// suite lays out is the surface the probe laid out, and that the fixture's kernel reads what
    /// the probe's kernel read.
    ///
    /// It is also the only check that can say so: the case's own verdict compares our side with
    /// the recording, and would read exact however the recording was taken.
    #[test]
    fn the_recorded_reads_are_the_committed_probes() {
        let recording = crate::recording::read(TEXEL_READS).expect("the committed recording");
        let answer = recording
            .answers
            .first()
            .expect("the recording holds an answer");
        assert!(
            answer.provenance.measured_elsewhere().is_none(),
            "the answer is no longer carried, so it names no other chip: {:?}",
            answer.provenance
        );
        let outcome = answer
            .outcomes
            .first()
            .expect("the answer holds an outcome");
        assert_eq!(answer.outcomes.len(), 1, "the reads are deterministic");
        let node = outcome
            .transcript
            .root
            .child(STEPS_SITE)
            .unwrap_or_else(|| panic!("the recording holds no {STEPS_SITE}"));
        let recorded = match &node.observed.value {
            Value::Bytes(bytes) => bytes.clone(),
            other => panic!("{STEPS_SITE}: the recording holds {}", other.kind()),
        };
        let probe = measured().expect("the committed measurement");
        assert_eq!(
            crate::sha256_hex(&recorded),
            crate::sha256_hex(&probe),
            "the recorded reads and the committed probe's"
        );
        assert_eq!(recorded, probe);
    }

    /// The layout covers every step and names the texel's row, so a diff over 960 texels says
    /// which row and which channel moved.
    #[test]
    fn the_step_layout_names_the_row() {
        let layout = step_layout();
        assert_eq!(layout.extent(), TEXELS * 6);
        // The last texel of the last row, its blue step.
        assert_eq!(
            layout.at(0, H as u32 - 1, (W * 3) as u32 - 1),
            TEXELS * 6 - 2
        );
    }

    /// The projection to steps accepts what Metal answers and refuses what it cannot: a channel
    /// off the `1 / 2048` grid is an error rather than a rounded step.
    #[test]
    fn the_projection_refuses_a_value_off_the_grid() {
        let mut bytes = Vec::new();
        for texel in 0..TEXELS {
            let step = (texel % 2049) as f32 / STEPS;
            for lane in [step, step, step, 1.0] {
                bytes.extend(lane.to_le_bytes());
            }
        }
        let (steps, alpha_off) = TexelReads::steps(&bytes).expect("every value is a step");
        assert_eq!(steps.len(), TEXELS * 6);
        assert_eq!(alpha_off, 0);
        assert_eq!(
            u16::from_le_bytes([steps[6], steps[7]]),
            1,
            "texel 1's red step"
        );
        // One channel moved half a step: the projection says so rather than rounding it.
        let mut off = bytes.clone();
        let half = (0.5f32 / STEPS + 1.0 / STEPS).to_le_bytes();
        off[16..20].copy_from_slice(&half);
        let why = TexelReads::steps(&off).expect_err("half a step is not a step");
        assert!(why.contains("texel 1 channel 0"), "{why}");
        // An alpha other than one is counted rather than refused: it is a fact of its own site.
        let mut alpha = bytes.clone();
        alpha[12..16].copy_from_slice(&0.5f32.to_le_bytes());
        let (_, counted) = TexelReads::steps(&alpha).expect("the steps still read");
        assert_eq!(counted, 1);
    }
}

/// The safety guard the format sweep rests on, run on a device.
///
/// Record mode hands every descriptor of [`FormatDescriptors`] to Metal, and a descriptor Metal's
/// own validation stops **ends Metal's process**. The runtime's model is what tells the two apart:
/// `Error::Nil` is "plain Metal answers nil, the process going on", while `Error::Invalid` is
/// "Metal's validation stops it, ending the process". So a sweep that makes the runtime answer
/// anything but an answer is a sweep that must not be recorded, and this says so before anybody
/// runs `conformance record`.
///
/// Its failure mode was exercised while writing it: adding the lane's 2D multisample descriptor to
/// the sweep fails here, naming the format and `TypeForFormat`.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod device_tests {
    use super::*;
    use crate::device::{self, Notices, ScratchCache, Venues};
    use crate::runtime_side::RuntimeSide;

    /// The venue the guard opens: a GPU where the tests are built for one, else a CPU ICD.
    const VENUE: Venues = if cfg!(m2v_gpu) {
        Venues::Gpu
    } else {
        Venues::Cpu
    };

    #[test]
    fn no_descriptor_of_the_family_is_one_metals_validation_stops() {
        let mut notices = Notices::default();
        let cache = ScratchCache::new().unwrap();
        let runtime = device::open(VENUE, &notices, &cache).unwrap();
        let mut side = RuntimeSide::new(
            &runtime,
            notices.open("no_descriptor_of_the_family_is_one_metals_validation_stops"),
        );
        let mut stopped = Vec::new();
        for (query, asks_texture) in FormatDescriptors.queries() {
            if let Err(failure) = side.heap_texture_size_and_align(&query) {
                stopped.push(format!("{}: {:?}", query.site(), failure.recorded()));
            }
            if asks_texture {
                if let Err(failure) = side.makes_texture(&query) {
                    stopped.push(format!(
                        "{} makes_texture: {:?}",
                        query.site(),
                        failure.recorded()
                    ));
                }
            }
        }
        assert!(
            stopped.is_empty(),
            "these descriptors do not answer on the runtime, so Metal's validation may stop them \
             and record mode would end Metal's process; take them out of the sweep:\n{}",
            stopped.join("\n")
        );
    }
}
