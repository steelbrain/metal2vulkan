//! Whether a gather and a nearest sample of 32-bit float texels keep a **denormal's exact word**.
//!
//! This is the one family where the interesting answer is a driver's habit rather than a format's
//! rule: `runtime_lane::denormal_word_fetches` measured that Metal keeps a denormal texel through
//! its samplers, while RADV's gathers and nearest samples, NVIDIA's gathers and ANV's nearest
//! samples hand back **the zero of the texel's sign** — and every one of those devices' plain
//! fetches keeps it. A denormal flushed to zero is a value a comparison in floats would call
//! almost right, so every site here records unsigned words and compares them exactly.
//!
//! The owned fixture `denormal_texels` is the one written for this question. Lane `i` reads the
//! coordinate `in[i].xy` and writes eight `float4`s, each a site of its own because each is its own
//! question: a device may flush under one sampler state and keep under another.
//!
//! | Site | What it is |
//! |---|---|
//! | `gather-r-edge` | `r.gather` through a nearest clamp-to-edge sampler |
//! | `sample-r-edge` | `r.sample` through the same |
//! | `gather-rgba-edge-y` | `rgba.gather` of component `y`, clamp to edge |
//! | `gather-rgba-repeat-w` | `rgba.gather` of component `w`, repeating |
//! | `sample-rgba-edge` | `rgba.sample`, clamp to edge |
//! | `gather-r-repeat` | `r.gather`, repeating |
//! | `sample-rgba-repeat-offset` | `rgba.sample` at a texel offset of `(1, -1)`, repeating |
//! | `sample-array-layer1-edge` | `arr.sample` of layer 1 of a two-slice array |
//!
//! The coordinates are the lane's own: a gather at each of the nine interior texel corners of the
//! 4 by 4 texture, where a gather's four taps are four different texels, and a nearest sample at
//! each of the sixteen texel centres, where the word states one texel. The two sets are one buffer
//! and every access runs over both, so a gather's answer at a texel centre and a sample's at a
//! corner are recorded too rather than being left out as uninteresting — they are the rows that
//! would catch a tie broken the other way.
//!
//! The lane also read the texture through a **runtime** sampler slot, through an argument buffer's
//! sampler, and as a heap texture, on its own generated AIR; `denormal_texels` holds constexpr
//! samplers only and no committed fixture reads a denormal texture through the other three with
//! one texture bound. They are not ported here, and the module docs of
//! `runtime_lane::denormal_word_fetches` record that Metal's runtime and argument-buffer samplers
//! filtered exactly as its constexpr samplers of the same state.
//!
//! Every output word holds [`BLANK`] before the dispatch, so a word the kernel did not reach is a
//! recorded value rather than whatever the allocator left.
//!
//! Nothing here can fault a GPU: one compute pass, twenty-five threads, three small textures, no
//! loop, and every coordinate inside or one texel outside a clamped or repeating texture.

use crate::case::{call, made, Abort, Bindings, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::texture::{TextureRequest, Upload};
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};

/// The fixture the kernel lives in, and its entry point.
const FIXTURE: &str = "denormal_texels";

/// The textures' side: 4 by 4, which the lane's corners and centres are of.
const SIDE: u32 = 4;

/// The texels of one level.
const TEXELS: usize = (SIDE * SIDE) as usize;

/// The slices the array texture at slot 2 holds; the kernel samples layer 1.
const SLICES: u32 = 2;

/// The `float4`s the kernel writes a lane.
const RESULTS: usize = 8;

/// The byte every output word holds before the dispatch.
pub const BLANK: u8 = 0xab;

/// The texels of the 4 by 4 `R32Float` texture, row by row, as the lane wrote them: denormals of
/// both signs, signed zeros, the smallest normal, and ordinary values.
const WORDS: [u32; TEXELS] = [
    0x0000_0001,
    0x8000_0002,
    0x007f_ffff,
    0x807f_ffff,
    0x0040_0000,
    0x8000_0000,
    0x0000_0000,
    0x0080_0000,
    0x3f80_0000,
    0xbf00_0000,
    0x0012_3456,
    0x8065_4321,
    0x4120_0000,
    0x0000_0100,
    0x8000_0001,
    0x3e80_0000,
];

/// How far apart two channels of one texel are in [`WORDS`], so each word lands at every channel
/// position of the four-channel texture and a fetch that flushed one channel and not another shows
/// as a texel rather than as a hash.
const STRIDE: usize = 5;

/// The sites the eight results are recorded at, in the order the kernel writes them.
const RESULT_SITES: [&str; RESULTS] = [
    "gather-r-edge",
    "sample-r-edge",
    "gather-rgba-edge-y",
    "gather-rgba-repeat-w",
    "sample-rgba-edge",
    "gather-r-repeat",
    "sample-rgba-repeat-offset",
    "sample-array-layer1-edge",
];

/// The one-channel texture's bytes.
fn red_bytes() -> Vec<u8> {
    WORDS.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// The four-channel texture's bytes: each texel's four channels taken [`STRIDE`] apart.
fn rgba_bytes() -> Vec<u8> {
    (0..TEXELS)
        .flat_map(|texel| (0..4).map(move |channel| WORDS[(texel + STRIDE * channel) % TEXELS]))
        .flat_map(u32::to_le_bytes)
        .collect()
}

/// The array texture's slice `slice`: slice 0 holds the words as they are and slice 1 holds them
/// rotated, so a sample that read the wrong layer reads a word that is in no other place of it.
fn slice_bytes(slice: u32) -> Vec<u8> {
    let turn = slice as usize * 7;
    (0..TEXELS)
        .map(|texel| WORDS[(texel + turn) % TEXELS])
        .flat_map(u32::to_le_bytes)
        .collect()
}

/// The nine interior texel corners, in the lane's own order: a gather at one of these reads four
/// different texels.
fn corners() -> Vec<(u32, u32)> {
    (0..SIDE - 1)
        .flat_map(|down| (0..SIDE - 1).map(move |across| (across, down)))
        .collect()
}

/// The sixteen texel centres, row major: a nearest sample at one of these states one texel.
fn centres() -> Vec<(u32, u32)> {
    (0..SIDE)
        .flat_map(|down| (0..SIDE).map(move |across| (across, down)))
        .collect()
}

/// Every coordinate the case reads at: the corners first, then the centres, as the lane's two
/// sweeps ran.
fn coordinates() -> Vec<[f32; 2]> {
    let side = SIDE as f32;
    let corner = corners()
        .into_iter()
        .map(move |(across, down)| [(across + 1) as f32 / side, (down + 1) as f32 / side]);
    let centre = centres()
        .into_iter()
        .map(move |(across, down)| [(across as f32 + 0.5) / side, (down as f32 + 0.5) / side]);
    corner.chain(centre).collect()
}

/// The coordinate buffer's bytes: one `float4` a lane, the coordinate in `xy` and zero elsewhere.
fn coordinate_bytes() -> Vec<u8> {
    coordinates()
        .into_iter()
        .flat_map(|[across, down]| [across, down, 0.0, 0.0])
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// The lanes the case dispatches.
fn lanes() -> usize {
    coordinates().len()
}

/// The output buffer before the dispatch: eight `float4`s a lane, every byte [`BLANK`].
fn blank_output() -> Vec<u8> {
    vec![BLANK; lanes() * RESULTS * 16]
}

/// One result's four words of each lane, gathered out of the whole output.
///
/// Gathered rather than recorded as a column of the whole buffer, because eight sites each holding
/// the whole buffer would store the same bytes eight times. The eight results partition a lane's
/// 128 bytes, so nothing is left out of the recording by it.
fn result(output: &[u8], index: usize) -> Vec<u8> {
    (0..lanes())
        .flat_map(|lane| {
            let at = (lane * RESULTS + index) * 16;
            output[at..at + 16].to_vec()
        })
        .collect()
}

/// The layout a gathered result is compared under: one row a lane, four words a row, so the
/// element a diff names is the coordinate and the gather tap or the channel inside it.
fn layout() -> ByteLayout {
    let length = (lanes() * 16) as u32;
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(length),
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: 4,
        row_pitch: 16,
        rows: lanes() as u32,
        image_pitch: length,
        images: 1,
        // A row is its four words and the rows abut, so there is no byte outside an element.
        gaps: Gaps::Ignored,
    }
}

/// The one-channel texture at slot 0.
fn red_request<'a>(uploads: &'a [Upload<'a>]) -> TextureRequest<'a> {
    TextureRequest::read_2d(PixelFormat::R32Float.as_raw(), SIDE, SIDE).loaded(uploads)
}

/// The four-channel texture at slot 1.
fn rgba_request<'a>(uploads: &'a [Upload<'a>]) -> TextureRequest<'a> {
    TextureRequest::read_2d(PixelFormat::RGBA32Float.as_raw(), SIDE, SIDE).loaded(uploads)
}

/// The two-slice array texture at slot 2, whose layer 1 the kernel samples.
fn array_request<'a>(uploads: &'a [Upload<'a>]) -> TextureRequest<'a> {
    let mut request = TextureRequest::read_2d(PixelFormat::R32Float.as_raw(), SIDE, SIDE);
    request.texture_type = TextureType::Type2DArray;
    request.slices = SLICES;
    request.usage = TextureUsage::SHADER_READ;
    request.loaded(uploads)
}

/// The upload a whole level 0 of a 4 by 4 texture of `texel_bytes` bytes a texel takes.
fn upload<'a>(bytes: &'a [u8], texel_bytes: u32, slice: u32) -> Upload<'a> {
    Upload {
        level: 0,
        slice,
        origin: [0, 0, 0],
        size: [SIDE, SIDE, 1],
        bytes,
        row_pitch: SIDE * texel_bytes,
        rows: SIDE,
        image_pitch: 0,
    }
}

/// The case: every gather and nearest sample of the denormal textures.
pub struct DenormalWordFetches;

/// The one case of the family.
pub fn all() -> Vec<DenormalWordFetches> {
    vec![DenormalWordFetches]
}

impl DenormalWordFetches {
    /// The fixture under its one kernel.
    fn fixture() -> Fixture {
        Fixture::new(FIXTURE)
    }
}

impl Case for DenormalWordFetches {
    fn name(&self) -> &'static str {
        "denormal-word-fetches"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<crate::allowance::Allowance> {
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the notices
        // site alone: every value of this family still compares against Metal's exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09), where it fired for all this case
        // and each row was divergent on that field and nothing else (replay-nvidia.log).
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: FIXTURE.to_owned(),
                source_sha256: crate::sha256_hex(Self::fixture().text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Self::fixture();
        let red = red_bytes();
        let rgba = rgba_bytes();
        let (first, second) = (slice_bytes(0), slice_bytes(1));
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            // Each texture's whole descriptor and its starting bytes, so a texture whose shape or
            // words moved makes the case unrecorded rather than comparing against other texels.
            Input::bytes("texture:red", &red_request(&[upload(&red, 4, 0)]).words()),
            Input::bytes(
                "texture:rgba",
                &rgba_request(&[upload(&rgba, 16, 0)]).words(),
            ),
            Input::bytes(
                "texture:array",
                &array_request(&[upload(&first, 4, 0), upload(&second, 4, 1)]).words(),
            ),
            Input::bytes("buffer:coordinates", &coordinate_bytes()),
            Input::bytes("buffer:blank", &blank_output()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Self::fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, loaded.entry),
        )?;
        let red = red_bytes();
        let rgba = rgba_bytes();
        let (first, second) = (slice_bytes(0), slice_bytes(1));
        let red_texture = made(
            sites,
            "texture/red",
            side.texture(&red_request(&[upload(&red, 4, 0)])),
        )?;
        let rgba_texture = made(
            sites,
            "texture/rgba",
            side.texture(&rgba_request(&[upload(&rgba, 16, 0)])),
        )?;
        let array = made(
            sites,
            "texture/array",
            side.texture(&array_request(&[
                upload(&first, 4, 0),
                upload(&second, 4, 1),
            ])),
        )?;
        let coordinates = made(
            sites,
            "buffer/coordinates",
            side.buffer(&coordinate_bytes()),
        )?;
        let output = made(sites, "buffer/out", side.buffer(&blank_output()))?;
        let textures: [Handle; 3] = [red_texture, rgba_texture, array];
        let command_buffer = made(
            sites,
            "dispatch",
            side.dispatch_threads(
                pipeline,
                &Bindings::new(&textures, &[coordinates, output]),
                &[],
                [lanes() as u32, 1, 1],
                [lanes() as u32, 1, 1],
            ),
        )?;
        let status = call(sites, "status", side.wait(command_buffer), Clone::clone)?;
        if status != Value::unit_variant("Completed") {
            return Err(Abort {
                at: "status".to_owned(),
                failure: Failure::named("CommandBufferStatus", crate::diff::print(&status)),
            });
        }
        let read = call(
            sites,
            "read-back",
            side.read_buffer(output, 0, lanes() * RESULTS * 16),
            |_| Value::Absent,
        )?;
        let mut words = Node::branch("words", Vec::new());
        for (index, name) in RESULT_SITES.iter().enumerate() {
            words = words.with(Node::leaf(
                *name,
                Observed::bytes(result(&read, index), layout()),
            ));
        }
        sites.push(words);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The word of texel `(across, down)` of the one-channel texture.
    fn texel(across: u32, down: u32) -> u32 {
        WORDS[(SIDE * down + across) as usize]
    }

    /// The inputs hold what the family turns on: denormals of both signs, both signed zeros, the
    /// smallest normal, and the largest denormal. A sweep that lost one would ask about flushing
    /// without holding the value that flushes.
    #[test]
    fn the_words_hold_denormals_of_both_signs_and_both_zeros() {
        let denormals: Vec<u32> = WORDS
            .iter()
            .copied()
            .filter(|word| word & 0x7f80_0000 == 0 && word & 0x007f_ffff != 0)
            .collect();
        assert!(denormals.len() >= 6, "{denormals:?}");
        assert!(denormals.iter().any(|word| word >> 31 == 0));
        assert!(denormals.iter().any(|word| word >> 31 == 1));
        assert!(WORDS.contains(&0x0000_0000) && WORDS.contains(&0x8000_0000));
        // The smallest normal, whose neighbour below is a denormal: the pair a flush separates.
        assert!(WORDS.contains(&0x0080_0000) && WORDS.contains(&0x007f_ffff));
    }

    /// The coordinates are the nine interior corners then the sixteen centres, each distinct, and
    /// the buffer holds one `float4` for each.
    #[test]
    fn the_coordinates_are_the_corners_then_the_centres() {
        let taken = coordinates();
        assert_eq!(taken.len(), 9 + 16);
        assert_eq!(lanes(), taken.len());
        let bits: BTreeSet<[u32; 2]> = taken
            .iter()
            .map(|[across, down]| [across.to_bits(), down.to_bits()])
            .collect();
        assert_eq!(bits.len(), taken.len(), "every coordinate is its own");
        assert_eq!(coordinate_bytes().len(), taken.len() * 16);
        // The first nine are corners: a multiple of a quarter, inside the texture.
        for [across, down] in &taken[..9] {
            assert_eq!(across * 4.0, (across * 4.0).round());
            assert_eq!(down * 4.0, (down * 4.0).round());
        }
    }

    /// Each texture states bytes for its whole level, and the array's two slices hold different
    /// words, so a sample that read layer 0 instead of layer 1 cannot pass.
    #[test]
    fn each_texture_states_its_own_bytes() {
        assert_eq!(red_bytes().len(), TEXELS * 4);
        assert_eq!(rgba_bytes().len(), TEXELS * 16);
        assert_eq!(slice_bytes(0).len(), TEXELS * 4);
        assert_eq!(slice_bytes(0), red_bytes());
        assert_ne!(slice_bytes(0), slice_bytes(1));
        assert_eq!(layout().extent(), lanes() * 16);
        assert_eq!(blank_output().len(), lanes() * RESULTS * 16);
    }

    /// A result gathers the bytes it names and no other, and the eight results partition a lane.
    ///
    /// A guard over distinct per-byte values, so a result taking another's index cannot pass by
    /// coincidence.
    #[test]
    fn a_result_gathers_its_own_bytes() {
        let output: Vec<u8> = (0..lanes() * RESULTS * 16)
            .map(|at| (at % 251) as u8)
            .collect();
        assert_eq!(RESULT_SITES.len(), RESULTS);
        let named: BTreeSet<&str> = RESULT_SITES.iter().copied().collect();
        assert_eq!(named.len(), RESULTS, "the sites are distinct");
        for index in 0..RESULTS {
            let taken = result(&output, index);
            assert_eq!(taken.len(), lanes() * 16);
            for lane in 0..lanes() {
                for byte in 0..16 {
                    assert_eq!(
                        taken[lane * 16 + byte],
                        output[(lane * RESULTS + index) * 16 + byte],
                        "result {index} lane {lane} byte {byte}"
                    );
                }
            }
        }
    }

    /// Metal's recorded words are the texels themselves, for the two accesses whose answer the
    /// lane pins: a clamp-to-edge gather at an interior corner reads the four texels around it in
    /// Metal's gather order, and a clamp-to-edge nearest sample at a texel centre reads that texel
    /// with 0, 0 and 1 beside it.
    ///
    /// This is the cross-check the port exists for. The lane asserts the same rule against a live
    /// device, which says nothing on a host with no Metal; here it is held against Metal's own
    /// recorded answer, so a driver that starts flushing shows as the texel that moved.
    #[test]
    fn metals_recorded_words_are_the_texels_the_lane_pins() {
        use metal2vulkan_runtime::MetalLevel;
        let case = DenormalWordFetches;
        let Ok(recording) = crate::recording::read(case.name()) else {
            // Recorded in a later commit of this batch.
            return;
        };
        let answer = recording
            .answer(MetalLevel::NEWEST)
            .unwrap_or_else(|why| panic!("{why}"));
        let taken = answer.deterministic().expect("one outcome");
        let words = taken.root.child("words").expect("the words site");
        let read = |name: &str| -> Vec<u32> {
            let site = words.child(name).unwrap_or_else(|| panic!("no {name}"));
            match &site.observed.value {
                Value::Bytes(bytes) => bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| u32::from_le_bytes(*word))
                    .collect(),
                other => panic!("{name} holds {}", other.kind()),
            }
        };
        let gathered = read("gather-r-edge");
        let sampled = read("sample-r-edge");
        assert_eq!(gathered.len(), lanes() * 4);
        assert_eq!(sampled.len(), lanes() * 4);
        // The first nine lanes are the interior corners, where Metal's gather order is the
        // lower-left, lower-right, upper-right, upper-left tap.
        for (lane, (across, down)) in corners().into_iter().enumerate() {
            let want = [
                texel(across, down + 1),
                texel(across + 1, down + 1),
                texel(across + 1, down),
                texel(across, down),
            ];
            for (tap, &expected) in want.iter().enumerate() {
                let got = gathered[4 * lane + tap];
                assert_eq!(
                    got, expected,
                    "gather at corner ({across}, {down}) tap {tap}: Metal recorded {got:#010x}, \
                     the texel is {expected:#010x}"
                );
            }
        }
        // The last sixteen lanes are the texel centres, where a nearest sample states one texel.
        for (index, (across, down)) in centres().into_iter().enumerate() {
            let lane = 9 + index;
            let want = [texel(across, down), 0, 0, 1f32.to_bits()];
            for (channel, &expected) in want.iter().enumerate() {
                let got = sampled[4 * lane + channel];
                assert_eq!(
                    got, expected,
                    "nearest sample at ({across}, {down}) channel {channel}: Metal recorded \
                     {got:#010x}, the texel is {expected:#010x}"
                );
            }
        }
    }
}
