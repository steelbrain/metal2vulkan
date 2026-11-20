//! A texture's extent at a level read from a buffer, ported from `runtime_lane::level_sizes`.
//!
//! A shader may ask for `get_width(lod)` at a level the texture does not have, and Metal answers
//! something rather than refusing: `max(1, e >> (lod & 255))` of the first level's extent, a shift
//! of 32 or more answering 1. The lane holds that as a rule measured on this M3 Max, so the lane's
//! own check is the rule against itself.
//!
//! **The port re-measures it here.** Record mode runs both kernels on Metal at every level of
//! `LEVELS`, so each recording holds Metal's own words and the rule becomes a committed
//! cross-check (the test `every_recorded_extent_is_the_lanes_rule`). The levels are the ones that
//! make the rule visible: inside the texture's three levels, past them, at 31 and 32 (a shift the
//! width of the word), at 255 and 256 (where Metal takes the level's low eight bits), and at
//! `u32::MAX`.
//!
//! Two cases, because they ask two different things of a device: a **sampled** 2D texture, whose
//! size queries the shader makes of the texture itself, and a pair of **storage** textures (a 2D
//! one and a cube), whose level count a storage image cannot query at all, so the binder pushes
//! it. A venue that holds neither is two honest gaps rather than one hidden by the other.
//!
//! Every texture is private and holds nothing: the case reads size queries, not texels, so there
//! is nothing to upload and nothing a venue's layout could change.
//!
//! Nothing here can fault a GPU: one compute pass a level, one thread, no texel read or write, no
//! loop.

use crate::case::{call, made, Abort, Bindings, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::texture::TextureRequest;
use crate::transcript::{ByteLayout, Digest, Element, Node, Observed, Sites, Value};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The levels each query is asked at: inside a three-level texture's count, past it, and past 255,
/// where Metal takes the level's low eight bits.
const LEVELS: [u32; 13] = [
    0,
    1,
    2,
    3,
    5,
    7,
    31,
    32,
    255,
    256,
    257,
    0x8000_0001,
    u32::MAX,
];

/// The byte every output buffer holds before a level runs, so a word the kernel did not write is a
/// recorded value rather than whatever the allocator left.
const BLANK: u8 = 0xab;

/// Metal's extent at level `level` of a texture whose first level's extent is `extent`:
/// `max(1, extent >> (level & 255))`, a shift of 32 or more answering 1.
pub fn extent(extent: u32, level: u32) -> u32 {
    extent.checked_shr(level & 255).unwrap_or(0).max(1)
}

/// Which pair of queries a case asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Which {
    /// `a4_query_dynamic_level` over a sampled 2D texture: its width and height at the level, and
    /// its level count.
    Sampled,
    /// `txt_level_queries` over a storage 2D texture and a storage cube: the 2D texture's level
    /// count, its width and height at the level, and the cube's level count.
    Storage,
}

impl Which {
    fn name(self) -> &'static str {
        match self {
            Which::Sampled => "level-sizes-sampled",
            Which::Storage => "level-sizes-storage",
        }
    }

    /// The fixture, whose entry point is its own name.
    fn fixture(self) -> Fixture {
        Fixture::new(match self {
            Which::Sampled => "a4_query_dynamic_level",
            Which::Storage => "txt_level_queries",
        })
    }

    /// The words the kernel writes.
    fn words(self) -> usize {
        match self {
            Which::Sampled => 3,
            Which::Storage => 4,
        }
    }

    /// The textures the case binds, at texture 0 up.
    fn textures(self) -> Vec<TextureRequest<'static>> {
        match self {
            // A sampled 2D texture of three levels, 37 by 5: an odd extent, so a shift that
            // rounds is visible.
            Which::Sampled => {
                vec![
                    TextureRequest::read_2d(PixelFormat::RGBA8Unorm.as_raw(), 37, 5)
                        .levels(3)
                        .private(),
                ]
            }
            // A storage 2D texture of three levels, 100 by 40, and a storage cube of two levels,
            // 8 texels a face.
            Which::Storage => vec![
                TextureRequest::read_2d(PixelFormat::RGBA8Unorm.as_raw(), 100, 40)
                    .levels(3)
                    .used(TextureUsage::SHADER_WRITE)
                    .private(),
                TextureRequest::read_2d(PixelFormat::RGBA16Float.as_raw(), 8, 8)
                    .cube(8)
                    .levels(2)
                    .used(TextureUsage::SHADER_WRITE)
                    .private(),
            ],
        }
    }
}

/// The case for one pair of queries.
pub struct LevelSizes {
    which: Which,
}

/// Both cases.
pub fn all() -> Vec<LevelSizes> {
    [Which::Sampled, Which::Storage]
        .into_iter()
        .map(|which| LevelSizes { which })
        .collect()
}

impl LevelSizes {
    /// The bytes the output buffer holds, and the layout its words are read under.
    fn output(&self) -> (Vec<u8>, ByteLayout) {
        let bytes = vec![BLANK; self.which.words() * 4];
        let layout = ByteLayout::buffer(0, bytes.len() as u64, Element::U32);
        (bytes, layout)
    }
}

impl Case for LevelSizes {
    fn name(&self) -> &'static str {
        self.which.name()
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.which.fixture();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{}{extension}", fixture.name),
                    entry: fixture.entry.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.which.fixture();
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("buffer:1", &self.output().0),
        ];
        // Each texture's whole descriptor, so a texture whose extent, level count, type, usage or
        // storage moves is `Unrecorded` rather than compared against Metal's answer for another
        // shape.
        for (at, request) in (0..).zip(self.which.textures()) {
            inputs.push(Input::bytes(format!("texture:{at}"), &request.words()));
        }
        // The levels asked for, so a level list that moves is a change to the case's inputs.
        inputs.push(Input::bytes(
            "levels",
            &LEVELS
                .iter()
                .flat_map(|level| level.to_le_bytes())
                .collect::<Vec<u8>>(),
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded = self
            .which
            .fixture()
            .loaded()
            .map_err(|why| stop("library", why))?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, loaded.entry),
        )?;
        let textures = (0..)
            .zip(self.which.textures())
            .map(|(at, request)| made(sites, &format!("texture/{at}"), side.texture(&request)))
            .collect::<Result<Vec<_>, Abort>>()?;
        let (blank, layout) = self.output();
        for level in LEVELS {
            let at = |site: &str| format!("level/{level}/{site}");
            let level_buffer = made(sites, &at("buffer/lod"), side.buffer(&level.to_le_bytes()))?;
            let out = made(sites, &at("buffer/out"), side.buffer(&blank))?;
            let command_buffer = made(
                sites,
                &at("dispatch"),
                side.dispatch_threads(
                    pipeline,
                    &Bindings::new(&textures, &[level_buffer, out]),
                    &[],
                    [1, 1, 1],
                    [1, 1, 1],
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
                side.read_buffer(out, 0, blank.len()),
                |_| Value::Absent,
            )?;
            sites.push(
                Node::branch(format!("level/{level}"), Vec::new())
                    .with(Node::leaf("words", Observed::bytes(read, layout.clone()))),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The words the lane's rule gives at `level` for `which`'s kernel, word by word: what the
    /// recording is held against, and the one thing in this module the case itself does not use.
    fn rule(which: Which, level: u32) -> Vec<u32> {
        match which {
            Which::Sampled => vec![extent(37, level), extent(5, level), 3],
            Which::Storage => vec![3, extent(100, level), extent(40, level), 2],
        }
    }

    /// The rule answers one word per word the kernel writes, so the cross-check covers the whole
    /// read-back rather than its first words.
    #[test]
    fn the_rule_answers_every_word_the_kernel_writes() {
        for which in [Which::Sampled, Which::Storage] {
            for level in LEVELS {
                assert_eq!(rule(which, level).len(), which.words(), "{}", which.name());
            }
        }
    }

    /// The rule's own edges: a shift inside the word, the low eight bits of the level, a shift of
    /// the word's width and one past it.
    #[test]
    fn the_rule_takes_the_levels_low_eight_bits_and_floors_at_one() {
        assert_eq!(extent(37, 0), 37);
        assert_eq!(extent(37, 1), 18);
        assert_eq!(extent(37, 5), 1);
        assert_eq!(extent(37, 31), 1);
        // 32 and 256 are a shift of 32 and of 0: the first answers 1, the second the extent.
        assert_eq!(extent(37, 32), 1);
        assert_eq!(extent(37, 256), 37);
        assert_eq!(extent(37, 257), 18);
        assert_eq!(extent(100, u32::MAX), 1);
    }

    /// Every word of every committed recording is the lane's rule's.
    ///
    /// This is the cross-check the port exists for: the recordings hold what Metal answered on
    /// this M3 Max, and the rule is what the lane fitted. A difference is a finding — Metal's
    /// answer at a level past the texture moved — rather than something to edit away.
    #[test]
    fn every_recorded_extent_is_the_lanes_rule() {
        let mut unrecorded = Vec::new();
        let mut differ = Vec::new();
        for which in [Which::Sampled, Which::Storage] {
            let recording = match crate::recording::read(which.name()) {
                Ok(recording) => recording,
                Err(_) => {
                    unrecorded.push(which.name());
                    continue;
                }
            };
            for answer in &recording.answers {
                for outcome in &answer.outcomes {
                    for level in LEVELS {
                        let node = outcome
                            .transcript
                            .root
                            .child(&format!("level/{level}"))
                            .and_then(|node| node.child("words"));
                        let Some(node) = node else {
                            differ.push(format!("{}: no site for level {level}", which.name()));
                            continue;
                        };
                        let Value::Bytes(bytes) = &node.observed.value else {
                            differ.push(format!("{}: level {level} has no bytes", which.name()));
                            continue;
                        };
                        for (at, want) in rule(which, level).into_iter().enumerate() {
                            let word = u32::from_le_bytes(
                                bytes[at * 4..at * 4 + 4].try_into().expect("four bytes"),
                            );
                            if word != want {
                                differ.push(format!(
                                    "{} level {level} word {at}: Metal {word}, the rule's {want}",
                                    which.name()
                                ));
                            }
                        }
                    }
                }
            }
        }
        assert!(
            differ.is_empty(),
            "Metal's recorded extents differ from the lane's rule:\n{}",
            differ.join("\n")
        );
        assert!(
            unrecorded.is_empty(),
            "no recording yet for {unrecorded:?}: run `conformance record` on a Mac"
        );
    }
}
