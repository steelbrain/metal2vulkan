//! Implicit-LOD samples and a fine derivative pair after each construct a fragment reconverges
//! from, and through a texture a ternary on two texture handles chooses.
//!
//! Two owned fixtures, one case each:
//!
//! - `reconvergence-constructs` (`reconverge_sample`) draws eight fragment functions, each after or
//!   inside one construct: an if/else on data, a loop of the pixel's own trip count, a break to
//!   the loop's own merge, a switch of one label per case, a discard, selections and a loop
//!   uniform over 8 by 8 blocks (`rc_grid`), and a uniform loop's second iteration after one pixel
//!   of each quad ended the first early, by a continue to the loop's own continue target
//!   (`rc_continue_quad`) or by an if/else at the body's end (`rc_latches`).
//! - `reconvergence-handle-choices` (`reconverge_select`) samples inside a selection uniform over
//!   8 by 8 blocks through a texture a ternary on two handles chooses: per pixel (`rc_select`), per
//!   4 by 4 block (`rc_select_quad`), and by a constant the whole draw shares (`rc_pick`), the last
//!   also over 64 small triangles, one in each 8 by 8 cell, whose edges cross quads
//!   (`rc_pick_cells`).
//!
//! Each fragment writes `(nearest, dfdx(v), dfdy(v), trilinear)`. The nearest sample is of an
//! 11-level `R16Float` source of 1024 by 2 texels whose level n holds n + 1 (21 + n in the second
//! source), at a coordinate whose scale is 2^q texels a pixel for a q uniform over each quad, so
//! the implicit level of detail is exactly q and the sample reads level q's value. The pair is the
//! fine derivative of the lane's own value v, and the trilinear sample is at a level of detail of
//! q + log2(1.5). Each reads the quad's other lanes.
//!
//! Metal defines every value these draws observe. Each is taken where every lane of the quad
//! arrives together: after a construct that closed (an if/else, a loop of the pixel's own trip
//! count, a break to the loop's own merge, a switch, a discard, which Metal runs as a demote so
//! the discarded lane stays a helper), in the iteration after a loop's own back edge (the
//! continue to the loop's own continue target, and the if/else whose arms both end the
//! iteration), or inside control flow uniform over the quad. A construct after which MSL leaves a
//! derivative undefined (an early return inside a selection, a break or a continue to a
//! construct around the innermost one, a construct a quad entered diverged) is not here, because
//! Metal's answer there would be a device's luck rather than a contract.
//!
//! Each draw covers a 64 by 64 `RGBA16Float` target cleared to -1, read back through a blit. A
//! draw's row holds two sites over its texels, each a column of the channels, packed so that the
//! recording holds each texel once:
//!
//! - `level-and-pair`: the nearest sample and the fine pair, three halves a pixel. A row of the
//!   layout is a row of pixels, and its element 3x + c is channel c of pixel x. This is what
//!   reconvergence decides.
//! - `trilinear`: the trilinear sample, one half a pixel, element x of row y. Its bits are the
//!   device's filter and level-of-detail arithmetic (rule 1, `device-sampling`), so where a venue
//!   answers other bits than Metal's, the site is a `known-gaps-rt.tsv` row scoped to that venue,
//!   and the venue's `level-and-pair` site still compares exactly.
//!
//! The clear value is part of the answer: a pixel a draw discards or does not cover holds -1 in
//! every channel.
//!
//! Nothing here can fault a GPU: one triangle (or 64 small ones) a draw over one 64 by 64 target,
//! and every loop in the fixtures is bounded by a constant or by a data word below 7.

use crate::case::{call, drawn_or_refused, made, Abort, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    StageFunction, TargetSpec,
};
use crate::texture::{TextureRequest, Upload};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType};

/// The target's side in pixels.
const SIDE: u32 = 64;

/// The target's format: four halves a pixel.
const FORMAT: PixelFormat = PixelFormat::RGBA16Float;

/// Bytes one row of the target takes.
const ROW_PITCH: u32 = SIDE * 8;

/// Bytes the whole target takes, which is its draw's slice of the read-back buffer.
const TARGET_BYTES: u32 = SIDE * ROW_PITCH;

/// Every channel of the clear value.
const CLEAR: f64 = -1.0;

/// The sources' size at level 0, and their levels: down to 1 by 1.
const SOURCE: [u32; 2] = [1024, 2];
const LEVELS: u32 = 11;

/// What `pick` holds: the second source.
const PICK: u32 = 1;

/// One draw: the site its row is compared at, its vertex function, its fragment function, its
/// vertex count, and whether its fragment reads `pick` at buffer 1, which the draw then binds. No
/// other draw binds it: Metal's API validation reports a bound buffer no function reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Draw {
    site: &'static str,
    vertex: &'static str,
    fragment: &'static str,
    vertices: u32,
    pick: bool,
}

impl Draw {
    /// The draw of `fragment` over one triangle that covers the target.
    const fn covering(vertex: &'static str, fragment: &'static str) -> Self {
        Self {
            site: fragment,
            vertex,
            fragment,
            vertices: 3,
            pick: false,
        }
    }

    /// The draw, binding `pick`.
    const fn picking(self) -> Self {
        Self { pick: true, ..self }
    }
}

/// `reconverge_sample`'s draws, in order.
const CONSTRUCTS: [Draw; 8] = [
    Draw::covering("rc_cover", "rc_if"),
    Draw::covering("rc_cover", "rc_loop"),
    Draw::covering("rc_cover", "rc_break"),
    Draw::covering("rc_cover", "rc_switch"),
    Draw::covering("rc_cover", "rc_discard"),
    Draw::covering("rc_cover", "rc_grid"),
    Draw::covering("rc_cover", "rc_continue_quad"),
    Draw::covering("rc_cover", "rc_latches"),
];

/// `reconverge_select`'s draws, in order: the last is `rc_pick` over a small triangle in each 8
/// by 8 cell (`rs_cells`, three vertices a cell).
const HANDLE_CHOICES: [Draw; 4] = [
    Draw::covering("rs_cover", "rc_select"),
    Draw::covering("rs_cover", "rc_select_quad"),
    Draw::covering("rs_cover", "rc_pick").picking(),
    Draw {
        site: "rc_pick_cells",
        vertex: "rs_cells",
        fragment: "rc_pick",
        vertices: 3 * 64,
        pick: true,
    },
];

/// The one colour attachment every pipeline writes.
const COLORS: [Option<ColorState>; 1] = [Some(ColorState::new(FORMAT))];

/// The target every draw renders into: private, since the blit reads it back.
fn target_spec() -> TargetSpec {
    TargetSpec::read(FORMAT, [SIDE, SIDE])
}

/// The pipeline of `draw`: its vertex and fragment functions out of the case's one library.
fn pipeline_spec(draw: &Draw) -> RenderPipelineSpec<'static> {
    RenderPipelineSpec::of(
        StageFunction::named(draw.vertex),
        Some(StageFunction::named(draw.fragment)),
        &COLORS,
    )
}

/// The 256 data words every fragment reads at buffer 0: word i a hash of i, and words 250 to 252
/// the uniform loops' trip count (2), sampling iteration (1) and continuing iteration (0).
fn words() -> Vec<u32> {
    let mut words: Vec<u32> = (0..256u32)
        .map(|i| {
            let mut x = i.wrapping_mul(0x9E37_79B1);
            x ^= x >> 16;
            x = x.wrapping_mul(0x85EB_CA6B);
            x ^ (x >> 13)
        })
        .collect();
    words[250..253].copy_from_slice(&[2, 1, 0]);
    words
}

/// The data words' bytes.
fn data_bytes() -> Vec<u8> {
    words().iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// The half of a small integer `value` (below 2048, so exact).
fn half_of(value: u32) -> u16 {
    let e = value.ilog2();
    ((15 + e) << 10 | ((value << 10) >> e) & 1023) as u16
}

/// Level `level`'s size in texels.
fn level_size(level: u32) -> [u32; 2] {
    SOURCE.map(|side| (side >> level).max(1))
}

/// A source's levels, level 0 first: level n holds `base + n` in every texel.
fn source_levels(base: u32) -> Vec<Vec<u8>> {
    (0..LEVELS)
        .map(|level| {
            let [width, height] = level_size(level);
            (0..width * height)
                .flat_map(|_| half_of(base + level).to_le_bytes())
                .collect()
        })
        .collect()
}

/// The uploads that load `levels`, one a level.
fn uploads(levels: &[Vec<u8>]) -> Vec<Upload<'_>> {
    levels
        .iter()
        .zip(0u32..)
        .map(|(bytes, level)| {
            let [width, height] = level_size(level);
            Upload {
                level,
                slice: 0,
                origin: [0, 0, 0],
                size: [width, height, 1],
                bytes,
                row_pitch: width * 2,
                rows: height,
                image_pitch: 0,
            }
        })
        .collect()
}

/// The request of a source loaded with `uploads`: shared, read by a shader, every level.
fn source_request<'a>(uploads: &'a [Upload<'a>]) -> TextureRequest<'a> {
    TextureRequest::read_2d(PixelFormat::R16Float.as_raw(), SOURCE[0], SOURCE[1])
        .levels(LEVELS)
        .loaded(uploads)
}

/// The region every site's texels are of: the whole target.
fn target_region() -> Region {
    Region::Texture {
        level: 0,
        slice: 0,
        aspect: Aspect::Color,
        sample: 0,
        origin: [0, 0, 0],
        size: [SIDE, SIDE, 1],
    }
}

/// The layout of a column of `channels` halves a pixel, a row of the layout a row of pixels.
fn column_layout(channels: u32) -> ByteLayout {
    ByteLayout {
        region: target_region(),
        element: Element::F16,
        row_offset: 0,
        row_elements: SIDE * channels,
        row_pitch: SIDE * channels * 2,
        rows: SIDE,
        image_pitch: SIDE * SIDE * channels * 2,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// A target's texels split into its two columns: the first three channels of each pixel, and
/// the fourth.
fn columns(target: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut level_and_pair = Vec::with_capacity(target.len() / 4 * 3);
    let mut trilinear = Vec::with_capacity(target.len() / 4);
    for texel in target.chunks(8) {
        level_and_pair.extend_from_slice(&texel[..6]);
        trilinear.extend_from_slice(&texel[6..]);
    }
    (level_and_pair, trilinear)
}

/// One of the two cases: a fixture and the draws over it.
pub struct Reconvergence {
    name: &'static str,
    fixture: &'static str,
    draws: &'static [Draw],
    /// The first level's value of each source the fragments bind, at textures 0 up.
    sources: &'static [u32],
}

/// The cases the suite holds.
pub fn all() -> Vec<Reconvergence> {
    vec![
        Reconvergence {
            name: "reconvergence-constructs",
            fixture: "reconverge_sample",
            draws: &CONSTRUCTS,
            sources: &[1],
        },
        Reconvergence {
            name: "reconvergence-handle-choices",
            fixture: "reconverge_select",
            draws: &HANDLE_CHOICES,
            sources: &[1, 21],
        },
    ]
}

impl Reconvergence {
    /// The fixture under the entry point `entry`.
    fn fixture(&self, entry: &'static str) -> Fixture {
        Fixture {
            name: self.fixture,
            entry,
        }
    }

    /// Whether any draw binds `pick`.
    fn picks(&self) -> bool {
        self.draws.iter().any(|draw| draw.pick)
    }

    /// Every entry the case's pipelines name, once each, vertex functions first.
    fn entries(&self) -> Vec<&'static str> {
        let mut entries: Vec<&'static str> = Vec::new();
        for draw in self.draws {
            if !entries.contains(&draw.vertex) {
                entries.push(draw.vertex);
            }
        }
        for draw in self.draws {
            if !entries.contains(&draw.fragment) {
                entries.push(draw.fragment);
            }
        }
        entries
    }
}

impl Case for Reconvergence {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for entry in self.entries() {
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{}{extension}", self.fixture),
                    entry: entry.to_owned(),
                    source_sha256: crate::sha256_hex(
                        self.fixture(entry).text(extension)?.as_bytes(),
                    ),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture(self.draws[0].fragment);
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("target", &target_spec().words()),
            Input::bytes(
                "pipelines",
                &self
                    .draws
                    .iter()
                    .flat_map(|draw| pipeline_spec(draw).words())
                    .collect::<Vec<u8>>(),
            ),
            Input::bytes(
                "draws",
                self.draws
                    .iter()
                    .map(|draw| format!("{}:{}:{}", draw.site, draw.vertices, draw.pick))
                    .collect::<Vec<String>>()
                    .join(",")
                    .as_bytes(),
            ),
            Input::bytes("buffer:0", &data_bytes()),
            Input::bytes("clear", &CLEAR.to_le_bytes()),
        ];
        if self.picks() {
            inputs.push(Input::bytes("buffer:1", &PICK.to_le_bytes()));
        }
        for (index, base) in self.sources.iter().enumerate() {
            let levels = source_levels(*base);
            let uploads = uploads(&levels);
            inputs.push(Input::bytes(
                format!("texture:{index}"),
                &source_request(&uploads).words(),
            ));
            for (level, bytes) in levels.iter().enumerate() {
                inputs.push(Input::bytes(
                    format!("texture:{index}:level-{level}"),
                    bytes,
                ));
            }
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = self
            .fixture(self.draws[0].fragment)
            .loaded()
            .map_err(|why| Abort {
                at: "library".to_owned(),
                failure: Failure::named("FixtureNotRead", why),
            })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        // One target for every draw: each pass clears it, and the pieces of one command buffer
        // run in the order the case lists them, so a row sees only its own draw.
        let target = made(sites, "target", side.target(&target_spec()))?;
        let mut sources = Vec::with_capacity(self.sources.len());
        for (index, base) in self.sources.iter().enumerate() {
            let levels = source_levels(*base);
            let uploads = uploads(&levels);
            sources.push(made(
                sites,
                &format!("texture/{index}"),
                side.texture(&source_request(&uploads)),
            )?);
        }
        let data = made(sites, "buffer/0", side.buffer(&data_bytes()))?;
        let pick = match self.picks() {
            true => Some(made(sites, "buffer/1", side.buffer(&PICK.to_le_bytes()))?),
            false => None,
        };
        let read_back = made(
            sites,
            "buffer/read-back",
            side.zeroed_buffer(self.draws.len() * TARGET_BYTES as usize),
        )?;

        let mut pipelines = Vec::with_capacity(self.draws.len());
        for draw in self.draws {
            pipelines.push(made(
                sites,
                &format!("{}/pipeline", draw.site),
                side.render_pipeline(library, &pipeline_spec(draw)),
            )?);
        }
        // Every draw in one command buffer: a pass and a blit each, the blits writing one slice of
        // the read-back buffer apiece, so the case costs one commit and one wait.
        let mut work = Vec::with_capacity(self.draws.len() * 2);
        for (at, (draw, pipeline)) in self.draws.iter().zip(&pipelines).enumerate() {
            let mut steps = vec![
                RenderStep::Pipeline(*pipeline),
                RenderStep::FragmentBuffer {
                    buffer: data,
                    offset: 0,
                    index: 0,
                },
            ];
            if let (true, Some(pick)) = (draw.pick, pick) {
                steps.push(RenderStep::FragmentBuffer {
                    buffer: pick,
                    offset: 0,
                    index: 1,
                });
            }
            for (index, source) in (0u32..).zip(&sources) {
                steps.push(RenderStep::FragmentTexture {
                    texture: *source,
                    index,
                });
            }
            steps.push(RenderStep::Draw {
                primitive: PrimitiveType::Triangle,
                first: 0,
                count: draw.vertices,
                instances: 1,
            });
            work.push(RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, [CLEAR; 4])]).drawing(steps),
            ));
            work.push(RenderWork::ReadTexture {
                texture: target,
                into: read_back,
                offset: at as u64 * u64::from(TARGET_BYTES),
                row_pitch: ROW_PITCH,
                size: [SIDE, SIDE],
                plane: Plane::Whole,
                at: ReadAt::BASE,
            });
        }
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(read_back, 0, self.draws.len() * TARGET_BYTES as usize),
            |_| Value::Absent,
        )?;

        for (at, draw) in self.draws.iter().enumerate() {
            let first = at * TARGET_BYTES as usize;
            let (level_and_pair, trilinear) = columns(&bytes[first..first + TARGET_BYTES as usize]);
            let row = Node::branch(draw.site, Vec::new())
                .with(Node::leaf(
                    "level-and-pair",
                    Observed::bytes(level_and_pair, column_layout(3)),
                ))
                .with(Node::leaf(
                    "trilinear",
                    Observed::bytes(trilinear, column_layout(1)),
                ));
            // A row whose pass or read our side refused has no texels: the refusal is the row's
            // answer and compares as one.
            let refused = (outcome.refusal(2 * at)).or_else(|| outcome.refusal(2 * at + 1));
            sites.push(drawn_or_refused(row, refused));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The fixtures' hash of `(x, y)`.
    fn hash(x: u32, y: u32) -> u32 {
        let mut h = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77);
        h ^= h >> 15;
        h = h.wrapping_mul(0x2C1B_3C6D);
        h ^ (h >> 12)
    }

    /// The value of half `bits`, for the halves these targets hold (no NaN or infinity).
    fn half(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = i32::from((bits >> 10) & 31);
        let fraction = f32::from(bits & 1023);
        sign * match exponent {
            0 => fraction * 2f32.powi(-24),
            _ => (1.0 + fraction / 1024.0) * 2f32.powi(exponent - 15),
        }
    }

    /// The nearest sample `draw` writes at pixel `(x, y)`, and for `rc_pick` the fine pair too,
    /// as the fixtures compute them: `None` where the model does not say (a pixel `rc_pick_cells`
    /// does not cover is checked apart).
    fn model(draw: &Draw, words: &[u32], x: u32, y: u32) -> (f32, Option<[f32; 2]>) {
        let word = |at: u32| words[(at & 255) as usize];
        let h = hash(x, y);
        let b = hash(x >> 3, y >> 3);
        let q = hash(x >> 1, y >> 1) % 8;
        let level = |base: u32| (base + q) as f32;
        let v = |x: u32, y: u32| (word(hash(x, y)) & 255) as f32;
        let pair = [v(x | 1, y) - v(x & !1, y), v(x, y | 1) - v(x, y & !1)];
        match draw.fragment {
            "rc_discard" if word(h) % 5 == 0 => (CLEAR as f32, None),
            "rc_grid" => {
                if word(b) & 1 == 0 {
                    return ((word(b >> 24) & 15) as f32, None);
                }
                if word(b >> 8) & 1 == 0 {
                    return (level(1), None);
                }
                let trips = 1 + word(b >> 16) % 3;
                let sum = (0..trips)
                    .filter(|&i| word(b.wrapping_add(i)) & 2 != 0)
                    .map(|i| ((q + i) % 8 + 1) as f32)
                    .sum();
                (sum, None)
            }
            "rc_select" | "rc_select_quad" | "rc_pick" if word(b) & 1 == 0 => (0.0, None),
            "rc_select" => (level(if word(h >> 8) & 1 != 0 { 21 } else { 1 }), None),
            "rc_select_quad" => {
                let c = hash(x >> 2, y >> 2);
                (level(if word(c >> 8) & 1 != 0 { 21 } else { 1 }), None)
            }
            "rc_pick" => (level(if PICK != 0 { 21 } else { 1 }), Some(pair)),
            _ => (level(1), None),
        }
    }

    /// The case's bookkeeping holds: each case's sites are distinct, the columns' layouts cover
    /// the split target exactly, and every entry a pipeline names is one its fixture defines.
    #[test]
    fn the_draws_the_layouts_and_the_fixtures_agree() {
        for case in all() {
            let sites: BTreeSet<&str> = case.draws.iter().map(|draw| draw.site).collect();
            assert_eq!(sites.len(), case.draws.len(), "{}", case.name);
            let text = case
                .fixture(case.draws[0].fragment)
                .text(RUNTIME_TEXT)
                .expect("the fixture's text");
            for entry in case.entries() {
                assert!(
                    text.contains(&format!("@{entry}(")),
                    "{}: {entry} is not in {}",
                    case.name,
                    case.fixture
                );
            }
        }
        let (level_and_pair, trilinear) = columns(&vec![0; TARGET_BYTES as usize]);
        assert_eq!(level_and_pair.len(), column_layout(3).extent());
        assert_eq!(trilinear.len(), column_layout(1).extent());
        // Every level value is a small integer, so its half is exact.
        for value in 1..=31 {
            assert_eq!(half(half_of(value)), value as f32);
        }
    }

    /// Every nearest sample Metal recorded reads the level the fixtures' hash chose, every pixel a
    /// draw discarded or did not cover holds the clear value, and `rc_pick`'s fine pair is the
    /// difference the fixture's values make.
    ///
    /// It reads the recordings, so it runs where there is no GPU, and says whether Metal's answer
    /// is the one the module describes; it never stands in for that answer.
    #[test]
    fn metal_s_recorded_samples_read_the_level_the_hash_chose() {
        let words = words();
        let mut checked = 0usize;
        let mut differ = Vec::new();
        for case in all() {
            let Ok(recording) = crate::recording::read(case.name()) else {
                println!("{}: no recording in the tree yet", case.name());
                continue;
            };
            for recorded in &recording.answers {
                for outcome in &recorded.outcomes {
                    for draw in case.draws {
                        let Some(Value::Bytes(bytes)) = outcome
                            .transcript
                            .root
                            .child(draw.site)
                            .and_then(|node| node.child("level-and-pair"))
                            .map(|node| &node.observed.value)
                        else {
                            continue;
                        };
                        checked += 1;
                        for (at, texel) in bytes.chunks(6).enumerate() {
                            let (x, y) = (at as u32 % SIDE, at as u32 / SIDE);
                            let got: Vec<f32> = texel
                                .chunks(2)
                                .map(|b| half(u16::from_le_bytes([b[0], b[1]])))
                                .collect();
                            let uncovered = got.iter().all(|&c| c == CLEAR as f32);
                            if draw.site == "rc_pick_cells" && uncovered {
                                continue;
                            }
                            let (nearest, pair) = model(draw, &words, x, y);
                            let pair_differs = pair.is_some_and(|pair| pair != [got[1], got[2]]);
                            if got[0] != nearest || pair_differs {
                                differ.push(format!(
                                    "{}: {}: ({x}, {y}) holds {got:?}, the model {nearest} {pair:?}",
                                    case.name(),
                                    draw.site
                                ));
                            }
                        }
                    }
                }
            }
        }
        for line in differ.iter().take(20) {
            println!("RECONVERGENCE-DIFFERS: {line}");
        }
        assert!(
            differ.is_empty(),
            "{} texels differ from the model",
            differ.len()
        );
        println!("{checked} recorded draws read as the model says");
    }
}
