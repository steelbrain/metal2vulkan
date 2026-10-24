//! Which pixels a point covers against its viewport.
//!
//! The owned fixture `rt6_restart`'s vertex function places a point of one pixel where buffer 0
//! holds it, at depth `0.25` with `w` 1, and `rt6_render_state`'s `rt6_white` fragment function
//! writes white; the pass is one 8 by 8 `RGBA8Unorm` target cleared to zero. One draw a row, each
//! row a viewport and a position.
//!
//! The two functions come from two fixtures, which is why a render stage names its own library
//! ([`StageFunction::of`]): there is no one library holding both.
//!
//! The viewports are the ones a Metal probe measured: fractional, whole, mirrored and flipped
//! edges, and the positions are the probe's offsets along each axis, from inside the viewport to
//! three times past its edge, plus Metal's own guard-band bound at `1.5` and the next `float`
//! either side of it. Metal keeps a point while its centre lies within `1.5 w` of the viewport's
//! centre on each axis and `0 <= z <= w`, and Vulkan culls one whose centre leaves the clip volume
//! at `w`, so these rows are exactly where the two part.
//!
//! The case records what Metal wrote, not the lane's table: record mode runs every draw on native
//! Metal. The lane's own table (`PROBE_9`) and its fitted rule (`metal_pixels`) are a cross-check
//! this module's tests apply to the recorded coverages, so the port says whether the two agree
//! instead of copying one into the other.
//!
//! **All 294 draws go into one command buffer**: a pass a row and a blit a row, the blits writing
//! one row of a single read-back buffer each, so the whole case costs one commit and one wait
//! rather than 294. One target serves every row, because each pass clears it and the pieces of one
//! command buffer run in the order the case lists them.
//!
//! A row's site is the **set** of pixels the draw covered, because rasterizing one point produces
//! a shape and no order: the comparison's set diff then prints what is missing, what is extra and
//! what both sides hold, which is the one screen a reader needs to see a point that drifted by a
//! pixel or was culled where Metal kept it. A draw that covered nothing records
//! [`Negative::DrewNothing`] rather than an empty set, so "the point was culled" is a stated fact.
//! What the draws put in the pixels they covered is one further site over the whole case, the set
//! of distinct words written, which is white and would move for a colour or a partial coverage
//! value; the two sites together are every target the case produced, without one 256-byte shape a
//! row (294 rows of those put the recording over the suite's 512 KiB ceiling at 1.1 MiB).
//!
//! Nothing here can fault a GPU: one render pass a row, one point, one 8 by 8 target, no loop and
//! no index buffer.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    StageFunction, TargetSpec, ViewportRect,
};
use crate::transcript::{Digest, Negative, Node, Observed, Sites, Value};
use metal2vulkan::metal::{PixelFormat, PrimitiveType};
use metal2vulkan_runtime::Degradation;

/// The fixture the vertex function lives in.
const POSITIONS: Fixture = Fixture {
    name: "rt6_restart",
    entry: "rt6_restart_vertex",
};

/// The fixture the white fragment function lives in.
const WHITE: Fixture = Fixture {
    name: "rt6_render_state",
    entry: "rt6_white",
};

/// The target's side in pixels.
const SIDE: u32 = 8;

/// The target's format: one byte a channel, so "white" is a texel of all ones and the diff reads a
/// pixel as one word.
const FORMAT: PixelFormat = PixelFormat::RGBA8Unorm;

/// Bytes one row of the target takes.
const ROW_PITCH: u32 = SIDE * 4;

/// Bytes one row's whole target takes, which is its slice of the read-back buffer.
const TARGET_BYTES: u32 = SIDE * ROW_PITCH;

/// The probe's viewports, as `origin x`, `origin y`, `width`, `height`: fractional origins, a
/// fractional extent, a whole one, and the two that cover a band of the target.
const VIEWPORTS: [[f64; 4]; 7] = [
    [4.5, 0.0, 1.0, 8.0],
    [4.25, 0.0, 1.5, 8.0],
    [4.7, 0.0, 1.0, 8.0],
    [4.2, 0.0, 1.0, 8.0],
    [0.0, 2.75, 8.0, 2.5],
    [3.25, 2.75, 1.5, 2.5],
    [4.0, 0.0, 1.0, 8.0],
];

/// The probe's offsets along one axis, in the viewport's own normalized coordinates: its centre,
/// then each step out to three times past its edge.
const OFFSETS: [f32; 17] = [
    0.0, 0.5, -0.5, 0.9, -0.9, 0.99, -0.99, 1.01, -1.01, 1.1, -1.1, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0,
];

/// The row's name for a draw through viewport `viewport` at `offset` along `axis`.
fn row_name(viewport: usize, axis: usize, offset: f32) -> String {
    let along = if axis == 0 { "x" } else { "y" };
    // The offset's bits name the row, so the two `float`s either side of 1.5 are distinct sites
    // and a reader can tell them apart.
    format!("v{viewport}/{along}/{:08x}", offset.to_bits())
}

/// Every draw the case makes: the probe's offsets along each axis of each viewport, and Metal's
/// guard-band bound with the next `float` either side of it, which the probe's table does not
/// reach.
fn rows() -> Vec<(usize, usize, f32)> {
    let mut rows = Vec::new();
    for viewport in 0..VIEWPORTS.len() {
        for axis in 0..2 {
            for offset in OFFSETS {
                rows.push((viewport, axis, offset));
            }
            // The `float`s either side of Metal's guard-band bound. 1.5 and -1.5 themselves are
            // already in `OFFSETS`, and a row repeated would name one site twice.
            for offset in [
                1.5f32.next_up(),
                1.5f32.next_down(),
                (-1.5f32).next_down(),
                (-1.5f32).next_up(),
            ] {
                rows.push((viewport, axis, offset));
            }
        }
    }
    rows
}

/// The position bytes for `offset` along `axis`, as the vertex binding takes them inline: the other
/// axis stays at the viewport's centre, as the probe drew it.
fn position(axis: usize, offset: f32) -> Vec<u8> {
    let mut at = [0.0f32; 2];
    at[axis] = offset;
    at.iter().flat_map(|axis| axis.to_le_bytes()).collect()
}

/// The viewport `at`, over the whole depth range.
fn viewport(at: usize) -> ViewportRect {
    let [x, y, width, height] = VIEWPORTS[at];
    ViewportRect::new([x, y], [width, height])
}

/// The target every row draws into: private, since the blit reads it back.
fn target_spec() -> TargetSpec {
    TargetSpec::read(FORMAT, [SIDE; 2])
}

/// The one colour attachment the pipeline writes.
const COLORS: [Option<ColorState>; 1] = [Some(ColorState::new(FORMAT))];

/// The pixels of one row's bytes the draw covered, as the set the row records.
fn coverage(bytes: &[u8]) -> Value {
    let covered: Vec<String> = bytes
        .chunks(4)
        .enumerate()
        .filter(|(_, texel)| texel.iter().any(|channel| *channel != 0))
        .map(|(at, _)| format!("{},{}", (at as u32) % SIDE, (at as u32) / SIDE))
        .collect();
    if covered.is_empty() {
        return Value::Nothing(Negative::DrewNothing);
    }
    Value::set(covered)
}

/// The distinct words one row's draw wrote, as the roll-up site collects them.
///
/// The coverage set says which pixels the draw reached and this says what it put there, so the two
/// together are the whole target without one shape a row. Rolled up over every row it is a set of
/// one word — white — and a draw that wrote some other colour, or a partial coverage value, moves
/// it.
fn written_words(bytes: &[u8]) -> Vec<String> {
    let mut words: Vec<String> = bytes
        .chunks(4)
        .filter(|texel| texel.iter().any(|channel| *channel != 0))
        .map(|texel| {
            format!(
                "{:#010x}",
                u32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]])
            )
        })
        .collect();
    words.sort();
    words.dedup();
    words
}

/// The case.
pub struct WidePoints;

/// The case the suite holds.
pub fn all() -> Vec<WidePoints> {
    vec![WidePoints]
}

impl Case for WidePoints {
    fn name(&self) -> &'static str {
        "point-viewport-coverage"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // It loosens the notices site alone: every coverage still compares exactly, and the
        // comparison fails if the notice did not fire, so a device that stops needing it has to
        // drop this line rather than carry a stale excuse.
        //
        // A device whose point size is not Metal's draws the point through the runtime's own
        // handling and says so. Measured: lavapipe announces it here and MoltenVK does not, so the
        // case reads `announced` on lavapipe and `exact` on MoltenVK. No other notice fired for
        // this case on either venue, and a tolerance no run justifies is not carried.
        //
        // The same device also converts the float the fragment writes to the normalized colour
        // attachment in the shader (`shader-color-conversion`, lavapipe). Declared, not loosened:
        // every coverage still compares exactly.
        //
        // The positions and the read-back are `Shared`, and a device with no unified memory keeps
        // a `Shared` allocation in system memory because that is the only memory both sides can
        // reach (`shared-in-system-memory`). That is the venue's memory model, not this case's
        // point.
        [
            Degradation::DevicePointSize,
            Degradation::ShaderColorConversion,
        ]
        .into_iter()
        .map(|kind| crate::replay::notice_allowance(kind.code()))
        .chain([
            // The positions and the read-back are `Shared`
            // ([`crate::replay::shared_memory_notice`]).
            crate::replay::shared_memory_notice(),
        ])
        .collect()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for fixture in [POSITIONS, WHITE] {
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{}{extension}", fixture.name),
                    entry: fixture.entry.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = Vec::new();
        for fixture in [POSITIONS, WHITE] {
            for (name, extension) in [("air", RUNTIME_TEXT), ("air-typed", METAL_TEXT)] {
                inputs.push(Input {
                    name: format!("{name}:{}", fixture.name),
                    digest: Digest::sha256(fixture.text(extension)?.as_bytes()),
                });
            }
        }
        inputs.push(Input::bytes("target", &target_spec().words()));
        // The viewports and the positions are one fingerprint each rather than one a row, because
        // they are generated from two constants: a row that moved moved the whole table, and 301
        // digests a reader scrolls past cost more than they say.
        let viewports: Vec<u8> = (0..VIEWPORTS.len())
            .flat_map(|at| viewport(at).words())
            .collect();
        inputs.push(Input::bytes("viewports", &viewports));
        let positions: Vec<u8> = rows()
            .iter()
            .flat_map(|&(viewport, axis, offset)| {
                let mut words = row_name(viewport, axis, offset).into_bytes();
                words.extend(position(axis, offset));
                words
            })
            .collect();
        inputs.push(Input::bytes("positions", &positions));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let vertex_source = POSITIONS
            .loaded()
            .map_err(|why| stop("library/vertex", why))?;
        let fragment_source = WHITE
            .loaded()
            .map_err(|why| stop("library/fragment", why))?;
        let vertex_library = made(
            sites,
            "library/vertex",
            side.library(&vertex_source.source()),
        )?;
        let fragment_library = made(
            sites,
            "library/fragment",
            side.library(&fragment_source.source()),
        )?;
        // The vertex stage comes from the library the call names and the fragment stage names its
        // own, which is the whole reason a stage may name one.
        let pipeline = made(
            sites,
            "pipeline",
            side.render_pipeline(
                vertex_library,
                &RenderPipelineSpec::of(
                    StageFunction::named(POSITIONS.entry),
                    Some(StageFunction::of(fragment_library, WHITE.entry)),
                    &COLORS,
                ),
            ),
        )?;
        // One target for every row: each pass clears it, and the pieces of one command buffer run
        // in the order the case lists them, so a row sees only its own draw.
        let target = made(sites, "target", side.target(&target_spec()))?;
        let rows = rows();
        let read_back = made(
            sites,
            "buffer/read-back",
            side.zeroed_buffer(rows.len() * TARGET_BYTES as usize),
        )?;

        let mut work = Vec::with_capacity(rows.len() * 2);
        for (at, &(viewport_at, axis, offset)) in rows.iter().enumerate() {
            work.push(RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, [0.0; 4])]).drawing(vec![
                    RenderStep::Viewport(viewport(viewport_at)),
                    RenderStep::Pipeline(pipeline),
                    // The position reaches buffer 0 inline: two floats are what
                    // `setVertexBytes:length:atIndex:` is for, and a buffer a row would be 294
                    // allocations saying the same thing.
                    RenderStep::VertexBytes {
                        bytes: position(axis, offset),
                        index: 0,
                    },
                    RenderStep::Draw {
                        primitive: PrimitiveType::Point,
                        first: 0,
                        count: 1,
                        instances: 1,
                    },
                ]),
            ));
            work.push(RenderWork::ReadTexture {
                texture: target,
                into: read_back,
                offset: at as u64 * u64::from(TARGET_BYTES),
                row_pitch: ROW_PITCH,
                size: [SIDE; 2],
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
            side.read_buffer(read_back, 0, rows.len() * TARGET_BYTES as usize),
            |_| Value::Absent,
        )?;

        let mut written: Vec<String> = Vec::new();
        for (at, &(viewport_at, axis, offset)) in rows.iter().enumerate() {
            let name = row_name(viewport_at, axis, offset);
            let first = at * TARGET_BYTES as usize;
            let target_bytes = &bytes[first..first + TARGET_BYTES as usize];
            written.extend(written_words(target_bytes));
            let mut row = Node::branch(name, Vec::new()).with(Node::leaf(
                "coverage",
                Observed::value(coverage(target_bytes)),
            ));
            // What each of the row's two pieces of work answered, which is `None` on Metal's side:
            // a piece of ours that refuses compares against Metal answering and reads as a refusal
            // rather than a divergence.
            for (piece, what) in [(2 * at, "pass"), (2 * at + 1, "read")] {
                if let Some(Some(failure)) = outcome.refused.get(piece) {
                    row = row.with(Node::leaf(what, Observed::value(failure.recorded())));
                }
            }
            sites.push(row);
        }
        written.sort();
        written.dedup();
        // What every draw of the case put in the pixels it covered, rolled up: the fragment
        // function writes opaque white and nothing blends, so this is one word, and a draw that
        // wrote another colour or a partial coverage value moves this site rather than hiding
        // inside a coverage set that happens to match.
        sites.leaf("written-words", Observed::value(Value::set(written)));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// No pixel drawn, in the lane's table.
    const NONE: u8 = u8::MAX;

    /// The pixel the lane's own Metal probe stored, `8 x + y`, through each viewport at each
    /// offset along `x`, then along `y`; [`NONE`] where it drew none. Copied from
    /// `runtime_lane::wide_points::PROBE_9` to be checked against the recording, never to stand in
    /// for it.
    const PROBE_9: [[[u8; 17]; 2]; 7] = [
        [
            [
                35, 43, 35, 43, 35, 43, 35, 43, 35, 43, 35, 43, 35, NONE, NONE, NONE, NONE,
            ],
            [
                35, 33, 37, 32, 39, 32, 39, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
                NONE,
            ],
        ],
        [
            [
                35, 43, 35, 43, 35, 43, 35, 43, 35, 43, 35, NONE, NONE, NONE, NONE, NONE, NONE,
            ],
            [
                35, 33, 37, 32, 39, 32, 39, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
                NONE,
            ],
        ],
        [
            [
                43, 43, 35, 43, 35, 43, 35, 43, 35, 43, 35, 43, 35, NONE, NONE, NONE, NONE,
            ],
            [
                43, 41, 45, 40, 47, 40, 47, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
                NONE,
            ],
        ],
        [
            [
                35, 35, 35, 43, 35, 43, 35, 43, 35, 43, 35, 43, NONE, NONE, NONE, NONE, NONE,
            ],
            [
                35, 33, 37, 32, 39, 32, 39, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
                NONE,
            ],
        ],
        [
            [
                27, 43, 11, 59, 3, 59, 3, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
                NONE,
            ],
            [
                27, 27, 28, 26, 29, 26, 29, 26, 29, 26, 29, 26, 29, NONE, NONE, NONE, NONE,
            ],
        ],
        [
            [
                27, 35, 27, 35, 27, 35, 27, 35, 27, 35, 27, NONE, NONE, NONE, NONE, NONE, NONE,
            ],
            [
                27, 27, 28, 26, 29, 26, 29, 26, 29, 26, 29, 26, 29, NONE, NONE, NONE, NONE,
            ],
        ],
        [
            [
                35, 35, 35, 35, 35, 35, 35, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
                NONE,
            ],
            [
                35, 33, 37, 32, 39, 32, 39, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
                NONE,
            ],
        ],
    ];

    /// The pixels `(x, y)` the lane's fitted rule keeps of a point of `size` at the clip-space
    /// `(x, y, z, w)` through `viewport` in an 8 by 8 target: Metal keeps the point while its
    /// centre lies within `1.5 w` on each axis and `0 <= z <= w`, and covers the pixels whose
    /// centres its square `[c - s/2, c + s/2)` reaches, inside the pixels the viewport overlaps.
    fn fitted_pixels(viewport: [f64; 4], [x, y, z, w]: [f32; 4], size: f64) -> Vec<(u32, u32)> {
        let guard = 1.5 * f64::from(w);
        let drawn = w > 0.0
            && (0.0..=w).contains(&z)
            && f64::from(x).abs() <= guard
            && f64::from(y).abs() <= guard;
        if !drawn {
            return Vec::new();
        }
        let [origin_x, origin_y, width, height] = viewport;
        let centre = [
            origin_x + (f64::from(x / w) + 1.0) / 2.0 * width,
            origin_y + (1.0 - f64::from(y / w)) / 2.0 * height,
        ];
        let reached = |origin: f64, extent: f64| {
            let (low, high) = (origin.min(origin + extent), origin.max(origin + extent));
            (low.floor().max(0.0) as u32)..(high.ceil().min(f64::from(SIDE)) as u32)
        };
        let covers = |pixel: u32, centre: f64| {
            let at = f64::from(pixel) + 0.5;
            centre - size / 2.0 <= at && at < centre + size / 2.0
        };
        let mut kept = Vec::new();
        for pixel_y in reached(origin_y, height) {
            for pixel_x in reached(origin_x, width) {
                if covers(pixel_x, centre[0]) && covers(pixel_y, centre[1]) {
                    kept.push((pixel_x, pixel_y));
                }
            }
        }
        kept
    }

    /// The pixels a recorded coverage set names, as the rule spells them.
    fn recorded(value: &Value) -> Option<BTreeSet<String>> {
        match value {
            Value::Set(covered) => Some(covered.clone()),
            Value::Nothing(Negative::DrewNothing) => Some(BTreeSet::new()),
            _ => None,
        }
    }

    /// The pixels the rule keeps, as the coverage set spells them.
    fn as_set(pixels: &[(u32, u32)]) -> BTreeSet<String> {
        pixels.iter().map(|(x, y)| format!("{x},{y}")).collect()
    }

    /// The lane's table is its own fitted rule's, row for row: the two expectations the lane
    /// carried agree before either is compared with Metal's bytes.
    #[test]
    fn the_lane_s_table_is_its_own_fitted_rule_s() {
        let mut checked = 0usize;
        for (at, viewport) in VIEWPORTS.iter().enumerate() {
            for (axis, codes) in PROBE_9[at].iter().enumerate() {
                for (&offset, &code) in OFFSETS.iter().zip(codes) {
                    let mut position = [0.0f32; 2];
                    position[axis] = offset;
                    let expected: Vec<(u32, u32)> = (code != NONE)
                        .then(|| (u32::from(code / SIDE as u8), u32::from(code % SIDE as u8)))
                        .into_iter()
                        .collect();
                    assert_eq!(
                        fitted_pixels(*viewport, [position[0], position[1], 0.25, 1.0], 1.0),
                        expected,
                        "viewport {viewport:?} at {position:?}"
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, VIEWPORTS.len() * 2 * OFFSETS.len());
    }

    /// Each row names a distinct site and a distinct position, so the comparison pairs the sides'
    /// rows and the input fingerprint tells them apart.
    #[test]
    fn the_rows_name_distinct_sites_and_positions() {
        let rows = rows();
        let named: BTreeSet<String> = rows
            .iter()
            .map(|&(viewport, axis, offset)| row_name(viewport, axis, offset))
            .collect();
        assert_eq!(named.len(), rows.len());
    }

    /// Every coverage Metal recorded holds exactly the pixels the lane's fitted rule keeps.
    ///
    /// This is the cross-check the port owes the lane: the lane carried a probe's table and a rule,
    /// the case carries Metal's coverages, and this says whether they agree — over the rows the
    /// table never reached as well, since the rule covers those. It reads the recording, so it runs
    /// where there is no GPU.
    #[test]
    fn metal_s_recorded_coverages_hold_the_fitted_rule_s_pixels() {
        let case = WidePoints;
        let Ok(recording) = crate::recording::read(case.name()) else {
            println!("{}: no recording in the tree yet", case.name());
            return;
        };
        let mut checked = 0usize;
        let mut differ = Vec::new();
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                for (at, axis, offset) in rows() {
                    let name = row_name(at, axis, offset);
                    let Some(got) = outcome
                        .transcript
                        .root
                        .child(&name)
                        .and_then(|node| node.child("coverage"))
                        .and_then(|node| recorded(&node.observed.value))
                    else {
                        continue;
                    };
                    let mut position = [0.0f32; 2];
                    position[axis] = offset;
                    let expected = as_set(&fitted_pixels(
                        VIEWPORTS[at],
                        [position[0], position[1], 0.25, 1.0],
                        1.0,
                    ));
                    checked += 1;
                    if got != expected {
                        differ.push(format!("{name}: Metal {got:?}, the rule {expected:?}"));
                    }
                }
            }
        }
        assert!(checked > 0, "the recording holds no row");
        for line in differ.iter().take(20) {
            println!("WIDE-POINTS-RULE-DIFFERS: {line}");
        }
        assert!(
            differ.is_empty(),
            "{} of {checked} recorded coverages differ from the lane's fitted rule",
            differ.len()
        );
    }
}
