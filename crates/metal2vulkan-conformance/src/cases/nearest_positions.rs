//! Where a nearest filter lands on a repeating and a mirrored axis, through a runtime sampler.
//!
//! The owned fixture `a2_clamp_to_zero` samples one float, one uint and one int texture five ways a
//! thread, and its third result is the sample through the sampler state bound at `[[sampler(0)]]`.
//! Here the float texture is 13 by 11 `R32Float`, neither axis a power of two, whose texel
//! `(x, y)` holds `x + 16 y`; the sampler state is nearest on both filters, repeating on both axes
//! in one row of the case and mirroring in the other. Every lane's coordinate is `(u, 0.5)`, each
//! `u` a texel edge of the 13-texel axis shifted whole periods away from `[0, 1)` and one to three
//! units in the last place either side of it, which is where a device that scales before it wraps
//! reads the next texel.
//!
//! The case records what Metal wrote, not a rule: record mode runs the fixture on native Metal, so
//! the recording holds Metal's whole output buffer, and the runtime lane's fitted rule
//! (`runtime_lane::nearest_positions::metal_texel`) is a cross-check this module's tests apply to
//! the recorded bytes rather than a second expectation to keep in step.
//!
//! One site a row holds the whole output buffer, and its layout's row is one thread's results with
//! the column the case is about: the four words of the runtime sampler's sample. A diff therefore
//! names the lane and the channel, so a sample that moved reads as "lane 231, word 0" of that
//! column rather than a byte offset.
//!
//! The other sixteen words a thread writes — the two constexpr clamp-to-zero samples and the uint
//! and int reads — are **recorded and not compared**, and the reason is a measured one rather than
//! tidiness. At the three coordinates one to three units in the last place below `+0.0`
//! (`0x8000_0001` to `0x8000_0003`, lanes 289, 291 and 293 of the lane's own set), Metal's
//! constexpr `clamp_to_zero` samplers read texel `(0, 5)`, 80, and its linear one 40, where a
//! Vulkan device reads the border, 0; MoltenVK matches Metal and lavapipe does not. That is a
//! difference of the clamp-to-zero addressing mode, not of where a nearest filter lands, it is
//! nobody's announced notice and no accepted difference covers it, and this case must not invent a
//! tolerance for it. The bytes stay in the recording, so widening the column costs a re-record and
//! nothing else once that gap is closed or written down.
//!
//! **Where that border test lies is measured here, not guessed.** Two edge ladders walk up to each
//! clamp-to-zero edge from outside it, `2^-1` down to `2^-40` of a texel, and a bracket of eight
//! magnitudes straddles `float`'s smallest normal. Metal reads the border at **every** rung down to
//! `9e-13` of a texel, so the boundary is not a fixed-point quantum of the coordinate; and it reads
//! the border at the smallest normal `float` (`1.1754944e-38`, bits `0x0080_0000`) while reading
//! texel 0 at the largest subnormal (`1.1754942e-38`, bits `0x007f_ffff`), one unit in the last
//! place away. **The boundary is `float`'s own subnormal edge: Metal flushes a subnormal texture
//! coordinate to zero before the address mode decides, and a Vulkan device that keeps it reads the
//! border.** That is what the fix has to reproduce, and the two tests below hold Metal to it.
//!
//! The sampler's `r` axis is not part of the case: a 2D sample never consults it, so the two rows
//! differ in the `s` and `t` modes alone.
//!
//! Nothing here can fault a GPU: one compute pass a row, three read-only textures, two buffers, no
//! loop, and every sample inside a bound texture.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Bindings, Case, Failure, Handle, Input, MetalSource, SamplerSpec, Side,
    TextureRole, TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{PixelFormat, SamplerAddressMode, SamplerMinMagFilter};

/// The fixture the case runs, whose entry point is its own name.
const FIXTURE: Fixture = Fixture::new("a2_clamp_to_zero");

/// The float texture's size: neither axis a power of two, so a scale-then-wrap device parts from
/// Metal at a texel edge.
const WIDTH: u32 = 13;
const HEIGHT: u32 = 11;

/// The side of the uint and int textures the fixture also samples. They are not the case's
/// subject; they are bound because the kernel declares them, and their words compare like any
/// other bytes Metal wrote.
const SMALL: u32 = 4;

/// The `uint4` results one thread writes: nearest and linear through the constexpr clamp-to-zero
/// samplers, the runtime sampler's sample, and the uint and int reads.
const RESULTS: u32 = 5;

/// The bytes one thread's results take.
const LANE_BYTES: u32 = RESULTS * 16;

/// The byte, inside a thread's results, of the runtime sampler's sample: the fixture's third
/// `uint4`, which is the column this case compares.
const SAMPLED_AT: u32 = 2 * 16;

/// The words the runtime sampler's sample is: the four channels of one `float4`.
const SAMPLED_WORDS: u32 = 4;

/// The compared column lies inside a thread's results, checked where it costs nothing: a column
/// that reached past a lane would read another thread's words.
const _: () = assert!(SAMPLED_AT + SAMPLED_WORDS * 4 <= LANE_BYTES);

/// The byte the output buffer holds before a row runs, so a word no thread wrote is a recorded
/// value rather than whatever the allocator left. The lane's own fill.
const FILL: u8 = 0xab;

/// One row of the case: the address mode both axes take, under the name its sites carry.
struct Row {
    name: &'static str,
    address: SamplerAddressMode,
}

/// Both rows: the repeating axis and the mirrored one, which wrap a coordinate outside `[0, 1)`
/// into one period by the two rules Metal has.
const ROWS: &[Row] = &[
    Row {
        name: "repeat",
        address: SamplerAddressMode::Repeat,
    },
    Row {
        name: "mirrored",
        address: SamplerAddressMode::MirrorRepeat,
    },
];

impl Row {
    /// The sampler state the row makes: nearest on both filters, the row's mode on both axes.
    fn sampler(&self) -> SamplerSpec {
        SamplerSpec::filtered(SamplerMinMagFilter::Nearest).addressed(self.address)
    }
}

/// The lane's own coordinates: each texel edge of the 13-texel axis, shifted by -3 to 3 whole
/// periods (0 left out, since inside `[0, 1)` scaling before the wrap and wrapping before the scale
/// agree), and one to three units in the last place either side of each edge.
///
/// These are the coordinates the lane's fitted rule was fitted over, so they are the ones the
/// rule's cross-check reads.
fn lane_us() -> Vec<f32> {
    let mut out = Vec::new();
    for period in [-3i32, -2, -1, 1, 2, 3] {
        for edge in 0..=WIDTH as i32 {
            let at = (period * WIDTH as i32 + edge) as f32 / WIDTH as f32;
            out.push(at);
            let (mut up, mut down) = (at, at);
            for _ in 0..3 {
                up = up.next_up();
                down = down.next_down();
                out.push(up);
                out.push(down);
            }
        }
    }
    out
}

/// The rungs of each edge ladder: `2^-1` down to `2^-EDGE_RUNGS` of one texel.
///
/// 40 rungs reach an unnormalized offset of about `9e-14`, which is still a normal `float` by a
/// wide margin, so the ladder tells a fixed-point quantum (a boundary at some `2^-k`) apart from
/// denormals being flushed to zero (a boundary near `1.2e-38`, below every rung).
const EDGE_RUNGS: u32 = 40;

/// The coordinates that walk up to each clamp-to-zero edge from outside it: `-(2^-k)/13` below
/// `0`, and `1 + (2^-k)/13` above the last texel, for `k` of 1 to [`EDGE_RUNGS`].
///
/// They are here to locate the one place Metal's constexpr `clamp_to_zero` samplers read a texel
/// where a border is expected (the module's documentation). An unnormalized offset of `2^-k` is
/// what a coordinate quantized to `k - 1` fractional bits rounds to zero from, so the rung where
/// Metal stops reading texel 0 names the quantum — if a quantum is what it is.
fn edge_us() -> Vec<f32> {
    let mut out = Vec::new();
    for rung in 1..=EDGE_RUNGS {
        let offset = (f64::powi(2.0, -(rung as i32)) / f64::from(WIDTH)) as f32;
        out.push(-offset);
        // Past the last texel the rungs below about `6e-8` round to exactly `1.0` in `float`, so
        // the upper ladder's fine end is the coordinate `1.0` itself. That is a coordinate worth
        // recording in its own right, and the ladder's test reads it as such.
        out.push(1.0 + offset);
    }
    out.extend(SUBNORMAL_BRACKET.iter().map(|offset| -offset));
    out
}

/// The magnitudes that bracket `float`'s smallest normal, below the first texel: two normals well
/// above it, the smallest normal itself and the one above it, then the largest subnormal and three
/// subnormals below that.
///
/// The ladder above shows the border test is not at any quantum down to `1e-12` of a texel while
/// the three coordinates one to three units in the last place below `+0.0` read texel 0, so the
/// boundary lies somewhere between. These eight say where, and the one boundary that can lie there
/// is `float`'s own: a subnormal flushed to zero.
const SUBNORMAL_BRACKET: [f32; 8] = [
    1.0e-30,
    1.0e-37,
    f32::MIN_POSITIVE,
    1.2e-38,
    1.1754942e-38,
    1.0e-40,
    1.0e-42,
    1.0e-44,
];

/// Every coordinate the case samples at: the lane's own, then the edge ladders.
fn us() -> Vec<f32> {
    let mut out = lane_us();
    out.extend(edge_us());
    out
}

/// The coordinate buffer's bytes: `(u, 0.5)` a thread, which is the centre row of the 11-texel
/// axis.
fn coords() -> Vec<u8> {
    us().iter()
        .flat_map(|&u| [u, 0.5])
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// The float texture's bytes: texel `(x, y)` holds `x + 16 y` as a `float`, so a sample names the
/// texel it read.
fn float_texels() -> Vec<u8> {
    (0..HEIGHT)
        .flat_map(|y| (0..WIDTH).map(move |x| (x, y)))
        .flat_map(|(x, y)| ((x + 16 * y) as f32).to_le_bytes())
        .collect()
}

/// The uint and int textures' bytes: every texel 1, as the lane bound them.
fn small_texels() -> Vec<u8> {
    (0..SMALL * SMALL)
        .flat_map(|_| 1u32.to_le_bytes())
        .collect()
}

/// The texture specs the kernel's three slots take, in binding order.
fn texture_specs() -> [TextureSpec; 3] {
    [
        TextureSpec::new(
            PixelFormat::R32Float,
            [WIDTH, HEIGHT],
            WIDTH * 4,
            TextureRole::Read,
        ),
        TextureSpec::new(
            PixelFormat::R32Uint,
            [SMALL, SMALL],
            SMALL * 4,
            TextureRole::Read,
        ),
        TextureSpec::new(
            PixelFormat::R32Sint,
            [SMALL, SMALL],
            SMALL * 4,
            TextureRole::Read,
        ),
    ]
}

/// The output buffer's length: one thread's results a lane.
fn output_bytes() -> usize {
    us().len() * LANE_BYTES as usize
}

/// The layout a row's output buffer is read under: one row of the layout per thread, whose
/// elements are the runtime sampler's four channels, so a diff names the lane and the channel.
///
/// The value under the layout is the whole buffer, so the words outside the column are recorded;
/// they are not compared, for the reason the module's documentation gives.
fn layout() -> ByteLayout {
    let lanes = us().len() as u32;
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: output_bytes() as u64,
        },
        element: Element::U32,
        row_offset: SAMPLED_AT,
        row_elements: SAMPLED_WORDS,
        row_pitch: LANE_BYTES,
        rows: lanes,
        image_pitch: lanes * LANE_BYTES,
        images: 1,
        gaps: Gaps::Ignored,
    }
}

/// The case.
pub struct NearestPositions;

/// The case the suite holds.
pub fn all() -> Vec<NearestPositions> {
    vec![NearestPositions]
}

impl Case for NearestPositions {
    fn name(&self) -> &'static str {
        "nearest-sample-positions"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // One declaration and no tolerance: the output buffer the threads write their samples into is `Shared`, which is
        // what [`crate::replay::shared_memory_notice`] is for. It loosens the notices site alone,
        // so every value this case reads still compares exactly.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{}{extension}", FIXTURE.name),
                entry: FIXTURE.entry.to_owned(),
                source_sha256: crate::sha256_hex(FIXTURE.text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(FIXTURE.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(FIXTURE.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("buffer:0", &coords()),
            Input::bytes("buffer:1", &vec![FILL; output_bytes()]),
            Input::bytes("texture:float", &float_texels()),
            Input::bytes("texture:uint", &small_texels()),
            Input::bytes("texture:int", &small_texels()),
        ];
        for row in ROWS {
            inputs.push(Input::bytes(
                format!("sampler:{}", row.name),
                &row.sampler().words(),
            ));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = FIXTURE.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, FIXTURE.entry),
        )?;
        let specs = texture_specs();
        let textures: Vec<Handle> = [
            ("float", float_texels()),
            ("uint", small_texels()),
            ("int", small_texels()),
        ]
        .iter()
        .zip(&specs)
        .map(|((name, bytes), spec)| {
            made(
                sites,
                &format!("texture/{name}"),
                side.texture_2d_in_role(spec, bytes),
            )
        })
        .collect::<Result<Vec<Handle>, Abort>>()?;
        let coordinates = made(sites, "buffer/coords", side.buffer(&coords()))?;
        let lanes = us().len() as u32;

        for row in ROWS {
            let at = |site: &str| format!("{}/{site}", row.name);
            let sampler = made(sites, &at("sampler"), side.sampler(&row.sampler()))?;
            let output = made(
                sites,
                &at("buffer/out"),
                side.buffer(&vec![FILL; output_bytes()]),
            )?;
            let command_buffer = made(
                sites,
                &at("dispatch"),
                side.dispatch_threads(
                    pipeline,
                    &Bindings::new(&textures, &[coordinates, output]).sampled(&[sampler]),
                    &[],
                    [lanes, 1, 1],
                    [lanes / 4, 1, 1],
                ),
            )?;
            call(
                sites,
                &at("status"),
                side.wait(command_buffer),
                Clone::clone,
            )?;
            let bytes = call(
                sites,
                &at("read-back"),
                side.read_buffer(output, 0, output_bytes()),
                |_| Value::Absent,
            )?;
            sites.push(
                Node::branch(row.name, Vec::new())
                    .with(Node::leaf("sampled", Observed::bytes(bytes, layout()))),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// `x`, exact in `f64`, rounded to `f32` toward positive infinity.
    fn rounded_up(x: f64) -> f32 {
        let near = x as f32;
        if f64::from(near) < x {
            near.next_up()
        } else {
            near
        }
    }

    /// `x`, exact in `f64`, rounded to the nearest `f32` with ties away from zero.
    fn rounded_away(x: f64) -> f32 {
        let near = x as f32;
        let off = f64::from(near) - x;
        if off == 0.0 {
            return near;
        }
        let other = if off > 0.0 {
            near.next_down()
        } else {
            near.next_up()
        };
        if (f64::from(other) - x).abs() == off.abs() && other.abs() > near.abs() {
            other
        } else {
            near
        }
    }

    /// Texel `index` of an axis of `n` wrapped, or mirrored, into the level.
    fn wrapped(index: i64, n: u32, mirrored: bool) -> u32 {
        let n = i64::from(n);
        let inside = if mirrored {
            let period = index.rem_euclid(2 * n);
            if period < n {
                period
            } else {
                2 * n - 1 - period
            }
        } else {
            index.rem_euclid(n)
        };
        u32::try_from(inside).expect("a texel inside the level")
    }

    /// The texel the runtime lane's fitted rule says Metal's nearest filter reads at `u`: wrapped
    /// into one period first, that subtraction rounded up, then scaled with ties away from zero.
    ///
    /// This is the cross-check, not the expectation. The case's expectation is the recording, which
    /// record mode takes by running the fixture on Metal.
    fn fitted_texel(u: f32, n: u32, mirrored: bool) -> u32 {
        let u = f64::from(u);
        let start = if mirrored {
            2.0 * (u / 2.0).floor()
        } else {
            u.floor()
        };
        let scaled = rounded_away(f64::from(rounded_up(u - start)) * f64::from(n));
        wrapped(scaled.floor() as i64, n, mirrored)
    }

    /// The texel a device that scales before it wraps reads at `u`.
    fn scaled_first(u: f32, n: u32, mirrored: bool) -> u32 {
        wrapped((u * n as f32).floor() as i64, n, mirrored)
    }

    /// The red channel of the runtime sampler's sample at lane `lane` of `bytes`:
    /// `out[5 t + 2].x` read as a `float`.
    fn sampled_red(bytes: &[u8], lane: usize) -> f32 {
        let at = lane * LANE_BYTES as usize + SAMPLED_AT as usize;
        f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    }

    /// The lanes reach where the two rules part, on both address modes: a case whose coordinates
    /// all lay inside `[0, 1)` would pass on a device that scales before it wraps and prove
    /// nothing.
    #[test]
    fn the_lanes_reach_where_the_rules_part() {
        assert_eq!(us().len() % 4, 0, "four whole threadgroups");
        let us = lane_us();
        for mirrored in [false, true] {
            let deciding = us
                .iter()
                .filter(|&&u| fitted_texel(u, WIDTH, mirrored) != scaled_first(u, WIDTH, mirrored))
                .count();
            assert!(deciding > 0, "mirrored {mirrored}: no deciding lane");
        }
    }

    /// The roundings the fitted rule needs are the ones it names: a tie at `f32`'s precision goes
    /// away from zero, which is not what `as f32` does, and the wrap's subtraction rounds up.
    #[test]
    fn the_fitted_rule_s_roundings_are_its_own() {
        let tie = 1.0 + f64::powi(2.0, -24);
        assert_eq!(rounded_away(tie), 1.0f32.next_up());
        assert_eq!(rounded_away(-tie), -(1.0f32.next_up()));
        assert_eq!(tie as f32, 1.0, "nearest even takes the other neighbour");
        assert_eq!(rounded_up(1.0 + f64::EPSILON), 1.0f32.next_up());
    }

    /// The case's own bookkeeping holds: the layout covers the buffer exactly, every row names a
    /// distinct site, and each row's sampler words differ, so the input fingerprint tells the rows
    /// apart.
    #[test]
    fn the_rows_and_the_layout_agree_with_the_buffer() {
        // The compared column is the runtime sampler's `float4` and lies wholly inside a lane, so
        // the layout reads the words the case is about and no byte past the buffer.
        assert!(layout().extent() <= output_bytes());
        assert_eq!(SAMPLED_AT + SAMPLED_WORDS * 4, 3 * 16);
        let named: BTreeSet<&str> = ROWS.iter().map(|row| row.name).collect();
        assert_eq!(named.len(), ROWS.len());
        let words: BTreeSet<Vec<u8>> = ROWS.iter().map(|row| row.sampler().words()).collect();
        assert_eq!(words.len(), ROWS.len());
        assert_eq!(coords().len(), us().len() * 8);
        assert_eq!(float_texels().len() as u32, WIDTH * HEIGHT * 4);
    }

    /// Metal's recorded samples are the texel the fitted rule names, lane for lane and bit for bit,
    /// on both address modes.
    ///
    /// This is the cross-check the port owes the lane: the lane carried a rule, the case carries
    /// Metal's bytes, and this test says whether the two agree. It reads the recording rather than
    /// a device, so it runs where there is no GPU at all; where no recording is in the tree yet it
    /// says so and stops, since a test that silently passes on a missing file proves nothing.
    #[test]
    fn metal_s_recorded_samples_are_the_fitted_rule_s_texels() {
        let case = NearestPositions;
        let Ok(recording) = crate::recording::read(case.name()) else {
            println!(
                "{}: no recording in the tree yet, so nothing to cross-check",
                case.name()
            );
            return;
        };
        let us = lane_us();
        let row = (0.5 * HEIGHT as f32).floor();
        let mut checked = 0usize;
        let mut differ = Vec::new();
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                for (name, mirrored) in [("repeat", false), ("mirrored", true)] {
                    let Some(bytes) = outcome
                        .transcript
                        .root
                        .child(name)
                        .and_then(|node| node.child("sampled"))
                        .and_then(|node| match &node.observed.value {
                            Value::Bytes(bytes) => Some(bytes),
                            _ => None,
                        })
                    else {
                        continue;
                    };
                    for (lane, &u) in us.iter().enumerate() {
                        let expected = fitted_texel(u, WIDTH, mirrored) as f32 + 16.0 * row;
                        let got = sampled_red(bytes, lane);
                        checked += 1;
                        if got.to_bits() != expected.to_bits() {
                            differ.push(format!(
                                "{name} lane {lane} u {u:e} ({:08x}): Metal {got}, the rule \
                                 {expected}",
                                u.to_bits()
                            ));
                        }
                    }
                }
            }
        }
        assert!(checked > 0, "the recording holds no sampled row");
        for line in differ.iter().take(20) {
            println!("NEAREST-RULE-DIFFERS: {line}");
        }
        assert!(
            differ.is_empty(),
            "{} of {checked} recorded samples differ from the lane's fitted rule",
            differ.len()
        );
    }

    /// Where Metal's constexpr `clamp_to_zero` border test actually lies, read off each edge
    /// ladder's rungs.
    ///
    /// Prints one line a rung — the unnormalized offset, and whether Metal read texel 0 or the
    /// border — so the boundary is visible, and asserts the shape the ladder must have whatever the
    /// boundary is: the rungs nearest the edge read a texel and the rungs furthest from it read the
    /// border, with **one** crossing. A ladder that alternated would mean the mechanism is neither
    /// a quantum nor a flush, and this would say so rather than quietly averaging it away.
    #[test]
    fn metal_s_clamp_to_zero_border_test_has_one_crossing_a_ladder() {
        let case = NearestPositions;
        let Ok(recording) = crate::recording::read(case.name()) else {
            println!("{}: no recording in the tree yet", case.name());
            return;
        };
        let lanes = lane_us().len();
        let ladder = edge_us();
        for recorded in &recording.answers {
            for outcome in &recorded.outcomes {
                for name in ["repeat", "mirrored"] {
                    let Some(Value::Bytes(bytes)) = outcome
                        .transcript
                        .root
                        .child(name)
                        .and_then(|node| node.child("sampled"))
                        .map(|node| &node.observed.value)
                    else {
                        continue;
                    };
                    // Below the first texel and past the last are two ladders interleaved in
                    // `edge_us`, so each is read on its own stride of two.
                    for (edge, first) in [("below-zero", 0usize), ("past-one", 1usize)] {
                        let mut read = Vec::new();
                        for rung in 0..EDGE_RUNGS as usize {
                            let lane = lanes + 2 * rung + first;
                            let at = lane * LANE_BYTES as usize;
                            let word = f32::from_le_bytes([
                                bytes[at],
                                bytes[at + 1],
                                bytes[at + 2],
                                bytes[at + 3],
                            ]);
                            let offset = f64::from(ladder[2 * rung + first]);
                            let unnormalized = (offset - offset.round()).abs() * f64::from(WIDTH);
                            println!(
                                "CLAMP-TO-ZERO {name} {edge} rung {} (offset {unnormalized:e} of a \
                                 texel): Metal read {word}",
                                rung + 1
                            );
                            read.push(word != 0.0);
                        }
                        // One crossing: every rung that read a texel is nearer the edge than every
                        // rung that read the border. The rungs run from furthest to nearest, so the
                        // sequence of "read a texel" must be non-decreasing.
                        // (`SUBNORMAL_BRACKET` is read separately, below.)
                        let crossings = read.windows(2).filter(|pair| pair[0] != pair[1]).count();
                        assert!(
                            crossings <= 1,
                            "{name} {edge}: {crossings} crossings in the ladder: {read:?}"
                        );
                    }
                }
            }
        }
    }

    /// Metal's constexpr `clamp_to_zero` border test below the first texel is `float`'s own
    /// subnormal boundary: a normal coordinate however small reads the border, and a subnormal one
    /// reads texel 0.
    ///
    /// This is the answer the ladder above sets up. It is a bit test, not a tolerance: the two
    /// sides of the boundary are the smallest normal `float` and the largest subnormal, which are
    /// one unit in the last place apart, so nothing between them can be in doubt.
    #[test]
    fn metal_flushes_a_subnormal_coordinate_to_the_first_texel() {
        let case = NearestPositions;
        let Ok(recording) = crate::recording::read(case.name()) else {
            println!("{}: no recording in the tree yet", case.name());
            return;
        };
        let first = lane_us().len() + 2 * EDGE_RUNGS as usize;
        let mut rows = 0usize;
        for recorded in &recording.answers {
            for outcome in &recorded.outcomes {
                for name in ["repeat", "mirrored"] {
                    let Some(Value::Bytes(bytes)) = outcome
                        .transcript
                        .root
                        .child(name)
                        .and_then(|node| node.child("sampled"))
                        .map(|node| &node.observed.value)
                    else {
                        continue;
                    };
                    rows += 1;
                    for (at, &offset) in SUBNORMAL_BRACKET.iter().enumerate() {
                        let byte = (first + at) * LANE_BYTES as usize;
                        let word = f32::from_le_bytes([
                            bytes[byte],
                            bytes[byte + 1],
                            bytes[byte + 2],
                            bytes[byte + 3],
                        ]);
                        let subnormal = offset != 0.0 && offset.abs() < f32::MIN_POSITIVE;
                        println!(
                            "CLAMP-TO-ZERO {name} bracket -{offset:e} ({}): Metal read {word}",
                            if subnormal { "subnormal" } else { "normal" }
                        );
                        if subnormal {
                            assert_eq!(
                                word, 80.0,
                                "{name}: -{offset:e} is subnormal and Metal read {word}, not \
                                 texel (0, 5)"
                            );
                        } else {
                            assert_eq!(
                                word, 0.0,
                                "{name}: -{offset:e} is a normal float and Metal read {word}, not \
                                 the border"
                            );
                        }
                    }
                }
            }
        }
        assert!(rows > 0, "the recording holds no sampled row");
    }

    /// The words outside the compared column are recorded, and the one place Metal's constexpr
    /// `clamp_to_zero` samplers read a texel where a border is expected is exactly the three
    /// coordinates just below `+0.0`.
    ///
    /// This pins the reason the column is narrow. It is read off the recording, so it says what
    /// Metal did rather than what a rule predicts, and it moves the day Metal's own answer moves.
    #[test]
    fn metal_reads_a_texel_at_the_three_coordinates_just_below_zero() {
        let case = NearestPositions;
        let Ok(recording) = crate::recording::read(case.name()) else {
            println!("{}: no recording in the tree yet", case.name());
            return;
        };
        let us = us();
        // The three lanes are the coordinate one, two and three units in the last place below
        // `+0.0`, which `f32::next_down` reaches from `+0.0` and nothing else in the set does.
        let below_zero: Vec<usize> = us
            .iter()
            .enumerate()
            .filter(|(_, &u)| u < 0.0 && u > -1.0e-44)
            .map(|(lane, _)| lane)
            .collect();
        // The three subnormal coordinates come from the lane's own set; the edge ladder's rungs
        // are all normal floats, so they are not among them.
        assert_eq!(below_zero.len(), 3, "{below_zero:?}");
        let mut rows = 0usize;
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                for name in ["repeat", "mirrored"] {
                    let Some(Value::Bytes(bytes)) = outcome
                        .transcript
                        .root
                        .child(name)
                        .and_then(|node| node.child("sampled"))
                        .map(|node| &node.observed.value)
                    else {
                        continue;
                    };
                    rows += 1;
                    assert_eq!(bytes.len(), output_bytes(), "{name}: a narrowed recording");
                    for &lane in &below_zero {
                        let at = lane * LANE_BYTES as usize;
                        let word = |word: usize| {
                            let at = at + word * 4;
                            f32::from_le_bytes([
                                bytes[at],
                                bytes[at + 1],
                                bytes[at + 2],
                                bytes[at + 3],
                            ])
                        };
                        assert_eq!(word(0), 80.0, "{name} lane {lane}: constexpr nearest");
                        assert_eq!(word(4), 40.0, "{name} lane {lane}: constexpr linear");
                    }
                }
            }
        }
        assert!(rows > 0, "the recording holds no sampled row");
    }
}
