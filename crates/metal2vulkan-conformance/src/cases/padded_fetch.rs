//! What a fragment function reads of a padded 10-bit colour attachment through programmable
//! blending (`[[color(0)]]`).
//!
//! `R10Unorm_X6`, `RG10Unorm_X12`, their `_sRGB` twins and `RGB10A2Unorm_sRGB` hold 10-bit
//! normalized codes ([`super::padded_blends`]). No Vulkan format holds them, so the runtime renders
//! such a target as its raw unsigned words and a fetch reads the words over an ordered read of the
//! attachment (the pixel interlock or a rasterization-order input attachment), decoding each code
//! to its value rounded to `half`. Metal's fetch reads so for a loaded code, a `float4` input and a
//! `half4` one alike (the M3 Max under macOS 27.0.1, `probes/padded_fetch.m`), and the missing
//! lanes of a one- or two-channel format as `(0, 0, 1)`. After a clear, or an earlier draw of the
//! same pass, Metal's tile holds the clear's `half`, or the draw's output rounded to `half` toward
//! zero, unclamped, which the stored codes do not keep: the runtime announces
//! `extended-range-blends-stored-codes` there, and those cases record the difference.
//!
//! Each case is a 128 x 64 target, color 0 of the padded format, beside an `RGBA32Float` color 1;
//! the owned fixture `pfx_fetch`'s `pfx_fetch_read` (a `float4` input) or `pfx_fetch_read_half`
//! (a `half4` one) reads color 0, whose write mask is none, and stores what it read into color 1,
//! the texels the case compares:
//!
//! - `load-float` and `load-half`: color 0 loaded with every 10-bit code in each channel (the
//!   channels disagreeing, `RGB10A2Unorm_sRGB`'s alpha cycling its four codes), its padding bits
//!   zero;
//! - `clear-float`: color 0 cleared to `(0.33371, -0.25, 1.25, 0.2)`, channels that are no code's
//!   value and past both ends of `[0, 1]`;
//! - `draw-float`: color 0 loaded as for `load`, then `pfx_fetch_fill` writes it unblended from
//!   [`super::extended_range_targets`]' 8,192 `float4` outputs (values across and past `[0, 1]`,
//!   code boundaries, NaN, the infinities and both zeros) before the read.
//!
//! Every format takes `load-float`; `R10Unorm_X6` and `RGB10A2Unorm_sRGB`, one linear and one
//! sRGB with an alpha, take the rest. Color 0's words are not compared: the question is the read.
//!
//! Nothing here can fault a GPU: single-sampled targets of the case's own extent, one or two
//! full-screen triangles a pass, no loop, and every fetch inside its buffer.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Input, MetalSource, Side};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    TargetSpec,
};
use crate::transcript::{Aspect, ByteLayout, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{ColorWriteMask, PixelFormat};
use metal2vulkan_runtime::Degradation;

use super::extended_range_targets::{
    fixture, fixture_inputs, float_bytes, not_read, program, shape, Format, HEIGHT, WIDTH,
};
use super::padded_blends::PADDED;

/// The owned fixture: a fill of color 0 and the reads of it into color 1.
const FETCH: &str = "pfx_fetch";
const VERTEX: &str = "pfx_fetch_vertex";
const FILL: &str = "pfx_fetch_fill";
const READ_FLOAT: &str = "pfx_fetch_read";
const READ_HALF: &str = "pfx_fetch_read_half";

/// The texels one draw covers.
const COUNT: usize = (WIDTH * HEIGHT) as usize;

/// Color 1's bytes a texel: an `RGBA32Float`.
const READ_BYTES: u32 = 16;

/// The `clear-float` cases' clear of color 0: no code's value, and past both ends of `[0, 1]`.
const CLEAR: [f64; 4] = [0.33371, -0.25, 1.25, 0.2];

/// What the pass leaves in color 0 before the read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leg {
    /// Loaded with every code ([`codes`]).
    Load,
    /// Cleared to [`CLEAR`].
    Clear,
    /// Loaded with every code, then written unblended by `pfx_fetch_fill`.
    Draw,
}

/// A case: one padded format, what color 0 holds, and the read's input type.
pub struct PaddedFetch {
    name: &'static str,
    format: Format,
    leg: Leg,
    /// Whether the read is the `half4` input rather than the `float4` one.
    half: bool,
}

/// Every case of the family.
pub fn all() -> Vec<PaddedFetch> {
    // The names are written out rather than built, so a case's recording file is named by something
    // a reader can grep for.
    const LOADS: [&str; 5] = [
        "padded-fetch-r10x6-load-float",
        "padded-fetch-r10x6-srgb-load-float",
        "padded-fetch-rg10x12-load-float",
        "padded-fetch-rg10x12-srgb-load-float",
        "padded-fetch-rgb10a2-srgb-load-float",
    ];
    let mut cases: Vec<PaddedFetch> = PADDED
        .iter()
        .zip(LOADS)
        .map(|(format, name)| PaddedFetch {
            name,
            format: *format,
            leg: Leg::Load,
            half: false,
        })
        .collect();
    let rest: [(usize, Leg, bool, &str); 6] = [
        (0, Leg::Load, true, "padded-fetch-r10x6-load-half"),
        (4, Leg::Load, true, "padded-fetch-rgb10a2-srgb-load-half"),
        (0, Leg::Clear, false, "padded-fetch-r10x6-clear-float"),
        (
            4,
            Leg::Clear,
            false,
            "padded-fetch-rgb10a2-srgb-clear-float",
        ),
        (0, Leg::Draw, false, "padded-fetch-r10x6-draw-float"),
        (4, Leg::Draw, false, "padded-fetch-rgb10a2-srgb-draw-float"),
    ];
    cases.extend(
        rest.into_iter()
            .map(|(index, leg, half, name)| PaddedFetch {
                name,
                format: PADDED[index],
                leg,
                half,
            }),
    );
    cases
}

/// Color 0's words for texel `i`: every 10-bit code in each channel across a row's 1,024 texels,
/// the channels disagreeing (`i`, `7i + 3` and `13i + 500`, each mod 1024), `RGB10A2Unorm_sRGB`'s
/// alpha `i` mod 4; a padded word's six low bits zero.
fn texel(format: &Format, i: u32) -> Vec<u8> {
    let codes = [i % 1024, (i * 7 + 3) % 1024, (i * 13 + 500) % 1024, i % 4];
    match format.word {
        Element::U32 => {
            let word = codes[0] | codes[1] << 10 | codes[2] << 20 | codes[3] << 30;
            word.to_le_bytes().to_vec()
        }
        _ => codes[..format.words as usize]
            .iter()
            .flat_map(|&c| ((c << 6) as u16).to_le_bytes())
            .collect(),
    }
}

/// Color 0's words before the pass: [`texel`] at every texel.
fn codes(format: &Format) -> Vec<u8> {
    (0..COUNT as u32).flat_map(|i| texel(format, i)).collect()
}

/// The layout of color 1's texels as the case reads them back: `RGBA32Float`, packed rows.
fn read_layout() -> ByteLayout {
    let pitch = WIDTH * READ_BYTES;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [WIDTH, HEIGHT, 1],
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: WIDTH * 4,
        row_pitch: pitch,
        rows: HEIGHT,
        image_pitch: pitch * HEIGHT,
        images: 1,
        // The rows are packed, so the sites cover every byte and there is no gap.
        gaps: Gaps::Ignored,
    }
}

impl PaddedFetch {
    /// Color 0: shared, its words going in through `replaceRegion:`, and of shader-write usage,
    /// which keeps Metal from its optimized layout ([`super::padded_blends`]).
    fn target(&self) -> TargetSpec {
        TargetSpec::read(self.format.format, [WIDTH, HEIGHT])
            .written()
            .shared()
    }

    /// Color 1, which the read writes.
    fn reads() -> TargetSpec {
        TargetSpec::read(PixelFormat::RGBA32Float, [WIDTH, HEIGHT])
    }

    /// The read function.
    fn read(&self) -> &'static str {
        if self.half {
            READ_HALF
        } else {
            READ_FLOAT
        }
    }

    /// The attachment states of the read: color 0 of mask none, color 1 written.
    fn read_colors(&self) -> [Option<ColorState>; 2] {
        [
            Some(ColorState::new(self.format.format).masked(ColorWriteMask::NONE)),
            Some(ColorState::new(PixelFormat::RGBA32Float)),
        ]
    }

    /// The attachment states of the fill: color 0 written, color 1 of mask none.
    fn fill_colors(&self) -> [Option<ColorState>; 2] {
        [
            Some(ColorState::new(self.format.format)),
            Some(ColorState::new(PixelFormat::RGBA32Float).masked(ColorWriteMask::NONE)),
        ]
    }

    /// Whether the pass loads color 0's words.
    fn loads(&self) -> bool {
        self.leg != Leg::Clear
    }
}

impl Case for PaddedFetch {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). A pass that loads and
        // stores a padded colour attachment announces `extended-range-bits-kept` (color 0's words
        // are not compared). A read after a clear or an earlier draw of the pass announces
        // `extended-range-blends-stored-codes`, and a draw that reads in the interlock what the
        // fill wrote may split the pass (`render-pass-split`, the venue's). Each loosens the
        // notices site alone: every value read still compares exactly, and a read the stored codes
        // move is a known gap.
        let mut named = vec![
            crate::replay::shared_memory_notice(),
            crate::replay::notice_allowance(Degradation::ExtendedRangeBitsKept.code()),
        ];
        if self.leg != Leg::Load {
            named.push(crate::replay::notice_allowance(
                Degradation::ExtendedRangeBlendsStoredCodes.code(),
            ));
        }
        if self.leg == Leg::Draw {
            named.push(crate::replay::interlocked_split_notice());
        }
        named
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut entries = vec![VERTEX, self.read()];
        if self.leg == Leg::Draw {
            entries.push(FILL);
        }
        super::extended_range_targets::shader_refs(FETCH, &entries)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs("program", FETCH)?;
        let read = self.read_colors();
        inputs.extend([
            Input::bytes("format", &self.format.format.as_raw().to_le_bytes()),
            Input::bytes(
                "pipeline:read",
                &RenderPipelineSpec::new(VERTEX, self.read(), &read).words(),
            ),
            Input::bytes("target", &self.target().words()),
            Input::bytes("target:reads", &Self::reads().words()),
        ]);
        if self.loads() {
            inputs.push(Input::bytes("texture:codes", &codes(&self.format)));
        } else {
            let clear: Vec<u8> = CLEAR.iter().flat_map(|c| c.to_le_bytes()).collect();
            inputs.push(Input::bytes("clear", &clear));
        }
        if self.leg == Leg::Draw {
            let fill = self.fill_colors();
            inputs.extend([
                Input::bytes(
                    "pipeline:fill",
                    &RenderPipelineSpec::new(VERTEX, FILL, &fill).words(),
                ),
                Input::bytes("buffer:values", &float_bytes()),
                Input::bytes("buffer:shape", &shape(WIDTH, 1)),
            ]);
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = fixture(FETCH, VERTEX)
            .loaded()
            .map_err(|why| not_read("library", why))?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let read_colors = self.read_colors();
        let reader = made(
            sites,
            "pipeline/read",
            side.render_pipeline(
                library,
                &RenderPipelineSpec::new(VERTEX, self.read(), &read_colors),
            ),
        )?;
        let mut steps = Vec::new();
        if self.leg == Leg::Draw {
            let fill_colors = self.fill_colors();
            let filler = made(
                sites,
                "pipeline/fill",
                side.render_pipeline(
                    library,
                    &RenderPipelineSpec::new(VERTEX, FILL, &fill_colors),
                ),
            )?;
            let values = made(sites, "buffer/values", side.buffer(&float_bytes()))?;
            let extent = made(sites, "buffer/shape", side.buffer(&shape(WIDTH, 1)))?;
            steps.extend([
                RenderStep::Pipeline(filler),
                RenderStep::FragmentBuffer {
                    buffer: values,
                    offset: 0,
                    index: 0,
                },
                RenderStep::FragmentBuffer {
                    buffer: extent,
                    offset: 0,
                    index: 1,
                },
                RenderStep::triangle(),
            ]);
        }
        steps.extend([RenderStep::Pipeline(reader), RenderStep::triangle()]);
        let target = made(sites, "target", side.target(&self.target()))?;
        let wide = made(sites, "target/reads", side.target(&Self::reads()))?;
        let color0 = match self.loads() {
            true => {
                call(
                    sites,
                    "replace-region",
                    side.replace_region(
                        target,
                        [WIDTH, HEIGHT],
                        WIDTH * self.format.texel,
                        &codes(&self.format),
                    ),
                    |_| Value::Absent,
                )?;
                PassColor::loaded(target)
            }
            false => PassColor::cleared(target, CLEAR),
        };
        let read_bytes = (WIDTH * HEIGHT * READ_BYTES) as usize;
        let reads = made(sites, "buffer/reads", side.zeroed_buffer(read_bytes))?;
        let work = vec![
            RenderWork::Pass(
                RenderPass::clearing(vec![color0, PassColor::cleared(wide, [-7.0; 4])])
                    .drawing(steps),
            ),
            RenderWork::ReadTexture {
                texture: wide,
                into: reads,
                offset: 0,
                row_pitch: WIDTH * READ_BYTES,
                size: [WIDTH, HEIGHT],
                plane: Plane::Whole,
                at: ReadAt::BASE,
            },
        ];
        let outcome = call(sites, "render", side.render(&work), |_| Value::Absent)?;
        program(sites, "program", &outcome, &work);
        let read = call(
            sites,
            "read-back/reads",
            side.read_buffer(reads, 0, read_bytes),
            |_| Value::Absent,
        )?;
        sites.push(Node::leaf("reads", Observed::bytes(read, read_layout())));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every case names a distinct row and a distinct question, so no two share a recording.
    #[test]
    fn the_cases_name_distinct_rows() {
        let cases = all();
        let named: BTreeSet<&str> = cases.iter().map(|case| case.name).collect();
        assert_eq!(named.len(), cases.len());
        let asked: BTreeSet<(u32, String, bool)> = cases
            .iter()
            .map(|case| {
                (
                    case.format.format.as_raw(),
                    format!("{:?}", case.leg),
                    case.half,
                )
            })
            .collect();
        assert_eq!(asked.len(), cases.len());
        for case in &cases {
            assert!(case.name.contains(case.format.slug), "{}", case.name);
        }
    }

    /// Color 0's words hold every 10-bit code in each channel, the six low bits of a padded word
    /// zero, and cover the target.
    #[test]
    fn the_codes_cover_every_code_and_the_target() {
        for format in &PADDED {
            let bytes = codes(format);
            assert_eq!(
                bytes.len(),
                COUNT * format.texel as usize,
                "{}",
                format.slug
            );
            let mut seen = [BTreeSet::new(), BTreeSet::new(), BTreeSet::new()];
            for chunk in bytes.chunks(format.texel as usize) {
                match format.word {
                    Element::U32 => {
                        let w = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        for (k, set) in seen.iter_mut().enumerate() {
                            set.insert((w >> (10 * k)) & 0x3ff);
                        }
                    }
                    _ => {
                        for (k, pair) in chunk.chunks(2).enumerate() {
                            let w = u16::from_le_bytes([pair[0], pair[1]]);
                            assert_eq!(w & 0x3f, 0, "{}", format.slug);
                            seen[k].insert(u32::from(w >> 6));
                        }
                    }
                }
            }
            let channels = match format.word {
                Element::U32 => 3,
                _ => format.words as usize,
            };
            for set in &seen[..channels] {
                assert_eq!(set.len(), 1024, "{}", format.slug);
            }
        }
    }
}
