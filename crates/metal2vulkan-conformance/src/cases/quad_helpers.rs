//! A fragment's quad and simdgroup operations over quads a triangle covers in part, whose
//! uncovered pixels run as helpers.
//!
//! The owned fixture `quad_helpers` holds one fragment function per quad operation, each drawn in
//! a pass of its own over 64 small triangles, one in each 8 by 8 cell of a 64 by 64 `RGBA16Float`
//! target (`qh_cells`, `reconverge_select`'s `rs_cells`), whose edges cross quads, so a quad on an
//! edge runs one to three helpers:
//!
//! - `qh_any`: `quad_any` of the helper query, 1 in a quad with a helper;
//! - `qh_all`: `quad_all` of its negation, 0 in a quad with a helper;
//! - `qh_ballot`: `quad_ballot` of the helper query, the quad's helper lanes;
//! - `qh_is_first`: `quad_is_first`, quad lane 0 where the helpers take part;
//! - `qh_active`: `quad_active_threads_mask`, every lane where the helpers take part.
//!
//! Each pixel writes `(answer, dfdx(v + answer), dfdy(v + answer), quad lane)`, v a per-quad
//! plane of data words whose fine and coarse derivatives are alike.
//!
//! Three simdgroup functions follow, each drawn in four passes of one triangle over three pixels
//! of the quad at (2, 2), the helper at quad lane 0 to 3 (`qh_one`): those three pixels and the
//! helper are the draw's only fragments, so the simdgroup that shades them holds that quad alone.
//! Which pixels share a fragment simdgroup, and its lane order, are the device's (the runtime's
//! `fragment-simdgroup-grouping`; MSL states neither), so only answers that name no lane are
//! here: `qs_arith` (`simd_sum`, `simd_min`, `simd_max` of a byte of the pixel's hash), `qs_bits`
//! (`simd_and`, `simd_or`, `simd_xor` of 11 bits of it) and `qs_votes` (`simd_any` of the helper
//! query and `simd_all` of its negation, and how many bits the low 32 of `simd_ballot` of the
//! query and of `simd_active_threads_mask` set). `simd_is_first`, `simd_broadcast_first` and the
//! prefix sums and products name a lane and are not here. Each pixel writes the three answers and
//! `dfdx` of v plus their sum. A pass reads back the 8 by 8 corner of the target the quad lies in.
//!
//! **What Metal defines.** An active thread is a thread that is executing, the simdgroup and quad
//! functions act on the active threads, and a helper thread is active until it has computed its
//! gradients (MSL 4.1 sections 6.10.2 and 6.10.3: "Helper threads only execute to compute
//! gradients for quad-groups in a fragment shader and then become inactive"). Each function runs
//! its operations in control flow uniform over the draw and then takes a derivative of a value
//! their answers feed, so every helper still has a gradient to compute when they run, and is one
//! of their active threads. After a function's last derivative a helper's part is not Metal's to
//! state, and no operation here stands there.
//!
//! The clear value, -1 in every channel, is part of the answer: a pixel a draw does not cover
//! holds it, and so does a helper's, whose outputs Metal discards.
//!
//! Nothing here can fault a GPU: one draw of at most 64 small triangles a pass, one 64 by 64
//! target, and no loop in the fixture.

use crate::allowance::Allowance;
use crate::case::{call, drawn_or_refused, made, Abort, Case, Failure, Input, MetalSource, Side};
use crate::degradation::Degradation;
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    StageFunction, TargetSpec,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType};

/// The fixture every function of the case lives in.
const FIXTURE: &str = "quad_helpers";

/// The target's side in pixels.
const SIDE: u32 = 64;

/// The side of the corner a one-quad pass reads back: the quad at (2, 2) and the pixels around it.
const CORNER: u32 = 8;

/// The target's format: four halves a pixel.
const FORMAT: PixelFormat = PixelFormat::RGBA16Float;

/// Bytes one texel takes.
const TEXEL: u32 = 8;

/// Every channel of the clear value.
const CLEAR: f64 = -1.0;

/// The quad functions, each drawn over `qh_cells`' 64 triangles.
const QUAD: [&str; 5] = ["qh_any", "qh_all", "qh_ballot", "qh_is_first", "qh_active"];

/// The simdgroup functions, each drawn over `qh_one`'s triangle with the helper at each quad lane.
const SIMD: [&str; 3] = ["qs_arith", "qs_bits", "qs_votes"];

/// One pass: its site, its fragment function, its vertex function, its first vertex and vertex
/// count, and the side of the square at the target's origin it reads back.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Pass {
    site: String,
    fragment: &'static str,
    vertex: &'static str,
    first: u32,
    vertices: u32,
    side: u32,
}

impl Pass {
    /// The bytes its read takes.
    fn bytes(&self) -> u32 {
        self.side * self.side * TEXEL
    }
}

/// The case's passes, in order: each quad function over the cells, then each simdgroup function
/// once for each quad lane the helper takes.
fn passes() -> Vec<Pass> {
    let quad = QUAD.iter().map(|&fragment| Pass {
        site: fragment.to_owned(),
        fragment,
        vertex: "qh_cells",
        first: 0,
        vertices: 3 * 64,
        side: SIDE,
    });
    let simd = SIMD.iter().flat_map(|&fragment| {
        (0..4u32).map(move |lane| Pass {
            site: format!("{fragment}-helper-{lane}"),
            fragment,
            vertex: "qh_one",
            first: 3 * lane,
            vertices: 3,
            side: CORNER,
        })
    });
    quad.chain(simd).collect()
}

/// The one colour attachment every pipeline writes.
const COLORS: [Option<ColorState>; 1] = [Some(ColorState::new(FORMAT))];

/// The target every pass renders into: private, since a blit reads it back.
fn target_spec() -> TargetSpec {
    TargetSpec::read(FORMAT, [SIDE, SIDE])
}

/// The pipeline of a fragment function and the vertex function its passes draw with.
fn pipeline_spec(vertex: &'static str, fragment: &'static str) -> RenderPipelineSpec<'static> {
    RenderPipelineSpec::of(
        StageFunction::named(vertex),
        Some(StageFunction::named(fragment)),
        &COLORS,
    )
}

/// The vertex function each fragment function's passes draw with.
fn vertex_of(fragment: &str) -> &'static str {
    match QUAD.contains(&fragment) {
        true => "qh_cells",
        false => "qh_one",
    }
}

/// Every fragment function, quad functions first.
fn fragments() -> impl Iterator<Item = &'static str> {
    QUAD.into_iter().chain(SIMD)
}

/// The 256 data words every fragment reads at buffer 0: word i a hash of i.
fn words() -> Vec<u32> {
    (0..256u32)
        .map(|i| {
            let mut x = i.wrapping_mul(0x9E37_79B1);
            x ^= x >> 16;
            x = x.wrapping_mul(0x85EB_CA6B);
            x ^ (x >> 13)
        })
        .collect()
}

/// The data words' bytes.
fn data_bytes() -> Vec<u8> {
    words().iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// The layout of a square of `side` texels at the target's origin, a row of the layout a row of
/// pixels and its element 4x + c channel c of pixel x.
fn layout(side: u32) -> ByteLayout {
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [side, side, 1],
        },
        element: Element::F16,
        row_offset: 0,
        row_elements: side * 4,
        row_pitch: side * TEXEL,
        rows: side,
        image_pitch: side * side * TEXEL,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The case.
pub struct QuadHelpers;

impl QuadHelpers {
    /// The fixture under the entry point `entry`.
    fn fixture(entry: &'static str) -> Fixture {
        Fixture {
            name: FIXTURE,
            entry,
        }
    }

    /// Every entry the case's pipelines name, once each, vertex functions first.
    fn entries() -> Vec<&'static str> {
        ["qh_cells", "qh_one"]
            .into_iter()
            .chain(fragments())
            .collect()
    }
}

impl Case for QuadHelpers {
    fn name(&self) -> &'static str {
        "quad-helpers"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Declarations only, so the notices site stays readable; neither loosens a value. Our side
        // records `quads-split-by-clipping` for every fragment function with a quad or simdgroup
        // operation, and `fragment-simdgroup-grouping` for every one with a simdgroup operation,
        // on every device. A value either moves stays divergent, a ledger row.
        [
            Degradation::QuadsSplitByClipping,
            Degradation::FragmentSimdgroupGrouping,
        ]
        .into_iter()
        .map(|kind| crate::replay::notice_allowance(kind.code()))
        .collect()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for entry in Self::entries() {
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
        let fixture = Self::fixture(QUAD[0]);
        Ok(vec![
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
                &fragments()
                    .flat_map(|fragment| pipeline_spec(vertex_of(fragment), fragment).words())
                    .collect::<Vec<u8>>(),
            ),
            Input::bytes(
                "passes",
                passes()
                    .iter()
                    .map(|pass| {
                        format!(
                            "{}:{}:{}:{}:{}",
                            pass.site, pass.vertex, pass.first, pass.vertices, pass.side
                        )
                    })
                    .collect::<Vec<String>>()
                    .join(",")
                    .as_bytes(),
            ),
            Input::bytes("buffer:0", &data_bytes()),
            Input::bytes("clear", &CLEAR.to_le_bytes()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Self::fixture(QUAD[0]).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        // One target for every pass: each pass clears it, and the pieces of one command buffer
        // run in the order the case lists them, so a row sees only its own draw.
        let target = made(sites, "target", side.target(&target_spec()))?;
        let data = made(sites, "buffer/0", side.buffer(&data_bytes()))?;
        let passes = passes();
        let total: usize = passes.iter().map(|pass| pass.bytes() as usize).sum();
        let read_back = made(sites, "buffer/read-back", side.zeroed_buffer(total))?;

        // One pipeline a fragment function, each in force for the passes after it until the next:
        // `passes` lists each function's passes together, in `fragments`' order.
        let mut work = Vec::with_capacity(passes.len() * 2);
        let mut offset = 0u64;
        let mut current = None;
        for pass in &passes {
            let pipeline = match current {
                Some((fragment, pipeline)) if fragment == pass.fragment => pipeline,
                _ => {
                    let spec = pipeline_spec(pass.vertex, pass.fragment);
                    let pipeline = made(
                        sites,
                        &format!("{}/pipeline", pass.fragment),
                        side.render_pipeline(library, &spec),
                    )?;
                    current = Some((pass.fragment, pipeline));
                    pipeline
                }
            };
            // Every pass in one command buffer: a pass and a blit each, the blits writing one
            // slice of the read-back buffer apiece, so the case costs one commit and one wait.
            let steps = vec![
                RenderStep::Pipeline(pipeline),
                RenderStep::FragmentBuffer {
                    buffer: data,
                    offset: 0,
                    index: 0,
                },
                RenderStep::Draw {
                    primitive: PrimitiveType::Triangle,
                    first: pass.first,
                    count: pass.vertices,
                    instances: 1,
                },
            ];
            work.push(RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, [CLEAR; 4])]).drawing(steps),
            ));
            work.push(RenderWork::ReadTexture {
                texture: target,
                into: read_back,
                offset,
                row_pitch: pass.side * TEXEL,
                size: [pass.side, pass.side],
                plane: Plane::Whole,
                at: ReadAt::BASE,
            });
            offset += u64::from(pass.bytes());
        }
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(read_back, 0, total),
            |_| Value::Absent,
        )?;

        let mut first = 0usize;
        for (at, pass) in passes.iter().enumerate() {
            let end = first + pass.bytes() as usize;
            let row = Node::branch(pass.site.clone(), Vec::new()).with(Node::leaf(
                "texels",
                Observed::bytes(bytes[first..end].to_vec(), layout(pass.side)),
            ));
            // A row whose pass or read our side refused has no texels: the refusal is the row's
            // answer and compares as one.
            let refused = (outcome.refusal(2 * at)).or_else(|| outcome.refusal(2 * at + 1));
            sites.push(drawn_or_refused(row, refused));
            first = end;
        }
        Ok(())
    }
}

/// The cases the suite holds.
pub fn all() -> Vec<QuadHelpers> {
    vec![QuadHelpers]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The fixture's hash of `(x, y)`.
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

    /// The fixture's v at pixel `(x, y)`.
    fn v(words: &[u32], x: u32, y: u32) -> f32 {
        let word = |at: u32| words[(at & 255) as usize];
        let h = hash(x >> 1, y >> 1);
        let a = (word(h) & 255) as f32;
        let b = (2 + (word(h >> 8) & 7)) as f32;
        let c = (2 + (word(h >> 16) & 7)) as f32;
        a + b * (x & 1) as f32 + c * (y & 1) as f32
    }

    /// The quad lane of pixel `(x, y)`.
    fn lane(x: u32, y: u32) -> u32 {
        (x & 1) | (y & 1) << 1
    }

    /// The texels of `bytes`, a square of `side` texels, as four channels each.
    fn texels(bytes: &[u8], side: u32) -> Vec<[f32; 4]> {
        assert_eq!(bytes.len(), (side * side * TEXEL) as usize);
        bytes
            .chunks(TEXEL as usize)
            .map(|texel| {
                let mut c = [0f32; 4];
                for (at, b) in texel.chunks(2).enumerate() {
                    c[at] = half(u16::from_le_bytes([b[0], b[1]]));
                }
                c
            })
            .collect()
    }

    /// What quad function `fragment` writes at each covered pixel of a quad whose helper lanes
    /// are `helpers`, where the helpers take part: one texel a quad lane, `None` at a helper.
    fn quad_model(
        fragment: &str,
        words: &[u32],
        qx: u32,
        qy: u32,
        helpers: u32,
    ) -> [Option<[f32; 4]>; 4] {
        let answer = |l: u32| -> f32 {
            match fragment {
                "qh_any" => (helpers != 0) as u32 as f32,
                "qh_all" => (helpers == 0) as u32 as f32,
                "qh_ballot" => helpers as f32,
                "qh_is_first" => (l == 0) as u32 as f32,
                "qh_active" => 15.0,
                other => unreachable!("no quad function {other}"),
            }
        };
        let u = |x: u32, y: u32| v(words, x, y) + answer(lane(x, y));
        let mut out = [None; 4];
        for (l, slot) in out.iter_mut().enumerate() {
            let l = l as u32;
            if helpers & (1 << l) != 0 {
                continue;
            }
            let (x, y) = (2 * qx + (l & 1), 2 * qy + (l >> 1));
            *slot = Some([
                answer(l),
                u(x | 1, y) - u(x & !1, y),
                u(x, y | 1) - u(x, y & !1),
                l as f32,
            ]);
        }
        out
    }

    /// What simdgroup function `fragment` writes at covered pixel `(x, y)` of the quad at (2, 2),
    /// where the helper takes part: the same wherever the helper lies.
    fn simd_model(fragment: &str, words: &[u32], x: u32, y: u32) -> [f32; 4] {
        let w = |x: u32, y: u32| words[(hash(x, y) & 255) as usize];
        let quad: Vec<u32> = (0..4).map(|l| w(2 + (l & 1), 2 + (l >> 1))).collect();
        let r = match fragment {
            "qs_arith" => {
                let bytes: Vec<u32> = quad.iter().map(|w| w & 255).collect();
                [
                    bytes.iter().sum::<u32>(),
                    *bytes.iter().min().unwrap_or(&0),
                    *bytes.iter().max().unwrap_or(&0),
                ]
            }
            "qs_bits" => {
                let bits: Vec<u32> = quad.iter().map(|w| w & 2047).collect();
                [
                    bits.iter().fold(2047, |a, b| a & b),
                    bits.iter().fold(0, |a, b| a | b),
                    bits.iter().fold(0, |a, b| a ^ b),
                ]
            }
            // Any lane a helper, not every lane not one; one bit of the ballot; four lanes active.
            "qs_votes" => [1, 1, 4],
            other => unreachable!("no simdgroup function {other}"),
        };
        let sum = (r[0] + r[1] + r[2]) as f32;
        let dfdx = (v(words, x | 1, y) + sum) - (v(words, x & !1, y) + sum);
        [r[0] as f32, r[1] as f32, r[2] as f32, dfdx]
    }

    /// The case's bookkeeping holds: its sites are distinct, the fixture defines every entry a
    /// pipeline names, the passes' reads fill the read-back buffer exactly, and every value the
    /// model writes is an integer a half holds exactly.
    #[test]
    fn the_passes_the_layouts_and_the_fixture_agree() {
        let passes = passes();
        let sites: BTreeSet<&str> = passes.iter().map(|pass| pass.site.as_str()).collect();
        assert_eq!(sites.len(), passes.len());
        let text = QuadHelpers::fixture(QUAD[0])
            .text(RUNTIME_TEXT)
            .expect("the fixture's text");
        for entry in QuadHelpers::entries() {
            assert!(
                text.contains(&format!("@{entry}(")),
                "{entry} is not in {FIXTURE}"
            );
        }
        for pass in &passes {
            assert_eq!(
                pass.bytes() as usize,
                layout(pass.side).extent(),
                "{}",
                pass.site
            );
        }
        let words = words();
        for fragment in SIMD {
            for (x, y) in [(2, 2), (3, 3)] {
                for value in simd_model(fragment, &words, x, y) {
                    assert!(
                        value.fract() == 0.0 && value.abs() < 2048.0,
                        "{fragment}: {value}"
                    );
                }
            }
        }
    }

    /// Every texel Metal recorded is the model's, where a quad's helpers are active threads of its
    /// quad and simdgroup operations: each covered pixel of a quad holds what the quad's covered
    /// pixels and helpers together answer, and every other pixel holds the clear value.
    ///
    /// It reads the recording, so it runs where there is no GPU, and says whether Metal's answer
    /// is the one the module describes; it never stands in for that answer.
    #[test]
    fn metal_s_recorded_answers_count_the_helpers() {
        let words = words();
        let Ok(recording) = crate::recording::read(QuadHelpers.name()) else {
            println!("quad-helpers: no recording in the tree yet");
            return;
        };
        let mut checked = 0usize;
        let mut differ = Vec::new();
        for recorded in &recording.answers {
            for outcome in &recorded.outcomes {
                for pass in passes() {
                    let Some(Value::Bytes(bytes)) = outcome
                        .transcript
                        .root
                        .child(&pass.site)
                        .and_then(|node| node.child("texels"))
                        .map(|node| &node.observed.value)
                    else {
                        continue;
                    };
                    checked += 1;
                    let got = texels(bytes, pass.side);
                    let at = |x: u32, y: u32| got[(y * pass.side + x) as usize];
                    let clear = [CLEAR as f32; 4];
                    let mut expected = vec![clear; got.len()];
                    if pass.side == SIDE {
                        // A quad's helpers are the lanes the draw left at the clear value; a quad
                        // with no covered lane did not run.
                        for qy in 0..SIDE / 2 {
                            for qx in 0..SIDE / 2 {
                                let helpers = (0..4u32)
                                    .filter(|&l| at(2 * qx + (l & 1), 2 * qy + (l >> 1)) == clear)
                                    .fold(0, |m, l| m | 1 << l);
                                if helpers == 15 {
                                    continue;
                                }
                                let model = quad_model(pass.fragment, &words, qx, qy, helpers);
                                for (l, texel) in (0u32..).zip(model) {
                                    if let Some(texel) = texel {
                                        let (x, y) = (2 * qx + (l & 1), 2 * qy + (l >> 1));
                                        expected[(y * SIDE + x) as usize] = texel;
                                    }
                                }
                            }
                        }
                    } else {
                        let helper = pass.first / 3;
                        for l in (0..4u32).filter(|&l| l != helper) {
                            let (x, y) = (2 + (l & 1), 2 + (l >> 1));
                            expected[(y * CORNER + x) as usize] =
                                simd_model(pass.fragment, &words, x, y);
                        }
                    }
                    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
                        if g != e {
                            let (x, y) = (i as u32 % pass.side, i as u32 / pass.side);
                            differ.push(format!(
                                "{}: ({x}, {y}) holds {g:?}, the model {e:?}",
                                pass.site
                            ));
                        }
                    }
                }
            }
        }
        for line in differ.iter().take(20) {
            println!("QUAD-HELPERS-DIFFERS: {line}");
        }
        assert!(
            differ.is_empty(),
            "{} texels differ from the model",
            differ.len()
        );
        println!("{checked} recorded passes read as the model says");
    }
}
