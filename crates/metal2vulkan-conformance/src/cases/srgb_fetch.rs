//! What a fragment function reads back out of an **sRGB** colour attachment it is drawing into.
//!
//! `fragment_fetch`'s `fetch_ordered` takes `float4 dst [[color(0), raster_order_group(0)]]` and
//! returns half of it. Over a plain 8-bit unorm target that is a halving of the code; over an sRGB
//! one it is a halving of the **decoded** value, re-encoded on the way back, so each colour byte
//! follows `HALVED` while alpha, which sRGB leaves linear, follows `(a + 1) / 2`.
//!
//! **`HALVED` is a measured table and cannot be derived.** It differs from the textbook sRGB
//! round trip -- decode `c / 255`, halve, encode, round -- in **15 of its 256 codes, each by
//! exactly one** (`tests::the_table_is_not_the_textbook_round_trip` pins which codes, so a later
//! tidying of the table into the formula fails). Metal's own decode and encode are what the
//! recording holds; the table is the lane's measurement of them, kept here as the cross-check.
//!
//! The pass, as the probe this ports drew it: a 64² `BGRA8Unorm_sRGB` or `RGBA8Unorm_sRGB` target
//! holding in texel `i` the memory bytes `i`, `37 i + i / 256`, `101 i + 7 (i / 256)`, `i / 16`,
//! each mod 256, so every byte takes every value. The pass loads and stores it and draws
//! `fetch_ordered` one, two or three times, every channel written, no blending. Each draw reads
//! what the last one stored, so `N` draws apply the one-draw map `N` times.
//!
//! A fourth row attaches a `Depth32Float_Stencil8` attachment, cleared to 1.0 by a pass before it
//! and loaded, under a depth state of `Always` without writes, and draws three times. Its answer
//! is the three-draw answer: a depth test that drops no fragment changes nothing. **It is here for
//! a second reason** -- its pipeline is the first in the suite to declare a depth and a stencil
//! format beside a colour answer that could move if the declaration were wrong, which is the field
//! our side was not forwarding at all until this round.
//!
//! Not ported: the multisampled rows, and the two depth-`Less` rows whose earlier pass writes 0.25
//! into the target's left half under a colour mask of no channel. Those need a half-extent draw and
//! a per-half expectation; they are the next commit in this family, not this one.
//!
//! Nothing here can fault a GPU: at most four render passes of at most three triangles each over a
//! 64² target, no loop in the fixture, and one blit a row.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, PassDepth, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep,
    RenderWork, StageFunction, TargetSpec,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, PixelFormat};
use metal2vulkan_runtime::{Degradation, StoreAction};

/// The fixture the vertex function comes from, and the function.
const VERTEX_FIXTURE: &str = "rt6_resolves";
const VERTEX: &str = "rt6_resolve_full";

/// The fixture the fetch comes from, and the function that reads `[[color(0)]]` in a raster order
/// group and returns half of it.
const FETCH_FIXTURE: &str = "fragment_fetch";
const FRAGMENT: &str = "fetch_ordered";

/// The triangle's depth, which the depth row's `Always` test compares and never fails.
const Z: f32 = 0.5;

/// The target's side, as the probe's: every memory byte takes every one of the 256 codes.
const SIDE: u32 = 64;

/// Bytes a texel takes.
const TEXEL: u32 = 4;

/// Bytes one target takes.
const TARGET_BYTES: usize = (SIDE * SIDE * TEXEL) as usize;

/// The depth and stencil format the fourth row attaches.
const DEPTH_STENCIL: PixelFormat = PixelFormat::Depth32FloatStencil8;

/// The depth the row's earlier pass clears to, which `Always` ignores.
const DEPTH_CLEAR: f64 = 1.0;

/// The memory bytes every target starts at, in memory order.
fn initial() -> Vec<u8> {
    (0..(SIDE * SIDE) as usize)
        .flat_map(|i| [i, 37 * i + i / 256, 101 * i + 7 * (i / 256), i / 16].map(|b| b as u8))
        .collect()
}

/// One row of a case: how many draws, and whether a depth and stencil attachment is there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Row {
    draws: usize,
    depth: bool,
}

/// The four rows each format runs, in order.
const ROWS: [Row; 4] = [
    Row {
        draws: 1,
        depth: false,
    },
    Row {
        draws: 2,
        depth: false,
    },
    Row {
        draws: 3,
        depth: false,
    },
    Row {
        draws: 3,
        depth: true,
    },
];

/// The site name a row's answer is emitted at.
fn row_name(row: &Row) -> String {
    match row.depth {
        true => format!("x{}-depth", row.draws),
        false => format!("x{}", row.draws),
    }
}

/// The depth state the depth row sets: `Always` without writes, so no fragment is dropped.
fn depth_state() -> DepthStencilDescriptor {
    let mut state = DepthStencilDescriptor::default();
    state.depth_compare_function = CompareFunction::Always;
    state.depth_write_enabled = false;
    state
}

/// The layout a target is read under: one row of the layout per row of texels, one element per
/// byte, so a diff names the texel and which of its memory bytes moved.
fn layout() -> ByteLayout {
    let pitch = SIDE * TEXEL;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [SIDE, SIDE, 1],
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: pitch,
        row_pitch: pitch,
        rows: SIDE,
        image_pitch: SIDE * pitch,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// One case: one sRGB target format, over all four rows.
pub struct SrgbFetch {
    name: &'static str,
    format: PixelFormat,
}

/// The suite's two cases.
pub fn all() -> Vec<SrgbFetch> {
    vec![
        SrgbFetch {
            name: "fetch-ordered-bgra8unorm-srgb",
            format: PixelFormat::BGRA8UnormSRGB,
        },
        SrgbFetch {
            name: "fetch-ordered-rgba8unorm-srgb",
            format: PixelFormat::RGBA8UnormSRGB,
        },
    ]
}

/// The fixture `name` under the entry point `entry`.
fn fixture(name: &'static str, entry: &'static str) -> Fixture {
    Fixture { name, entry }
}

impl SrgbFetch {
    /// The target: shared, because its codes go in through `replaceRegion:`, and a shader also
    /// reads it, as the lane's does.
    fn target_spec(&self) -> TargetSpec {
        TargetSpec::read(self.format, [SIDE, SIDE]).shared()
    }

    /// The one colour attachment, every channel written, no blending.
    fn colors(&self) -> [Option<ColorState>; 1] {
        [Some(ColorState::new(self.format))]
    }

    /// The pipeline of a row: the two stages, and the depth and stencil formats only where the row
    /// attaches them, as Metal's descriptor must state them.
    fn pipeline<'a>(
        &self,
        fetch: Handle,
        colors: &'a [Option<ColorState>],
        row: &Row,
    ) -> RenderPipelineSpec<'a> {
        let spec = RenderPipelineSpec::of(
            StageFunction::named(VERTEX),
            Some(StageFunction::of(fetch, FRAGMENT)),
            colors,
        );
        match row.depth {
            true => spec.depth_stencil(Some(DEPTH_STENCIL), Some(DEPTH_STENCIL)),
            false => spec,
        }
    }
}

impl Case for SrgbFetch {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The notice set alone, never a texel: how a venue reaches the attachment is a route, and
        // each route announces itself -- a scratch image through the pixel interlock (MoltenVK),
        // a shader colour conversion through an input attachment (lavapipe). Both are performance
        // notices. Every byte compares exactly, which is the owner's rule for this family.
        //
        // A third route the same pass takes on lavapipe: an attachment it does not store back is
        // Vulkan's `DONT_CARE` where the driver is measured to leave the earlier texels there,
        // announced as `dont-care-store`. Measured on both rows of this module and on neither row
        // of `super::fetch_ordered`, so it is named here and not there. It says nothing about the
        // texels either -- every byte still compares exactly.
        [
            Degradation::FetchThroughScratch,
            Degradation::ShaderColorConversion,
            Degradation::DontCareStore,
        ]
        .iter()
        .map(|notice| crate::replay::notice_allowance(notice.code()))
        .chain([
            // The buffer each pass reads its target back through is `Shared`
            // ([`crate::replay::shared_memory_notice`]).
            crate::replay::shared_memory_notice(),
        ])
        .collect()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for (name, entry) in [(VERTEX_FIXTURE, VERTEX), (FETCH_FIXTURE, FRAGMENT)] {
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{name}{extension}"),
                    entry: entry.to_owned(),
                    source_sha256: crate::sha256_hex(
                        fixture(name, entry).text(extension)?.as_bytes(),
                    ),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = Vec::new();
        for (label, name, entry) in [
            ("vertex", VERTEX_FIXTURE, VERTEX),
            ("fetch", FETCH_FIXTURE, FRAGMENT),
        ] {
            let held = fixture(name, entry);
            inputs.push(Input {
                name: format!("{label}:air"),
                digest: Digest::sha256(held.text(RUNTIME_TEXT)?.as_bytes()),
            });
            inputs.push(Input {
                name: format!("{label}:air-typed"),
                digest: Digest::sha256(held.text(METAL_TEXT)?.as_bytes()),
            });
        }
        let colors = self.colors();
        inputs.extend([
            Input::bytes("target", &self.target_spec().words()),
            Input::bytes("buffer:z", &Z.to_le_bytes()),
            Input {
                name: "texels:initial".to_owned(),
                digest: Digest::sha256(&initial()),
            },
        ]);
        // Each row's pipeline, so the depth row's declared formats are part of the fingerprint.
        for row in &ROWS {
            inputs.push(Input::bytes(
                format!("pipeline:{}", row_name(row)),
                &self.pipeline(Handle(1), &colors, row).words(),
            ));
        }
        inputs.push(Input::bytes(
            "depth",
            &[
                DEPTH_STENCIL.as_raw().to_le_bytes().to_vec(),
                DEPTH_CLEAR.to_bits().to_le_bytes().to_vec(),
            ]
            .concat(),
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let not_read = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let vertex_source = fixture(VERTEX_FIXTURE, VERTEX)
            .loaded()
            .map_err(|why| not_read("library/vertex", why))?;
        let vertex_library = made(
            sites,
            "library/vertex",
            side.library(&vertex_source.source()),
        )?;
        let fetch_source = fixture(FETCH_FIXTURE, FRAGMENT)
            .loaded()
            .map_err(|why| not_read("library/fetch", why))?;
        let fetch_library = made(sites, "library/fetch", side.library(&fetch_source.source()))?;

        let colors = self.colors();
        let z = made(sites, "buffer/z", side.buffer(&Z.to_le_bytes()))?;
        let state = made(
            sites,
            "state/always",
            side.depth_stencil_state(&depth_state()),
        )?;
        let codes = initial();

        // One target a row, every one seeded before anything is committed: the host cannot write a
        // texture between two passes of one command buffer, and each row starts from the probe's
        // codes.
        let mut work = Vec::new();
        let mut reads = Vec::new();
        for row in &ROWS {
            let at = row_name(row);
            let target = made(
                sites,
                &format!("target/{at}"),
                side.target(&self.target_spec()),
            )?;
            call(
                sites,
                &format!("replace-region/{at}"),
                side.replace_region(target, [SIDE, SIDE], SIDE * TEXEL, &codes),
                |_| Value::Absent,
            )?;
            let pipeline = made(
                sites,
                &format!("pipeline/{at}"),
                side.render_pipeline(vertex_library, &self.pipeline(fetch_library, &colors, row)),
            )?;
            let mut steps = vec![
                RenderStep::Pipeline(pipeline),
                RenderStep::vertex_buffer(z, 0),
            ];
            if row.depth {
                steps.push(RenderStep::DepthStencilState(state));
            }
            for _ in 0..row.draws {
                steps.push(RenderStep::triangle());
            }
            let mut pass = RenderPass::clearing(vec![PassColor::loaded(target)]).drawing(steps);
            if row.depth {
                // The depth attachment is cleared and stored by a pass of its own first, so the
                // drawing pass loads a depth it did not clear -- which is how the probe drew it.
                let depth = made(
                    sites,
                    &format!("target/{at}/depth"),
                    side.target(&TargetSpec::plain(DEPTH_STENCIL, [SIDE, SIDE])),
                )?;
                work.push(RenderWork::Pass(
                    RenderPass::clearing(Vec::new())
                        .with_depth(PassDepth::cleared(depth, DEPTH_CLEAR)),
                ));
                let mut loaded = PassDepth::cleared(depth, DEPTH_CLEAR);
                loaded.load = metal2vulkan_runtime::LoadAction::Load;
                pass = pass.with_depth(loaded.stored(StoreAction::DontCare));
            }
            work.push(RenderWork::Pass(pass));
            let texels = made(
                sites,
                &format!("buffer/{at}"),
                side.zeroed_buffer(TARGET_BYTES),
            )?;
            work.push(RenderWork::ReadTexture {
                texture: target,
                into: texels,
                offset: 0,
                row_pitch: SIDE * TEXEL,
                size: [SIDE, SIDE],
                plane: Plane::Whole,
                at: ReadAt::BASE,
            });
            reads.push((at, texels));
        }

        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        let mut program = Node::branch("program", Vec::new()).with(Node::leaf(
            "status",
            Observed::value(outcome.status.clone()),
        ));
        for (piece, refused) in work.iter().zip(&outcome.refused) {
            if let Some(failure) = refused {
                program = program.with(Node::leaf(
                    piece.call(),
                    Observed::value(failure.recorded()),
                ));
            }
        }
        if let Some(error) = &outcome.error {
            program = program.with(Node::leaf("error", Observed::value(error.clone())));
        }
        sites.push(program);

        for (at, texels) in reads {
            let stored = call(
                sites,
                &format!("read-back/{at}"),
                side.read_buffer(texels, 0, TARGET_BYTES),
                |_| Value::Absent,
            )?;
            sites.push(Node::leaf(at, Observed::bytes(stored, layout())));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// What Metal stores in an sRGB colour byte holding `c` when a draw returns half of what it
    /// fetched: the code decoded by Metal's sRGB decode, halved, and encoded by Metal's encode.
    ///
    /// A measurement, not a formula: see the module's note and
    /// `tests::the_table_is_not_the_textbook_round_trip`.
    const HALVED: [u8; 256] = [
        0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 6, 6, 6, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14, 15, 15, 16,
        17, 18, 18, 19, 20, 21, 21, 22, 23, 23, 24, 25, 26, 27, 27, 28, 29, 30, 30, 31, 32, 33, 33,
        34, 35, 36, 36, 37, 38, 38, 39, 40, 41, 42, 42, 43, 44, 44, 45, 46, 47, 48, 48, 49, 50, 50,
        51, 52, 53, 53, 54, 55, 56, 57, 57, 58, 59, 59, 60, 61, 62, 62, 63, 64, 65, 65, 66, 67, 68,
        68, 69, 70, 71, 71, 72, 73, 74, 74, 75, 76, 77, 77, 78, 79, 80, 80, 81, 82, 83, 83, 84, 85,
        86, 86, 87, 88, 89, 89, 90, 91, 92, 92, 93, 94, 95, 95, 96, 97, 98, 98, 99, 100, 101, 101,
        102, 103, 104, 104, 105, 106, 107, 107, 108, 109, 110, 110, 111, 112, 113, 113, 114, 115,
        116, 116, 117, 118, 119, 119, 120, 121, 122, 122, 123, 124, 125, 125, 126, 127, 128, 128,
        129, 130, 131, 131, 132, 133, 134, 134, 135, 136, 137, 137, 138, 139, 140, 140, 141, 142,
        143, 143, 144, 145, 146, 146, 147, 148, 149, 149, 150, 151, 152, 152, 153, 154, 155, 155,
        156, 157, 158, 158, 159, 160, 161, 161, 162, 163, 164, 164, 165, 166, 167, 167, 168, 169,
        170, 170, 171, 172, 173, 173, 174, 175, 176, 176, 177, 178, 179, 179, 180, 181, 182, 182,
        183, 184, 185, 185, 186, 187, 188,
    ];

    /// What Metal stores in the **alpha** byte, which sRGB leaves linear: `(a + 1) / 2`, the plain
    /// unorm halving rounded to nearest. The recording holds the measured answer; this is the rule
    /// the tests below hold it to.
    fn halved_alpha(a: u8) -> u8 {
        u16::from(a).div_ceil(2) as u8
    }

    /// Which memory byte of a texel of `format` is alpha: the fourth in both orders, since
    /// `BGRA8Unorm_sRGB` reverses only the three colour channels.
    const ALPHA_BYTE: usize = 3;

    /// Metal's bytes after `draws` draws: each colour byte through `HALVED` once a draw, alpha
    /// through the linear rule once a draw.
    fn expected(draws: usize) -> Vec<u8> {
        let mut bytes = initial();
        for (at, byte) in bytes.iter_mut().enumerate() {
            for _ in 0..draws {
                *byte = match at % 4 == ALPHA_BYTE {
                    true => halved_alpha(*byte),
                    false => HALVED[usize::from(*byte)],
                };
            }
        }
        bytes
    }

    /// The textbook sRGB round trip: decode, halve, encode, round to nearest. `f64::round` breaks
    /// a tie away from zero, which is the spelling this comparison is against -- the count below
    /// moves by one code under a round-half-to-even tie rule, so the rule is named rather than
    /// assumed.
    fn textbook(c: u8) -> u8 {
        let u = f64::from(c) / 255.0;
        let linear = match u <= 0.04045 {
            true => u / 12.92,
            false => ((u + 0.055) / 1.055).powf(2.4),
        };
        let halved = linear * 0.5;
        let encoded = match halved <= 0.0031308 {
            true => 12.92 * halved,
            false => 1.055 * halved.powf(1.0 / 2.4) - 0.055,
        };
        (encoded * 255.0).round() as u8
    }

    /// `HALVED` is Metal's measurement and **not** the textbook formula: it differs in 15 of the
    /// 256 codes, each by exactly one. Pinning the codes and the size of the disagreement is what
    /// stops someone replacing the table with the formula, which would be wrong in those 15 and
    /// right everywhere a casual check would look.
    ///
    /// The codes are pinned as a set rather than only counted, because the count alone depends on
    /// how the textbook spelling breaks a tie (see `textbook`).
    #[test]
    fn the_table_is_not_the_textbook_round_trip() {
        let differing: Vec<u8> = (0..=255u8)
            .filter(|c| HALVED[usize::from(*c)] != textbook(*c))
            .collect();
        assert_eq!(
            differing,
            vec![5, 13, 17, 20, 21, 24, 28, 32, 40, 44, 48, 52, 60, 68, 80],
            "which codes Metal's table and the formula disagree on"
        );
        for code in &differing {
            let ours = i32::from(HALVED[usize::from(*code)]);
            let theirs = i32::from(textbook(*code));
            assert_eq!(
                (ours - theirs).abs(),
                1,
                "code {code} differs by more than one"
            );
        }
    }

    /// The table is a halving: monotone, never above its input, and it reaches zero. A table typo
    /// that put a code above its input, or broke the ordering, fails here.
    #[test]
    fn the_table_is_a_halving() {
        assert_eq!(HALVED[0], 0);
        for c in 0..=255usize {
            assert!(HALVED[c] <= c as u8, "code {c} halved up to {}", HALVED[c]);
            if c > 0 {
                assert!(HALVED[c] >= HALVED[c - 1], "the table falls at {c}");
            }
        }
        // sRGB halves far less aggressively than a linear halving would, which is the whole point
        // of the family: half of the decoded 255 encodes back to 188, not 128.
        assert_eq!(HALVED[255], 188);
    }

    /// Alpha is linear, so it follows the plain unorm rule and **not** the table: the two rules
    /// disagree on nearly every code, so a case that fed alpha through the colour table would
    /// differ from Metal in most texels.
    #[test]
    fn alpha_is_not_a_colour_channel() {
        assert_eq!(halved_alpha(0), 0);
        assert_eq!(halved_alpha(255), 128);
        let same = (0..=255u8)
            .filter(|a| halved_alpha(*a) == HALVED[usize::from(*a)])
            .count();
        assert!(same < 32, "the two rules agree on {same} codes of 256");
    }

    /// Every byte of a target takes every code, which is why the probe's side is 64.
    #[test]
    fn every_byte_takes_every_code() {
        let codes = initial();
        assert_eq!(codes.len(), TARGET_BYTES);
        for byte in 0..3usize {
            let seen: BTreeSet<u8> = codes.iter().skip(byte).step_by(4).copied().collect();
            assert_eq!(seen.len(), 256, "memory byte {byte} misses codes");
        }
    }

    /// The four rows are four questions: the three draw counts answer differently, and the depth
    /// row answers exactly what three draws answer, since `Always` drops no fragment. That last
    /// equality is the row's whole claim, so it is stated here rather than left implied.
    #[test]
    fn the_rows_are_four_questions() {
        let mut seen = BTreeSet::new();
        for draws in [1, 2, 3] {
            assert!(seen.insert(expected(draws)), "two draw counts answer alike");
        }
        let depth_row = ROWS[3];
        assert!(depth_row.depth);
        assert_eq!(depth_row.draws, 3);
        let names: BTreeSet<String> = ROWS.iter().map(row_name).collect();
        assert_eq!(names.len(), ROWS.len(), "two rows share a site name");
    }

    /// Only the depth row's pipeline declares a depth and a stencil format, and declaring them
    /// changes the pipeline's fingerprint -- which is what makes the row able to notice a side that
    /// does not state them.
    #[test]
    fn only_the_depth_row_declares_the_formats() {
        for case in all() {
            let colors = case.colors();
            let plain = case.pipeline(Handle(1), &colors, &ROWS[2]);
            let with_depth = case.pipeline(Handle(1), &colors, &ROWS[3]);
            assert_eq!(plain.depth_format, None);
            assert_eq!(with_depth.depth_format, Some(DEPTH_STENCIL));
            assert_eq!(with_depth.stencil_format, Some(DEPTH_STENCIL));
            assert_ne!(plain.words(), with_depth.words());
        }
    }

    /// The layout covers a target exactly.
    #[test]
    fn the_layout_covers_the_target() {
        let layout = layout();
        assert_eq!(
            (layout.rows * layout.row_elements * layout.element.size()) as usize,
            TARGET_BYTES
        );
        assert_eq!(all().len(), 2);
    }
}
