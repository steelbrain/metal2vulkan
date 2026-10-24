//! What a fragment function reads back out of the colour attachment it is drawing into, when the
//! read is inside the fragment interlock.
//!
//! `fragment_fetch`'s `fetch_ordered` takes `float4 dst [[color(0), raster_order_group(0)]]` and
//! returns half of it. Metal orders such a read after every earlier fragment of the same pixel, so
//! `N` draws of the fullscreen triangle halve each texel `N` times in a defined order, and the
//! answer is a per-byte rule rather than a race: the stored code read as the correctly rounded
//! `c / 255`, halved, and written back as the exact product with 255 rounded to nearest. The
//! module's tests state that rule and check what it implies of the recorded bytes.
//!
//! The family is where the owner's rule bites hardest — if Metal allows it, we must allow it — so
//! every byte compares exactly and the case carries no allowance.
//!
//! The pass, as the probe this ports drew it: a 64² `BGRA8Unorm` or `RGBA8Unorm` target holding in
//! texel `i = y · 64 + x` the memory bytes `i`, `37 i + i / 256`, `101 i + 7 (i / 256)`, `i / 16`,
//! each mod 256, so that **every byte takes every value**. The pass loads and stores it and draws
//! `fetch_ordered` one to three times under a colour write mask of every channel or of red and
//! alpha alone. Beside it, colour 1 is an `RGBA16Float` attachment the function **never writes**,
//! cleared and stored: a pipeline whose fragment function returns one attachment while a second is
//! attached is a shape that has been refused before, so what colour 1 holds afterwards is part of
//! the recorded answer rather than a detail.
//!
//! Four cases: each format under each mask, three draw counts a case. Which memory byte a mask
//! keeps is the point of having both formats — `BGRA8Unorm` holds blue, green, red and alpha in
//! that order, so a mask of red and alpha writes bytes 2 and 3 of a `BGRA` texel and bytes 0 and 3
//! of an `RGBA` one. A device that wrote the attachment through a channel-swapped view without
//! swapping the mask would come out exact on one format and divergent on the other.
//!
//! **One departure from the lane**, stated because it is a departure: the lane's target is private
//! and filled by a buffer-to-texture blit, and the case's is of shared storage and filled by
//! `replaceRegion:`, because a case has no vocabulary for a blit into a texture and arbitrary raw
//! codes reach a texture no other way. The recorded answer says whether Metal cares; it does not,
//! on this oracle.
//!
//! Multisampled rows are **not** ported. The lane draws the same at four samples, but filling and
//! reading a multisampled target sample by sample is the validation crate's own kernel machinery,
//! and lavapipe has no 2x multisampling to replay it on. They stay a lane row.
//!
//! Nothing here can fault a GPU: three render passes of at most three triangles each over a 64²
//! target, no loop in either fixture, and six blits.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    StageFunction, TargetSpec,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{ColorWriteMask, PixelFormat};
use metal2vulkan_runtime::Degradation;

/// The fixture the vertex function comes from, and the function: Metal's fullscreen triangle at a
/// depth its vertex buffer holds.
const VERTEX_FIXTURE: &str = "rt6_resolves";
const VERTEX: &str = "rt6_resolve_full";

/// The fixture the fetch comes from, and the function that reads `[[color(0)]]` in a raster order
/// group and returns half of it.
const FETCH_FIXTURE: &str = "fragment_fetch";
const FRAGMENT: &str = "fetch_ordered";

/// The triangle's depth. The pass attaches no depth, so any depth inside the clip volume draws the
/// same; 0.5 is the probe's.
const Z: f32 = 0.5;

/// The targets' side, as the probe's: 4,096 texels, so every byte of three of the four channels
/// takes every one of the 256 codes.
const SIDE: u32 = 64;

/// Bytes a texel of either 8-bit format takes.
const TEXEL: u32 = 4;

/// Bytes a texel of colour 1, an `RGBA16Float` attachment.
const WIDE_TEXEL: u32 = 8;

/// Bytes one target takes.
const TARGET_BYTES: usize = (SIDE * SIDE * TEXEL) as usize;

/// Bytes one colour 1 takes.
const WIDE_BYTES: usize = (SIDE * SIDE * WIDE_TEXEL) as usize;

/// Colour 1's format: an attachment the fragment function never writes.
const WIDE: PixelFormat = PixelFormat::RGBA16Float;

/// Colour 1's clear.
const CLEAR: [f64; 4] = [0.25, 0.5, 0.75, 1.0];

/// The draw counts each case runs, in order: one pass a count, over its own target.
const DRAWS: [usize; 3] = [1, 2, 3];

/// The memory bytes every target starts at, in memory order.
fn initial() -> Vec<u8> {
    (0..(SIDE * SIDE) as usize)
        .flat_map(|i| [i, 37 * i + i / 256, 101 * i + 7 * (i / 256), i / 16].map(|b| b as u8))
        .collect()
}

/// The layout a target is read under: one row of the layout per row of texels, one element per
/// byte, so a diff names the texel and which of its memory bytes moved.
fn layout(texel: u32, element: Element) -> ByteLayout {
    let pitch = SIDE * texel;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [SIDE, SIDE, 1],
        },
        element,
        row_offset: 0,
        row_elements: pitch / element.size(),
        row_pitch: pitch,
        rows: SIDE,
        image_pitch: SIDE * pitch,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// One case: one target format under one colour write mask.
pub struct FetchOrdered {
    name: &'static str,
    format: PixelFormat,
    mask: ColorWriteMask,
}

/// Red and alpha alone: the mask the probe drew beside a mask of every channel.
fn red_alpha() -> ColorWriteMask {
    ColorWriteMask::RED.union(ColorWriteMask::ALPHA)
}

/// The suite's four cases: each of the two formats under each of the two masks.
pub fn all() -> Vec<FetchOrdered> {
    vec![
        FetchOrdered {
            name: "fetch-ordered-bgra8unorm",
            format: PixelFormat::BGRA8Unorm,
            mask: ColorWriteMask::ALL,
        },
        FetchOrdered {
            name: "fetch-ordered-bgra8unorm-red-alpha",
            format: PixelFormat::BGRA8Unorm,
            mask: red_alpha(),
        },
        FetchOrdered {
            name: "fetch-ordered-rgba8unorm",
            format: PixelFormat::RGBA8Unorm,
            mask: ColorWriteMask::ALL,
        },
        FetchOrdered {
            name: "fetch-ordered-rgba8unorm-red-alpha",
            format: PixelFormat::RGBA8Unorm,
            mask: red_alpha(),
        },
    ]
}

/// The fixture `name` under the entry point `entry`.
fn fixture(name: &'static str, entry: &'static str) -> Fixture {
    Fixture { name, entry }
}

impl FetchOrdered {
    /// The target the case fetches: shared, because its codes go in through `replaceRegion:`, and
    /// a shader also reads it, as the lane's does.
    fn target_spec(&self) -> TargetSpec {
        TargetSpec::read(self.format, [SIDE, SIDE]).shared()
    }

    /// Colour 1's target: a plain private `RGBA16Float` render target the pass clears and stores.
    fn wide_spec(&self) -> TargetSpec {
        TargetSpec::plain(WIDE, [SIDE, SIDE])
    }

    /// The pipeline's two attachments: the fetched one under the case's mask, and colour 1 with
    /// Metal's defaults.
    fn colors(&self) -> [Option<ColorState>; 2] {
        [
            Some(ColorState::new(self.format).masked(self.mask)),
            Some(ColorState::new(WIDE)),
        ]
    }

    /// The pipeline: the fullscreen triangle and `fetch_ordered`, whose stages come from two
    /// libraries.
    fn pipeline<'a>(
        &self,
        fetch: Handle,
        colors: &'a [Option<ColorState>],
    ) -> RenderPipelineSpec<'a> {
        RenderPipelineSpec::of(
            StageFunction::named(VERTEX),
            Some(StageFunction::of(fetch, FRAGMENT)),
            colors,
        )
    }
}

impl Case for FetchOrdered {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Two, and both loosen the **notice set** alone: how a venue reaches the attachment is a
        // route, and each route announces itself. Reading it through the pixel interlock may mean
        // rendering the pass into a scratch image (`fetch-through-scratch`, MoltenVK here), and
        // reading it through an input attachment may mean converting the colour in the shader
        // (`shader-color-conversion`, lavapipe here). Both are performance notices, so neither says
        // anything about the texels -- every byte still compares exactly, which is the owner's rule
        // for this family: if Metal allows it we must allow it, so a byte that differs is a bug and
        // a refusal is a bug, and neither gets a tolerance here. A venue announces one of the two,
        // so the other is reported as having fired on no venue of that run, which is honest.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for all 4 rows of the family and each
        // was divergent on that field and nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        std::iter::once(crate::replay::shared_memory_notice())
            .chain(
                [
                    Degradation::FetchThroughScratch,
                    Degradation::ShaderColorConversion,
                ]
                .iter()
                .map(|notice| crate::replay::notice_allowance(notice.code())),
            )
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
            Input::bytes("target:color1", &self.wide_spec().words()),
            // The handle a stage names never reaches the fingerprint, so the fetch library's
            // number here only names the spec whose words these are.
            Input::bytes("pipeline", &self.pipeline(Handle(1), &colors).words()),
            Input::bytes("buffer:z", &Z.to_le_bytes()),
            // The codes the fetch reads are the question, so they are an input whole.
            Input {
                name: "texels:initial".to_owned(),
                digest: Digest::sha256(&initial()),
            },
            Input::bytes("mask", &self.mask.as_raw().to_le_bytes()),
            Input::bytes(
                "draws",
                &DRAWS
                    .iter()
                    .flat_map(|n| n.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
        ]);
        inputs.push(Input::bytes(
            "clear",
            &CLEAR
                .iter()
                .flat_map(|channel| channel.to_bits().to_le_bytes())
                .collect::<Vec<u8>>(),
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
        let pipeline = made(
            sites,
            "pipeline",
            side.render_pipeline(vertex_library, &self.pipeline(fetch_library, &colors)),
        )?;
        let z = made(sites, "buffer/z", side.buffer(&Z.to_le_bytes()))?;

        // One target and one colour 1 for each draw count, every target seeded with the same codes
        // before anything is committed: the host cannot write a texture between two passes of one
        // command buffer, and a count must start from the codes the probe started from.
        let codes = initial();
        let mut targets = Vec::with_capacity(DRAWS.len());
        for draws in DRAWS {
            let target = made(
                sites,
                &format!("target/{draws}"),
                side.target(&self.target_spec()),
            )?;
            call(
                sites,
                &format!("replace-region/{draws}"),
                side.replace_region(target, [SIDE, SIDE], SIDE * TEXEL, &codes),
                |_| Value::Absent,
            )?;
            let wide = made(
                sites,
                &format!("target/{draws}/color1"),
                side.target(&self.wide_spec()),
            )?;
            targets.push((target, wide));
        }

        let mut work = Vec::new();
        let mut reads = Vec::new();
        for (at, draws) in DRAWS.iter().enumerate() {
            let (target, wide) = targets[at];
            let mut steps = vec![
                RenderStep::Pipeline(pipeline),
                RenderStep::vertex_buffer(z, 0),
            ];
            for _ in 0..*draws {
                steps.push(RenderStep::triangle());
            }
            work.push(RenderWork::Pass(
                RenderPass::clearing(vec![
                    PassColor::loaded(target),
                    PassColor::cleared(wide, CLEAR),
                ])
                .drawing(steps),
            ));
            let texels = made(
                sites,
                &format!("buffer/{draws}/texels"),
                side.zeroed_buffer(TARGET_BYTES),
            )?;
            let wide_bytes = made(
                sites,
                &format!("buffer/{draws}/color1"),
                side.zeroed_buffer(WIDE_BYTES),
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
            work.push(RenderWork::ReadTexture {
                texture: wide,
                into: wide_bytes,
                offset: 0,
                row_pitch: SIDE * WIDE_TEXEL,
                size: [SIDE, SIDE],
                plane: Plane::Whole,
                at: ReadAt::BASE,
            });
            reads.push((*draws, texels, wide_bytes));
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

        for (draws, texels, wide_bytes) in reads {
            let stored = call(
                sites,
                &format!("read-back/{draws}/texels"),
                side.read_buffer(texels, 0, TARGET_BYTES),
                |_| Value::Absent,
            )?;
            let wide = call(
                sites,
                &format!("read-back/{draws}/color1"),
                side.read_buffer(wide_bytes, 0, WIDE_BYTES),
                |_| Value::Absent,
            )?;
            sites.push(
                Node::branch(format!("x{draws}"), Vec::new())
                    .with(Node::leaf(
                        "texels",
                        Observed::bytes(stored, layout(TEXEL, Element::U8)),
                    ))
                    .with(Node::leaf(
                        "color1",
                        Observed::bytes(wide, layout(WIDE_TEXEL, Element::U16)),
                    )),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// What Metal stores in an 8-bit unorm byte holding `c` when a draw returns half of what it
    /// fetched: the code read as the correctly rounded `c / 255`, halved, then written as the exact
    /// product with 255 rounded to nearest. For `0 < c < 255` the product is never a tie, so the
    /// tie rule is only reached at the ends and `round_ties_even` is Metal's there too.
    fn halved(c: u8) -> u8 {
        let read = f32::from(c) / 255.0;
        (f64::from(read) * 0.5 * 255.0).round_ties_even() as u8
    }

    /// Which **memory** bytes of a texel of `format` the mask `mask` writes. A `BGRA8Unorm` texel holds
    /// blue, green, red and alpha in that order, so the mask's channels land on other bytes than an
    /// `RGBA8Unorm` texel's.
    fn written(format: PixelFormat, mask: ColorWriteMask) -> [bool; 4] {
        let [r, g, b, a] = [
            ColorWriteMask::RED,
            ColorWriteMask::GREEN,
            ColorWriteMask::BLUE,
            ColorWriteMask::ALPHA,
        ]
        .map(|channel| mask.contains(channel));
        match format {
            PixelFormat::BGRA8Unorm => [b, g, r, a],
            _ => [r, g, b, a],
        }
    }

    /// Metal's bytes after `draws` draws over a target of `format` under `mask`: each written byte
    /// halved once a draw, each byte the mask leaves out keeping the code it started at.
    fn expected(format: PixelFormat, mask: ColorWriteMask, draws: usize) -> Vec<u8> {
        let written = written(format, mask);
        let mut bytes = initial();
        for (at, byte) in bytes.iter_mut().enumerate() {
            if written[at % 4] {
                for _ in 0..draws {
                    *byte = halved(*byte);
                }
            }
        }
        bytes
    }

    /// The `half` codes Metal stores for colour 1's clear.
    const CLEARED: [u16; 4] = [0x3400, 0x3800, 0x3a00, 0x3c00];

    /// The bytes Metal leaves in colour 1: its clear, every texel, since the function never
    /// writes it. The recording holds the measured answer; this is what the test says it must be.
    fn cleared_wide() -> Vec<u8> {
        CLEARED
            .iter()
            .flat_map(|code| code.to_le_bytes())
            .collect::<Vec<u8>>()
            .repeat((SIDE * SIDE) as usize)
    }

    /// Metal's halving rule, by the lane's own spelling, over every code: the three draw counts a
    /// case runs each apply it that many times, and no code leaves the byte range.
    #[test]
    fn the_rule_covers_every_code() {
        for c in 0..=255u8 {
            let once = halved(c);
            assert!(once <= c, "{c} halved up to {once}");
            // Halving is monotone and reaches zero, so three draws never stall above the second.
            assert!(halved(once) <= once);
        }
        // The two ends, which are the only codes whose product with 255 can tie.
        assert_eq!(halved(0), 0);
        assert_eq!(halved(255), 128);
        assert_eq!(halved(1), 1);
    }

    /// Every byte of a target takes every code, which is why the probe's side is 64: three of the
    /// four memory bytes sweep all 256 codes, and the fourth sweeps the 256 values of `i / 16`.
    #[test]
    fn every_byte_takes_every_code() {
        let codes = initial();
        assert_eq!(codes.len(), TARGET_BYTES);
        for byte in 0..3usize {
            let seen: BTreeSet<u8> = codes.iter().skip(byte).step_by(4).copied().collect();
            assert_eq!(seen.len(), 256, "memory byte {byte} misses codes");
        }
    }

    /// The two formats answer differently under a mask of red and alpha, which is the point of
    /// carrying both: the mask keeps other **memory** bytes in each, so a device that swapped the
    /// attachment's channels without swapping the mask is exact on one and divergent on the other.
    #[test]
    fn the_mask_lands_on_other_bytes_in_each_format() {
        let mask = red_alpha();
        assert_eq!(
            written(PixelFormat::BGRA8Unorm, mask),
            [false, false, true, true]
        );
        assert_eq!(
            written(PixelFormat::RGBA8Unorm, mask),
            [true, false, false, true]
        );
        let bgra = expected(PixelFormat::BGRA8Unorm, mask, 1);
        let rgba = expected(PixelFormat::RGBA8Unorm, mask, 1);
        assert_ne!(bgra, rgba);
        // Under every channel the two formats halve the same memory bytes, so the masked rows are
        // the only ones that tell the orders apart.
        assert_eq!(
            expected(PixelFormat::BGRA8Unorm, ColorWriteMask::ALL, 1),
            expected(PixelFormat::RGBA8Unorm, ColorWriteMask::ALL, 1)
        );
    }

    /// Each further draw moves bytes the earlier count did not: the three counts are three
    /// different answers rather than one answer recorded three times.
    #[test]
    fn each_draw_count_is_its_own_answer() {
        for case in all() {
            let mut seen = BTreeSet::new();
            for draws in DRAWS {
                assert!(
                    seen.insert(expected(case.format, case.mask, draws)),
                    "{}: two draw counts answer the same bytes",
                    case.name
                );
            }
        }
    }

    /// The case's bookkeeping holds: four rows of distinct names, each layout covers its target
    /// exactly, and colour 1's clear is the `half` codes the recording compares.
    #[test]
    fn the_cases_bookkeeping_holds() {
        let rows = all();
        assert_eq!(rows.len(), 4);
        let names: BTreeSet<&str> = rows.iter().map(|row| row.name).collect();
        assert_eq!(names.len(), rows.len());
        for (texel, element, bytes) in [
            (TEXEL, Element::U8, TARGET_BYTES),
            (WIDE_TEXEL, Element::U16, WIDE_BYTES),
        ] {
            let layout = layout(texel, element);
            assert_eq!(
                (layout.rows * layout.row_elements * layout.element.size()) as usize,
                bytes
            );
        }
        assert_eq!(cleared_wide().len(), WIDE_BYTES);
        for (at, channel) in CLEAR.iter().enumerate() {
            // Each clear channel is exactly representable as a `half`, so the stored code is the
            // clear and not a rounding of it.
            assert_eq!(f64::from(half_of(CLEARED[at])), *channel);
        }
    }

    /// An IEEE binary16 code as an `f32`, so the test reads the clear codes without a dependency.
    fn half_of(code: u16) -> f32 {
        let sign = u32::from(code >> 15) << 31;
        let exponent = u32::from((code >> 10) & 0x1f);
        let mantissa = u32::from(code & 0x3ff);
        let bits = match exponent {
            0 if mantissa == 0 => sign,
            0 => unreachable!("the clear codes are all normal"),
            _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
        };
        f32::from_bits(bits)
    }
}
