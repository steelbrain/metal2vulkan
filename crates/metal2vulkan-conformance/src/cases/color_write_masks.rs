//! Which channels of a texel a draw under a color write mask changes.
//!
//! `writeMask` is ordinary Metal state and Metal honours every shape of it, so the suite asks the
//! question at the three shapes that tell a mask that is read from one that is ignored: **no**
//! channel, **some** of them, and **all** of them.
//!
//! The family exists because a mask of no channel was not honoured and nothing caught it: the
//! tile-store family (`snorm_tile_store`) asked for one, our side drew every channel, and the
//! divergence read as our implementation refusing a legal Metal state. It was the suite's own
//! runtime side dropping the field (`runtime_side::color_descriptor`), which the Metal side set.
//! One case of a mask of no channel is therefore not enough: a mask of *some* channels goes
//! through the same field, and a fix for one that left the other broken would be half a fix.
//!
//! So each case loads a target whose every texel holds four **different** bytes, draws the
//! fixture's fullscreen triangle of white over all of it under one mask, and reads the texels
//! back. White writes `0xff`, so a channel the mask keeps reads as its own prefill byte and a
//! channel the mask writes reads as `0xff`: the four channels are told apart by value, and a mask
//! applied to the wrong channel fails rather than passing by coincidence.
//!
//! `RGBA8Unorm` is the format because its codes survive Metal's tile store unchanged, so **the
//! texels are exact on every device** and no case loosens them. That is the point of the format
//! choice: the tile-store family's masked draw can only be compared under a notice, and a mask is
//! too ordinary a piece of state to be checked only where a notice is in the way.
//!
//! The one site that is loosened is the notices site, by `shader-color-conversion`: a device whose
//! own float-to-normalized conversion is not Metal's has the fragment function round each stored
//! channel itself (`metal2vulkan::pipeline::color_writes`), which it announces. Which device does
//! that is the device's answer and not the case's — lavapipe announces it, MoltenVK on an Apple
//! GPU does not — and it changes no texel here, because white converts to `0xff` either way. The
//! case of a mask of no channel does not announce it at all: with nothing stored there is no
//! conversion to round, which is itself a small confirmation that the mask reached the pipeline.
//!
//! The mask of every channel is the control. Without it a side that quietly stopped drawing at all
//! would pass both of the other two, and the family would prove nothing.
//!
//! # `MTLColorWriteMaskUnspecialized`, and why it is not a placeholder here
//!
//! The fourth and fifth cases are `MTLColorWriteMaskUnspecialized` (`0x10`), which MTL4 defines as
//! the placeholder for a mask a render command specializes later. A **classic**
//! `MTLRenderPipelineDescriptor` takes it and draws, measured in a throwaway probe on this Apple
//! M3 Max, macOS 27.0.1:
//!
//! - a pipeline whose `writeMask` is `0x10` stores **every** channel, and so does one whose mask is
//!   `0x18` or `0x1f`. So the bit **dominates** the four channel bits rather than being ignored:
//!   ignoring it would leave `0x10` as a mask of no channel, which stores nothing, and
//!   `0x10 | Red` as red alone;
//! - Metal's setter keeps five bits and no more — `0x20` reads back as `0x0`, `0x30` as `0x10` and
//!   `0xffffffff` as `0x1f` — so there is no sixth bit to ask about.
//!
//! Each of the two cases is therefore told from the answer the bit-dropped mask would give, which
//! is the whole reason both are here rather than one: `write-mask-unspecialized` differs from
//! `write-mask-none`'s answer and `write-mask-unspecialized-and-red` from a mask of red alone. What
//! neither can be told from is `write-mask-all`, because every channel is every channel — that
//! agreement is **by construction** and is asserted as such below rather than left to look like a
//! coincidence.
//!
//! Our side refused this mask until this round (`metal2vulkan::metal::state::color_write_mask`
//! answered a `PlaceholderEnumValue` descriptor error), which is a refusal where Metal draws, so
//! the two cases are what hold the fix.
//!
//! Nothing here can fault a GPU: one 4 by 4 target a case, one triangle, no loop.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{Abort, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, RenderPass, RenderPipelineSpec, RenderStep, TargetSpec,
};
use crate::replay::NOTICES_SITE;
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{ColorWriteMask, PixelFormat};
use metal2vulkan_runtime::Degradation;

/// The owned fixture whose vertex function is Metal's fullscreen triangle at the depth a vertex
/// buffer holds, and whose fragment function returns white. The tile-store family's fixture.
const FIXTURE: &str = "rt6_resolves";
const VERTEX: &str = "rt6_resolve_full";
const FRAGMENT: &str = "rt6_resolve_white";

/// The target's side.
const SIDE: u32 = 4;

/// The format every case renders, whose codes Metal's tile store leaves alone, so every field of
/// every case is exact.
const FORMAT: PixelFormat = PixelFormat::RGBA8Unorm;

/// The depth the draw takes, which its vertex buffer holds as one `float`.
const DEPTH: f32 = 0.5;

/// The bytes of one texel the host writes before the pass: four **different** codes, one a
/// channel, so a channel is told from another by value.
const PREFILL: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

/// What one case asks about: a write mask, and the name its recording takes.
#[derive(Clone, Copy)]
struct Subject {
    name: &'static str,
    mask: ColorWriteMask,
}

/// The three shapes of a mask that tell one that is read from one that is ignored.
const SUBJECTS: &[Subject] = &[
    Subject {
        name: "write-mask-none",
        mask: ColorWriteMask::NONE,
    },
    Subject {
        name: "write-mask-red-alpha",
        mask: ColorWriteMask::RED.union(ColorWriteMask::ALPHA),
    },
    Subject {
        name: "write-mask-all",
        mask: ColorWriteMask::ALL,
    },
    Subject {
        name: "write-mask-unspecialized",
        mask: ColorWriteMask::UNSPECIALIZED,
    },
    Subject {
        name: "write-mask-unspecialized-and-red",
        mask: ColorWriteMask::UNSPECIALIZED.union(ColorWriteMask::RED),
    },
];

/// One case of the family.
pub struct ColorWriteMaskCase {
    subject: Subject,
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<ColorWriteMaskCase> {
    SUBJECTS
        .iter()
        .map(|subject| ColorWriteMaskCase { subject: *subject })
        .collect()
}

/// The bytes of a `SIDE` by `SIDE` `RGBA8Unorm` target whose every texel holds [`PREFILL`].
fn filled() -> Vec<u8> {
    PREFILL.repeat((SIDE * SIDE) as usize)
}

/// The layout the target's whole level 0 is read under: one element a channel, so a diff names the
/// row and the channel inside it rather than a byte offset.
fn target_layout() -> ByteLayout {
    let row_pitch = 4 * SIDE;
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
        row_elements: row_pitch,
        row_pitch,
        rows: SIDE,
        image_pitch: row_pitch * SIDE,
        images: 1,
        gaps: Gaps::Compared,
    }
}

impl ColorWriteMaskCase {
    /// The fixture the case runs.
    fn fixture(&self) -> Fixture {
        Fixture::new(FIXTURE)
    }

    /// The pipeline the case makes: the fullscreen triangle and white into `RGBA8Unorm` under the
    /// case's own write mask.
    fn pipeline_spec<'a>(&self, colors: &'a [Option<ColorState>]) -> RenderPipelineSpec<'a> {
        RenderPipelineSpec::new(VERTEX, FRAGMENT, colors)
    }

    /// The one color state the pipeline holds.
    fn color_state(&self) -> [Option<ColorState>; 1] {
        [Some(ColorState::new(FORMAT).masked(self.subject.mask))]
    }
}

impl Case for ColorWriteMaskCase {
    fn name(&self) -> &'static str {
        self.subject.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The texels are not loosened: `RGBA8Unorm` codes survive Metal's store, so which channels
        // the mask kept is an exact answer on every device or it is a bug. The notices site is,
        // by the one notice a draw that stores a converted channel may announce, because which
        // device rounds the conversion in the shader is the device's answer and not the case's and
        // it moves no texel here.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for all 5 rows of the family and each
        // was divergent on that field and nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        vec![
            crate::replay::shared_memory_notice(),
            Allowance::new(
                format!("case/{NOTICES_SITE}/*"),
                Loosening::Unchecked,
                Justification::notice(Degradation::ShaderColorConversion),
            ),
        ]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.fixture();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{}{extension}", fixture.name),
                    entry: format!("{VERTEX}+{FRAGMENT}"),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        let colors = self.color_state();
        let target = TargetSpec::plain(FORMAT, [SIDE, SIDE]).shared();
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("render-pipeline", &self.pipeline_spec(&colors).words()),
            Input::bytes("target", &target.words()),
            Input::bytes("texels", &filled()),
            Input::bytes("buffer:0", &DEPTH.to_le_bytes()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: String, failure: Failure| Abort { at, failure };
        let loaded = self.fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = crate::case::made(sites, "library", side.library(&loaded.source()))?;
        let colors = self.color_state();
        let state = crate::case::made(
            sites,
            "render-pipeline",
            side.render_pipeline(library, &self.pipeline_spec(&colors)),
        )?;
        let target = crate::case::made(
            sites,
            "target",
            side.target(&TargetSpec::plain(FORMAT, [SIDE, SIDE]).shared()),
        )?;
        let depth = crate::case::made(sites, "buffer:0", side.buffer(&DEPTH.to_le_bytes()))?;
        crate::case::call(
            sites,
            "wrote-texels",
            side.replace_region(target, [SIDE, SIDE], 4 * SIDE, &filled()),
            |_| Value::Absent,
        )?;
        let steps = vec![
            RenderStep::Pipeline(state),
            RenderStep::vertex_buffer(depth, 0),
            RenderStep::triangle(),
        ];
        let outcome = side
            .render(&[crate::render::RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::loaded(target)]).drawing(steps),
            )])
            .map_err(|failure| stop("drawn/pass".to_owned(), failure))?;
        let mut calls = Node::branch("calls", Vec::new());
        for (index, refused) in outcome.refused.iter().enumerate() {
            calls = calls.with(Node::leaf(
                format!("{index}-render"),
                Observed::value(match refused {
                    Some(failure) => failure.recorded(),
                    None => Value::ok(Value::Absent),
                }),
            ));
        }
        let mut node = Node::branch("drawn", Vec::new()).with(calls);
        if let Some(failure) = outcome.refused.iter().flatten().next() {
            sites.push(node);
            return Err(Abort {
                at: "drawn/calls/0-render".to_owned(),
                failure: failure.clone(),
            });
        }
        node = node.with(Node::leaf(
            "status",
            Observed::value(Value::ok(outcome.status.clone())),
        ));
        node = node.with(Node::leaf(
            "command-buffer-error",
            Observed::value(match &outcome.error {
                Some(error) => Value::Variant {
                    name: "Some".to_owned(),
                    payload: Box::new(error.clone()),
                },
                None => Value::unit_variant("None"),
            }),
        ));
        node = node.with(Node::leaf(
            "texels",
            Observed::bytes(
                side.read_texture(target, [SIDE, SIDE], 4 * SIDE)
                    .map_err(|failure| stop("drawn/texels".to_owned(), failure))?,
                target_layout(),
            ),
        ));
        sites.push(node);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! What Metal's answer must be, stated channel by channel and checked against the recordings.
    //!
    //! The rule is Metal's own and needs no measurement to state: a channel the mask names takes
    //! the fragment's white (`0xff`), and a channel it does not keeps its prefill byte. The test is
    //! here because a recording that disagrees means either the mask is not Metal state or the
    //! recording was taken with a different pipeline, and the test says which case.

    use super::*;
    use crate::recording::Recording;
    use crate::transcript::Transcript;

    /// The texel Metal leaves under `mask`: `0xff` in each channel it names, the prefill byte in
    /// each it does not.
    fn expected(mask: ColorWriteMask) -> [u8; 4] {
        // Measured: the placeholder bit dominates the four channel bits and stores every channel,
        // so a mask holding it is every channel whatever else it names.
        if mask.contains(ColorWriteMask::UNSPECIALIZED) {
            return [0xff; 4];
        }
        let bits = [
            ColorWriteMask::RED,
            ColorWriteMask::GREEN,
            ColorWriteMask::BLUE,
            ColorWriteMask::ALPHA,
        ];
        let mut texel = PREFILL;
        for (channel, bit) in texel.iter_mut().zip(bits) {
            if mask.contains(bit) {
                *channel = 0xff;
            }
        }
        texel
    }

    /// The one outcome of `name`'s recorded Metal answer; a case of this family is settled, so its
    /// runs fold into one.
    fn only_outcome(name: &str) -> Transcript {
        let recording: Recording = crate::recording::read(name)
            .unwrap_or_else(|why| panic!("{name}: no recording ({why})"));
        let answer = recording
            .answers
            .first()
            .unwrap_or_else(|| panic!("{name}: no answer"));
        assert_eq!(answer.outcomes.len(), 1, "{name}: more than one outcome");
        answer.outcomes[0].transcript.clone()
    }

    /// The bytes the site at `path` of `transcript` holds.
    fn bytes(transcript: &Transcript, path: &str) -> Vec<u8> {
        let mut node = &transcript.root;
        for name in path.trim_start_matches("case/").split('/') {
            node = node
                .child(name)
                .unwrap_or_else(|| panic!("no site {path} (stopped at {name})"));
        }
        let Value::Bytes(bytes) = &node.observed.value else {
            panic!("{path} holds no bytes");
        };
        bytes.clone()
    }

    /// Metal wrote exactly the channels each mask names and kept the others, in every texel.
    #[test]
    fn metal_wrote_exactly_the_channels_each_mask_names() {
        for subject in SUBJECTS {
            let recorded = only_outcome(subject.name);
            assert_eq!(
                bytes(&recorded, "case/drawn/texels"),
                expected(subject.mask).repeat((SIDE * SIDE) as usize),
                "{}",
                subject.name
            );
        }
    }

    /// The three channel masks are three shapes a mask can have and each is a different answer, so
    /// none of those cases can be passed by the answer another expects.
    ///
    /// The two placeholder cases are **not** in that set, and this test says why rather than
    /// quietly excluding them: each stores every channel, so each agrees with `write-mask-all` by
    /// construction. What tells them apart from a mask whose placeholder bit was dropped is the
    /// next test.
    #[test]
    fn the_three_channel_masks_ask_three_different_questions() {
        let channel: Vec<[u8; 4]> = SUBJECTS
            .iter()
            .filter(|s| !s.mask.contains(ColorWriteMask::UNSPECIALIZED))
            .map(|s| expected(s.mask))
            .collect();
        assert_eq!(channel.len(), 3);
        for (i, a) in channel.iter().enumerate() {
            for b in &channel[i + 1..] {
                assert_ne!(a, b, "two masks expect the same texel");
            }
        }
        // And none of them is the prefill-and-white confusion: the unmasked case writes every
        // channel, the masked-off case writes none.
        assert_eq!(expected(ColorWriteMask::NONE), PREFILL);
        assert_eq!(expected(ColorWriteMask::ALL), [0xff; 4]);
    }

    /// Each placeholder case stores something the **bit-dropped** mask would not, which is what
    /// makes the pair non-vacuous, and Metal's own recording says so.
    ///
    /// `0x10` with the bit dropped is a mask of no channel, which keeps the prefill; `0x10 | Red`
    /// with it dropped is red alone. Both recordings hold every channel written instead.
    #[test]
    fn metal_reads_the_placeholder_bit_as_every_channel_and_not_as_nothing() {
        let dropped = |mask: ColorWriteMask| mask.intersection(ColorWriteMask::ALL);
        for (name, mask) in [
            ("write-mask-unspecialized", ColorWriteMask::UNSPECIALIZED),
            (
                "write-mask-unspecialized-and-red",
                ColorWriteMask::UNSPECIALIZED.union(ColorWriteMask::RED),
            ),
        ] {
            let recorded = only_outcome(name);
            let texels = bytes(&recorded, "case/drawn/texels");
            assert_eq!(
                texels,
                [0xffu8; 4].repeat((SIDE * SIDE) as usize),
                "{name}: Metal did not store every channel"
            );
            // The answer the bit-dropped mask would have given is a different image, so the case
            // cannot pass for a side that dropped the bit.
            let without = expected(dropped(mask));
            assert_ne!(
                without, [0xff; 4],
                "{name}: the bit-dropped mask stores every channel too, so the case asks nothing"
            );
        }
    }

    /// No case of the family loosens a texel: the notices site is the only loosened site, and the
    /// one notice it names is in the degrade catalogue.
    ///
    /// This is the family's whole tolerance story in one assertion. A case that ever loosened the
    /// texels would fail here, which is the field the family exists to compare.
    #[test]
    fn only_the_notices_site_is_loosened() {
        for case in all() {
            for allowance in case.allowances() {
                assert_eq!(
                    allowance.path,
                    format!("case/{NOTICES_SITE}/*"),
                    "{}: an allowance outside the notices site",
                    case.name()
                );
                match &allowance.justified_by {
                    // The venue's own memory notice, which every row of the family declares
                    // because every row reads its answer back through a `Shared` buffer, and the
                    // one notice a converted channel may carry.
                    Justification::Degradation(code) => assert!(
                        code == Degradation::SharedInSystemMemory.code()
                            || code == Degradation::ShaderColorConversion.code(),
                        "{}: an unexpected notice {code}",
                        case.name()
                    ),
                    other => panic!("{}: an unexpected tolerance: {other:?}", case.name()),
                }
            }
            assert_eq!(case.metal_source(), MetalSource::Run);
            case.shaders().unwrap();
            let inputs = case.inputs().unwrap();
            assert!(
                inputs.iter().any(|input| input.name == "render-pipeline"),
                "{}: no render pipeline in its inputs",
                case.name()
            );
        }
    }

    /// Every case fingerprints its own pipeline, so a recording cannot be read for the wrong mask —
    /// the two placeholder cases included, whose stored texels agree.
    #[test]
    fn each_mask_fingerprints_its_own_pipeline() {
        let words: Vec<Vec<u8>> = all()
            .iter()
            .map(|case| {
                let colors = case.color_state();
                case.pipeline_spec(&colors).words()
            })
            .collect();
        for (i, a) in words.iter().enumerate() {
            for b in &words[i + 1..] {
                assert_ne!(a, b, "two masks fingerprint the same pipeline");
            }
        }
    }
}
