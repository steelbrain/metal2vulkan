//! What Metal's tile store leaves in the texels a draw did not write: a signed normalized
//! attachment's minimum codes, and an `RGB9E5Float` attachment's unnormalized words.
//!
//! Ported from the runtime lane `runtime_lane/snorm_tile_store.rs`. The lane's claim is Metal's
//! behaviour, measured on this same chip: once a pass **draws**, Metal writes each 32x32 tile back
//! through its channels' *values*, so a channel's minimum code (-128, or -32768), which reads as
//! -1.0 exactly as the code above it does, is stored as that code above (-127, -32767) in every
//! texel of the tile — including the texels no fragment wrote. An `RGB9E5Float` attachment's tiles
//! go back the same way, so a word whose largest mantissa is below 256 under an exponent above 0 is
//! stored normalized. A pass that **draws nothing** keeps every code.
//!
//! So each case asks the question twice over one target: a pass that loads it and draws the
//! fixture's fullscreen triangle under a write mask of **no channel** — a draw that writes no
//! texel, which still makes the pass a drawing pass — and then a pass that loads the same codes and
//! encodes nothing at all. Metal's answer to both is **run** here rather than copied: record mode
//! takes it on this M3 Max, and the module's own tests cross-check the recordings against the
//! lane's table, which is the evidence the port owes the lane.
//!
//! One case a format, so a format a device cannot render is its own row rather than a refusal that
//! costs the other five their sites.
//!
//! The runtime keeps the device's load and store by default (`docs/FAST-PATHS.md`), so where the
//! device's store is not Metal's the codes stay and the draw announces `snorm-minimum-kept` or
//! `shared-exponent-kept`, naming the attachment. That is the one field of these cases that may
//! differ from Metal, and each case names the notice that justifies it; every other field is exact.
//!
//! Nothing here can fault a GPU: one 4 by 4 target a case, one triangle, no loop.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{Abort, Case, Failure, Handle, Input, MetalSource, Side};
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
/// buffer holds, and whose fragment function returns white.
const FIXTURE: &str = "rt6_resolves";
const VERTEX: &str = "rt6_resolve_full";
const FRAGMENT: &str = "rt6_resolve_white";

/// The target's side, as the lane's is.
const SIDE: u32 = 4;

/// The depth the lane's draw takes, which its vertex buffer holds as one `float`.
const DEPTH: f32 = 0.5;

/// What one case asks about: a format, the bytes of one texel, and what the host writes into
/// every channel of every texel before each pass.
#[derive(Clone, Copy)]
struct Subject {
    /// The case's name, which is its recording's file name.
    name: &'static str,
    format: PixelFormat,
    /// Bytes one texel takes.
    texel: u32,
    /// The bytes of the code the host writes into every channel: a channel's minimum signed code,
    /// or `RGB9E5Float`'s unnormalized word.
    code: &'static [u8],
    /// How a diff reads one element of the target's bytes.
    element: Element,
    /// The notice a draw in a pass that loads this format announces where the device's store is
    /// not Metal's.
    notice: Degradation,
}

/// The six signed normalized formats Metal's tile store rewrites, each with its channel's minimum
/// code, and `RGB9E5Float` with the unnormalized word the lane measured (mantissas 1, 2 and 3
/// under the exponent 20: its largest mantissa below 256).
const SUBJECTS: &[Subject] = &[
    Subject {
        name: "snorm-store-r8",
        format: PixelFormat::R8Snorm,
        texel: 1,
        code: &[0x80],
        element: Element::I8,
        notice: Degradation::SnormMinimumKept,
    },
    Subject {
        name: "snorm-store-rg8",
        format: PixelFormat::RG8Snorm,
        texel: 2,
        code: &[0x80],
        element: Element::I8,
        notice: Degradation::SnormMinimumKept,
    },
    Subject {
        name: "snorm-store-rgba8",
        format: PixelFormat::RGBA8Snorm,
        texel: 4,
        code: &[0x80],
        element: Element::I8,
        notice: Degradation::SnormMinimumKept,
    },
    Subject {
        name: "snorm-store-r16",
        format: PixelFormat::R16Snorm,
        texel: 2,
        code: &[0x00, 0x80],
        element: Element::I16,
        notice: Degradation::SnormMinimumKept,
    },
    Subject {
        name: "snorm-store-rg16",
        format: PixelFormat::RG16Snorm,
        texel: 4,
        code: &[0x00, 0x80],
        element: Element::I16,
        notice: Degradation::SnormMinimumKept,
    },
    Subject {
        name: "snorm-store-rgba16",
        format: PixelFormat::RGBA16Snorm,
        texel: 8,
        code: &[0x00, 0x80],
        element: Element::I16,
        notice: Degradation::SnormMinimumKept,
    },
    Subject {
        name: "shared-exponent-store",
        format: PixelFormat::RGB9E5Float,
        texel: 4,
        code: &UNNORMALIZED,
        element: Element::U32,
        notice: Degradation::SharedExponentKept,
    },
];

/// The `RGB9E5Float` word the lane writes: mantissas 1, 2 and 3 under the exponent 20.
const UNNORMALIZED: [u8; 4] = (1u32 | 2 << 9 | 3 << 18 | 20 << 27).to_le_bytes();

/// The word Metal's store leaves in its place: mantissas 128, 256 and 384 under the exponent 13,
/// which is the same value normalized. The lane's table, which [`tests`] checks the recording
/// against, and nothing else reads.
#[cfg(test)]
const NORMALIZED: [u8; 4] = (128u32 | 256 << 9 | 384 << 18 | 13 << 27).to_le_bytes();

/// One case of the family.
pub struct SnormTileStore {
    subject: Subject,
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<SnormTileStore> {
    SUBJECTS
        .iter()
        .map(|subject| SnormTileStore { subject: *subject })
        .collect()
}

/// The degrade notices a pass that loads one of these attachments and draws may announce, each of
/// which a case of this family loosens by.
///
/// It is a list and not a device query, because a case never branches on the device: a case asks by
/// running and reads what came back. The list holds exactly the kinds measured on the two venues of
/// this round, MoltenVK 1.4.2 on this M3 Max and lavapipe in the local container. A third venue
/// that announces a kind not here fails the case until its kind is added, which is the model
/// working: an announced difference nobody named is not an announced difference.
const PASS_NOTICES: &[Degradation] = &[
    // The device's own load and store keeps a signed normalized minimum code where Metal's store
    // raises it (`docs/FAST-PATHS.md`, the tile store).
    Degradation::SnormMinimumKept,
    // The same for an `RGB9E5Float` word Metal's store normalizes.
    Degradation::SharedExponentKept,
    // The exact tile store, which the suite's replay selects by default: the load pass rewrites
    // the tiles no fragment reached as Metal's store leaves them, so the texels **are** Metal's
    // and the notice says the rewrite happened. Measured in the container; the two kinds above are
    // what the fast default announces instead (`--default-policy`), where the codes stay and the
    // texels differ.
    Degradation::UnreachedTilesRewritten,
    // The device's own compiler rewrote the stage's fast-math arithmetic as it made the pipeline,
    // which MoltenVK 1.4.2 on this M3 Max announces for this fixture's stages.
    Degradation::FastMathRewrites,
    // The buffer each pass reads the stored tiles back through is `Shared`, and a device with no
    // unified memory keeps a `Shared` allocation in system memory because that is the only memory
    // both sides can reach. It belongs to the venue's memory model rather than to the tile store
    // this family is about, and a device whose driver is Metal announces nothing for it.
    Degradation::SharedInSystemMemory,
];

/// The bytes of a `SIDE` by `SIDE` target of `texel` bytes a texel whose every channel holds
/// `code`.
fn filled(texel: u32, code: &[u8]) -> Vec<u8> {
    code.repeat((texel * SIDE * SIDE) as usize / code.len())
}

/// The layout the target's whole level 0 is read under: one element a channel, packed, so a diff
/// names the row and the channel inside it rather than a byte offset.
fn target_layout(texel: u32, element: Element) -> ByteLayout {
    let row_pitch = texel * SIDE;
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
        row_elements: row_pitch / element.size(),
        row_pitch,
        rows: SIDE,
        image_pitch: row_pitch * SIDE,
        images: 1,
        gaps: Gaps::Compared,
    }
}

impl SnormTileStore {
    /// The fixture the case runs.
    fn fixture(&self) -> Fixture {
        Fixture::new(FIXTURE)
    }

    /// The pipeline the case makes: the fullscreen triangle and white into its format under a
    /// write mask of **none** of its channels, so its draws write no texel.
    fn pipeline_spec<'a>(&self, colors: &'a [Option<ColorState>]) -> RenderPipelineSpec<'a> {
        RenderPipelineSpec::new(VERTEX, FRAGMENT, colors)
    }

    /// The one color state the pipeline holds.
    fn color_state(&self) -> [Option<ColorState>; 1] {
        [Some(
            ColorState::new(self.subject.format).masked(ColorWriteMask::NONE),
        )]
    }

    /// Run one pass over `target` and emit what the pass answered and what the target then held:
    /// `steps` encoded into a pass that loads the attachment and stores it.
    fn pass(
        &self,
        side: &mut dyn Side,
        at: &str,
        target: Handle,
        steps: Vec<RenderStep>,
        sites: &mut Sites,
    ) -> Result<(), Abort> {
        let stop = |at: String, failure: Failure| Abort { at, failure };
        let texel = self.subject.texel;
        let outcome = side
            .render(&[crate::render::RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::loaded(target)]).drawing(steps.clone()),
            )])
            .map_err(|failure| stop(format!("{at}/pass"), failure))?;
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
        let mut node = Node::branch(at, Vec::new()).with(calls);
        if let Some(failure) = outcome.refused.iter().flatten().next() {
            sites.push(node);
            return Err(Abort {
                at: format!("{at}/calls/0-render"),
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
                side.read_texture(target, [SIDE, SIDE], texel * SIDE)
                    .map_err(|failure| stop(format!("{at}/texels"), failure))?,
                target_layout(texel, self.subject.element),
            ),
        ));
        sites.push(node);
        Ok(())
    }
}

impl Case for SnormTileStore {
    fn name(&self) -> &'static str {
        self.subject.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The drawn texels are loosened by the one kind that explains a differing texel: the
        // notice the fast default announces where the device's store is not Metal's. The notices
        // site is loosened by every kind a pass of this family may announce, because which one
        // fires is the device's and the policy's answer rather than the case's. Nothing else is
        // loosened: the pass's calls, its status, its command-buffer error and the codes a pass of
        // no draw left are exact.
        let mut allowances = vec![Allowance::new(
            "case/drawn/texels",
            Loosening::Unchecked,
            Justification::notice(self.subject.notice),
        )];
        for kind in PASS_NOTICES {
            allowances.push(Allowance::new(
                format!("case/{NOTICES_SITE}/*"),
                Loosening::Unchecked,
                Justification::notice(*kind),
            ));
        }
        allowances
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
        let target = TargetSpec::plain(self.subject.format, [SIDE, SIDE]).shared();
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
            Input::bytes("texels", &filled(self.subject.texel, self.subject.code)),
            Input::bytes("buffer:0", &DEPTH.to_le_bytes()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
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
        let texel = self.subject.texel;
        let target = crate::case::made(
            sites,
            "target",
            side.target(&TargetSpec::plain(self.subject.format, [SIDE, SIDE]).shared()),
        )?;
        let depth = crate::case::made(sites, "buffer:0", side.buffer(&DEPTH.to_le_bytes()))?;
        let codes = filled(texel, self.subject.code);
        let write = |side: &mut dyn Side, at: &str, sites: &mut Sites| -> Result<(), Abort> {
            crate::case::call(
                sites,
                at,
                side.replace_region(target, [SIDE, SIDE], texel * SIDE, &codes),
                |_| Value::Absent,
            )?;
            Ok(())
        };
        // The drawing pass: the minimum codes loaded, one triangle drawn under a write mask of no
        // channel, and what the store left.
        write(side, "wrote-codes", sites)?;
        self.pass(
            side,
            "drawn",
            target,
            vec![
                RenderStep::Pipeline(state),
                RenderStep::vertex_buffer(depth, 0),
                RenderStep::triangle(),
            ],
            sites,
        )?;
        // The same codes again, and a pass that encodes nothing at all.
        write(side, "wrote-codes-again", sites)?;
        self.pass(side, "no-draw", target, Vec::new(), sites)
    }
}

#[cfg(test)]
mod tests {
    //! The cross-check the port owes the lane: the codes Metal's own store left, as the recordings
    //! hold them, against the lane's table.
    //!
    //! The table is the lane's (`runtime_lane/snorm_tile_store.rs`), restated here because a
    //! conformance case may not depend on the validation crate. It is the claim under test, not a
    //! helper: if a recorded texel disagrees, either Apple's store moved or the lane's table was
    //! never Metal's, and the test says which format.

    use super::*;
    use crate::recording::Recording;
    use crate::transcript::Transcript;

    /// The code Metal's store leaves in place of a signed normalized minimum: the code above it.
    fn above(code: &[u8]) -> Vec<u8> {
        let mut raised = code.to_vec();
        raised[0] += 1;
        raised
    }

    /// The one outcome of `name`'s recorded Metal answer. A case of this family is settled, so its
    /// runs folded into one.
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

    /// Metal's store raised every signed normalized minimum code to the code above it, in every
    /// texel of the tile, and a pass that drew nothing kept every one: the lane's claim, measured.
    #[test]
    fn the_recorded_drawing_pass_raises_every_snorm_minimum() {
        for subject in SUBJECTS {
            if subject.notice != Degradation::SnormMinimumKept {
                continue;
            }
            let recorded = only_outcome(subject.name);
            assert_eq!(
                bytes(&recorded, "case/drawn/texels"),
                filled(subject.texel, &above(subject.code)),
                "{}: the drawing pass",
                subject.name
            );
            assert_eq!(
                bytes(&recorded, "case/no-draw/texels"),
                filled(subject.texel, subject.code),
                "{}: the pass of no draw",
                subject.name
            );
        }
    }

    /// Metal's store normalized every unnormalized `RGB9E5Float` word, and a pass that drew
    /// nothing kept every one.
    #[test]
    fn the_recorded_drawing_pass_normalizes_a_shared_exponent_word() {
        let recorded = only_outcome("shared-exponent-store");
        assert_eq!(
            bytes(&recorded, "case/drawn/texels"),
            NORMALIZED.repeat((SIDE * SIDE) as usize),
            "the drawing pass"
        );
        assert_eq!(
            bytes(&recorded, "case/no-draw/texels"),
            UNNORMALIZED.repeat((SIDE * SIDE) as usize),
            "the pass of no draw"
        );
    }

    /// Every case of the family states its inputs and its pipeline, and loosens **only** the
    /// drawing pass's texels and its notices site, each by a degrade notice it names.
    ///
    /// This is the family's whole tolerance story in one assertion: the pass's calls, its status,
    /// its command-buffer error and the codes a pass of no draw left are exact or they are a bug.
    /// A case that ever loosened one of those would fail here.
    #[test]
    fn every_case_loosens_the_drawn_texels_and_its_notices_alone() {
        for case in all() {
            let inputs = case.inputs().unwrap();
            assert!(
                inputs.iter().any(|input| input.name == "render-pipeline"),
                "{}: no render pipeline in its inputs",
                case.name()
            );
            assert_eq!(case.metal_source(), MetalSource::Run);
            case.shaders().unwrap();
            for allowance in case.allowances() {
                assert!(
                    allowance.path == "case/drawn/texels"
                        || allowance.path == format!("case/{NOTICES_SITE}/*"),
                    "{}: an allowance outside the drawn texels and the notices: {}",
                    case.name(),
                    allowance.path
                );
                match &allowance.justified_by {
                    Justification::Degradation(code) => assert!(
                        PASS_NOTICES.iter().any(|kind| kind.code() == code),
                        "{}: {code} is not a tile-store notice",
                        case.name()
                    ),
                    other => panic!("{}: an unexpected tolerance: {other:?}", case.name()),
                }
            }
        }
    }
}
