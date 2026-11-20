//! A render state's texture slots and how its functions access them.
//!
//! The question is Metal's reflection: what
//! `newRenderPipelineStateWithDescriptor:options:reflection:error:` states about the texture
//! arguments of a vertex function (one `access::read` texture at 0) and a fragment function
//! (`access::read` at 0, `access::write` at 1, `access::read_write` at 2, an array of two
//! `access::write` at 3, an `access::write` at 5 the function never references, and a sampled
//! texture at 6). The runtime answers it through `RenderPipelineState::texture_access`.
//!
//! **Metal's answer as a value.** Record mode asks Metal itself, on an Apple M3 Max, and the
//! recording holds Metal's per-slot answer as a value (an Apple M4 under macOS 27.0.1 answered the
//! same list).
//!
//! **One stage at a time.** Metal states its reflection as one binding list per stage, so a single
//! `MTLRenderStages` bit has a Metal answer and the case asks for one bit at a time — vertex,
//! fragment, tile, object and mesh. A *union* of stages and no stage at all are a composition rule
//! of our own API with no Metal counterpart, so they stay out of the transcript.
//!
//! **Why the shader is a fixture and not inline AIR text.** An inline opaque-pointer AIR string
//! assembles with `xcrun metal-as` and packages with `xcrun metallib` once a target triple is added
//! — but Metal then refuses the pipeline with "Failed to upgrade function bitcode" (on an Apple M3
//! Max), so such text cannot be put in front of Metal's compiler at all. The case therefore names
//! the owned fixture
//! `reflection/texture_accesses`, whose MSL declares the same texture arguments at the same slots
//! and whose generated AIR carries the same `air.texture` metadata, down to the array length of
//! two at slot 3 and `air.arg_unused` at slot 5.
//!
//! Nothing here reaches the GPU: a library is compiled, a pipeline state is created, and its
//! reflection is read on the CPU.

use crate::case::{made, Abort, Case, Failure, Input, MetalSource, ShaderSource, Side};
use crate::fixture::{METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{Digest, Node, Observed, Sites, Value};
use metal2vulkan::metal::{BindingAccess, PixelFormat, RenderStages};
use metal2vulkan_devtools::{fixtures_dir, llvm_assemble, AssembleError};

/// The fixture's name under the owned `reflection` fixtures.
const FIXTURE: &str = "texture_accesses";

/// The color attachment the pipeline writes.
const ATTACHMENT: PixelFormat = PixelFormat::RGBA8Unorm;

/// The vertex function.
const VERTEX: &str = "texture_accesses_vertex";

/// The fragment function.
const FRAGMENT: &str = "texture_accesses_fragment";

/// The stages the case asks about, each a single `MTLRenderStages` bit, with the name its site
/// takes. A union of bits is left out: see the module documentation.
const STAGES: [(&str, RenderStages); 5] = [
    ("vertex", RenderStages::VERTEX),
    ("fragment", RenderStages::FRAGMENT),
    ("tile", RenderStages::TILE),
    ("object", RenderStages::OBJECT),
    ("mesh", RenderStages::MESH),
];

/// The texture slots the case asks about in every stage: 0 through 7, which covers every slot the
/// functions name (0 to 3, the array's second element at 4, 5 and 6) and the slot past them.
const SLOTS: [u64; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

/// The fixture's text in the dialect `extension` names.
///
/// The owned `reflection` fixtures sit beside the `engine` ones under the public fixtures, which
/// [`crate::fixture::Fixture`] reads, and are reached through the same devtools manifest, so a
/// host without Metal gets them from the git-indexed tree like every other committed fixture.
fn text(extension: &str) -> Result<String, String> {
    let path = fixtures_dir()
        .join("reflection")
        .join(format!("{FIXTURE}{extension}"));
    std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))
}

/// The AIR bitcode the runtime parses, as `llvm-as` assembles the fixture's `.ll`.
fn bitcode() -> Result<Vec<u8>, String> {
    let path = fixtures_dir()
        .join("reflection")
        .join(format!("{FIXTURE}{RUNTIME_TEXT}"));
    llvm_assemble(&path).map_err(|error| match error {
        AssembleError::Missing(why) => format!("no llvm-as: {why}"),
        AssembleError::Rejected(why) => format!("llvm-as rejected {FIXTURE}: {why}"),
    })
}

/// The case.
pub struct TextureAccesses;

/// The one case of the family.
pub fn all() -> Vec<TextureAccesses> {
    vec![TextureAccesses]
}

/// What one slot's answer records as: the Metal enumerant the access is, or `None` where no used
/// texture argument covers the slot.
fn access_value(access: Option<BindingAccess>) -> Value {
    match access {
        Some(access) => Value::Variant {
            name: "Some".to_owned(),
            payload: Box::new(Value::unit_variant(access.name())),
        },
        None => Value::unit_variant("None"),
    }
}

impl Case for TextureAccesses {
    fn name(&self) -> &'static str {
        "render-state-texture-accesses"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for entry in [VERTEX, FRAGMENT] {
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("reflection/{FIXTURE}{extension}"),
                    entry: entry.to_owned(),
                    source_sha256: crate::sha256_hex(text(extension)?.as_bytes()),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // Both committed texts, the attachment format the state is made for, and the stages and
        // slots the case asks about: a change to any of them is a different question, and the
        // recording's answer is then no answer for this run.
        let asked: Vec<u8> = STAGES
            .iter()
            .flat_map(|(_, stage)| stage.as_raw().to_le_bytes())
            .chain(SLOTS.iter().flat_map(|slot| slot.to_le_bytes()))
            .collect();
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("attachment", &ATTACHMENT.as_raw().to_le_bytes()),
            Input::bytes("stages-and-slots", &asked),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let bitcode = bitcode().map_err(|why| stop("library", why))?;
        let air_text = text(METAL_TEXT).map_err(|why| stop("library", why))?;
        let library = made(
            sites,
            "library",
            side.library(&ShaderSource {
                bitcode: &bitcode,
                air_text: &air_text,
                entry: VERTEX,
            }),
        )?;
        let pipeline = made(
            sites,
            "pipeline",
            side.render_pipeline_with_reflection(library, VERTEX, FRAGMENT, ATTACHMENT),
        )?;
        let mut answers = Node::branch("texture_access", Vec::new());
        for (name, stage) in STAGES {
            let at = format!("texture_access/{name}");
            let accesses = match side.texture_accesses(pipeline, stage, &SLOTS) {
                Ok(accesses) => accesses,
                Err(failure) => {
                    sites.push(Node::leaf(at.clone(), Observed::value(failure.recorded())));
                    return Err(Abort { at, failure });
                }
            };
            let mut stage_node = Node::branch(name, Vec::new());
            for (slot, access) in SLOTS.iter().zip(accesses) {
                stage_node = stage_node.with(Node::leaf(
                    slot.to_string(),
                    Observed::value(access_value(access)),
                ));
            }
            answers = answers.with(stage_node);
        }
        sites.push(answers);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every stage the case asks about is one `MTLRenderStages` bit, which is what Metal states a
    /// binding list for. A union would have no Metal answer, and a side answers `Unavailable` for
    /// one, so a case that grew a union here would stop recording rather than record a guess.
    #[test]
    fn every_stage_asked_about_is_one_bit() {
        for (name, stage) in STAGES {
            assert_eq!(stage.as_raw().count_ones(), 1, "{name}");
        }
    }

    /// The fixture declares the texture arguments the probe declares, at the same slots, with the
    /// array of two at slot 3 and the unreferenced argument at slot 5. This is what makes the case
    /// the probe's question rather than a new one, and it reads the committed AIR rather than
    /// trusting the MSL beside it.
    #[test]
    fn the_fixture_declares_the_probes_texture_arguments() {
        let air = text(METAL_TEXT).expect("the typed AIR");
        for wanted in [
            r#"!"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"r""#,
            r#"!"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"w""#,
            r#"!"air.location_index", i32 2, i32 1, !"air.read_write""#,
            r#"!"air.location_index", i32 3, i32 2, !"air.write""#,
            r#"!"air.location_index", i32 5, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"unused", !"air.arg_unused""#,
            r#"!"air.location_index", i32 6, i32 1, !"air.sample""#,
            r#"!"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"vt""#,
        ] {
            assert!(air.contains(wanted), "the fixture's AIR lacks {wanted}");
        }
    }

    /// The case's inputs cover both committed texts, so a regenerated fixture makes the case
    /// unrecorded rather than comparing against Metal's answer for other functions.
    #[test]
    fn the_inputs_fingerprint_both_texts() {
        let inputs = TextureAccesses.inputs().expect("the inputs");
        for (name, extension) in [("air", RUNTIME_TEXT), ("air-typed", METAL_TEXT)] {
            let input = inputs
                .iter()
                .find(|input| input.name == name)
                .unwrap_or_else(|| panic!("the {name} input"));
            assert_eq!(
                input.digest,
                Digest::sha256(text(extension).expect("the text").as_bytes())
            );
        }
    }
}
