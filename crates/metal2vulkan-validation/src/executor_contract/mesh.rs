//! The executor contract of a mesh render pipeline's draw: what an object or mesh function's row
//! or case needs that the executors do not supply, decided from its reflected structure
//! ([`ShaderInterface::mesh_stage`], its bindings) and, for a case, the stages its draw names
//! ([`MeshDraw`]). Classification and the case checker read the one function,
//! [`mesh_requirements`]. Both executors draw a mesh render pipeline: the Metal oracle
//! (`metal::platform`) and the engine's candidate through the runtime (`runtime_case`).
//!
//! - [`ToolingRequirement::MeshCompanion`]: a stage the pipeline needs that nothing supplies.
//!   Metal links a mesh render pipeline from an object function (optional), a mesh function, and a
//!   fragment function wherever it rasterizes: its descriptor validation stops a rasterizing
//!   pipeline without one ("fragmentFunction must not be nil", M3 Max, macOS 27.0.1), and a
//!   pipeline with rasterization disabled draws with none, its mesh function's device writes
//!   landing (measured on the same device). So an object function needs a mesh stage, a mesh
//!   function that reads a payload needs the object stage that writes it, and a mesh function
//!   observed through its rasterized output needs a fragment stage. A case supplies each by naming
//!   a partner of its own module, or by stating what generated stages take the partners' places
//!   (`companion::MeshStages`): a mesh case's observer fragment of its output, an object case's
//!   mesh function and observer fragment of its launch ([`MeshObservation::Launch`]), a mesh
//!   case's object function that writes an authored payload
//!   ([`ObjectLaunch`](crate::case::ObjectLaunch)); the schema refuses
//!   a rasterizing draw with neither ([`MeshDraw`]), so a case never reaches Metal's abort. A row,
//!   which names none, needs each that its structure does and nothing generates: a mesh function
//!   whose device writes are observable draws with rasterization off, and one that writes its
//!   position draws with the generated observer of it, so neither needs a fragment stage; nor does
//!   one that sets none of its vertices or primitives ([`sets_mesh_outputs`]) and writes no
//!   observable memory, which has no effect a stage observes: it draws with rasterization off and
//!   an empty observation (`OutputSelection::None`), as a kernel that writes nothing does. An
//!   object function whose payload the launch observer reads ([`launch_payload_words`]) draws with
//!   the generated mesh stage, and a mesh function whose payload a generated object function
//!   writes ([`object_payload_bytes`]) with the generated object stage.

use super::writes_observable_memory;
use crate::air::reflect::{ShaderInterface, ShaderStage};
use crate::case::{MeshDraw, MeshObservation, Stage, MESH_PAYLOAD_BYTES};
use crate::requirement::ToolingRequirement;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// What an object or mesh row's draw is (`triage::StructuralTriage::mesh_draw`, which the focused
/// audit selects the family by): its stage, the payload it writes (an object function) or reads (a
/// mesh function), and the stages a row of it needs that nothing supplies ([`missing_stages`]),
/// which say which generated stage would let a case draw it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshDrawFacts {
    pub stage: Stage,
    /// The function declares a `[[payload]]` parameter.
    pub payload: bool,
    /// Its `air.arg_type_size`, where the AIR states one.
    pub payload_size: Option<u32>,
    /// The stages a row needs that nothing supplies, in pipeline order.
    pub missing: Vec<Stage>,
}

/// The most payload bytes the generated launch observer reads: its mesh function draws one
/// triangle (three vertices) for the row's grid pixel and one for each 16 bytes of payload, and
/// Metal's mesh function writes at most 256 vertices, so 85 triangles hold 84 payload pixels.
pub const LAUNCH_PAYLOAD_BYTES: u32 = 84 * 16;

/// The 4-byte words of an object function's payload that the generated launch observer reads
/// ([`MeshObservation::Launch`]): its declared size `payload` (`None`: the function declares no
/// payload; `Some(None)`: the AIR states no size) in whole words, at most
/// [`LAUNCH_PAYLOAD_BYTES`]; or why the observer cannot read it.
pub fn launch_payload_words(payload: Option<Option<u32>>) -> Result<u32, String> {
    match payload {
        None => Ok(0),
        Some(None) => {
            Err("the object function's payload states no size (air.arg_type_size)".into())
        }
        Some(Some(size)) if size % 4 != 0 => Err(format!(
            "the object function's payload holds {size} bytes, not whole 4-byte words: not \
             modelled"
        )),
        Some(Some(size)) if size > LAUNCH_PAYLOAD_BYTES => Err(format!(
            "the object function's payload holds {size} bytes, past the {LAUNCH_PAYLOAD_BYTES} \
             the generated launch observer reads"
        )),
        Some(Some(size)) => Ok(size / 4),
    }
}

/// The payload bytes a generated object function writes for a mesh function whose payload's
/// declared size is `size` (`None`: the AIR states none): whole 4-byte words, at most Metal's
/// [`MESH_PAYLOAD_BYTES`]; or why it cannot write them.
pub fn object_payload_bytes(size: Option<u32>) -> Result<u32, String> {
    let size = size.ok_or("the mesh function's payload states no size (air.arg_type_size)")?;
    if size % 4 != 0 {
        return Err(format!(
            "the mesh function's payload holds {size} bytes, not whole 4-byte words: not modelled"
        ));
    }
    if size as usize > MESH_PAYLOAD_BYTES {
        return Err(format!(
            "the mesh function's payload holds {size} bytes, past Metal's {MESH_PAYLOAD_BYTES}"
        ));
    }
    Ok(size)
}

/// The [`MeshDrawFacts`] of a row whose entry's text is `source_ll` and whose reflection is
/// `reflection`; `None` for any stage but an object or mesh function's.
pub fn mesh_draw_facts(source_ll: &str, reflection: &ShaderInterface) -> Option<MeshDrawFacts> {
    let own = reflection.mesh_stage.as_ref()?;
    let stage = match reflection.stage {
        ShaderStage::Object => Stage::Object,
        ShaderStage::Mesh => Stage::Mesh,
        _ => return None,
    };
    Some(MeshDrawFacts {
        stage,
        payload: own.payload.is_some(),
        payload_size: own.payload.as_ref().and_then(|payload| payload.size),
        missing: missing_stages(source_ll, reflection, None)
            .into_iter()
            .map(|(stage, _)| stage)
            .collect(),
    })
}

/// The pipeline stages a row or case of an object or mesh function needs and nothing supplies,
/// each with why: for a case, against the partners its `draw` names (a rasterizing draw names its
/// fragment function, which the schema checks); for a row (`draw` `None`), against none. Empty for
/// any other stage.
pub fn missing_stages(
    source_ll: &str,
    reflection: &ShaderInterface,
    draw: Option<&MeshDraw>,
) -> Vec<(Stage, &'static str)> {
    let Some(own) = &reflection.mesh_stage else {
        return Vec::new();
    };
    let names = |stage| draw.is_some_and(|draw| draw.partner(stage).is_some());
    let mut missing = Vec::new();
    match reflection.stage {
        ShaderStage::Object => {
            // Its draw rasterizes or writes through the mesh stage it launches: the partner a
            // case names, or the generated launch observer, which a case states and a row draws
            // with wherever it reads the payload.
            let generated = match draw {
                Some(draw) => draw.observation == Some(MeshObservation::Launch),
                None => {
                    launch_payload_words(own.payload.as_ref().map(|payload| payload.size)).is_ok()
                }
            };
            if !names(Stage::Mesh) && !generated {
                missing.push((
                    Stage::Mesh,
                    "an object function's payload and grid are drawn only by a mesh stage",
                ));
            }
        }
        ShaderStage::Mesh => {
            // The object stage that writes its payload: the partner a case names, or the
            // generated object function, which a case states and a row draws with wherever it
            // writes the payload.
            let generated = match draw {
                Some(draw) => draw.object_launch.is_some(),
                None => own
                    .payload
                    .as_ref()
                    .is_some_and(|payload| object_payload_bytes(payload.size).is_ok()),
            };
            if own.payload.is_some() && !names(Stage::Object) && !generated {
                missing.push((
                    Stage::Object,
                    "a mesh function's payload is written only by an object stage",
                ));
            }
            // A row is observed through its device writes where it has any, with rasterization
            // off, and otherwise through its rasterized output: the generated observer fragment
            // reads the position it writes (`air.set_position_mesh`). One that sets no vertex or
            // primitive and writes nothing has no effect: rasterization off observes all of it,
            // the empty output. A case's draw names the fragment function it rasterizes with, or
            // the observation its observer reads.
            let sets_position =
                own.writes_position() && source_ll.contains("@air.set_position_mesh(");
            if draw.is_none()
                && !writes_observable_memory(source_ll)
                && !sets_position
                && sets_mesh_outputs(source_ll)
            {
                missing.push((
                    Stage::Fragment,
                    "a mesh function that sets vertices or primitives without their position, \
                     and writes no observable memory, has no output a fragment stage observes",
                ));
            }
        }
        ShaderStage::Vertex
        | ShaderStage::TessellationEvaluation
        | ShaderStage::Fragment
        | ShaderStage::Kernel => {}
    }
    missing
}

/// Whether the entry whose text is `source_ll` sets any of its mesh's vertices or primitives: it
/// calls a mesh output setter (`air.set_*_mesh`) other than its primitive count's. One that sets
/// only a count draws primitives over no vertex it wrote, so rasterizing them shows nothing it
/// defined.
pub fn sets_mesh_outputs(source_ll: &str) -> bool {
    use metal2vulkan_air::intrinsics::{AirIntrinsic, Mesh};
    super::air_call_counts(source_ll).keys().any(|name| {
        matches!(
            AirIntrinsic::parse(name),
            Some(AirIntrinsic::Mesh(op))
                if !matches!(op, Mesh::PrimitiveCount | Mesh::ThreadgroupsPerGrid)
        )
    })
}

/// The mesh requirements of a row or case (`draw` `None` for a row) whose entry's reflection is
/// `reflection`: [`ToolingRequirement::MeshCompanion`] where [`missing_stages`] names a stage.
/// Empty for any other stage. An object function's threadgroup memory needs nothing: both
/// executors set its lengths (`setObjectThreadgroupMemoryLength:atIndex:`).
pub fn mesh_requirements(
    source_ll: &str,
    reflection: &ShaderInterface,
    draw: Option<&MeshDraw>,
) -> BTreeSet<ToolingRequirement> {
    let mut requirements = BTreeSet::new();
    if !missing_stages(source_ll, reflection, draw).is_empty() {
        requirements.insert(ToolingRequirement::MeshCompanion);
    }
    requirements
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::reflect::{
        inspect, MeshField, MeshFieldRole, MeshTopology, ResourceAccess, ResourceKind,
    };
    use crate::case::MeshGrid;

    /// The text of the owned public fixture `engine/<name>.ll`.
    fn fixture(name: &str) -> String {
        std::fs::read_to_string(format!(
            "{}/fixtures/public/engine/{name}.ll",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap()
    }

    /// The projection of `module` onto its entry `entry` of `stage`, and its reflection.
    fn reflected(module: &str, stage: Stage, entry: &str) -> (String, ShaderInterface) {
        let text = crate::air::entry_text(module, entry).unwrap().into_owned();
        let reflection = inspect(&text, stage, entry, &[]).unwrap();
        (text, reflection)
    }

    fn draw(object: Option<&str>, mesh: Option<&str>, fragment: Option<&str>) -> MeshDraw {
        MeshDraw {
            object: object.map(str::to_string),
            mesh: mesh.map(str::to_string),
            fragment: fragment.map(str::to_string),
            grid: MeshGrid::Threadgroups([1, 1, 1]),
            threads_per_object_threadgroup: [1, 1, 1],
            threads_per_mesh_threadgroup: [1, 1, 1],
            rasterization: fragment.is_some(),
            observation: None,
            object_launch: None,
        }
    }

    /// An object and a mesh function reflect as a kernel's interface at their own stage, with
    /// their own facts: the payload's size, alignment and type, the object function's grid
    /// handle and the grid it sets, and the mesh function's vertex and primitive fields (builtin
    /// role or user key, and type), counts and topology.
    #[test]
    fn object_and_mesh_functions_reflect_their_own_stage_s_facts() {
        let module = fixture("mesh_draws");
        let (_, object) = reflected(&module, Stage::Object, "mesh_draws_object");
        assert_eq!(object.stage, ShaderStage::Object);
        let own = object.mesh_stage.as_ref().unwrap();
        let payload = own.payload.as_ref().unwrap();
        assert_eq!(
            (payload.param_index, payload.size, payload.align),
            (0, Some(32), Some(16))
        );
        assert_eq!(payload.type_name.as_deref(), Some("DrawsPayload"));
        assert!(own.mesh_grid && own.sets_mesh_grid);
        assert_eq!(own.outputs, None);
        // The grid an entry sets is read from what it reaches, not from its module: the whole
        // module reflects each entry as its projection does, and its mesh function, whose
        // module's object function sets a grid, sets none.
        for (stage, entry, sets) in [
            (Stage::Object, "mesh_draws_object", true),
            (Stage::Mesh, "mesh_draws_grid", false),
        ] {
            let whole = inspect(&module, stage, entry, &[]).unwrap();
            assert_eq!(whole.mesh_stage.unwrap().sets_mesh_grid, sets, "{entry}");
        }
        let [buffer] = &object.bindings[..] else {
            panic!("{:?}", object.bindings);
        };
        assert_eq!(
            (buffer.kind, buffer.metal_index, buffer.access),
            (ResourceKind::Buffer, 0, Some(ResourceAccess::ReadOnly))
        );

        let (_, grid) = reflected(&module, Stage::Mesh, "mesh_draws_grid");
        assert_eq!(grid.stage, ShaderStage::Mesh);
        assert!(grid.bindings.is_empty());
        let own = grid.mesh_stage.as_ref().unwrap();
        assert_eq!((own.payload.is_some(), own.mesh_grid), (false, false));
        let outputs = own.outputs.as_ref().unwrap();
        assert_eq!(
            (
                outputs.max_vertices,
                outputs.max_primitives,
                outputs.topology
            ),
            (3, 1, MeshTopology::Triangle)
        );
        assert_eq!(
            outputs.vertex,
            [MeshField {
                role: MeshFieldRole::Position,
                type_name: Some("float4".into()),
            }]
        );
        assert_eq!(
            outputs.primitive,
            [("1a", 0), ("1b", 1)].map(|(name, slot)| MeshField {
                role: MeshFieldRole::PrimitiveData {
                    slot,
                    key: format!("generated({name}Dv4_j)"),
                },
                type_name: Some("uint4".into()),
            })
        );
        assert!(own.writes_position());

        let (_, mesh) = reflected(&module, Stage::Mesh, "mesh_draws_mesh");
        let payload = mesh.mesh_stage.as_ref().unwrap().payload.as_ref().unwrap();
        assert_eq!((payload.param_index, payload.size), (1, Some(32)));
    }

    /// A row names no partner: an object function needs a mesh stage unless the generated launch
    /// observer reads its payload, and a mesh function that reads a payload needs the object stage
    /// writing it unless a generated object function writes it. A mesh function needs no fragment
    /// stage where
    /// one is generated or none is needed: one that writes its position draws with the generated
    /// observer of it, one that writes device memory with rasterization off, and one that sets no
    /// vertex or primitive and writes nothing (an empty body, or a primitive count alone, constant
    /// or read from its payload) with rasterization off and an empty observation; one that sets
    /// vertices without their position and writes nothing needs a fragment stage nothing can
    /// observe it through. Each such row carries `mesh_companion`, and a row that needs no stage
    /// carries no mesh requirement: both executors draw it.
    #[test]
    fn a_row_needs_the_stages_its_structure_does() {
        let draws = fixture("mesh_draws");
        let barrier = fixture("mesh_barrier");
        let grid = fixture("mesh_object_grid");
        let threadgroup = fixture("icb_object_threadgroup");
        let limits = fixture("mesh_pipeline_limits");
        let rows = [
            (&draws, Stage::Object, "mesh_draws_object", &[][..]),
            (&draws, Stage::Mesh, "mesh_draws_grid", &[][..]),
            (&draws, Stage::Mesh, "mesh_draws_mesh", &[][..]),
            (&barrier, Stage::Mesh, "mesh_barrier_writer", &[][..]),
            (&grid, Stage::Mesh, "mesh_object_grid_mesh", &[][..]),
            (
                &threadgroup,
                Stage::Mesh,
                "icb_object_threadgroup_mesh",
                &[][..],
            ),
            (&limits, Stage::Mesh, "mesh_pipeline_limits_mesh", &[][..]),
            (
                &limits,
                Stage::Mesh,
                "mesh_pipeline_limits_mesh_alone",
                &[][..],
            ),
        ];
        for (module, entry) in [
            (&grid, "mesh_object_grid_mesh"),
            (&threadgroup, "icb_object_threadgroup_mesh"),
            (&limits, "mesh_pipeline_limits_mesh"),
        ] {
            let (text, _) = reflected(module, Stage::Mesh, entry);
            assert!(!sets_mesh_outputs(&text), "{entry}");
            assert!(!writes_observable_memory(&text), "{entry}");
        }
        for (module, stage, entry, stages) in rows {
            let (text, reflection) = reflected(module, stage, entry);
            let missing = missing_stages(&text, &reflection, None)
                .into_iter()
                .map(|(stage, _)| stage)
                .collect::<Vec<_>>();
            assert_eq!(missing, stages, "{entry}");
            let expected = if stages.is_empty() {
                BTreeSet::new()
            } else {
                BTreeSet::from([ToolingRequirement::MeshCompanion])
            };
            assert_eq!(
                mesh_requirements(&text, &reflection, None),
                expected,
                "{entry}"
            );
        }
        // A mesh function whose vertices state no position, and that writes no device memory.
        let (text, mut unobserved) = reflected(&draws, Stage::Mesh, "mesh_draws_grid");
        let outputs = unobserved
            .mesh_stage
            .as_mut()
            .and_then(|own| own.outputs.as_mut())
            .unwrap();
        outputs
            .vertex
            .retain(|field| field.role != MeshFieldRole::Position);
        let missing = missing_stages(&text, &unobserved, None);
        assert_eq!(
            missing.iter().map(|(stage, _)| *stage).collect::<Vec<_>>(),
            [Stage::Fragment]
        );
        assert_eq!(
            mesh_requirements(&text, &unobserved, None),
            BTreeSet::from([ToolingRequirement::MeshCompanion])
        );
        // A mesh function whose payload no generated object function writes: of no stated size,
        // or not whole words.
        let (text, mut mesh) = reflected(&draws, Stage::Mesh, "mesh_draws_mesh");
        for size in [None, Some(30)] {
            mesh.mesh_stage
                .as_mut()
                .and_then(|own| own.payload.as_mut())
                .unwrap()
                .size = size;
            let missing = missing_stages(&text, &mesh, None);
            assert_eq!(
                missing.iter().map(|(stage, _)| *stage).collect::<Vec<_>>(),
                [Stage::Object]
            );
        }
        // An object function whose payload the launch observer cannot read: of no stated size,
        // or not whole words.
        let (text, mut object) = reflected(&draws, Stage::Object, "mesh_draws_object");
        for size in [None, Some(30)] {
            object
                .mesh_stage
                .as_mut()
                .and_then(|own| own.payload.as_mut())
                .unwrap()
                .size = size;
            let missing = missing_stages(&text, &object, None);
            assert_eq!(
                missing.iter().map(|(stage, _)| *stage).collect::<Vec<_>>(),
                [Stage::Mesh]
            );
            assert_eq!(
                mesh_requirements(&text, &object, None),
                BTreeSet::from([ToolingRequirement::MeshCompanion])
            );
        }
    }

    /// The launch observer reads a payload at its declared size in whole words, up to the pixels
    /// its triangles cover, and an object function without one through its grid alone.
    #[test]
    fn the_launch_observer_reads_whole_words_of_a_stated_payload() {
        assert_eq!(launch_payload_words(None), Ok(0));
        assert_eq!(launch_payload_words(Some(Some(32))), Ok(8));
        assert_eq!(
            launch_payload_words(Some(Some(LAUNCH_PAYLOAD_BYTES))),
            Ok(LAUNCH_PAYLOAD_BYTES / 4)
        );
        for unread in [None, Some(6), Some(LAUNCH_PAYLOAD_BYTES + 16)] {
            assert!(launch_payload_words(Some(unread)).is_err(), "{unread:?}");
        }
    }

    /// A generated object function writes a payload of a stated size in whole words, up to
    /// Metal's 16 KB.
    #[test]
    fn a_generated_object_writes_whole_words_of_a_stated_payload() {
        assert_eq!(object_payload_bytes(Some(32)), Ok(32));
        assert_eq!(
            object_payload_bytes(Some(MESH_PAYLOAD_BYTES as u32)),
            Ok(MESH_PAYLOAD_BYTES as u32)
        );
        for unwritten in [None, Some(6), Some(MESH_PAYLOAD_BYTES as u32 + 4)] {
            assert!(object_payload_bytes(unwritten).is_err(), "{unwritten:?}");
        }
    }

    /// An object function's threadgroup memory needs nothing of a case that names its partners:
    /// both executors set its lengths.
    #[test]
    fn an_object_function_s_threadgroup_memory_needs_nothing() {
        let module = fixture("mesh_draws");
        let (text, mut object) = reflected(&module, Stage::Object, "mesh_draws_object");
        let named = draw(None, Some("mesh_draws_mesh"), Some("mesh_draws_fragment"));
        let mut threadgroup = object.bindings[0].clone();
        threadgroup.kind = ResourceKind::ThreadgroupBuffer;
        object.bindings.push(threadgroup);
        assert!(mesh_requirements(&text, &object, Some(&named)).is_empty());
    }

    /// A case supplies a stage by naming its partner, and its draw names the fragment function it
    /// rasterizes with (the schema's check); what it leaves absent is still missing. Any other
    /// stage's entry has no mesh requirement.
    #[test]
    fn a_case_needs_the_stages_its_draw_does_not_name() {
        let module = fixture("mesh_draws");
        let (object_text, object) = reflected(&module, Stage::Object, "mesh_draws_object");
        let (mesh_text, mesh) = reflected(&module, Stage::Mesh, "mesh_draws_mesh");
        let missing = |text: &str, reflection: &ShaderInterface, draw: &MeshDraw| {
            missing_stages(text, reflection, Some(draw))
                .into_iter()
                .map(|(stage, _)| stage)
                .collect::<Vec<_>>()
        };
        let named = draw(None, Some("mesh_draws_mesh"), Some("mesh_draws_fragment"));
        assert_eq!(missing(&object_text, &object, &named), []);
        assert!(mesh_requirements(&object_text, &object, Some(&named)).is_empty());
        assert_eq!(
            missing(&object_text, &object, &draw(None, None, Some("f"))),
            [Stage::Mesh]
        );
        assert_eq!(
            missing(&object_text, &object, &draw(None, Some("m"), None)),
            []
        );
        // An object case's launch observation supplies its mesh stage.
        let mut launched = draw(None, None, None);
        launched.rasterization = true;
        launched.observation = Some(MeshObservation::Launch);
        assert_eq!(missing(&object_text, &object, &launched), []);
        assert!(mesh_requirements(&object_text, &object, Some(&launched)).is_empty());
        assert_eq!(
            missing(&mesh_text, &mesh, &draw(Some("o"), None, Some("f"))),
            []
        );
        assert_eq!(
            missing(&mesh_text, &mesh, &draw(None, None, Some("f"))),
            [Stage::Object]
        );
        assert_eq!(missing(&mesh_text, &mesh, &draw(Some("o"), None, None)), []);
        // A mesh case's object launch supplies its object stage.
        let mut payload = draw(None, None, Some("f"));
        payload.object_launch = Some(crate::case::ObjectLaunch {
            payload_b64: "AAAAAA==".into(),
            mesh_threadgroups: [1, 1, 1],
        });
        assert_eq!(missing(&mesh_text, &mesh, &payload), []);
        assert!(mesh_requirements(&mesh_text, &mesh, Some(&payload)).is_empty());

        let kernel = fixture("../kernel_copy_word");
        let reflection = inspect(&kernel, Stage::Kernel, "copy_word", &[]).unwrap();
        assert_eq!(reflection.mesh_stage, None);
        assert!(mesh_requirements(&kernel, &reflection, None).is_empty());
    }
}
