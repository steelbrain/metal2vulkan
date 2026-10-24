//! The second seed case: a typed answer, where Metal answers otherwise than the command runs.
//!
//! An indirect render command the CPU encoded is read back three ways: `getCommandType`, the draw
//! getter, and the read-back of the whole record. For Metal's private primitive types 7, 8 and 9
//! the draw getter answers a draw of points — 0 — while the record keeps the type the command
//! draws, and `getCommandType` of an indexed draw of type 9 answers none. The case sweeps all ten
//! types in both forms and compares every one of the three answers.
//!
//! Nothing here executes the commands, so nothing faults: the case only encodes and reads back.
//!
//! Metal's answer is **carried**, and re-recording it is **forbidden**: Metal exposes this
//! read-back through private getters alone, so there is no public call for a Metal side to make.
//! The values are our own measurement of this chip, taken by the runtime lane that holds the
//! table, and record mode refuses the case rather than pretending it can ask Metal again.

use crate::case::{Abort, Case, Input, MetalSource, Side};
use crate::recording::ShaderRef;
use crate::replay::NOTICES_SITE;
use crate::transcript::{Node, Observed, Sites, Transcript, Value};

/// The case's name.
pub const PRIVATE_TYPE_GETTERS: &str = "indirect-private-type-getters";

/// Where the carried values were measured.
const MEASURED_IN: &str =
    "the runtime lane's indirect command buffer table (an Apple M3 Max, macOS 27.0.1)";

/// The runs the carried table stands on: the lane's own probes of the ten types in both forms.
pub const MEASURED_RUNS: u32 = 1;

/// Why record mode must not run the case on Metal.
const NO_PUBLIC_CALL: &str =
    "Metal answers this read-back through private getters alone, so no public call exists to \
     record it from";

/// Metal's ten primitive types, by raw value: 0 to 4 are the public ones, 5 to 9 the private ones.
const TYPES: std::ops::RangeInclusive<u64> = 0..=9;

/// `MTLIndirectCommandTypeDraw`.
const DRAW: u64 = 0x1;

/// `MTLIndirectCommandTypeDrawIndexed`.
const DRAW_INDEXED: u64 = 0x2;

/// The first private primitive type the draw getter answers as points.
const FIRST_ANSWERED_AS_POINTS: u64 = 7;

/// The type whose indexed draw `getCommandType` answers none for.
const NO_COMMAND_TYPE_WHEN_INDEXED: u64 = 9;

/// A case that compares an indirect render command's read-back with Metal's private getters.
pub struct IndirectPrivateTypes {
    name: &'static str,
}

/// The seed case.
pub fn private_type_getters() -> IndirectPrivateTypes {
    IndirectPrivateTypes {
        name: PRIVATE_TYPE_GETTERS,
    }
}

/// The site a type's form sits under.
fn form(indexed: bool) -> &'static str {
    if indexed {
        "indexed"
    } else {
        "direct"
    }
}

/// What the getters answer for `primitive_type` in the form `indexed`, as Metal's measured table
/// states it: `getCommandType`, the draw getter's primitive type, and the whole record's.
fn metal_answers(primitive_type: u64, indexed: bool) -> (u64, Option<u64>, Option<u64>) {
    let command_type = match (indexed, primitive_type) {
        (false, _) => DRAW,
        (true, NO_COMMAND_TYPE_WHEN_INDEXED) => 0,
        (true, _) => DRAW_INDEXED,
    };
    let answered = if primitive_type >= FIRST_ANSWERED_AS_POINTS {
        0
    } else {
        primitive_type
    };
    (command_type, Some(answered), Some(primitive_type))
}

/// The node one read-back emits.
fn read_back_node(
    indexed: bool,
    command_type: u64,
    answered: Option<u64>,
    recorded: Option<u64>,
) -> Node {
    let optional = |value: Option<u64>| match value {
        Some(value) => Value::Variant {
            name: "Some".to_owned(),
            payload: Box::new(Value::Uint(value)),
        },
        None => Value::unit_variant("None"),
    };
    Node::branch(form(indexed), Vec::new())
        .with(Node::leaf(
            "command_type",
            Observed::value(Value::Uint(command_type)),
        ))
        .with(Node::leaf(
            "answered_primitive_type",
            Observed::value(optional(answered)),
        ))
        .with(Node::leaf(
            "recorded_primitive_type",
            Observed::value(optional(recorded)),
        ))
}

impl IndirectPrivateTypes {
    /// The transcript Metal's measured table states.
    pub fn measured_transcript(&self) -> Result<Transcript, String> {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs()?));
        for primitive_type in TYPES {
            let mut node = Node::branch(format!("type/{primitive_type}"), Vec::new());
            for indexed in [false, true] {
                let (command_type, answered, recorded) = metal_answers(primitive_type, indexed);
                node = node.with(read_back_node(indexed, command_type, answered, recorded));
            }
            sites.push(node);
        }
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        Ok(sites.finish())
    }
}

impl Case for IndirectPrivateTypes {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::carried_only(MEASURED_IN, NO_PUBLIC_CALL)
    }

    fn allowances(&self) -> Vec<crate::allowance::Allowance> {
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the notices
        // site alone: every value of this family still compares against Metal's exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09), where it fired for all this case
        // and each row was divergent on that field and nothing else (replay-nvidia.log).
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        Ok(Vec::new())
    }

    fn carried_transcript(&self) -> Result<Option<(u32, Transcript)>, String> {
        Ok(Some((MEASURED_RUNS, self.measured_transcript()?)))
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // What the case feeds: the primitive types it sets, in order, and the two forms. An
        // indexed draw's own parameters are the same for every type (6 indices of 16 bits from a
        // buffer's start, one instance, no base), so the fingerprint names them once.
        let mut words = Vec::new();
        for primitive_type in TYPES {
            words.extend(primitive_type.to_le_bytes());
        }
        Ok(vec![
            Input::bytes("primitive-types", &words),
            Input::bytes("forms", b"direct,indexed"),
            Input::bytes(
                "indexed-draw",
                b"count 6, UInt16, offset 0, 1 instance, no base",
            ),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for primitive_type in TYPES {
            let mut node = Node::branch(format!("type/{primitive_type}"), Vec::new());
            for indexed in [false, true] {
                match side.indirect_draw_read_back(primitive_type, indexed) {
                    Ok(read) => {
                        node = node.with(read_back_node(
                            indexed,
                            read.command_type,
                            read.answered_primitive_type,
                            read.recorded_primitive_type,
                        ));
                    }
                    Err(failure) => {
                        let at = format!("type/{primitive_type}/{}", form(indexed));
                        sites.push(node);
                        sites.push(Node::leaf(
                            form(indexed),
                            Observed::value(failure.recorded()),
                        ));
                        return Err(Abort { at, failure });
                    }
                }
            }
            sites.push(node);
        }
        Ok(())
    }
}
