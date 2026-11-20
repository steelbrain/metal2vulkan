//! What a function says about **itself**: `MTLFunction`'s own getters, and the two objects they
//! answer with — `MTLVertexAttribute`, `MTLAttribute` and `MTLFunctionConstant`.
//!
//! [`crate::function_bindings`] compares what a function's reflection says its **arguments** are.
//! This is the other side of the same object: the kind of entry point it is, the patch it
//! evaluates, the stage-input attributes it declares, and the function constants it reads. Four
//! containers that the suite reached nowhere — `MTLFunction` at one item of twelve, and
//! `MTLAttribute`, `MTLVertexAttribute` and `MTLFunctionConstant` at none of theirs.
//!
//! Every field here is a number, an enumeration or a name, so the whole tree compares exactly and
//! nothing in it is the host's own. The one fact that is not portable is the `device` object
//! itself, and what the case asks of it is the one portable question the getter answers: did the
//! function come from the case's own device ([`FunctionFacts::device_is_the_cases`]), the shape
//! [`crate::argument_encoders::EncoderFacts`] already uses.
//!
//! # Order
//!
//! `vertexAttributes` and `stageInputAttributes` answer **declaration order**, which both sides
//! state, so an attribute's site is named by its position and the position is part of the answer.
//! `functionConstantsDictionary` answers a dictionary, which has no order at all, so a constant's
//! site is named by its own name — unique among a dictionary's keys by construction, which is what
//! keeps two sibling sites from sharing a name.
//!
//! # A specialized function
//!
//! A function made with constant values (`newFunctionWithName:constantValues:error:`) is the
//! function specialized: it leaves out an attribute its constants gate off, keeps the others, and
//! answers **no** function constants. That is the only way to reach `MTLFunctionConstantValues`,
//! whose three setters a case states as [`ConstantValue`]s.

use crate::case::field;
use crate::transcript::{Node, NodeName, Observed, Value};
use metal2vulkan::metal::{DataType, FunctionOptions, FunctionType, PatchType};

/// What one function answers of itself: every `MTLFunction` getter whose answer is portable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionFacts {
    /// `name`.
    pub name: String,
    /// `functionType`.
    pub function_type: FunctionType,
    /// `patchType`.
    pub patch_type: PatchType,
    /// `patchControlPointCount`: `None` where Metal answers -1 (no patch, or a patch that states
    /// no count).
    pub patch_control_point_count: Option<u32>,
    /// `vertexAttributes`, in declaration order; `None` is Metal's nil.
    pub vertex_attributes: Option<Vec<AttributeFacts>>,
    /// `stageInputAttributes`, in declaration order; `None` is Metal's nil.
    pub stage_input_attributes: Option<Vec<AttributeFacts>>,
    /// `functionConstantsDictionary`, by the constant's own name. A dictionary has no order, so
    /// the sites are named by the key rather than by a position neither side promises.
    pub constants: Vec<ConstantFacts>,
    /// `options`: the `MTLFunctionOptions` bits the function was made with.
    pub options: crate::case::Field<FunctionOptions>,
    /// Whether `device` is the device the case's other objects were made on.
    ///
    /// The device object itself is not comparable — its name and its identity are the host's — so
    /// what the case asks is the one portable question the getter answers.
    pub device_is_the_cases: crate::case::Field<bool>,
}

/// One stage-input attribute: `MTLVertexAttribute` and `MTLAttribute`, which state the same six
/// facts and are the two types `vertexAttributes` and `stageInputAttributes` answer with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttributeFacts {
    /// `name`.
    pub name: String,
    /// `attributeIndex`: its `[[attribute(n)]]`.
    pub attribute_index: u32,
    /// `attributeType`.
    pub attribute_type: DataType,
    /// `isActive`: whether the function reads the attribute.
    pub active: bool,
    /// `isPatchData`: a post-tessellation vertex function's per-patch input.
    pub patch_data: bool,
    /// `isPatchControlPointData`: a field of a post-tessellation vertex function's control points.
    pub patch_control_point_data: bool,
}

/// One function constant: `MTLFunctionConstant`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstantFacts {
    /// The dictionary's key, which is the constant's name.
    pub name: String,
    /// `type`.
    pub data_type: DataType,
    /// `index`: its `[[function_constant(n)]]`.
    pub index: u32,
    /// `required`: whether a function made from it needs a value for the constant.
    pub required: bool,
}

/// One value a case supplies for a function constant: where it goes, its Metal data type and its
/// bytes, exactly as `MTLFunctionConstantValues` takes them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstantValue {
    /// Which `MTLFunctionConstantValues` setter the value goes through.
    pub at: ConstantAt,
    /// The `MTLDataType` the setter is given.
    pub data_type: DataType,
    /// The bytes, one value's for [`ConstantAt::Index`] and [`ConstantAt::Name`] and `count`
    /// values' for [`ConstantAt::Range`].
    pub bytes: Vec<u8>,
}

impl ConstantValue {
    /// A `bool` constant at `index`, set through `setConstantValue:type:atIndex:`.
    pub fn bool_at(index: u32, value: bool) -> Self {
        Self {
            at: ConstantAt::Index(index),
            data_type: DataType::Bool,
            bytes: vec![u8::from(value)],
        }
    }

    /// A `bool` constant named `name`, set through `setConstantValue:type:withName:`.
    pub fn bool_named(name: &str, value: bool) -> Self {
        Self {
            at: ConstantAt::Name(name.to_owned()),
            data_type: DataType::Bool,
            bytes: vec![u8::from(value)],
        }
    }

    /// `count` `bool` constants from `first`, set through `setConstantValues:type:withRange:`.
    pub fn bools_from(first: u32, values: &[bool]) -> Self {
        Self {
            at: ConstantAt::Range {
                first,
                count: values.len() as u32,
            },
            data_type: DataType::Bool,
            bytes: values.iter().map(|value| u8::from(*value)).collect(),
        }
    }

    /// The words a case's input fingerprint covers: the setter, where it writes, the type and the
    /// bytes.
    pub fn words(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match &self.at {
            ConstantAt::Index(index) => {
                out.extend(b"index");
                out.extend(index.to_le_bytes());
            }
            ConstantAt::Range { first, count } => {
                out.extend(b"range");
                out.extend(first.to_le_bytes());
                out.extend(count.to_le_bytes());
            }
            ConstantAt::Name(name) => {
                out.extend(b"name");
                out.extend(name.as_bytes());
            }
        }
        out.extend(self.data_type.as_raw().to_le_bytes());
        out.extend(&self.bytes);
        out
    }
}

/// Which `MTLFunctionConstantValues` setter a [`ConstantValue`] goes through, which is the whole
/// of that class's own surface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstantAt {
    /// `setConstantValue:type:atIndex:`.
    Index(u32),
    /// `setConstantValues:type:withRange:`: `count` values from `first`.
    Range { first: u32, count: u32 },
    /// `setConstantValue:type:withName:`.
    Name(String),
}

/// An enumeration as a site records it: its Metal enumerator name beside its ordinal, so a diff
/// reads `MTLFunctionTypeVertex` rather than 1 and a value neither side has a name for is still
/// readable. The shape [`crate::function_bindings`] records an enumeration in.
fn named(name: &'static str, raw: u32) -> Value {
    Value::fields([
        ("name", Value::Lines(vec![name.to_owned()])),
        ("raw", Value::Uint(u64::from(raw))),
    ])
}

/// An `Option<u32>` as a site records it.
fn optional(value: Option<u32>) -> Value {
    match value {
        Some(value) => Value::Variant {
            name: "Some".to_owned(),
            payload: Box::new(Value::Uint(u64::from(value))),
        },
        None => Value::unit_variant("None"),
    }
}

impl FunctionFacts {
    /// The facts as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let mut node = Node::branch(name, Vec::new())
            .with(Node::leaf(
                "name",
                Observed::value(Value::Lines(vec![self.name.clone()])),
            ))
            .with(Node::leaf(
                "function_type",
                Observed::value(named(
                    self.function_type.name(),
                    self.function_type.as_raw(),
                )),
            ))
            .with(Node::leaf(
                "patch_type",
                Observed::value(named(self.patch_type.name(), self.patch_type.as_raw())),
            ))
            .with(Node::leaf(
                "patch_control_point_count",
                Observed::value(optional(self.patch_control_point_count)),
            ))
            .with(Node::leaf(
                "options",
                field(&self.options, |held| options(*held)),
            ))
            .with(Node::leaf(
                "device",
                field(&self.device_is_the_cases, |held| Value::Bool(*held)),
            ));
        node = node.with(attributes("vertex_attributes", &self.vertex_attributes));
        node = node.with(attributes(
            "stage_input_attributes",
            &self.stage_input_attributes,
        ));
        let mut dictionary = Node::branch("function_constants", Vec::new()).with(Node::leaf(
            "count",
            Observed::value(Value::Uint(self.constants.len() as u64)),
        ));
        for constant in &self.constants {
            dictionary = dictionary.with(constant.node(format!("constant/{}", constant.name)));
        }
        node.with(dictionary)
    }
}

/// `MTLFunctionOptions` as a site records it: the names of the bits set, and the raw value beside
/// them so a bit no name declares is still readable.
fn options(options: FunctionOptions) -> Value {
    let names = FunctionOptions::KNOWN
        .iter()
        .filter(|(bit, _)| !bit.is_empty() && options.contains(*bit))
        .map(|(_, name)| *name);
    Value::fields([
        ("names", Value::set(names)),
        ("raw", Value::Uint(u64::from(options.as_raw()))),
    ])
}

/// One of the two attribute lists as a site records it: Metal's nil as a leaf, a list as a branch
/// of its length and one child a position.
///
/// A list of none and nil are two different answers and compare as two different shapes, which is
/// what the two forms buy: a side that answered an empty array where the other answered nil is a
/// readable difference rather than an equal one.
fn attributes(name: &'static str, held: &Option<Vec<AttributeFacts>>) -> Node {
    let Some(held) = held else {
        return Node::leaf(name, Observed::value(Value::unit_variant("None")));
    };
    let mut node = Node::branch(name, Vec::new()).with(Node::leaf(
        "count",
        Observed::value(Value::Uint(held.len() as u64)),
    ));
    for (at, attribute) in held.iter().enumerate() {
        node = node.with(attribute.node(format!("attribute/{at}")));
    }
    node
}

impl AttributeFacts {
    /// The attribute's six facts under one node.
    fn node(&self, name: impl Into<NodeName>) -> Node {
        Node::branch(name, Vec::new())
            .with(Node::leaf(
                "name",
                Observed::value(Value::Lines(vec![self.name.clone()])),
            ))
            .with(Node::leaf(
                "attribute_index",
                Observed::value(Value::Uint(u64::from(self.attribute_index))),
            ))
            .with(Node::leaf(
                "attribute_type",
                Observed::value(named(
                    self.attribute_type.name(),
                    self.attribute_type.as_raw(),
                )),
            ))
            .with(Node::leaf(
                "active",
                Observed::value(Value::Bool(self.active)),
            ))
            .with(Node::leaf(
                "patch_data",
                Observed::value(Value::Bool(self.patch_data)),
            ))
            .with(Node::leaf(
                "patch_control_point_data",
                Observed::value(Value::Bool(self.patch_control_point_data)),
            ))
    }
}

impl ConstantFacts {
    /// The constant's facts under one node. Its name is the dictionary's key, and the site is
    /// named by it, so the name is in the site rather than repeated as a child.
    fn node(&self, name: impl Into<NodeName>) -> Node {
        Node::branch(name, Vec::new())
            .with(Node::leaf(
                "data_type",
                Observed::value(named(self.data_type.name(), self.data_type.as_raw())),
            ))
            .with(Node::leaf(
                "index",
                Observed::value(Value::Uint(u64::from(self.index))),
            ))
            .with(Node::leaf(
                "required",
                Observed::value(Value::Bool(self.required)),
            ))
    }
}

/// One stage-input attribute the translator reflects, in the vocabulary every side answers in.
pub(crate) fn attribute(attribute: &metal2vulkan::reflection::Attribute) -> AttributeFacts {
    AttributeFacts {
        name: attribute.name().to_owned(),
        attribute_index: attribute.attribute_index(),
        attribute_type: attribute.attribute_type(),
        active: attribute.is_active(),
        patch_data: attribute.is_patch_data(),
        patch_control_point_data: attribute.is_patch_control_point_data(),
    }
}

/// The constant values the case states, as the translator's `MTLFunctionConstantValues`, which
/// the runtime takes.
///
/// The runtime holds values by index and by name, which is the two single setters; a range is
/// **expanded** into one value an index from its first, which is what
/// `setConstantValues:type:withRange:` means. The bytes are split by the type's own size, so a
/// range of three `bool`s is three one-byte values at three indices.
pub(crate) fn constant_values(
    values: &[ConstantValue],
) -> metal2vulkan::library::FunctionConstantValues {
    let mut built = metal2vulkan::library::FunctionConstantValues::defaults();
    for value in values {
        let one = |bytes: &[u8]| {
            metal2vulkan::library::FunctionConstantValue::new(value.data_type, bytes.to_vec())
        };
        match &value.at {
            ConstantAt::Index(index) => built = built.with_index(*index, one(&value.bytes)),
            ConstantAt::Name(name) => built = built.with_name(name.clone(), one(&value.bytes)),
            ConstantAt::Range { first, count } => {
                let count = *count as usize;
                let each = match count {
                    0 => 0,
                    count => value.bytes.len() / count,
                };
                for step in 0..count {
                    let at = step * each;
                    built = built.with_index(
                        first + step as u32,
                        one(&value.bytes[at..at.saturating_add(each).min(value.bytes.len())]),
                    );
                }
            }
        }
    }
    built
}
