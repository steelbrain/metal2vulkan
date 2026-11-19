//! What a **library** says about itself, and the function a descriptor makes of it.
//!
//! [`crate::function_facts`] compares what one function answers of itself and
//! [`crate::function_bindings`] what its reflection says its arguments are. Both reach their
//! function through `newFunctionWithName:`, so the object that holds them — `MTLLibrary` — was
//! compared at two items of eleven: the two function-making calls. Its own four getters
//! (`device`, `functionNames`, `type`, `installName`) and `newFunctionWithDescriptor:error:` were
//! compared nowhere, and `MTLFunctionDescriptor` was a container no case touched at all.
//!
//! The getters are not decoration. `functionNames` is what a consumer enumerates a library by
//! before it knows any entry point's name; `type` is what separates an executable library from a
//! dynamic one, which refuses every `newFunctionWithName:`; and `installName` is the name a
//! dynamic library is resolved against. Our side answers the three from the metallib container's
//! own header and function list, so what this family holds to Metal is that reading of the
//! container.
//!
//! # `functionNames` is a set and a count, and the order is Metal's own
//!
//! Metal answers an `NSArray` and our side a `Vec`, but the site holds the names as a **set**
//! beside their **count**, because the order is not a fact either side states. Measured on the
//! oracle while recording: given the same metallib bytes, Metal answers the seven functions of
//! `px_samplers` in an order that is neither the container's function-list order, nor its reverse,
//! nor alphabetical, while our side answers the container's own order — the same order whether it
//! reads the metallib or the bare AIR bitcode. The repository's own reflection probe already reads
//! this getter `.sorted()` for that reason.
//!
//! What the set buys is that the comparison still bites: a missing name, an extra name, a name
//! spelled differently reads as exactly that in the diff, and the count beside it catches a side
//! that answered one name twice where a set would have collapsed the pair. What it gives up is
//! stated here rather than hidden: a side that answers the right names in another order passes
//! this family, and the order divergence above is an open item rather than a case.
//!
//! # A descriptor is three facts a function then carries
//!
//! `MTLFunctionDescriptor` has no getters a case can usefully read back off the descriptor: what
//! it is *for* is the function it makes. So its three settable facts are compared through that
//! function — `name` through [`crate::function_facts::FunctionFacts::name`], `constantValues` through the specialized
//! attribute lists and the empty constants dictionary, and `options` through
//! [`crate::function_facts::FunctionFacts::options`], which every case made by name answers `MTLFunctionOptionNone` for.
//! A descriptor whose options a side dropped reads as a different `options` field rather than as
//! nothing.

use crate::case::field;
use crate::transcript::{Node, NodeName, Observed, Value};
use metal2vulkan::library::LibraryType;
use metal2vulkan::metal::FunctionOptions;

/// What one library answers of itself: every `MTLLibrary` getter whose answer is portable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibraryFacts {
    /// `functionNames`, in the order the side answers them. The site compares the names as a set
    /// and their count, not their order (the module's prose says what was measured).
    pub function_names: Vec<String>,
    /// `type`.
    pub library_type: LibraryType,
    /// `installName`; `None` is Metal's nil.
    pub install_name: Option<String>,
    /// Whether `device` is the device the case's other objects were made on.
    ///
    /// The device object itself is not comparable — its name and its identity are the host's — so
    /// what the case asks is the one portable question the getter answers, as
    /// [`crate::function_facts::FunctionFacts::device_is_the_cases`] asks it of a function.
    pub device_is_the_cases: crate::case::Field<bool>,
}

/// `MTLFunctionDescriptor` as a case states it: the three facts a side sets on one before
/// `newFunctionWithDescriptor:error:`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionDescriptorSpec {
    /// `name`.
    pub name: String,
    /// `constantValues`: the values the descriptor carries, or none for nil.
    pub constants: Vec<crate::function_facts::ConstantValue>,
    /// `options`.
    pub options: FunctionOptions,
}

impl FunctionDescriptorSpec {
    /// The bytes a case's input fingerprint covers: the name, the options and each constant value.
    pub fn words(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(b"name");
        out.extend(self.name.as_bytes());
        out.extend(b"options");
        out.extend(self.options.as_raw().to_le_bytes());
        for value in &self.constants {
            out.extend(b"constant");
            out.extend(value.words());
        }
        out
    }
}

/// `MTLLibraryType` as a site records it: its Metal enumerator name beside the ordinal Apple's
/// header gives it, so a difference reads `MTLLibraryTypeDynamic` against
/// `MTLLibraryTypeExecutable` rather than 1 against 0.
///
/// Written here rather than taken from the type, which carries neither a name nor a raw value: the
/// mapping is the enumeration's own and the match is exhaustive over what the vocabulary holds.
/// The ordinals are `MTLLibraryType`'s (0 and 1), not the metallib header's `file_type` byte (0 and
/// 2) our side reads the type out of.
fn library_type(library_type: LibraryType) -> Value {
    let (name, raw) = match library_type {
        LibraryType::Dynamic => ("MTLLibraryTypeDynamic", 1u32),
        // `Executable` and whatever a later variant of a `#[non_exhaustive]` enumeration adds.
        // A new variant recorded as the executable one would be a silent wrong answer, so the
        // name says what was matched and the raw value stays the one Apple's header gives.
        LibraryType::Executable => ("MTLLibraryTypeExecutable", 0),
        _ => ("unknown", u32::MAX),
    };
    Value::fields([
        ("name", Value::Lines(vec![name.to_owned()])),
        ("raw", Value::Uint(u64::from(raw))),
    ])
}

impl LibraryFacts {
    /// The facts as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        // The count beside the set rather than the set alone: a set of seven names answered by a
        // side that listed one of them twice is the same set, and the count is what makes that
        // readable.
        let names = Node::branch("function_names", Vec::new())
            .with(Node::leaf(
                "count",
                Observed::value(Value::Uint(self.function_names.len() as u64)),
            ))
            .with(Node::leaf(
                "names",
                Observed::value(Value::set(self.function_names.iter().cloned())),
            ));
        Node::branch(name, Vec::new())
            .with(names)
            .with(Node::leaf(
                "library_type",
                Observed::value(library_type(self.library_type)),
            ))
            .with(Node::leaf(
                "install_name",
                Observed::value(match &self.install_name {
                    Some(held) => Value::Variant {
                        name: "Some".to_owned(),
                        payload: Box::new(Value::Lines(vec![held.clone()])),
                    },
                    None => Value::unit_variant("None"),
                }),
            ))
            .with(Node::leaf(
                "device",
                field(&self.device_is_the_cases, |held| Value::Bool(*held)),
            ))
    }
}
