//! Every `MTLPipelineOption` value, and every combination of them, against what each one makes a
//! pipeline creation answer.
//!
//! `MTLPipelineOption` is three defined bits and a `None`:
//!
//! | bit | name | what it is claimed to do |
//! |---|---|---|
//! | — | `MTLPipelineOptionNone` | the creation answers no reflection |
//! | `1 << 0` | `MTLPipelineOptionBindingInfo` (and the deprecated `MTLPipelineOptionArgumentInfo`, the same bit) | the creation answers the pipeline's reflection |
//! | `1 << 1` | `MTLPipelineOptionBufferTypeInfo` | the same, with the buffers' types |
//! | `1 << 2` | `MTLPipelineOptionFailOnBinaryArchiveMiss` | a creation that lists a binary archive fails unless an archive holds its pipeline |
//!
//! Three bits are eight values, and [`MASKS`] is all eight: every single bit, every pair, the
//! triple, and `None`. The suite reached exactly **one** of them before this family —
//! `BindingInfo`, through [`crate::cases::function_bindings`] — so the claims the other seven
//! stand on were the runtime's own documentation and nothing else.
//!
//! # The two claims the sweep is pointed at
//!
//! - **`BufferTypeInfo` is the same as `BindingInfo`.** The runtime's documentation said so
//!   outright: "`MTLPipelineOptionBufferTypeInfo`: the same as
//!   `BINDING_INFO` (Metal's reflection carries the buffers' types under either)". If that is
//!   wrong, the mask `1 << 1` alone is where it shows: Metal would answer a reflection with no
//!   buffer struct type, or no reflection at all, and we would answer the full one.
//! - **`FailOnBinaryArchiveMiss` does nothing without an archive.** The runtime's documentation
//!   says a creation "that lists at least one archive is refused unless the archives hold the
//!   archive key of each of its stages" and that "without archives, or without the option,
//!   archives only make creation faster". No case lists an archive here, so every mask with
//!   `1 << 2` in it must answer exactly what the same mask without it answers — and a creation
//!   that refused instead would be a refusal of something plain Metal does.
//!
//! # What is compared, and why it is the whole reflection
//!
//! Each case records whether the creation was refused, whether a reflection came, and — where one
//! came — **every binding of it**, in the shape [`crate::function_bindings`] already compares: the
//! name, the type, the access, the index, whether it is used, whether it is an argument, and the
//! buffer, texture or threadgroup facts each kind carries, the buffer's struct members among them.
//! A reflection compared only by its presence would say nothing about which option filled it in.
//!
//! # Why a value is seen to change the answer
//!
//! `MTLPipelineOptionNone` answers **no** reflection and every mask holding `1 << 0` or `1 << 1`
//! answers one, so the sweep's own rows disagree with each other at the reflection site. That is
//! what makes the family a measurement rather than eight identical passes: a side that ignored its
//! options and always reflected, or never did, fails at least one row of every sweep.
//!
//! # Why this family is safe
//!
//! Each case makes one library and compiles one kernel. Nothing is allocated for the device, no
//! command buffer is made, nothing is dispatched, and no undefined option bit is ever passed:
//! `MASKS` holds only the bits Apple's header names, so no case asks Metal to interpret a value it
//! has no meaning for.

use crate::function_bindings::BindingFacts;
use crate::transcript::{Node, NodeName, Observed, Value};

/// Every `MTLPipelineOption` value: the three bits Apple's header names, in all eight
/// combinations, each with the name a site spells it by.
///
/// The deprecated `MTLPipelineOptionArgumentInfo` is not a ninth row: it is the same bit as
/// `MTLPipelineOptionBindingInfo` (`1 << 0`), so a row for it would be the same creation under
/// another spelling. Bits above `1 << 2` are not here either, and deliberately: Apple's header
/// names none of them, so a creation asked for one is asking what an undefined value means, which
/// is not a question this suite answers for either implementation.
pub const MASKS: [(u64, &str); 8] = [
    (0, "none"),
    (1, "binding-info"),
    (2, "buffer-type-info"),
    (3, "binding-info-and-buffer-type-info"),
    (4, "fail-on-binary-archive-miss"),
    (5, "binding-info-and-fail-on-miss"),
    (6, "buffer-type-info-and-fail-on-miss"),
    (7, "all-three"),
];

/// What one creation under one options value answered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnderOptions {
    /// Whether the creation was refused.
    ///
    /// A bool and not the error: an `NSError`'s words and a typed Rust error's are not comparable
    /// between implementations, for the reason [`crate::pipeline_handlers`] states. Every row of
    /// the sweep must answer `false` — a refusal of a creation plain Metal makes is a gap of ours
    /// — so this site is the one that catches an option bit we refuse and Metal accepts.
    pub refused: bool,
    /// The reflection's bindings, in the order the side answers them, or `None` where the creation
    /// answered no reflection at all (`MTLPipelineOptionNone`, as Metal leaves `reflection` nil).
    pub bindings: Option<Vec<BindingFacts>>,
}

impl UnderOptions {
    /// The answer of a creation that was refused.
    pub fn refused() -> Self {
        Self {
            refused: true,
            bindings: None,
        }
    }

    /// The sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let node = Node::branch(name, Vec::new()).with(Node::leaf(
            "refused",
            Observed::value(Value::Bool(self.refused)),
        ));
        match &self.bindings {
            // The count beside the bindings, as `library_facts` keeps one beside its name set: a
            // side that answered one binding where Metal answers four differs at a site a reader
            // sees before walking four subtrees.
            Some(bindings) => {
                let mut held = Node::branch("reflection", Vec::new()).with(Node::leaf(
                    "count",
                    Observed::value(Value::Uint(bindings.len() as u64)),
                ));
                for (at, binding) in bindings.iter().enumerate() {
                    held = held.with(binding.node(format!("binding-{at}")));
                }
                node.with(held)
            }
            None => node.with(Node::leaf(
                "reflection",
                Observed::value(Value::unit_variant("None")),
            )),
        }
    }
}
