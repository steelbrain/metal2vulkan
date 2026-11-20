//! `MTLRenderPipelineFunctionsDescriptor`: the three per-stage lists of functions a program hands
//! `newRenderPipelineStateWithAdditionalBinaryFunctions:error:`, and what the descriptor answers
//! for each after a program has written them.
//!
//! The class was reached at none of its four items. What a case can say about it is a **round
//! trip**: three properties, written in a stated order and read straight back. That is a small
//! claim and the family [`crate::cases::pipeline_functions`] states it as one — but it is the only
//! claim the class has, because the one call that consumes the descriptor needs a binary archive
//! (the reason [`crate::pipeline_facts`] gives for leaving every binary-function item out), and
//! nothing else observes it.
//!
//! # What is compared, and what cannot be
//!
//! **Lengths and names, with Metal's nil folded to a length of zero.** Metal's getters answer nil
//! for a property no program wrote, and an empty array for one written with one; the runtime has
//! no descriptor **object** at all — a program hands it a
//! `RenderPipelineFunctionsDescriptor`
//! whose three fields are `Vec`s — so nil and empty are one answer on our side and there is
//! nothing there to observe the difference with. That is the same fold, for the same reason,
//! that [`crate::compute_descriptors::SuppliedLists`] applies to `preloadedLibraries` and
//! `linkedFunctions`, and it is stated here rather than widened quietly: a case that compared
//! nil against empty would be comparing a shape one side cannot hold.
//!
//! **A name, not a count.** The list holds the functions' `name`s in order, so a side that
//! dropped one, reordered two, or wrote a stage's list into another stage's property reads a
//! different list rather than the same length. A count alone would pass all three of those.
//!
//! **Our side's answer is a round trip through its own `Vec`.** The family therefore pins Metal's
//! behaviour and our type's contract; it does not exercise a computation of ours, and says so
//! rather than reading as more than it is.

use crate::transcript::{Node, NodeName, Observed, Value};

/// One write a case makes to one of the descriptor's three properties.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Write<'a> {
    /// `setXAdditionalBinaryFunctions:` with the library's functions of these names, in order.
    /// An empty slice is Metal's empty array.
    Functions(&'a [&'a str]),
    /// `setXAdditionalBinaryFunctions:nil`, which puts the property back to what `init` left.
    Nil,
}

impl Write<'_> {
    /// The words a case's input fingerprint covers.
    fn words(&self) -> Vec<u8> {
        match self {
            Write::Functions(names) => {
                let mut out = b"functions".to_vec();
                for name in *names {
                    out.push(b'/');
                    out.extend(name.as_bytes());
                }
                out
            }
            Write::Nil => b"nil".to_vec(),
        }
    }
}

/// What one case writes to the descriptor: the writes each property takes, in order.
///
/// A sequence rather than one value a property, because "set and then set again" and "set and then
/// put back to nil" are the two questions a mutable property has beyond its first write, and a
/// single value could state neither.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FunctionsDescriptorSpec<'a> {
    /// The writes `vertexAdditionalBinaryFunctions` takes.
    pub vertex: &'a [Write<'a>],
    /// The writes `fragmentAdditionalBinaryFunctions` takes.
    pub fragment: &'a [Write<'a>],
    /// The writes `tileAdditionalBinaryFunctions` takes.
    pub tile: &'a [Write<'a>],
}

impl<'a> FunctionsDescriptorSpec<'a> {
    /// The descriptor `init` leaves: no write to any property.
    pub const fn untouched() -> Self {
        Self {
            vertex: &[],
            fragment: &[],
            tile: &[],
        }
    }

    /// The descriptor with these writes to the three properties.
    pub const fn new(
        vertex: &'a [Write<'a>],
        fragment: &'a [Write<'a>],
        tile: &'a [Write<'a>],
    ) -> Self {
        Self {
            vertex,
            fragment,
            tile,
        }
    }

    /// Every name the case needs a function of, so a side makes each one once.
    pub fn names(&self) -> Vec<&'a str> {
        let mut out = Vec::new();
        for writes in [self.vertex, self.fragment, self.tile] {
            for write in writes {
                if let Write::Functions(names) = write {
                    for name in *names {
                        if !out.contains(name) {
                            out.push(*name);
                        }
                    }
                }
            }
        }
        out
    }

    /// The words a case's input fingerprint covers: every write to every property, in order.
    pub fn words(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for (stage, writes) in [
            ("vertex", self.vertex),
            ("fragment", self.fragment),
            ("tile", self.tile),
        ] {
            out.extend(stage.as_bytes());
            for write in writes {
                out.push(b'|');
                out.extend(write.words());
            }
        }
        out
    }
}

/// What the descriptor answers after a case's writes: each property's functions by `name`, in
/// order, with a nil list folded to no names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionsDescriptorFacts {
    /// `vertexAdditionalBinaryFunctions`.
    pub vertex: Vec<String>,
    /// `fragmentAdditionalBinaryFunctions`.
    pub fragment: Vec<String>,
    /// `tileAdditionalBinaryFunctions`.
    pub tile: Vec<String>,
}

/// One property's answer as a site records it: its length, then its names in order, so a length
/// that agrees with a list that does not is a readable difference.
fn list(name: &'static str, names: &[String]) -> Node {
    let mut node = Node::branch(name, Vec::new()).with(Node::leaf(
        "count",
        Observed::value(Value::Uint(names.len() as u64)),
    ));
    for (at, held) in names.iter().enumerate() {
        node = node.with(Node::leaf(
            format!("name/{at}"),
            Observed::value(Value::Lines(vec![held.clone()])),
        ));
    }
    node
}

impl FunctionsDescriptorFacts {
    /// The facts as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        Node::branch(name, Vec::new())
            .with(list("vertex_additional_binary_functions", &self.vertex))
            .with(list("fragment_additional_binary_functions", &self.fragment))
            .with(list("tile_additional_binary_functions", &self.tile))
    }
}
