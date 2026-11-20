//! A pipeline state made through Metal's `completionHandler:` form, beside the **same descriptor**
//! made through the synchronous sibling: the vocabulary for `MTLDevice`'s seven asynchronous
//! creation calls.
//!
//! `MTLDevice` has seven creation calls that answer through a block rather than returning:
//!
//! | handler form | synchronous sibling |
//! |---|---|
//! | `newRenderPipelineStateWithDescriptor:completionHandler:` | `newRenderPipelineStateWithDescriptor:error:` |
//! | `newRenderPipelineStateWithDescriptor:options:completionHandler:` | `newRenderPipelineStateWithDescriptor:options:reflection:error:` |
//! | `newComputePipelineStateWithFunction:completionHandler:` | `newComputePipelineStateWithFunction:error:` |
//! | `newComputePipelineStateWithFunction:options:completionHandler:` | `newComputePipelineStateWithFunction:options:reflection:error:` |
//! | `newComputePipelineStateWithDescriptor:options:completionHandler:` | `newComputePipelineStateWithDescriptor:options:reflection:error:` |
//! | `newRenderPipelineStateWithTileDescriptor:options:completionHandler:` | `newRenderPipelineStateWithTileDescriptor:options:reflection:error:` |
//! | `newRenderPipelineStateWithMeshDescriptor:options:completionHandler:` | `newRenderPipelineStateWithMeshDescriptor:options:reflection:error:` |
//!
//! The coverage map (`scripts/metal-coverage/coverage.tsv`) records each handler form as
//! implemented **by its synchronous sibling**, with the note "handler form: the call returns what
//! the handler receives". So the claim under test is exactly that note: *the asynchronous form
//! delivers what the synchronous one returns*. A case therefore takes **both paths on one side,
//! from one descriptor**, and emits both answers.
//!
//! # What each case holds, and why each site is there
//!
//! - **the arrival** ([`Arrival`]). The handler either ran inside the bound or it did not, and the
//!   site says which. This is not decoration: a timeout and a success must not look alike in a
//!   transcript, or a family of asynchronous cases passes by never running anything. Our leg
//!   answers [`Arrival::Answered`] by construction (below), so a Metal release whose handler stops
//!   arriving reads as a divergence at this site rather than as a quiet pass.
//! - **what was delivered**: whether a state came, and whether an error came. Two bools, not one:
//!   Metal's block takes `(state, error)` and a delivery of neither, or of both, is a different
//!   behaviour from a delivery of one. The **error path** is where this bites — an implementation
//!   that returned a nil state with a nil error, or that delivered a state beside an error, differs
//!   here and nowhere else.
//! - **the state's own facts**, where a state came: the whole portable surface
//!   [`crate::pipeline_facts`] already compares for the kind of state the descriptor makes
//!   ([`StateFacts`]). A creation that merely answered non-nil proves nothing, so what the case
//!   compares is what the delivered object says about itself.
//! - **the same three, through the synchronous sibling**, from the same descriptor.
//! - **whether the two paths agreed**, as the side itself read them ([`BothPaths::agree`]).
//!
//! # Why the error's text is not compared
//!
//! Metal's refusal is an `NSError` whose domain, code and localized description are Apple's; ours
//! is a typed `metal2vulkan_runtime::Error`. Neither is a translation of the other and no
//! implementation could match the words, so what the case compares is **that the creation was
//! refused and how the refusal arrived**, not what it said. That is the same choice
//! [`crate::library_facts`] makes for a device object's identity: the portable question, stated
//! outright, rather than a field that would differ on every host.
//!
//! The error text is not silently dropped either. A case whose Metal leg *answered* where ours
//! refused is still a `Refused` verdict with our typed error printed, because that comparison runs
//! at the state's own sites, not here.
//!
//! # Why our leg takes the synchronous call for both paths
//!
//! `metal2vulkan-runtime` has no asynchronous creation form: the coverage map names the
//! synchronous call as each handler form's counterpart, and that is the whole of what we
//! implement. So our side takes that call twice, and the arrival site reads `Answered` on our leg
//! because a call that returned *is* a delivery. This is stated rather than hidden: what the case
//! holds to Metal is the delivered state's facts, the two bools, and that Metal's two paths agree
//! with each other. The absence of an asynchronous form of ours is a reported gap, not a
//! comparison.
//!
//! # Why nothing here can hang, and nothing holds the GPU
//!
//! Neither leg calls `waitUntilCompleted`, which Metal has no bounded form of, and neither waits
//! on a handler without a bound: the Metal leg waits on the handler's **own channel** for [`WAIT`]
//! and records a timeout as [`Arrival::NeverArrived`], which is a fact and not a retry. No case of
//! the family allocates a buffer or a texture, makes a command buffer, dispatches or draws: a
//! pipeline creation is a compile, which the project's 20-second and 500-MiB ceilings already
//! cover.

use crate::case::{MeshFunctions, TileFacts, TilePipelineSpec};
use crate::pipeline_facts::{ComputeStateFacts, ComputeStateSpec, RenderStateFacts};
use crate::render::RenderPipelineSpec;
use crate::transcript::{Negative, Node, NodeName, Observed, Value};

/// The longest the Metal leg waits on a creation handler's own channel.
///
/// A bound, not a measurement: an answer that did not arrive inside it is recorded as
/// [`Arrival::NeverArrived`], so no leg can leave the case's thread waiting for work that will not
/// come. Thirty seconds is the same bound [`crate::host_detach::WAIT`] uses, and it is well above
/// the project's own 20-second ceiling for a translation, so a handler that is merely slow is a
/// ceiling breach to diagnose rather than a timeout this site hides.
pub const WAIT: std::time::Duration = std::time::Duration::from_secs(30);

/// The imageblock extent every case of the family asks
/// `imageblockMemoryLengthForDimensions:` about, which is
/// [`crate::cases::pipeline_facts`]' own: three different axes, so a side that read the width
/// where the height belongs answers a different number.
pub const IMAGEBLOCK: [u32; 3] = [8, 4, 3];

/// Which of the seven calls a case makes, and the descriptor it makes it from.
///
/// The descriptor is the **same value** both paths are given, which is what makes the two
/// comparable: a case states one descriptor and the side makes it twice.
#[derive(Clone, Copy, Debug)]
pub enum Form<'a> {
    /// `newRenderPipelineStateWithDescriptor:completionHandler:`, or its
    /// `options:completionHandler:` form where `options` holds Metal's option bits.
    Render {
        spec: &'a RenderPipelineSpec<'a>,
        /// `None` takes the call with no options at all; `Some(bits)` takes the
        /// `options:completionHandler:` call with those raw `MTLPipelineOption` bits.
        options: Option<u64>,
    },
    /// `newComputePipelineStateWithFunction:completionHandler:`, or its
    /// `options:completionHandler:` form.
    ComputeFunction {
        entry: &'a str,
        /// As [`Form::Render`]'s.
        options: Option<u64>,
    },
    /// `newComputePipelineStateWithDescriptor:options:completionHandler:`, which has no
    /// option-less form.
    ComputeDescriptor {
        spec: &'a ComputeStateSpec<'a>,
        options: u64,
    },
    /// `newRenderPipelineStateWithTileDescriptor:options:completionHandler:`.
    Tile {
        spec: &'a TilePipelineSpec<'a>,
        options: u64,
    },
    /// `newRenderPipelineStateWithMeshDescriptor:options:completionHandler:`.
    Mesh {
        functions: &'a MeshFunctions<'a>,
        options: u64,
    },
}

impl Form<'_> {
    /// The handler call this form names, which is how a site and a report spell it.
    pub fn call(&self) -> &'static str {
        match self {
            Form::Render { options: None, .. } => {
                "newRenderPipelineStateWithDescriptor:completionHandler:"
            }
            Form::Render {
                options: Some(_), ..
            } => "newRenderPipelineStateWithDescriptor:options:completionHandler:",
            Form::ComputeFunction { options: None, .. } => {
                "newComputePipelineStateWithFunction:completionHandler:"
            }
            Form::ComputeFunction {
                options: Some(_), ..
            } => "newComputePipelineStateWithFunction:options:completionHandler:",
            Form::ComputeDescriptor { .. } => {
                "newComputePipelineStateWithDescriptor:options:completionHandler:"
            }
            Form::Tile { .. } => {
                "newRenderPipelineStateWithTileDescriptor:options:completionHandler:"
            }
            Form::Mesh { .. } => {
                "newRenderPipelineStateWithMeshDescriptor:options:completionHandler:"
            }
        }
    }

    /// The words a case's input fingerprint covers: which call, its options, and the whole
    /// descriptor it is made from.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.call().as_bytes().to_vec();
        let mut options = |bits: Option<u64>| match bits {
            None => out.extend(b"no-options"),
            Some(bits) => out.extend(bits.to_le_bytes()),
        };
        match self {
            Form::Render { spec, options: o } => {
                options(*o);
                out.extend(spec.words());
            }
            Form::ComputeFunction { entry, options: o } => {
                options(*o);
                out.extend(entry.as_bytes());
            }
            Form::ComputeDescriptor { spec, options: o } => {
                options(Some(*o));
                out.extend(spec.words());
            }
            Form::Tile { spec, options: o } => {
                options(Some(*o));
                out.extend(spec.words());
            }
            Form::Mesh {
                functions,
                options: o,
            } => {
                options(Some(*o));
                out.extend(functions.object.unwrap_or("").as_bytes());
                out.push(u8::from(functions.object.is_some()));
                out.extend(functions.mesh.as_bytes());
                out.extend(functions.fragment.as_bytes());
                out.extend(functions.attachments.to_le_bytes());
            }
        }
        out
    }
}

/// Whether the handler ran inside [`WAIT`].
///
/// Its own type rather than an `Option`, because the two outcomes are **facts of the case** that a
/// reader must be able to tell apart at a glance in a diff: `Answered` is a delivery and
/// `NeverArrived` is the recorded negative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arrival {
    /// The handler ran and the case read what it was given.
    Answered,
    /// The handler had not run when the bound ran out. Nothing was delivered.
    NeverArrived,
}

impl Arrival {
    /// The value the arrival site holds.
    pub fn value(self) -> Value {
        match self {
            Arrival::Answered => Value::unit_variant("Answered"),
            Arrival::NeverArrived => Value::Nothing(Negative::NeverReturned),
        }
    }
}

/// The facts the delivered state answers of itself, by the kind of state the descriptor makes.
///
/// Three shapes and not one: a compute state, a drawing state and a tile state answer different
/// getters, and [`crate::pipeline_facts`] already states which. A mesh state is an
/// `MTLRenderPipelineState`, so it answers [`StateFacts::Render`]'s getters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateFacts {
    Compute(ComputeStateFacts),
    Render(RenderStateFacts),
    Tile(TileFacts),
}

impl StateFacts {
    /// The facts as the node a site holds.
    fn node(&self, name: impl Into<NodeName>) -> Node {
        match self {
            StateFacts::Compute(facts) => facts.node(name),
            StateFacts::Render(facts) => facts.node(name),
            StateFacts::Tile(facts) => facts.node(name),
        }
    }
}

/// What one of the two paths delivered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Delivered {
    /// Whether a pipeline state came.
    pub state: bool,
    /// Whether an error came with it. Metal's block takes both, and a creation that failed
    /// delivers a nil state and a non-nil error; our synchronous call's `Err` is the same fact.
    pub error: bool,
    /// Whether a reflection came, for a form that has an options parameter; `None` for
    /// `newRenderPipelineStateWithDescriptor:completionHandler:` and
    /// `newComputePipelineStateWithFunction:completionHandler:`, which have none and so answer no
    /// reflection at all.
    ///
    /// A `Some(false)` is a fact rather than an absence: an options value with no reflection bit
    /// in it is a creation that must answer **no** reflection, and
    /// [`crate::cases::pipeline_handlers`] sweeps the bit both ways so that this site is seen to
    /// move.
    pub reflection: Option<bool>,
    /// What the delivered state answers of itself, or `None` where none came.
    pub facts: Option<StateFacts>,
}

impl Delivered {
    /// The delivery of a state whose facts are `facts`, by a form with no options parameter.
    pub fn made(facts: StateFacts) -> Self {
        Self {
            state: true,
            error: false,
            reflection: None,
            facts: Some(facts),
        }
    }

    /// The delivery of a state whose facts are `facts`, by a form whose options decided whether a
    /// reflection came.
    pub fn reflected(facts: StateFacts, reflection: bool) -> Self {
        Self {
            state: true,
            error: false,
            reflection: Some(reflection),
            facts: Some(facts),
        }
    }

    /// The delivery of a refusal: no state, an error, and no reflection where the form has one.
    pub fn refused(options: bool) -> Self {
        Self {
            state: false,
            error: true,
            reflection: options.then_some(false),
            facts: None,
        }
    }

    /// The sites this delivery holds, under one node.
    fn node(&self, name: impl Into<NodeName>) -> Node {
        let node = Node::branch(name, Vec::new())
            .with(Node::leaf(
                "delivered-a-state",
                Observed::value(Value::Bool(self.state)),
            ))
            .with(Node::leaf(
                "delivered-an-error",
                Observed::value(Value::Bool(self.error)),
            ))
            .with(Node::leaf(
                "delivered-a-reflection",
                Observed::value(match self.reflection {
                    Some(held) => Value::Bool(held),
                    None => Value::unit_variant("TheCallTakesNoOptions"),
                }),
            ));
        match &self.facts {
            Some(facts) => node.with(facts.node("facts")),
            None => node.with(Node::leaf(
                "facts",
                Observed::value(Value::unit_variant("NoState")),
            )),
        }
    }
}

/// The handler path's whole answer: whether it arrived, and what it delivered where it did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Handled {
    pub arrival: Arrival,
    /// What the handler was given; `None` exactly where [`Arrival::NeverArrived`].
    pub delivered: Option<Delivered>,
}

impl Handled {
    /// The handler's delivery of `delivered`.
    pub fn answered(delivered: Delivered) -> Self {
        Self {
            arrival: Arrival::Answered,
            delivered: Some(delivered),
        }
    }

    /// The handler that had not run when the bound ran out.
    pub fn never_arrived() -> Self {
        Self {
            arrival: Arrival::NeverArrived,
            delivered: None,
        }
    }
}

/// One descriptor taken through both paths on one side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BothPaths {
    /// The `completionHandler:` form.
    pub handler: Handled,
    /// The synchronous sibling, from the same descriptor.
    pub synchronous: Delivered,
    /// Whether the two paths delivered the same thing, as this side read them.
    ///
    /// Derived on the side rather than by the comparison, because it is a question **about one
    /// side**: "your asynchronous form and your synchronous form agree". A side with one path
    /// answers `true` by construction and says so in this module's prose; a Metal release where
    /// the two drifted answers `false` here, which is the one site that states that drift as a
    /// fact of its own rather than leaving a reader to compare two subtrees by eye.
    pub agree: bool,
}

impl BothPaths {
    /// The two paths' answers, with `agree` computed from them.
    ///
    /// Computed here and not by either side, so neither side can answer `true` while its two
    /// deliveries differ.
    pub fn new(handler: Handled, synchronous: Delivered) -> Self {
        let agree = handler.delivered.as_ref() == Some(&synchronous);
        Self {
            handler,
            synchronous,
            agree,
        }
    }

    /// The sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let handler = Node::branch("through-the-handler", Vec::new())
            .with(Node::leaf(
                "arrival",
                Observed::value(self.handler.arrival.value()),
            ))
            .with(match &self.handler.delivered {
                Some(delivered) => delivered.node("delivery"),
                None => Node::leaf(
                    "delivery",
                    Observed::value(Value::Nothing(Negative::NeverReturned)),
                ),
            });
        Node::branch(name, Vec::new())
            .with(handler)
            .with(self.synchronous.node("synchronously"))
            .with(Node::leaf(
                "the-two-paths-agree",
                Observed::value(Value::Bool(self.agree)),
            ))
    }
}
