//! Visible-function tables: the handles a pipeline state gives, the tables it makes, and what a
//! shader reads through one.
//!
//! Four Metal containers meet here and the suite reached **none** of them:
//! `MTLFunctionHandle`, `MTLVisibleFunctionTable`, `MTLVisibleFunctionTableDescriptor`, and the
//! two selector pairs that join them to a pipeline state — `functionHandleWithFunction:` and
//! `newVisibleFunctionTableWithDescriptor:` on a compute state, and the `:stage:` spellings of
//! both on a render state. `MTLVisibleFunctionTable.h` and `MTLFunctionHandle.h` were two of the
//! four SDK headers with no exercised item at all.
//!
//! # Why a table is worth executing and not only reading back
//!
//! A table's getters say almost nothing: a count the case itself set, a label, a resource ID that
//! is the host's own number. A case that read those back would compare a round trip. What a table
//! *is* for is the call it makes, and the owned fixture `engine/visible_table_caller` makes that
//! call observable in four words:
//!
//! ```text
//! kernel visible_table_dispatch(device uint *out, visible_function_table<uint(uint,uint)> ops)
//!     r = 0xffffffff                     a null table (no table bound)
//!     r = 0xfffffffe                     a table of no entries
//!     r = 0xfffffffd                     a null entry
//!     r = ops[tid % size](tid, 3)        otherwise
//! ```
//!
//! So over four threads a table holding `[table_add, table_mul]` writes `3, 3, 5, 9` and the same
//! table holding `[table_mul, table_add]` writes `0, 4, 6, 6`. That is the family's control: the
//! words depend on **which function each entry holds**, not merely on a table having been bound,
//! and no amount of binding plumbing produces one vector from the other order.
//!
//! # The three safety bounds, which are measured and not guessed
//!
//! Two of these end a process and the third is recorded in the runtime's own documentation. No
//! case here may reach any of them:
//!
//! - **A render table of no stage faults.** `metal2vulkan_runtime`'s `function_table.rs` records
//!   `newVisibleFunctionTableWithDescriptor:stage:` with an empty `MTLRenderStages` faulting on an
//!   M4 under macOS 27.0.1. [`StageAsked`] therefore has no value for it and
//!   [`crate::cases::visible_tables`] names it unexercised by choice.
//! - **The device's maximum table size ends the process.** A table's allocation is one word an
//!   entry, so `functionCount` at the device's maximum asks for `max_buffer_length` bytes; Metal
//!   ended the process on an M4 from 2³¹ entries. The size sweep stops at 3, and the maximum is
//!   named unexercised by choice with that reason.
//! - **A handle of another pipeline state is Metal's validation failure**, and plain Metal's
//!   answer for it was never measured. Every handle a case here places comes from the state that
//!   made the table.
//!
//! # What our side folds, and why that is exact
//!
//! Metal answers **nil** from `functionHandleWithFunction:` for a function the state's stage does
//! not link, and from the `:stage:` pair for a `stage` that is more than one stage. The runtime
//! answers the first as `Ok(None)` and the second as `Err(Error::Nil)`, whose documented meaning
//! is "plain Metal answers nil here" — so [`Asked::of`] folds `Error::Nil` to the same `None`
//! Metal's nil records as, and records every other error as a refusal. The fold is the same one
//! [`crate::pipeline_functions`] applies to a nil array, for the same reason: a shape one side
//! cannot hold is not a difference to compare. A side that answered a handle where Metal answered
//! nil, or refused with any other error, still reads differently.
//!
//! # `gpuResourceID` is the host's number
//!
//! `MTLVisibleFunctionTable.gpuResourceID` answers an identifier of this process's own device, and
//! the runtime's documentation says Metal hands out the lowest free one of a space of its own from
//! 1 — a number that depends on what else the process has allocated and so belongs in no
//! recording. The call is still made, and the two derivations that *are* portable are recorded in
//! its place ([`TableFacts::resource_id_is_non_zero`], [`TableFacts::a_second_tables_id_differs`]),
//! which is how [`crate::pipeline_facts`] and the rest handle a `device` whose identity no
//! transcript can hold.

use crate::transcript::{Node, NodeName, Observed, Value};
use metal2vulkan::metal::{FunctionType, RenderStages};

/// Which pipeline state gives the handles and makes the table, and so which pair of selectors a
/// case reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Maker {
    /// A compute state: `functionHandleWithFunction:` and
    /// `newVisibleFunctionTableWithDescriptor:`.
    Compute,
    /// A drawing render state's stage, through `functionHandleWithFunction:stage:` and
    /// `newVisibleFunctionTableWithDescriptor:stage:`. The state's vertex function is the case's
    /// entry and it has no fragment function, so one stage of it links functions and the rest do
    /// not.
    DrawingStage(StageAsked),
    /// A tile render state's stage, through the same `:stage:` pair. Its tile function is the
    /// case's entry, so `MTLRenderStageTile` is the stage that links.
    TileStage(StageAsked),
}

impl Maker {
    /// The stage the case asks of a render state, or `None` for a compute state.
    pub fn stage(self) -> Option<StageAsked> {
        match self {
            Maker::Compute => None,
            Maker::DrawingStage(stage) | Maker::TileStage(stage) => Some(stage),
        }
    }

    /// The words a case's input fingerprint covers.
    pub fn words(&self) -> Vec<u8> {
        let (kind, stage) = match self {
            Maker::Compute => ("compute", None),
            Maker::DrawingStage(stage) => ("drawing", Some(stage)),
            Maker::TileStage(stage) => ("tile", Some(stage)),
        };
        let mut out = kind.as_bytes().to_vec();
        if let Some(stage) = stage {
            out.push(b'/');
            out.extend(stage.stages().as_raw().to_le_bytes());
        }
        out
    }
}

/// The `MTLRenderStages` a case passes to the `:stage:` pair.
///
/// Every value of the option set that a state can be asked about, and one shape of two stages at
/// once. There is deliberately **no value for an empty set**: a table of no stage faults on an M4
/// under macOS 27.0.1, which the runtime records, so the suite cannot ask it and says so rather
/// than carrying a value no case may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageAsked {
    /// `MTLRenderStageVertex`.
    Vertex,
    /// `MTLRenderStageFragment`.
    Fragment,
    /// `MTLRenderStageTile`.
    Tile,
    /// `MTLRenderStageObject`.
    Object,
    /// `MTLRenderStageMesh`.
    Mesh,
    /// `MTLRenderStageVertex | MTLRenderStageFragment`: more than one stage, for which Metal
    /// answers the handle and the table nil.
    VertexAndFragment,
}

impl StageAsked {
    /// The raw option set the selector takes.
    pub fn stages(self) -> RenderStages {
        match self {
            StageAsked::Vertex => RenderStages::VERTEX,
            StageAsked::Fragment => RenderStages::FRAGMENT,
            StageAsked::Tile => RenderStages::TILE,
            StageAsked::Object => RenderStages::OBJECT,
            StageAsked::Mesh => RenderStages::MESH,
            StageAsked::VertexAndFragment => RenderStages::VERTEX.union(RenderStages::FRAGMENT),
        }
    }
}

/// Which list of `MTLLinkedFunctions` a case puts a function in.
///
/// The two that decide whether a function has a handle at all: `functions` numbers the handles a
/// table can hold, and `privateFunctions` is reached by a visible-function reference alone, so a
/// function only there has **no** handle and Metal answers nil for it. `binaryFunctions` is left
/// out because Metal requires each of its functions to have been made with
/// `MTLFunctionOptionCompileToBinary`, which needs a binary archive — the reason
/// [`crate::pipeline_facts`] gives for leaving every binary-function item out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkedList {
    /// `MTLLinkedFunctions.functions`.
    Functions,
    /// `MTLLinkedFunctions.privateFunctions`.
    PrivateFunctions,
}

impl LinkedList {
    fn words(self) -> &'static [u8] {
        match self {
            LinkedList::Functions => b"functions",
            LinkedList::PrivateFunctions => b"private",
        }
    }
}

/// What a case links into the pipeline stage: each function of the functions library by name,
/// with the list it goes in, in the order the case states them.
///
/// The order is part of the question and not a detail: a handle's word is the function's position
/// among the stage's callable linked functions, so linking the same three functions in another
/// order is a different pipeline, and a side that numbered them by name, by library order or by
/// anything but the list's order reads different words out of the dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinkedSpec<'a> {
    /// Each function's name in the functions library and the list it is linked in.
    pub functions: &'a [(&'a str, LinkedList)],
}

impl<'a> LinkedSpec<'a> {
    /// The functions `functions` names, every one in `MTLLinkedFunctions.functions`.
    pub const fn callable(functions: &'a [(&'a str, LinkedList)]) -> Self {
        Self { functions }
    }

    /// The words a case's input fingerprint covers.
    pub fn words(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for (name, list) in self.functions {
            out.extend(name.as_bytes());
            out.push(b':');
            out.extend(list.words());
            out.push(b'|');
        }
        out
    }
}

/// One entry a case writes into the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entry<'a> {
    /// The handle of this function of the functions library.
    Function(&'a str),
    /// Metal's nil handle, which leaves the entry null.
    Nil,
}

impl Entry<'_> {
    fn words(&self) -> Vec<u8> {
        match self {
            Entry::Function(name) => name.as_bytes().to_vec(),
            Entry::Nil => b"nil".to_vec(),
        }
    }
}

/// Which setter writes the table's entries.
///
/// Both of `MTLVisibleFunctionTable`'s two, because they are two items and a side that implemented
/// the range form as "the first handle, repeated" or that ignored its `location` writes a table the
/// one-at-a-time form would not. The range form writes the whole of the case's entries in one
/// call, from index 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Writes {
    /// `setFunction:atIndex:`, once an entry.
    OneAtATime,
    /// `setFunctions:withRange:` over every entry at once.
    WholeRange,
}

impl Writes {
    fn words(self) -> &'static [u8] {
        match self {
            Writes::OneAtATime => b"set_function",
            Writes::WholeRange => b"set_functions",
        }
    }
}

/// Whether the encoder binds the table at the shader's table index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binding {
    /// `setVisibleFunctionTable:atBufferIndex:` of the case's table.
    Bound,
    /// Nothing is bound at the index, which is Metal's null table.
    Unbound,
}

/// What one case asks: the state to make, what to link into it, which handles to ask for, the
/// table to make and fill, and the grid to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableSpec<'a> {
    /// Which state gives the handles and makes the table.
    pub maker: Maker,
    /// The entry point of the caller library the state is made from.
    pub entry: &'a str,
    /// What the stage links.
    pub linked: LinkedSpec<'a>,
    /// The functions a case asks `functionHandleWithFunction:` of, in order. A name here need not
    /// be linked: a function the stage does not link is the shape Metal answers nil for.
    pub handles: &'a [&'a str],
    /// `MTLVisibleFunctionTableDescriptor.functionCount`.
    pub function_count: u32,
    /// The entries the case writes, entry `i` at index `i`. Shorter than `function_count` leaves
    /// the rest as the table was made, which is null.
    pub entries: &'a [Entry<'a>],
    /// Which setter writes them.
    pub writes: Writes,
    /// Whether a second table of the same descriptor is made, so the case can say whether two
    /// tables of one state answer different resource IDs.
    pub second_table: bool,
    /// Whether the encoder binds the table.
    pub binding: Binding,
    /// The threads the case runs, which is also the words it reads back. `None` runs nothing: the
    /// case reads the handles and the table alone, which is every case of a render stage, because
    /// a stage that links nothing has no dispatch to make.
    pub threads: Option<u32>,
}

impl TableSpec<'_> {
    /// The words a case's input fingerprint covers: every field, so a spec that moves is a change
    /// to the case's inputs and a recording of the old one is no answer for the new.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.maker.words();
        out.push(b'/');
        out.extend(self.entry.as_bytes());
        out.push(b'/');
        out.extend(self.linked.words());
        for name in self.handles {
            out.extend(name.as_bytes());
            out.push(b',');
        }
        out.extend(self.function_count.to_le_bytes());
        for entry in self.entries {
            out.push(b'|');
            out.extend(entry.words());
        }
        out.extend(self.writes.words());
        out.push(u8::from(self.second_table));
        out.push(u8::from(matches!(self.binding, Binding::Bound)));
        out.extend(self.threads.unwrap_or(0).to_le_bytes());
        out.push(u8::from(self.threads.is_some()));
        out
    }

    /// Whether the case runs a dispatch.
    pub fn runs(&self) -> bool {
        self.threads.is_some()
    }
}

/// What one `MTLFunctionHandle` answers: the whole of its portable surface, which is two getters.
///
/// **`functionType` is not here**, although `objc2-metal` binds it: the runtime's
/// `FunctionHandle` has no such getter, so the coverage map marks the item unimplemented, and a
/// site only Metal's leg could fill is a site that compares nothing. The function's **own**
/// `functionType` is compared instead, at its own site beside the handle
/// ([`TableOutcome::function_types`]), which is the implemented item and answers
/// `MTLFunctionTypeVisible` for every `[[visible]]` function. Likewise `gpuResourceID`, which
/// Metal documents as meaningful only for an intersection function.
#[derive(Clone, Debug, PartialEq)]
pub struct HandleFacts {
    /// `name`, which is the library function's own.
    pub name: String,
    /// Whether `device` is the device the case's other objects were made on. A device object is
    /// not a value a transcript can hold, so this is the whole of what the getter promises.
    pub device_is_the_cases: bool,
}

/// What one asked-for handle answered: Metal's object, Metal's nil, or a refusal.
#[derive(Clone, Debug, PartialEq)]
pub enum Asked {
    /// A handle, and what it answers.
    Handle(HandleFacts),
    /// Metal's nil: a function the stage does not link, or a `stage` of more than one stage.
    Nil,
    /// The call was refused. Metal refuses neither shape a case here asks, so this is a
    /// divergence wherever it appears on one side alone.
    Refused(crate::case::Failure),
}

/// An enumeration as a site records it: its Metal enumerator name beside its ordinal, the shape
/// [`crate::function_facts`] and [`crate::function_bindings`] record one in, so a diff reads
/// `MTLFunctionTypeVisible` rather than 5 and the depth census reads the value out of the
/// recording by its own name.
fn named(name: &'static str, raw: u32) -> Value {
    Value::fields([
        ("name", Value::Lines(vec![name.to_owned()])),
        ("raw", Value::Uint(u64::from(raw))),
    ])
}

/// The fold this module's header states, in the one place both answers go through:
/// `Error::Nil`'s documented meaning in the runtime is "plain Metal answers nil (no object, no
/// error), and the process goes on", so it records as the nil Metal's own answer records as. Any
/// other error is a refusal, which reads divergent against Metal's object or nil.
///
/// `metal_nil` is what the folded answer is: a side's own nil for a call it made and that
/// answered nothing.
fn folded<T, N: Fn() -> T, H: Fn(F) -> T, F>(
    answer: Result<Option<F>, metal2vulkan_runtime::Error>,
    metal_nil: N,
    held: H,
    refused: impl Fn(crate::case::Failure) -> T,
) -> T {
    match answer {
        Ok(Some(facts)) => held(facts),
        Ok(None) => metal_nil(),
        Err(metal2vulkan_runtime::Error::Nil(_)) => metal_nil(),
        Err(error) => refused(crate::case::Failure::of(&error)),
    }
}

impl Asked {
    /// The answer of a side whose call gives `Option` and whose `Error::Nil` is Metal's nil
    /// (the `folded` helper below is the one place the rule lives).
    pub fn of(answer: Result<Option<HandleFacts>, metal2vulkan_runtime::Error>) -> Self {
        folded(answer, || Asked::Nil, Asked::Handle, Asked::Refused)
    }

    /// The answer of a side whose call answers Metal's nil-able object and no error, which is
    /// Metal's own leg.
    pub fn answered(facts: Option<HandleFacts>) -> Self {
        facts.map_or(Asked::Nil, Asked::Handle)
    }

    /// The facts as one site, so both sides emit one shape whichever way the call answered.
    fn node(&self, name: impl Into<NodeName>) -> Node {
        match self {
            Asked::Handle(facts) => Node::branch(name, Vec::new())
                .with(Node::leaf(
                    "name",
                    Observed::value(Value::Lines(vec![facts.name.clone()])),
                ))
                .with(Node::leaf(
                    "device_is_the_cases",
                    Observed::value(Value::Bool(facts.device_is_the_cases)),
                )),
            Asked::Nil => Node::leaf(name, Observed::value(Value::unit_variant("Nil"))),
            Asked::Refused(failure) => Node::leaf(name, Observed::value(failure.recorded())),
        }
    }
}

/// What one `MTLVisibleFunctionTable` answers.
///
/// **There is no `functionCount` here**, and that is Metal's shape and not a choice: the
/// `MTLVisibleFunctionTable` protocol declares `gpuResourceID`, `setFunction:atIndex:` and
/// `setFunctions:withRange:` and no count at all. The count lives on the **descriptor**, where
/// this family reads it back ([`TableOutcome::descriptor_function_count`]); the runtime's table
/// answers no count either, so there is nothing to compare a site against.
#[derive(Clone, Debug, PartialEq)]
pub struct TableFacts {
    /// Whether `gpuResourceID` is non-zero. The identifier itself is the host's own number
    /// (this module's header says why), and this is one of its two portable derivations.
    pub resource_id_is_non_zero: bool,
    /// Whether a second table the same state made from the same descriptor answers a different
    /// `gpuResourceID`; `None` where the case made no second table. The other portable
    /// derivation: two live tables are two objects, so a side that answered one identifier for
    /// both — or a constant — reads differently.
    pub a_second_tables_id_differs: Option<bool>,
    /// Whether `device` is the device the case's other objects were made on.
    pub device_is_the_cases: bool,
}

/// What the table creation answered: a table, Metal's nil, or a refusal.
#[derive(Clone, Debug, PartialEq)]
pub enum Made {
    Table(TableFacts),
    /// Metal's nil, which the `:stage:` creation answers for more than one stage.
    Nil,
    Refused(crate::case::Failure),
}

impl Made {
    /// The answer of a side whose creation gives a `Result` and whose `Error::Nil` is Metal's nil,
    /// folded as [`Asked::of`] folds a handle's, through the same `folded` helper.
    pub fn of(answer: Result<Option<TableFacts>, metal2vulkan_runtime::Error>) -> Self {
        folded(answer, || Made::Nil, Made::Table, Made::Refused)
    }

    /// The answer of Metal's own leg, whose creation answers a nil-able object and no error.
    pub fn answered(facts: Option<TableFacts>) -> Self {
        facts.map_or(Made::Nil, Made::Table)
    }

    fn node(&self, name: impl Into<NodeName>) -> Node {
        match self {
            Made::Table(facts) => {
                let node = Node::branch(name, Vec::new())
                    .with(Node::leaf(
                        "resource_id_is_non_zero",
                        Observed::value(Value::Bool(facts.resource_id_is_non_zero)),
                    ))
                    .with(Node::leaf(
                        "device_is_the_cases",
                        Observed::value(Value::Bool(facts.device_is_the_cases)),
                    ));
                match facts.a_second_tables_id_differs {
                    Some(differs) => node.with(Node::leaf(
                        "a_second_tables_id_differs",
                        Observed::value(Value::Bool(differs)),
                    )),
                    None => node,
                }
            }
            Made::Nil => Node::leaf(name, Observed::value(Value::unit_variant("Nil"))),
            Made::Refused(failure) => Node::leaf(name, Observed::value(failure.recorded())),
        }
    }
}

/// What one case observed: each handle it asked for, the table it made, and the words the
/// dispatch wrote through that table.
#[derive(Clone, Debug, PartialEq)]
pub struct TableOutcome {
    /// One answer per name of [`TableSpec::handles`], in that order.
    pub handles: Vec<Asked>,
    /// `MTLVisibleFunctionTableDescriptor.functionCount`, read back off the descriptor after the
    /// case set it and before the creation took it.
    ///
    /// A round trip, and it is the whole of what the item offers — the descriptor has one
    /// property and the table it makes has no count at all — so the family states it as one
    /// rather than reading as more, the same way [`crate::pipeline_functions`] states a
    /// descriptor whose only claim is a round trip. What it still catches is a side that clamped
    /// the count, that lost a count of 0, or whose descriptor's default was not 0.
    pub descriptor_function_count: u64,
    /// `MTLFunction.functionType` of each name of [`TableSpec::handles`], in that order, read off
    /// the function the handle was asked of.
    ///
    /// It is here and not on [`HandleFacts`] for the reason that type gives, and it is read for
    /// every asked function and not only the ones with a handle: a function with no handle still
    /// has a type, and this family is the suite's only route to `MTLFunctionTypeVisible` — no
    /// other case makes an `MTLFunction` of a `[[visible]]` function at all.
    pub function_types: Vec<FunctionType>,
    /// What the table creation answered.
    pub table: Made,
    /// Whether every entry the case asked for was written. A side that refused a write says so
    /// here, and the words below then say what the table it did write calls.
    pub entries_written: Result<(), crate::case::Failure>,
    /// The words the dispatch wrote, one a thread; empty where the case runs nothing.
    pub words: Vec<u32>,
    /// The command buffer's status, or [`Value::Absent`] where the case runs nothing.
    pub status: Value,
}

impl TableOutcome {
    /// The outcome of a case that only read the handles and the table.
    pub fn read_only(
        handles: Vec<Asked>,
        descriptor_function_count: u64,
        function_types: Vec<FunctionType>,
        table: Made,
    ) -> Self {
        Self {
            handles,
            descriptor_function_count,
            function_types,
            table,
            entries_written: Ok(()),
            words: Vec::new(),
            status: Value::Absent,
        }
    }

    /// The facts as the sites a case emits under one node, so both sides emit one shape.
    ///
    /// The handles are named by the function they were asked of rather than by position, so a
    /// difference names the function; the names a case asks are distinct, which
    /// [`crate::cases::visible_tables`] states, so no two sites of one name can arise here.
    pub fn node(&self, name: impl Into<NodeName>, asked: &[&str]) -> Node {
        let mut node = Node::branch(name, Vec::new());
        for (held, of) in self.handles.iter().zip(asked) {
            node = node.with(held.node(format!("handle/{of}")));
        }
        for (held, of) in self.function_types.iter().zip(asked) {
            // The enumerator's own name beside its ordinal, so the value compares as a name and
            // the depth census reads `MTLFunctionTypeVisible` out of the recording.
            node = node.with(Node::leaf(
                format!("function_type/{of}"),
                Observed::value(named(held.name(), held.as_raw())),
            ));
        }
        node = node.with(Node::leaf(
            "descriptor_function_count",
            Observed::value(Value::Uint(self.descriptor_function_count)),
        ));
        node = node.with(self.table.node("table"));
        node = node.with(Node::leaf(
            "entries_written",
            Observed::value(match &self.entries_written {
                Ok(()) => Value::ok(Value::Absent),
                Err(failure) => failure.recorded(),
            }),
        ));
        node = node.with(Node::leaf(
            "words",
            Observed::value(Value::List(
                self.words
                    .iter()
                    .map(|word| Value::Uint(u64::from(*word)))
                    .collect(),
            )),
        ));
        node.with(Node::leaf("status", Observed::value(self.status.clone())))
    }
}
