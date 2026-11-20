//! Visible-function tables, swept: the handles a pipeline state gives, the table it makes, the
//! entries a case writes into it, and what a shader calls through it.
//!
//! # What the suite measured before this
//!
//! **Nothing.** `MTLVisibleFunctionTable.h` and `MTLFunctionHandle.h` were two of the four SDK
//! headers with no exercised item at all, and the two selector pairs that reach them — a compute
//! state's `functionHandleWithFunction:` and `newVisibleFunctionTableWithDescriptor:`, and the
//! `:stage:` spellings a render state has — were unexercised beside them. The only visible-table
//! items any case reached were `MTLArgumentEncoder`'s two, which write a table's *resource ID*
//! into an argument buffer without ever making one. So every claim the runtime makes about a
//! handle, a table, an entry or a table call rested on prose and on its own interpreter.
//!
//! # Why this family executes rather than reads back
//!
//! A table's getters are a count the case itself set, a label, and a resource identifier that is
//! the host's own number. A family built on those would compare a round trip. What a table is
//! *for* is the call, and the owned fixture `engine/visible_table_caller` makes the call a vector
//! of words ([`crate::visible_tables`] prints its four answers). Over twelve threads a table
//! holding `[table_add, table_mul]` writes `3, 3, 5, 9, 7, 15, …` and the **same table holding
//! `[table_mul, table_add]`** writes `0, 4, 6, 6, 12, 8, …`.
//!
//! That pair is the family's control and it is what makes the whole sweep non-vacuous: the words
//! depend on *which function each entry holds*, so a side that bound a table correctly but
//! numbered its handles by name, by library order, or by anything but the order of
//! `MTLLinkedFunctions.functions` reads one vector where the other belongs. No amount of binding
//! plumbing turns one into the other.
//!
//! # Which values move the observation, and which cannot
//!
//! - **`functionCount` 0, 1, 2, 3 each move the words**, and by construction: the shader calls
//!   entry `tid % size`, so the size decides which thread calls what, and a size of 0 takes the
//!   empty-table branch instead. Four of four sizes a case may ask.
//! - **The device's maximum `functionCount` is unexercised by choice.** A table's allocation is
//!   one word an entry over a buffer of the device's `max_buffer_length`, and Metal **ended the
//!   process** on an M4 from 2³¹ entries (recorded in the runtime's `function_table.rs`). The
//!   sweep stops at 3 rather than discovering that again.
//! - **A table of no entries and a table whose entries are all nil are different rows**, and
//!   Metal distinguishes them: the shader's `size()` guard writes `0xfffffffe` for the first and
//!   its null-entry guard writes `0xfffffffd` for the second. Two rows, two answers, not one row
//!   asked twice.
//! - **An unbound index moves it again**, to `0xffffffff`: Metal's null table, which is what an
//!   index nothing was bound at reads as.
//! - **Both entry setters are swept and are *not* indistinguishable by construction.**
//!   `setFunctions:withRange:` writes the same two entries `setFunction:atIndex:` writes, so the
//!   two rows are expected to agree — and a side that implemented the range form as "the first
//!   handle, repeated", that ignored its `location`, or that wrote the range backwards reads a
//!   different vector at exactly that row.
//! - **A `functionCount` of 3 with two entries written** leaves the third as the creation left it.
//!   Expected to agree with nothing else in the sweep: its words hold `0xfffffffd` where the
//!   nil-entry row holds one and the size-3 row holds `table_sub`'s answer.
//! - **`MTLLinkedFunctions.privateFunctions` against `functions` moves the handle.** The private
//!   row links all four functions and puts `table_sub` alone in `privateFunctions`, so against the
//!   size-3 row — which links the same four and gets `table_sub`'s handle — it differs at exactly
//!   that handle. That is the whole claim about the list: a function reachable by a
//!   visible-function reference but holding no table word.
//! - **A private function and an unlinked one are two rows, not one.** The private row keeps
//!   `table_half` callable and the unlinked row links neither it nor `table_sub`, so the two
//!   differ at `handle/table_half`. Written the obvious way — the private row linking only the two
//!   it needed — the two transcripts came out **byte-identical** and one of them compared nothing;
//!   the recordings said so, and the rows were separated for that reason.
//! - **Neither row moves the words.** `table_add` and `table_mul` hold the same positions among
//!   the callable functions in all three, so the dispatch reads the base two-entry vector and the
//!   handles are the only sites that move. That is the point: the two rows are about the handle
//!   selector, and a row whose words also moved could not say which call the difference came
//!   from.
//! - **The `MTLRenderStages` sweep moves the handle and not the words.** Of the five single
//!   stages, `MTLRenderStageVertex` is the one the case's state has a function of, so it alone
//!   answers a handle; `MTLRenderStageFragment`, `MTLRenderStageTile`, `MTLRenderStageObject` and
//!   `MTLRenderStageMesh` answer no handle and still **make** a table, which is the runtime's
//!   claim under test. `MTLRenderStageVertex | MTLRenderStageFragment` answers both nil. Five of
//!   five single values and one compound; the empty set is unexercised by choice because plain
//!   Metal **faults** on a table of no stage (an M4 under macOS 27.0.1, recorded in the runtime).
//!
//! # What is deliberately not a case
//!
//! **An entry holding a function of another type.** `table_half` is `float(float)` and the table
//! is `uint(uint, uint)`; Metal's own documentation makes a call through the wrong type
//! undefined, so a case comparing it would be comparing undefined behaviour on two
//! implementations and would pin whatever this machine happened to do. The *handle* of
//! `table_half` is asked for and compared — that is defined — and no case places it.
//!
//! **A handle of another pipeline state in another state's table.** Metal's validation stops it
//! and plain Metal's answer was never measured, so every handle every case here places comes from
//! the state that made the table.
//!
//! **A dispatch through a render stage's table.** The two `:stage:` selectors are what this family
//! reaches, and it reaches them by making the state, taking the handle and making the table. The
//! vertex and tile *callers* exist as committed fixtures and executing them is a round of its own.
//!
//! # Why this family is safe
//!
//! Each case makes two libraries, one pipeline state, at most two tables of at most three
//! entries, and the compute cases encode one dispatch of twelve single-thread threadgroups over a
//! 48-byte buffer. Nothing is sampled, no texture is made, no attachment is written and no case
//! waits on anything but its own command buffer. The three shapes with a recorded way to end a
//! process — a table of no stage, a table near the device's maximum, and a handle of another
//! state — are each named above and reached by no case.
//!
//! # Exactness
//!
//! Every field is a number, a name, an enumerator or a bool. The family declares no tolerance and
//! announces nothing: no float, no texture object and no sampler is touched.

use crate::case::{Abort, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{Digest, Node, Observed, Sites};
use crate::visible_tables::{
    Binding, Entry, LinkedList, LinkedSpec, Maker, StageAsked, TableSpec, Writes,
};

/// The site the caller library's make sits at.
const CALLER_LIBRARY: &str = "caller_library";

/// The site the functions library's make sits at.
const FUNCTIONS_LIBRARY: &str = "functions_library";

/// The site the case's whole answer sits at.
const OUTCOME: &str = "outcome";

/// The library that defines the visible functions a stage links and a table holds.
const FUNCTIONS: &str = "visible_table_functions";

/// The caller whose kernel calls entry `tid % size` of its table.
const COMPUTE_CALLER: &str = "visible_table_caller";
const COMPUTE_ENTRY: &str = "visible_table_dispatch";

/// The caller whose **vertex** function calls through a table, which is the state the
/// `:stage:` pair is read off.
const VERTEX_CALLER: &str = "vft_vertex_caller";
const VERTEX_ENTRY: &str = "vft_vertex_dispatch";

/// The threads a compute case runs, which is also the words it reads back.
///
/// Twelve rather than four: six divides it, so each entry of a table of 1, 2 or 3 is called more
/// than once and a vector says which entry answered where rather than only that the first did. It
/// is also what keeps `table_sub`'s wrap readable — entry 2 of the size-3 table answers
/// `2 - 3`, whose bit pattern is the null-table sentinel's, and at twelve threads the same entry
/// also answers `5 - 3` and `8 - 3` and `11 - 3` as plain small numbers, so no word of a vector
/// has to be read as a sentinel.
const THREADS: u32 = 12;

/// Every function of the functions library of the table's type, then the one of another type, in
/// the order the cases link them.
///
/// A handle's word is the function's position among the stage's callable linked functions, so this
/// order is part of every case's question.
const EVERY: [(&str, LinkedList); 4] = [
    ("table_add", LinkedList::Functions),
    ("table_mul", LinkedList::Functions),
    ("table_sub", LinkedList::Functions),
    ("table_half", LinkedList::Functions),
];

/// `table_add` and `table_mul` alone, for the cases about a function with no handle.
const TWO: [(&str, LinkedList); 2] = [
    ("table_add", LinkedList::Functions),
    ("table_mul", LinkedList::Functions),
];

/// Every function of the library again, but with `table_sub` in `privateFunctions`.
///
/// Three of the four stay in `functions`, so this differs from [`EVERY`] at exactly one function
/// and from [`TWO`] at exactly one other: against `EVERY` it says what `privateFunctions` costs a
/// function (its handle), and against `TWO` it says that a **private** function and an
/// **unlinked** one are not the same row. Were `table_half` left out here too, this row and the
/// unlinked one would have byte-identical transcripts and one of them would compare nothing —
/// which is how they were first written, and what the recordings showed.
///
/// `table_add` and `table_mul` keep their positions among the callable functions either way, so
/// the dispatch's words are the base two-entry row's and the only sites that move are the two
/// handles.
const EVERY_WITH_A_PRIVATE: [(&str, LinkedList); 4] = [
    ("table_add", LinkedList::Functions),
    ("table_mul", LinkedList::Functions),
    ("table_sub", LinkedList::PrivateFunctions),
    ("table_half", LinkedList::Functions),
];

/// The handles every compute case asks for: each function of the library, which is also what
/// says a `[[visible]]` function's handle answers `MTLFunctionTypeVisible`.
const ALL_FOUR: [&str; 4] = ["table_add", "table_mul", "table_sub", "table_half"];

/// The handles the render-stage cases ask for.
const JUST_TWO: [&str; 2] = ["table_add", "table_mul"];

/// The entries the base two-entry table holds, and the same two swapped, which is the family's
/// control.
const ADD_MUL: [Entry<'static>; 2] = [Entry::Function("table_add"), Entry::Function("table_mul")];
const MUL_ADD: [Entry<'static>; 2] = [Entry::Function("table_mul"), Entry::Function("table_add")];

/// A compute case's spec with the fields every one of them shares.
const fn compute(function_count: u32, entries: &'static [Entry<'static>]) -> TableSpec<'static> {
    TableSpec {
        maker: Maker::Compute,
        entry: COMPUTE_ENTRY,
        linked: LinkedSpec::callable(&EVERY),
        handles: &ALL_FOUR,
        function_count,
        entries,
        writes: Writes::OneAtATime,
        second_table: false,
        binding: Binding::Bound,
        threads: Some(THREADS),
    }
}

/// A render-stage case's spec: the drawing state `VERTEX_CALLER`'s vertex function makes, asked
/// about `stage`, with the entries `entries`.
const fn stage(stage: StageAsked, entries: &'static [Entry<'static>]) -> TableSpec<'static> {
    TableSpec {
        maker: Maker::DrawingStage(stage),
        entry: VERTEX_ENTRY,
        linked: LinkedSpec::callable(&TWO),
        handles: &JUST_TWO,
        function_count: 2,
        entries,
        writes: Writes::OneAtATime,
        second_table: false,
        binding: Binding::Unbound,
        threads: None,
    }
}

/// Every case of the family, each named after what it sweeps.
const CASES: [(&str, TableSpec<'static>); 18] = [
    // The size sweep: every `functionCount` a case may ask, each of which moves the words.
    ("visible-tables-compute-size-0", compute(0, &[])),
    (
        "visible-tables-compute-size-1",
        compute(1, &[Entry::Function("table_add")]),
    ),
    (
        "visible-tables-compute-size-2",
        TableSpec {
            second_table: true,
            ..compute(2, &ADD_MUL)
        },
    ),
    // The control: the same table, the same two handles, the other order.
    (
        "visible-tables-compute-size-2-swapped",
        compute(2, &MUL_ADD),
    ),
    (
        "visible-tables-compute-size-3",
        compute(
            3,
            &[
                Entry::Function("table_add"),
                Entry::Function("table_mul"),
                Entry::Function("table_sub"),
            ],
        ),
    ),
    // The entry contents: every entry nil, one entry nil, and a size past what was written.
    (
        "visible-tables-compute-every-entry-nil",
        compute(2, &[Entry::Nil, Entry::Nil]),
    ),
    (
        "visible-tables-compute-one-entry-nil",
        compute(2, &[Entry::Function("table_add"), Entry::Nil]),
    ),
    (
        "visible-tables-compute-count-3-two-entries-written",
        compute(3, &ADD_MUL),
    ),
    // The other setter, over the same two entries.
    (
        "visible-tables-compute-whole-range",
        TableSpec {
            writes: Writes::WholeRange,
            ..compute(2, &ADD_MUL)
        },
    ),
    // No table bound at the shader's index, which is Metal's null table.
    (
        "visible-tables-compute-table-unbound",
        TableSpec {
            binding: Binding::Unbound,
            ..compute(2, &ADD_MUL)
        },
    ),
    // The two ways a function has no handle.
    (
        "visible-tables-compute-a-private-function-has-no-handle",
        TableSpec {
            linked: LinkedSpec::callable(&EVERY_WITH_A_PRIVATE),
            ..compute(2, &ADD_MUL)
        },
    ),
    (
        "visible-tables-compute-an-unlinked-function-has-no-handle",
        TableSpec {
            linked: LinkedSpec::callable(&TWO),
            ..compute(2, &ADD_MUL)
        },
    ),
    // The `:stage:` pair, over every single `MTLRenderStages` value and one compound.
    (
        "visible-tables-stage-vertex",
        TableSpec {
            second_table: true,
            ..stage(StageAsked::Vertex, &ADD_MUL)
        },
    ),
    (
        "visible-tables-stage-fragment",
        stage(StageAsked::Fragment, &[]),
    ),
    ("visible-tables-stage-tile", stage(StageAsked::Tile, &[])),
    (
        "visible-tables-stage-object",
        stage(StageAsked::Object, &[]),
    ),
    ("visible-tables-stage-mesh", stage(StageAsked::Mesh, &[])),
    (
        "visible-tables-stage-vertex-and-fragment",
        stage(StageAsked::VertexAndFragment, &[]),
    ),
];

/// A case over one state, one table and one dispatch.
pub struct VisibleTablesCase {
    name: &'static str,
    spec: TableSpec<'static>,
}

/// Every case of the family.
pub fn all() -> Vec<VisibleTablesCase> {
    CASES
        .into_iter()
        .map(|(name, spec)| VisibleTablesCase { name, spec })
        .collect()
}

impl VisibleTablesCase {
    /// The caller fixture the case's state is made from.
    fn caller(&self) -> Fixture {
        Fixture {
            name: match self.spec.maker {
                Maker::Compute => COMPUTE_CALLER,
                Maker::DrawingStage(_) | Maker::TileStage(_) => VERTEX_CALLER,
            },
            entry: self.spec.entry,
        }
    }

    /// The library of visible functions, whose own kernel is its entry point.
    fn functions(&self) -> Fixture {
        Fixture {
            name: FUNCTIONS,
            entry: FUNCTIONS,
        }
    }
}

impl Case for VisibleTablesCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for fixture in [self.caller(), self.functions()] {
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                let text = fixture.text(extension)?;
                shaders.push(ShaderRef {
                    fixture: format!("{}{extension}", fixture.name),
                    entry: fixture.entry.to_owned(),
                    source_sha256: crate::sha256_hex(text.as_bytes()),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = Vec::new();
        for (fixture, at) in [(self.caller(), "caller"), (self.functions(), "functions")] {
            for (extension, name) in [(RUNTIME_TEXT, "air"), (METAL_TEXT, "air-typed")] {
                inputs.push(Input {
                    name: format!("{at}-{name}"),
                    digest: Digest::sha256(fixture.text(extension)?.as_bytes()),
                });
            }
        }
        // The whole spec: a different state, a different link order, a different table or a
        // different grid is a different question, and a recording of one is no answer for another.
        inputs.push(Input::bytes("table", &self.spec.words()));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let caller = self
            .caller()
            .loaded()
            .map_err(|why| stop(CALLER_LIBRARY, why))?;
        let functions = self
            .functions()
            .loaded()
            .map_err(|why| stop(FUNCTIONS_LIBRARY, why))?;
        let caller = crate::case::made(sites, CALLER_LIBRARY, side.library(&caller.source()))?;
        let functions =
            crate::case::made(sites, FUNCTIONS_LIBRARY, side.library(&functions.source()))?;
        match side.visible_function_table(caller, functions, &self.spec) {
            Ok(outcome) => {
                sites.push(outcome.node(OUTCOME, self.spec.handles));
                Ok(())
            }
            Err(failure) => {
                sites.push(Node::leaf(OUTCOME, Observed::value(failure.recorded())));
                Err(Abort {
                    at: OUTCOME.to_owned(),
                    failure,
                })
            }
        }
    }
}
