//! What a **compute pipeline descriptor** answers of itself: `MTLComputePipelineDescriptor`'s own
//! surface, and what `reset` puts back.
//!
//! [`crate::render_descriptors`] is the same question of `MTLRenderPipelineDescriptor`, and the two
//! are built the same way for the same reason: a descriptor is where Metal's **defaults** live, and
//! a program that sets two fields of sixteen gets Metal's value for the other fourteen. This
//! family shares that module's [`Shape`] rather than declaring a second one.
//!
//! [`crate::pipeline_facts`] is the pipeline **state**'s side: what the made object echoes back.
//! Five of the descriptor's items were reached that way — `label`, `computeFunction`,
//! `stageInputDescriptor`, `supportIndirectCommandBuffers` and `requiredThreadsPerThreadgroup` —
//! and the rest of the surface was reached by nothing at all.
//!
//! The runtime has no descriptor object. Its
//! [`ComputePipelineDescriptor`](metal2vulkan::pipeline::ComputePipelineDescriptor) is a plain
//! struct a consumer fills, so its defaults are the ones
//! [`ComputePipelineDescriptor::new`](metal2vulkan::pipeline::ComputePipelineDescriptor::new)
//! writes: the call-stack depth of 1, the absent thread limit, the absent required threadgroup
//! size, the empty buffer-mutability map, the three flags off. Those were read off Apple's
//! documentation and nothing in the tree ran Metal to check one. `reset` is the same claim from
//! the other end: Metal's descriptor is mutable and `reset` puts every field back, while the
//! runtime's counterpart is `ComputePipelineDescriptor::new` again (the coverage map says so),
//! because a fresh struct **is** the reset.
//!
//! # What the site observes, and what it cannot
//!
//! **`stageInputDescriptor` is deliberately not here.** It is the one item of the surface already
//! compared against Metal (`indirect_commands` and `stage_in_regions` both state one), and no shape
//! of this family sets one, so a site for it could only ever hold the same `None` on both sides of
//! every case — a field that cannot differ, which is noise in a transcript rather than a
//! measurement. The whole of what a stated stage-input descriptor means belongs to
//! `MTLStageInputOutputDescriptor`'s own family.
//!
//! Four getters answer **objects a program supplies**, and the runtime takes each one somewhere
//! other than the descriptor:
//!
//! - `binaryArchives` is `PipelineOptions.binary_archives`;
//! - `preloadedLibraries` and its deprecated spelling `insertLibraries` are the kernel stage's
//!   `FunctionDescriptor.libraries`;
//! - `linkedFunctions` is the kernel stage's `FunctionDescriptor.linked`.
//!
//! For those the site records **how many** of each the list holds, never the objects, and a nil
//! list is none — the shape [`crate::render_descriptors`] gives its reasoning for. **This family
//! therefore compares those four items at their default alone**: it says the lists start empty and
//! that a reset empties them, and says nothing about a descriptor carrying a real archive or
//! dynamic library. That is a narrower case, not a loosened one; the wider case waits on cases for
//! `MTLBinaryArchive` and `MTLDynamicLibrary`, which no case of the suite reaches.
//!
//! `preloadedLibraries` and `insertLibraries` are read **both**, off a descriptor that set neither:
//! that the two are one property is a claim the runtime makes by holding a single
//! `FunctionDescriptor.libraries`, and reading both getters is what checks it, exactly as
//! [`crate::render_descriptors`] reads Metal's deprecated `sampleCount` beside `rasterSampleCount`.
//!
//! [`Shape::Reset`] leaves the three per-stage lists out, for the measured reason
//! [`SuppliedLists`] states.
//!
//! `shaderValidation` is absent: Apple's enumeration has no counterpart in the runtime, and a
//! getter whose answer nothing of ours can state is not a comparison.
//!
//! # Why every case of this family is safe
//!
//! Nothing here is compiled, allocated or submitted. A case makes one library, one function of it,
//! and a descriptor it never hands to `newComputePipelineStateWithDescriptor:`. No command buffer
//! is made, so no case can fault the device, trip its watchdog or reach a queue.
//!
//! The one hazard a descriptor's own surface carries is a **setter** Metal's validation stops, and
//! there is exactly one in reach: `setLabel:` with nil ends Metal's process (measured on this
//! family's oracle while recording [`crate::pipeline_facts`]). No shape here sets a nil label — the
//! label is set only where the shape states one — and the default shape reads the nil Metal's own
//! `init` left.

use crate::case::field;
use crate::render_descriptors::{optional, ordinal, LinkedCounts, Shape};
use crate::transcript::{Node, NodeName, Observed, Value};
use metal2vulkan::metal;

/// Every field the set shape moves away from Metal's default, each to a value no other field of
/// the record holds, so a side that wired one getter to its neighbour answers a different number
/// rather than passing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Away {
    pub label: &'static str,
    pub threadgroup_size_is_multiple_of_execution_width: bool,
    /// `maxTotalThreadsPerThreadgroup`; Metal's default is 0, the device's own maximum.
    pub max_total_threads_per_threadgroup: u32,
    /// `buffers[i].mutability`, by index.
    pub buffers: &'static [(u32, metal::Mutability)],
    pub support_indirect_command_buffers: bool,
    pub support_adding_binary_functions: bool,
    pub max_call_stack_depth: u32,
    /// `requiredThreadsPerThreadgroup`; Metal's default is the zero size, no requirement.
    pub required_threads_per_threadgroup: [u32; 3],
}

/// The one table of stated values, each away from Metal's default.
///
/// The numbers are chosen apart: a thread limit of 96, a call-stack depth of 3, and a required
/// threadgroup of 8 by 6 by 2 whose three axes differ. No two of them are equal, so a side that
/// answered one getter from another's field reads wrong, and a side that read the required
/// threadgroup's width where its height belongs reads 8 for 6.
///
/// The required threadgroup is 96 threads, which is the stated thread limit exactly: nothing of
/// this family compiles the descriptor, so neither number reaches a pipeline, and a table where
/// the requirement exceeded the limit would still have recorded whatever Metal's setters stored.
/// Choosing them consistent costs nothing and keeps the descriptor one a program could use.
pub const AWAY: Away = Away {
    label: "conformance-compute-descriptor",
    threadgroup_size_is_multiple_of_execution_width: true,
    max_total_threads_per_threadgroup: 96,
    buffers: &[
        (0, metal::Mutability::Immutable),
        (2, metal::Mutability::Mutable),
    ],
    support_indirect_command_buffers: true,
    support_adding_binary_functions: true,
    max_call_stack_depth: 3,
    required_threads_per_threadgroup: [8, 6, 2],
};

impl Away {
    /// The words a case's input fingerprint covers: every stated field, so a table that moves is a
    /// change to every case of the family.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.label.as_bytes().to_vec();
        for word in [
            self.max_total_threads_per_threadgroup,
            self.max_call_stack_depth,
            self.required_threads_per_threadgroup[0],
            self.required_threads_per_threadgroup[1],
            self.required_threads_per_threadgroup[2],
        ] {
            out.extend(word.to_le_bytes());
        }
        for flag in [
            self.threadgroup_size_is_multiple_of_execution_width,
            self.support_indirect_command_buffers,
            self.support_adding_binary_functions,
        ] {
            out.push(u8::from(flag));
        }
        for (index, mutability) in self.buffers {
            out.extend(index.to_le_bytes());
            out.extend(mutability.as_raw().to_le_bytes());
        }
        out
    }
}

/// What one case makes and reads: the kernel, and which of the three shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescriptorSpec<'a> {
    /// `computeFunction`'s entry in the case's library.
    pub kernel: &'a str,
    pub shape: Shape,
}

impl<'a> DescriptorSpec<'a> {
    /// The descriptor of `kernel`, in `shape`.
    pub const fn new(kernel: &'a str, shape: Shape) -> Self {
        Self { kernel, shape }
    }

    /// The words a case's input fingerprint covers: the kernel, the shape, and — where the shape
    /// sets anything — the whole stated table.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.kernel.as_bytes().to_vec();
        out.push(self.shape.word());
        if self.shape.sets() {
            out.extend(AWAY.words());
        }
        out
    }
}

/// The lists of objects a program supplies, by how many each holds.
///
/// [`Shape::Reset`] reads none of them, and the reason is Metal's own behaviour plus a third-party
/// defect, both measured on this family's oracle for the render descriptor first
/// ([`crate::render_descriptors::SuppliedLists`]): **`reset` leaves a preloaded-library list nil**
/// where `init` leaves a non-nil empty array, and `objc2-metal` 0.3.2 declares
/// `MTLComputePipelineDescriptor.preloadedLibraries` non-null, so its generated accessor panics on
/// the nil rather than answering it. Nothing of the runtime's answers that question anyway — it has
/// no descriptor object, so a program's own `Vec` is the counterpart and there is no reset to
/// observe — so the reset shape leaves the three sites out rather than reading a nil it cannot
/// record and our side cannot have.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SuppliedLists {
    /// `preloadedLibraries`' length.
    pub preloaded_libraries: u64,
    /// `insertLibraries`' length, Metal's **deprecated** spelling of the same property; a nil list
    /// is 0.
    pub insert_libraries: u64,
    /// `linkedFunctions`' four lengths; a nil object, or a list it answers nil for, is 0.
    pub linked_functions: LinkedCounts,
}

/// What a compute pipeline descriptor answers of itself. Each field is one Metal getter's answer,
/// an enumeration by its raw ordinal so an ordinal neither side names still compares.
#[derive(Clone, Debug, PartialEq)]
pub struct ComputeDescriptorFacts {
    /// `label`; `None` is Metal's nil.
    pub label: crate::case::Field<Option<String>>,
    /// `computeFunction`'s own `name`; `None` where the descriptor holds none.
    pub compute_function: Option<String>,
    pub threadgroup_size_is_multiple_of_execution_width: bool,
    pub max_total_threads_per_threadgroup: u64,
    /// `buffers[i].mutability` for `i` below [`crate::render_descriptors::READ_BUFFERS`].
    pub buffers: Vec<u32>,
    pub support_indirect_command_buffers: bool,
    /// `binaryArchives`' length; a nil list is 0.
    pub binary_archives: crate::case::Field<u64>,
    /// The three lists of supplied objects, which [`Shape::Reset`] does not read
    /// ([`SuppliedLists`] says why).
    pub supplied: Option<SuppliedLists>,
    pub support_adding_binary_functions: bool,
    pub max_call_stack_depth: u64,
    /// `requiredThreadsPerThreadgroup`, as three axes; the zero size where none is required.
    pub required_threads_per_threadgroup: [u64; 3],
}

impl ComputeDescriptorFacts {
    /// The facts as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let value = Observed::value;
        let mut node = Node::branch(name, Vec::new())
            .with(Node::leaf("label", field(&self.label, optional)))
            .with(Node::leaf(
                "compute_function",
                value(optional(&self.compute_function)),
            ))
            .with(Node::leaf(
                "threadgroup_size_is_multiple_of_execution_width",
                value(Value::Bool(
                    self.threadgroup_size_is_multiple_of_execution_width,
                )),
            ))
            .with(Node::leaf(
                "max_total_threads_per_threadgroup",
                value(Value::Uint(self.max_total_threads_per_threadgroup)),
            ))
            .with(Node::leaf(
                "buffers",
                value(Value::List(
                    self.buffers
                        .iter()
                        .map(|raw| ordinal(*raw, metal::Mutability::name))
                        .collect(),
                )),
            ))
            .with(Node::leaf(
                "support_indirect_command_buffers",
                value(Value::Bool(self.support_indirect_command_buffers)),
            ))
            .with(Node::leaf(
                "binary_archives",
                field(&self.binary_archives, |count| Value::Uint(*count)),
            ))
            .with(Node::leaf(
                "support_adding_binary_functions",
                value(Value::Bool(self.support_adding_binary_functions)),
            ))
            .with(Node::leaf(
                "max_call_stack_depth",
                value(Value::Uint(self.max_call_stack_depth)),
            ))
            .with(Node::leaf(
                "required_threads_per_threadgroup",
                value(crate::pipeline_facts::size(
                    &self.required_threads_per_threadgroup,
                )),
            ));
        if let Some(supplied) = &self.supplied {
            node = node
                .with(Node::leaf(
                    "preloaded_libraries",
                    value(Value::Uint(supplied.preloaded_libraries)),
                ))
                .with(Node::leaf(
                    "insert_libraries",
                    value(Value::Uint(supplied.insert_libraries)),
                ))
                .with(Node::leaf(
                    "linked_functions",
                    value(supplied.linked_functions.value()),
                ));
        }
        node
    }
}
