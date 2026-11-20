//! What a pipeline **state** says about itself: `MTLComputePipelineState`'s and
//! `MTLRenderPipelineState`'s own getters, and the descriptor fields each one echoes back.
//!
//! A pipeline state is immutable, so everything it answers was decided by its descriptor or by the
//! function it compiled. That makes each getter one of two kinds of question, and the two are not
//! worth the same:
//!
//! - **a round trip** — `label` and `supportIndirectCommandBuffers` come straight from the
//!   descriptor, `requiredThreadsPerThreadgroup` from a compute descriptor's, and comparing them
//!   is comparing that we kept what the program handed us;
//! - **a fact about the compiled thing** — `staticThreadgroupMemoryLength`,
//!   `imageblockSampleLength` and `imageblockMemoryLengthForDimensions:` are computed from the
//!   function and the attachment formats, and our side computes them itself, by rules measured on
//!   an Apple M3 Max and an M4. Those are the ones with something to get wrong.
//!
//! # What is left out, and why
//!
//! **A compute state's `maxTotalThreadsPerThreadgroup` and `threadExecutionWidth` are not here.**
//! They answer the **device's** limits for the kernel, which is the reason
//! [`Side::static_threadgroup_memory_length`] already states at its own site: a venue that answers
//! less is not diverging from Metal, it is a venue with a smaller device, and a transcript has no
//! way to say that of a number. Capping them from the descriptor does not fix it — the answer is
//! still the least of several limits, one of them the device's — so the two stay out and are a
//! named hole rather than a comparison that would fail honestly on a smaller venue.
//!
//! A **render** state's `maxTotalThreadsPerThreadgroup` is a different question and is here: a
//! drawing state answers 0, which is a constant and not a limit.
//!
//! **`gpuResourceID` is not here.** A resource ID is each host's own number, for exactly the
//! reason [`crate::argument_encoders`] gives for every word an encoder writes, and a recording
//! holds nothing host-specific.
//!
//! [`Side::static_threadgroup_memory_length`]: crate::case::Side::static_threadgroup_memory_length

use crate::case::{field, Field};
use crate::transcript::{Node, NodeName, Observed, Value};

/// A compute pipeline state a case makes from a whole descriptor:
/// `MTLComputePipelineDescriptor`'s fields whose answer the state itself gives back.
///
/// Only those fields. A descriptor field nothing observes would be a call the coverage map counts
/// and no case checks, which is worth nothing to a reader; the fields that are here are each read
/// back at the matching getter of [`ComputeStateFacts`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComputeStateSpec<'a> {
    /// The kernel's name in the library.
    pub entry: &'a str,
    /// `label`; `None` is Metal's nil, which the state answers as nil.
    pub label: Option<&'a str>,
    /// `supportIndirectCommandBuffers`.
    pub support_indirect_command_buffers: bool,
    /// `requiredThreadsPerThreadgroup`; `None` is Metal's zero size, no requirement.
    pub required_threads_per_threadgroup: Option<[u32; 3]>,
}

impl<'a> ComputeStateSpec<'a> {
    /// The kernel `entry` with Metal's defaults for every other field.
    pub const fn new(entry: &'a str) -> Self {
        Self {
            entry,
            label: None,
            support_indirect_command_buffers: false,
            required_threads_per_threadgroup: None,
        }
    }

    /// The descriptor with `label`.
    #[must_use]
    pub const fn labelled(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    /// The descriptor with `supportIndirectCommandBuffers` set.
    #[must_use]
    pub const fn for_indirect_commands(mut self) -> Self {
        self.support_indirect_command_buffers = true;
        self
    }

    /// The descriptor requiring every dispatch to have the threadgroup size `threads`.
    #[must_use]
    pub const fn requiring(mut self, threads: [u32; 3]) -> Self {
        self.required_threads_per_threadgroup = Some(threads);
        self
    }

    /// The words a case's input fingerprint covers: every field, so a descriptor that moves is a
    /// change to the case's inputs.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.entry.as_bytes().to_vec();
        out.extend(self.label.unwrap_or("").as_bytes());
        out.push(u8::from(self.label.is_some()));
        out.push(u8::from(self.support_indirect_command_buffers));
        for axis in self.required_threads_per_threadgroup.unwrap_or([0, 0, 0]) {
            out.extend(axis.to_le_bytes());
        }
        out
    }
}

/// What a compute pipeline state answers of itself. Each field is a Metal getter's answer, and a
/// [`Field`] one a side may refuse alone, the state's other getters still compared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeStateFacts {
    /// `label`, which the state echoes from its descriptor; `None` is nil.
    pub label: Field<Option<String>>,
    /// Whether `device` is the device the case's other objects were made on. The device object
    /// itself is not comparable, so this is the one portable question the getter answers.
    pub device_is_the_cases: Field<bool>,
    /// `staticThreadgroupMemoryLength`: the bytes the kernel's own `threadgroup` arrays take.
    pub static_threadgroup_memory_length: u64,
    /// `supportIndirectCommandBuffers`, echoed from the descriptor.
    pub support_indirect_command_buffers: bool,
    /// `imageblockMemoryLengthForDimensions:` of the dimensions the case asked about.
    pub imageblock_memory_length: Field<u64>,
    /// `requiredThreadsPerThreadgroup`, as three axes; a zero size where the descriptor requires
    /// none.
    pub required_threads_per_threadgroup: [u64; 3],
}

/// What a render pipeline state answers of itself. Each field is a Metal getter's answer, and a
/// [`Field`] one a side may refuse alone, the state's other getters still compared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderStateFacts {
    /// `label`, which the state echoes from its descriptor; `None` is nil.
    pub label: Field<Option<String>>,
    /// Whether `device` is the device the case's other objects were made on.
    pub device_is_the_cases: Field<bool>,
    /// `maxTotalThreadsPerThreadgroup`, a tile threadgroup's: 0 for a drawing state, which is a
    /// constant and not the device's limit, so it compares exactly on every venue.
    pub max_total_threads_per_threadgroup: Field<u64>,
    /// `threadgroupSizeMatchesTileSize`: false for a drawing state.
    pub threadgroup_size_matches_tile_size: Field<bool>,
    /// `imageblockSampleLength`: the color attachments' texels packed. `None` where the side
    /// answered a typed error for it rather than a length.
    pub imageblock_sample_length: Option<u64>,
    /// `imageblockMemoryLengthForDimensions:` of the dimensions the case asked about, which for a
    /// drawing state is the sample length times the dimensions and the raster sample count.
    pub imageblock_memory_length: Option<u64>,
    /// `supportIndirectCommandBuffers`, echoed from the descriptor.
    pub support_indirect_command_buffers: bool,
    /// `requiredThreadsPerTileThreadgroup`, as three axes; a zero size for a drawing state.
    pub required_threads_per_tile_threadgroup: Field<[u64; 3]>,
}

/// `label` as a site records it.
fn label(held: &Option<String>) -> Value {
    match held {
        Some(held) => Value::Variant {
            name: "Some".to_owned(),
            payload: Box::new(Value::Lines(vec![held.clone()])),
        },
        None => Value::unit_variant("None"),
    }
}

/// A `MTLSize` as a site records it, in the shape [`crate::case::RateMapFacts`] records one.
///
/// Shared with the three descriptor families whose `requiredThreadsPerThreadgroup` is an
/// `MTLSize` ([`crate::compute_descriptors`], [`crate::tile_descriptors`]), so one Metal size reads
/// the same wherever a site records one.
pub(crate) fn size(axes: &[u64; 3]) -> Value {
    Value::fields([
        ("width", Value::Uint(axes[0])),
        ("height", Value::Uint(axes[1])),
        ("depth", Value::Uint(axes[2])),
    ])
}

/// A `u64` a side may have answered a typed error for, as a site records it.
fn optional(held: Option<u64>) -> Value {
    match held {
        Some(held) => Value::Variant {
            name: "Some".to_owned(),
            payload: Box::new(Value::Uint(held)),
        },
        None => Value::unit_variant("None"),
    }
}

impl ComputeStateFacts {
    /// The facts as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        Node::branch(name, Vec::new())
            .with(Node::leaf("label", field(&self.label, label)))
            .with(Node::leaf(
                "device",
                field(&self.device_is_the_cases, |held| Value::Bool(*held)),
            ))
            .with(Node::leaf(
                "static_threadgroup_memory_length",
                Observed::value(Value::Uint(self.static_threadgroup_memory_length)),
            ))
            .with(Node::leaf(
                "support_indirect_command_buffers",
                Observed::value(Value::Bool(self.support_indirect_command_buffers)),
            ))
            .with(Node::leaf(
                "imageblock_memory_length",
                field(&self.imageblock_memory_length, |length| {
                    Value::Uint(*length)
                }),
            ))
            .with(Node::leaf(
                "required_threads_per_threadgroup",
                Observed::value(size(&self.required_threads_per_threadgroup)),
            ))
    }
}

impl RenderStateFacts {
    /// The facts as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        Node::branch(name, Vec::new())
            .with(Node::leaf("label", field(&self.label, label)))
            .with(Node::leaf(
                "device",
                field(&self.device_is_the_cases, |held| Value::Bool(*held)),
            ))
            .with(Node::leaf(
                "max_total_threads_per_threadgroup",
                field(&self.max_total_threads_per_threadgroup, |threads| {
                    Value::Uint(*threads)
                }),
            ))
            .with(Node::leaf(
                "threadgroup_size_matches_tile_size",
                field(&self.threadgroup_size_matches_tile_size, |matches| {
                    Value::Bool(*matches)
                }),
            ))
            .with(Node::leaf(
                "imageblock_sample_length",
                Observed::value(optional(self.imageblock_sample_length)),
            ))
            .with(Node::leaf(
                "imageblock_memory_length",
                Observed::value(optional(self.imageblock_memory_length)),
            ))
            .with(Node::leaf(
                "support_indirect_command_buffers",
                Observed::value(Value::Bool(self.support_indirect_command_buffers)),
            ))
            .with(Node::leaf(
                "required_threads_per_tile_threadgroup",
                field(&self.required_threads_per_tile_threadgroup, size),
            ))
    }
}
