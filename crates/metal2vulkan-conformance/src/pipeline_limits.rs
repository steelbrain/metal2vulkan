//! What a pipeline's **creation** answers for a descriptor's thread and grid limits: refused, or
//! made, and then what the made state's limit getters answer.
//!
//! Two descriptors: a mesh render pipeline's thread maxima, mesh-grid maximum, payload length and
//! vertex amplification count, beside its object function's own
//! `[[max_total_threadgroups_per_mesh_grid]]`; and a compute pipeline's
//! `maxTotalThreadsPerThreadgroup` beside its kernel's. The probes
//! `mesh_pipeline_limits.m` and `compute_pipeline_limits.m` measured both on an M3 Max; the cases
//! are their rows that answer, not the ones whose descriptor validation ends the process.
//!
//! # A refusal is an answer
//!
//! Each case asks what one descriptor does, and "Metal refuses it" is one of the answers, so a
//! refusal of Metal's own rule is an `Ok` of [`Creation::refused`] on both sides, for the reason
//! [`Side::tile_pipeline_bindings`](crate::case::Side::tile_pipeline_bindings) states. The
//! runtime's answer is refused where its typed error is Metal's rule
//! (`Compile(InvalidDescriptor(..))`, `Invalid(..)`); any other typed error is the runtime's own
//! refusal, an `Err` the comparison counts as one. Which rule refused is not compared: Metal's
//! error is a description, the runtime's a typed problem, and the translator's tests hold each
//! problem to the probe's descriptions.
//!
//! # The thread maxima are compared up to Vulkan's floor
//!
//! A made state's `maxTotalThreadsPerObjectThreadgroup`, `maxTotalThreadsPerMeshThreadgroup` and a
//! compute state's `maxTotalThreadsPerThreadgroup` are the least of Metal's rule (the
//! descriptor's maximum, else the function's, else 1024) and the device's limit for the stage, for
//! the reason [`crate::pipeline_facts`] gives for leaving the compute one out. Each is recorded as
//! the least of the answer and [`THREAD_FLOOR`], the fewest invocations of a workgroup every
//! Vulkan device takes (`maxComputeWorkGroupInvocations`, `maxTaskWorkGroupInvocations` and
//! `maxMeshWorkGroupInvocations` are each at least 128): below the floor the answer is Metal's rule
//! alone on every venue, so a stated maximum of 64 compares exactly and an answer of 1024 compares
//! as 128 whatever the device holds. A difference above the floor is not seen.
//!
//! # What a state without an object function does not read
//!
//! An M3 Max answers 3 for `maxTotalThreadsPerObjectThreadgroup` and 32 for
//! `objectThreadExecutionWidth` of a mesh state without an object function, and the runtime 0
//! for both (docs/API.md, accepted differences): such a state's two object-stage getters are not
//! read.

use crate::pipeline_facts::size;
use crate::transcript::{Node, NodeName, Observed, Value};

/// The fewest invocations of a workgroup every Vulkan device takes, in each stage a limit getter
/// answers for.
pub const THREAD_FLOOR: u64 = 128;

/// A mesh render pipeline a case makes: its functions, rasterization disabled and no fragment
/// function, and the descriptor's limit fields, each 0 for Metal's default. No case states a
/// required threadgroup size ([`crate::cases::pipeline_limits`] says why), so both are Metal's
/// zero size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshLimitsSpec<'a> {
    /// `objectFunction`'s entry, where one is set.
    pub object: Option<&'a str>,
    /// `meshFunction`'s entry.
    pub mesh: &'a str,
    /// `maxTotalThreadsPerObjectThreadgroup`.
    pub max_object: u32,
    /// `maxTotalThreadsPerMeshThreadgroup`.
    pub max_mesh: u32,
    /// `maxTotalThreadgroupsPerMeshGrid`.
    pub max_grid: u32,
    /// `payloadMemoryLength`.
    pub payload: u32,
    /// `maxVertexAmplificationCount`; Metal's default is 1.
    pub amplification: u32,
}

impl<'a> MeshLimitsSpec<'a> {
    /// The descriptor of `object` and `mesh` with Metal's default for every limit.
    pub const fn new(object: Option<&'a str>, mesh: &'a str) -> Self {
        Self {
            object,
            mesh,
            max_object: 0,
            max_mesh: 0,
            max_grid: 0,
            payload: 0,
            amplification: 1,
        }
    }

    /// The words a case's input fingerprint covers: every field.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.object.unwrap_or("").as_bytes().to_vec();
        out.push(0);
        out.extend(self.mesh.as_bytes());
        out.push(0);
        for word in [
            self.max_object,
            self.max_mesh,
            self.max_grid,
            self.payload,
            self.amplification,
        ] {
            out.extend(word.to_le_bytes());
        }
        out
    }
}

/// A compute pipeline a case makes: its kernel and the descriptor's
/// `maxTotalThreadsPerThreadgroup`, 0 for Metal's default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComputeLimitsSpec<'a> {
    /// `computeFunction`'s entry.
    pub entry: &'a str,
    /// `maxTotalThreadsPerThreadgroup`.
    pub max_total_threads: u32,
}

impl ComputeLimitsSpec<'_> {
    /// The words a case's input fingerprint covers: both fields.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.entry.as_bytes().to_vec();
        out.push(0);
        out.extend(self.max_total_threads.to_le_bytes());
        out
    }
}

/// What a creation answered: refused by Metal's rule, or made, with what the state then answers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Creation<T> {
    /// Whether the creation was refused.
    pub refused: bool,
    /// The made state's answers; `None` where it was refused.
    pub state: Option<T>,
}

impl<T> Creation<T> {
    /// The answer of a creation refused by Metal's rule.
    pub fn refused() -> Self {
        Self {
            refused: true,
            state: None,
        }
    }

    /// The answer of a made state.
    pub fn made(state: T) -> Self {
        Self {
            refused: false,
            state: Some(state),
        }
    }
}

/// What a made mesh render pipeline state answers of its limits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshStateLimits {
    /// `maxTotalThreadsPerObjectThreadgroup`, at most [`THREAD_FLOOR`]; `None` without an object
    /// function.
    pub max_total_threads_per_object_threadgroup: Option<u64>,
    /// `maxTotalThreadsPerMeshThreadgroup`, at most [`THREAD_FLOOR`].
    pub max_total_threads_per_mesh_threadgroup: u64,
    /// `objectThreadExecutionWidth`; `None` without an object function.
    pub object_thread_execution_width: Option<u64>,
    /// `meshThreadExecutionWidth`.
    pub mesh_thread_execution_width: u64,
    /// `maxTotalThreadgroupsPerMeshGrid`.
    pub max_total_threadgroups_per_mesh_grid: u64,
    /// `requiredThreadsPerObjectThreadgroup`.
    pub required_threads_per_object_threadgroup: [u64; 3],
    /// `requiredThreadsPerMeshThreadgroup`.
    pub required_threads_per_mesh_threadgroup: [u64; 3],
    /// `maxTotalThreadsPerThreadgroup`, which a drawing state answers 0 for.
    pub max_total_threads_per_threadgroup: u64,
}

/// What a made compute pipeline state answers of its limit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeStateLimits {
    /// `maxTotalThreadsPerThreadgroup`, at most [`THREAD_FLOOR`].
    pub max_total_threads_per_threadgroup: u64,
}

/// `answer` as a thread maximum is recorded: at most [`THREAD_FLOOR`].
pub fn floored(answer: u64) -> u64 {
    answer.min(THREAD_FLOOR)
}

/// An answer only some states give, as a site records it.
fn optional(held: Option<u64>) -> Value {
    match held {
        Some(held) => Value::Variant {
            name: "Some".to_owned(),
            payload: Box::new(Value::Uint(held)),
        },
        None => Value::unit_variant("None"),
    }
}

impl Creation<MeshStateLimits> {
    /// The answer as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let value = Observed::value;
        let mut node = Node::branch(name, Vec::new())
            .with(Node::leaf("refused", value(Value::Bool(self.refused))));
        if let Some(state) = &self.state {
            node = node
                .with(Node::leaf(
                    "max_total_threads_per_object_threadgroup",
                    value(optional(state.max_total_threads_per_object_threadgroup)),
                ))
                .with(Node::leaf(
                    "max_total_threads_per_mesh_threadgroup",
                    value(Value::Uint(state.max_total_threads_per_mesh_threadgroup)),
                ))
                .with(Node::leaf(
                    "object_thread_execution_width",
                    value(optional(state.object_thread_execution_width)),
                ))
                .with(Node::leaf(
                    "mesh_thread_execution_width",
                    value(Value::Uint(state.mesh_thread_execution_width)),
                ))
                .with(Node::leaf(
                    "max_total_threadgroups_per_mesh_grid",
                    value(Value::Uint(state.max_total_threadgroups_per_mesh_grid)),
                ))
                .with(Node::leaf(
                    "required_threads_per_object_threadgroup",
                    value(size(&state.required_threads_per_object_threadgroup)),
                ))
                .with(Node::leaf(
                    "required_threads_per_mesh_threadgroup",
                    value(size(&state.required_threads_per_mesh_threadgroup)),
                ))
                .with(Node::leaf(
                    "max_total_threads_per_threadgroup",
                    value(Value::Uint(state.max_total_threads_per_threadgroup)),
                ));
        }
        node
    }
}

impl Creation<ComputeStateLimits> {
    /// The answer as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let mut node = Node::branch(name, Vec::new()).with(Node::leaf(
            "refused",
            Observed::value(Value::Bool(self.refused)),
        ));
        if let Some(state) = &self.state {
            node = node.with(Node::leaf(
                "max_total_threads_per_threadgroup",
                Observed::value(Value::Uint(state.max_total_threads_per_threadgroup)),
            ));
        }
        node
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A maximum below the floor is recorded as it is, and one above it as the floor.
    #[test]
    fn a_thread_maximum_is_recorded_up_to_the_floor() {
        assert_eq!(floored(64), 64);
        assert_eq!(floored(128), 128);
        assert_eq!(floored(1024), 128);
    }
}
