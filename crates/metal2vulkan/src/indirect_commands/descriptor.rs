//! A command buffer's descriptor: what it allows, the header it initializes, and the checks a
//! decoded record must pass (the host's side of the format, with `host`).

use super::host::{get_word, CommandBufferError};
use super::{
    put_word, Action, BufferStage, CommandRecord, DrawKind, Inheritance, StateSource,
    BINDING_COUNT, COMMAND_BUFFER_HEADER_BYTES, COMPUTE, DRAW, DRAW_INDEXED, DRAW_INDEXED_PATCHES,
    DRAW_MESH_THREADGROUPS, DRAW_MESH_THREADS, DRAW_PATCHES, DYNAMIC_STRIDE, GROUPS,
    INHERIT_BUFFERS, INHERIT_PIPELINE, RENDER, THREADS,
};

/// What a command buffer allows, fixed when it is made (Metal's
/// `MTLIndirectCommandBufferDescriptor`, as far as the record format needs it): a compute buffer's
/// dispatch kinds, kernel-buffer and threadgroup-memory binding counts and dynamic strides, or a
/// render buffer's draw kinds, vertex, fragment, object and mesh binding counts and dynamic
/// strides. It writes
/// the header's policy ([`encode_header`](Self::encode_header)) and checks each record a producer
/// wrote against it ([`validate_record`](Self::validate_record)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommandDescriptor {
    Compute {
        dispatch: ComputeDispatch,
        max_buffer_bind_count: u32,
        max_threadgroup_memory_bind_count: u32,
        support_dynamic_attribute_stride: bool,
    },
    Render {
        draws: RenderDraws,
        max_vertex_buffer_bind_count: u32,
        max_fragment_buffer_bind_count: u32,
        max_object_buffer_bind_count: u32,
        max_mesh_buffer_bind_count: u32,
        support_dynamic_attribute_stride: bool,
    },
}

/// The draws a render descriptor allows: Metal's `Draw` (draws of primitives), `DrawIndexed`,
/// `DrawPatches`, `DrawIndexedPatches`, `DrawMeshThreadgroups` and `DrawMeshThreads`, at least
/// one of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderDraws {
    pub primitives: bool,
    pub indexed: bool,
    pub patches: bool,
    pub indexed_patches: bool,
    pub mesh_threadgroups: bool,
    pub mesh_threads: bool,
}

impl RenderDraws {
    /// The header's policy bits of the draws.
    fn bits(self) -> u32 {
        [
            (self.primitives, DRAW),
            (self.indexed, DRAW_INDEXED),
            (self.patches, DRAW_PATCHES),
            (self.indexed_patches, DRAW_INDEXED_PATCHES),
            (self.mesh_threadgroups, DRAW_MESH_THREADGROUPS),
            (self.mesh_threads, DRAW_MESH_THREADS),
        ]
        .into_iter()
        .filter(|&(allowed, _)| allowed)
        .fold(0, |bits, (_, bit)| bits | bit)
    }

    /// Whether the records of a buffer of these draws hold `action`, a draw or none: a draw of a
    /// kind they allow, or of a kind whose record a kind they allow holds ([`DrawKind::holders`]).
    fn allow(self, action: &Action) -> bool {
        let kind = match action {
            Action::Empty => return true,
            Action::DrawPrimitives { .. } => DrawKind::Primitives,
            Action::DrawIndexedPrimitives { .. } => DrawKind::Indexed,
            Action::DrawPatches { .. } | Action::DrawIndexedPatches { .. } => DrawKind::Patches,
            Action::DrawMeshThreadgroups { .. } | Action::DrawMeshThreads { .. } => DrawKind::Mesh,
            Action::DispatchThreadgroups { .. } | Action::DispatchThreads { .. } => return false,
        };
        self.bits() & kind.holders() != 0
    }
}

/// The dispatches a compute descriptor allows: Metal's `ConcurrentDispatch` (threadgroups),
/// `ConcurrentDispatchThreads` (threads), or both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeDispatch {
    Threadgroups,
    Threads,
    Both,
}

impl CommandDescriptor {
    pub fn validate(&self) -> Result<(), String> {
        let counts = match *self {
            Self::Compute {
                max_buffer_bind_count,
                max_threadgroup_memory_bind_count,
                ..
            } => {
                if max_threadgroup_memory_bind_count > super::THREADGROUP_BUFFER_ARGUMENT_COUNT {
                    return Err(
                        "indirect threadgroup-memory binding count exceeds the record ABI".into(),
                    );
                }
                [max_buffer_bind_count, 0, 0, 0]
            }
            Self::Render {
                draws,
                max_vertex_buffer_bind_count,
                max_fragment_buffer_bind_count,
                max_object_buffer_bind_count,
                max_mesh_buffer_bind_count,
                ..
            } => {
                if draws.bits() == 0 {
                    return Err("a render command buffer descriptor allows no draw".into());
                }
                [
                    max_vertex_buffer_bind_count,
                    max_fragment_buffer_bind_count,
                    max_object_buffer_bind_count,
                    max_mesh_buffer_bind_count,
                ]
            }
        };
        if counts.into_iter().any(|count| count > BINDING_COUNT as u32) {
            return Err("indirect buffer binding count exceeds the record ABI".into());
        }
        Ok(())
    }

    /// Whether the records of a buffer of this descriptor hold `action`: a compute buffer's a
    /// dispatch of either kind, and a render buffer's a draw of a kind it allows or of a kind
    /// whose record a kind it allows holds, as Metal's records do, which it sizes by the largest
    /// kind a buffer allows (our probes of a kernel's commands and of the CPU's setters).
    pub fn holds(&self, action: &Action) -> bool {
        match *self {
            Self::Compute { .. } => match action {
                Action::Empty
                | Action::DispatchThreadgroups { .. }
                | Action::DispatchThreads { .. } => true,
                Action::DrawPrimitives { .. }
                | Action::DrawIndexedPrimitives { .. }
                | Action::DrawPatches { .. }
                | Action::DrawIndexedPatches { .. }
                | Action::DrawMeshThreadgroups { .. }
                | Action::DrawMeshThreads { .. } => false,
            },
            Self::Render { draws, .. } => draws.allow(action),
        }
    }

    /// Check command-owned state. Parent bindings are governed by the parent encoder contract.
    pub fn validate_record(&self, record: &CommandRecord) -> Result<(), String> {
        self.validate()?;
        // Each stage of the record's family, with its count (`BufferStage::of`).
        let (counts, threadgroup_count, dynamic_stride) = match *self {
            Self::Compute {
                max_buffer_bind_count,
                max_threadgroup_memory_bind_count,
                support_dynamic_attribute_stride,
                ..
            } => (
                [max_buffer_bind_count, 0, 0, 0],
                max_threadgroup_memory_bind_count,
                support_dynamic_attribute_stride,
            ),
            // A mesh draw's object threadgroup lengths, at any index the record holds: Metal
            // keeps one at or past the descriptor's count.
            Self::Render {
                draws,
                max_vertex_buffer_bind_count,
                max_fragment_buffer_bind_count,
                max_object_buffer_bind_count,
                max_mesh_buffer_bind_count,
                support_dynamic_attribute_stride,
            } => (
                [
                    max_mesh_buffer_bind_count,
                    max_vertex_buffer_bind_count,
                    max_fragment_buffer_bind_count,
                    max_object_buffer_bind_count,
                ],
                match draws.mesh_threadgroups || draws.mesh_threads {
                    true => super::THREADGROUP_COUNT as u32,
                    false => 0,
                },
                support_dynamic_attribute_stride,
            ),
        };
        if !self.holds(&record.action) {
            return Err("indirect action is not enabled by its descriptor".into());
        }
        let render = matches!(self, Self::Render { .. });
        for (stage, count) in BufferStage::of(render).into_iter().zip(counts) {
            for (index, binding) in record.bindings.buffers(stage).iter().enumerate() {
                let Some(binding) = binding else {
                    continue;
                };
                if index >= count as usize {
                    return Err(format!(
                        "indirect {stage:?} binding {index} exceeds its descriptor"
                    ));
                }
                if binding.attribute_stride.is_some()
                    && (!dynamic_stride
                        || !matches!(stage, BufferStage::Kernel | BufferStage::Vertex))
                {
                    return Err("indirect dynamic attribute stride is not enabled".into());
                }
            }
        }
        if record.threadgroup_memory_lengths[threadgroup_count as usize..]
            .iter()
            .any(|&length| length != 0)
        {
            return Err("indirect threadgroup-memory binding exceeds its descriptor".into());
        }
        Ok(())
    }
}

impl CommandDescriptor {
    /// Encode capacity, a clear fault flag, and immutable descriptor policy without allocating
    /// replay records. Keep this header intact while producer commands can access it.
    pub fn encode_header(
        &self,
        capacity: u32,
        inheritance: Inheritance,
    ) -> Result<[u8; COMMAND_BUFFER_HEADER_BYTES], CommandBufferError> {
        self.validate().map_err(CommandBufferError::Descriptor)?;
        let (mut flags, counts, dynamic) = match *self {
            Self::Compute {
                dispatch,
                max_buffer_bind_count,
                max_threadgroup_memory_bind_count,
                support_dynamic_attribute_stride,
            } => (
                COMPUTE
                    | match dispatch {
                        ComputeDispatch::Threadgroups => GROUPS,
                        ComputeDispatch::Threads => THREADS,
                        ComputeDispatch::Both => GROUPS | THREADS,
                    },
                max_buffer_bind_count | (max_threadgroup_memory_bind_count << 24),
                support_dynamic_attribute_stride,
            ),
            // A render descriptor's object and mesh counts take the kernel and threadgroup
            // counts' bytes (`layout::COUNTS`).
            Self::Render {
                draws,
                max_vertex_buffer_bind_count,
                max_fragment_buffer_bind_count,
                max_object_buffer_bind_count,
                max_mesh_buffer_bind_count,
                support_dynamic_attribute_stride,
            } => (
                RENDER | draws.bits(),
                max_object_buffer_bind_count
                    | (max_vertex_buffer_bind_count << 8)
                    | (max_fragment_buffer_bind_count << 16)
                    | (max_mesh_buffer_bind_count << 24),
                support_dynamic_attribute_stride,
            ),
        };
        if dynamic {
            flags |= DYNAMIC_STRIDE;
        }
        if inheritance.pipeline == StateSource::Parent {
            flags |= INHERIT_PIPELINE;
        }
        if inheritance.buffers == StateSource::Parent {
            flags |= INHERIT_BUFFERS;
        }
        let mut header = [0; COMMAND_BUFFER_HEADER_BYTES];
        for (word, value) in [capacity, 0, flags, counts].into_iter().enumerate() {
            put_word(&mut header, word, value);
        }
        Ok(header)
    }

    /// Check a completed producer's header against its original capacity and policy, including
    /// the sticky encoding fault. Record contents and native resource state are separate checks.
    pub fn validate_header(
        &self,
        bytes: &[u8],
        capacity: u32,
        inheritance: Inheritance,
    ) -> Result<(), CommandBufferError> {
        let expected = self.encode_header(capacity, inheritance)?;
        if bytes.len() != expected.len() {
            return Err(CommandBufferError::Length);
        }
        if bytes[..4] != expected[..4] || bytes[8..] != expected[8..] {
            return Err(CommandBufferError::HeaderPolicyMismatch);
        }
        match get_word(bytes, 1) {
            0 => Ok(()),
            faults => Err(CommandBufferError::EncodingFault(faults)),
        }
    }
}
