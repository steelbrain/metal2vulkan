//! What a guarded native setter checks before it encodes: each command-encoder operation's
//! signature as a typed helper and the descriptor policy its write requires of the record header,
//! emitted as LLVM text. Copied from the product's host-record format with the guard; the policy
//! bits are the header bits `CommandDescriptor::encode_header` writes into the table
//! ([`super::NativeCommandPolicies`]), so the two cannot be read apart.

use crate::air::calls::{BufferStage, CommandEncoderOperation};

// The header's policy bits (`metal2vulkan::bind::indirect_commands`' descriptor encoding).
const COMPUTE: u32 = 1;
const RENDER: u32 = 2;
const DRAW: u32 = 16;
const DYNAMIC_STRIDE: u32 = 32;
const INHERIT_PIPELINE: u32 = 64;
const INHERIT_BUFFERS: u32 = 128;
const DRAW_INDEXED: u32 = 256;
const DRAW_PATCHES: u32 = 512;
const DRAW_INDEXED_PATCHES: u32 = 1024;
const DRAW_MESH_THREADGROUPS: u32 = 2048;
const DRAW_MESH_THREADS: u32 = 4096;

/// The descriptor policy an encoder operation's write requires of the record header: its
/// policy bits masked with `required | forbidden` equal `required`, and hold one of `any` where
/// `any` is not 0, the command index is below the capacity, a binding index is below its count
/// (byte `binding` of the header's packed counts), a dynamic stride is all ones unless the
/// descriptor enables dynamic strides, and a threadgroup length is a multiple of 16. A write
/// that fails any of them sets the fault flag instead and writes nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Policy {
    pub(super) required: u32,
    pub(super) forbidden: u32,
    /// The draw kinds of which the descriptor must allow one (0: no such check): those whose
    /// Metal record holds the operation's draw, as the product's lowering checks them
    /// (`indirect_commands::DrawKind::holders`).
    pub(super) any: u32,
    pub(super) binding: Option<u8>,
    pub(super) dynamic_stride: bool,
    pub(super) length: bool,
}

impl CommandEncoderOperation {
    pub(super) fn signature(self, name: &str) -> Option<String> {
        let tail = match self {
            Self::UnsupportedBufferEncoding | Self::Unmodeled => return None,
            Self::Buffer {
                stage,
                address_space,
            } => {
                let stride = if matches!(stage, BufferStage::Kernel | BufferStage::Vertex) {
                    ", i64 %stride"
                } else {
                    ""
                };
                format!(", ptr addrspace({address_space}) %buffer{stride}, i32 %binding")
            }
            Self::Dispatch { .. } => ", <3 x i32> %grid, <3 x i32> %local".to_string(),
            Self::DrawMesh { .. } => {
                ", <3 x i32> %grid, <3 x i32> %object_threads, <3 x i32> %mesh_threads".to_string()
            }
            Self::Pipeline { .. } => ", ptr addrspace(1) %pipeline".to_string(),
            Self::Barrier { .. } | Self::Reset { .. } => String::new(),
            Self::ThreadgroupMemory => ", i32 %length, i32 %binding".to_string(),
            Self::StageInRegion => ", <3 x i32> %origin, <3 x i32> %size".to_string(),
            Self::ImageblockSize => ", <2 x i16> %size".to_string(),
            Self::Copy => ", ptr addrspace(1) %source, i32 %source_command".to_string(),
            Self::Draw => {
                ", i32 %primitive, i32 %start, i32 %count, i32 %instances, i32 %base_instance"
                    .to_string()
            }
            Self::DrawIndexed { address_space, .. } => format!(
                ", i32 %primitive, i32 %count, ptr addrspace({address_space}) %indices, \
                 i32 %instances, i32 %base_vertex, i32 %base_instance"
            ),
            Self::DrawPatches {
                patch_indices,
                factors,
            } => format!(
                ", i32 %control_points, i32 %patch_start, i32 %patch_count, \
                 ptr addrspace({patch_indices}) %patch_indices, i32 %instances, \
                 i32 %base_instance, ptr addrspace({factors}) %factors, i32 %instance_stride, \
                 float %scale"
            ),
            Self::DrawIndexedPatches {
                patch_indices,
                control_point_indices,
                factors,
            } => format!(
                ", i32 %control_points, i32 %patch_start, i32 %patch_count, \
                 ptr addrspace({patch_indices}) %patch_indices, \
                 ptr addrspace({control_point_indices}) %control_point_indices, i32 %instances, \
                 i32 %base_instance, ptr addrspace({factors}) %factors, i32 %instance_stride, \
                 float %scale"
            ),
        };
        Some(format!(
            "define internal void @{name}(ptr addrspace(1) %records, i32 %command{tail})"
        ))
    }

    /// The descriptor policy this operation's write requires of the header ([`Policy`]); `None`
    /// for an unsupported buffer encoding.
    pub(super) fn policy(self) -> Option<Policy> {
        let family = |render: bool| if render { RENDER } else { COMPUTE };
        let plain = |required| Policy {
            required,
            forbidden: 0,
            any: 0,
            binding: None,
            dynamic_stride: false,
            length: false,
        };
        // A draw needs a render descriptor whose records hold its kind: primitives every render
        // buffer's, indexed draws an indexed or a patch buffer's, patch and indexed patch draws a
        // patch buffer's, mesh draws a buffer of either mesh kind's. A draw of primitives on a
        // buffer of mesh draws alone writes too: Metal draws nothing for it.
        let patch_records = DRAW_PATCHES | DRAW_INDEXED_PATCHES;
        let mesh_records = DRAW_MESH_THREADGROUPS | DRAW_MESH_THREADS;
        let draw = |any| Policy {
            any,
            ..plain(RENDER)
        };
        Some(match self {
            Self::Buffer { stage, .. } => Policy {
                required: family(stage != BufferStage::Kernel),
                forbidden: INHERIT_BUFFERS,
                any: 0,
                // A render descriptor's object and mesh counts take the bytes of the kernel and
                // threadgroup counts, which a compute descriptor's alone hold.
                binding: Some(match stage {
                    BufferStage::Kernel | BufferStage::Object => 0,
                    BufferStage::Vertex => 1,
                    BufferStage::Fragment => 2,
                    BufferStage::Mesh => 3,
                }),
                dynamic_stride: matches!(stage, BufferStage::Kernel | BufferStage::Vertex),
                length: false,
            },
            // Every compute descriptor allows a dispatch kind, and its records hold either.
            Self::Dispatch { .. } => plain(COMPUTE),
            Self::Pipeline { render } => Policy {
                forbidden: INHERIT_PIPELINE,
                ..plain(family(render))
            },
            Self::Barrier { render, .. } | Self::Reset { render } => plain(family(render)),
            Self::ThreadgroupMemory => Policy {
                binding: Some(3),
                length: true,
                ..plain(COMPUTE)
            },
            Self::StageInRegion | Self::ImageblockSize | Self::Copy => plain(COMPUTE),
            Self::Draw => draw(DRAW | DRAW_INDEXED | patch_records | mesh_records),
            Self::DrawIndexed { .. } => draw(DRAW_INDEXED | patch_records),
            Self::DrawPatches { .. } | Self::DrawIndexedPatches { .. } => draw(patch_records),
            Self::DrawMesh { .. } => draw(mesh_records),
            Self::UnsupportedBufferEncoding | Self::Unmodeled => return None,
        })
    }

    /// The typed spelling each pointer operand's declaration must have, by its operand index, as
    /// the overload names it: a buffer setter's buffer of bytes, an indexed draw's index buffer of
    /// its index type, a patch draw's patch index buffer of `uint`s and factor buffer of bytes, an
    /// indexed patch draw's likewise with its control-point index buffer of bytes. A
    /// declaration of another pointee is refused, so the guard never forwards a pointer of a type
    /// the overload does not state.
    pub(super) fn typed_pointers(self) -> Vec<(usize, String)> {
        match self {
            Self::Buffer { address_space, .. } => {
                vec![(2, format!("i8 addrspace({address_space})*"))]
            }
            Self::DrawIndexed {
                address_space,
                wide,
            } => {
                let bits = if wide { 32 } else { 16 };
                vec![(4, format!("i{bits} addrspace({address_space})*"))]
            }
            Self::DrawPatches {
                patch_indices,
                factors,
            } => vec![
                (5, format!("i32 addrspace({patch_indices})*")),
                (8, format!("i8 addrspace({factors})*")),
            ],
            Self::DrawIndexedPatches {
                patch_indices,
                control_point_indices,
                factors,
            } => vec![
                (5, format!("i32 addrspace({patch_indices})*")),
                (6, format!("i8 addrspace({control_point_indices})*")),
                (9, format!("i8 addrspace({factors})*")),
            ],
            _ => Vec::new(),
        }
    }

    /// Emit the same policy checks for opaque record storage and typed native policy words.
    pub(super) fn emit_policy_checks(self, body: &mut String, typed: bool) -> Option<()> {
        use std::fmt::Write;
        let pointer = if typed {
            "i32 addrspace(1)*"
        } else {
            "ptr addrspace(1)"
        };
        writeln!(body, "  %capacity = load i32, {pointer} %records, align 4\n  %command_valid = icmp ult i32 %command, %capacity").unwrap();
        let Policy {
            required,
            forbidden,
            any,
            binding,
            dynamic_stride,
            length,
        } = self.policy()?;
        header_field(body, "policy_ptr", 2, typed);
        writeln!(body, "  %policy = load i32, {pointer} %policy_ptr, align 4\n  %policy_bits = and i32 %policy, {}\n  %policy_valid = icmp eq i32 %policy_bits, {required}\n  %valid0 = and i1 %command_valid, %policy_valid", required | forbidden).unwrap();
        let mut valid = "%valid0";
        if any != 0 {
            writeln!(body, "  %held_kinds = and i32 %policy, {any}\n  %kind_held = icmp ne i32 %held_kinds, 0\n  %valid_kind = and i1 {valid}, %kind_held").unwrap();
            valid = "%valid_kind";
        }
        if let Some(byte) = binding {
            header_field(body, "counts_ptr", 3, typed);
            writeln!(body, "  %counts = load i32, {pointer} %counts_ptr, align 4\n  %shifted_count = lshr i32 %counts, {}\n  %binding_count = and i32 %shifted_count, 255\n  %binding_valid = icmp ult i32 %binding, %binding_count", u32::from(byte) * 8).unwrap();
            body.push_str("  %valid1 = and i1 %valid0, %binding_valid\n");
            valid = "%valid1";
        }
        if dynamic_stride {
            writeln!(body, "  %dynamic_bit = and i32 %policy, {}\n  %dynamic_enabled = icmp ne i32 %dynamic_bit, 0\n  %static_stride = icmp eq i64 %stride, -1\n  %stride_valid = or i1 %dynamic_enabled, %static_stride\n  %valid2 = and i1 {valid}, %stride_valid", DYNAMIC_STRIDE).unwrap();
            valid = "%valid2";
        }
        if length {
            writeln!(body, "  %length_remainder = and i32 %length, 15\n  %length_valid = icmp eq i32 %length_remainder, 0\n  %valid3 = and i1 {valid}, %length_valid").unwrap();
            valid = "%valid3";
        }
        if self == Self::Copy {
            emit_copy_checks(body, typed, valid);
            valid = "%valid4";
        }
        writeln!(
            body,
            "  br i1 {valid}, label %write, label %invalid\ninvalid:"
        )
        .unwrap();
        header_field(body, "fault", 1, typed);
        writeln!(body, "  call void @air.atomic.global.store.i32({pointer} %fault, i32 1, i32 0, i32 2, i1 true)\n  ret void\nwrite:").unwrap();
        Some(())
    }
}

fn header_field(body: &mut String, name: &str, word: usize, typed: bool) {
    header_field_of(body, "records", name, word, typed)
}

/// [`header_field`] of the header at `%{records}`.
fn header_field_of(body: &mut String, records: &str, name: &str, word: usize, typed: bool) {
    use std::fmt::Write;
    let (element, pointer, offset) = if typed {
        ("i32", "i32 addrspace(1)*", word)
    } else {
        ("i8", "ptr addrspace(1)", word * 4)
    };
    writeln!(
        body,
        "  %{name} = getelementptr {element}, {pointer} %{records}, i64 {offset}"
    )
    .unwrap();
}

/// A copy's checks beyond the destination's policy, into `%valid4` from `valid`: the source
/// command (`%source_command`) is below the capacity of the source's header (`%source_records`),
/// and either the two headers
/// are of equal descriptors (policy bits and binding counts), or the destination inherits its
/// pipeline and buffers from a source that inherits neither, of the same descriptor otherwise,
/// the destination binding no kernel buffer, onto a command other than 0. Metal's copy between
/// any other descriptors depends on its private record layout, and its copy onto an inheriting
/// buffer's command 0 runs that command with zero grid sizes (M3 Max and M4), so the guard faults
/// them.
fn emit_copy_checks(body: &mut String, typed: bool, valid: &str) {
    use std::fmt::Write;
    let pointer = if typed {
        "i32 addrspace(1)*"
    } else {
        "ptr addrspace(1)"
    };
    let inherits = INHERIT_PIPELINE | INHERIT_BUFFERS;
    writeln!(body, "  %source_capacity = load i32, {pointer} %source_records, align 4\n  %source_present = icmp ult i32 %source_command, %source_capacity").unwrap();
    header_field_of(body, "source_records", "source_policy_ptr", 2, typed);
    header_field_of(body, "source_records", "source_counts_ptr", 3, typed);
    header_field(body, "copy_counts_ptr", 3, typed);
    writeln!(
        body,
        "  %source_policy = load i32, {pointer} %source_policy_ptr, align 4
  %source_counts = load i32, {pointer} %source_counts_ptr, align 4
  %copy_counts = load i32, {pointer} %copy_counts_ptr, align 4
  %same_policy = icmp eq i32 %source_policy, %policy
  %same_counts = icmp eq i32 %source_counts, %copy_counts
  %equal = and i1 %same_policy, %same_counts
  %source_inherits = and i32 %source_policy, {inherits}
  %source_owned = icmp eq i32 %source_inherits, 0
  %inheriting_policy = or i32 %source_policy, {inherits}
  %inheriting = icmp eq i32 %policy, %inheriting_policy
  %kernel_buffers = and i32 %copy_counts, 255
  %no_kernel_buffers = icmp eq i32 %kernel_buffers, 0
  %other_counts = and i32 %copy_counts, -256
  %source_other_counts = and i32 %source_counts, -256
  %same_other_counts = icmp eq i32 %other_counts, %source_other_counts
  %into0 = and i1 %source_owned, %inheriting
  %into1 = and i1 %into0, %no_kernel_buffers
  %not_first = icmp ne i32 %command, 0
  %into2 = and i1 %into1, %not_first
  %into_inheriting = and i1 %into2, %same_other_counts
  %explained = or i1 %equal, %into_inheriting
  %copy_valid = and i1 %source_present, %explained
  %valid4 = and i1 {valid}, %copy_valid"
    )
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan::bind::indirect_commands::{
        CommandDescriptor, ComputeDispatch, Inheritance, RenderDraws, StateSource,
    };

    // The dispatch kinds' header bits, which no operation's policy names.
    const GROUPS: u32 = 4;
    const THREADS: u32 = 8;

    fn word(header: &[u8], index: usize) -> u32 {
        u32::from_le_bytes(header[index * 4..index * 4 + 4].try_into().unwrap())
    }

    /// The guard reads the header the product's `CommandDescriptor::encode_header` writes: the
    /// capacity in word 0, the fault flag in word 1, the policy bits in word 2, and the binding
    /// counts byte by byte in word 3 (`header_field`'s words, `Policy`'s bits and bytes). A change
    /// to either side's encoding fails here instead of faulting every guarded write.
    #[test]
    fn the_guard_reads_the_header_the_product_encodes() {
        let sources = [StateSource::Command, StateSource::Parent];
        for pipeline in sources {
            for buffers in sources {
                let inheritance = Inheritance { pipeline, buffers };
                let inherited = if pipeline == StateSource::Parent {
                    INHERIT_PIPELINE
                } else {
                    0
                } | if buffers == StateSource::Parent {
                    INHERIT_BUFFERS
                } else {
                    0
                };
                for dynamic in [false, true] {
                    let stride = if dynamic { DYNAMIC_STRIDE } else { 0 };
                    for (dispatch, bits) in [
                        (ComputeDispatch::Threadgroups, GROUPS),
                        (ComputeDispatch::Threads, THREADS),
                        (ComputeDispatch::Both, GROUPS | THREADS),
                    ] {
                        let header = CommandDescriptor::Compute {
                            dispatch,
                            max_buffer_bind_count: 5,
                            max_threadgroup_memory_bind_count: 3,
                            support_dynamic_attribute_stride: dynamic,
                        }
                        .encode_header(7, inheritance)
                        .unwrap();
                        assert_eq!(word(&header, 0), 7);
                        assert_eq!(word(&header, 1), 0);
                        assert_eq!(word(&header, 2), COMPUTE | bits | stride | inherited);
                        let counts = word(&header, 3);
                        assert_eq!(counts & 255, 5, "kernel buffers: byte 0");
                        assert_eq!(counts >> 24 & 255, 3, "threadgroup memory: byte 3");
                    }
                    for (primitives, indexed, patches, indexed_patches, groups, threads) in [
                        (true, false, false, false, false, false),
                        (false, true, false, false, false, false),
                        (false, false, true, false, false, false),
                        (false, false, false, true, false, false),
                        (false, false, false, false, true, false),
                        (false, false, false, false, false, true),
                        (true, true, true, true, true, true),
                    ] {
                        let header = CommandDescriptor::Render {
                            draws: RenderDraws {
                                primitives,
                                indexed,
                                patches,
                                indexed_patches,
                                mesh_threadgroups: groups,
                                mesh_threads: threads,
                            },
                            max_vertex_buffer_bind_count: 6,
                            max_fragment_buffer_bind_count: 4,
                            max_object_buffer_bind_count: 2,
                            max_mesh_buffer_bind_count: 5,
                            support_dynamic_attribute_stride: dynamic,
                        }
                        .encode_header(9, inheritance)
                        .unwrap();
                        let draws = [
                            (primitives, DRAW),
                            (indexed, DRAW_INDEXED),
                            (patches, DRAW_PATCHES),
                            (indexed_patches, DRAW_INDEXED_PATCHES),
                            (groups, DRAW_MESH_THREADGROUPS),
                            (threads, DRAW_MESH_THREADS),
                        ]
                        .into_iter()
                        .filter(|&(allowed, _)| allowed)
                        .fold(0, |bits, (_, bit)| bits | bit);
                        assert_eq!(word(&header, 0), 9);
                        assert_eq!(word(&header, 1), 0);
                        assert_eq!(word(&header, 2), RENDER | draws | stride | inherited);
                        let counts = word(&header, 3);
                        assert_eq!(counts & 255, 2, "object buffers: byte 0");
                        assert_eq!(counts >> 8 & 255, 6, "vertex buffers: byte 1");
                        assert_eq!(counts >> 16 & 255, 4, "fragment buffers: byte 2");
                        assert_eq!(counts >> 24 & 255, 5, "mesh buffers: byte 3");
                    }
                }
            }
        }
        // The bytes `Policy` reads its binding counts from.
        let byte = |stage| {
            CommandEncoderOperation::Buffer {
                stage,
                address_space: 1,
            }
            .policy()
            .and_then(|policy| policy.binding)
        };
        assert_eq!(byte(BufferStage::Kernel), Some(0));
        assert_eq!(byte(BufferStage::Vertex), Some(1));
        assert_eq!(byte(BufferStage::Fragment), Some(2));
        assert_eq!(byte(BufferStage::Object), Some(0));
        assert_eq!(byte(BufferStage::Mesh), Some(3));
        assert_eq!(
            CommandEncoderOperation::ThreadgroupMemory
                .policy()
                .and_then(|policy| policy.binding),
            Some(3)
        );
        // Each draw requires its own kind's bit, as the product's lowering does
        // (`indirect_commands::CommandEncoderOperation::policy`).
        let required = |operation: CommandEncoderOperation| operation.policy().unwrap().required;
        let any = |operation: CommandEncoderOperation| operation.policy().unwrap().any;
        // A draw requires a render descriptor that allows a kind whose records hold it, as the
        // product's lowering does (`indirect_commands::CommandEncoderOperation::policy`); a
        // dispatch, a compute one.
        let patch_records = DRAW_PATCHES | DRAW_INDEXED_PATCHES;
        let mesh_records = DRAW_MESH_THREADGROUPS | DRAW_MESH_THREADS;
        assert_eq!(required(CommandEncoderOperation::Draw), RENDER);
        assert_eq!(
            any(CommandEncoderOperation::Draw),
            DRAW | DRAW_INDEXED | patch_records | mesh_records
        );
        for threads in [false, true] {
            let mesh = CommandEncoderOperation::DrawMesh { threads };
            assert_eq!(required(mesh), RENDER);
            assert_eq!(any(mesh), mesh_records);
        }
        for exact_threads in [false, true] {
            let dispatch = CommandEncoderOperation::Dispatch { exact_threads };
            assert_eq!(required(dispatch), COMPUTE);
            assert_eq!(any(dispatch), 0);
        }
        for (address_space, wide) in [(1, false), (2, true)] {
            let indexed = CommandEncoderOperation::DrawIndexed {
                address_space,
                wide,
            };
            assert_eq!(required(indexed), RENDER);
            assert_eq!(any(indexed), DRAW_INDEXED | patch_records);
        }
        assert_eq!(
            required(CommandEncoderOperation::Reset { render: false }),
            COMPUTE
        );
        assert_eq!(required(CommandEncoderOperation::StageInRegion), COMPUTE);
        assert_eq!(required(CommandEncoderOperation::ImageblockSize), COMPUTE);
        assert_eq!(
            required(CommandEncoderOperation::Reset { render: true }),
            RENDER
        );
        for (patch_indices, factors) in [(1, 1), (2, 1), (1, 2)] {
            let patches = CommandEncoderOperation::DrawPatches {
                patch_indices,
                factors,
            };
            assert_eq!(required(patches), RENDER);
            assert_eq!(any(patches), patch_records);
        }
        for (patch_indices, control_point_indices, factors) in [(1, 1, 1), (2, 2, 2), (1, 2, 1)] {
            let indexed_patches = CommandEncoderOperation::DrawIndexedPatches {
                patch_indices,
                control_point_indices,
                factors,
            };
            assert_eq!(required(indexed_patches), RENDER);
            assert_eq!(any(indexed_patches), patch_records);
        }
    }
}
