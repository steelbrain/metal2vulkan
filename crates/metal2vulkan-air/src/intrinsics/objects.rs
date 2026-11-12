//! Object families: simdgroup async copies and events, visible and intersection function tables,
//! acceleration-structure queries, tensor descriptors, and indirect-command encoding.
//!
//! These decode to typed signatures only; the positions that are not handles are ordinary
//! operands whose meaning lowering owns (or refuses). Ray intersection (`air.intersect`) is its
//! own family (`intersect`).

use super::sig::{HandleKind, Operand, Signature};
use super::ty::{Elem, NumTy};
use crate::ir::AddrSpace;

fn num(elem: Elem, lanes: u32) -> Operand {
    Operand::Num(NumTy { elem, lanes })
}

/// A byte-pointer overload token `p<space>i8`.
fn byte_ptr(token: &str) -> Option<AddrSpace> {
    let space = token.strip_prefix('p')?.strip_suffix("i8")?;
    AddrSpace::from_raw(space.parse().ok()?).ok()
}

/// An index-pointer overload token `p<space>i16` or `p<space>i32` in device or constant memory,
/// as an indexed draw's index buffer takes: the space and whether the indices are 32-bit.
fn index_ptr(token: &str) -> Option<(AddrSpace, bool)> {
    let token = token.strip_prefix('p')?;
    let (space, wide) = match (token.strip_suffix("i16"), token.strip_suffix("i32")) {
        (Some(space), _) => (space, false),
        (_, Some(space)) => (space, true),
        _ => return None,
    };
    let space = AddrSpace::from_raw(space.parse().ok()?).ok()?;
    matches!(space, AddrSpace::Device | AddrSpace::Constant).then_some((space, wide))
}

/// Simdgroup asynchronous copies and their events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AsyncCopy {
    /// `air.simdgroup_async_copy_2d.p<dst>i8.p<src>i8`: `event (i64, i64, dst, i64, i64,
    /// <2 x i64>, src, i64, i64, <2 x i64>, <2 x i64>, i32)` between device and threadgroup
    /// memory.
    Copy2d {
        /// The destination's address space: device or threadgroup.
        dst: AddrSpace,
        /// The source's address space: the other of the two.
        src: AddrSpace,
    },
    /// `air.get_null_simdgroup_event`: `event ()`.
    NullEvent,
    /// `air.is_null_simdgroup_event`: `i1 (event)`.
    IsNullEvent,
    /// `air.wait_simdgroup_events`: `void (i32 count, ptr events)`.
    WaitEvents,
}

impl AsyncCopy {
    pub(super) fn parse(rest: &str) -> Option<AsyncCopy> {
        match rest {
            "get_null_simdgroup_event" => return Some(AsyncCopy::NullEvent),
            "is_null_simdgroup_event" => return Some(AsyncCopy::IsNullEvent),
            "wait_simdgroup_events" => return Some(AsyncCopy::WaitEvents),
            _ => {}
        }
        let (dst, src) = rest
            .strip_prefix("simdgroup_async_copy_2d.")?
            .split_once('.')?;
        let (dst, src) = (byte_ptr(dst)?, byte_ptr(src)?);
        let pair = matches!(
            (dst, src),
            (AddrSpace::Threadgroup, AddrSpace::Device)
                | (AddrSpace::Device, AddrSpace::Threadgroup)
        );
        pair.then_some(AsyncCopy::Copy2d { dst, src })
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        let event = Operand::Handle(HandleKind::Event);
        let (i64s, v2i64) = (num(Elem::I64, 1), num(Elem::I64, 2));
        match *self {
            AsyncCopy::Copy2d { dst, src } => Signature::new(
                event,
                &[
                    i64s,
                    i64s,
                    Operand::Ptr(dst),
                    i64s,
                    i64s,
                    v2i64,
                    Operand::Ptr(src),
                    i64s,
                    i64s,
                    v2i64,
                    v2i64,
                    num(Elem::I32, 1),
                ],
            ),
            AsyncCopy::NullEvent => Signature::new(event, &[]),
            AsyncCopy::IsNullEvent => Signature::new(num(Elem::I1, 1), &[event]),
            AsyncCopy::WaitEvents => Signature::new(
                Operand::Void,
                &[num(Elem::I32, 1), Operand::Ptr(AddrSpace::Thread)],
            ),
        }
    }
}

/// Visible and intersection function tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FunctionTable {
    /// `air.get_function_pointer_visible_function_table`: `ptr (table, i32 index)`.
    GetFunctionPointer,
    /// `air.get_size_visible_function_table`: `i32 (table)`.
    Size,
    /// `air.is_null_visible_function_table`: `i1 (table)`.
    IsNull,
    /// `air.get_null_intersection_function_table`: `table ()`.
    NullIntersection,
    /// `air.set_buffer_intersection_function_table.p1i8`: `void (table, ptr addrspace(1),
    /// i32 index)`.
    SetIntersectionBuffer,
    /// Word `word` of the visible-function table whose resource id is `id`: `i32 (i64 id, i32
    /// word)`, the words of `pipeline::table` (1 for a table, its size, its entries), and 0 for
    /// an id that names no table the pipeline state made (the null table). Not an AIR
    /// intrinsic: no name parses to it. Devirtualization makes it of an operation on a table
    /// loaded from memory, an argument buffer's member, and the access family reads it through the
    /// stage's function-table directory and set-1 heap.
    Word,
}

impl FunctionTable {
    pub(super) fn parse(rest: &str) -> Option<FunctionTable> {
        Some(match rest {
            "get_function_pointer_visible_function_table" => FunctionTable::GetFunctionPointer,
            "get_size_visible_function_table" => FunctionTable::Size,
            "is_null_visible_function_table" => FunctionTable::IsNull,
            "get_null_intersection_function_table" => FunctionTable::NullIntersection,
            "set_buffer_intersection_function_table.p1i8" => FunctionTable::SetIntersectionBuffer,
            _ => return None,
        })
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        let vft = Operand::Handle(HandleKind::VisibleFunctionTable);
        let ift = Operand::Handle(HandleKind::IntersectionFunctionTable);
        let i32s = num(Elem::I32, 1);
        match self {
            FunctionTable::GetFunctionPointer => {
                Signature::new(Operand::Ptr(AddrSpace::Thread), &[vft, i32s])
            }
            FunctionTable::Size => Signature::new(i32s, &[vft]),
            FunctionTable::IsNull => Signature::new(num(Elem::I1, 1), &[vft]),
            FunctionTable::NullIntersection => Signature::new(ift, &[]),
            FunctionTable::SetIntersectionBuffer => {
                Signature::new(Operand::Void, &[ift, Operand::Ptr(AddrSpace::Device), i32s])
            }
            FunctionTable::Word => Signature::new(i32s, &[num(Elem::I64, 1), i32s]),
        }
    }
}

/// Instance acceleration-structure queries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccelerationStructure {
    /// `i32 (instance structure)`.
    InstanceCount,
    /// `structure (instance structure, i32 index)`.
    PrimitiveStructure,
    /// `ptr addrspace(1) (instance structure)`: the instance descriptors.
    DataPointer,
}

impl AccelerationStructure {
    pub(super) fn parse(rest: &str) -> Option<AccelerationStructure> {
        Some(match rest {
            "get_instance_count_instance_acceleration_structure" => {
                AccelerationStructure::InstanceCount
            }
            "get_primitive_acceleration_structure_instance_acceleration_structure" => {
                AccelerationStructure::PrimitiveStructure
            }
            "get_data_pointer_instance_acceleration_structure" => {
                AccelerationStructure::DataPointer
            }
            _ => return None,
        })
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        let s = Operand::Handle(HandleKind::AccelerationStructure);
        match self {
            AccelerationStructure::InstanceCount => Signature::new(num(Elem::I32, 1), &[s]),
            AccelerationStructure::PrimitiveStructure => Signature::new(s, &[s, num(Elem::I32, 1)]),
            AccelerationStructure::DataPointer => {
                Signature::new(Operand::Ptr(AddrSpace::Device), &[s])
            }
        }
    }
}

/// Tensor descriptor operations (Apple's opaque tensor ABI; typed signatures only).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tensor {
    /// `air.get_descriptor_size_tensor`: `i16 (i16, i16)`.
    DescriptorSize,
    /// `air.get_extent_private_tensor.i32`: `i32 (ptr, i16, i16)`.
    Extent,
    /// `air.init_strided_private_tensor.i32.{global,local}`: `void (ptr, i16, ptr data, ptr,
    /// ptr, i8)`, `data` in device or threadgroup memory; in AIR 2.9 (deployment target macOS
    /// 27) `void (ptr, i16, ptr data, ptr, ptr, i8, ptr, i32, i8)`, the added pointer in
    /// `data`'s space.
    InitStrided {
        /// The data pointer's address space: device (`global`) or threadgroup (`local`).
        data: AddrSpace,
        /// The AIR 2.9 form: three more operands, a pointer in `data`'s space, an `i32` and
        /// an `i8`. Metal's header passes null, 0 and -1 for a strided tensor's.
        trailing: bool,
    },
    /// `air.slice_private_tensor_private_tensor.s.i32`: `void (ptr, ptr, i16, ptr, ptr)`.
    Slice,
}

impl Tensor {
    pub(super) fn parse(rest: &str) -> Option<Tensor> {
        Some(match rest {
            "get_descriptor_size_tensor" => Tensor::DescriptorSize,
            "get_extent_private_tensor.i32" => Tensor::Extent,
            "init_strided_private_tensor.i32.global" => Tensor::InitStrided {
                data: AddrSpace::Device,
                trailing: false,
            },
            "init_strided_private_tensor.i32.local" => Tensor::InitStrided {
                data: AddrSpace::Threadgroup,
                trailing: false,
            },
            "slice_private_tensor_private_tensor.s.i32" => Tensor::Slice,
            _ => return None,
        })
    }

    /// The AIR 2.9 form of an AIR 2.8 strided initialization, the other declaration its name
    /// may have.
    pub(super) fn alternative(self) -> Option<Tensor> {
        match self {
            Tensor::InitStrided {
                data,
                trailing: false,
            } => Some(Tensor::InitStrided {
                data,
                trailing: true,
            }),
            _ => None,
        }
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        let p = Operand::Ptr(AddrSpace::Thread);
        let i16s = num(Elem::I16, 1);
        match *self {
            Tensor::DescriptorSize => Signature::new(i16s, &[i16s, i16s]),
            Tensor::Extent => Signature::new(num(Elem::I32, 1), &[p, i16s, i16s]),
            Tensor::InitStrided {
                data,
                trailing: false,
            } => Signature::new(
                Operand::Void,
                &[p, i16s, Operand::Ptr(data), p, p, num(Elem::I8, 1)],
            ),
            Tensor::InitStrided {
                data,
                trailing: true,
            } => Signature::new(
                Operand::Void,
                &[
                    p,
                    i16s,
                    Operand::Ptr(data),
                    p,
                    p,
                    num(Elem::I8, 1),
                    Operand::Ptr(data),
                    num(Elem::I32, 1),
                    num(Elem::I8, 1),
                ],
            ),
            Tensor::Slice => Signature::new(Operand::Void, &[p, p, i16s, p, p]),
        }
    }
}

/// The stage whose buffer an indirect command sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CommandStage {
    /// `set_kernel_buffer_compute_command`: a compute command's kernel buffer.
    Kernel,
    /// `set_vertex_buffer_render_command`: a render command's vertex buffer.
    Vertex,
    /// `set_fragment_buffer_render_command`: a render command's fragment buffer.
    Fragment,
    /// `set_object_buffer_render_command`: a render command's object buffer.
    Object,
    /// `set_mesh_buffer_render_command`: a render command's mesh buffer.
    Mesh,
}

/// The fixed-function state a render command's setter sets (MSL 4.0 `render_command::set_*`): each
/// a `void (cmd, i32, ...)` whose values follow the index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RenderState {
    /// `air.set_cull_mode_render_command`: an `i32` cull mode.
    CullMode,
    /// `air.set_depth_bias_render_command`: three `float`s, the bias, its slope scale and its
    /// clamp.
    DepthBias,
    /// `air.set_depth_clip_mode_render_command`: an `i32` depth clip mode.
    DepthClipMode,
    /// `air.set_front_facing_winding_render_command`: an `i32` winding.
    FrontFacingWinding,
    /// `air.set_triangle_fill_mode_render_command`: an `i32` fill mode.
    TriangleFillMode,
    /// `air.set_depth_stencil_state_render_command`: a `depth_stencil_state` handle (`ptr
    /// addrspace(1)`), as Metal's compiler declares it.
    DepthStencilState,
}

/// Indirect-command-buffer encoding. Every operation takes the command handle and its index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IndirectCommand {
    /// `air.concurrent_dispatch_{threadgroups,threads}_compute_command`: `void (cmd, i32,
    /// <3 x i32>, <3 x i32>)`.
    Dispatch {
        /// `concurrent_dispatch_threads` (a grid in threads) rather than
        /// `concurrent_dispatch_threadgroups` (in threadgroups).
        exact_threads: bool,
    },
    /// `air.set_pipeline_state_compute_command`: `void (cmd, i32, pipeline)`, the pipeline a
    /// `compute_pipeline_state` handle (`ptr addrspace(1)`), not memory the command reads.
    SetComputePipeline,
    /// `air.{set,clear}_barrier_compute_command`: `void (cmd, i32)`; `enabled` for `set`.
    ComputeBarrier {
        /// `true` for `set_barrier`, `false` for `clear_barrier`.
        enabled: bool,
    },
    /// `air.set_threadgroup_memory_length_compute_command`: `void (cmd, i32, i32, i32)`.
    SetThreadgroupMemoryLength,
    /// `air.set_stage_in_region_compute_command`: `void (cmd, i32, <3 x i32>, <3 x i32>)`: the
    /// region's origin, then its size.
    SetStageInRegion,
    /// `air.set_imageblock_size_compute_command`: `void (cmd, i32, <2 x i16>)`: the width, then
    /// the height.
    SetImageblockSize,
    /// `air.reset_compute_command`: `void (cmd, i32)`.
    ResetComputeCommand,
    /// `air.copy_compute_command`: `void (cmd, i32, cmd, i32)`: the command, then the source
    /// command buffer and its command's index (the probe).
    CopyComputeCommand,
    /// `air.draw_primitives_render_command`: `void (cmd, i32, i32, i32, i32, i32, i32)`: the
    /// index, primitive type, vertex start, vertex count, instance count, base instance.
    DrawPrimitives,
    /// `air.draw_indexed_primitives_render_command.p<space>i<16|32>`: `void (cmd, i32, i32, i32,
    /// ptr, i32, i32, i32)`: the index, primitive type, index count, the index buffer (16-bit or
    /// 32-bit indices, as `wide` says, the overload naming the index type), instance count, base
    /// vertex, base instance.
    DrawIndexedPrimitives {
        /// The address space of the index buffer: device or constant.
        space: AddrSpace,
        /// Whether the indices are 32-bit rather than 16-bit.
        wide: bool,
    },
    /// `air.draw_patches_render_command.p<space>i32.p<space>i8`: `void (cmd, i32, i32, i32, i32,
    /// ptr, i32, i32, ptr, i32, float)`: the index, control points, patch start, patch count, the
    /// patch index buffer (`uint`s; null for none), instance count, base instance, the
    /// tessellation factor buffer, its instance stride, and a factor scale; the overload names each
    /// buffer's space.
    DrawPatches {
        /// The address space of the patch index buffer: device or constant.
        patch_indices: AddrSpace,
        /// The address space of the tessellation factor buffer: device or constant.
        factors: AddrSpace,
    },
    /// `air.set_pipeline_state_render_command`: `void (cmd, i32, pipeline)`, the pipeline a
    /// `render_pipeline_state` handle, as the compute setter's.
    SetRenderPipeline,
    /// `air.reset_render_command`: `void (cmd, i32)`.
    ResetRenderCommand,
    /// `air.set_{kernel,vertex}_buffer_*_command.p<space>i8`: `void (cmd, i32, ptr, i64, i32)`,
    /// the attribute stride before the buffer index (`strided`); the fragment, object and mesh
    /// forms have no `i64`. AIR 2.5 (Metal compiler 31001.723) declares the kernel and vertex
    /// setters without it too: MSL's plain setter, which today's compiler spells with an `i64` of
    /// all ones, the pipeline's own stride.
    SetBuffer {
        /// The stage whose buffer the command sets.
        stage: CommandStage,
        /// The address space of the buffer: device or constant.
        space: AddrSpace,
        /// Whether the declaration takes the attribute stride (`i64`) before the buffer index.
        strided: bool,
    },
    /// A render command's fixed-function state setter ([`RenderState`]).
    SetRenderState(RenderState),
    /// `air.draw_indexed_patches_render_command.p<space>i32.p<space>i8.p<space>i8`: `void (cmd,
    /// i32, i32, i32, i32, ptr, ptr, i32, i32, ptr, i32, float)`: the index, control points,
    /// patch start, patch count, the patch index buffer (`uint`s; null for none), the
    /// control-point index buffer (bytes of the pipeline's index type; null for none), instance
    /// count, base instance, the tessellation factor buffer, its instance stride, and a factor
    /// scale; the overload names each buffer's space.
    DrawIndexedPatches {
        /// The address space of the patch index buffer: device or constant.
        patch_indices: AddrSpace,
        /// The address space of the control-point index buffer: device or constant.
        control_point_indices: AddrSpace,
        /// The address space of the tessellation factor buffer: device or constant.
        factors: AddrSpace,
    },
    /// `air.{set,clear}_barrier_render_command`: `void (cmd, i32)`; `enabled` for `set`.
    RenderBarrier {
        /// `true` for `set_barrier`, `false` for `clear_barrier`.
        enabled: bool,
    },
    /// `air.draw_mesh_{threadgroups,threads}_render_command`: `void (cmd, i32, <3 x i32>,
    /// <3 x i32>, <3 x i32>)`: the index, the grid (in threadgroups, or in threads where
    /// `threads`), the threads of an object threadgroup and of a mesh threadgroup.
    DrawMesh {
        /// `draw_mesh_threads` (a grid in threads) rather than `draw_mesh_threadgroups`.
        threads: bool,
    },
    /// `air.set_object_threadgroup_memory_length_render_command`: `void (cmd, i32, i32, i32)`:
    /// the index, the length, then its binding.
    SetObjectThreadgroupMemoryLength,
}

impl IndirectCommand {
    /// Whether the operation writes its pointer operand's guest address into the command's
    /// record, reading none of its bytes: a buffer setter's buffer, an indexed draw's index
    /// buffer, a patch draw's patch index and factor buffers, and an indexed patch draw's
    /// control-point index buffer too.
    pub fn records_address(self) -> bool {
        matches!(
            self,
            IndirectCommand::SetBuffer { .. }
                | IndirectCommand::DrawIndexedPrimitives { .. }
                | IndirectCommand::DrawPatches { .. }
                | IndirectCommand::DrawIndexedPatches { .. }
        )
    }

    pub(super) fn parse(rest: &str) -> Option<IndirectCommand> {
        match rest {
            "concurrent_dispatch_threadgroups_compute_command" => {
                return Some(IndirectCommand::Dispatch {
                    exact_threads: false,
                })
            }
            "concurrent_dispatch_threads_compute_command" => {
                return Some(IndirectCommand::Dispatch {
                    exact_threads: true,
                })
            }
            "set_pipeline_state_compute_command" => {
                return Some(IndirectCommand::SetComputePipeline)
            }
            "set_barrier_compute_command" => {
                return Some(IndirectCommand::ComputeBarrier { enabled: true })
            }
            "clear_barrier_compute_command" => {
                return Some(IndirectCommand::ComputeBarrier { enabled: false })
            }
            "set_threadgroup_memory_length_compute_command" => {
                return Some(IndirectCommand::SetThreadgroupMemoryLength)
            }
            "set_stage_in_region_compute_command" => {
                return Some(IndirectCommand::SetStageInRegion)
            }
            "set_imageblock_size_compute_command" => {
                return Some(IndirectCommand::SetImageblockSize)
            }
            "reset_compute_command" => return Some(IndirectCommand::ResetComputeCommand),
            "copy_compute_command" => return Some(IndirectCommand::CopyComputeCommand),
            "draw_primitives_render_command" => return Some(IndirectCommand::DrawPrimitives),
            "set_pipeline_state_render_command" => return Some(IndirectCommand::SetRenderPipeline),
            "reset_render_command" => return Some(IndirectCommand::ResetRenderCommand),
            "set_barrier_render_command" => {
                return Some(IndirectCommand::RenderBarrier { enabled: true })
            }
            "clear_barrier_render_command" => {
                return Some(IndirectCommand::RenderBarrier { enabled: false })
            }
            "draw_mesh_threadgroups_render_command" => {
                return Some(IndirectCommand::DrawMesh { threads: false })
            }
            "draw_mesh_threads_render_command" => {
                return Some(IndirectCommand::DrawMesh { threads: true })
            }
            "set_object_threadgroup_memory_length_render_command" => {
                return Some(IndirectCommand::SetObjectThreadgroupMemoryLength)
            }
            "set_cull_mode_render_command" => {
                return Some(IndirectCommand::SetRenderState(RenderState::CullMode))
            }
            "set_depth_bias_render_command" => {
                return Some(IndirectCommand::SetRenderState(RenderState::DepthBias))
            }
            "set_depth_clip_mode_render_command" => {
                return Some(IndirectCommand::SetRenderState(RenderState::DepthClipMode))
            }
            "set_front_facing_winding_render_command" => {
                return Some(IndirectCommand::SetRenderState(
                    RenderState::FrontFacingWinding,
                ))
            }
            "set_triangle_fill_mode_render_command" => {
                return Some(IndirectCommand::SetRenderState(
                    RenderState::TriangleFillMode,
                ))
            }
            "set_depth_stencil_state_render_command" => {
                return Some(IndirectCommand::SetRenderState(
                    RenderState::DepthStencilState,
                ))
            }
            _ => {}
        }
        let (family, ptr) = rest.split_once('.')?;
        if family == "draw_indexed_primitives_render_command" {
            let (space, wide) = index_ptr(ptr)?;
            return Some(IndirectCommand::DrawIndexedPrimitives { space, wide });
        }
        if family == "draw_patches_render_command" {
            let (indices, factors) = ptr.split_once('.')?;
            let (patch_indices, true) = index_ptr(indices)? else {
                return None;
            };
            let factors = byte_ptr(factors)?;
            return matches!(factors, AddrSpace::Device | AddrSpace::Constant).then_some(
                IndirectCommand::DrawPatches {
                    patch_indices,
                    factors,
                },
            );
        }
        if family == "draw_indexed_patches_render_command" {
            let mut buffers = ptr.split('.');
            let (Some(indices), Some(points), Some(factors), None) = (
                buffers.next(),
                buffers.next(),
                buffers.next(),
                buffers.next(),
            ) else {
                return None;
            };
            let (patch_indices, true) = index_ptr(indices)? else {
                return None;
            };
            let control_point_indices = byte_ptr(points)?;
            let factors = byte_ptr(factors)?;
            return [control_point_indices, factors]
                .iter()
                .all(|space| matches!(space, AddrSpace::Device | AddrSpace::Constant))
                .then_some(IndirectCommand::DrawIndexedPatches {
                    patch_indices,
                    control_point_indices,
                    factors,
                });
        }
        let stage = match family {
            "set_kernel_buffer_compute_command" => CommandStage::Kernel,
            "set_vertex_buffer_render_command" => CommandStage::Vertex,
            "set_fragment_buffer_render_command" => CommandStage::Fragment,
            "set_object_buffer_render_command" => CommandStage::Object,
            "set_mesh_buffer_render_command" => CommandStage::Mesh,
            _ => return None,
        };
        let space = byte_ptr(ptr)?;
        matches!(space, AddrSpace::Device | AddrSpace::Constant).then_some(
            IndirectCommand::SetBuffer {
                stage,
                space,
                strided: matches!(stage, CommandStage::Kernel | CommandStage::Vertex),
            },
        )
    }

    /// The other declaration the same name may have: AIR 2.5's kernel or vertex buffer setter,
    /// without the attribute stride (`void (cmd, i32, ptr, i32)`), MSL's plain setter.
    pub(super) fn alternative(self) -> Option<IndirectCommand> {
        match self {
            IndirectCommand::SetBuffer {
                stage,
                space,
                strided: true,
            } => Some(IndirectCommand::SetBuffer {
                stage,
                space,
                strided: false,
            }),
            _ => None,
        }
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        let cmd = Operand::Handle(HandleKind::Command);
        let i32s = num(Elem::I32, 1);
        let v3 = num(Elem::I32, 3);
        let void = Operand::Void;
        match *self {
            IndirectCommand::Dispatch { .. } => Signature::new(void, &[cmd, i32s, v3, v3]),
            IndirectCommand::SetComputePipeline => Signature::new(
                void,
                &[cmd, i32s, Operand::Handle(HandleKind::ComputePipelineState)],
            ),
            IndirectCommand::ComputeBarrier { .. }
            | IndirectCommand::RenderBarrier { .. }
            | IndirectCommand::ResetComputeCommand
            | IndirectCommand::ResetRenderCommand => Signature::new(void, &[cmd, i32s]),
            IndirectCommand::SetRenderPipeline => Signature::new(
                void,
                &[cmd, i32s, Operand::Handle(HandleKind::RenderPipelineState)],
            ),
            IndirectCommand::DrawIndexedPrimitives { space, .. } => Signature::new(
                void,
                &[cmd, i32s, i32s, i32s, Operand::Ptr(space), i32s, i32s, i32s],
            ),
            IndirectCommand::DrawPatches {
                patch_indices,
                factors,
            } => Signature::new(
                void,
                &[
                    cmd,
                    i32s,
                    i32s,
                    i32s,
                    i32s,
                    Operand::Ptr(patch_indices),
                    i32s,
                    i32s,
                    Operand::Ptr(factors),
                    i32s,
                    num(Elem::F32, 1),
                ],
            ),
            IndirectCommand::DrawIndexedPatches {
                patch_indices,
                control_point_indices,
                factors,
            } => Signature::new(
                void,
                &[
                    cmd,
                    i32s,
                    i32s,
                    i32s,
                    i32s,
                    Operand::Ptr(patch_indices),
                    Operand::Ptr(control_point_indices),
                    i32s,
                    i32s,
                    Operand::Ptr(factors),
                    i32s,
                    num(Elem::F32, 1),
                ],
            ),
            IndirectCommand::SetThreadgroupMemoryLength
            | IndirectCommand::SetObjectThreadgroupMemoryLength => {
                Signature::new(void, &[cmd, i32s, i32s, i32s])
            }
            IndirectCommand::DrawMesh { .. } => Signature::new(void, &[cmd, i32s, v3, v3, v3]),
            IndirectCommand::SetStageInRegion => Signature::new(void, &[cmd, i32s, v3, v3]),
            IndirectCommand::CopyComputeCommand => Signature::new(void, &[cmd, i32s, cmd, i32s]),
            IndirectCommand::SetImageblockSize => {
                Signature::new(void, &[cmd, i32s, num(Elem::I16, 2)])
            }
            IndirectCommand::DrawPrimitives => {
                Signature::new(void, &[cmd, i32s, i32s, i32s, i32s, i32s, i32s])
            }
            IndirectCommand::SetBuffer {
                space,
                strided: false,
                ..
            } => Signature::new(void, &[cmd, i32s, Operand::Ptr(space), i32s]),
            IndirectCommand::SetBuffer {
                space,
                strided: true,
                ..
            } => Signature::new(
                void,
                &[cmd, i32s, Operand::Ptr(space), num(Elem::I64, 1), i32s],
            ),
            IndirectCommand::SetRenderState(RenderState::DepthBias) => {
                let f32s = num(Elem::F32, 1);
                Signature::new(void, &[cmd, i32s, f32s, f32s, f32s])
            }
            IndirectCommand::SetRenderState(RenderState::DepthStencilState) => Signature::new(
                void,
                &[cmd, i32s, Operand::Handle(HandleKind::DepthStencilState)],
            ),
            IndirectCommand::SetRenderState(_) => Signature::new(void, &[cmd, i32s, i32s]),
        }
    }
}
