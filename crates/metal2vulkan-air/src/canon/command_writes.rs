//! What an entry's indirect-command operations may write into the records they reach
//! ([`CommandWrites`], `Entry::command_writes`).

use crate::abi::EntryPoint;
use crate::intrinsics::{AirIntrinsic, IndirectCommand};
use crate::ir::{Call, Callee, Const, FuncId, InstKind, Module, ModuleParts, Value};

/// What a function's indirect-command operations may write into the records of the indirect
/// command buffers they reach, read from its AIR: the kinds of operation its body may call
/// (`operations`, the `CommandWrites::*` bits) and the primitive types its draws may pass
/// (`primitive_types`). A bit that is clear is proven: no operation of the function writes it.
/// A set bit says only that one may (an operation on a path that never runs sets it too).
///
/// A draw's primitive type (`draw_primitives` and `draw_indexed_primitives`) counts as the value
/// its operand holds where that is a constant of Metal's five (`MTLPrimitiveType`'s raw values,
/// bit `n` for value `n`), as a point draw where it is a constant past them (Metal's encoding
/// writes one so: the translator's `lower::access::encode`), and as all five where it is anything
/// else. A call the walk cannot see
/// into (an indirect call, an intrinsic outside the closed set, a call to a declaration other
/// than the entry's control-point functions) may write anything: [`CommandWrites::ANY`].
/// Records the function does not write are not its own fact: a command buffer's other writers
/// (the CPU, other functions, blit copies) add theirs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CommandWrites {
    /// The operations it may call, as the `CommandWrites::*` operation bits.
    pub operations: u32,
    /// The primitive types its draws may pass: bit `n` for `MTLPrimitiveType` `n` (0 point,
    /// 1 line, 2 line strip, 3 triangle, 4 triangle strip).
    pub primitive_types: u8,
}

impl CommandWrites {
    /// Nothing: a function that calls no indirect-command operation and makes no call the walk
    /// cannot see into.
    pub const NONE: CommandWrites = CommandWrites {
        operations: 0,
        primitive_types: 0,
    };
    /// `draw_primitives`.
    pub const DRAW: u32 = 1;
    /// `draw_indexed_primitives`.
    pub const DRAW_INDEXED: u32 = 1 << 1;
    /// `draw_patches` and `draw_indexed_patches`.
    pub const DRAW_PATCHES: u32 = 1 << 2;
    /// `draw_mesh_threadgroups` and `draw_mesh_threads`.
    pub const DRAW_MESH: u32 = 1 << 3;
    /// `concurrent_dispatch_threadgroups` and `concurrent_dispatch_threads`.
    pub const DISPATCH: u32 = 1 << 4;
    /// `set_pipeline_state` of a compute or a render command: a record that names a state.
    pub const PIPELINE: u32 = 1 << 5;
    /// A buffer setter of any stage.
    pub const BUFFER: u32 = 1 << 6;
    /// `set_barrier` of a compute or a render command.
    pub const BARRIER: u32 = 1 << 7;
    /// `clear_barrier` of a compute or a render command.
    pub const CLEAR_BARRIER: u32 = 1 << 8;
    /// A render command's fixed-function state setter (cull mode, depth bias, depth clip mode,
    /// front-facing winding, triangle fill mode, depth-stencil state).
    pub const RENDER_STATE: u32 = 1 << 9;
    /// A threadgroup-memory length setter: a compute command's, or a render command's object
    /// threadgroup's.
    pub const THREADGROUP_MEMORY: u32 = 1 << 10;
    /// `set_stage_in_region`.
    pub const STAGE_IN_REGION: u32 = 1 << 11;
    /// `set_imageblock_size`.
    pub const IMAGEBLOCK_SIZE: u32 = 1 << 12;
    /// `reset` of a compute or a render command.
    pub const RESET: u32 = 1 << 13;
    /// `copy_command`: a compute command's record from another command buffer's, its pipeline
    /// included.
    pub const COPY: u32 = 1 << 14;
    /// Every operation bit.
    pub const ALL_OPERATIONS: u32 = (1 << 15) - 1;
    /// Every primitive type's bit.
    pub const ALL_PRIMITIVE_TYPES: u8 = 0b1_1111;
    /// Anything: every operation, every primitive type.
    pub const ANY: CommandWrites = CommandWrites {
        operations: CommandWrites::ALL_OPERATIONS,
        primitive_types: CommandWrites::ALL_PRIMITIVE_TYPES,
    };

    /// Both writers' writes.
    pub const fn union(self, other: CommandWrites) -> CommandWrites {
        CommandWrites {
            operations: self.operations | other.operations,
            primitive_types: self.primitive_types | other.primitive_types,
        }
    }

    /// Whether it writes nothing.
    pub const fn is_empty(self) -> bool {
        self.operations == 0 && self.primitive_types == 0
    }

    /// Whether it may call an operation of any of `bits`.
    pub const fn may(self, bits: u32) -> bool {
        self.operations & bits != 0
    }

    /// Whether a draw it makes may pass the primitive type of raw value `raw` (`MTLPrimitiveType`;
    /// a value past Metal's five is never one).
    pub const fn may_draw_primitive_type(self, raw: u32) -> bool {
        raw < 5 && self.primitive_types & (1 << raw) != 0
    }

    /// The writes of one operation `op`, called by `call`.
    fn of(op: IndirectCommand, call: &Call, parts: &ModuleParts) -> CommandWrites {
        use IndirectCommand as C;
        let operations = match op {
            C::DrawPrimitives => CommandWrites::DRAW,
            C::DrawIndexedPrimitives { .. } => CommandWrites::DRAW_INDEXED,
            C::DrawPatches { .. } | C::DrawIndexedPatches { .. } => CommandWrites::DRAW_PATCHES,
            C::DrawMesh { .. } => CommandWrites::DRAW_MESH,
            C::Dispatch { .. } => CommandWrites::DISPATCH,
            C::SetComputePipeline | C::SetRenderPipeline => CommandWrites::PIPELINE,
            C::SetBuffer { .. } => CommandWrites::BUFFER,
            C::ComputeBarrier { enabled } | C::RenderBarrier { enabled } => {
                if enabled {
                    CommandWrites::BARRIER
                } else {
                    CommandWrites::CLEAR_BARRIER
                }
            }
            C::SetRenderState(_) => CommandWrites::RENDER_STATE,
            C::SetThreadgroupMemoryLength | C::SetObjectThreadgroupMemoryLength => {
                CommandWrites::THREADGROUP_MEMORY
            }
            C::SetStageInRegion => CommandWrites::STAGE_IN_REGION,
            C::SetImageblockSize => CommandWrites::IMAGEBLOCK_SIZE,
            C::ResetComputeCommand | C::ResetRenderCommand => CommandWrites::RESET,
            C::CopyComputeCommand => CommandWrites::COPY,
        };
        // Both draws take the primitive type after the command and its index.
        let primitive_types = match op {
            C::DrawPrimitives | C::DrawIndexedPrimitives { .. } => {
                match call.args.get(2).and_then(|&v| literal(parts, v)) {
                    Some(raw) if raw < 5 => 1 << raw,
                    Some(_) => 1,
                    None => CommandWrites::ALL_PRIMITIVE_TYPES,
                }
            }
            _ => 0,
        };
        CommandWrites {
            operations,
            primitive_types,
        }
    }
}

/// The literal integer `v`, when it is one.
fn literal(parts: &ModuleParts, v: Value) -> Option<u64> {
    let Value::Const(c) = v else {
        return None;
    };
    match parts.consts.get(c)?.value {
        Const::Int(x) => Some(x),
        _ => None,
    }
}

/// What the function bodies of `module` may write through indirect-command operations
/// (`Entry::command_writes`): each operation's writes, or [`CommandWrites::ANY`] at a call the
/// walk cannot see into, as `calls` counts them. Every instruction is visited once.
pub(super) fn command_writes(module: &Module, ep: &EntryPoint) -> CommandWrites {
    let parts = module.parts();
    let declared = |f: FuncId| module.func(f).is_none_or(|f| f.body.is_none());
    let mut writes = CommandWrites::NONE;
    let insts = parts
        .funcs
        .iter()
        .filter_map(|(_, f)| f.body.as_ref())
        .flat_map(|body| body.insts.values());
    for inst in insts {
        let InstKind::Call(call) = &inst.kind else {
            continue;
        };
        match call.callee {
            Callee::Air {
                op: AirIntrinsic::IndirectCommand(op),
                ..
            } => writes = writes.union(CommandWrites::of(op, call, parts)),
            Callee::Air { .. } | Callee::Llvm { .. } => {}
            Callee::Indirect { .. } | Callee::UnknownIntrinsic(_) | Callee::Unbound { .. } => {
                return CommandWrites::ANY
            }
            Callee::Direct(f) => {
                if declared(f) && !ep.is_control_point_function(f) {
                    return CommandWrites::ANY;
                }
            }
        }
    }
    writes
}
