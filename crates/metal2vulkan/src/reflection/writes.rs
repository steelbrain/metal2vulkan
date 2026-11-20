//! Whether a function may store through the object bound at one of its argument-table slots.
//!
//! The question a consumer asks of a state it is about to bind resources to: *may this function
//! write the object at this slot?* A slot the function cannot store through holds a resource no
//! draw or dispatch of the state can dirty, so a read-only buffer may sit there.
//!
//! It is answered from the entry's AIR metadata ABI alone: each parameter's role, the access its
//! declaration states (`air.read`, `air.write`, `air.read_write`, `air.sample`) and the slot it
//! binds (`air.location_index`). The declared access is the whole of it; the address space beside
//! it (`air.address_space`) is not read, and needs not be, since Apple's compiler writes
//! `air.read` for a `constant` or `const device` buffer and `air.read_write` for a `device` one.
//!
//! It is the same field the reflection derives [`Binding::access`](super::Binding::access) from,
//! so the two answers agree by construction, and
//! `writes_tests::every_fixture_kernel_answers_and_agrees_with_its_reflection` holds them
//! together over the recorded fixtures. The values are Metal's `MTLBinding.access` (the
//! measurements are
//! [`reflection`](super)'s): a `device` buffer is read-write whether the function reads it, writes
//! it or both; a `constant` or `const device` buffer is read-only; a texture read or sampled is
//! read-only, one written write-only, one `read_write` read-write; samplers, function tables,
//! acceleration structures and imageblocks are read-only; threadgroup memory is read-write; an
//! object function's payload is read-write and a mesh function's read-only.
//!
//! It derives **no type description**: not a pointee's data type, struct members, array strides,
//! sizes or alignments. That is the whole difference from a reflection, and the reason this
//! question is answerable where a reflection is refused. A reflection must give Metal's
//! `MTLType` for every binding, and refuses the function where Metal's answer for one argument's
//! *type* has not been measured (an argument buffer's member the AIR gives no argument index, a
//! pipeline's stride for an array of structs holding a struct, a type name the data-type table
//! does not know). None of those describe an access, so none of them need stop this answer: a
//! function whose reflection is refused still states, for every argument, what it may do to it.
//!
//! So a state that compiles always answers, and the answer is per slot: an argument whose own
//! access or slot cannot be determined is one [`SlotWrite`], naming the parameter and the
//! [`SlotWriteGap`] that left it undetermined, and the other arguments keep their answers. No
//! uncertainty is spread over the rest of the state, and an undetermined slot never reads as
//! writable nor as not writable: [`SlotWrite::may_write`] answers
//! [`SlotWritability::Unknown`] for it.

use crate::air::abi::{
    AbiInt, AccelerationStructureKind, Access, EntryPoint, ImageblockKind, ParamBinding, ParamRole,
    Slot, Stage,
};
use crate::metal::{BindingAccess, BindingType};

/// Whether a function may store through the object bound at one argument-table slot.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SlotWritability {
    /// The function cannot store through the slot: its declaration is read-only, so no store it
    /// performs can land on the object bound there (a `constant` or `const device` buffer, a
    /// texture declared `access::read` or `access::sample`, a sampler, a visible-function or
    /// intersection-function table, an acceleration structure, an imageblock, or a mesh
    /// function's payload).
    Never,
    /// The function may store through the slot: its declaration is write-only or read-write (a
    /// `device` buffer, which Metal answers read-write whether the function reads it, writes it
    /// or neither, a texture declared `access::write` or `access::read_write`, threadgroup
    /// memory, or an object function's payload).
    Maybe,
    /// Undetermined for this slot alone, for the reason named. It is neither [`Never`] nor
    /// [`Maybe`]: a caller that must be safe treats it as writable, and one that reports gaps
    /// reports this one.
    ///
    /// [`Never`]: SlotWritability::Never
    /// [`Maybe`]: SlotWritability::Maybe
    Unknown(SlotWriteGap),
}

impl SlotWritability {
    /// Whether the answer is [`SlotWritability::Never`]: the one answer that lets a caller bind a
    /// resource it must not let the state write.
    pub fn is_never(self) -> bool {
        matches!(self, SlotWritability::Never)
    }

    /// The gap that left the answer undetermined; `None` for a determined one.
    pub fn gap(self) -> Option<SlotWriteGap> {
        match self {
            SlotWritability::Unknown(gap) => Some(gap),
            SlotWritability::Never | SlotWritability::Maybe => None,
        }
    }
}

/// Why one slot's writability is undetermined.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SlotWriteGap {
    /// Only a function constant states the argument's slot or the length of its span
    /// (`air.location_index` naming a global, `[[buffer(base + 2)]]`), and the values the entry
    /// was specialized with did not fold it to a literal. Which slots the argument binds is
    /// therefore not known, so every slot of its binding type answers `Unknown` rather than let
    /// one read as `Never` that the argument may cover.
    UnresolvedIndex,
    /// A `[[payload]]` argument of a stage that is neither an object nor a mesh function. Metal's
    /// access for one has been measured in those two stages alone (read-write in the object
    /// function that fills it, read-only in the mesh function that reads it).
    UnmeasuredPayloadStage,
    /// The argument's role is one only an argument-buffer member takes: an inline constant, an
    /// indirect command buffer, a render or compute pipeline state, or a depth-stencil state. A
    /// top-level parameter of such a role is a shape no measurement covers, so what Metal binds
    /// it to and what it answers for it are both unknown.
    MemberRole,
}

/// What a function may do to the object bound at one of its argument-table slots: one per
/// argument that binds a slot, in the order the function declares its arguments, as
/// [`RenderPipelineDescriptor::slot_writes`] and its compute, mesh and tile forms answer.
///
/// [`RenderPipelineDescriptor::slot_writes`]: crate::pipeline::RenderPipelineDescriptor::slot_writes
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SlotWrite {
    /// The argument's name (`air.arg_name`), as its reflection names it; empty for an argument
    /// the metadata leaves unnamed.
    pub name: String,
    /// The argument's parameter index in the entry's signature, which names it whatever else
    /// about it is undetermined.
    pub parameter: u32,
    /// The kind of binding the argument is, as [`Binding::binding_type`](super::Binding::binding_type)
    /// answers it; `None` where the role determines none ([`SlotWriteGap::MemberRole`]).
    pub binding_type: Option<BindingType>,
    /// The argument-table index the argument binds (`air.location_index`), `u32::MAX` for a
    /// payload and 0 for an imageblock, as Metal answers; `None` where only a function constant
    /// states it ([`SlotWriteGap::UnresolvedIndex`]).
    pub index: Option<u32>,
    /// The number of consecutive slots the argument binds from [`index`](Self::index), 1 for a
    /// single argument and `N` for an `array<T, N>` of textures or samplers; `None` where only a
    /// function constant states it.
    pub count: Option<u32>,
    /// Whether the function references the argument (`air.arg_unused` unset), as
    /// [`Binding::is_used`](super::Binding::is_used) answers it. An unused argument still binds
    /// its slot, and its declaration still states what a store through it could do, so it is
    /// answered like any other.
    pub used: bool,
    /// The argument's declared access, exactly as [`Binding::access`](super::Binding::access)
    /// answers it for the same argument (Metal's `MTLBindingAccess`); `None` where the
    /// declaration states none, which is where [`writes`](Self::writes) is
    /// [`SlotWritability::Unknown`] for a reason other than the slot
    /// ([`SlotWriteGap::UnmeasuredPayloadStage`], [`SlotWriteGap::MemberRole`]).
    ///
    /// It is stated even where the slots the argument binds are not
    /// ([`SlotWriteGap::UnresolvedIndex`]): what the function may do to the argument is known,
    /// and only which slots it applies to is not.
    pub access: Option<BindingAccess>,
    /// What the function may do to the object bound at the slot.
    pub writes: SlotWritability,
}

impl SlotWrite {
    /// Whether a function whose arguments answer `slots` may store through the object bound at
    /// `binding_type` slot `index`.
    ///
    /// [`SlotWritability::Maybe`] where an argument that covers the slot may be stored through,
    /// [`SlotWritability::Unknown`] where none may but one is undetermined or binds a slot the
    /// entry does not state, and [`SlotWritability::Never`] otherwise, a slot no argument covers
    /// included (nothing binds it, so nothing stores through it).
    ///
    /// An argument whose slot is undetermined ([`SlotWriteGap::UnresolvedIndex`]) may be the one
    /// at `index`, so it leaves every slot of its binding type `Unknown` rather than letting one
    /// read as `Never`.
    pub fn may_write(
        slots: &[SlotWrite],
        binding_type: BindingType,
        index: u32,
    ) -> SlotWritability {
        let mut gap = None;
        for s in slots {
            // An argument of another binding type covers no slot of this one. One whose role
            // determines no binding type at all (`MemberRole`) is not ruled out: what Metal
            // binds it to is unknown, so it may be this slot.
            if s.binding_type.is_some_and(|t| t != binding_type) {
                continue;
            }
            let covers = match (s.index, s.count, s.binding_type) {
                (Some(first), Some(count), Some(_)) => {
                    index >= first && index - first < count.max(1)
                }
                // A slot or span the metadata leaves open may be the one asked for.
                _ => true,
            };
            if !covers {
                continue;
            }
            match s.writes {
                SlotWritability::Maybe => return SlotWritability::Maybe,
                SlotWritability::Unknown(g) => gap = gap.or(Some(g)),
                SlotWritability::Never => {}
            }
        }
        match gap {
            Some(gap) => SlotWritability::Unknown(gap),
            None => SlotWritability::Never,
        }
    }
}

/// The slots of each stage of one pipeline, as the pipeline descriptors' `slot_writes` answer:
/// a stage the pipeline does not have has none.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SlotWrites {
    pub(crate) kernel: Vec<SlotWrite>,
    pub(crate) vertex: Vec<SlotWrite>,
    pub(crate) fragment: Vec<SlotWrite>,
    pub(crate) tile: Vec<SlotWrite>,
    pub(crate) object: Vec<SlotWrite>,
    pub(crate) mesh: Vec<SlotWrite>,
}

impl SlotWrites {
    /// The kernel's arguments; none for a pipeline without a compute stage.
    pub fn kernel_slots(&self) -> &[SlotWrite] {
        &self.kernel
    }

    /// The vertex function's arguments; none for a pipeline without a vertex stage. A vertex
    /// descriptor's buffers are not among them: they bind no argument of the function, and a
    /// pipeline never stores through one.
    pub fn vertex_slots(&self) -> &[SlotWrite] {
        &self.vertex
    }

    /// The fragment function's arguments; none for a pipeline without a fragment stage.
    pub fn fragment_slots(&self) -> &[SlotWrite] {
        &self.fragment
    }

    /// The mesh function's arguments; none for a pipeline without a mesh stage.
    pub fn mesh_slots(&self) -> &[SlotWrite] {
        &self.mesh
    }

    /// Every stage's answers, in the order the stages are listed above.
    pub fn stages(&self) -> [&[SlotWrite]; 6] {
        [
            &self.kernel,
            &self.vertex,
            &self.fragment,
            &self.tile,
            &self.object,
            &self.mesh,
        ]
    }
}

/// `MTLBindingAccess` of a declared access, as `derive::binding_access` answers it for the same
/// argument's reflection.
fn declared(access: Access) -> BindingAccess {
    match access {
        Access::Read | Access::Sample => BindingAccess::ReadOnly,
        Access::Write => BindingAccess::WriteOnly,
        Access::ReadWrite => BindingAccess::ReadWrite,
    }
}

/// Whether a store may land through a binding of declared access `access`.
fn writability(access: BindingAccess) -> SlotWritability {
    match access {
        BindingAccess::ReadOnly => SlotWritability::Never,
        BindingAccess::WriteOnly | BindingAccess::ReadWrite => SlotWritability::Maybe,
    }
}

/// An index or count the metadata states, `None` for one only a function constant gives.
fn stated(v: AbiInt) -> Option<u32> {
    match v {
        AbiInt::Const(x) => Some(x),
        AbiInt::Global(_) => None,
    }
}

/// What entry `ep` may do to each slot its arguments bind, in declaration order: the arguments
/// that bind an argument-table slot, builtins and stage inputs left out as a reflection leaves
/// them out. Infallible by construction, since it reads each parameter's role, declared access
/// and slot alone and derives no type description.
pub(crate) fn entry_slot_writes(ep: &EntryPoint) -> Vec<SlotWrite> {
    let mut out = Vec::new();
    for p in &ep.params {
        if let Some(w) = slot_write(ep, p) {
            out.push(w);
        }
    }
    out
}

/// The answer for parameter `p` of `ep`; `None` for a parameter that binds no argument-table slot.
fn slot_write(ep: &EntryPoint, p: &ParamBinding) -> Option<SlotWrite> {
    // An index and count a slot states, with the span of a single argument where the metadata
    // gives no count.
    let at = |slot: Slot| (stated(slot.index), stated(slot.count).map(|c| c.max(1)));
    let (binding_type, index, count, access) = match &p.role {
        ParamRole::Buffer(b) | ParamRole::ArgumentBuffer(b) => {
            let (index, count) = at(b.location);
            (
                Some(BindingType::Buffer),
                index,
                count,
                Ok(declared(b.access)),
            )
        }
        ParamRole::Threadgroup(b) => {
            let (index, count) = at(b.location);
            (
                Some(BindingType::ThreadgroupMemory),
                index,
                count,
                Ok(declared(b.access)),
            )
        }
        ParamRole::Texture(t) => {
            let (index, count) = at(t.location);
            (
                Some(BindingType::Texture),
                index,
                count,
                Ok(declared(t.access)),
            )
        }
        ParamRole::Sampler { location } => {
            let (index, count) = at(*location);
            (
                Some(BindingType::Sampler),
                index,
                count,
                Ok(BindingAccess::ReadOnly),
            )
        }
        ParamRole::FunctionTable(o) => {
            let (index, count) = at(o.location);
            (
                Some(BindingType::VisibleFunctionTable),
                index,
                count,
                Ok(BindingAccess::ReadOnly),
            )
        }
        ParamRole::IntersectionTable(o) => {
            let (index, count) = at(o.location);
            (
                Some(BindingType::IntersectionFunctionTable),
                index,
                count,
                Ok(BindingAccess::ReadOnly),
            )
        }
        ParamRole::AccelerationStructure { kind, arg } => {
            let (index, count) = at(arg.location);
            let binding_type = match kind {
                AccelerationStructureKind::Instance => BindingType::InstanceAccelerationStructure,
                AccelerationStructureKind::Primitive => BindingType::PrimitiveAccelerationStructure,
            };
            (
                Some(binding_type),
                index,
                count,
                Ok(BindingAccess::ReadOnly),
            )
        }
        // An imageblock binds index 0 and is read-only, whichever the function does with it, as
        // Metal answers.
        ParamRole::Imageblock(ib) => {
            let binding_type = match ib.kind {
                ImageblockKind::Implicit | ImageblockKind::Explicit => BindingType::Imageblock,
                ImageblockKind::Data => BindingType::ImageblockData,
            };
            (
                Some(binding_type),
                Some(0),
                Some(1),
                Ok(BindingAccess::ReadOnly),
            )
        }
        // An object function fills its payload and a mesh function reads it; Metal's access for
        // one in any other stage is unmeasured.
        ParamRole::Payload { .. } => {
            let access = match ep.stage {
                Stage::Object { .. } => Ok(BindingAccess::ReadWrite),
                Stage::Mesh { .. } => Ok(BindingAccess::ReadOnly),
                Stage::Kernel { .. } | Stage::Vertex { .. } | Stage::Fragment { .. } => {
                    Err(SlotWriteGap::UnmeasuredPayloadStage)
                }
            };
            (
                Some(BindingType::ObjectPayload),
                Some(u32::MAX),
                Some(1),
                access,
            )
        }
        // Builtins, stage inputs, the mesh grid and the mesh output bind no argument-table slot.
        ParamRole::Builtin(_)
        | ParamRole::StageIn { .. }
        | ParamRole::VertexInput { .. }
        | ParamRole::FragmentInput(_)
        | ParamRole::ColorInput { .. }
        | ParamRole::PatchInput(_)
        | ParamRole::MeshGridProperties
        | ParamRole::Mesh(_) => return None,
        // Roles of argument-buffer members only.
        ParamRole::IndirectConstant { .. }
        | ParamRole::CommandBuffer(_)
        | ParamRole::ComputePipelineState(_)
        | ParamRole::RenderPipelineState(_)
        | ParamRole::DepthStencilState(_) => (None, None, None, Err(SlotWriteGap::MemberRole)),
    };
    // The declaration states the access; a slot or span only a function constant states leaves
    // which slots that access applies to unknown.
    let writes = match (access, index, count) {
        (Err(gap), _, _) => SlotWritability::Unknown(gap),
        (Ok(access), Some(_), Some(_)) => writability(access),
        (Ok(_), _, _) => SlotWritability::Unknown(SlotWriteGap::UnresolvedIndex),
    };
    Some(SlotWrite {
        name: p.name.as_ref().map(ToString::to_string).unwrap_or_default(),
        parameter: p.param,
        binding_type,
        index,
        count,
        used: !p.unused,
        access: access.ok(),
        writes,
    })
}
