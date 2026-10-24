//! Metal's reflection of functions and pipelines: what each argument binds, in Metal's shape
//! (`MTLFunctionReflection`, `MTLComputePipelineReflection`, `MTLRenderPipelineReflection`,
//! `id<MTLBinding>` and its buffer, texture and threadgroup kinds, and the `MTLType` family that
//! describes a buffer's contents).
//!
//! [`Function::reflection`](crate::library::Function::reflection) answers for a library's
//! function, as `-[MTLLibrary reflectionForFunctionWithName:]` does,
//! [`Function::properties`](crate::library::Function::properties) and
//! [`Function::function_constants_dictionary`](crate::library::Function::function_constants_dictionary)
//! what `MTLFunction` states about it ([`FunctionProperties`] lists the rules), and each
//! pipeline
//! descriptor's `reflection` ([`ComputePipelineDescriptor::reflection`],
//! [`RenderPipelineDescriptor::reflection`], [`TileRenderPipelineDescriptor::reflection`]) for the
//! pipeline it describes, as the `reflection:` out-parameter of Metal's creation calls does (the
//! runtime's `_with_reflection` creation forms return it). Every answer comes from the functions'
//! AIR metadata and the descriptor, never from a guess: a binding whose shape Metal's answer has
//! not been measured for is refused (`Error::Unsupported` with [`Feature::Reflection`], naming the
//! argument's type).
//!
//! What Metal answers, measured on an M3 Max (macOS 26.7) and an M4 (macOS 27.0.1), identically,
//! over the fixtures in `validation/fixtures/public/reflection/` (their recorded answers are
//! `metal-answers.txt` there, which the tests compare against line for line):
//!
//! - **Presence.** A metallib function answers only when the library carries its reflection (the
//!   function's `RFLT` tag, which the compiler driver writes and `metallib` of `.air` files does
//!   not); otherwise Metal answers nil. A visible function answers no bindings.
//! - **Order.** Bindings come in the order the function declares its arguments. A library lists
//!   the ones function constants gate; a pipeline leaves out the ones its constants gate off. An
//!   argument the function never references is listed with [`Binding::is_used`] false.
//! - **Access.** A device buffer is read-write even when the function only reads or only writes
//!   it; a constant or `const` buffer is read-only. A texture read or sampled is read-only, one
//!   written is write-only, `read_write` is read-write. Samplers, function tables and
//!   acceleration structures are read-only; threadgroup memory is read-write. An imageblock, a
//!   tile function's implicit one and an explicit one in a tile function or a compute kernel, is
//!   an `Imageblock` binding at index 0, read-only, and a fragment function's imageblock data an
//!   `ImageblockData` binding at index 0, read-only.
//! - **Indices.** An argument's index is its `air.location_index`: a literal, also after
//!   arguments a function constant gates (Apple's compiler numbers implicit indices in declaration
//!   order whatever the constant). One a function constant computes (`[[buffer(base + 2)]]`) is
//!   answered by a pipeline from the specialized entry; a library's answer is refused, since
//!   Metal's states values the constant never gives (28, 4294967236 and 18 for `base`,
//!   `base + 2` and `base * 2`, on both chips).
//! - **Types.** A buffer's data type is `Struct` when its pointee is a struct (an atomic pointee is
//!   a struct with one member `__s`), else the pointee's scalar, vector or matrix type; a packed
//!   vector reports its unpacked type. Every buffer states itself as a pointer. A texture array
//!   argument (`array<texture2d<T>, N>`) has array length `N`; a `texture2d_array` has array
//!   length 1.
//! - **Argument buffers.** A member's argument index is its `[[id]]` (implicit ids count up in
//!   declaration order), relative to the struct that holds it; an array member's argument index
//!   stride is the number of ids one element takes. An array's stride is its element size, except
//!   that a pipeline's answer gives an array of textures, samplers or pointers, or of structs
//!   whose members are all such resources, stride 0; a struct array with a constant member keeps
//!   its element size, and a pipeline's struct array whose struct holds no constant but a struct
//!   is refused, its stride unmeasured. A render or compute pipeline state member is
//!   `RenderPipeline` or `ComputePipeline` data, and a visible function table member
//!   `VisibleFunctionTable` data, an array of them strided as resources. An intersection
//!   function table or an acceleration structure member is ray tracing, refused.
//! - **Descriptor buffers.** A pipeline's vertex descriptor (a compute pipeline's stage input
//!   descriptor) adds, before the function's bindings, one read-only buffer binding per buffer
//!   its attributes read, named `vertexBuffer.N` (`stageInBuffer.N`) with `N` counting from 0 in
//!   buffer index order: its index is the buffer index, its data size the layout's stride
//!   (`NSUIntegerMax` for a dynamic stride), its alignment 4, its data type `Struct` with no
//!   struct type, [`Binding::is_argument`] false, and [`Binding::is_used`] true exactly when one
//!   of its attributes is one the function reads: not one it declares and never reads, one a
//!   constant gates off, one it does not declare, or a post-tessellation function's control-point
//!   or patch input. An attribute whose buffer has no layout stops Metal at an assertion ("has no
//!   stride"), read or not and without API validation too: the crate's answer is the descriptor's
//!   error (`DescriptorError::VertexLayout`, `DescriptorError::StageInputLayout`).
//! - **Compiled stages.** A stage the compile layer already compiled (`StageSource::Compiled`)
//!   answers what its compile kept of the entry it compiled, the same as its function's stage.
//!
//! Not answered: tensor bindings and reference types (no tensor support), an object or mesh
//! function's payload (mesh functions do not decode), and `MTLFunctionReflection.userAnnotation`
//! (its AIR form is unmeasured).

mod derive;
mod function;
mod linked;
pub(crate) use linked::dropped_fragment_inputs;
pub use linked::{DirectArgumentUse, LinkedUseAnswers, LinkedUseGap, UnwrittenFragmentInputs};
mod live;
mod types;
mod writes;
pub(crate) use writes::entry_slot_writes;
pub use writes::{SlotWritability, SlotWrite, SlotWriteGap, SlotWrites};

#[cfg(test)]
mod linked_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod writes_tests;

pub(crate) use derive::{
    bindings, needs_resolution, read_attributes, specialization_decides_reads, specialized_reads,
    Channel, UnspecializedReads,
};
pub(crate) use function::{
    entry_properties, metallib_constants, metallib_properties, module_constants, visible_properties,
};
pub use function::{Attribute, FunctionConstant, FunctionProperties};

use std::collections::{BTreeMap, BTreeSet};

use crate::air;
use crate::budget::Meter;
use crate::error::{Error, Feature, InternalError, InternalKind, Unsupported};
use crate::library::function::FunctionSource;
use crate::library::Function;
use crate::metal::{
    AttributeFormat, BindingAccess, BindingType, DataType, TextureType, VertexFormat,
};
use crate::pipeline::{
    ComputePipelineDescriptor, MeshPipelineDescriptor, RenderPipelineDescriptor, StageSource,
    TileRenderPipelineDescriptor,
};

/// `MTLFunctionReflection`: a function's bindings as its library answers them.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FunctionReflection {
    bindings: Vec<Binding>,
    user_annotation: Option<String>,
}

impl FunctionReflection {
    /// `bindings`, in the order the function declares its arguments.
    pub fn bindings(&self) -> &[Binding] {
        &self.bindings
    }

    /// `userAnnotation`: the string the function's `[[user_annotation("...")]]` passes
    /// (`air.user_annotation`), `None` (nil) for a function without one, as Metal answers on the
    /// M3 Max and the M4.
    pub fn user_annotation(&self) -> Option<&str> {
        self.user_annotation.as_deref()
    }
}

/// The reflection of function `f` from its AIR (whether its library carries one is the caller's
/// question): no bindings nor annotation for a visible function, else its entry's arguments and
/// user annotation.
pub(crate) fn function_reflection(
    f: &Function,
    m: &mut Meter,
) -> Result<FunctionReflection, Error> {
    let answer = |ep: &air::abi::EntryPoint, structs: &[air::abi::StructInfo]| {
        Ok(FunctionReflection {
            bindings: bindings(ep, structs, Channel::Library, &BTreeSet::new())?,
            user_annotation: ep.user_annotation.as_ref().map(ToString::to_string),
        })
    };
    crate::pipeline::compile::guarded(|| match f.source()? {
        FunctionSource::Visible => Ok(FunctionReflection {
            bindings: Vec::new(),
            user_annotation: None,
        }),
        FunctionSource::Module { bitcode, index } => {
            let decoded = air::decode(bitcode, m)?;
            let ep = decoded.abi.entries.get(index as usize).ok_or_else(|| {
                Error::Internal(InternalError::new(
                    InternalKind::Invariant,
                    "a library entry outside its module",
                ))
            })?;
            answer(ep, &decoded.abi.structs)
        }
        FunctionSource::Metallib(lf) => {
            let mut decoded = air::decode(lf.bitcode, m)?;
            decoded.abi.check_container(&decoded.module, &lf, m)?;
            let ep = air::canon::select(&decoded.module, &decoded.abi, lf.name)?;
            answer(ep, &decoded.abi.structs)
        }
    })
}

/// `MTLComputePipelineReflection`: a compute pipeline's bindings.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComputePipelineReflection {
    bindings: Vec<Binding>,
}

impl ComputePipelineReflection {
    /// `bindings`: the stage input descriptor's buffers first, then the kernel's arguments in
    /// declaration order.
    pub fn bindings(&self) -> &[Binding] {
        &self.bindings
    }
}

/// `MTLRenderPipelineReflection`: a render pipeline's bindings, per stage. A stage the pipeline
/// does not have has none.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RenderPipelineReflection {
    pub(crate) vertex: Vec<Binding>,
    pub(crate) fragment: Vec<Binding>,
    pub(crate) tile: Vec<Binding>,
    pub(crate) object: Vec<Binding>,
    pub(crate) mesh: Vec<Binding>,
}

impl RenderPipelineReflection {
    /// `vertexBindings`: the vertex descriptor's buffers first, then the vertex function's
    /// arguments.
    pub fn vertex_bindings(&self) -> &[Binding] {
        &self.vertex
    }

    /// `fragmentBindings`.
    pub fn fragment_bindings(&self) -> &[Binding] {
        &self.fragment
    }

    /// `tileBindings`.
    pub fn tile_bindings(&self) -> &[Binding] {
        &self.tile
    }

    /// `objectBindings`: a mesh pipeline's object function's arguments; none for a pipeline
    /// without an object stage.
    pub fn object_bindings(&self) -> &[Binding] {
        &self.object
    }

    /// `meshBindings`: a mesh pipeline's mesh function's arguments; none for a pipeline without
    /// a mesh stage.
    pub fn mesh_bindings(&self) -> &[Binding] {
        &self.mesh
    }
}

/// What a pipeline's reflection takes of one stage: its bindings, and the attribute indices its
/// entry reads (`[[attribute(n)]]`), which decide whether a descriptor's buffer is used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StageReflection {
    pub bindings: Vec<Binding>,
    pub attributes: BTreeSet<u32>,
}

/// The reflection of stage `source` in a pipeline, its calls nesting `depth` deep through visible
/// functions. A stage the compile layer already compiled answers what its compile kept, from the
/// entry it compiled (`FunctionArtifact::reflection`); one whose answer was unmeasured kept none
/// and is refused (its library function's reflection names the binding).
fn stage(source: &StageSource, depth: u32, m: &mut Meter) -> Result<StageReflection, Error> {
    match source {
        StageSource::Function(fd) => crate::pipeline::compile::stage_reflection(fd, depth, m),
        StageSource::Compiled(c) => {
            c.artifact().reflection.clone().ok_or_else(|| {
                Error::Unsupported(Unsupported::not_implemented(Feature::Reflection))
            })
        }
    }
}

/// What stage `source`, its calls nesting `depth` deep, may store through each slot its arguments
/// bind ([`SlotWrite`]): from the entry its constants specialize where they decide a slot, else
/// from the decoded entry. A stage the compile layer already compiled answers what its compile
/// kept of the entry it compiled (`FunctionArtifact::slot_writes`, which every compile keeps),
/// decoding nothing.
///
/// It derives no type description, so no argument's unmeasured type refuses it: a stage that
/// compiles answers.
fn stage_writes(source: &StageSource, depth: u32, m: &mut Meter) -> Result<Vec<SlotWrite>, Error> {
    match source {
        StageSource::Function(fd) => crate::pipeline::compile::stage_slot_writes(fd, depth, m),
        StageSource::Compiled(c) => Ok(c.artifact().slot_writes.clone()),
    }
}

/// Whether stage `source`, its calls nesting `depth` deep, uses a direct texture or sampler
/// argument that its reflection lists as used: decided from the arguments' roles and use alone
/// (`compile::stage_uses_direct_arguments`), so an argument whose reflection Metal's answer has
/// not been measured for decides nothing. A stage the compile layer already compiled answers from
/// the reflection its compile kept, refused where it kept none.
pub(crate) fn stage_direct(source: &StageSource, depth: u32, m: &mut Meter) -> Result<bool, Error> {
    match source {
        StageSource::Function(fd) => {
            crate::pipeline::compile::stage_uses_direct_arguments(fd, depth, m)
        }
        StageSource::Compiled(_) => Ok(stage(source, depth, m)?.bindings.iter().any(|binding| {
            binding.is_used()
                && matches!(
                    binding.binding_type(),
                    BindingType::Texture | BindingType::Sampler
                )
        })),
    }
}

/// The buffers a vertex or stage input descriptor adds before a stage's own bindings, named
/// `<prefix>.N`: one per buffer a set attribute (`(attribute index, buffer index)` in
/// `attributes`; the caller leaves out those of format `Invalid`, whose buffers Metal's
/// reflection does not list) reads, in buffer index order, its data size its layout's stride
/// (`strides`, by buffer index; `None` for a dynamic stride, whose data size is `u64::MAX`, Metal's
/// `NSUIntegerMax`), used when one of its attributes is one the stage reads (`read`). The caller
/// has checked the layouts (`check_layouts`): Metal stops at an assertion where a set attribute's
/// buffer "has no stride", whether the stage reads the attribute or not, without API validation
/// too.
fn descriptor_buffers(
    prefix: &str,
    attributes: impl IntoIterator<Item = (u32, u32)>,
    strides: &BTreeMap<u32, Option<u32>>,
    read: &BTreeSet<u32>,
) -> Vec<Binding> {
    let mut used: BTreeMap<u32, bool> = BTreeMap::new();
    for (attribute, buffer) in attributes {
        *used.entry(buffer).or_default() |= read.contains(&attribute);
    }
    let mut out = Vec::with_capacity(used.len());
    for (n, (index, used)) in used.into_iter().enumerate() {
        // Every buffer has a layout (checked), so `None` is a dynamic stride.
        let stride = strides
            .get(&index)
            .copied()
            .flatten()
            .map_or(u64::MAX, u64::from);
        let pointer = PointerType::new(
            DataType::Struct,
            BindingAccess::ReadOnly,
            4,
            stride,
            false,
            TypeDetail::Plain,
        );
        let buffer = BufferBinding::new(4, stride, DataType::Struct, None, Some(pointer));
        out.push(Binding::new(
            format!("{prefix}.{n}"),
            BindingType::Buffer,
            BindingAccess::ReadOnly,
            index,
            used,
            false,
            BindingKind::Buffer(buffer),
        ));
    }
    out
}

impl ComputePipelineDescriptor {
    /// The reflection of the pipeline this descriptor describes, as the `reflection:`
    /// out-parameter of `newComputePipelineStateWithDescriptor:options:reflection:error:` answers
    /// it (the rules are [`reflection`](crate::reflection)'s): the stage input descriptor's
    /// buffers, then the kernel's arguments, of the entry the kernel's constants specialize where
    /// they decide one. It decodes the kernel's AIR under `budget`; a kernel the compile layer
    /// already compiled (`StageSource::Compiled`) answers what its compile kept of the same entry,
    /// decoding nothing.
    ///
    /// Refused: an argument whose answer Metal's has not been measured for (`Unsupported` with
    /// `Feature::Reflection`), a stage input attribute whose buffer has no layout
    /// (`InvalidDescriptor(StageInputLayout)`, where Metal stops at an assertion), and the
    /// decode's and specialization's own errors.
    pub fn reflection(&self, budget: &crate::Budget) -> Result<ComputePipelineReflection, Error> {
        let mut m = Meter::new(budget);
        let kernel = stage(&self.kernel, self.max_call_stack_depth, &mut m)?;
        let mut bindings = match &self.stage_input {
            Some(s) => {
                s.check_layouts()?;
                descriptor_buffers(
                    "stageInBuffer",
                    s.attributes
                        .iter()
                        .filter(|(_, a)| a.format != AttributeFormat::Invalid)
                        .map(|(&i, a)| (i, a.buffer_index)),
                    &s.layouts.iter().map(|(&b, l)| (b, l.stride)).collect(),
                    &kernel.attributes,
                )
            }
            None => Vec::new(),
        };
        bindings.extend(kernel.bindings);
        Ok(ComputePipelineReflection { bindings })
    }

    /// Whether the kernel uses a direct texture or sampler argument (`[[texture(n)]]`,
    /// `[[sampler(n)]]`, a texture array among them) that the pipeline's
    /// [`reflection`](Self::reflection) lists as used, of the entry its constants specialize where
    /// they decide one: what `newComputePipelineStateWithDescriptor:` refuses with
    /// `supportIndirectCommandBuffers`. It decides from the arguments' roles and use alone, so an
    /// argument whose reflection Metal's answer has not been measured for, which `reflection`
    /// refuses, decides nothing here. It decodes the kernel's AIR under `budget`; a kernel the
    /// compile layer already compiled (`StageSource::Compiled`) answers from what its compile kept.
    ///
    /// Refused: a compiled kernel that kept no reflection (`Unsupported` with
    /// `Feature::Reflection`), and the decode's and specialization's own errors.
    pub fn uses_direct_arguments(&self, budget: &crate::Budget) -> Result<bool, Error> {
        let mut m = Meter::new(budget);
        stage_direct(&self.kernel, self.max_call_stack_depth, &mut m)
    }

    /// What the kernel may store through each slot its arguments bind
    /// ([`writes`](crate::reflection::SlotWrite), whose rules are
    /// [`reflection`](crate::reflection)'s), of the entry its constants specialize where they
    /// decide a slot: the answers in [`SlotWrites::kernel_slots`]. The stage input descriptor's
    /// buffers are not among them, since they bind no argument of the kernel. It decodes the
    /// kernel's AIR under `budget`; a kernel the compile layer already compiled answers what its
    /// compile kept, decoding nothing.
    ///
    /// Unlike [`reflection`](Self::reflection) it derives no type description, so an argument
    /// whose *type* Metal's answer has not been measured for does not refuse it: a kernel that
    /// compiles answers, and an argument whose own slot or access is undetermined is one
    /// [`SlotWritability::Unknown`] answer among determined ones. Refused only by the decode's
    /// and specialization's own errors.
    pub fn slot_writes(&self, budget: &crate::Budget) -> Result<SlotWrites, Error> {
        let mut m = Meter::new(budget);
        Ok(SlotWrites {
            kernel: stage_writes(&self.kernel, self.max_call_stack_depth, &mut m)?,
            ..SlotWrites::default()
        })
    }
}

impl RenderPipelineDescriptor {
    /// The reflection of the drawing pipeline this descriptor describes, as the `reflection:`
    /// out-parameter of `newRenderPipelineStateWithDescriptor:options:reflection:error:` answers
    /// it: the vertex descriptor's buffers, then the vertex function's arguments, and the fragment
    /// function's, each of the entry its constants specialize where they decide one. It decodes the
    /// functions' AIR under `budget`. Refused as [`ComputePipelineDescriptor::reflection`] refuses,
    /// a vertex attribute whose buffer has no layout as `InvalidDescriptor(VertexLayout)`.
    pub fn reflection(&self, budget: &crate::Budget) -> Result<RenderPipelineReflection, Error> {
        let mut m = Meter::new(budget);
        let vertex_stage = stage(&self.vertex, self.max_vertex_call_stack_depth, &mut m)?;
        let mut vertex = match &self.vertex_descriptor {
            Some(d) => {
                d.check_layouts()?;
                descriptor_buffers(
                    "vertexBuffer",
                    d.attributes
                        .iter()
                        .filter(|(_, a)| a.format != VertexFormat::Invalid)
                        .map(|(&i, a)| (i, a.buffer_index)),
                    &d.layouts.iter().map(|(&b, l)| (b, l.stride)).collect(),
                    &vertex_stage.attributes,
                )
            }
            None => Vec::new(),
        };
        vertex.extend(vertex_stage.bindings);
        let fragment = match &self.fragment {
            Some(f) => stage(f, self.max_fragment_call_stack_depth, &mut m)?.bindings,
            None => Vec::new(),
        };
        Ok(RenderPipelineReflection {
            vertex,
            fragment,
            ..RenderPipelineReflection::default()
        })
    }

    /// What each stage of the drawing pipeline this descriptor describes may store through each
    /// slot its arguments bind ([`ComputePipelineDescriptor::slot_writes`]'s answer per stage):
    /// [`SlotWrites::vertex_slots`] and [`SlotWrites::fragment_slots`]. The vertex descriptor's
    /// buffers are not among the vertex answers, since they bind no argument of the function.
    ///
    /// This is the question [`reflection`](Self::reflection) also answers, in
    /// [`Binding::access`], without the type description that makes a reflection refuse a
    /// function whose argument types Metal's answer has not been measured for. A state that
    /// compiles answers here. Refused only by the decode's and specialization's own errors.
    pub fn slot_writes(&self, budget: &crate::Budget) -> Result<SlotWrites, Error> {
        let mut m = Meter::new(budget);
        let vertex = stage_writes(&self.vertex, self.max_vertex_call_stack_depth, &mut m)?;
        let fragment = match &self.fragment {
            Some(f) => stage_writes(f, self.max_fragment_call_stack_depth, &mut m)?,
            None => Vec::new(),
        };
        Ok(SlotWrites {
            vertex,
            fragment,
            ..SlotWrites::default()
        })
    }
}

impl MeshPipelineDescriptor {
    /// The reflection of the mesh pipeline this descriptor describes, as the `reflection:`
    /// out-parameter of `newRenderPipelineStateWithMeshDescriptor:options:reflection:error:`
    /// answers it: the object function's arguments where it has one, the mesh function's, and the
    /// fragment function's where it rasterizes, an object or mesh function's payload among them;
    /// none for the vertex and tile stages. Each stage is the entry its constants specialize where
    /// they decide one, its calls nesting Metal's default depth of 1, as
    /// `Device::compile_mesh_pipeline` compiles it. It decodes the functions' AIR under `budget`.
    /// Refused as [`ComputePipelineDescriptor::reflection`] refuses.
    pub fn reflection(&self, budget: &crate::Budget) -> Result<RenderPipelineReflection, Error> {
        let mut m = Meter::new(budget);
        let object = match &self.object {
            Some(o) => stage(o, 1, &mut m)?.bindings,
            None => Vec::new(),
        };
        let mesh = stage(&self.mesh, 1, &mut m)?.bindings;
        let fragment = match &self.fragment {
            Some(f) => stage(f, 1, &mut m)?.bindings,
            None => Vec::new(),
        };
        Ok(RenderPipelineReflection {
            fragment,
            object,
            mesh,
            ..RenderPipelineReflection::default()
        })
    }

    /// What each stage of the mesh pipeline this descriptor describes may store through each slot
    /// its arguments bind ([`ComputePipelineDescriptor::slot_writes`]'s answer per stage):
    /// [`SlotWrites::object_slots`], [`SlotWrites::mesh_slots`] and
    /// [`SlotWrites::fragment_slots`], an object or mesh function's payload among them. Each
    /// stage's calls nest Metal's default depth of 1, as `Device::compile_mesh_pipeline` compiles
    /// it. Refused only by the decode's and specialization's own errors.
    pub fn slot_writes(&self, budget: &crate::Budget) -> Result<SlotWrites, Error> {
        let mut m = Meter::new(budget);
        let object = match &self.object {
            Some(o) => stage_writes(o, 1, &mut m)?,
            None => Vec::new(),
        };
        let mesh = stage_writes(&self.mesh, 1, &mut m)?;
        let fragment = match &self.fragment {
            Some(f) => stage_writes(f, 1, &mut m)?,
            None => Vec::new(),
        };
        Ok(SlotWrites {
            fragment,
            object,
            mesh,
            ..SlotWrites::default()
        })
    }
}

impl TileRenderPipelineDescriptor {
    /// The reflection of the tile pipeline this descriptor describes, as the `reflection:`
    /// out-parameter of `newRenderPipelineStateWithTileDescriptor:options:reflection:error:`
    /// answers it: the tile function's arguments, a tile function's implicit imageblock among them.
    /// It decodes the function's AIR under `budget`. Refused as
    /// [`ComputePipelineDescriptor::reflection`] refuses.
    pub fn reflection(&self, budget: &crate::Budget) -> Result<RenderPipelineReflection, Error> {
        let mut m = Meter::new(budget);
        let tile = stage(&self.tile, self.max_call_stack_depth, &mut m)?.bindings;
        Ok(RenderPipelineReflection {
            tile,
            ..RenderPipelineReflection::default()
        })
    }

    /// What the tile function of the pipeline this descriptor describes may store through each
    /// slot its arguments bind ([`ComputePipelineDescriptor::slot_writes`]'s answer for the tile
    /// stage): [`SlotWrites::tile_slots`], a tile function's implicit imageblock among them.
    /// Refused only by the decode's and specialization's own errors.
    pub fn slot_writes(&self, budget: &crate::Budget) -> Result<SlotWrites, Error> {
        let mut m = Meter::new(budget);
        Ok(SlotWrites {
            tile: stage_writes(&self.tile, self.max_call_stack_depth, &mut m)?,
            ..SlotWrites::default()
        })
    }
}

/// `id<MTLBinding>`: one argument a function or pipeline binds, with the fields of the kind it
/// conforms to ([`Binding::buffer`], [`Binding::threadgroup`], [`Binding::texture`]).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Binding {
    pub(crate) name: String,
    pub(crate) binding_type: BindingType,
    pub(crate) access: BindingAccess,
    pub(crate) index: u32,
    pub(crate) used: bool,
    pub(crate) argument: bool,
    pub(crate) kind: BindingKind,
}

/// The kind-specific fields of a binding (the protocol it conforms to beyond `MTLBinding`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) enum BindingKind {
    /// `MTLBinding` alone: a sampler, a function table, an acceleration structure, an
    /// imageblock.
    Plain,
    Buffer(BufferBinding),
    Threadgroup(ThreadgroupBinding),
    Texture(TextureBinding),
    ObjectPayload(ObjectPayloadBinding),
}

impl Binding {
    pub(crate) fn new(
        name: String,
        binding_type: BindingType,
        access: BindingAccess,
        index: u32,
        used: bool,
        argument: bool,
        kind: BindingKind,
    ) -> Binding {
        Binding {
            name,
            binding_type,
            access,
            index,
            used,
            argument,
            kind,
        }
    }

    /// `name`: the argument's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `type`.
    pub fn binding_type(&self) -> BindingType {
        self.binding_type
    }

    /// `access`.
    pub fn access(&self) -> BindingAccess {
        self.access
    }

    /// `index`: the argument table index.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// `isUsed`: whether the function references the argument.
    pub fn is_used(&self) -> bool {
        self.used
    }

    /// `isArgument`: whether the function declares the binding as an argument.
    pub fn is_argument(&self) -> bool {
        self.argument
    }

    /// The `MTLBufferBinding` fields, for a buffer.
    pub fn buffer(&self) -> Option<&BufferBinding> {
        match &self.kind {
            BindingKind::Buffer(b) => Some(b),
            _ => None,
        }
    }

    /// The `MTLThreadgroupBinding` fields, for threadgroup memory.
    pub fn threadgroup(&self) -> Option<&ThreadgroupBinding> {
        match &self.kind {
            BindingKind::Threadgroup(t) => Some(t),
            _ => None,
        }
    }

    /// The `MTLTextureBinding` fields, for a texture.
    pub fn texture(&self) -> Option<&TextureBinding> {
        match &self.kind {
            BindingKind::Texture(t) => Some(t),
            _ => None,
        }
    }

    /// The `MTLObjectPayloadBinding` fields, for an object or mesh function's payload.
    pub fn object_payload(&self) -> Option<&ObjectPayloadBinding> {
        match &self.kind {
            BindingKind::ObjectPayload(p) => Some(p),
            _ => None,
        }
    }
}

/// `MTLObjectPayloadBinding`'s fields: an object function's payload, which it writes and the mesh
/// functions it launches read (`[[payload]]`).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectPayloadBinding {
    pub(crate) alignment: u64,
    pub(crate) data_size: u64,
}

impl ObjectPayloadBinding {
    pub(crate) fn new(alignment: u64, data_size: u64) -> ObjectPayloadBinding {
        ObjectPayloadBinding {
            alignment,
            data_size,
        }
    }

    /// `objectPayloadAlignment`: the payload type's alignment.
    pub fn object_payload_alignment(&self) -> u64 {
        self.alignment
    }

    /// `objectPayloadDataSize`: the payload type's size.
    pub fn object_payload_data_size(&self) -> u64 {
        self.data_size
    }
}

/// `MTLBufferBinding`'s fields.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BufferBinding {
    pub(crate) alignment: u64,
    pub(crate) data_size: u64,
    pub(crate) data_type: DataType,
    pub(crate) struct_type: Option<StructType>,
    pub(crate) pointer_type: Option<PointerType>,
}

impl BufferBinding {
    pub(crate) fn new(
        alignment: u64,
        data_size: u64,
        data_type: DataType,
        struct_type: Option<StructType>,
        pointer_type: Option<PointerType>,
    ) -> BufferBinding {
        BufferBinding {
            alignment,
            data_size,
            data_type,
            struct_type,
            pointer_type,
        }
    }

    /// `bufferAlignment`: the pointee's alignment.
    pub fn buffer_alignment(&self) -> u64 {
        self.alignment
    }

    /// `bufferDataSize`: the pointee's size.
    pub fn buffer_data_size(&self) -> u64 {
        self.data_size
    }

    /// `bufferDataType`.
    pub fn buffer_data_type(&self) -> DataType {
        self.data_type
    }

    /// `bufferStructType`: the pointee's members, when it is a struct.
    pub fn buffer_struct_type(&self) -> Option<&StructType> {
        self.struct_type.as_ref()
    }

    /// `bufferPointerType`: the buffer as a pointer (every buffer the function declares has one).
    pub fn buffer_pointer_type(&self) -> Option<&PointerType> {
        self.pointer_type.as_ref()
    }
}

/// `MTLThreadgroupBinding`'s fields.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThreadgroupBinding {
    pub(crate) alignment: u64,
    pub(crate) data_size: u64,
}

impl ThreadgroupBinding {
    pub(crate) fn new(alignment: u64, data_size: u64) -> ThreadgroupBinding {
        ThreadgroupBinding {
            alignment,
            data_size,
        }
    }

    /// `threadgroupMemoryAlignment`: the element's alignment.
    pub fn threadgroup_memory_alignment(&self) -> u64 {
        self.alignment
    }

    /// `threadgroupMemoryDataSize`: the element's size.
    pub fn threadgroup_memory_data_size(&self) -> u64 {
        self.data_size
    }
}

/// `MTLTextureBinding`'s fields.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TextureBinding {
    pub(crate) texture_type: TextureType,
    pub(crate) data_type: DataType,
    pub(crate) depth: bool,
    pub(crate) array_length: u32,
}

impl TextureBinding {
    pub(crate) fn new(
        texture_type: TextureType,
        data_type: DataType,
        depth: bool,
        array_length: u32,
    ) -> TextureBinding {
        TextureBinding {
            texture_type,
            data_type,
            depth,
            array_length,
        }
    }

    /// `textureType`.
    pub fn texture_type(&self) -> TextureType {
        self.texture_type
    }

    /// `textureDataType`: the component type the texture is declared with.
    pub fn texture_data_type(&self) -> DataType {
        self.data_type
    }

    /// `isDepthTexture`.
    pub fn is_depth_texture(&self) -> bool {
        self.depth
    }

    /// `arrayLength`: `N` for `array<texture2d<T>, N>`, else 1.
    pub fn array_length(&self) -> u32 {
        self.array_length
    }
}

/// `MTLStructType`: a struct's members in declaration order.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StructType {
    pub(crate) members: Vec<StructMember>,
}

impl StructType {
    pub(crate) fn new(members: Vec<StructMember>) -> StructType {
        StructType { members }
    }

    /// `members`.
    pub fn members(&self) -> &[StructMember] {
        &self.members
    }

    /// `memberByName:`.
    pub fn member_by_name(&self, name: &str) -> Option<&StructMember> {
        self.members.iter().find(|m| m.name == name)
    }
}

/// `MTLStructMember`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StructMember {
    pub(crate) name: String,
    pub(crate) offset: u64,
    pub(crate) data_type: DataType,
    pub(crate) argument_index: u32,
    pub(crate) detail: TypeDetail,
}

/// What an `MTLStructMember` or an `MTLArrayType`'s element is beyond its data type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) enum TypeDetail {
    Plain,
    Struct(StructType),
    Array(Box<ArrayType>),
    TextureReference(TextureReferenceType),
    Pointer(Box<PointerType>),
}

impl StructMember {
    pub(crate) fn new(
        name: String,
        offset: u64,
        data_type: DataType,
        argument_index: u32,
        detail: TypeDetail,
    ) -> StructMember {
        StructMember {
            name,
            offset,
            data_type,
            argument_index,
            detail,
        }
    }

    /// `name`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `offset`: the member's byte offset in its struct.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// `dataType`.
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// `argumentIndex`: an argument-buffer member's `[[id]]`, relative to the struct that holds
    /// it; 0 outside argument buffers.
    pub fn argument_index(&self) -> u32 {
        self.argument_index
    }

    /// `structType`.
    pub fn struct_type(&self) -> Option<&StructType> {
        self.detail.struct_type()
    }

    /// `arrayType`.
    pub fn array_type(&self) -> Option<&ArrayType> {
        self.detail.array_type()
    }

    /// `textureReferenceType`.
    pub fn texture_reference_type(&self) -> Option<&TextureReferenceType> {
        self.detail.texture_reference_type()
    }

    /// `pointerType`.
    pub fn pointer_type(&self) -> Option<&PointerType> {
        self.detail.pointer_type()
    }
}

impl TypeDetail {
    fn struct_type(&self) -> Option<&StructType> {
        match self {
            TypeDetail::Struct(s) => Some(s),
            _ => None,
        }
    }

    fn array_type(&self) -> Option<&ArrayType> {
        match self {
            TypeDetail::Array(a) => Some(a),
            _ => None,
        }
    }

    fn texture_reference_type(&self) -> Option<&TextureReferenceType> {
        match self {
            TypeDetail::TextureReference(t) => Some(t),
            _ => None,
        }
    }

    fn pointer_type(&self) -> Option<&PointerType> {
        match self {
            TypeDetail::Pointer(p) => Some(p),
            _ => None,
        }
    }
}

/// `MTLArrayType`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArrayType {
    pub(crate) element_type: DataType,
    pub(crate) array_length: u32,
    pub(crate) stride: u64,
    pub(crate) argument_index_stride: u32,
    pub(crate) element: TypeDetail,
}

impl ArrayType {
    pub(crate) fn new(
        element_type: DataType,
        array_length: u32,
        stride: u64,
        argument_index_stride: u32,
        element: TypeDetail,
    ) -> ArrayType {
        ArrayType {
            element_type,
            array_length,
            stride,
            argument_index_stride,
            element,
        }
    }

    /// `elementType`.
    pub fn element_type(&self) -> DataType {
        self.element_type
    }

    /// `arrayLength`.
    pub fn array_length(&self) -> u32 {
        self.array_length
    }

    /// `stride`: the bytes between elements (0 for an argument buffer's array of resources in a
    /// pipeline's answer).
    pub fn stride(&self) -> u64 {
        self.stride
    }

    /// `argumentIndexStride`: the argument ids one element takes (0 outside argument buffers).
    pub fn argument_index_stride(&self) -> u32 {
        self.argument_index_stride
    }

    /// `elementStructType`.
    pub fn element_struct_type(&self) -> Option<&StructType> {
        self.element.struct_type()
    }

    /// `elementArrayType`.
    pub fn element_array_type(&self) -> Option<&ArrayType> {
        self.element.array_type()
    }

    /// `elementTextureReferenceType`.
    pub fn element_texture_reference_type(&self) -> Option<&TextureReferenceType> {
        self.element.texture_reference_type()
    }

    /// `elementPointerType`.
    pub fn element_pointer_type(&self) -> Option<&PointerType> {
        self.element.pointer_type()
    }
}

/// `MTLPointerType`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PointerType {
    pub(crate) element_type: DataType,
    pub(crate) access: BindingAccess,
    pub(crate) alignment: u64,
    pub(crate) data_size: u64,
    pub(crate) element_is_argument_buffer: bool,
    pub(crate) element: TypeDetail,
}

impl PointerType {
    pub(crate) fn new(
        element_type: DataType,
        access: BindingAccess,
        alignment: u64,
        data_size: u64,
        element_is_argument_buffer: bool,
        element: TypeDetail,
    ) -> PointerType {
        PointerType {
            element_type,
            access,
            alignment,
            data_size,
            element_is_argument_buffer,
            element,
        }
    }

    /// `elementType`.
    pub fn element_type(&self) -> DataType {
        self.element_type
    }

    /// `access`.
    pub fn access(&self) -> BindingAccess {
        self.access
    }

    /// `alignment`: the pointee's alignment.
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// `dataSize`: the pointee's size.
    pub fn data_size(&self) -> u64 {
        self.data_size
    }

    /// `elementIsArgumentBuffer`.
    pub fn element_is_argument_buffer(&self) -> bool {
        self.element_is_argument_buffer
    }

    /// `elementStructType`.
    pub fn element_struct_type(&self) -> Option<&StructType> {
        self.element.struct_type()
    }

    /// `elementArrayType`.
    pub fn element_array_type(&self) -> Option<&ArrayType> {
        self.element.array_type()
    }
}

/// `MTLTextureReferenceType`: an argument-buffer member's texture.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TextureReferenceType {
    pub(crate) data_type: DataType,
    pub(crate) texture_type: TextureType,
    pub(crate) access: BindingAccess,
    pub(crate) depth: bool,
}

impl TextureReferenceType {
    pub(crate) fn new(
        data_type: DataType,
        texture_type: TextureType,
        access: BindingAccess,
        depth: bool,
    ) -> TextureReferenceType {
        TextureReferenceType {
            data_type,
            texture_type,
            access,
            depth,
        }
    }

    /// `textureDataType`.
    pub fn texture_data_type(&self) -> DataType {
        self.data_type
    }

    /// `textureType`.
    pub fn texture_type(&self) -> TextureType {
        self.texture_type
    }

    /// `access`.
    pub fn access(&self) -> BindingAccess {
        self.access
    }

    /// `isDepthTexture`.
    pub fn is_depth_texture(&self) -> bool {
        self.depth
    }
}
