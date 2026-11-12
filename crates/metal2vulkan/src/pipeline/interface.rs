//! `PipelineInterface`: the binder's input and a Metal-reflection-shaped view for the consumer.
//!
//! Per stage and slot it states the argument kind, how the shader accesses it (proven, never
//! guessed), the buffer form and view the function selected, and the proven window. It replaced
//! the old engine's `ShaderReflection`, which 0.2.0 removed.

use std::collections::BTreeMap;

pub use crate::air::canon::CommandWrites;
use crate::metal;
use crate::pipeline::descriptor::TessellationDescriptor;
use crate::pipeline::sym::{PushField, ResourceSlot, SampledType, SlotRef, SpecKey, Stage};
use crate::vk;

/// The interface of a pipeline.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PipelineInterface {
    /// Every statically used Metal argument slot.
    #[cfg_attr(feature = "serde", serde(with = "crate::pipeline::serde_pairs"))]
    pub slots: BTreeMap<SlotRef, SlotInterface>,
    /// `[[threadgroup(n)]]` arguments of the kernel.
    pub threadgroup: Vec<ThreadgroupArgument>,
    /// Some function reads the resident root table (a `Psb` slot or a dynamic root), so the binder
    /// builds it, with an entry for every `Psb` slot.
    pub needs_pointer_translation: bool,
    /// Some function resolves texture or sampler handles, so the binder builds the handle table and
    /// the set-1 heap.
    pub needs_handle_translation: bool,
    /// Each stage's push placement, in stage order: the push bytes and `ArgumentTable` contents the
    /// binder writes.
    pub push: Vec<PushPlacement>,
    /// Some stage's function may reach an indirect command buffer
    /// (`StageInterface::reaches_command_buffers`): only such a pipeline can encode, reset or
    /// copy indirect commands; `false` is proven.
    #[cfg_attr(feature = "serde", serde(default))]
    pub reaches_command_buffers: bool,
    /// What the stages' functions may write into the indirect command buffers they reach
    /// (`StageInterface::command_writes`), all stages' together: a clear bit is proven of every
    /// function of the pipeline.
    #[cfg_attr(feature = "serde", serde(default))]
    pub command_writes: CommandWrites,
    /// The stages whose function may write a texture (`StageInterface::writes_textures`), in the
    /// pipeline's stage order: a stage not listed writes no texture, which is proven.
    #[cfg_attr(feature = "serde", serde(default))]
    pub texture_writing_stages: Vec<Stage>,
    /// The samplers a pipeline state numbers for each of its Metal functions
    /// (`StageInterface::numbered_samplers`, [`NumberedSampler`]), each function's in Metal's
    /// order, the functions in the pipeline's stage order: what the runtime gives sampler
    /// resource IDs as Metal's pipeline creation does.
    #[cfg_attr(feature = "serde", serde(default))]
    pub numbered_samplers: Vec<(Stage, Vec<NumberedSampler>)>,
    /// The kernel's `[[stage_in]]` fetches, each as its function was compiled under it
    /// (`StageInterface::io`'s `stage_in`), in attribute order. A dispatch decides
    /// `SpecKey::StageInReach` from those of a thread-position step.
    #[cfg_attr(feature = "serde", serde(default))]
    pub stage_in: Vec<crate::pipeline::stage_io::StageInFetch>,
}

/// What one Metal argument slot is and how the shader uses it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SlotInterface {
    Buffer(BufferSlot),
    Texture(TextureSlot),
    Sampler(SamplerSlot),
}

/// A buffer argument.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BufferSlot {
    /// What the module does to the buffer through the slot: the function's loads and stores, an
    /// atomic as both, and a read where a store merges a partial word through the sub-word
    /// store's two atomics, which read the word they modify.
    pub access: AccessSet,
    /// The byte window the shader can touch, relative to the bind offset.
    pub window: Window,
    /// How the function reaches the buffer.
    pub form: BufferForm,
    /// What the slot holds: a buffer, or a function table the function reads as a buffer of
    /// words. Metal binds each with its own call, and its API validation stops a dispatch whose
    /// slot holds the other kind.
    pub kind: BufferKind,
}

/// What a buffer slot holds: the function's parameter at its index in the AIR ABI (`air.buffer`,
/// `air.visible_function_table`, `air.intersection_function_table`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BufferKind {
    /// A buffer (`setBuffer:offset:atIndex:`).
    Buffer,
    /// A visible-function table (`setVisibleFunctionTable:atBufferIndex:`), read as the words of
    /// `pipeline::table`.
    VisibleFunctionTable,
    /// An intersection function table (`setIntersectionFunctionTable:atBufferIndex:`), read as
    /// its words.
    IntersectionFunctionTable,
}

/// A texture argument.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TextureSlot {
    /// The texture type the shader declares.
    pub texture_type: metal::TextureType,
    /// The numeric type the shader reads or writes.
    pub sampled_type: SampledType,
    /// A depth texture sampled with comparison.
    pub depth_compare: bool,
    pub access: TextureAccess,
}

/// How a texture is used (Metal `access::` qualifiers plus sampling).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TextureAccess {
    pub sample: bool,
    pub read: bool,
    pub write: bool,
}

impl TextureAccess {
    /// An access set from its three bits.
    pub fn new(sample: bool, read: bool, write: bool) -> TextureAccess {
        TextureAccess {
            sample,
            read,
            write,
        }
    }
}

/// A sampler argument.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SamplerSlot {
    /// The shader samples with depth comparison.
    pub compare: bool,
    /// A coordinate-sensitive sampling op uses it, so its coordinate mode is a runtime fact.
    pub coordinate_sensitive: bool,
}

/// A sampler a pipeline state numbers for one of its functions, as Metal's does: each takes a plain
/// sampler resource ID (`gpuResourceID` of the sampler states made after it start past it) unless
/// an equal one of the device holds one. Measured on an Apple M4 (macOS 27.0.1) and an M3 Max
/// (macOS 26.7).
///
/// [`CompiledFunction::numbered_samplers`](crate::library::CompiledFunction::numbered_samplers)
/// lists a function's in Metal's order, and [`PipelineInterface::numbered_samplers`] each of a
/// pipeline's functions': its constexpr samplers in the reverse of the module's
/// `!air.sampler_states` order, then [`Read`] where it reads a texture, then [`Write`] where it
/// writes one.
///
/// [`Read`]: NumberedSampler::Read
/// [`Write`]: NumberedSampler::Write
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NumberedSampler {
    /// A constexpr sampler the specialized function samples, gathers or queries a level of
    /// texture through, by its canonical AIR state words (as `SamplerRef::Constexpr` names it):
    /// two of equal state are one.
    Constexpr([u64; 2]),
    /// The sampler of a texture `read`: the specialized function reads a texel of a texture of
    /// any type (a texture buffer's, a depth texture's, a multisampled one's), through an
    /// argument, an argument buffer or any handle. A size query, an atomic, or a texture argument
    /// the function never reads takes none.
    Read,
    /// The sampler of a texture `write`: the specialized function writes a texel of a texture of
    /// any type (a texture buffer's too).
    Write,
}

/// Which memory operations the shader performs on a resource. `Unused` is both false.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AccessSet {
    pub read: bool,
    pub write: bool,
}

impl AccessSet {
    /// Neither read nor written.
    pub const UNUSED: AccessSet = AccessSet {
        read: false,
        write: false,
    };
    /// Read only.
    pub const READ: AccessSet = AccessSet {
        read: true,
        write: false,
    };
    /// Written only.
    pub const WRITE: AccessSet = AccessSet {
        read: false,
        write: true,
    };
    /// Read and written.
    pub const READ_WRITE: AccessSet = AccessSet {
        read: true,
        write: true,
    };
}

/// The proven byte window of a buffer slot, relative to its bind offset.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Window {
    /// `[start, start + len)`; `start` may be negative.
    Bounded { start: i64, len: u64 },
    /// Anywhere from the bind offset to the end of the allocation (always the case when the shader
    /// has a dynamic root).
    Whole,
}

/// How a function reaches a buffer slot.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BufferForm {
    /// Through a descriptor of this view kind.
    Descriptor(DescriptorView),
    /// Through a buffer device address (dynamic roots, per-stage limit fallback, or a
    /// `BufferRangeLimit` variant); the bound buffer needs a device address.
    Psb,
}

impl BufferForm {
    /// Whether the form reads through a uniform-buffer view: the views whose range a plan starts
    /// at the bind offset where the device's view base alignment divides it
    /// (`SpecKey::RangeAtBindOffset`, `BindState::view_base_alignment`).
    pub fn uniform_view(self) -> bool {
        matches!(
            self,
            BufferForm::Descriptor(DescriptorView::UniformWords | DescriptorView::UniformVec4)
        )
    }
}

/// The descriptor view kind of a descriptor-form slot.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DescriptorView {
    /// A storage buffer.
    Storage,
    /// A uniform buffer viewed as `u32[]` (needs `uniformBufferStandardLayout`).
    UniformWords,
    /// A uniform buffer viewed as a `uvec4` array (std140-valid everywhere).
    UniformVec4,
}

/// How a compiled function fetches one attribute it reads
/// ([`CompiledFunction::attribute_fetches`](crate::library::CompiledFunction::attribute_fetches)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AttributeFetch {
    /// Fixed-function vertex input: the pipeline's vertex input state fetches it.
    FixedFunction,
    /// In the shader, from the buffer at Metal index `buffer_index`.
    Pulled { buffer_index: u32 },
}

/// The interface of one compiled function: what link and the binder need without the module's
/// words.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StageInterface {
    pub stage: Stage,
    /// Every statically used Metal argument slot of the function.
    #[cfg_attr(feature = "serde", serde(with = "crate::pipeline::serde_pairs"))]
    pub slots: BTreeMap<SlotRef, SlotInterface>,
    /// `[[threadgroup(n)]]` arguments (kernels).
    pub threadgroup: Vec<ThreadgroupArgument>,
    /// Where each push field lives: the stage's push range and `ArgumentTable` layout.
    pub push: PushPlacement,
    /// Every descriptor binding symbol the module declares, synthetic ones included, with the
    /// descriptor it needs; sorted by symbol (the link contract).
    pub descriptors: Vec<StageDescriptor>,
    /// Every specialization constant the module declares, sorted by key.
    pub spec: Vec<StageSpec>,
    /// The function reads the resident root table: it has a `Psb` slot or a dynamic root, so the
    /// binder builds the root table with an entry for every `Psb` slot.
    pub needs_pointer_translation: bool,
    /// The function resolves texture or sampler handles.
    pub needs_handle_translation: bool,
    /// The stage-IO interface (vertex and fragment functions; empty for kernels): what every
    /// `Location` key of the module is, and every vertex attribute the function reads with the
    /// fetch it was compiled under. It is the one contract link numbers locations and derives
    /// vertex input from.
    pub io: crate::pipeline::stage_io::StageIo,
    /// A post-tessellation vertex function's evaluation stage: the module is a
    /// `TessellationEvaluation` entry, bound through the vertex argument table as Metal binds it.
    pub tessellation: Option<TessellationStage>,
    /// The function writes memory another invocation or a later command can read: it stores to
    /// a buffer or a texture, or updates one atomically (`spirv::requirements::writes_memory`).
    /// Metal runs a multisampled fragment function with such a side effect once per covered
    /// sample. The `ExecutionStatus` buffer's words, which the resident-miss count updates, are not
    /// the function's, nor a color's interlocked storage form's or `half` shadow's, which carry
    /// Metal's color attachment write.
    #[cfg_attr(feature = "serde", serde(default))]
    pub writes_memory: bool,
    /// A mesh function compiled for a device without `VK_EXT_mesh_shader`: the module is a
    /// `GLCompute` entry that writes its threadgroups' vertices and primitives as these records,
    /// which the pipeline's synthesized vertex stage draws.
    #[cfg_attr(feature = "serde", serde(default))]
    pub mesh_records: Option<MeshRecords>,
    /// A mesh function's primitive: its indices, 1 (points), 2 (lines) or 3 (triangles), on
    /// every device, whether the device draws the mesh or the function writes records. `None`
    /// for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub mesh_corners: Option<u32>,
    /// An object or mesh function compiled for a device without `VK_EXT_mesh_shader` (a
    /// `GLCompute` entry): its payload's words (0 without a payload), which its object
    /// threadgroup's slot of the draw's work buffer holds (`MeshRecords::slot_words`). `None`
    /// for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub emulated_payload: Option<u32>,
    /// An object or mesh function's payload arena's bytes (`memory::arena::payload_bytes` of its
    /// `[[payload]]` parameter's `air.arg_type_size`, a mesh function's at least
    /// `LoweringInputs::object_payload`), which its accesses of the payload read and write; `None`
    /// for a function without a payload. Link holds a mesh function's to at least its pipeline's
    /// object function's (`NeedsInput(ObjectPayload)`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub payload: Option<u32>,
    /// The function may reach an indirect command buffer, of the entry its function constants
    /// specialized (a constant that gates an argument or a path off leaves it out); `false` only
    /// where the function provably touches none. It is `true` where the function takes one
    /// (`air.command_buffer`), as an argument or as a member of a buffer it takes at any depth of
    /// the buffer's argument-buffer layouts, whatever the body does with it; and where the body
    /// calls an indirect-command operation on any pointer, a command buffer the metadata names or
    /// one the body casts from other bytes, loads from a member no layout describes, or makes
    /// from an integer (Metal gives a command buffer's records no address, so an operation is the
    /// only way to them), or makes a call it cannot see into (`Entry::may_encode_indirect_commands`
    /// of `metal2vulkan-air`). Only such a function can encode, reset or copy indirect commands.
    #[cfg_attr(feature = "serde", serde(default))]
    pub reaches_command_buffers: bool,
    /// What the function's indirect-command operations may write into the records they reach,
    /// of the entry its function constants specialized ([`CommandWrites`]: the operations it
    /// may call and the primitive types its draws may pass; `Entry::command_writes` of
    /// `metal2vulkan-air`). A clear bit is proven: no operation of the function writes it. Empty
    /// where `reaches_command_buffers` is `false`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub command_writes: CommandWrites,
    /// The function may write a texture, of the entry its function constants specialized;
    /// `false` only where the function provably writes none. It is `true` where the function
    /// takes a texture declared `access::write` or `access::read_write`, as an argument or as a
    /// member of a buffer it takes at any depth of the buffer's argument-buffer layouts,
    /// whatever the body does with it; and where the body calls a texture operation that stores
    /// a texel (a `write`, an imageblock slice written to a texture, or an atomic other than a
    /// load) through any handle, an argument buffer's, a heap's or a resource ID included, or
    /// makes a call it cannot see into (`Entry::may_write_textures` of `metal2vulkan-air`).
    /// Color attachment outputs, imageblock and tile stores, and stores to the bytes of a
    /// buffer a texture views are not texture writes.
    #[cfg_attr(feature = "serde", serde(default))]
    pub writes_textures: bool,
    /// The samplers a pipeline state of the function numbers for it, of the entry its function
    /// constants specialized, in Metal's order ([`NumberedSampler`]): the runtime gives each a
    /// sampler resource ID as Metal's pipeline creation does.
    #[cfg_attr(feature = "serde", serde(default))]
    pub numbered_samplers: Vec<NumberedSampler>,
    /// A fragment function's texture slots whose every texel read is at the fragment's own pixel,
    /// sorted: an `OpImageFetch` at level 0, with no offset, of a single-sampled, non-arrayed 2D
    /// image, at Metal's saturating `uint2` of `[[position]]`'s x and y (`read(uint2(p.xy))`),
    /// clamped into the image or not. On a framebuffer below `2^24` pixels a side, such a slot's
    /// reads lie inside the rectangle bounding the draw's scissors, a pixel wider each way for
    /// its helper invocations, clamped as the coordinate is. Empty for every other stage.
    #[cfg_attr(feature = "serde", serde(default))]
    pub own_pixel_textures: Vec<u32>,
    /// Those of `own_pixel_textures` a draw can read through a snapshot an earlier draw of the
    /// same primitives took, sorted (`spirv::requirements::snapshot_textures`): no value a fetch
    /// of the slot returns reaches, directly or through a call, a derivative, a sample, a quad or
    /// subgroup operation, a branch, an integer divisor, a dynamic vector index, or a store other
    /// than to a `Function`, `Private` or `Output` variable, so a helper invocation's value of it
    /// reaches no fragment's result and gives it no undefined behavior; and the function writes no
    /// memory and holds no interlock. Its variant (`LoweringInputs::snapshot_textures`) reads the slot through
    /// the snapshot. Empty for every other stage.
    #[cfg_attr(feature = "serde", serde(default))]
    pub snapshot_textures: Vec<u32>,
    /// Why each slot of `own_pixel_textures` that `snapshot_textures` leaves out is left out:
    /// every [`SnapshotRefusal`] that holds for it, sorted by slot and then by refusal, so a slot
    /// may appear more than once. Empty for every other stage, and for a function compiled to read
    /// slots through snapshots (`LoweringInputs::snapshot_textures`), which judges none.
    #[cfg_attr(feature = "serde", serde(default))]
    pub snapshot_refusals: Vec<(u32, SnapshotRefusal)>,
    /// The levels of its bound view that each texture slot's reads reach, one entry a slot, sorted
    /// by slot ([`LevelReads`]): what a runtime needs to tell whether a slot whose texture presents
    /// a color attachment of the draw's render pass instance at some level reads that level.
    /// Empty for a kernel.
    #[cfg_attr(feature = "serde", serde(default))]
    pub level_reads: Vec<LevelReads>,
    /// The private memory a thread of the function keeps, in bytes: its `thread` variables and
    /// arrays a run-time index reaches, its call stacks, and the constants it holds per thread so
    /// (`ProfileFields::function_storage` holds it). What
    /// `PrecisionNote::FunctionStorageAssumed`'s announcement names.
    #[cfg_attr(feature = "serde", serde(default))]
    pub private_bytes: u64,
}

/// What leaves an own-pixel texture slot of a fragment function out of the slots a draw can read
/// through a snapshot an earlier draw of the same primitives took
/// (`StageInterface::snapshot_refusals`, `StageModule::snapshot_refusals`). A helper invocation
/// reads a constant from the snapshot, so the slot is left out where a helper's value of it could
/// reach another invocation's result, give it undefined behavior, or where the snapshot cannot be
/// made beside the function. The first eight say where a value that a fetch of the slot reaches,
/// through any instruction, phi, call, or `Function`, `Private` or `Output` variable, is used.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SnapshotRefusal {
    /// It decides a conditional branch or a switch, but for a branch that only decides whether
    /// the invocation demotes itself.
    Branch,
    /// It is an operand of a derivative.
    Derivative,
    /// It is an operand of a sample, a gather or a level-of-detail query of any image.
    Sample,
    /// It is an operand of a subgroup or quad operation.
    GroupOperation,
    /// It is stored to a storage class other than `Function`, `Private` or `Output`, or through
    /// a function parameter's pointer.
    Store,
    /// It is an operand of an atomic.
    Atomic,
    /// It is an operand of an image write or a texel pointer.
    ImageWrite,
    /// It is an operand an instruction's behavior is undefined for at some value: an integer
    /// divisor, the dividend of a signed division or remainder whose divisor may be `-1`, or a
    /// dynamic vector component index.
    UndefinedOperand,
    /// The function writes memory (`StageInterface::writes_memory`).
    WritesMemory,
    /// The module holds a fragment interlock critical section.
    Interlock,
    /// The module has a `Coherent` resource or uses the Vulkan memory model.
    Coherent,
    /// Anything else: the slot's index is 128 or more, its values take more than 16 passes over
    /// the module to settle, or one is an operand of a ray query, a mesh output count or a call
    /// whose callee's parameters it does not match, or the entry's returned value.
    Other,
}

/// The levels of its bound view that a texture slot's reads reach, each counted from the view's
/// first level (`StageInterface::level_reads`, `StageModule::level_reads`), as the module's
/// instructions decide them (`spirv::requirements::level_reads`, structural). A sample's level is
/// its sampler's to bound, so the fact lists each sampler the slot is sampled through apart, and a
/// runtime asks [`LevelReads::highest_level`] and [`LevelReads::lowest_level`] with the samplers
/// it binds. Vulkan's rules decide
/// the levels (Image Level(s) Selection): a sample reads `clamp(λ, 0, q)` past the view's first
/// level, `λ` its level of detail clamped into the sampler's `[minLod, maxLod]`; a gather reads
/// the first level; a fetch and a size query read the level they name.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LevelReads {
    /// The texture slot.
    pub slot: u32,
    /// Some use of the slot may reach any level of its view, or answers by the view's level
    /// count: a fetch or size query at a level the module computes, a level-count query beyond the
    /// containment of a constant level (`UMin(c, levels - 1)` and `c < levels`), a level-of-detail
    /// query, a sample with a minimum level of detail operand, a storage access, or any use the
    /// walk does not follow (a call, a store, a phi or select of the image).
    pub any: bool,
    /// The highest level a fetch, a size query or a gather reads, each a constant (a gather reads
    /// level 0), or the highest constant a level-count containment compares; `None` where none
    /// does.
    pub named: Option<u32>,
    /// The lowest level whose texels a fetch or a gather reads, each a constant (a gather reads
    /// level 0); `None` where none does. A size query reads no texel, nor does a level-count
    /// containment's comparison: each raises `named` alone.
    pub lowest_read: Option<u32>,
    /// The slot's samples, by the sampler each reads through, sorted by sampler.
    pub sampled: Vec<SampledLevels>,
}

/// The samples of a texture slot through one sampler ([`LevelReads::sampled`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SampledLevels {
    /// The sampler: a runtime sampler's slot, or a constexpr sampler.
    pub sampler: crate::pipeline::sym::SamplerRef,
    /// The bits of the highest explicit level of detail among its samples that is a constant
    /// `f32` (a NaN where one is); `None` where none is.
    pub highest_lod_bits: Option<u32>,
    /// The bits of the lowest explicit level of detail among its samples that is a constant `f32`
    /// (a NaN where one is); `None` where none is.
    pub lowest_lod_bits: Option<u32>,
    /// Some sample's level of detail is the module's to compute: implicit, biased, from
    /// gradients, or an explicit level that is no constant. The sampler's `maxLod` bounds it.
    pub computed: bool,
}

/// A sampler's level selection as a runtime binds it ([`SampledLevels::level`]): its
/// `minLod`, `maxLod` and `mipLodBias`, whether its mipmap mode is `LINEAR`, and whether every
/// texel of the view is finite ([`SamplerLevels::finite_texels`]), where a linear blend of a level
/// with weight 0 reads that level's texel to no effect.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerLevels {
    pub min_lod: f32,
    pub max_lod: f32,
    pub bias: f32,
    pub linear: bool,
    pub finite_texels: bool,
}

impl SamplerLevels {
    /// The levels of Vulkan sampler `desc` over a view whose texels `finite_texels` says are
    /// finite.
    pub fn of(desc: &vk::SamplerDesc, finite_texels: bool) -> SamplerLevels {
        SamplerLevels {
            min_lod: desc.min_lod,
            max_lod: desc.max_lod,
            bias: desc.mip_lod_bias,
            linear: desc.mipmap_mode == vk::SamplerMipmapMode::LINEAR,
            finite_texels,
        }
    }

    /// Whether every texel a view of `format` holds reads as a finite number: an 8-, 10- or
    /// 16-bit normalized or sRGB color format, or an integer one. `false` for every other format,
    /// a float one among them, whose texels may be infinities or NaNs.
    pub fn finite_texels(format: vk::Format) -> bool {
        use vk::Format as F;
        matches!(
            format,
            F::R8_UNORM
                | F::R8_SNORM
                | F::R8G8_UNORM
                | F::R8G8_SNORM
                | F::R8G8B8A8_UNORM
                | F::R8G8B8A8_SNORM
                | F::R8G8B8A8_SRGB
                | F::B8G8R8A8_UNORM
                | F::B8G8R8A8_SRGB
                | F::A2B10G10R10_UNORM_PACK32
                | F::A2R10G10B10_UNORM_PACK32
                | F::R16_UNORM
                | F::R16_SNORM
                | F::R16G16_UNORM
                | F::R16G16_SNORM
                | F::R16G16B16A16_UNORM
                | F::R16G16B16A16_SNORM
                | F::R5G6B5_UNORM_PACK16
                | F::B5G6R5_UNORM_PACK16
                | F::R8_UINT
                | F::R8_SINT
                | F::R8G8_UINT
                | F::R8G8_SINT
                | F::R8G8B8A8_UINT
                | F::R8G8B8A8_SINT
                | F::R16_UINT
                | F::R16_SINT
                | F::R16G16_UINT
                | F::R16G16_SINT
                | F::R16G16B16A16_UINT
                | F::R16G16B16A16_SINT
                | F::R32_UINT
                | F::R32_SINT
                | F::R32G32_UINT
                | F::R32G32_SINT
                | F::R32G32B32A32_UINT
                | F::R32G32B32A32_SINT
        )
    }
}

impl SampledLevels {
    /// The highest explicit constant level of detail among the samples.
    pub fn highest_lod(&self) -> Option<f32> {
        self.highest_lod_bits.map(f32::from_bits)
    }

    /// The highest level, counted from the view's first, these samples reach through a sampler
    /// of levels `s`; `None` where they may reach any.
    ///
    /// A sample's level of detail is at most `L`: `maxLod`, or for a constant explicit `c` with
    /// nothing computed, `c` plus the sampler's bias where it is positive (Vulkan adds the
    /// sampler's bias to an explicit level of detail too), clamped into `[minLod, maxLod]`; at
    /// least 0. The nearest mipmap mode reads level `floor(L + 0.5)` at most (both of Vulkan's
    /// roundings); the linear one reads `floor(L) + 1` with the weight `L - floor(L)`, so where
    /// `L` is a whole level and the texels are finite it reads level `L` to effect alone. A
    /// sampler whose `minLod` exceeds its `maxLod`, or whose bounds are no finite numbers,
    /// reaches any level.
    pub fn level(&self, s: SamplerLevels) -> Option<u32> {
        // A last level past every texture's: Metal's largest has 15 levels.
        const PAST: f32 = 64.0;
        if !(s.min_lod.is_finite() && s.max_lod.is_finite()) || s.min_lod > s.max_lod {
            return None;
        }
        let computed = self.computed.then_some(s.max_lod);
        // A negative bias is not counted on to lower the level.
        let explicit = (self.highest_lod()).map(|c| match c.is_nan() {
            true => s.max_lod,
            false => (c + s.bias.max(0.0)).clamp(s.min_lod, s.max_lod),
        });
        let top = (computed.into_iter().chain(explicit)).fold(0.0_f32, |top, l| top.max(l));
        if top >= PAST {
            return None;
        }
        let level = match s.linear {
            true if s.finite_texels && top.fract() == 0.0 => top,
            true => top.floor() + 1.0,
            false => (top + 0.5).floor(),
        };
        Some(level as u32)
    }

    /// The lowest explicit constant level of detail among the samples.
    pub fn lowest_lod(&self) -> Option<f32> {
        self.lowest_lod_bits.map(f32::from_bits)
    }

    /// The lowest level, counted from the view's first, these samples may read through a sampler
    /// of levels `s`, over a view of more levels than it (a view whose last level is lower reads
    /// that one in its place: the LOD Operation clamps into `[0, q]`).
    ///
    /// A sample's level of detail is at least `L`: `minLod` where one is computed (the LOD
    /// Operation clamps it into `[minLod, maxLod]` after any bias), and for the lowest constant
    /// explicit `c`, `c` plus the sampler's bias where it is negative, clamped so; at least 0.
    /// Either mipmap mode reads level `floor(L)` or a higher one: the nearest one
    /// `ceil(L + 0.5) - 1` or `floor(L + 0.5)`, the linear one `floor(L)` and the next. A NaN
    /// constant, a sampler whose `minLod` exceeds its `maxLod` (the level of detail is undefined),
    /// or one whose bounds are no finite numbers, reaches the first level.
    pub fn lowest(&self, s: SamplerLevels) -> u32 {
        if !(s.min_lod.is_finite() && s.max_lod.is_finite()) || s.min_lod > s.max_lod {
            return 0;
        }
        let computed = self.computed.then_some(s.min_lod);
        let explicit = match self.lowest_lod() {
            Some(c) if c.is_nan() => return 0,
            // A positive bias is not counted on to raise the level, and a NaN one is none, as a
            // device's sampler takes it (`metal::sampler::device_lod_bias`).
            Some(c) => Some((c + s.bias.min(0.0)).clamp(s.min_lod, s.max_lod)),
            None => None,
        };
        let low = (computed.into_iter().chain(explicit)).reduce(f32::min);
        low.map_or(0, |low| low.max(0.0).floor() as u32)
    }
}

impl LevelReads {
    /// The highest level, counted from the view's first, that the slot's reads reach, where
    /// `sampled` answers for each of its samplers' samples the highest level they reach through
    /// the sampler the runtime binds ([`SampledLevels::level`]); `None` where a read may reach
    /// any level ([`LevelReads::any`]) or `sampled` answers `None`. A view of the same first level
    /// and more levels than this answers every read of the slot as the whole view does: each
    /// reads the same level and texels (Image Level(s) Selection, at a level of detail that does
    /// not depend on the level count).
    pub fn highest_level(&self, sampled: impl Fn(&SampledLevels) -> Option<u32>) -> Option<u32> {
        if self.any {
            return None;
        }
        let mut highest = self.named.unwrap_or(0);
        for samples in &self.sampled {
            highest = highest.max(sampled(samples)?);
        }
        Some(highest)
    }

    /// The lowest level, counted from the view's first, that the slot's reads may reach, where
    /// `sampled` answers for each of its samplers' samples the lowest level they may reach
    /// through the sampler the runtime binds ([`SampledLevels::lowest`]): 0 where a read may reach
    /// any level ([`LevelReads::any`]); `None` where no read reaches a level's texels (a size
    /// query, a level-count containment's comparison). Over a view of the same first level whose
    /// last level is lower, a read reaches that last level in its place.
    pub fn lowest_level(&self, sampled: impl Fn(&SampledLevels) -> u32) -> Option<u32> {
        if self.any {
            return Some(0);
        }
        let samples = self.sampled.iter().map(sampled);
        self.lowest_read.into_iter().chain(samples).min()
    }
}

/// An emulated mesh function's records: what each of its threadgroups writes to the draw's records
/// buffer (`SyntheticBinding::MeshRecords`), all of it words.
///
/// Mesh threadgroup `w`, its number among the draw's mesh threadgroups (object threadgroup by
/// object threadgroup, each one's grid x fastest), owns [`MeshRecords::group_words`] words from
/// the draw's first word plus `w · group_words()`: its primitive count, then `vertices` vertex
/// records of `vertex_words` words each, then `primitives` primitive records of
/// `primitive_words` words each, every primitive record beginning with the primitive's
/// `corners` indices. Each field sits at its offset in its record, one word per lane. A
/// threadgroup zeroes its words before its body runs, so an index, a culled flag, or a render
/// target slice or viewport index the function does not write reads 0, as Metal's does.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MeshRecords {
    /// The mesh type's `NV`.
    pub vertices: u32,
    /// The mesh type's `NP`.
    pub primitives: u32,
    /// The indices of one primitive: 1 (points), 2 (lines) or 3 (triangles).
    pub corners: u32,
    pub vertex_words: u32,
    pub primitive_words: u32,
    /// Every field, in record order: the vertex record's, then the primitive record's.
    pub fields: Vec<RecordField>,
    /// The words of an object threadgroup's slot in the draw's work buffer: its grid's three
    /// counts and a zero, then the larger payload of the pipeline's object and mesh functions
    /// (in a function's own interface, its own payload).
    pub slot_words: u32,
}

impl MeshRecords {
    /// The words of one threadgroup's records: its count, its vertex records and its primitive
    /// records.
    pub fn group_words(&self) -> u64 {
        1 + u64::from(self.vertices) * u64::from(self.vertex_words)
            + u64::from(self.primitives) * u64::from(self.primitive_words)
    }

    /// The word of a threadgroup's records where its primitive records begin.
    pub fn primitive_base(&self) -> u64 {
        1 + u64::from(self.vertices) * u64::from(self.vertex_words)
    }

    /// The Vulkan list topology of the records' primitives.
    pub fn topology(&self) -> vk::PrimitiveTopology {
        match self.corners {
            1 => vk::PrimitiveTopology::POINT_LIST,
            2 => vk::PrimitiveTopology::LINE_LIST,
            _ => vk::PrimitiveTopology::TRIANGLE_LIST,
        }
    }
}

/// One field of a vertex or primitive record ([`MeshRecords`]).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RecordField {
    pub value: RecordValue,
    /// In the primitive record, else in the vertex record.
    pub per_primitive: bool,
    /// Its first word in its record.
    pub offset: u32,
    /// Its words: its lanes, or the length of the clip distances.
    pub words: u32,
    /// What each word holds: a `float`'s bits, a `uint`, or an `int`'s bits (a `bool` is a
    /// `uint`, 0 or 1).
    pub numeric: crate::pipeline::stage_io::IoNumeric,
}

/// What a [`RecordField`] holds.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RecordValue {
    /// `[[position]]`.
    Position,
    /// `[[point_size]]`.
    PointSize,
    /// `[[clip_distance]]`.
    ClipDistance,
    /// `[[primitive_culled]]`, 1 where set.
    Culled,
    /// The primitive's render target slice (`Layer`): its `[[render_target_array_index]]` where
    /// that lies below the pass's length, else 0.
    Layer,
    /// The primitive's viewport (`ViewportIndex`): its `[[viewport_array_index]]` where that lies
    /// below the draw's viewport count, else 0.
    ViewportIndex,
    /// A user varying: vertex data, primitive data, or a raw index (the link numbers its
    /// location by key).
    Varying(crate::pipeline::sym::VaryingKey),
}

/// What a tessellation evaluation module (a post-tessellation vertex function) needs of the
/// pipeline's control stage: its patch domain and Metal's control points per patch. It reads the
/// patch index and the instance as `uint` patch inputs at
/// [`TessellationStage::PATCH_INDEX_LOCATION`] and [`TessellationStage::INSTANCE_LOCATION`],
/// which the control stage writes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TessellationStage {
    pub domain: PatchDomain,
    /// `[[patch(_, N)]]`: the control points of each patch the function reads (0 when the
    /// function does not state it).
    pub control_points: u32,
    /// `Some` for a function compiled for a device without `tessellationShader`: the module is a
    /// plain `Vertex` entry that draws the tessellator's triangles itself
    /// (`GraphicsState::emulated_tessellation`), baking in this render pipeline descriptor
    /// tessellation state (`LoweringInputs::tessellation`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub emulated: Option<TessellationDescriptor>,
}

impl TessellationStage {
    /// The patch input holding Metal's patch index (`[[patch_id]]`).
    pub const PATCH_INDEX_LOCATION: u32 = 0;
    /// The patch input holding Metal's instance index (`[[instance_id]]`).
    pub const INSTANCE_LOCATION: u32 = 1;
    /// The patch input holding the element of each of the patch's control points: a `uvec4` array
    /// of `ceil(N / 4)` elements, control point `k`'s element in component `k % 4` of element `k /
    /// 4`, from this location on.
    pub const CONTROL_POINT_LOCATION: u32 = 2;

    /// The evaluation stage of a `domain` patch of `control_points` control points.
    pub fn new(domain: PatchDomain, control_points: u32) -> TessellationStage {
        TessellationStage {
            domain,
            control_points,
            emulated: None,
        }
    }

    /// This stage, emulated under the descriptor tessellation state `desc`.
    pub fn emulated(self, desc: TessellationDescriptor) -> TessellationStage {
        TessellationStage {
            emulated: Some(desc),
            ..self
        }
    }
}

/// The patch type of a post-tessellation vertex function (`[[patch(triangle|quad, N)]]`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PatchDomain {
    /// `[[patch(triangle, N)]]`.
    Triangle,
    /// `[[patch(quad, N)]]`.
    Quad,
}

impl PipelineInterface {
    /// The buffer slots of `stage` (bit `i` for index `i`, below 64) its function reads through a
    /// uniform view ([`BufferForm::uniform_view`]): the slots whose bind offset class can tell the
    /// device's view base alignment apart (`StageTable::at_bind_views`).
    pub fn uniform_view_buffers(&self, stage: Stage) -> u64 {
        let mut slots = 0;
        for (slot, interface) in &self.slots {
            if let SlotInterface::Buffer(b) = interface {
                if slot.stage == stage && slot.index < u64::BITS && b.form.uniform_view() {
                    slots |= 1 << slot.index;
                }
            }
        }
        slots
    }
}

impl StageInterface {
    /// The Vulkan stage of the module: the evaluation stage for a post-tessellation vertex
    /// function (the vertex stage where it is emulated), the compute stage for an emulated object
    /// or mesh function, else the Metal stage's.
    pub fn vk_stage(&self) -> vk::ShaderStage {
        match (self.tessellation, self.emulated_payload) {
            (Some(t), _) if t.emulated.is_some() => vk::ShaderStage::VERTEX,
            (Some(_), _) => vk::ShaderStage::TESSELLATION_EVALUATION,
            (None, Some(_)) => vk::ShaderStage::COMPUTE,
            (None, None) => crate::device::derive::vk_stage(self.stage),
        }
    }

    /// Whether the module declares specialization constant `key`.
    #[cfg(test)]
    pub fn declares_spec(&self, key: SpecKey) -> bool {
        self.spec.iter().any(|s| s.key == key)
    }

    /// The keys the function reads: its specialization constants' keys, then the keys it reads
    /// as data from its `PushField::Specialized` fields.
    #[cfg(test)]
    pub fn read_keys(&self) -> Vec<SpecKey> {
        let pushed = self.push.fields.iter().filter_map(|f| match f.field {
            crate::pipeline::sym::PushField::Specialized(key) => Some(key),
            _ => None,
        });
        self.spec.iter().map(|s| s.key).chain(pushed).collect()
    }

    /// Whether the function reads key `key`, specialized or as data.
    #[cfg(test)]
    pub fn reads_key(&self, key: SpecKey) -> bool {
        self.read_keys().contains(&key)
    }

    /// The function is a vertex function with a `[[point_size]]` output that the module writes
    /// (its `PointSize` builtin, not the 1.0 the module writes for a function without one). Metal
    /// refuses such a function in a render pipeline whose input primitive topology class is line or
    /// triangle (measured on an M3 Max and an M4). A post-tessellation vertex function's point size
    /// sizes nothing and is dropped, so it answers `false`.
    pub fn writes_point_size(&self) -> bool {
        self.io
            .builtins
            .contains(&crate::pipeline::stage_io::StageBuiltin::PointSize { default: false })
    }
}

/// One descriptor binding a function's module declares: link numbers it in set 0 and lists it in
/// the `PipelineLayout` with exactly this type and count.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StageDescriptor {
    pub slot: ResourceSlot,
    pub ty: vk::DescriptorType,
    /// Descriptor array size (1 for a single descriptor; 0 for a set-1 heap array, whose size
    /// link sets).
    pub count: u32,
}

impl StageDescriptor {
    /// A descriptor of `ty` and `count` for `slot`.
    pub fn new(slot: ResourceSlot, ty: vk::DescriptorType, count: u32) -> StageDescriptor {
        StageDescriptor { slot, ty, count }
    }
}

/// One specialization constant a function's module declares.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StageSpec {
    pub key: SpecKey,
    /// Size in bytes of the `VkSpecializationMapEntry` (4 for `bool` and 32-bit scalars).
    pub size: u32,
    /// The module's default value, as its low bytes.
    pub default: u64,
    /// The bits of the value no instruction of the module can read and no specialized note
    /// tests (`spirv::spec_bits`): a binder clears them, so values that differ only there give
    /// one program and one pipeline. 0 where every bit may be read.
    #[cfg_attr(feature = "serde", serde(default))]
    pub unread_bits: u64,
}

impl StageSpec {
    /// A constant `key` of `size` bytes with `default`, every bit of it read.
    pub fn new(key: SpecKey, size: u32, default: u64) -> StageSpec {
        StageSpec {
            key,
            size,
            default,
            unread_bits: 0,
        }
    }
}

/// The push placement of one function: each field's location in the stage's push window or its
/// `ArgumentTable`, in the fixed kind order.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PushPlacement {
    pub stage: Stage,
    /// Every field, in placement order.
    pub fields: Vec<PlacedField>,
    /// `(offset, size)` of the stage's push-constant range, when any field is pushed. The offset
    /// is the stage window's start (vertex 0, fragment 64, compute 0).
    pub push_range: Option<(u32, u32)>,
    /// Byte size of the stage's `ArgumentTable` uniform buffer (a multiple of 16); 0 without one.
    pub argument_table_size: u32,
}

impl PushPlacement {
    /// The value each `PushField::Specialized` field placed in the push-constant block holds in
    /// push-constant bytes `push` (the block's bytes from offset 0), by its key. A field in the
    /// `ArgumentTable` is the caller's to read.
    #[cfg(test)]
    pub fn pushed_values(&self, push: &[u8]) -> Vec<(SpecKey, u64)> {
        self.fields
            .iter()
            .filter_map(|f| match (f.field, f.loc) {
                (PushField::Specialized(key), PushLoc::Push { offset }) => {
                    let at = offset as usize;
                    let word = push
                        .get(at..at + 4)
                        .unwrap_or_else(|| panic!("no push bytes for {key:?} at {at}"));
                    Some((
                        key,
                        u64::from(u32::from_le_bytes([word[0], word[1], word[2], word[3]])),
                    ))
                }
                _ => None,
            })
            .collect()
    }
}

/// One placed push field.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlacedField {
    pub field: PushField,
    pub loc: PushLoc,
    /// Size in bytes.
    pub size: u32,
}

/// Where a push field lives.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PushLoc {
    /// At this byte offset of the push-constant block (absolute, inside the stage window).
    Push { offset: u32 },
    /// At this byte offset of the stage's `ArgumentTable` uniform buffer.
    ArgumentTable { offset: u32 },
}

/// A `[[threadgroup(n)]]` argument of a kernel; its length is a runtime fact.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThreadgroupArgument {
    pub stage: Stage,
    pub index: u32,
    /// The alignment the arena gives the argument.
    pub alignment: u32,
}
