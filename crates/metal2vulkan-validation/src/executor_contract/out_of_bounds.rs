//! The out-of-bounds containment contract: how a candidate run of a case that declares
//! out-of-bounds targets (`case::out_of_bounds`) is judged.
//!
//! One rule, shared by the engine candidate when it runs a case, by a stored row's
//! re-decision (`CandidateObservation::status_holds`), and by the coverage report's containment
//! rows:
//!
//! - **The form and mechanism** of each target are what its compiled module does with it, never
//!   only what the case says ([`containment`]): a buffer is a storage or uniform buffer as its
//!   descriptor view is, or a device address; a buffer slot the product reaches through a descriptor view is
//!   checked in the shader or left to the device's buffer robustness (the two encodings); a slot
//!   reached by its device address and the threadgroup arena are checked in the shader in every
//!   leg; a texel buffer and an image follow the product's texel containment
//!   (`DeviceProfile::device_reads_texels` and `device_drops_texel_writes`): a texel buffer is left
//!   to `robustBufferAccess2` and otherwise checked in the shader; an image's reads are left to
//!   `robustImageAccess2` and its writes to `robustImageAccess` at either level, each otherwise
//!   checked in the shader. A buffer reached through a pointer loaded from an argument buffer is a
//!   device address the resident set resolves, a thread-local array a typed object whose index is
//!   checked against its length, and an argument buffer's texture array a handle table the loaded
//!   resource id is looked up in: each checked in the shader in every leg. An attribute buffer is
//!   fetched by fixed-function vertex input, left to the device's buffer robustness, or pulled in
//!   the shader through the slot its fetch names, contained as that slot is ([`Fetch`]); a direct
//!   indexed draw's index buffer is bounded by `robustBufferAccess2` or the binder's clamp; and a
//!   top-level texture array's dynamic index is clamped onto its elements in the shader.
//! - **The read rule** ([`read_rule`]) says what an out-of-bounds read of the target may return:
//!   zero under an in-shader check and under `robustBufferAccess2` (Vulkan's "Robust Buffer
//!   Access 2": "reads ... outside of the checked range will return zero values"), anything for
//!   an atomic's return under robust buffer access ("will return an undefined value"), and under
//!   `robustBufferAccess` alone zero, `(0, 0, 0, x)`, or a value of the buffer's own memory. A
//!   read through a handle past a handle array returns a texel of a texture the case binds, or
//!   zero (the null handle). An index read past an index buffer is index 0 or none, so the
//!   vertex ids only a read past it gives stay unshaded. A fixed-function vertex fetch past an
//!   attribute buffer the runtime path places in a heap of its own reads the heap's memory: the
//!   runtime binds a placed vertex buffer to the end of the heap's one buffer, as Metal's fetch
//!   reads a heap neighbour's bytes, and the run records that cause ([`ReadCause`],
//!   [`TargetRun::read_rule`]).
//! - **Nothing escapes**: every byte of the guard the candidate allocates around a buffer (an
//!   argument buffer's included) or texture buffer target is unchanged ([`TargetRun::escaped`]),
//!   and so is the memory bound to it past its end ([`TargetRun::past_end`]) except under
//!   `robustBufferAccess` alone, where a write may change any byte of the target's own memory, so
//!   the output bytes of a written buffer target are not compared there either. A target the
//!   runtime path places in a heap of its own has no memory of its own past its length padded to
//!   16: every other byte of the heap is its guard, so a write there escapes even under
//!   `robustBufferAccess` alone, which lets it land in the heap's memory, and the run records that
//!   cause ([`EscapeCause`]). An attribute buffer and an index buffer are guarded as a buffer is.
//!   An image's memory is opaque: its out-of-bounds writes are judged by its compared contents
//!   alone. A thread-local array and a handle array have no guard: what their accesses past them
//!   reach is the case's own layout, whose output bytes witness it.
//! - Every other output byte compares with Metal under the run's comparison class, as any case's
//!   does ([`matches()`]). A form whose out-of-bounds result Metal defines (a texture or a texture
//!   buffer) has no read bytes: its whole output compares with Metal.

use crate::case::out_of_bounds::{ContainmentForm, OutOfBoundsResource, OutOfBoundsTarget};
use crate::case::{AuthoredCase, OutputSelection};
use crate::coverage::{ExecutionClass, RobustBufferForm};
use crate::observation_contract::{ComparisonClass, ObservationLanes};
use base64::Engine as _;
use metal2vulkan::device::{Containment, DeviceProfile, Robustness};
use metal2vulkan::library::CompiledFunction;
use metal2vulkan::pipeline::{AttributeFetch, StrategyCensus};
use metal2vulkan::pipeline::{BufferForm, DescriptorView, SlotInterface};
use metal2vulkan::pipeline::{SlotClass, SlotRef, Stage as SymStage};
use metal2vulkan::pipeline::{StageInputDescriptor, VertexInputState};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// The memory an out-of-bounds target's accesses reach in its compiled module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Medium {
    /// A storage or uniform buffer descriptor view (`BufferForm::Descriptor`).
    DescriptorView,
    /// The buffer's device address (`BufferForm::Psb`).
    DeviceAddress,
    /// The threadgroup arena.
    Arena,
    /// A texel buffer view.
    TexelBuffer,
    /// An image view.
    Image,
    /// A thread-local object the module keeps as a typed variable, each array level's index
    /// checked against its length (the sized arrays).
    TypedObject,
    /// The handle table a texture's resource id, loaded from an argument buffer, is looked up in:
    /// an id of no resident texture is the null handle.
    HandleTable,
    /// Vulkan's fixed-function vertex input, fetching a vertex buffer binding (only under buffer
    /// robustness).
    VertexInput,
    /// A draw's index buffer, fetched by Vulkan's index fetch.
    IndexBuffer,
    /// A top-level handle array's element slots: an index is clamped onto the array's elements in
    /// the shader (the clamped handle-array load).
    HandleArray,
}

/// The check an out-of-bounds target's accesses take in its compiled module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Check {
    /// The shader checks the access: the normative check (a load yields zero, a store or an atomic
    /// is skipped), or an image read's clamp and write's guard.
    Shader,
    /// The shader clamps an image's reads, and the device's `robustImageAccess` without the
    /// second level discards its writes.
    ShaderReads,
    /// The device's `robustBufferAccess` without the second level.
    Robust,
    /// The device's `robustBufferAccess2` (or `robustImageAccess2`).
    Robust2,
    /// The binder bounded the command on the CPU: a direct indexed draw took only the indices
    /// inside its index buffer (the clamp, before the padded copy). No executor contains a target
    /// so now; committed records of earlier runs state it.
    Binder,
    /// A direct indexed draw reads a copy of its indices padded with zeros to every position it
    /// reads, so no index fetch lies past the bound range, where Vulkan states no index: each
    /// position past where Metal's fetch stops reads index 0.
    Padded,
}

/// How a compiled module contains one out-of-bounds target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mechanism {
    pub medium: Medium,
    pub check: Check,
}

impl Mechanism {
    /// `<medium>:<check>`, as the coverage report prints it.
    pub fn label(self) -> String {
        let medium = match self.medium {
            Medium::DescriptorView => "descriptor-view",
            Medium::DeviceAddress => "device-address",
            Medium::Arena => "arena",
            Medium::TexelBuffer => "texel-buffer",
            Medium::Image => "image",
            Medium::TypedObject => "typed-object",
            Medium::HandleTable => "handle-table",
            Medium::VertexInput => "vertex-input",
            Medium::IndexBuffer => "index-buffer",
            Medium::HandleArray => "handle-array",
        };
        let check = match self.check {
            Check::Shader => "shader",
            Check::ShaderReads => "shader-reads",
            Check::Robust => "robust",
            Check::Robust2 => "robust2",
            Check::Binder => "binder",
            Check::Padded => "padded",
        };
        format!("{medium}:{check}")
    }

    /// The form of the robust buffer encoding this mechanism leaves to the device
    /// ([`RobustBufferForm`]): a descriptor view's loads and stores or atomics, a texel buffer, and
    /// fixed-function vertex input, under buffer robustness; `None` for every in-shader check, for
    /// the binder's and the padded copy's, for images, and for an index fetch, which only
    /// `robustBufferAccess2` bounds.
    pub fn robust_buffer_form(self, form: ContainmentForm) -> Option<RobustBufferForm> {
        match (self.medium, self.check) {
            (_, Check::Shader | Check::ShaderReads | Check::Binder | Check::Padded) => None,
            (Medium::DescriptorView, _) if form == ContainmentForm::BufferAtomic => {
                Some(RobustBufferForm::Atomic)
            }
            (Medium::DescriptorView, _) => Some(RobustBufferForm::LoadStore),
            (Medium::TexelBuffer, _) => Some(RobustBufferForm::TexelBuffer),
            (Medium::VertexInput, _) => Some(RobustBufferForm::VertexInput),
            (
                Medium::DeviceAddress
                | Medium::Arena
                | Medium::Image
                | Medium::TypedObject
                | Medium::HandleTable
                | Medium::IndexBuffer
                | Medium::HandleArray,
                _,
            ) => None,
        }
    }
}

/// One target as a candidate run contained it: what the census records of an out-of-bounds run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetRun {
    pub resource: OutOfBoundsResource,
    /// The form the executed module exercised on the target ([`containment`]): the declared
    /// one, except that a buffer's is how its compiled slot reaches it.
    pub form: ContainmentForm,
    /// How the executed module contained it ([`containment`]).
    pub mechanism: Mechanism,
    /// The executed module writes the target (its interface's access set).
    pub written: bool,
    /// Bytes of the guard around the target, outside the memory bound to it, that the run
    /// changed: 0 when nothing escaped. Every buffer (an argument buffer's included), attribute
    /// buffer, index buffer, and texture buffer target has a guard (it is host-visible buffer
    /// memory, [`guarded`]); `None` for the threadgroup arena and thread memory, which no host
    /// memory holds, for an image, whose memory is opaque, and for a handle array, whose accesses
    /// stay in its argument buffer or the case's bound texture slots.
    pub escaped: Option<u64>,
    /// Bytes of the memory bound to a guarded target past its end that the run changed: its memory
    /// requirement past its length in an allocation of its own, its length padded to 16 in a heap
    /// of its own, filled with [`PAD_FILL`]. Only robustness 1 may write there ("within the memory
    /// range(s) bound to the buffer"); `None` without a guard.
    #[serde(default)]
    pub past_end: Option<u64>,
    /// What explains the guard bytes the run changed ([`TargetRun::escaped`]) without the
    /// device breaking Vulkan's containment, where something does ([`EscapeCause`]). The run
    /// escaped all the same: such a row is never a match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub escape_cause: Option<EscapeCause>,
    /// Why an out-of-bounds read of the target may return other values than its mechanism's
    /// ([`read_rule`]) because of where the candidate placed it ([`ReadCause`]); the rule the
    /// run's reads are judged by is [`TargetRun::read_rule`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub read_cause: Option<ReadCause>,
}

impl TargetRun {
    /// What an out-of-bounds read of the target may return in this run: its mechanism's rule
    /// ([`read_rule`]), or the memory of the heap the candidate placed it in where a
    /// [`ReadCause`] says the read reaches it ([`ReadRule::HeapWords`]).
    pub fn read_rule(&self) -> ReadRule {
        match self.read_cause {
            Some(ReadCause::HeapBoundRange) => ReadRule::HeapWords,
            None => read_rule(self.form, self.mechanism, self.written),
        }
    }
}

/// Why an out-of-bounds read of a target may return other values than its mechanism's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadCause {
    /// Fixed-function vertex input fetched the target ([`Medium::VertexInput`]), and the
    /// candidate placed it in a heap of its own (the runtime path): the runtime binds a placed
    /// vertex buffer at its offset in the heap's one buffer with no size, so the binding's range,
    /// which buffer robustness checks the fetch against, runs to the heap's end, and a fetch past
    /// the target reads the heap's bytes there. Metal's own fetch past a placed buffer reads its
    /// heap neighbour's bytes, a result Metal leaves undefined.
    HeapBoundRange,
}

/// Why a run changed guard bytes around a target its device contained as Vulkan allows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EscapeCause {
    /// `robustBufferAccess` alone contained the target ([`Check::Robust`]), and the candidate
    /// placed it in a heap of its own (the runtime path): Vulkan lets an out-of-bounds write change
    /// any byte of the memory bound to the buffer, which for a heap's buffer is the whole heap, so
    /// the write landed in the guard. On Metal such a write changes a heap neighbour's bytes: a
    /// real finding, recorded as an escape.
    HeapBoundMemory,
}

/// The out-of-bounds facts of one candidate run, in the order of the case's targets.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutOfBoundsRun {
    pub targets: Vec<TargetRun>,
}

impl OutOfBoundsRun {
    /// Whether the run's output may differ from Metal's row and still match: a target whose
    /// out-of-bounds read may return a non-zero value ([`read_rule`]) where the row records zero
    /// ([`metal_recorded`]), or, under robust buffer access alone, which a write past an output
    /// target may change. Which bytes may differ, and to what, [`matches()`] decides over the case.
    pub fn tolerates_differences(&self) -> bool {
        self.targets
            .iter()
            .any(|target| target.read_rule() != ReadRule::Zero)
    }

    /// Whether the run changed memory around a target that its containment does not allow: a
    /// guard byte, or a padding byte under anything but robustness 1. Such a run is no match
    /// ([`matches()`]) even when its output equals Metal's.
    pub fn escapes(&self) -> bool {
        self.targets.iter().any(|target| {
            target.escaped.is_some_and(|bytes| bytes > 0)
                || (target.past_end.is_some_and(|bytes| bytes > 0)
                    && target.mechanism.check != Check::Robust)
        })
    }
}

/// What an out-of-bounds read of a target may return.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadRule {
    /// Zero bytes.
    Zero,
    /// Each 32-bit word is zero, a component of Vulkan's `(0, 0, 0, x)`, or a word of the target's
    /// initial bytes (`robustBufferAccess` without the second level, on a target the module
    /// does not write).
    ZeroOrOwnWords,
    /// Any value.
    Any,
    /// Each 32-bit word is zero (the null handle's texel) or a word of the initial bytes of a
    /// texture the case binds (a handle the table resolves).
    ResidentTexels,
    /// Each 32-bit word is a word of the memory of the heap of its own the candidate placed the
    /// target in (the target's initial bytes, the padding fill past its length, the guard fill
    /// around it), or, past the heap's end, zero or a component of Vulkan's `(0, 0, 0, x)`
    /// ([`ReadCause::HeapBoundRange`]).
    HeapWords,
}

/// What an out-of-bounds read of a target of `form` contained by `mechanism` may return, where
/// the module writes the target if `written`: the mechanism's rule, which a run's reads are judged
/// by unless where the candidate placed the target says otherwise ([`TargetRun::read_rule`]).
pub fn read_rule(form: ContainmentForm, mechanism: Mechanism, written: bool) -> ReadRule {
    match (mechanism.medium, mechanism.check) {
        // An id past the handle array is another handle of the argument buffer, which the table
        // resolves to a resident texture or the null handle. An index past a top-level array is
        // clamped onto its last element, a texture the case binds.
        (Medium::HandleTable | Medium::HandleArray, _) => ReadRule::ResidentTexels,
        // An index read past the buffer is index 0 from the padded copy (and was under robust
        // buffer access 2, and the binder drew none); the case's in-range indices include index 0
        // (its checker rule), so none reaches a byte the in-range indices do not, and the declared
        // read bytes, which only a vertex id outside the buffer's indices reaches, stay zero.
        (Medium::IndexBuffer, _) | (_, Check::Binder | Check::Padded) => ReadRule::Zero,
        // An in-shader check yields zero for a load and for a skipped atomic; a device address, the
        // arena, and a typed object take it in every leg.
        (Medium::DeviceAddress | Medium::Arena | Medium::TypedObject, _)
        | (_, Check::Shader | Check::ShaderReads) => ReadRule::Zero,
        // Robust buffer access 2: a load returns zero, an atomic an undefined value.
        (_, Check::Robust2) if form == ContainmentForm::BufferAtomic => ReadRule::Any,
        (_, Check::Robust2) => ReadRule::Zero,
        // Robust buffer access: a value of the buffer's memory, zero, or (0, 0, 0, x); a write may
        // have changed the memory, and an atomic returns an undefined value.
        (_, Check::Robust) if written || form == ContainmentForm::BufferAtomic => ReadRule::Any,
        (_, Check::Robust) => ReadRule::ZeroOrOwnWords,
    }
}

/// The `x` of Vulkan's `(0, 0, 0, x)` for a 4-component out-of-bounds read under robust buffer
/// access: 0, 1, or the largest positive integer for integer components (signed or unsigned), and
/// 0.0 or 1.0 for float components.
const VECTOR_FILLS: [u32; 5] = [0, 1, 0x7fff_ffff, 0xffff_ffff, 0x3f80_0000];

/// How a compiled module contains one out-of-bounds target ([`containment`]): the form the
/// module exercises on it, its mechanism, and whether the module writes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Contained {
    pub form: ContainmentForm,
    pub mechanism: Mechanism,
    pub written: bool,
}

/// How a compiled module fetches one attribute: by Vulkan's fixed-function vertex input, or in the
/// shader from the Metal buffer slot its fetch names (a pulled vertex or tessellation attribute, or
/// a kernel stage input).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fetch {
    FixedFunction,
    Pulled { buffer_index: u32 },
}

impl Fetch {
    /// Each attribute `function` fetches, by Metal attribute index
    /// (`CompiledFunction::attribute_fetches`): its fixed-function vertex attributes, its pulled
    /// vertex and tessellation attributes, and its kernel stage inputs, each with the buffer its
    /// fetch names.
    pub fn of_function(function: &CompiledFunction) -> BTreeMap<u32, Fetch> {
        function
            .attribute_fetches()
            .into_iter()
            .map(|(index, fetch)| {
                let fetch = match fetch {
                    AttributeFetch::FixedFunction => Fetch::FixedFunction,
                    AttributeFetch::Pulled { buffer_index } => Fetch::Pulled { buffer_index },
                };
                (index, fetch)
            })
            .collect()
    }

    /// Each stage input a kernel compiled under stage-input descriptor `descriptor` fetches, by
    /// Metal attribute index: in the shader, from the buffer its attribute names.
    pub fn of_stage_input(descriptor: Option<&StageInputDescriptor>) -> BTreeMap<u32, Fetch> {
        descriptor
            .iter()
            .flat_map(|descriptor| &descriptor.attributes)
            .map(|(&location, attribute)| {
                (
                    location,
                    Fetch::Pulled {
                        buffer_index: attribute.buffer_index,
                    },
                )
            })
            .collect()
    }

    /// Each attribute a render pipeline of vertex input state `state` fetches, by Metal attribute
    /// index, of the attributes `buffers` names with the Metal buffer each one's data is bound at
    /// (one attribute per buffer, as a case binds them): pulled in the shader from the buffer
    /// its pulled attribute names, or by fixed-function vertex input where a binding of the state
    /// fetches its buffer.
    pub fn of_vertex_input(
        state: &VertexInputState,
        buffers: impl IntoIterator<Item = (u32, u32)>,
    ) -> BTreeMap<u32, Fetch> {
        buffers
            .into_iter()
            .filter_map(|(location, buffer_index)| {
                let pulled = state
                    .pulled
                    .iter()
                    .find(|pulled| pulled.attribute == location)
                    .map(|pulled| Fetch::Pulled {
                        buffer_index: pulled.buffer_index,
                    });
                let fixed = state
                    .bindings
                    .iter()
                    .any(|binding| binding.buffer_index == buffer_index)
                    .then_some(Fetch::FixedFunction);
                Some((location, pulled.or(fixed)?))
            })
            .collect()
    }
}

/// What [`containment`] reads of one compiled module: the stage it was compiled for, its Metal
/// argument slots, how it fetches each attribute, whether it reads the resident root table and
/// resolves handles through the handle table, and the census of the strategies its lowering used.
#[derive(Clone, Copy, Debug)]
pub struct ModuleFacts<'a> {
    pub stage: SymStage,
    pub slots: &'a BTreeMap<SlotRef, SlotInterface>,
    /// How the module (for a render pipeline, the pipeline's vertex input with its module) fetches
    /// each attribute, by Metal attribute index ([`Fetch`]).
    pub attributes: &'a BTreeMap<u32, Fetch>,
    /// The module reads the resident root table: it has a device-address slot or a dynamic root
    /// (`needs_pointer_translation`).
    pub resident_roots: bool,
    /// The module resolves texture or sampler handles through the handle table
    /// (`needs_handle_translation`).
    pub handle_table: bool,
    pub strategies: &'a StrategyCensus,
}

impl ModuleFacts<'_> {
    fn uses(&self, strategy: &str) -> bool {
        self.strategies.entries.contains_key(strategy)
    }

    fn uses_family(&self, prefix: &str) -> bool {
        self.strategies
            .entries
            .keys()
            .any(|name| name.starts_with(prefix))
    }
}

/// The product census names (`StrategyCensus` keys) [`containment`] reads: what a pointer loaded
/// from memory (a dynamic root) does, and how the module keeps thread-local memory (a typed
/// variable checked per array level, or the private arena checked as a whole) and resolves an
/// argument buffer's texture handles.
mod census {
    /// A pointer loaded from memory: the dynamic root's creation (qualified by the load's reach).
    pub const LOAD_POINTER: &str = "access.dynamic.load_pointer";
    /// A load and a store through a dynamic root.
    pub const DYNAMIC_LOAD: &str = "access.dynamic.load";
    pub const DYNAMIC_STORE: &str = "access.dynamic.store";
    /// A typed thread-local load and store at a dynamic index, checked against each array level.
    pub const TYPED_LOAD_CLAMPED: &str = "access.typed.load.clamped";
    pub const TYPED_STORE_CLAMPED: &str = "access.typed.store.clamped";
    /// Thread-local memory in the private arena.
    pub const PRIVATE: &str = "access.private.";
    /// A texture handle loaded from an argument buffer, looked up in the handle table.
    pub const HANDLE_LOOKUP: &str = "access.handle.lookup";
    /// A handle loaded from a top-level handle array at a dynamic index clamped onto its elements.
    pub const HANDLE_ARRAY_LOAD_CLAMPED: &str = "access.handle.array_load.clamped";
}

/// How the compiled `module` contains `target`, compiled for `profile`, whose buffer and image
/// robustness are the leg's (`Robustness::Rba2` for `robustBufferAccess2`;
/// `robust_image_access2` for `robustImageAccess2`), read through the product's own containment
/// predicates (the `DeviceProfile` containment facts).
///
/// The form is the structure's, never only the case's word: a buffer is a `uniform_buffer` when
/// the module views it as a uniform buffer, a `storage_buffer` when it views it as a storage
/// buffer, and a `device_address` when it reaches it by its address, except that a
/// `buffer_atomic` target (one an AIR atomic accesses, which the case checker establishes) stays
/// one in every medium. A buffer an argument buffer points to is a `loaded_pointer` when the
/// module loads a pointer and accesses memory through it, thread memory a `sized_array` when the
/// module indexes a typed thread-local array at a dynamic index, and an argument buffer's texture
/// array a `handle_array_index` when the module looks its handles up in the handle table. `Err`
/// when the module does not access the target, reaches it in a way validation does not know (thread
/// memory in the private arena, whose check is the arena's bounds, not the array's), or contains
/// nothing (`Containment::UnsafeOff`, which validation never compiles).
pub fn containment(
    target: &OutOfBoundsTarget,
    module: &ModuleFacts<'_>,
    profile: &DeviceProfile,
) -> Result<Contained, String> {
    let ModuleFacts { stage, slots, .. } = *module;
    let robust = robust_check(profile)?;
    // A descriptor view's atomic is checked in the shader where the driver's robustness leaves
    // buffer atomics uncontained (`DeviceProfile::checks_atomics_in_shader`).
    let atomic_check = if profile.checks_atomics_in_shader() {
        Check::Shader
    } else {
        robust
    };
    let slot = |class, index| {
        slots
            .get(&SlotRef {
                stage,
                class,
                index,
            })
            .ok_or_else(|| {
                format!(
                    "the compiled module does not access {}",
                    target.resource.label()
                )
            })
    };
    let made = |form, medium, check, written| {
        Ok(Contained {
            form,
            mechanism: Mechanism { medium, check },
            written,
        })
    };
    let unknown = || {
        Err(format!(
            "{} has a buffer form validation does not know",
            target.resource.label()
        ))
    };
    match target.resource {
        OutOfBoundsResource::Buffer { binding } => match slot(SlotClass::Buffer, binding)? {
            SlotInterface::Buffer(buffer) => {
                let atomic = target.form == ContainmentForm::BufferAtomic;
                let written = buffer.access.write;
                match buffer.form {
                    BufferForm::Psb if atomic => made(
                        ContainmentForm::BufferAtomic,
                        Medium::DeviceAddress,
                        Check::Shader,
                        written,
                    ),
                    BufferForm::Psb => made(
                        ContainmentForm::DeviceAddress,
                        Medium::DeviceAddress,
                        Check::Shader,
                        written,
                    ),
                    BufferForm::Descriptor(view) => {
                        let (form, check) = match view {
                            DescriptorView::Storage if atomic => {
                                (ContainmentForm::BufferAtomic, atomic_check)
                            }
                            DescriptorView::Storage => (ContainmentForm::StorageBuffer, robust),
                            DescriptorView::UniformWords | DescriptorView::UniformVec4 => {
                                (ContainmentForm::UniformBuffer, robust)
                            }
                            _ => return unknown(),
                        };
                        made(form, Medium::DescriptorView, check, written)
                    }
                    _ => unknown(),
                }
            }
            _ => Err(format!("{} is not a buffer slot", target.resource.label())),
        },
        OutOfBoundsResource::ThreadgroupMemory { .. } => made(
            ContainmentForm::Threadgroup,
            Medium::Arena,
            Check::Shader,
            true,
        ),
        OutOfBoundsResource::ArgumentBufferBuffer { buffer_binding, .. } => {
            slot(SlotClass::Buffer, buffer_binding)?;
            let accessed = module.uses(census::DYNAMIC_LOAD) || module.uses(census::DYNAMIC_STORE);
            if !module.resident_roots || !module.uses_family(census::LOAD_POINTER) || !accessed {
                return Err(format!(
                    "the compiled module accesses no memory through a pointer loaded from {}",
                    target.resource.label()
                ));
            }
            // The loaded pointer resolves to the resident allocation holding it, and each access is
            // checked against that allocation's size in the shader, in every leg.
            made(
                ContainmentForm::LoadedPointer,
                Medium::DeviceAddress,
                Check::Shader,
                module.uses(census::DYNAMIC_STORE),
            )
        }
        OutOfBoundsResource::ThreadMemory => {
            if module.uses_family(census::PRIVATE) {
                return Err(
                    "the compiled module keeps thread memory in the private arena, whose check \
                     is the arena's bounds, not the array's"
                        .into(),
                );
            }
            let load = module.uses(census::TYPED_LOAD_CLAMPED);
            let store = module.uses(census::TYPED_STORE_CLAMPED);
            if !load && !store {
                return Err(
                    "the compiled module indexes no thread-local array at a dynamic index".into(),
                );
            }
            made(
                ContainmentForm::SizedArray,
                Medium::TypedObject,
                Check::Shader,
                store,
            )
        }
        OutOfBoundsResource::ArgumentBufferTextureArray { buffer_binding, .. } => {
            slot(SlotClass::Buffer, buffer_binding)?;
            if !module.handle_table || !module.uses_family(census::HANDLE_LOOKUP) {
                return Err(format!(
                    "the compiled module looks no handle of {} up in the handle table",
                    target.resource.label()
                ));
            }
            made(
                ContainmentForm::HandleArrayIndex,
                Medium::HandleTable,
                Check::Shader,
                false,
            )
        }
        OutOfBoundsResource::AttributeBuffer { location } => {
            // Fixed-function vertex input is left to the device's buffer robustness, and a pulled
            // fetch reads its buffer's slot, contained as that slot is.
            match module.attributes.get(&location).copied() {
                None => Err(format!(
                    "the compiled module fetches no attribute {location}"
                )),
                Some(Fetch::FixedFunction) if robust == Check::Shader => Err(format!(
                    "attribute {location} is fixed-function vertex input on a device without \
                     buffer robustness, which contains nothing"
                )),
                Some(Fetch::FixedFunction) => made(target.form, Medium::VertexInput, robust, false),
                Some(Fetch::Pulled { buffer_index }) => {
                    match slot(SlotClass::Buffer, buffer_index)? {
                        SlotInterface::Buffer(buffer) => match buffer.form {
                            BufferForm::Psb => {
                                made(target.form, Medium::DeviceAddress, Check::Shader, false)
                            }
                            BufferForm::Descriptor(
                                DescriptorView::Storage
                                | DescriptorView::UniformWords
                                | DescriptorView::UniformVec4,
                            ) => made(target.form, Medium::DescriptorView, robust, false),
                            _ => unknown(),
                        },
                        _ => Err(format!(
                            "attribute {location}'s buffer {buffer_index} is not a buffer slot"
                        )),
                    }
                }
            }
        }
        OutOfBoundsResource::IndexBuffer => {
            if stage != SymStage::Vertex {
                return Err("an index buffer is a vertex stage's draw's".into());
            }
            // Vulkan states no index past the bound range, robust or not, so on every device the
            // draw reads a copy of its indices padded with zeros.
            made(
                ContainmentForm::IndexBuffer,
                Medium::IndexBuffer,
                Check::Padded,
                false,
            )
        }
        OutOfBoundsResource::TextureArray { binding } => {
            slot(SlotClass::Texture, binding)?;
            if !module.uses(census::HANDLE_ARRAY_LOAD_CLAMPED) {
                return Err(format!(
                    "the compiled module clamps no index into {}",
                    target.resource.label()
                ));
            }
            made(
                ContainmentForm::HandleArrayIndex,
                Medium::HandleArray,
                Check::Shader,
                false,
            )
        }
        OutOfBoundsResource::Texture { binding } => match slot(SlotClass::Texture, binding)? {
            // The product's texel containment: a texel buffer is left to the device only where both
            // its reads and its writes are Metal's there (`robustBufferAccess2`).
            SlotInterface::Texture(texture) if target.form == ContainmentForm::TexelBuffer => {
                let device =
                    profile.device_reads_texels(true) && profile.device_drops_texel_writes(true);
                made(
                    ContainmentForm::TexelBuffer,
                    Medium::TexelBuffer,
                    if device {
                        Check::Robust2
                    } else {
                        Check::Shader
                    },
                    texture.access.write,
                )
            }
            // An image's writes are left to `robustImageAccess` at either level, its reads only
            // to the second level.
            SlotInterface::Texture(texture) => {
                let check = if !profile.device_drops_texel_writes(false) {
                    Check::Shader
                } else if profile.device_reads_texels(false) {
                    Check::Robust2
                } else {
                    Check::ShaderReads
                };
                made(
                    ContainmentForm::Texture,
                    Medium::Image,
                    check,
                    texture.access.write,
                )
            }
            _ => Err(format!("{} is not a texture slot", target.resource.label())),
        },
    }
}

/// The check `profile`'s leg gives every access its compiled modules leave to the device's buffer
/// robustness (the two encodings): a descriptor view's loads and stores (and atomics, unless
/// `DeviceProfile::checks_atomics_in_shader`) and fixed-function vertex input.
/// [`Check::Robust2`] under `robustBufferAccess2`, [`Check::Robust`] under `robustBufferAccess`
/// alone, and [`Check::Shader`] without buffer robustness; `Err` for a containment validation
/// never compiles (`Containment::UnsafeOff`) or does not know. The only source of
/// [`Check::Robust`]: every other medium's check is the shader's, the binder's, or
/// `robustBufferAccess2`'s.
pub fn robust_check(profile: &DeviceProfile) -> Result<Check, String> {
    match profile.containment() {
        Containment::UnsafeOff => {
            Err("containment is off: nothing contains an out-of-bounds access".into())
        }
        Containment::On {
            robust: Robustness::Rba2,
        } => Ok(Check::Robust2),
        Containment::On { .. } => Ok(if profile.checks_buffers_in_shader() {
            Check::Shader
        } else {
            Check::Robust
        }),
        _ => Err("a containment validation does not know".into()),
    }
}

/// The forms among `targets` that `class` leaves uncontained
/// ([`ExecutionClass::robust_buffers_uncontained`]): a run there would access memory outside the
/// target, so the candidate refuses it before any submission, a recorded driver limitation.
pub fn uncontained(
    targets: &[TargetRun],
    class: Option<ExecutionClass>,
) -> BTreeSet<RobustBufferForm> {
    let Some(class) = class else {
        return BTreeSet::new();
    };
    let limited = class.robust_buffers_uncontained();
    targets
        .iter()
        .filter_map(|target| target.mechanism.robust_buffer_form(target.form))
        .filter(|form| limited.contains(form))
        .collect()
}

/// Whether a candidate observation `candidate` of `case` matches its Metal observation `golden`
/// under `class`, with the out-of-bounds facts `run` of the run for a case that declares targets. A
/// case without targets compares as every case does.
pub fn matches(
    case: &AuthoredCase,
    class: &ComparisonClass,
    golden: &[u8],
    candidate: &[u8],
    run: Option<&OutOfBoundsRun>,
) -> bool {
    let lanes = ObservationLanes::of(case);
    let Some(declaration) = &case.out_of_bounds else {
        return class.matches(lanes, golden, candidate);
    };
    let Some(run) = run else {
        return false;
    };
    if run.targets.len() != declaration.targets.len() || golden.len() != candidate.len() {
        return false;
    }
    for (declared, ran) in declaration.targets.iter().zip(&run.targets) {
        if declared.resource != ran.resource || !contained_in_guard(declared, ran) {
            return false;
        }
    }
    let mut compared = candidate.to_vec();
    // Under robust buffer access alone a write past a target may change any byte of its memory:
    // an output that is such a target is not compared.
    if let OutputSelection::Buffer { binding, .. } = case.output {
        let loose = run.targets.iter().any(|target| {
            target.resource == (OutOfBoundsResource::Buffer { binding })
                && target.mechanism.check == Check::Robust
                && target.written
        });
        if loose {
            compared.copy_from_slice(golden);
        }
    }
    for read in &declaration.reads {
        let (Ok(start), Ok(end)) = (
            usize::try_from(read.offset),
            usize::try_from(read.offset.saturating_add(read.length)),
        ) else {
            return false;
        };
        let (Some(target), Some(declared), Some(bytes)) = (
            run.targets.get(read.target as usize),
            declaration.targets.get(read.target as usize),
            candidate.get(start..end),
        ) else {
            return false;
        };
        let allowed = match target.read_rule() {
            ReadRule::Zero => bytes.iter().all(|&byte| byte == 0),
            ReadRule::Any => true,
            ReadRule::ResidentTexels => {
                let texels = texture_words(case);
                bytes.as_chunks::<4>().0.iter().all(|word| {
                    let word = u32::from_le_bytes(*word);
                    word == 0 || texels.contains(&word)
                }) && bytes.len() % 4 == 0
            }
            ReadRule::ZeroOrOwnWords => {
                // The memory bound to the target past its end holds the padding fill.
                let own = own_words(case, declared.resource);
                let padding = u32::from_ne_bytes([PAD_FILL; 4]);
                bytes.as_chunks::<4>().0.iter().all(|word| {
                    let word = u32::from_le_bytes(*word);
                    VECTOR_FILLS.contains(&word) || own.contains(&word) || word == padding
                })
            }
            ReadRule::HeapWords => {
                let own = own_words(case, declared.resource);
                let fills = [PAD_FILL, GUARD_FILL].map(|fill| u32::from_ne_bytes([fill; 4]));
                bytes.as_chunks::<4>().0.iter().all(|word| {
                    let word = u32::from_le_bytes(*word);
                    VECTOR_FILLS.contains(&word) || own.contains(&word) || fills.contains(&word)
                }) && bytes.len() % 4 == 0
            }
        };
        if !allowed {
            return false;
        }
        compared[start..end].copy_from_slice(&golden[start..end]);
    }
    class.matches(lanes, golden, &compared)
}

/// Whether the run's guard around `ran` shows nothing escaped: a guarded target ([`guarded`]) is
/// host-visible memory the candidate guards, so its run must have counted no changed guard byte,
/// and no changed byte of the memory bound to it past its end unless robustness 1 contains it
/// (which may write anywhere in that memory). The threadgroup arena, thread memory, an image, and
/// a handle array have no guard.
fn contained_in_guard(declared: &OutOfBoundsTarget, ran: &TargetRun) -> bool {
    let guarded = guarded(declared);
    if !guarded {
        return ran.escaped.is_none() && ran.past_end.is_none();
    }
    ran.escaped == Some(0)
        && ran
            .past_end
            .is_some_and(|bytes| bytes == 0 || ran.mechanism.check == Check::Robust)
}

/// Whether the candidate allocates `target` inside a guard: a buffer, a buffer an argument
/// buffer points to, an attribute buffer, an index buffer, or a texture buffer, each host-visible
/// buffer memory.
pub fn guarded(target: &OutOfBoundsTarget) -> bool {
    matches!(
        target.resource,
        OutOfBoundsResource::Buffer { .. }
            | OutOfBoundsResource::ArgumentBufferBuffer { .. }
            | OutOfBoundsResource::AttributeBuffer { .. }
            | OutOfBoundsResource::IndexBuffer
    ) || target.form == ContainmentForm::TexelBuffer
}

/// The 32-bit words of `resource`'s initial bytes in `case`, at every 4-byte-aligned offset: a
/// buffer's, or an attribute buffer's (a vertex input, a tessellation input, or a kernel stage
/// input), the resources a read under robust buffer access alone may return a word of.
fn own_words(case: &AuthoredCase, resource: OutOfBoundsResource) -> BTreeSet<u32> {
    match resource {
        OutOfBoundsResource::Buffer { binding } => words(
            case.buffers
                .iter()
                .find(|buffer| buffer.binding == binding)
                .and_then(|buffer| {
                    buffer
                        .bytes_b64
                        .as_deref()
                        .or(buffer.initial_bytes_b64.as_deref())
                }),
        ),
        OutOfBoundsResource::AttributeBuffer { location } => {
            let tessellation = case
                .tessellation
                .iter()
                .flat_map(|draw| draw.control_points.iter().chain(&draw.patch_inputs));
            words(
                case.vertex_inputs
                    .iter()
                    .chain(tessellation)
                    .chain(&case.kernel_stage_inputs)
                    .find(|input| input.location == location)
                    .map(|input| input.bytes_b64.as_str()),
            )
        }
        _ => BTreeSet::new(),
    }
}

/// The 32-bit words of the initial bytes of every texture `case` binds, at a slot or in an
/// argument buffer: the texels a handle the table resolves may read.
fn texture_words(case: &AuthoredCase) -> BTreeSet<u32> {
    let bound = case.textures.iter().map(|texture| {
        texture
            .bytes_b64
            .as_deref()
            .or(texture.initial_bytes_b64.as_deref())
    });
    let embedded = case.argument_buffer_textures.iter().map(|texture| {
        texture
            .bytes_b64
            .as_deref()
            .or(texture.initial_bytes_b64.as_deref())
    });
    let arrayed = case
        .texture_arrays
        .iter()
        .flat_map(|array| &array.elements)
        .map(|element| {
            element
                .bytes_b64
                .as_deref()
                .or(element.initial_bytes_b64.as_deref())
        });
    bound
        .chain(embedded)
        .chain(arrayed)
        .flat_map(words)
        .collect()
}

/// The 32-bit words of base64 `encoded` bytes, at every 4-byte-aligned offset.
fn words(encoded: Option<&str>) -> BTreeSet<u32> {
    encoded
        .and_then(|encoded| {
            base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .ok()
        })
        .map(|bytes| {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word))
                .collect()
        })
        .unwrap_or_default()
}

/// The byte the Metal oracle fills a target's shadow with, and the candidate its guard with: a
/// value a contained read never returns (it is not zero) and no case's derivation needs.
pub const GUARD_FILL: u8 = 0xE7;

/// The byte the candidate fills the memory bound to a guarded target past its end with (its
/// memory requirement past its length): the target's own memory, which a read under robustness 1
/// may return, distinct from [`GUARD_FILL`] so such a read is told apart from one that escaped.
pub const PAD_FILL: u8 = 0x5A;

/// The output of a Metal run of `case` as the oracle records it: the bytes of every declared
/// out-of-bounds read zeroed, since Metal leaves them undefined (they read the shadow, whose
/// threadgroup bytes Metal does not even initialize), so every qualification run records the same
/// bytes there.
pub fn metal_recorded(case: &AuthoredCase, mut output: Vec<u8>) -> Vec<u8> {
    if let Some(declaration) = &case.out_of_bounds {
        for read in &declaration.reads {
            let start = usize::try_from(read.offset).unwrap_or(usize::MAX);
            let end =
                usize::try_from(read.offset.saturating_add(read.length)).unwrap_or(usize::MAX);
            if let Some(bytes) = output.get_mut(start..end) {
                bytes.fill(0);
            }
        }
    }
    output
}

/// Each declared out-of-bounds read of `case` in a Metal `output` before [`metal_recorded`]
/// zeroes it: `read=<offset>+<length> bytes=<hex>`, what Metal returned there (a diagnostic of
/// the run, never recorded).
pub fn undefined_reads(case: &AuthoredCase, output: &[u8]) -> Vec<String> {
    let Some(declaration) = &case.out_of_bounds else {
        return Vec::new();
    };
    declaration
        .reads
        .iter()
        .map(|read| {
            let start = usize::try_from(read.offset).unwrap_or(usize::MAX);
            let end =
                usize::try_from(read.offset.saturating_add(read.length)).unwrap_or(usize::MAX);
            let bytes = output.get(start..end).map_or_else(
                || "outside-the-output".to_string(),
                |bytes| bytes.iter().map(|byte| format!("{byte:02x}")).collect(),
            );
            format!("read={}+{} bytes={bytes}", read.offset, read.length)
        })
        .collect()
}

/// The `(reach_before, reach_after)` of the shadow or guard around the buffer at `binding`, when
/// `case` declares it an out-of-bounds target.
pub fn buffer_reach(case: &AuthoredCase, binding: u32) -> Option<(u32, u32)> {
    case.out_of_bounds
        .as_ref()?
        .targets
        .iter()
        .find_map(|target| {
            (target.resource == OutOfBoundsResource::Buffer { binding })
                .then_some((target.reach_before, target.reach_after))
        })
}

/// The `(reach_before, reach_after)` of the shadow or guard around the argument-buffer buffer at
/// `(buffer_binding, field_offset)`, when `case` declares it an out-of-bounds target.
pub fn argument_buffer_reach(
    case: &AuthoredCase,
    (buffer_binding, field_offset): (u32, u32),
) -> Option<(u32, u32)> {
    case.out_of_bounds
        .as_ref()?
        .targets
        .iter()
        .find_map(|target| {
            (target.resource
                == OutOfBoundsResource::ArgumentBufferBuffer {
                    buffer_binding,
                    field_offset,
                })
            .then_some((target.reach_before, target.reach_after))
        })
}

/// The `(reach_before, reach_after)` of the shadow or guard around `resource`, when `case`
/// declares it an out-of-bounds target: an attribute buffer's or the index buffer's, which the
/// case checker keeps at `reach_before` 0, so the shadow lies past the data.
pub fn reach(case: &AuthoredCase, resource: OutOfBoundsResource) -> Option<(u32, u32)> {
    case.out_of_bounds
        .as_ref()?
        .targets
        .iter()
        .find_map(|target| {
            (target.resource == resource).then_some((target.reach_before, target.reach_after))
        })
}

/// The bytes past the threadgroup memory at `binding` the Metal oracle allocates as its shadow,
/// when `case` declares it an out-of-bounds target.
pub fn threadgroup_reach(case: &AuthoredCase, binding: u32) -> u32 {
    case.out_of_bounds
        .as_ref()
        .and_then(|declaration| {
            declaration.targets.iter().find_map(|target| {
                (target.resource == OutOfBoundsResource::ThreadgroupMemory { binding })
                    .then_some(target.reach_after)
            })
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests;
