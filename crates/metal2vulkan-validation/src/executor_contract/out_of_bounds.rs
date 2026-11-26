//! The out-of-bounds containment contract: how a candidate run of a case that declares
//! out-of-bounds targets (`case::out_of_bounds`) is judged.
//!
//! One rule, shared by the engine candidate when it runs a case, by a stored row's
//! re-decision (`CandidateObservation::status_holds`), and by the coverage report's containment
//! rows:
//!
//! - **The form and mechanism** of each target are what its compiled module does with it, never
//!   only what the case says ([`containment`]): a buffer is a storage buffer through its
//!   descriptor, or a device address; a buffer slot the product reaches through a descriptor view,
//!   a slot reached by its device address and the threadgroup arena are checked in the shader in
//!   every leg, except that a kernel whose buffer arguments are bound as storage buffers of set 0
//!   (`ModuleFacts::robust_views`) leaves its views' loads and stores to its pipeline's
//!   `robustBufferAccess2` and checks only their atomics in the shader; a texel buffer and an image follow the product's texel containment
//!   (`DeviceProfile::device_reads_texels` and `device_drops_texel_writes`): a texel buffer is
//!   checked in the shader; an image's reads are left to `robustImageAccess2` and its writes to
//!   `robustImageAccess` at either level, each otherwise checked in the shader. A buffer reached through a pointer loaded from an argument buffer is a
//!   device address the resident set resolves, a thread-local array a typed object whose index is
//!   checked against its length, and an argument buffer's texture array a handle table the loaded
//!   resource id is looked up in: each checked in the shader in every leg. Under the identity
//!   domains (`DevicePolicy`) a loaded pointer is the address it holds, which nothing checks, so
//!   the candidate does not run the case ([`Check::Unchecked`]), and a loaded resource id's index
//!   is masked onto the descriptor arrays, whose elements no live texture owns are null
//!   descriptors. An attribute buffer is fetched by fixed-function vertex input, which nothing
//!   bounds ([`Check::Unchecked`]), or pulled in the shader through the slot its fetch names,
//!   contained as that slot is ([`Fetch`]); a direct
//!   indexed draw's index buffer is bounded by `robustBufferAccess2` or the binder's clamp; and a
//!   top-level texture array's dynamic index is clamped onto its elements in the shader.
//! - **The read rule** ([`read_rule`]) says what an out-of-bounds read of the target may return:
//!   zero under an in-shader check and under `robustBufferAccess2` (Vulkan's "Robust Buffer
//!   Access 2": "reads ... outside of the checked range will return zero values"), anything for
//!   an atomic's return under robust buffer access ("will return an undefined value"), and under
//!   `robustBufferAccess` alone zero, `(0, 0, 0, x)`, or a value of the buffer's own memory. A
//!   read through a handle past a handle array returns a texel of a texture the case binds, or
//!   zero (the null handle or the null descriptor). An index read past an index buffer is index 0
//!   or none, so the vertex ids only a read past it gives stay unshaded.
//! - **Nothing escapes**: every byte of the guard the candidate allocates around a buffer (an
//!   argument buffer's included) or texture buffer target is unchanged ([`TargetRun::escaped`]),
//!   and so is the memory bound to it past its end ([`TargetRun::past_end`]) except under
//!   `robustBufferAccess` alone, where a write may change any byte of the target's own memory, so
//!   the output bytes of a written buffer target are not compared there either. An attribute
//!   buffer and an index buffer are guarded as a buffer is.
//!   An image's memory is opaque: its out-of-bounds writes are judged by its compared contents
//!   alone. A thread-local array and a handle array have no guard: what their accesses past them
//!   reach is the case's own layout, whose output bytes witness it.
//! - Every other output byte compares with Metal under the run's comparison class, as any case's
//!   does ([`matches()`]). A form whose out-of-bounds result Metal defines (a texture or a texture
//!   buffer) has no read bytes: its whole output compares with Metal.

use crate::case::out_of_bounds::{ContainmentForm, OutOfBoundsResource, OutOfBoundsTarget};
use crate::case::{AuthoredCase, OutputSelection};
use crate::observation_contract::{ComparisonClass, ObservationLanes};
use base64::Engine as _;
use metal2vulkan::device::{DevicePolicy, DeviceProfile, HandleDomain, PointerDomain};
use metal2vulkan::library::CompiledFunction;
use metal2vulkan::pipeline::VertexInputState;
use metal2vulkan::pipeline::{AttributeFetch, StrategyCensus};
use metal2vulkan::pipeline::{BufferForm, SlotInterface};
use metal2vulkan::pipeline::{SlotClass, SlotRef, Stage as SymStage};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// The memory an out-of-bounds target's accesses reach in its compiled module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Medium {
    /// A storage buffer descriptor view (`BufferForm::Descriptor`).
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
    /// The descriptor arrays a texture's resource id, loaded from an argument buffer, indexes
    /// under the identity handle domain (`HandleDomain::Identity`), its index masked onto the
    /// arrays' elements: an element no live texture owns is the null descriptor.
    ResourceId,
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
    /// The device's `robustBufferAccess` without the second level. No executor contains a target
    /// so now; committed records of earlier runs state it.
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
    /// Nothing checks the access: neither the shader nor the device's robustness bounds it (a
    /// pointer loaded from memory under the identity pointer domain, `PointerDomain::Identity`,
    /// is the device address it holds, used as given, and fixed-function vertex input on a device
    /// without buffer robustness fetches past its buffer what the device gives). A run would reach
    /// memory no binding gave the case, so the candidate refuses the case before it makes the
    /// pipeline (`NotSupported::Uncontained`).
    Unchecked,
}

/// How a compiled module contains one out-of-bounds target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mechanism {
    pub medium: Medium,
    pub check: Check,
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
    /// requirement past its length in an allocation of its own, filled with [`PAD_FILL`]. Only
    /// robustness 1 may write there ("within the memory range(s) bound to the buffer"); `None`
    /// without a guard.
    #[serde(default)]
    pub past_end: Option<u64>,
}

impl TargetRun {
    /// What an out-of-bounds read of the target may return in this run: its mechanism's rule
    /// ([`read_rule`]).
    pub fn read_rule(&self) -> ReadRule {
        read_rule(self.form, self.mechanism, self.written)
    }
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
}

/// What an out-of-bounds read of a target of `form` contained by `mechanism` may return, where
/// the module writes the target if `written`: the rule a run's reads are judged by
/// ([`TargetRun::read_rule`]).
pub fn read_rule(form: ContainmentForm, mechanism: Mechanism, written: bool) -> ReadRule {
    match (mechanism.medium, mechanism.check) {
        // An unchecked access reads whatever the device holds there; the candidate never runs one.
        (_, Check::Unchecked) => ReadRule::Any,
        // An id past the handle array is another handle of the argument buffer, which the table
        // resolves to a resident texture or the null handle, and which the identity handle domain
        // masks onto the descriptor arrays, whose elements no live texture owns are null
        // descriptors. An index past a top-level array is clamped onto its last element, a texture
        // the case binds.
        (Medium::HandleTable | Medium::ResourceId | Medium::HandleArray, _) => {
            ReadRule::ResidentTexels
        }
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
/// resolves handles through the handle table, the domains it was compiled under, and the census of
/// the strategies its lowering used.
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
    /// The pointer and handle domains of the device the module was compiled for
    /// (`Device::policy`).
    pub policy: DevicePolicy,
    pub strategies: &'a StrategyCensus,
    /// The module's storage views are contained by its pipeline's robust buffer access 2, not by
    /// a check in the shader: a kernel whose buffer arguments are bound as storage buffers of set
    /// 0, which requires the pipeline-robustness feature (`VkFeature::PipelineRobustness` in its
    /// requirements). Its atomics keep their check in the shader.
    pub robust_views: bool,
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

/// How the compiled `module` contains `target`, compiled for `profile`, whose image robustness is
/// the leg's (`robust_image_access2` for `robustImageAccess2`), read through the product's own
/// containment predicates (the `DeviceProfile` containment facts).
///
/// The form is the structure's, never only the case's word: a buffer is a `uniform_buffer` when
/// the module views it as a uniform buffer, a `storage_buffer` when it views it as a storage
/// buffer, and a `device_address` when it reaches it by its address, except that a
/// `buffer_atomic` target (one an AIR atomic accesses, which the case checker establishes) stays
/// one in every medium. A buffer an argument buffer points to is a `loaded_pointer` when the
/// module loads a pointer and accesses memory through it, which the resident table checks and the
/// identity pointer domain leaves unchecked ([`Check::Unchecked`]), thread memory a `sized_array`
/// when the module indexes a typed thread-local array at a dynamic index, and an argument buffer's
/// texture array a `handle_array_index` when the module looks its handles up in the handle table,
/// or, under the identity handle domain, masks their ids onto the descriptor arrays. `Err`
/// when the module does not access the target, reaches it in a way validation does not know (thread
/// memory in the private arena, whose check is the arena's bounds, not the array's).
pub fn containment(
    target: &OutOfBoundsTarget,
    module: &ModuleFacts<'_>,
    profile: &DeviceProfile,
) -> Result<Contained, String> {
    let ModuleFacts { stage, slots, .. } = *module;
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
                    BufferForm::Descriptor if atomic => made(
                        ContainmentForm::BufferAtomic,
                        Medium::DescriptorView,
                        Check::Shader,
                        written,
                    ),
                    BufferForm::Descriptor => made(
                        ContainmentForm::StorageBuffer,
                        Medium::DescriptorView,
                        match module.robust_views {
                            true => Check::Robust2,
                            false => Check::Shader,
                        },
                        written,
                    ),
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
            // Under the resident table the loaded pointer resolves to the resident allocation
            // holding it, and each access is checked against that allocation's size in the shader,
            // in every leg. Under the identity pointer domain it is the device address it holds,
            // used as given: nothing checks the access.
            let (resolved, check) = match module.policy.pointer_domain {
                PointerDomain::ResidentTable => (module.resident_roots, Check::Shader),
                PointerDomain::Identity => (true, Check::Unchecked),
                _ => return Err("a pointer domain validation does not know".into()),
            };
            if !resolved || !module.uses_family(census::LOAD_POINTER) || !accessed {
                return Err(format!(
                    "the compiled module accesses no memory through a pointer loaded from {}",
                    target.resource.label()
                ));
            }
            made(
                ContainmentForm::LoadedPointer,
                Medium::DeviceAddress,
                check,
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
            // The handle table resolves a loaded id to a resident texture or the null handle; the
            // identity handle domain masks the id's index onto the descriptor arrays.
            let (resolved, medium) = match module.policy.handle_domain {
                HandleDomain::Table => (module.handle_table, Medium::HandleTable),
                HandleDomain::Identity => (true, Medium::ResourceId),
                _ => return Err("a handle domain validation does not know".into()),
            };
            if !resolved || !module.uses_family(census::HANDLE_LOOKUP) {
                return Err(format!(
                    "the compiled module resolves no handle of {}",
                    target.resource.label()
                ));
            }
            made(
                ContainmentForm::HandleArrayIndex,
                medium,
                Check::Shader,
                false,
            )
        }
        OutOfBoundsResource::AttributeBuffer { location } => {
            // Nothing bounds fixed-function vertex input (no buffer robustness is enabled), and a
            // pulled fetch reads its buffer's slot, contained as that slot is.
            match module.attributes.get(&location).copied() {
                None => Err(format!(
                    "the compiled module fetches no attribute {location}"
                )),
                Some(Fetch::FixedFunction) => {
                    made(target.form, Medium::VertexInput, Check::Unchecked, false)
                }
                Some(Fetch::Pulled { buffer_index }) => {
                    match slot(SlotClass::Buffer, buffer_index)? {
                        SlotInterface::Buffer(buffer) => match buffer.form {
                            BufferForm::Psb => {
                                made(target.form, Medium::DeviceAddress, Check::Shader, false)
                            }
                            BufferForm::Descriptor => {
                                made(target.form, Medium::DescriptorView, Check::Shader, false)
                            }
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
            // The product's texel containment: the shader checks every texel buffer access
            // (`DeviceProfile::device_reads_texels`).
            SlotInterface::Texture(texture) if target.form == ContainmentForm::TexelBuffer => made(
                ContainmentForm::TexelBuffer,
                Medium::TexelBuffer,
                Check::Shader,
                texture.access.write,
            ),
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
pub(crate) mod tests;
