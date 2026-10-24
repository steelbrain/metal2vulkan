//! A heap and the resources placed in it, as both sides describe and read one: `MTLHeap`'s own
//! getters and `MTLResource`'s, over an automatic and a placement heap.
//!
//! [`crate::sparse_query`] already spells a **sparse** heap and the three getters a mapping case
//! counts in. This is the rest of the object: the eight properties a heap answers about itself, the
//! two calls that place a buffer in one, the purgeable-state call, and the `MTLResource` surface
//! every buffer, texture and view answers — its heap and offset, its aliasing, its label and its
//! purgeable state.
//!
//! Every call here is a **pure query or a plain allocation**: nothing is encoded, committed or
//! dispatched, and no case reaches a queue. The one call that writes anything is
//! `setPurgeableState:` with `Empty`, which Metal may discard the resource's bytes for — so a case
//! asks it **last** of whatever it asks it of, and nothing is read through the object afterwards
//! but the state call's own answer.
//!
//! # Why a value read back is not the whole of a case here
//!
//! A descriptor field read back off the descriptor it was just set on is a value the case already
//! knew, and a case built of those passes without Metal deciding anything. So the facts recorded
//! here are the ones **Metal and only Metal** settles:
//!
//! - `size` is the descriptor's size **rounded**, and `currentAllocatedSize` and `usedSize` are
//!   figures of Metal's own allocator, not the case's;
//! - `hazardTrackingMode` of a heap whose descriptor said `Default` is the one Metal **resolves**
//!   it to, which the runtime models as untracked;
//! - `maxAvailableSizeWithAlignment:` is the longest buffer Metal's suballocator will place, which
//!   moves as buffers are placed and is 0 for a heap that suballocates nothing;
//! - a placed buffer's `heapOffset` is where Metal's suballocator put it, which a case that states
//!   no offset cannot know;
//! - `label` of an object nothing has labelled is Metal's answer, and so is the label of a label
//!   set to the empty string;
//! - `setPurgeableState:` answers the state **before** the call, so a chain of them is a reading of
//!   Metal's own transition rule rather than of the arguments the case handed.
//!
//! [`crate::cases::heap_facts`] and [`crate::cases::resource_facts`] state per case which of those
//! each row is for, and each family carries a positive control: a value nothing else in the case
//! leaves, so a run that measured nothing cannot read as a pass.
//!
//! # What this module leaves out, and why
//!
//! - **A heap over the caller's own bytes.** There is none: this round measured that
//!   `MTLHeapDescriptor` answers none of `setContents:`, `setNoCopy:` or `setDeallocator:`, and that
//!   `MTLDevice` does not respond to `newHeapWithBytesNoCopy:length:options:deallocator:`. The
//!   runtime's `Device::new_heap_with_bytes_no_copy` is a runtime extension with no Metal
//!   counterpart, so no case can compare it.
//! - **An `alignment` of 0** for `maxAvailableSizeWithAlignment:`. The runtime's model counts it as
//!   1, but nothing in this tree says Metal's suballocator takes it, and the answer our model gives
//!   is not evidence that it does — the same reason [`crate::device_queries`] leaves a length of 0
//!   out of the heap-buffer sweep.
//! - **An alignment that is not a power of two.** The runtime refuses it `Invalid`, which is this
//!   tree's reading of "Metal's own validation stops it, ending the process".
//! - **`newTextureWithDescriptor:` of an automatic heap, and the offset forms.** Those three are
//!   already exercised ([`crate::cases::sparse_heaps`], [`crate::cases::texture_fills`],
//!   [`crate::cases::two_plane_heap_blits`]); this module adds the two the suite never called.
//! - **Reading a resource's bytes after `Empty`.** The runtime models what Metal does to a shared
//!   heap's bytes there, and it is a fact worth a case, but it is a case about **contents** and so
//!   about a shape where a run that discarded nothing and a run that discarded everything can be
//!   told apart. Nothing here reads bytes.

use crate::case::Handle;
use crate::texture_query::Storage;
use crate::transcript::{Node, NodeName, Observed, Value};
use metal2vulkan_runtime::{CpuCacheMode, HazardTrackingMode, HeapType, PurgeableState};

/// The CPU cache modes a case of this module names, by Metal's own enumeration.
///
/// Both of them, because `cpuCacheMode` is one of the properties the sweep is about and a heap
/// descriptor that never moved it would record one answer twice.
pub const CACHE_MODES: [CpuCacheMode; 2] =
    [CpuCacheMode::DefaultCache, CpuCacheMode::WriteCombined];

/// The hazard tracking modes a case names, by Metal's own enumeration.
///
/// `Default` is in it and is the interesting one: Metal **resolves** it, and what it resolves to
/// for a heap is a fact no descriptor states.
pub const HAZARD_MODES: [HazardTrackingMode; 3] = [
    HazardTrackingMode::Default,
    HazardTrackingMode::Untracked,
    HazardTrackingMode::Tracked,
];

/// The word a CPU cache mode carries in a site's name.
///
/// Each of the runtime's four enumerations here is `#[non_exhaustive]`, so a mode Metal gains is
/// named by its raw value rather than silently sharing a site with one of these — the spelling
/// [`crate::sparse_query::page_word`] already uses.
pub fn cache_word(mode: CpuCacheMode) -> String {
    match mode {
        CpuCacheMode::DefaultCache => "default-cache".to_owned(),
        CpuCacheMode::WriteCombined => "write-combined".to_owned(),
        mode => format!("cache-{}", mode.raw()),
    }
}

/// The word a hazard tracking mode carries in a site's name.
pub fn hazard_word(mode: HazardTrackingMode) -> String {
    match mode {
        HazardTrackingMode::Default => "hazard-default".to_owned(),
        HazardTrackingMode::Untracked => "untracked".to_owned(),
        HazardTrackingMode::Tracked => "tracked".to_owned(),
        mode => format!("hazard-{}", mode.raw()),
    }
}

/// The word a heap type carries in a site's name.
pub fn heap_type_word(heap_type: HeapType) -> String {
    match heap_type {
        HeapType::Automatic => "automatic".to_owned(),
        HeapType::Placement => "placement".to_owned(),
        HeapType::Sparse => "sparse".to_owned(),
        heap_type => format!("heap-type-{}", heap_type.raw()),
    }
}

/// The word a purgeable state carries in a site's name.
pub fn purgeable_word(state: PurgeableState) -> String {
    match state {
        PurgeableState::KeepCurrent => "keep-current".to_owned(),
        PurgeableState::NonVolatile => "non-volatile".to_owned(),
        PurgeableState::Volatile => "volatile".to_owned(),
        PurgeableState::Empty => "empty".to_owned(),
        state => format!("purgeable-{}", state.raw()),
    }
}

/// A heap a case makes: every `MTLHeapDescriptor` field that decides what the heap's own getters
/// answer.
///
/// A **sparse** heap is [`crate::sparse_query::SparseHeapSpec`]'s and not this type's: its storage
/// is forced, its size rounds to a page rather than to 16 KiB, and the cases over it are about
/// mapping. This type is the two heaps that hold memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapSpec {
    /// `type`: `Automatic` or `Placement`. `Sparse` is refused by [`Self::is_well_formed`].
    pub heap_type: HeapType,
    /// `size`, in bytes, before Metal rounds it.
    pub size: u64,
    /// The storage mode, as the suite spells one.
    pub storage: Storage,
    /// `cpuCacheMode`.
    pub cache: CpuCacheMode,
    /// `hazardTrackingMode`, as the **descriptor** states it. What the heap answers may differ,
    /// which is one of the facts this family records.
    pub hazard: HazardTrackingMode,
}

impl HeapSpec {
    /// An automatic heap of `size` bytes in `storage`, the other two modes at Metal's defaults.
    pub const fn automatic(size: u64, storage: Storage) -> Self {
        Self {
            heap_type: HeapType::Automatic,
            size,
            storage,
            cache: CpuCacheMode::DefaultCache,
            hazard: HazardTrackingMode::Default,
        }
    }

    /// A placement heap of `size` bytes in `storage`, the other two modes at Metal's defaults.
    pub const fn placement(size: u64, storage: Storage) -> Self {
        Self {
            heap_type: HeapType::Placement,
            size,
            storage,
            cache: CpuCacheMode::DefaultCache,
            hazard: HazardTrackingMode::Default,
        }
    }

    /// The same heap whose descriptor states `cache`.
    #[must_use]
    pub const fn cached(mut self, cache: CpuCacheMode) -> Self {
        self.cache = cache;
        self
    }

    /// The same heap whose descriptor states `hazard`.
    #[must_use]
    pub const fn tracked(mut self, hazard: HazardTrackingMode) -> Self {
        self.hazard = hazard;
        self
    }

    /// Why this heap must never be handed to Metal, where it must not.
    ///
    /// One list, decided from the spec's own shape, and the only one. A case's guard reaches it
    /// with no device at all, which is the only proof such a check can have.
    ///
    /// - A **sparse** type: its descriptor's rules are another type's
    ///   ([`crate::sparse_query::SparseHeapSpec`]), and a sparse heap in any storage but `Private`
    ///   is one Metal stops.
    /// - **`Managed` storage**: Metal's validation stops a managed heap on this chip, and a managed
    ///   resource needs an explicit synchronization no case here encodes.
    /// - **`Memoryless` storage**: `MTLStorageModeMemoryless` describes a texture's bytes inside a
    ///   render pass, and nothing in this tree says Metal's heap allocator takes it. A memoryless
    ///   **texture** is a different question and is compared
    ///   ([`crate::cases::texture_descriptor_pairs`]).
    /// - A **size of 0**: nothing in this tree says Metal's heap allocator takes one.
    pub fn is_well_formed(&self) -> Option<String> {
        if matches!(self.heap_type, HeapType::Sparse) {
            return Some(
                "a sparse heap is crate::sparse_query::SparseHeapSpec's, not this type's".into(),
            );
        }
        if matches!(self.storage, Storage::Managed) {
            return Some("Metal's validation stops a managed heap".into());
        }
        if matches!(self.storage, Storage::Memoryless) {
            return Some(
                "nothing in this tree says Metal's heap allocator takes MTLStorageModeMemoryless"
                    .into(),
            );
        }
        if self.size == 0 {
            return Some(
                "nothing in this tree says Metal's heap allocator takes a size of 0".into(),
            );
        }
        None
    }

    /// The site a heap's answers sit under: every field, so no two heaps of a table share a site
    /// and a reader of a diff knows which heap moved without counting rows.
    pub fn site(&self) -> String {
        format!(
            "{}-{}-{}-{}-{}",
            heap_type_word(self.heap_type),
            self.size,
            self.storage.word(),
            cache_word(self.cache),
            hazard_word(self.hazard),
        )
    }

    /// The words the heap fingerprints as: the call it makes and every field it hands that call.
    pub fn words(&self) -> Vec<u8> {
        let mut out = b"newHeapWithDescriptor:".to_vec();
        out.extend(self.heap_type.raw().to_le_bytes());
        out.extend(self.size.to_le_bytes());
        out.extend(storage_word(self.storage));
        out.extend(self.cache.raw().to_le_bytes());
        out.extend(self.hazard.raw().to_le_bytes());
        out
    }
}

/// The byte a storage mode fingerprints as, the same spelling
/// [`crate::buffer_texture::BufferTextureRequest`] uses.
fn storage_word(storage: Storage) -> [u8; 1] {
    match storage {
        Storage::Shared => [0],
        Storage::Private => [1],
        Storage::Managed => [2],
        Storage::Memoryless => [3],
    }
}

/// What a heap answers about itself: the eight properties `MTLHeap` declares, and whether its
/// `device` is the device that made it.
///
/// `size` and `usedSize` are here as well as in [`crate::sparse_query::HeapFacts`], deliberately:
/// a heap that holds memory answers both differently from a sparse one, and reading them beside
/// `currentAllocatedSize` is what makes the three legible as a set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeapOwnFacts {
    /// `type`, by its raw `MTLHeapType`.
    pub heap_type: u64,
    /// `size`: the descriptor's size as Metal rounded it.
    pub size: u64,
    /// `usedSize`.
    pub used_size: u64,
    /// `currentAllocatedSize`.
    pub current_allocated_size: u64,
    /// `storageMode`, by its raw `MTLStorageMode`.
    pub storage_mode: u64,
    /// `cpuCacheMode`, by its raw `MTLCPUCacheMode`.
    pub cpu_cache_mode: u64,
    /// `hazardTrackingMode`, by its raw `MTLHazardTrackingMode`: what Metal **resolved** the
    /// descriptor's to.
    pub hazard_tracking_mode: u64,
    /// `resourceOptions`, by its raw `MTLResourceOptions`.
    pub resource_options: u64,
    /// `label`, as a label-bearing object answers one; `None` for nil.
    pub label: Option<String>,
    /// Whether `device`'s `registryID` is that of the device that made the heap.
    ///
    /// A Bool and not the object, because a recorded message may hold no address
    /// (`tests::no_recorded_message_holds_an_address`) and because the fact under test is the
    /// relation, not the pointer: what Metal settles is whether a heap answers its maker.
    ///
    /// `registryID` and not object identity, because object identity is not a question both APIs
    /// can be asked. Metal's device is one object per GPU, while the runtime's `Device` is a
    /// handle each object keeps a clone of, so a pointer comparison of the runtime's answers
    /// `false` for a heap that did answer its own device — which is exactly what the first run of
    /// this family recorded, and what makes a public identifier the only honest spelling.
    pub own_device: bool,
}

impl HeapOwnFacts {
    /// The facts as the sites a case emits under one node, so both sides emit one shape and a
    /// difference names the one getter that moved.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let number = |value: u64| Observed::value(Value::Uint(value));
        Node::branch(name, Vec::new())
            .with(Node::leaf("type", number(self.heap_type)))
            .with(Node::leaf("size", number(self.size)))
            .with(Node::leaf("used_size", number(self.used_size)))
            .with(Node::leaf(
                "current_allocated_size",
                number(self.current_allocated_size),
            ))
            .with(Node::leaf("storage_mode", number(self.storage_mode)))
            .with(Node::leaf("cpu_cache_mode", number(self.cpu_cache_mode)))
            .with(Node::leaf(
                "hazard_tracking_mode",
                number(self.hazard_tracking_mode),
            ))
            .with(Node::leaf(
                "resource_options",
                number(self.resource_options),
            ))
            .with(Node::leaf(
                "label",
                Observed::value(label_value(&self.label)),
            ))
            .with(Node::leaf(
                "own_device",
                Observed::value(Value::Bool(self.own_device)),
            ))
    }
}

/// A label as a site records it: the string as one line, or the absent field for nil.
///
/// `Value::Absent` rather than an empty string, because the whole question a label case asks is
/// whether Metal tells nil and `@""` apart, and a spelling that mapped both to `""` would answer it
/// by construction. The string itself goes in as a single line, which is the transcript's one shape
/// for text; it holds no address, because it is a label the case itself handed in.
pub fn label_value(label: &Option<String>) -> Value {
    match label {
        None => Value::Absent,
        Some(held) => Value::Lines(vec![held.clone()]),
    }
}

/// A buffer a case places in a heap: `newBufferWithLength:options:` of an automatic heap, or
/// `newBufferWithLength:options:offset:` of a placement one.
///
/// The options are always the heap's own, which both APIs require: the runtime refuses any other
/// `Invalid`, and that is this tree's reading of "Metal's validation stops it".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapBufferSpec {
    /// `length`, in bytes.
    pub length: u64,
    /// `offset`, for the placement form; `None` is the automatic form, which takes none.
    pub offset: Option<u64>,
}

impl HeapBufferSpec {
    /// A buffer of `length` bytes placed wherever an automatic heap's suballocator puts it.
    pub const fn suballocated(length: u64) -> Self {
        Self {
            length,
            offset: None,
        }
    }

    /// A buffer of `length` bytes at `offset` in a placement heap.
    pub const fn at(offset: u64, length: u64) -> Self {
        Self {
            length,
            offset: Some(offset),
        }
    }

    /// The call this buffer is made by, as the selector names it.
    pub const fn call(&self) -> &'static str {
        match self.offset {
            None => "newBufferWithLength:options:",
            Some(_) => "newBufferWithLength:options:offset:",
        }
    }

    /// The word the site's name carries.
    pub fn word(&self) -> String {
        match self.offset {
            None => format!("length-{}", self.length),
            Some(offset) => format!("length-{}-at-{offset}", self.length),
        }
    }

    /// The words the buffer fingerprints as: the call it makes and the numbers it hands it.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.call().as_bytes().to_vec();
        out.extend(self.length.to_le_bytes());
        out.extend(self.offset.unwrap_or(u64::MAX).to_le_bytes());
        out
    }
}

/// Which kind of object a case asks the `MTLResource` getters of.
///
/// One enumeration rather than two `Side` calls, because the getters are `MTLResource`'s and the
/// whole subject is that a buffer, a texture and a view answer the same protocol: a call per
/// receiver type would be two places for one answer to drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueriedResource {
    /// A buffer a case made, by its handle.
    Buffer(Handle),
    /// A texture, or a view of one, by its handle.
    Texture(Handle),
}

impl QueriedResource {
    /// The word the site's name carries, without the handle's number: a handle is a side's own
    /// counter, so a site keyed by one would name a different row on a side that made its objects
    /// in another order.
    pub const fn word(&self) -> &'static str {
        match self {
            QueriedResource::Buffer(_) => "buffer",
            QueriedResource::Texture(_) => "texture",
        }
    }
}

/// What a resource answers about itself: every `MTLResource` getter, read **before** any of this
/// family's setters has touched it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceFacts {
    /// `label`: nil for an object nothing has labelled, which is the fact a case reads this for.
    pub label: Option<String>,
    /// Whether `device`'s `registryID` is that of the device that made the resource
    /// ([`HeapOwnFacts::own_device`] says why it is a Bool and why it is the registry ID).
    pub own_device: bool,
    /// `storageMode`, by its raw `MTLStorageMode`.
    pub storage_mode: u64,
    /// `cpuCacheMode`, by its raw `MTLCPUCacheMode`.
    pub cpu_cache_mode: u64,
    /// `hazardTrackingMode`, by its raw `MTLHazardTrackingMode`.
    pub hazard_tracking_mode: u64,
    /// `resourceOptions`, by its raw `MTLResourceOptions`.
    pub resource_options: u64,
    /// Whether `heap` answered a heap at all.
    pub in_a_heap: bool,
    /// Whether the heap `heap` answered is the heap the case placed the resource in.
    ///
    /// `false` where the case placed it in none, so the two fields read together: "no heap" and
    /// "a heap that is not the one asked for" are different answers.
    pub expected_heap: bool,
    /// `heapOffset`.
    pub heap_offset: u64,
    /// `isAliasable`, before anything made it aliasable.
    pub aliasable: bool,
}

impl ResourceFacts {
    /// The facts as the sites a case emits under one node.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let number = |value: u64| Observed::value(Value::Uint(value));
        let flag = |held: bool| Observed::value(Value::Bool(held));
        Node::branch(name, Vec::new())
            .with(Node::leaf(
                "label",
                Observed::value(label_value(&self.label)),
            ))
            .with(Node::leaf("own_device", flag(self.own_device)))
            .with(Node::leaf("storage_mode", number(self.storage_mode)))
            .with(Node::leaf("cpu_cache_mode", number(self.cpu_cache_mode)))
            .with(Node::leaf(
                "hazard_tracking_mode",
                number(self.hazard_tracking_mode),
            ))
            .with(Node::leaf(
                "resource_options",
                number(self.resource_options),
            ))
            .with(Node::leaf("in_a_heap", flag(self.in_a_heap)))
            .with(Node::leaf("expected_heap", flag(self.expected_heap)))
            .with(Node::leaf("heap_offset", number(self.heap_offset)))
            .with(Node::leaf("aliasable", flag(self.aliasable)))
    }
}

/// The purgeable-state chain every case of this module asks, in order.
///
/// It is a chain and not a set because `setPurgeableState:` answers the state **before** the call:
/// `Volatile` tells what the object was when it was made, `Empty` tells that `Volatile` took, and
/// `KeepCurrent` — a query that changes nothing — tells that `Empty` took. Three answers that only
/// a run can give, and no two of them are the argument handed in.
///
/// `Empty` is third rather than last so that `KeepCurrent` can read it back; nothing is read
/// through the object after the chain but the chain's own answers.
pub const PURGEABLE_CHAIN: [PurgeableState; 3] = [
    PurgeableState::Volatile,
    PurgeableState::Empty,
    PurgeableState::KeepCurrent,
];

/// The three states `setPurgeableState:` can **put** an object in.
///
/// `MTLPurgeableState` has four values and this is all of them but `KeepCurrent`, which is not a
/// state at all: it is the query that reads one back without changing it, and it is how every
/// transition below is observed rather than a place a transition can go.
pub const PURGEABLE_STATES: [PurgeableState; 3] = [
    PurgeableState::NonVolatile,
    PurgeableState::Volatile,
    PurgeableState::Empty,
];

/// Every ordered transition between [`PURGEABLE_STATES`], including each state to itself: nine
/// pairs, each of which a case asks of an object of its own.
///
/// Both directions of each pair are in it by construction — `(a, b)` and `(b, a)` are separate
/// elements — which is the whole point: Metal's answer to `Volatile` after `Empty` need not be the
/// mirror of its answer to `Empty` after `Volatile`, and a sweep in one direction could not tell.
/// The same-state pairs are in it because "set what it already is" is the one place a side could
/// answer the argument it was handed instead of the state before the call.
pub fn purgeable_pairs() -> Vec<(PurgeableState, PurgeableState)> {
    let mut pairs = Vec::new();
    for from in PURGEABLE_STATES {
        for to in PURGEABLE_STATES {
            pairs.push((from, to));
        }
    }
    pairs
}

/// The labels every label case sets, in order, and what each is for.
///
/// - `""` first: whether Metal tells a label set to the empty string from a label never set is a
///   fact about Metal's own storage of the property, and the order puts it where the previous
///   answer was nil.
/// - a plain label: that a set takes at all.
/// - `""` again: whether a label can be cleared back to what an unlabelled object answers.
pub const LABEL_CHAIN: [&str; 3] = ["", "a label", ""];
