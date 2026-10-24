//! What an `MTLResource` answers about itself, asked of every kind of object that is one: a buffer
//! of the device's, a buffer placed in an automatic heap, a buffer placed in a placement heap at an
//! offset the case chose, a texture, and a view of a texture.
//!
//! `MTLResource` is the protocol every resource inherits, and before this family the suite reached
//! three of its twelve items — `storageMode`, `cpuCacheMode` and `hazardTrackingMode`, through
//! [`crate::cases::buffer_textures`] alone. The nine it never called are the ones that say where a
//! resource **lives** and what may be done to it: its `heap` and `heapOffset`, its aliasing, its
//! label, its `device`, the option word it packs its three modes into, and its purgeable state.
//!
//! # The cases, and what each one would catch
//!
//! - **`resource-own-getters`**: the ten answers of 22 resources — two lengths of a device
//!   buffer, a shared and a private texture, placement-heap buffers at offsets 0, 256 and 512 and
//!   one of length 0, a view, and the **whole** mode space of a heap-placed buffer: every storage
//!   mode against every CPU cache mode against every hazard tracking mode, twelve rows.
//!   The facts here are relations Metal settles and a descriptor does not: whether a buffer the
//!   device made answers **no** heap while one placed in a heap answers **that** heap, what
//!   `heapOffset` a resource outside a heap answers, what a **view's** heap and offset are, and
//!   what option word Metal packs a placed resource's modes into — which the runtime models as the
//!   heap's own rather than the call's.
//! - **`resource-aliasing`**: `isAliasable` before and after `makeAliasable`, with the heap's
//!   `usedSize` either side, over one resource of each kind — and then, in `TOGETHER`'s own
//!   sites, over **three buffers of one heap** with only the middle one made aliasable. The
//!   runtime's `usedSize` leaves a buffer made aliasable **out** of the sum, and the three-in-one
//!   shape is what makes that claim readable: a sum that lost the middle buffer's length and kept
//!   the other two is a different number from one that lost all three or none. The per-row half
//!   also records whether a placement-heap resource, which the heap never suballocated, answers
//!   aliasable at all.
//! - **`resource-purgeable-states`**: **every** ordered transition between the three states
//!   `setPurgeableState:` can put a resource in — nine pairs, both directions of each, each on a
//!   resource of its own — on a buffer of the device's, a heap-placed one, a placement-heap one
//!   and a texture. Each answer is the state **before** its call, read through the `KeepCurrent`
//!   query either side, so none of them is the argument handed in. Both directions are what the
//!   sweep is for: measured on a shared automatic heap, `Volatile` after `Empty` leaves the object
//!   **Empty** while `Empty` after `Volatile` takes, so the rule is not symmetric and a sweep in
//!   one direction could not have said so.
//! - **`resource-labels`**: [`crate::heap_query::LABEL_CHAIN`] on a buffer, a texture and a view of
//!   that texture, with the parent's label read **between** the view's sets. Two measured facts:
//!   whether Metal tells `nil` from `@""` — the runtime answers `Some("")`, which would be a
//!   divergence if Metal answers nil — and whether labelling a view labels its parent, which no
//!   descriptor states.
//!
//! # The positive control, and why this family needs one
//!
//! A sweep of getters is the shape that can pass having measured nothing, so each case is built so
//! that **the sides cannot agree by construction**:
//!
//! - every table holds both a resource in a heap and a resource outside one, so `in_a_heap` is
//!   recorded as true somewhere and false somewhere. This module's own tests assert that with no device;
//! - the placement rows state **three different offsets**, so `heapOffset` is three values and not
//!   one, and a side that answered 0 always would differ at two sites;
//! - the aliasing case reads the heap's `usedSize` both before and after `makeAliasable`, so a
//!   reading that never moved is a difference rather than a pass, and its three-in-one-heap sites
//!   leave two buffers **not** made aliasable, so a side that dropped every length from the sum
//!   differs from one that dropped the right one;
//! - the label chain's own control is the non-empty label: a side that dropped every `setLabel:`
//!   would answer `nil` where the other answers the label.
//!
//! # What each swept mode decides, and what it cannot
//!
//! A heap-placed resource never had a descriptor stating a CPU cache mode or a hazard tracking
//! mode: Metal gives it the **heap's**, which is why the getter sweep moves them through the heap
//! and why reading them back is Metal's composition rather than the case's own words. Measured
//! over the twelve mode rows on this chip:
//!
//! - **cache mode** gives `cpuCacheMode` 0 and 1, six rows each, and the low nibble of
//!   `resourceOptions` with it;
//! - **storage mode** gives `storageMode` 0 and 2, and the option word's storage nibble;
//! - **hazard tracking mode** gives `hazardTrackingMode` 1 for `Default` **and** for `Untracked`
//!   and 2 for `Tracked`: three asked values, two answers, and the two that coincide do so by
//!   Metal's own resolution. The contrast that makes it a measurement is the texture rows, where a
//!   resource **outside** a heap answers 2 for the same `Default` — so the resolution depends on
//!   where the resource lives and not on what was asked for;
//! - the twelve rows together give **eight** distinct `resourceOptions` words, which is the whole
//!   2 x 2 x 2 of storage, cache and resolved hazard.
//!
//! Two things this family cannot sweep, named rather than left implied: a **managed** resource,
//! because a managed heap is one Metal's validation stops and the suite's texture request states
//! private-or-shared rather than a storage mode; and a cache or hazard mode on a resource
//! **outside** a heap, because `Side::zeroed_buffer` and the texture request state neither.
//!
//! # Why every case here is safe
//!
//! Nothing is encoded, dispatched or committed and no case reaches a queue: resources are made,
//! getters are read, `makeAliasable` is called, and a resource the transition sweep made is asked
//! nothing but its own states. The textures are 64 by 64 `RGBA8Unorm`, the buffers at most 64 KiB,
//! and the heaps 1 MiB; the widest table is the transition sweep's four rows times nine pairs.
//!
//! The hazards are the heap descriptor's, named in [`crate::heap_query::HeapSpec`]'s own guard, and
//! the placement offsets': an offset that is not a multiple of 256, or a placement that runs past
//! the heap, is one Metal's validation stops. This module's own tests hold every row to both with no device.

use crate::allowance::Allowance;
use crate::case::{Abort, Case, Handle, Input, MetalSource, Side};
use crate::heap_query::{
    cache_word, hazard_word, purgeable_pairs, purgeable_word, HeapBufferSpec, HeapSpec,
    QueriedResource, CACHE_MODES, HAZARD_MODES, LABEL_CHAIN,
};
use crate::texture::TextureRequest;
use crate::texture_query::Storage;
use crate::texture_views::{ParentShape, ViewCall, ViewRequest};
use crate::transcript::{Node, Observed, Sites, Value};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::{CpuCacheMode, Degradation, HazardTrackingMode, PurgeableState};

/// The case names, as the report and the recordings spell them.
pub const OWN_GETTERS: &str = "resource-own-getters";
pub const ALIASING: &str = "resource-aliasing";
pub const PURGEABLE: &str = "resource-purgeable-states";
pub const LABELS: &str = "resource-labels";

/// The heap every case of the family places into: 1 MiB, as
/// [`crate::cases::heap_facts`]' heaps are.
const SIZE: u64 = 1 << 20;

/// Metal's placement grain for a heap-placed buffer: the offsets the placement rows state are
/// multiples of it, which is what keeps the rows out of Metal's validation.
const GRAIN: u64 = 256;

/// The texture every case that needs one makes: a 64 by 64 `RGBA8Unorm` of four levels, read by a
/// shader.
///
/// Four levels and not one, because a view of it has levels to start at, which is what makes the
/// view rows of this family and of [`crate::cases::texture_views`] one shape.
const LEVELS: u64 = 4;
const SIDE: u32 = 64;

/// That texture's shape, as a view request reads it.
const SHAPE: ParentShape = ParentShape::texture_2d(PixelFormat::RGBA8Unorm, LEVELS);

/// Where one resource of a table comes from, which is the whole of what a row varies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Row {
    /// `newBufferWithLength:options:` of the device: a buffer in no heap.
    ///
    /// No storage mode: the suite's vocabulary makes a device buffer in one storage mode only
    /// ([`Side::zeroed_buffer`]), and a second one would be a
    /// call of its own rather than a row of this table. `resourceOptions` still takes several
    /// values across the table, through the heaps and the textures, which is what the sweep needs.
    DeviceBuffer { length: u64 },
    /// `newTextureWithDescriptor:` of the device: a texture in no heap.
    DeviceTexture { storage: Storage },
    /// A view of a device texture: a resource with no memory of its own.
    View,
    /// `newBufferWithLength:options:` of an automatic heap of these modes.
    ///
    /// The CPU cache mode and the hazard tracking mode are the **heap's**, not the buffer's: Metal
    /// takes a heap-placed resource's from the heap, and that inheritance is the whole reason they
    /// are row fields. A buffer never had a descriptor stating either, so reading them back off it
    /// is Metal reporting what it composed rather than what the case wrote.
    HeapBuffer {
        length: u64,
        storage: Storage,
        cache: CpuCacheMode,
        hazard: HazardTrackingMode,
    },
    /// `newBufferWithLength:options:offset:` of a placement heap.
    PlacedBuffer {
        length: u64,
        offset: u64,
        storage: Storage,
    },
}

impl Row {
    /// The site this resource's answers sit under: every value the row states.
    pub fn site(&self) -> String {
        match self {
            Row::DeviceBuffer { length } => format!("device-buffer-{length}"),
            Row::DeviceTexture { storage } => format!("device-texture-{}", storage.word()),
            Row::View => "view-of-a-device-texture".to_owned(),
            Row::HeapBuffer {
                length,
                storage,
                cache,
                hazard,
            } => format!(
                "automatic-heap-buffer-{length}-{}-{}-{}",
                storage.word(),
                cache_word(*cache),
                hazard_word(*hazard)
            ),
            Row::PlacedBuffer {
                length,
                offset,
                storage,
            } => format!(
                "placement-heap-buffer-{length}-at-{offset}-{}",
                storage.word()
            ),
        }
    }

    /// The words the row fingerprints as.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.site().into_bytes();
        out.push(0);
        out
    }

    /// Whether the row's resource is placed in a heap, which is what the two heap getters are read
    /// against.
    pub const fn in_a_heap(&self) -> bool {
        matches!(self, Row::HeapBuffer { .. } | Row::PlacedBuffer { .. })
    }

    /// Why this row must never be handed to Metal, where it must not.
    ///
    /// One list, decided from the row's own shape. The shapes, each a failure of Metal's own
    /// validation and so a call that ends its process:
    ///
    /// - a placement offset that is not a multiple of `GRAIN`;
    /// - a placement whose bytes run past the heap's end;
    /// - a buffer of length 0 from an **automatic** heap, which that call refuses in both APIs
    ///   (plain Metal answers nil; the runtime answers a typed `Nil`), so it is a refusal row
    ///   rather than a reading of where a resource lives. The placement form does take 0, whose
    ///   Metal answer the runtime's own documentation names a probe for, so that row stays.
    pub fn is_well_formed(&self) -> Option<String> {
        match self {
            Row::PlacedBuffer { length, offset, .. } => {
                if !offset.is_multiple_of(GRAIN) {
                    return Some(format!(
                        "an offset of {offset} is not a multiple of {GRAIN}"
                    ));
                }
                if offset.checked_add(*length).is_none_or(|end| end > SIZE) {
                    return Some(format!(
                        "{length} bytes at {offset} run past the heap's {SIZE}"
                    ));
                }
                None
            }
            Row::HeapBuffer { length: 0, .. } => Some(
                "an automatic heap makes no buffer of 0 bytes, so the row would be a refusal \
                 rather than a reading of where a resource lives"
                    .into(),
            ),
            _ => None,
        }
    }
}

/// What one case of the family asks of each resource of its table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ask {
    /// Every `MTLResource` getter, read fresh.
    OwnGetters,
    /// `isAliasable` and `makeAliasable`, with the heap's `usedSize` either side.
    Aliasing,
    /// The purgeable chain.
    Purgeable,
    /// The label chain, and for a view its parent's label between the sets.
    Labels,
}

/// A case over one table of resources.
pub struct ResourceFactsCase {
    name: &'static str,
    ask: Ask,
    resources: Vec<Row>,
}

/// Every case of the family.
pub fn all() -> Vec<ResourceFactsCase> {
    vec![own_getters(), aliasing(), purgeable_states(), labels()]
}

/// The two storage modes a resource of this family takes: a managed one needs an explicit
/// synchronization no case here encodes, and a managed heap is one Metal's validation stops.
const BOTH_STORAGE: [Storage; 2] = [Storage::Shared, Storage::Private];

/// Every kind of resource `MTLResource` covers, and the **whole** mode space a resource can be in:
/// every storage mode against every CPU cache mode against every hazard tracking mode, with three
/// placement offsets and a placement of length 0.
///
/// The twelve mode rows go through heaps, because that is the only way this vocabulary can put a
/// resource in a cache mode or a hazard tracking mode at all: `MTLBuffer` and `MTLTexture` take
/// theirs from the heap they are placed in, and `Side::zeroed_buffer` states none. A row is
/// therefore `BOTH_STORAGE` x [`crate::heap_query::CACHE_MODES`] x
/// [`crate::heap_query::HAZARD_MODES`], and `hazardTrackingMode` is the field that makes the
/// sweep more than a packing check: Metal resolves `Default` on a heap-placed resource, and what
/// it resolves it to is not what any descriptor said.
fn own_getters() -> ResourceFactsCase {
    // Two lengths of a device buffer, which is the one row whose storage mode the suite's
    // vocabulary does not vary: the lengths are what keep two such rows apart.
    let mut resources = vec![
        Row::DeviceBuffer { length: 4096 },
        Row::DeviceBuffer { length: 1 },
    ];
    for storage in BOTH_STORAGE {
        resources.push(Row::DeviceTexture { storage });
        for cache in CACHE_MODES {
            for hazard in HAZARD_MODES {
                resources.push(Row::HeapBuffer {
                    length: 4096,
                    storage,
                    cache,
                    hazard,
                });
            }
        }
    }
    // Three offsets, so `heapOffset` is three values: a side that answered 0 always differs at two
    // sites rather than passing.
    for offset in [0, GRAIN, 2 * GRAIN] {
        resources.push(Row::PlacedBuffer {
            length: 4096,
            offset,
            storage: Storage::Shared,
        });
    }
    // A placement of 0 bytes: plain Metal makes a buffer of no bytes at the offset, which the
    // runtime's own documentation names a Metal probe for and nothing in this tree ever ran again.
    resources.push(Row::PlacedBuffer {
        length: 0,
        offset: 4 * GRAIN,
        storage: Storage::Shared,
    });
    resources.push(Row::View);
    ResourceFactsCase {
        name: OWN_GETTERS,
        ask: Ask::OwnGetters,
        resources,
    }
}

/// One resource of each kind that can be made aliasable, each in a heap of its own.
///
/// The claim that needs several resources in **one** heap is `TOGETHER`'s, emitted once per case
/// rather than per row: a row of this table makes its own heap, so its `usedSize` before and after
/// is "the only buffer's length, then nothing" and says only that the length left the sum.
fn aliasing() -> ResourceFactsCase {
    let mut resources = Vec::new();
    for length in [4096, 8192, 1 << 16] {
        resources.push(Row::HeapBuffer {
            length,
            storage: Storage::Shared,
            cache: CpuCacheMode::DefaultCache,
            hazard: HazardTrackingMode::Default,
        });
    }
    resources.push(Row::PlacedBuffer {
        length: 4096,
        offset: 0,
        storage: Storage::Shared,
    });
    resources.push(Row::DeviceBuffer { length: 4096 });
    resources.push(Row::DeviceTexture {
        storage: Storage::Private,
    });
    ResourceFactsCase {
        name: ALIASING,
        ask: Ask::Aliasing,
        resources,
    }
}

/// A buffer of the device's, a heap-placed one, a placement-heap one and a texture.
fn purgeable_states() -> ResourceFactsCase {
    ResourceFactsCase {
        name: PURGEABLE,
        ask: Ask::Purgeable,
        resources: vec![
            Row::DeviceBuffer { length: 4096 },
            Row::HeapBuffer {
                length: 4096,
                storage: Storage::Shared,
                cache: CpuCacheMode::DefaultCache,
                hazard: HazardTrackingMode::Default,
            },
            Row::PlacedBuffer {
                length: 4096,
                offset: 0,
                storage: Storage::Shared,
            },
            Row::DeviceTexture {
                storage: Storage::Private,
            },
        ],
    }
}

/// A buffer, a texture, a view of a texture, and a heap-placed buffer.
fn labels() -> ResourceFactsCase {
    ResourceFactsCase {
        name: LABELS,
        ask: Ask::Labels,
        resources: vec![
            Row::DeviceBuffer { length: 4096 },
            Row::DeviceTexture {
                storage: Storage::Shared,
            },
            Row::View,
            Row::HeapBuffer {
                length: 4096,
                storage: Storage::Shared,
                cache: CpuCacheMode::DefaultCache,
                hazard: HazardTrackingMode::Default,
            },
        ],
    }
}

/// The buffers the aliasing case places in **one** automatic heap, and which of them it makes
/// aliasable.
///
/// Three different lengths, and the **middle** one aliasable: the heap's `usedSize` afterwards is
/// then a number that tells three readings apart — the sum of all three, the sum without the
/// middle, and 0 — where one buffer alone could only tell "its length" from 0.
const TOGETHER: [(u64, bool); 3] = [(4096, false), (8192, true), (1 << 16, false)];

/// The sites a case emits.
mod site {
    /// The resource's ten getters, fresh.
    pub const FRESH: &str = "fresh";
    /// The heap the resource was placed in, before anything touched the resource.
    pub const HEAP_BEFORE: &str = "heap-before";
    /// That heap after `makeAliasable`.
    pub const HEAP_AFTER: &str = "heap-after";
    /// The resource's getters after `makeAliasable`.
    pub const ALIASED: &str = "aliased";
    /// One answer of the purgeable chain.
    pub const PURGEABLE: &str = "purgeable";
    /// The resource's getters after one label of the chain was set.
    pub const LABELLED: &str = "labelled";
    /// A view's parent's getters, read between the view's label sets.
    pub const PARENT: &str = "parent";
    /// Three buffers of one heap, with only the middle one made aliasable.
    pub const TOGETHER: &str = "three-in-one-heap";
}

/// The degrade notices one of these cases may announce. A list rather than a device query,
/// because a case never branches on the device: it runs and reads what came back, and a venue that
/// announces a kind not here fails the case until the kind is added.
///
/// Declaring a kind states what our side announced at the notices site and loosens no value:
/// every getter still answers exactly what Metal answers.
const PASS_NOTICES: &[Degradation] = &[
    // The resources these getters answer about include `Shared` buffers.
    // A device with no unified memory keeps a `Shared` allocation in system memory because that is
    // the only memory both sides can reach, so this notice belongs to the venue's memory model
    // rather than to anything these cases do. A device whose driver is Metal announces nothing
    // for it.
    Degradation::SharedInSystemMemory,
];

impl Case for ResourceFactsCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        PASS_NOTICES
            .iter()
            .map(|kind| crate::replay::notice_allowance(kind.code()))
            .collect()
    }

    fn shaders(&self) -> Result<Vec<crate::recording::ShaderRef>, String> {
        // Nothing here compiles: every answer is a getter of a resource.
        Ok(Vec::new())
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut words = Vec::new();
        for resource in &self.resources {
            words.extend(resource.words());
        }
        // The heaps and the texture the rows are made from are part of what the case feeds, so a
        // table whose heap size or texture shape moved is a change to the case's inputs.
        words.extend(automatic_heap(Storage::Shared).words());
        words.extend(placement_heap(Storage::Shared).words());
        words.extend(texture().words());
        words.extend(view_request().words());
        let asks: Vec<u8> = match self.ask {
            Ask::OwnGetters => b"getters".to_vec(),
            Ask::Aliasing => {
                let mut out = b"isAliasable,makeAliasable,usedSize".to_vec();
                for (length, aliasable) in TOGETHER {
                    out.extend(length.to_le_bytes());
                    out.push(u8::from(aliasable));
                }
                out
            }
            Ask::Purgeable => {
                let mut out = b"setPurgeableState:".to_vec();
                for (from, to) in purgeable_pairs() {
                    out.extend(from.raw().to_le_bytes());
                    out.extend(to.raw().to_le_bytes());
                }
                out
            }
            Ask::Labels => {
                let mut out = b"setLabel:".to_vec();
                for label in LABEL_CHAIN {
                    out.extend(label.as_bytes());
                    out.push(0);
                }
                out
            }
        };
        Ok(vec![
            Input::bytes("resources", &words),
            Input::bytes("asks", &asks),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for row in &self.resources {
            if matches!(self.ask, Ask::Purgeable) {
                sites.push(transitions(side, row));
                continue;
            }
            let mut node = Node::branch(row.site(), Vec::new());
            // Each row makes its own heap and its own texture, so one row that could not be made
            // leaves the others' answers readable and no row reads an object another row changed.
            let made = match make(side, row) {
                Ok(made) => made,
                Err(failure) => {
                    sites.push(
                        node.with(Node::leaf(site::FRESH, Observed::value(failure.recorded()))),
                    );
                    continue;
                }
            };
            node = node.with(facts(side, &made, site::FRESH));
            match self.ask {
                Ask::OwnGetters => {}
                Ask::Aliasing => node = aliasing_of(side, &made, node),
                // Taken below: the transition sweep makes a resource of its own for every pair,
                // because a resource already put in one state cannot be the `from` of another.
                Ask::Purgeable => {}
                Ask::Labels => node = labels_of(side, &made, node),
            }
            sites.push(node);
        }
        if matches!(self.ask, Ask::Aliasing) {
            sites.push(three_in_one_heap(side));
        }
        Ok(())
    }
}

/// `TOGETHER`'s buffers in one automatic heap: the heap's figures, then the middle buffer made
/// aliasable, then the heap's figures and every buffer's `isAliasable` again.
///
/// This is the shape the `usedSize` claim needs. Nothing is encoded or committed: three buffers are
/// placed, one call is made and the getters are read.
fn three_in_one_heap(side: &mut dyn Side) -> Node {
    let mut node = Node::branch(site::TOGETHER, Vec::new());
    let heap = match side.heap(&automatic_heap(Storage::Shared)) {
        Ok(heap) => heap,
        Err(failure) => return node.with(Node::leaf("heap", Observed::value(failure.recorded()))),
    };
    let mut placed = Vec::new();
    for (length, _) in TOGETHER {
        let at = format!("placed-{length}");
        match side.heap_buffer(heap, &HeapBufferSpec::suballocated(length)) {
            Ok(buffer) => placed.push((length, buffer)),
            Err(failure) => {
                node = node.with(Node::leaf(at, Observed::value(failure.recorded())));
            }
        }
    }
    let each = |side: &mut dyn Side, node: Node, round: &str, placed: &[(u64, Handle)]| {
        let mut node = node;
        for (length, buffer) in placed {
            let made = Made {
                resource: QueriedResource::Buffer(*buffer),
                heap: Some(heap),
                parent: None,
            };
            node = node.with(facts(side, &made, &format!("{round}/length-{length}")));
        }
        node
    };
    if let Some(before) = heap_facts(side, Some(heap), site::HEAP_BEFORE) {
        node = node.with(before);
    }
    node = each(side, node, site::FRESH, &placed);
    for (length, aliasable) in TOGETHER {
        if !aliasable {
            continue;
        }
        let Some((_, buffer)) = placed.iter().find(|(held, _)| *held == length) else {
            continue;
        };
        let answered = match side.make_resource_aliasable(QueriedResource::Buffer(*buffer)) {
            Ok(()) => Value::ok(Value::Absent),
            Err(failure) => failure.recorded(),
        };
        node = node.with(Node::leaf(
            format!("make-aliasable-{length}"),
            Observed::value(answered),
        ));
    }
    node = each(side, node, site::ALIASED, &placed);
    if let Some(after) = heap_facts(side, Some(heap), site::HEAP_AFTER) {
        node = node.with(after);
    }
    node
}

/// One row's objects, as a side made them.
struct Made {
    /// The resource the getters are asked of.
    resource: QueriedResource,
    /// The heap the row placed it in, where it placed it in one.
    heap: Option<Handle>,
    /// The texture a view was made of, where the row is a view.
    parent: Option<Handle>,
}

/// The automatic heap a row places into.
fn automatic_heap(storage: Storage) -> HeapSpec {
    HeapSpec::automatic(SIZE, storage)
}

/// The placement heap a row places into.
fn placement_heap(storage: Storage) -> HeapSpec {
    HeapSpec::placement(SIZE, storage)
}

/// The texture the texture and view rows are made of and from.
fn texture() -> TextureRequest<'static> {
    TextureRequest::read_2d(PixelFormat::RGBA8Unorm.as_raw(), SIDE, SIDE).levels(LEVELS as u32)
}

/// The view the view rows make: levels 1 and 2 of the texture, through the five-argument selector,
/// so a view's own `heapOffset` and label are read of a view that is not the whole texture.
fn view_request() -> ViewRequest {
    ViewRequest::whole(ViewCall::Swizzled, &SHAPE, PixelFormat::RGBA8Unorm).levels(1, 2)
}

/// Make one row's objects.
fn make(side: &mut dyn Side, row: &Row) -> Result<Made, crate::case::Failure> {
    match *row {
        Row::DeviceBuffer { length } => {
            let buffer = side.zeroed_buffer(length as usize)?;
            Ok(Made {
                resource: QueriedResource::Buffer(buffer),
                heap: None,
                parent: None,
            })
        }
        Row::DeviceTexture { storage } => {
            let mut request = texture();
            if matches!(storage, Storage::Private) {
                request = request.private();
            }
            let texture = side.texture(&request)?;
            Ok(Made {
                resource: QueriedResource::Texture(texture),
                heap: None,
                parent: None,
            })
        }
        Row::View => {
            let parent = side.texture(&texture())?;
            let view = side.texture_view(parent, &view_request())?;
            Ok(Made {
                resource: QueriedResource::Texture(view),
                heap: None,
                parent: Some(parent),
            })
        }
        Row::HeapBuffer {
            length,
            storage,
            cache,
            hazard,
        } => {
            let heap = side.heap(&automatic_heap(storage).cached(cache).tracked(hazard))?;
            let buffer = side.heap_buffer(heap, &HeapBufferSpec::suballocated(length))?;
            Ok(Made {
                resource: QueriedResource::Buffer(buffer),
                heap: Some(heap),
                parent: None,
            })
        }
        Row::PlacedBuffer {
            length,
            offset,
            storage,
        } => {
            let heap = side.heap(&placement_heap(storage))?;
            let buffer = side.heap_buffer(heap, &HeapBufferSpec::at(offset, length))?;
            Ok(Made {
                resource: QueriedResource::Buffer(buffer),
                heap: Some(heap),
                parent: None,
            })
        }
    }
}

/// The resource's ten getters at `at`.
fn facts(side: &mut dyn Side, made: &Made, at: &str) -> Node {
    match side.resource_facts(made.resource, made.heap) {
        Ok(facts) => facts.node(at.to_owned()),
        Err(failure) => Node::leaf(at.to_owned(), Observed::value(failure.recorded())),
    }
}

/// The heap's own figures at `at`, where there is a heap to read.
fn heap_facts(side: &mut dyn Side, heap: Option<Handle>, at: &str) -> Option<Node> {
    let heap = heap?;
    Some(match side.heap_own_facts(heap) {
        Ok(facts) => facts.node(at.to_owned()),
        Err(failure) => Node::leaf(at.to_owned(), Observed::value(failure.recorded())),
    })
}

/// `makeAliasable`, with the heap's figures and the resource's getters either side.
fn aliasing_of(side: &mut dyn Side, made: &Made, mut node: Node) -> Node {
    if let Some(before) = heap_facts(side, made.heap, site::HEAP_BEFORE) {
        node = node.with(before);
    }
    let answered = match side.make_resource_aliasable(made.resource) {
        Ok(()) => Value::ok(Value::Absent),
        Err(failure) => failure.recorded(),
    };
    node = node.with(Node::leaf("make-aliasable", Observed::value(answered)));
    node = node.with(facts(side, made, site::ALIASED));
    if let Some(after) = heap_facts(side, made.heap, site::HEAP_AFTER) {
        node = node.with(after);
    }
    node
}

/// Every ordered transition between the three purgeable states, each on a resource of `row` made
/// for that pair alone.
///
/// Four answers a pair, and only the last is a query of ours: the state Metal **made** the
/// resource in, the state before `from` was set, the state before `to` was set — which is what
/// `from` resolved to, and the transition's own answer — and the state `to` left behind.
/// [`crate::heap_query::purgeable_pairs`] says why both directions and the same-state pairs are
/// in it.
///
/// `Empty` is asked in both directions, which is why each pair gets its own resource: nothing is
/// read through a resource after it has been emptied but these answers.
fn transitions(side: &mut dyn Side, row: &Row) -> Node {
    let mut node = Node::branch(row.site(), Vec::new());
    for (from, to) in purgeable_pairs() {
        let at = format!(
            "{}/{}-then-{}",
            site::PURGEABLE,
            purgeable_word(from),
            purgeable_word(to)
        );
        let made = match make(side, row) {
            Ok(made) => made,
            Err(failure) => {
                node = node.with(Node::leaf(at, Observed::value(failure.recorded())));
                continue;
            }
        };
        let mut pair = Node::branch(at, Vec::new());
        for (name, state) in [
            ("fresh", PurgeableState::KeepCurrent),
            ("from", from),
            ("to", to),
            ("read-back", PurgeableState::KeepCurrent),
        ] {
            let answered = match side.resource_purgeable_state(made.resource, state) {
                Ok(was) => Value::ok(Value::Uint(was)),
                Err(failure) => failure.recorded(),
            };
            pair = pair.with(Node::leaf(name, Observed::value(answered)));
        }
        node = node.with(pair);
    }
    node
}

/// The label chain, and for a view its parent's label after each set.
fn labels_of(side: &mut dyn Side, made: &Made, mut node: Node) -> Node {
    for (round, label) in LABEL_CHAIN.iter().enumerate() {
        if let Err(failure) = side.set_resource_label(made.resource, label) {
            node = node.with(Node::leaf(
                format!("set-label-{round}"),
                Observed::value(failure.recorded()),
            ));
            continue;
        }
        let at = format!("{}/{round}-{}", site::LABELLED, label.len());
        node = node.with(facts(side, made, &at));
        // A view's parent, read after the view was labelled: whether labelling a view labels what
        // it is a view of is Metal's to settle, and no descriptor states it.
        if let Some(parent) = made.parent {
            let parent_at = format!("{}/{round}-{}", site::PARENT, label.len());
            let of_parent = Made {
                resource: QueriedResource::Texture(parent),
                heap: None,
                parent: None,
            };
            node = node.with(facts(side, &of_parent, &parent_at));
        }
    }
    node
}

/// The guards this family's safety argument and its claim to measure anything rest on, all with no
/// device at all.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap_query::PURGEABLE_STATES;
    use std::collections::BTreeSet;

    /// No row and no heap of the family is one Metal's validation stops.
    ///
    /// Verified to fail: making [`view_request`] ask for levels 3 and 4 of the four-level texture
    /// fails [`the_view_the_family_makes_is_one_metal_takes`] naming the range and the parent's
    /// level count. This half of the guard reads the same list for every row of every table.
    #[test]
    fn no_row_of_the_family_is_one_metals_validation_stops() {
        let mut stopped = Vec::new();
        for case in all() {
            for row in &case.resources {
                if let Some(why) = row.is_well_formed() {
                    stopped.push(format!("{}: {}: {why}", case.name, row.site()));
                }
            }
        }
        for heap in [
            automatic_heap(Storage::Shared),
            automatic_heap(Storage::Private),
            placement_heap(Storage::Shared),
            placement_heap(Storage::Private),
        ] {
            if let Some(why) = heap.is_well_formed() {
                stopped.push(format!("{}: {why}", heap.site()));
            }
        }
        assert!(
            stopped.is_empty(),
            "these rows would end Metal's process; take them out of the table:\n{}",
            stopped.join("\n")
        );
    }

    /// The view every row makes is one Metal's validation takes, against the shape of the texture
    /// the family makes it of.
    #[test]
    fn the_view_the_family_makes_is_one_metal_takes() {
        assert_eq!(
            view_request().is_well_formed(&SHAPE),
            None,
            "the view this family makes is one Metal's validation stops"
        );
    }

    /// Every table holds a resource in a heap and a resource outside one, so the two heap getters
    /// are recorded as both answers and not as one.
    ///
    /// This is the positive control: without it a table of heap-placed resources alone would
    /// record `in_a_heap` true everywhere, and a side that answered true unconditionally would
    /// pass.
    ///
    /// Verified to fail: dropping both rows of [`purgeable_states`] that are outside a heap — its
    /// `DeviceBuffer` and its `DeviceTexture` — fails here naming the case and the two counts.
    #[test]
    fn every_table_holds_a_resource_in_a_heap_and_one_outside_one() {
        for case in all() {
            let inside = case.resources.iter().filter(|row| row.in_a_heap()).count();
            let outside = case.resources.len() - inside;
            assert!(
                inside > 0 && outside > 0,
                "{}: {inside} resources in a heap and {outside} outside one, so `heap` and \
                 `heapOffset` are recorded as one answer and a side that always gave it would pass",
                case.name
            );
        }
    }

    /// The getter sweep states three different placement offsets, so `heapOffset` is three values.
    ///
    /// Verified to fail: narrowing [`own_getters`]' offset loop to `[0]` fails here, naming the
    /// one offset left.
    #[test]
    fn the_getter_sweep_states_more_than_one_placement_offset() {
        let offsets: BTreeSet<u64> = own_getters()
            .resources
            .iter()
            .filter_map(|row| match row {
                Row::PlacedBuffer { offset, .. } => Some(*offset),
                _ => None,
            })
            .collect();
        assert!(
            offsets.len() >= 3,
            "{} placement offsets, so a side answering a constant `heapOffset` would differ at no \
             site: {offsets:?}",
            offsets.len()
        );
    }

    /// The getter sweep states the **whole** mode space a heap-placed resource can be in: every
    /// storage mode against every CPU cache mode against every hazard tracking mode.
    ///
    /// This is the depth claim for `MTLResource`'s three mode getters. A table that stated one
    /// cache mode would compare `cpuCacheMode` and the low nibble of `resourceOptions` at one
    /// value, and a side that dropped the field entirely would pass.
    ///
    /// Verified to fail: taking only the first of `CACHE_MODES` in [`own_getters`] fails here,
    /// naming the six write-combined combinations it leaves out.
    #[test]
    fn the_getter_sweep_states_every_inherited_mode_combination() {
        let rows: BTreeSet<String> = own_getters().resources.iter().map(Row::site).collect();
        let mut missing = Vec::new();
        for storage in BOTH_STORAGE {
            for cache in CACHE_MODES {
                for hazard in HAZARD_MODES {
                    let row = Row::HeapBuffer {
                        length: 4096,
                        storage,
                        cache,
                        hazard,
                    };
                    if !rows.contains(&row.site()) {
                        missing.push(row.site());
                    }
                }
            }
        }
        assert!(
            missing.is_empty(),
            "no case puts a resource in these modes, so Metal's answer for them is compared \
             nowhere in the suite: {}",
            missing.join(", ")
        );
    }

    /// The transition sweep holds every ordered pair of the three settable purgeable states, in
    /// both directions, and reads each through the query that changes nothing.
    ///
    /// Verified to fail: keeping only the pairs where `from.raw() <= to.raw()` in
    /// [`crate::heap_query::purgeable_pairs`] fails here, naming the three it drops.
    #[test]
    fn the_transition_sweep_holds_every_ordered_pair() {
        let pairs: BTreeSet<(u64, u64)> = purgeable_pairs()
            .into_iter()
            .map(|(from, to)| (from.raw(), to.raw()))
            .collect();
        let mut missing = Vec::new();
        for from in PURGEABLE_STATES {
            for to in PURGEABLE_STATES {
                if !pairs.contains(&(from.raw(), to.raw())) {
                    missing.push(format!(
                        "{}-then-{}",
                        purgeable_word(from),
                        purgeable_word(to)
                    ));
                }
            }
        }
        assert!(
            missing.is_empty(),
            "the sweep never asks these transitions of a resource: {}",
            missing.join(", ")
        );
        assert!(
            purgeable_states()
                .resources
                .iter()
                .any(|row| matches!(row, Row::DeviceTexture { .. })),
            "no texture in the transition table, and a texture is the one resource whose states \
             this chip was measured to walk at all"
        );
    }

    /// `TOGETHER` places more than one buffer in one heap and leaves at least one of them
    /// **not** aliasable, which is what makes the `usedSize` claim readable: a sum that lost one
    /// buffer's length and kept the others is a different number from 0 and from the whole sum.
    ///
    /// Verified to fail: a `TOGETHER` whose three buffers are all made aliasable fails here,
    /// naming the count.
    #[test]
    fn the_aliasing_case_leaves_a_buffer_of_one_heap_not_aliasable() {
        let lengths: BTreeSet<u64> = TOGETHER.iter().map(|(length, _)| *length).collect();
        assert_eq!(
            lengths.len(),
            TOGETHER.len(),
            "two buffers of one heap share a length, so their sites share a name: {lengths:?}"
        );
        assert!(lengths.len() >= 2, "one buffer in the heap only");
        let aliased = TOGETHER.iter().filter(|(_, held)| *held).count();
        assert!(
            aliased > 0 && aliased < TOGETHER.len(),
            "{aliased} of {} buffers are made aliasable, so `usedSize` afterwards cannot be told \
             from 0 or from the whole sum",
            TOGETHER.len()
        );
    }

    /// No two rows of a case share a site, and each case fingerprints its own table.
    #[test]
    fn every_row_names_its_own_site_and_every_case_its_own_inputs() {
        for case in all() {
            let sites: BTreeSet<String> = case.resources.iter().map(Row::site).collect();
            assert_eq!(
                sites.len(),
                case.resources.len(),
                "{}: two rows share a site",
                case.name
            );
        }
        let fingerprints: BTreeSet<Vec<String>> = all()
            .iter()
            .map(|case| {
                case.inputs()
                    .expect("the inputs are the case's own words")
                    .into_iter()
                    .map(|input| input.digest.hex)
                    .collect()
            })
            .collect();
        assert_eq!(
            fingerprints.len(),
            all().len(),
            "two cases fingerprint alike"
        );
    }
}

/// The second half of this family's safety argument, run on a device: the **order of work** the
/// heap-size and sparse-heap families state, which is that a table is replayed on the runtime
/// first and only one that answers no `Error::Invalid` anywhere is handed to Metal.
///
/// [`crate::device::sites_the_runtime_calls_invalid`] is the shared walk; this is the family that
/// asks it. The device-free guards above check each row's shape against the rules; this checks the
/// runtime's own reading of every call the rows actually make, including the ones no rule above
/// names.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod device_tests {
    use super::*;
    use crate::case::Case;

    #[test]
    fn no_call_of_the_family_is_one_metals_validation_stops() {
        let cases: Vec<Box<dyn Case>> = all()
            .into_iter()
            .map(|case| Box::new(case) as Box<dyn Case>)
            .collect();
        let stopped =
            crate::device::sites_the_runtime_calls_invalid(&cases).expect("a device for the guard");
        assert!(
            stopped.is_empty(),
            "the runtime answers `Error::Invalid` at these sites, which is this tree's reading of \
             \"Metal's own validation stops this call\" -- so record mode would end Metal's \
             process. Take the rows behind them out of the tables:\n{}",
            stopped.join("\n")
        );
    }
}
