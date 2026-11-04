//! One stage's argument table: the buffers with their offsets, inline bytes, textures, and sampler
//! states with their LOD clamps an encoder binds by Metal index. A compute encoder has one, a
//! render encoder one per stage (vertex and fragment); every `set_*` call of either is one of these
//! methods, with the same refusals.
//!
//! Inline bytes (`setBytes:length:atIndex:` and its forms) are copied at the call into the command
//! buffer's upload arena (`upload.rs`), as Metal copies them, so the caller's memory may change at
//! once; the bind layer states them as `BoundBuffer::Bytes`, and the binder binds the copy where
//! the plan writes `WriteSource::InlineBytes` or a vertex buffer.
//!
//! What a bound texture's usage allows, and what a sampler's clamps sample, is checked at the
//! command that uses the slot, against the function's use of it; binding checks only the index
//! range, the object's device, a buffer offset against the buffer's length, and that a texture is
//! not memoryless (Metal's validation stops binding one; a Metal probe on an M4).
//!
//! Every object a table binds is held until its encoder's command buffer completes, as Metal (with
//! retained references) holds what an encoder was handed: a no-copy buffer bound, then released by
//! the program, ran its deallocator only after its command buffer completed, whether a command used
//! it, it was replaced before any did, or it was bound after the last (the M3 Max under macOS
//! 27.0.1, probes `dealloc-probe` and `dealloc-probe2`, compute and render encoders). A command
//! that uses the table retains what it binds (`ArgumentTable::retain`); an object replaced or
//! unbound that no such command retained, and one still bound as the encoder ends, is retained
//! then (`let_go`, `ArgumentTable::let_go_all`). Each object names the core that holds it from
//! such a retain (`retained::HeldBy`), so an object retained once is not retained again by that
//! core, whichever slot binds it: a bind before each draw, and a draw that binds objects an
//! earlier draw of the encoder bound, retain nothing of their own.
//!
//! A compute encoder's table differs in one answer, as plain Metal's compute encoder gave it on an
//! M4 where its API validation asserts: it holds a buffer bound past its length, which a dispatch
//! refuses only where the kernel reads the slot (a read there is past the buffer: memory decides
//! it). A render stage's table refuses that at the call, as Metal's validation does; plain Metal's
//! render form is not measured.
//!
//! An offset set at an index with no buffer bound is no change in either, as plain Metal's render
//! and compute encoders both take it ([`ArgumentTable::take_buffer_offset`]).

use super::address_map::SamplerMaps;
use super::binding::bound_sampler;
use super::buffer::{Buffer, ReadApart};
use super::device::lock;
use super::device::Device;
use super::encoder::EncoderCore;
use super::error::{invalid, Error, Problem};
use super::function_table::{FunctionTableProblem, VisibleFunctionTable};
use super::live_bind::Reshape;
use super::residency::whole_use;
use super::resource::ResourceKey;
use super::retained::HeldBy;
use super::sampler::SamplerState;
use super::texture::Texture;
use super::tracker::{Access, Use};
use super::upload::Upload;
use ash::vk;
use metal2vulkan::bind::{BindState, BoundBuffer, BoundSampler, BoundTexture, StageTable};
use metal2vulkan::pipeline::{
    BufferKind, PipelineInterface, SlotClass, SlotInterface, SlotRef, Stage,
};
use std::any::Any;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

/// Entries of a buffer table (`setBuffer:offset:atIndex:` and its stage forms).
pub(super) const BUFFER_SLOTS: u32 = 31;
/// Entries of a texture table (`setTexture:atIndex:` and its stage forms).
pub(super) const TEXTURE_SLOTS: u32 = 128;
/// Entries of a sampler table (`setSamplerState:atIndex:` and its stage forms).
pub(super) const SAMPLER_SLOTS: u32 = 16;
/// `MTLAttributeStrideStatic`: the stride argument that sets no per-bind stride, so the buffer
/// is read at its layout's own stride, as by the plain setters.
pub(super) const ATTRIBUTE_STRIDE_STATIC: u64 = u64::MAX;
/// The most bytes one `setBytes:length:atIndex:` (or a stage form) takes: Metal's documented
/// limit for inline bytes, 4 KiB.
pub(super) const INLINE_BYTES_LIMIT: u64 = 4096;

/// The stages whose buffer slots [`AtBindViews`] keeps, in its order.
const VIEW_STAGES: [Stage; 5] = [
    Stage::Vertex,
    Stage::Fragment,
    Stage::Kernel,
    Stage::Object,
    Stage::Mesh,
];

/// A pipeline state's buffer slots, per stage, whose bind offset class tells the device's uniform
/// view base alignment apart (`StageTable::at_bind_views`): at first those its functions read
/// through a uniform view (`PipelineInterface::uniform_view_buffers`), which a command bound at a
/// multiple of the alignment reads at its bind offset alone (`SpecKey::RangeAtBindOffset`). A slot
/// a command binds at an offset the alignment does not divide is left out from then on: a slot
/// bound at offsets that move takes one shape and one pipeline, its view read at its residual
/// (exact at any offset), not a shape and a pipeline for each in turn, whose draws would each
/// settle and switch pipelines on the encoding thread.
pub(crate) struct AtBindViews {
    /// Each stage's of [`VIEW_STAGES`], bit `i` for buffer index `i`.
    slots: [AtomicU64; VIEW_STAGES.len()],
    /// The device's view base alignment (`BindState::view_base_alignment_of`).
    alignment: u32,
}

impl AtBindViews {
    /// The slots `interface`'s functions read through a uniform view, under the view base
    /// alignment of `device`: on every device, so a command's shape and its plan are the ones the
    /// translator makes of its binds there. On a device that reads every uniform view at its
    /// residual (`pipeline::reads_views_at_bind`) only the key the plan gives takes its residual
    /// form (`RenderVariant::plan`, `Variant::keyed_regions`).
    pub(crate) fn new(device: &Device, interface: &PipelineInterface) -> AtBindViews {
        let limits = device.compile_device().profile().limits();
        AtBindViews {
            slots: VIEW_STAGES.map(|stage| AtomicU64::new(interface.uniform_view_buffers(stage))),
            alignment: BindState::view_base_alignment_of(limits),
        }
    }

    /// The device's view base alignment, which a bind state states
    /// (`BindState::view_base_alignment`).
    pub(crate) fn alignment(&self) -> u32 {
        self.alignment
    }

    /// Whether `stage`'s slots are `slots` (none for a stage it does not keep): a table read under
    /// them that binds the same offsets leaves the same out ([`ArgumentTable::at_bind_views`]),
    /// as it left out its own as it was read.
    pub(super) fn holds(&self, stage: Stage, slots: u64) -> bool {
        self.of(stage)
            .map_or(0, |kept| kept.load(Ordering::Relaxed))
            == slots
    }

    /// `stage`'s slots, none for a stage it does not keep.
    fn of(&self, stage: Stage) -> Option<&AtomicU64> {
        let at = VIEW_STAGES.iter().position(|s| *s == stage)?;
        Some(&self.slots[at])
    }
}

/// One stage's bound objects, by Metal index.
#[derive(Default)]
pub(super) struct ArgumentTable {
    /// A compute encoder's table ([`ArgumentTable::compute`]): it holds buffers bound past their
    /// length (the module docs).
    compute: bool,
    /// Each buffer with its bind offset; a visible-function table's words at offset 0.
    pub(super) buffers: BTreeMap<u32, (Buffer, u64)>,
    /// The visible-function tables among `buffers`, by the index they share with buffers: a
    /// later buffer at the index replaces the table, and a later table the buffer, as on Metal.
    pub(super) function_tables: BTreeMap<u32, VisibleFunctionTable>,
    /// The indices where a visible-function table was bound and an offset then set
    /// (`setBufferOffset:` and its forms): plain Metal reads the table there as the null table,
    /// of size 0 and every entry null, until a table or a buffer is bound at the index again.
    /// Neither `buffers` nor `function_tables` holds them.
    pub(super) null_tables: BTreeSet<u32>,
    /// The per-bind attribute strides of buffers in `buffers`, by index
    /// (`setBuffer:offset:attributeStride:atIndex:` and its forms): a layout of
    /// `MTLBufferLayoutStrideDynamic` reads its buffer at this stride. The plain setters bind
    /// with `MTLAttributeStrideStatic`, which sets none.
    pub(super) attribute_strides: BTreeMap<u32, u32>,
    pub(super) textures: BTreeMap<u32, Texture>,
    /// The indices of `textures` that hold a memoryless texture, a texture buffer's too: none,
    /// nearly always, so a command finds nothing to check ([`uses_memoryless`](Self::uses_memoryless)).
    memoryless: BTreeSet<u32>,
    /// Each sampler state with the LOD clamps its bind set
    /// (`setSamplerState:lodMinClamp:lodMaxClamp:atIndex:`), which replace the state's.
    pub(super) samplers: BTreeMap<u32, (SamplerState, Option<[f32; 2]>)>,
    /// The bytes of `setBytes:length:atIndex:` and its forms, by index: copied at the call
    /// ([`set_bytes`](Self::set_bytes)). Neither `buffers` nor `function_tables` holds the index.
    pub(super) bytes: BTreeMap<u32, InlineBytes>,
    /// What the table bound as an encoder's core last retained it ([`retain`](Self::retain)).
    retained: Retained,
    /// Its contents' stamp, drawn afresh by every call that may change them.
    stamp: Stamp,
    /// The slots its last calls changed, by the stamps they took ([`Changes`]).
    changes: Changes,
}

/// A copy binds what the table binds, under a stamp of its own: a copy changed in place (a
/// pass's copies of bytes its fragment stages wrote: `render/pass_start_reads.rs`) binds others.
impl Clone for ArgumentTable {
    fn clone(&self) -> ArgumentTable {
        ArgumentTable {
            compute: self.compute,
            buffers: self.buffers.clone(),
            function_tables: self.function_tables.clone(),
            null_tables: self.null_tables.clone(),
            attribute_strides: self.attribute_strides.clone(),
            textures: self.textures.clone(),
            memoryless: self.memoryless.clone(),
            samplers: self.samplers.clone(),
            bytes: self.bytes.clone(),
            retained: self.retained.unheld(),
            stamp: Stamp::fresh(),
            changes: Changes::default(),
        }
    }
}

/// What an argument table binds, named by a number drawn afresh for each new table (a copy
/// included) and at each call that may change what it binds, which no other table of the process
/// takes ([`ArgumentTable::stamp`]): while a table's stamp holds, it binds what it bound, by
/// identity. A table made for one command (an execution's own buffers) is filled in place before
/// any command reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Stamp(u64);

impl Stamp {
    /// A stamp no table took before.
    fn fresh() -> Stamp {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Stamp(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for Stamp {
    fn default() -> Stamp {
        Stamp::fresh()
    }
}

/// The slots of its last calls' changes a table keeps at least, once it noted as many, and the
/// most slots a call notes ([`Changes`]): it keeps at most twice as many, and a reader that read it
/// before those it keeps reads it whole.
const CHANGES_KEPT: usize = 16;

/// The slots a table's last calls changed (or may have: a refused call's too), each with the
/// stamp the table took at that call, oldest first, and the stamp before the oldest: a reader that
/// read the table under one of those stamps brings in the slots changed since alone
/// ([`ArgumentTable::refresh_changed`]); a reader under any other stamp reads it whole. A copy of
/// a table keeps none (its stamp is its own).
#[derive(Default)]
struct Changes {
    since: Option<Stamp>,
    slots: Vec<Changed>,
}

/// A slot a call changed, with the stamp the table took at the call: of `class` at `index`, and
/// whether the call kept the buffer bound there, moving its offset or attribute stride alone
/// (`offset`).
#[derive(Clone, Copy)]
struct Changed {
    stamp: Stamp,
    class: SlotClass,
    index: u32,
    offset: bool,
}

impl Changes {
    /// Note a call that took the table from `before` to `after`, changing the slots of `class`
    /// at `indices`, their offsets alone where `offset` ([`Changed`]): of more slots than are
    /// kept, it leaves none to read from.
    fn note(
        &mut self,
        (before, after): (Stamp, Stamp),
        class: SlotClass,
        indices: Range<u32>,
        offset: bool,
    ) {
        if indices.len() > CHANGES_KEPT {
            self.since = None;
            self.slots.clear();
            return;
        }
        if self.slots.is_empty() {
            self.since = Some(before);
        }
        self.slots.extend(indices.map(|index| Changed {
            stamp: after,
            class,
            index,
            offset,
        }));
        // Past twice the slots it keeps, the oldest calls go whole, at once, down to the newest
        // calls of at most `CHANGES_KEPT` slots: a reader under the stamp one of them took saw its
        // slots. At once, so a note moves the slots it keeps once in `CHANGES_KEPT` calls.
        if self.slots.len() > 2 * CHANGES_KEPT {
            let mut cut = self.slots.len() - CHANGES_KEPT;
            while cut < self.slots.len() && self.slots[cut].stamp == self.slots[cut - 1].stamp {
                cut += 1;
            }
            self.since = Some(self.slots[cut - 1].stamp);
            self.slots.drain(..cut);
        }
    }

    /// The slots changed since the table's stamp was `read`, where it keeps them.
    fn since(&self, read: Stamp) -> Option<impl Iterator<Item = Changed> + Clone + '_> {
        let from = match self.since == Some(read) {
            true => 0,
            false => (self.slots.iter()).rposition(|changed| changed.stamp == read)? + 1,
        };
        Some(self.slots[from..].iter().copied())
    }
}

/// The core an argument table was last retained into ([`EncoderCore::id`]) and the table's stamp
/// then ([`ArgumentTable::retain`]): while both hold, that core holds every object the table
/// binds.
#[derive(Default)]
struct Retained(Mutex<RetainedBy>);

impl Retained {
    /// A copy whose core is to look at every object at its next retain
    /// ([`RetainedBy::unheld`]): a table's copy, which may be changed in place without its calls
    /// (`render/pass_start_reads.rs`).
    fn unheld(&self) -> Retained {
        let mut copy = lock(&self.0).clone();
        copy.unheld = true;
        Retained(Mutex::new(copy))
    }
}

#[derive(Clone, Default)]
struct RetainedBy {
    core: Option<u64>,
    /// The table's stamp then: while it holds, the table binds those objects still.
    stamp: Option<Stamp>,
    /// Whether a call since bound an object that core did not hold then
    /// ([`ArgumentTable::note_bound`]): until one does, the core holds every object the table
    /// binds, so a retain looks at none ([`retain_into`]).
    unheld: bool,
    /// The objects the core holds of those the table bound since that retain, its own then
    /// among them, each by the address of its [`HeldBy`] (builds with debug assertions:
    /// [`check_held`](Self::check_held)). Its storage is kept from retain to retain, so a draw
    /// allocates as many times with debug assertions as without.
    #[cfg(debug_assertions)]
    held: Vec<usize>,
}

#[cfg(debug_assertions)]
impl RetainedBy {
    /// Note the objects `objects` binds now, as the table was just retained: the core holds them
    /// all.
    fn note_retained(&mut self, objects: Objects<'_>) {
        self.held.clear();
        let (buffers, textures, samplers) = objects;
        let bound = bound_in(buffers, textures, samplers);
        (self.held).extend(bound.map(|object| std::ptr::from_ref(object.held_by()).addr()));
    }

    /// That every object `objects` binds is one the core holds ([`retain_into`]'s premise where
    /// it looks at none): bound at its last retain, or bound since while the core held it.
    fn check_held(&self, objects: Objects<'_>) {
        let (buffers, textures, samplers) = objects;
        for object in bound_in(buffers, textures, samplers) {
            let address = std::ptr::from_ref(object.held_by()).addr();
            assert!(
                self.held.contains(&address),
                "an argument table binds an object its core neither retained at its last retain \
                 nor held as a call bound it, and no call since noted it unheld"
            );
        }
    }
}

/// What an argument table binds: its buffers, textures and sampler states, by index.
type Objects<'a> = (
    &'a BTreeMap<u32, (Buffer, u64)>,
    &'a BTreeMap<u32, Texture>,
    &'a BTreeMap<u32, (SamplerState, Option<[f32; 2]>)>,
);

/// An object an argument table binds, which a command that uses the table holds.
#[derive(Clone, Copy)]
enum Bound<'a> {
    Buffer(&'a Buffer),
    Texture(&'a Texture),
    Sampler(&'a SamplerState),
}

impl<'a> Bound<'a> {
    /// The core that holds it from a table's retain.
    fn held_by(self) -> &'a HeldBy {
        match self {
            Bound::Buffer(buffer) => buffer.held_by(),
            Bound::Texture(texture) => texture.held_by(),
            Bound::Sampler(sampler) => sampler.held_by(),
        }
    }

    /// The key its uses name, which a core that retains it keeps: none for a sampler state.
    fn key(self) -> Option<ResourceKey> {
        match self {
            Bound::Buffer(buffer) => Some(buffer.identity().key),
            Bound::Texture(texture) => Some(texture.use_key()),
            Bound::Sampler(_) => None,
        }
    }

    fn held(self) -> Arc<dyn Any + Send + Sync> {
        match self {
            Bound::Buffer(buffer) => buffer.held(),
            Bound::Texture(texture) => texture.held(),
            Bound::Sampler(sampler) => sampler.held(),
        }
    }
}

/// One `set*Bytes` call's copy of its bytes: a region of the command buffer's upload arena
/// (`upload.rs`) written at the call, so a later write of the caller's memory reaches no
/// command, as Metal copies the bytes at the call.
#[derive(Clone)]
pub(super) struct InlineBytes {
    /// The region: the bytes, then zeros to a multiple of 16 bytes, at least 16.
    pub(super) region: Upload,
    /// The call's length.
    pub(super) len: u32,
    /// The stride form's stride, its low 32 bits (`MTLAttributeStrideStatic` is `u32::MAX`);
    /// `None` for the plain setters (`BoundBuffer::Bytes`).
    pub(super) attribute_stride: Option<u32>,
}

/// Refuse `index` past a table of `limit` entries.
pub(super) fn check_index(call: &'static str, index: u32, limit: u32) -> Result<(), Error> {
    if index < limit {
        Ok(())
    } else {
        Err(invalid(call, Problem::IndexOutOfRange { index, limit }))
    }
}

/// Refuse `count` entries from `first` past a table of `limit` entries.
pub(super) fn check_range(
    call: &'static str,
    first: u32,
    count: usize,
    limit: u32,
) -> Result<(), Error> {
    if count == 0 {
        return Ok(());
    }
    let count = u32::try_from(count).unwrap_or(u32::MAX);
    check_index(call, first.saturating_add(count - 1), limit)
}

/// Refuse `buffer` at `offset` in `call` of an encoder of `device`: a buffer of another device,
/// and, in a render stage's table (not `compute`), an offset past its length
/// ([`ArgumentTable::compute`] holds it). The table's check, and a render encoder's at the call
/// (`render/table_facts.rs`).
pub(super) fn check_bound_buffer(
    device: &Device,
    call: &'static str,
    (buffer, offset): (&Buffer, u64),
    compute: bool,
) -> Result<(), Error> {
    super::encoder::check_device(device, call, buffer.device())?;
    let length = buffer.length();
    if offset > length && !compute {
        return Err(invalid(call, Problem::OffsetOutOfRange { offset, length }));
    }
    Ok(())
}

/// The per-bind stride `stride` names: none for `MTLAttributeStrideStatic`, else its low 32
/// bits. Plain Metal reads a stride past 32 bits so, and its validation takes it (an M4 under macOS
/// 27.0.1 and an M3 Max under 26.7, compute stage-in and vertex fetch, both binding forms: `2^32 +
/// 8` and `2^63 + 8` read at 8, `2^33 + 4` at 4, `2^32` at 0).
fn attribute_stride(stride: u64) -> Option<u32> {
    // Truncation is the measured answer.
    (stride != ATTRIBUTE_STRIDE_STATIC).then_some(stride as u32)
}

/// Bind `value` at `index` of `table`, or unbind the index when `value` is `None`: what was
/// bound there before.
fn set_entry<T>(table: &mut BTreeMap<u32, T>, index: u32, value: Option<T>) -> Option<T> {
    match value {
        Some(value) => table.insert(index, value),
        None => table.remove(&index),
    }
}

/// Bind the handle `value` at `index` of `table` with `with` (its offset, its clamps), or unbind
/// the index when `value` is `None`: the handle it replaced or unbound, another object than
/// `value` ([`let_go`]). The handle already bound there (the same object) keeps
/// its hold and takes `with` alone: binding it again takes no new reference, whose count the
/// threads that release the encoder's holds share.
fn set_handle<T: Clone + PartialEq, W>(
    table: &mut BTreeMap<u32, (T, W)>,
    index: u32,
    value: Option<&T>,
    with: W,
) -> Option<T> {
    match value {
        Some(value) => match table.entry(index) {
            Entry::Occupied(mut bound) if bound.get().0 == *value => {
                bound.get_mut().1 = with;
                None
            }
            Entry::Occupied(mut bound) => Some(bound.insert((value.clone(), with)).0),
            Entry::Vacant(slot) => {
                slot.insert((value.clone(), with));
                None
            }
        },
        None => table.remove(&index).map(|(handle, _)| handle),
    }
}

/// [`set_handle`] for a table of handles alone.
fn set_bare_handle<T: Clone + PartialEq>(
    table: &mut BTreeMap<u32, T>,
    index: u32,
    value: Option<&T>,
) -> Option<T> {
    match value {
        Some(value) => match table.entry(index) {
            Entry::Occupied(bound) if bound.get() == value => None,
            Entry::Occupied(mut bound) => Some(bound.insert(value.clone())),
            Entry::Vacant(slot) => {
                slot.insert(value.clone());
                None
            }
        },
        None => table.remove(&index),
    }
}

/// `buffer` bound at `offset`, read at `attribute_stride`, as the bind layer reads it
/// ([`ArgumentTable::stage_table`]), or `None` where it is bound past its length.
fn bound_buffer(
    buffer: &Buffer,
    offset: u64,
    attribute_stride: Option<u32>,
) -> Option<BoundBuffer> {
    if offset > buffer.length() {
        return None;
    }
    // The Metal address first, which keeps a no-copy buffer's import for good, then where the
    // buffer lies, in one look.
    let gpu_address = buffer.exposed_address(ReadApart::Binding).0;
    let (base, device_address) = buffer.base_and_device_address(ReadApart::Binding);
    Some(BoundBuffer::Buffer {
        length: base + buffer.length(),
        offset: base + offset,
        gpu_address: gpu_address - base,
        device_address: device_address.map(|address| address - base),
        attribute_stride,
    })
}

/// Inline bytes as the bind layer reads them.
fn bound_bytes(bytes: &InlineBytes) -> BoundBuffer {
    BoundBuffer::Bytes {
        len: bytes.len,
        attribute_stride: bytes.attribute_stride,
    }
}

/// `texture` as the bind layer reads it.
fn bound_texture(texture: &Texture) -> BoundTexture {
    match texture.texels() {
        Some(texels) => texels.bound(texture.pixel_format(), texture.record().sampled_swizzle()),
        // Its resource ID makes it a member of the handle tables that reach its slot: an argument
        // buffer can name it too.
        None => BoundTexture::Texture {
            facts: texture.facts(),
            resource_id: Some(texture.gpu_resource_id().0),
        },
    }
}

/// Whether `a` and `b` print alike (`{:?}`), told without printing them
/// ([`ArgumentTable::check_reading`]): every field as `==` compares it, but the descriptor's LOD
/// clamps, which print a float's sign (`0.0` and `-0.0` differ, where `==` takes them for equal)
/// and print every NaN alike (where `==` takes no NaN for equal). `tests/argument_table.rs` fails
/// when `BoundSampler` or `SamplerDescriptor` gains a field, so a new one is compared here too.
#[cfg(debug_assertions)]
pub(super) fn same_as_printed(a: &BoundSampler, b: &BoundSampler) -> bool {
    let alike = |x: f32, y: f32| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
    let clamps = alike(a.desc.lod_min_clamp, b.desc.lod_min_clamp)
        && alike(a.desc.lod_max_clamp, b.desc.lod_max_clamp);
    let [mut a, mut b] = [*a, *b];
    for desc in [&mut a.desc, &mut b.desc] {
        [desc.lod_min_clamp, desc.lod_max_clamp] = [0.0; 2];
    }
    clamps && a == b
}

impl ArgumentTable {
    /// What commands that run with this table's bindings, as an indirect command buffer's
    /// commands inherit the encoder's, may use of them, for the scheduler's order across queues
    /// (`CommandBufferShared::footprint`): each buffer over all its bytes and each texture over its
    /// whole image, read and written in every stage, as nothing bounds what their functions do
    /// there. `None` where it binds a visible-function table, whose functions reach more.
    pub(super) fn bound_uses(&self) -> Option<Vec<Use<ResourceKey>>> {
        if !self.function_tables.is_empty() {
            return None;
        }
        let stages = vk::PipelineStageFlags2::ALL_COMMANDS;
        let read = Access::new(stages, vk::AccessFlags2::MEMORY_READ);
        let write = Access::new(stages, vk::AccessFlags2::MEMORY_WRITE);
        let buffers = (self.buffers.values()).map(|(buffer, _)| whole_use(buffer, read, write));
        let textures = (self.textures.values()).map(|texture| texture.bound_use(read, write));
        Some(buffers.chain(textures).collect())
    }

    /// A compute encoder's table (the module docs).
    pub(super) fn compute() -> ArgumentTable {
        ArgumentTable {
            compute: true,
            ..ArgumentTable::default()
        }
    }

    /// The stamp of what the table binds now ([`Stamp`]).
    pub(super) fn stamp(&self) -> Stamp {
        self.stamp
    }

    /// A call that may change the slots of `class` at `indices` (before its checks: a refused one
    /// may have changed nothing, which a reader then reads again): a fresh stamp, and the slots
    /// noted ([`Changes`]).
    fn changed(&mut self, class: SlotClass, indices: Range<u32>) {
        self.changed_as(class, indices, false);
    }

    /// [`changed`](Self::changed), the call's change of the buffer slots at `indices` their
    /// offsets alone where `offset`: each binds the buffer it bound, and keeps binding it
    /// whatever the call's checks answer ([`Changed`]).
    fn changed_as(&mut self, class: SlotClass, indices: Range<u32>, offset: bool) {
        let before = self.stamp;
        self.stamp = Stamp::fresh();
        self.changes
            .note((before, self.stamp), class, indices, offset);
    }

    /// Whether buffer slot `index` binds `buffer` now: a buffer's offset moved there is the
    /// change of an offset alone ([`changed_as`](Self::changed_as)).
    fn binds_buffer(&self, index: u32, buffer: &Buffer) -> bool {
        (self.buffers.get(&index)).is_some_and(|(bound, _)| bound == buffer)
            && !self.function_tables.contains_key(&index)
    }

    /// A call through `core` binds `object`: where `core` does not hold it (from a retain of its
    /// own, `retained::HeldBy`), or is not the core the table was last retained into, the next
    /// retain looks at every object ([`retain_into`]); else `core` holds it until its command
    /// buffer completes, and keeps its key.
    fn note_bound(&mut self, core: &EncoderCore, object: Bound<'_>) {
        let id = core.id();
        let retained = self
            .retained
            .0
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        if retained.core != Some(id) || !object.held_by().is(id) {
            retained.unheld = true;
            return;
        }
        #[cfg(debug_assertions)]
        {
            let address = std::ptr::from_ref(object.held_by()).addr();
            if !retained.held.contains(&address) {
                retained.held.push(address);
            }
        }
    }

    /// Refuse `buffer` at `offset` in `call` of an encoder of `core`: a buffer of another device,
    /// and, in a render stage's table, an offset past its length (a compute table holds it).
    fn check_buffer(
        &self,
        core: &EncoderCore,
        call: &'static str,
        buffer: &Buffer,
        offset: u64,
    ) -> Result<(), Error> {
        check_bound_buffer(core.device(), call, (buffer, offset), self.compute)
    }

    /// `setBuffer:offset:atIndex:` (`None` unbinds the index) in an encoder of `core`: the
    /// stride form with `MTLAttributeStrideStatic`. Refused: an index past the table, a buffer of
    /// another device, and in a render stage's table an offset past the buffer's length.
    pub(super) fn set_buffer(
        &mut self,
        core: &mut EncoderCore,
        call: &'static str,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.set_buffer_with_stride(core, call, buffer, offset, ATTRIBUTE_STRIDE_STATIC, index)
    }

    /// `setBuffer:offset:attributeStride:atIndex:` (`None` unbinds the index) in an encoder of
    /// `core`, a stride past 32 bits taken modulo `2^32`. Refused as
    /// [`set_buffer`](Self::set_buffer) is.
    pub(super) fn set_buffer_with_stride(
        &mut self,
        core: &mut EncoderCore,
        call: &'static str,
        buffer: Option<&Buffer>,
        offset: u64,
        stride: u64,
        index: u32,
    ) -> Result<(), Error> {
        let same = buffer.is_some_and(|buffer| self.binds_buffer(index, buffer));
        self.changed_as(SlotClass::Buffer, index..index.saturating_add(1), same);
        check_index(call, index, BUFFER_SLOTS)?;
        let stride = attribute_stride(stride);
        if let Some(buffer) = buffer {
            self.check_buffer(core, call, buffer, offset)?;
            self.note_bound(core, Bound::Buffer(buffer));
        }
        let replaced = set_handle(&mut self.buffers, index, buffer, offset);
        set_entry(&mut self.attribute_strides, index, stride);
        self.function_tables.remove(&index);
        self.null_tables.remove(&index);
        self.bytes.remove(&index);
        if let Some(replaced) = replaced {
            let_go(core, Bound::Buffer(&replaced));
        }
        Ok(())
    }

    /// `setBuffers:offsets:withRange:`: `buffers[i]` at offset `offsets[i]` at index `first + i`,
    /// every stride `MTLAttributeStrideStatic`. Refused as [`set_buffer`](Self::set_buffer) is,
    /// and when the slices' lengths differ; a refusal binds none of them.
    pub(super) fn set_buffers(
        &mut self,
        core: &mut EncoderCore,
        call: &'static str,
        buffers: &[Option<&Buffer>],
        offsets: &[u64],
        first: u32,
    ) -> Result<(), Error> {
        self.set_buffers_with_strides(core, call, buffers, offsets, None, first)
    }

    /// `setBuffers:offsets:attributeStrides:withRange:`: as [`set_buffers`](Self::set_buffers),
    /// each buffer at stride `strides[i]` (modulo `2^32`) where `strides` is given. Refused as
    /// `set_buffers` is.
    pub(super) fn set_buffers_with_strides(
        &mut self,
        core: &mut EncoderCore,
        call: &'static str,
        buffers: &[Option<&Buffer>],
        offsets: &[u64],
        strides: Option<&[u64]>,
        first: u32,
    ) -> Result<(), Error> {
        self.changed(SlotClass::Buffer, span(first, buffers.len()));
        if buffers.len() != offsets.len() || strides.is_some_and(|s| s.len() != buffers.len()) {
            return Err(invalid(call, Problem::LengthMismatch));
        }
        check_range(call, first, buffers.len(), BUFFER_SLOTS)?;
        for (buffer, &offset) in buffers.iter().zip(offsets) {
            if let Some(buffer) = buffer {
                self.check_buffer(core, call, buffer, offset)?;
            }
        }
        let strides = (0..buffers.len()).map(|i| strides.and_then(|s| attribute_stride(s[i])));
        for (((buffer, &offset), stride), index) in
            buffers.iter().zip(offsets).zip(strides).zip(first..)
        {
            if let Some(buffer) = buffer {
                self.note_bound(core, Bound::Buffer(buffer));
            }
            let replaced = set_handle(&mut self.buffers, index, *buffer, offset);
            set_entry(&mut self.attribute_strides, index, stride);
            self.function_tables.remove(&index);
            self.null_tables.remove(&index);
            self.bytes.remove(&index);
            if let Some(replaced) = replaced {
                let_go(core, Bound::Buffer(&replaced));
            }
        }
        Ok(())
    }

    /// `setBufferOffset:atIndex:`: the stride form with `MTLAttributeStrideStatic`, so it clears
    /// the index's stride, as the plain buffer setters do. Measured for the vertex form on an M4:
    /// after a stride form, Metal's validation stops the draw for "providing
    /// `MTLAttributeStrideStatic` to `attributeStride` when stride at buffer-layout index 0 is
    /// dynamic", and Metal without it reads the buffer at stride 0. Every stage's form sets this
    /// one table. At an index holding a visible-function table, plain Metal then reads the null
    /// table there, of size 0 and every entry null, at any offset, 0 included, until a table or
    /// buffer is bound again (an M4 under macOS 27.0.1: the compute form at offsets 0 to 4096, the
    /// vertex form at 4 to 24; a second offset keeps it null; Metal's debug layer raises an
    /// exception there), and so does the runtime. At an index holding inline bytes, they read as
    /// zeros at any offset until the index is bound again, as plain Metal reads them (an M4).
    /// At an index with no buffer bound it keeps no offset, in a render stage's table as in a
    /// compute one ([`take_buffer_offset`](Self::take_buffer_offset)). Refused: an index past the
    /// table, and in a render stage's table an offset past a bound buffer's length.
    pub(super) fn set_buffer_offset(
        &mut self,
        core: &mut EncoderCore,
        call: &'static str,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.set_buffer_offset_with_stride(core, call, offset, ATTRIBUTE_STRIDE_STATIC, index)
    }

    /// `setBufferOffset:attributeStride:atIndex:` with `stride` (`MTLAttributeStrideStatic`
    /// clears the index's stride; a stride past 32 bits is taken modulo `2^32`). Refused as
    /// [`set_buffer_offset`](Self::set_buffer_offset) is.
    pub(super) fn set_buffer_offset_with_stride(
        &mut self,
        core: &mut EncoderCore,
        call: &'static str,
        offset: u64,
        stride: u64,
        index: u32,
    ) -> Result<(), Error> {
        // A buffer bound there keeps being bound, at another offset; inline bytes and a
        // visible-function table are another change.
        let same = !self.bytes.contains_key(&index)
            && !self.function_tables.contains_key(&index)
            && self.buffers.contains_key(&index);
        self.changed_as(SlotClass::Buffer, index..index.saturating_add(1), same);
        check_index(call, index, BUFFER_SLOTS)?;
        let stride = attribute_stride(stride);
        if let Some(len) = self.bytes.get(&index).map(|bytes| bytes.len) {
            // Inline bytes then read as zeros, at any offset, until the index is bound again:
            // plain Metal on an M4 read 64 zero bytes after `setBytes` of 64 then an offset of
            // 0, 16 or 128, in the compute, vertex and fragment forms (twice each; its debug layer
            // asserts "index(0) must have an existing buffer").
            let zeros = [0u8; INLINE_BYTES_LIMIT as usize];
            let padded = u64::from(len).next_multiple_of(16).max(16);
            let region = core.command_buffer().state().uploads.upload(
                core.device(),
                &zeros[..len as usize],
                padded,
            )?;
            self.bytes.insert(
                index,
                InlineBytes {
                    region,
                    len,
                    attribute_stride: stride,
                },
            );
            return Ok(());
        }
        if self.function_tables.remove(&index).is_some() {
            // The table reads as the null table until the index is bound again.
            let words = self.buffers.remove(&index);
            self.attribute_strides.remove(&index);
            self.null_tables.insert(index);
            if let Some((words, _)) = words {
                let_go(core, Bound::Buffer(&words));
            }
            return Ok(());
        }
        self.take_buffer_offset(call, offset, stride, index)
    }

    /// `setBufferOffset:attributeStride:atIndex:` at an index that binds a buffer, or binds
    /// nothing at all, with `stride`
    /// ([`set_buffer_offset_with_stride`](Self::set_buffer_offset_with_stride)'s inline bytes and
    /// visible-function tables are its own). It needs no encoder, since nothing here is retained
    /// or let go, so one test reads both tables' answer.
    ///
    /// Nothing bound keeps no offset, in a render stage's table as in a compute one: the index
    /// stays unbound, a later `setBuffer:offset:atIndex:` states its own offset, and a draw or a
    /// dispatch before that reads zeros, byte for byte as it reads an index never bound.
    ///
    /// An Apple M3 Max under macOS 27.0.1, three runs with Metal's API validation off in a fresh
    /// process and one with it on: `setVertexBufferOffset:64 atIndex:0` with nothing bound at 0
    /// returned, the draw ran its 3 vertices and 4096 fragments, the command buffer completed with
    /// no error, and the vertex function read `0x00000000` from `src[0]` — the same word a draw
    /// with no setter at all read, where the setter over a bound buffer read that buffer's word at
    /// 64, `0x22222222`. Metal's API validation alone objects, and
    /// `MTL_DEBUG_LAYER_ERROR_MODE=assert` ends the program there:
    /// `index(0) must have an existing buffer.` for the setter, then
    /// `Vertex Function(vread): missing Buffer binding at index 0 for src[0].` for the draw, which
    /// the never-bound run raises on its own.
    ///
    /// `metal2vulkan-conformance`'s `probes/buffer_offset_at_an_unbound_index.m` reads the same
    /// answer for an offset of 0, at an index that never held a buffer, on the same device and
    /// release: three runs a shape, one shape a process, API validation out of the environment.
    /// `setVertexBufferOffset:0 atIndex:3` with nothing bound at 3 returned, `endEncoding` and
    /// `commit` returned, the command buffer completed with no error, and the stages counted 3
    /// vertices and 256 fragments, having read `0x00000000` from `src[0]` -- the same word the
    /// never-bound control read, where a buffer bound at 3 read `0x11111111` at offset 0 and
    /// `0x22222222` after `setVertexBufferOffset:64`, so the setter demonstrably moves a bound
    /// buffer's offset. Under `MTL_DEBUG_LAYER=1` the layer alone objects, the command buffer
    /// still completing: `Set Vertex Buffer Offset Validation` /
    /// `index(3) must have an existing buffer.`, then the draw message the never-bound control
    /// raises on its own; `MTL_DEBUG_LAYER_ERROR_MODE=assert` ends the program there.
    ///
    /// Refused: an offset past a bound buffer's length in a render stage's table, which a compute
    /// one holds (the module docs).
    pub(super) fn take_buffer_offset(
        &mut self,
        call: &'static str,
        offset: u64,
        stride: Option<u32>,
        index: u32,
    ) -> Result<(), Error> {
        let compute = self.compute;
        let Some(bound) = self.buffers.get_mut(&index) else {
            return Ok(());
        };
        let length = bound.0.length();
        if offset > length && !compute {
            return Err(invalid(call, Problem::OffsetOutOfRange { offset, length }));
        }
        bound.1 = offset;
        set_entry(&mut self.attribute_strides, index, stride);
        Ok(())
    }

    /// `setBytes:length:atIndex:` (`stride` `None`) and `setBytes:length:attributeStride:atIndex:`
    /// (and every stage's forms) in an encoder of `core`: `bytes` copied now into the command
    /// buffer's upload arena, then zeros to a multiple of 16 bytes, in place of anything bound at
    /// `index`. A layout of `MTLBufferLayoutStrideDynamic` reads them at the stride form's
    /// stride, its low 32 bits, and at 0 after a plain setter (`BoundBuffer::Bytes`). Refused: an
    /// index past the table, and more than [`INLINE_BYTES_LIMIT`] bytes.
    pub(super) fn set_bytes(
        &mut self,
        core: &mut EncoderCore,
        call: &'static str,
        bytes: &[u8],
        stride: Option<u64>,
        index: u32,
    ) -> Result<(), Error> {
        self.changed(SlotClass::Buffer, index..index.saturating_add(1));
        check_index(call, index, BUFFER_SLOTS)?;
        let length = bytes.len() as u64;
        if length > INLINE_BYTES_LIMIT {
            return Err(invalid(
                call,
                Problem::InlineBytesLength {
                    length,
                    limit: INLINE_BYTES_LIMIT,
                },
            ));
        }
        // The bind layer binds at most the bytes padded to 16, at least 16.
        let padded = length.next_multiple_of(16).max(16);
        let region = core
            .command_buffer()
            .state()
            .uploads
            .upload(core.device(), bytes, padded)?;
        let replaced = self.buffers.remove(&index);
        self.attribute_strides.remove(&index);
        self.function_tables.remove(&index);
        self.null_tables.remove(&index);
        if let Some((replaced, _)) = replaced {
            let_go(core, Bound::Buffer(&replaced));
        }
        self.bytes.insert(
            index,
            InlineBytes {
                region,
                // At most `INLINE_BYTES_LIMIT`.
                len: length as u32,
                // Truncation is the measured answer for a buffer's (`attribute_stride`).
                attribute_stride: stride.map(|stride| stride as u32),
            },
        );
        Ok(())
    }

    /// `setTextures:withRange:` (and `setTexture:atIndex:` as a range of one): `textures[i]` at
    /// index `first + i`. A memoryless texture binds as plain Metal binds it (its API validation
    /// stops the call: "texture is Memoryless, and cannot be assigned"): a command whose function
    /// uses the slot fails its command buffer ([`uses_memoryless`](Self::uses_memoryless)).
    /// Refused: an index past the table, a texture of another device; a refusal binds none of
    /// them.
    pub(super) fn set_textures(
        &mut self,
        core: &mut EncoderCore,
        call: &'static str,
        textures: &[Option<&Texture>],
        first: u32,
    ) -> Result<(), Error> {
        self.changed(SlotClass::Texture, span(first, textures.len()));
        check_range(call, first, textures.len(), TEXTURE_SLOTS)?;
        for texture in textures.iter().flatten() {
            core.check_device(call, texture.device())?;
        }
        for (texture, index) in textures.iter().zip(first..) {
            match texture {
                Some(texture) if texture.is_memoryless() => self.memoryless.insert(index),
                _ => self.memoryless.remove(&index),
            };
            if let Some(texture) = texture {
                self.note_bound(core, Bound::Texture(texture));
            }
            if let Some(replaced) = set_bare_handle(&mut self.textures, index, *texture) {
                let_go(core, Bound::Texture(&replaced));
            }
        }
        Ok(())
    }

    /// Whether it binds nothing: no buffer, function table, null table, texture, sampler or bytes.
    pub(super) fn binds_nothing(&self) -> bool {
        self.buffers.is_empty()
            && self.function_tables.is_empty()
            && self.null_tables.is_empty()
            && self.textures.is_empty()
            && self.samplers.is_empty()
            && self.bytes.is_empty()
    }

    /// Whether a texture slot of `stage` that `interface`'s function statically uses holds a
    /// memoryless texture, a texture buffer's too. Plain Metal fails the command buffer of a
    /// command that runs such a function (`CommandBufferError::InvalidResource`), whether or not
    /// its run reads the slot, and of none whose function leaves the slot unused (an M3 Max under
    /// macOS 27.0.1).
    pub(super) fn uses_memoryless(&self, interface: &PipelineInterface, stage: Stage) -> bool {
        self.memoryless.iter().any(|&index| {
            let slot = SlotRef {
                stage,
                class: SlotClass::Texture,
                index,
            };
            interface.slots.contains_key(&slot)
        })
    }

    /// Bind `samplers` from index `first`, with `clamps` (minimums and maximums) when given.
    /// Refused: slices of different lengths, an index past the table, a state of another device;
    /// a refusal binds none of them. Metal takes any clamps; a sampler whose clamps are NaN or
    /// inverted is refused at the command that samples through it.
    pub(super) fn set_samplers(
        &mut self,
        core: &mut EncoderCore,
        call: &'static str,
        samplers: &[Option<&SamplerState>],
        clamps: Option<(&[f32], &[f32])>,
        first: u32,
    ) -> Result<(), Error> {
        self.changed(SlotClass::Sampler, span(first, samplers.len()));
        if let Some((min, max)) = clamps {
            if min.len() != samplers.len() || max.len() != samplers.len() {
                return Err(invalid(call, Problem::LengthMismatch));
            }
        }
        check_range(call, first, samplers.len(), SAMPLER_SLOTS)?;
        for sampler in samplers.iter().flatten() {
            core.check_device(call, sampler.device())?;
        }
        for (i, (sampler, index)) in samplers.iter().zip(first..).enumerate() {
            let clamp = clamps.map(|(min, max)| [min[i], max[i]]);
            if let Some(sampler) = sampler {
                self.note_bound(core, Bound::Sampler(sampler));
            }
            if let Some(replaced) = set_handle(&mut self.samplers, index, *sampler, clamp) {
                let_go(core, Bound::Sampler(&replaced));
            }
        }
        Ok(())
    }

    /// `setVisibleFunctionTables:withBufferRange:` (and `setVisibleFunctionTable:atBufferIndex:`
    /// as a range of one): `tables[i]` at buffer index `first + i`, in place of any buffer there;
    /// `None` unbinds the index. The table's words are the buffer the slot reads. Refused: an
    /// index past the buffer table, a table of another device; a refusal binds none of them.
    pub(super) fn set_function_tables(
        &mut self,
        core: &mut EncoderCore,
        call: &'static str,
        tables: &[Option<&VisibleFunctionTable>],
        first: u32,
    ) -> Result<(), Error> {
        self.changed(SlotClass::Buffer, span(first, tables.len()));
        check_range(call, first, tables.len(), BUFFER_SLOTS)?;
        for table in tables.iter().flatten() {
            core.check_device(call, table.device())?;
        }
        for (table, index) in tables.iter().zip(first..) {
            if let Some(table) = table {
                self.note_bound(core, Bound::Buffer(table.words()));
            }
            let replaced = set_entry(
                &mut self.buffers,
                index,
                table.map(|table| (table.words().clone(), 0)),
            );
            set_entry(&mut self.function_tables, index, table.cloned());
            self.attribute_strides.remove(&index);
            self.null_tables.remove(&index);
            self.bytes.remove(&index);
            // A table's words bound again at its index are the same object, held by the table.
            if let Some((replaced, _)) = replaced.filter(|(replaced, _)| {
                !self
                    .buffers
                    .get(&index)
                    .is_some_and(|(bound, _)| bound == replaced)
            }) {
                let_go(core, Bound::Buffer(&replaced));
            }
        }
        Ok(())
    }

    /// Refuse a command of `call` whose `stage` function writes a buffer, or reads an
    /// intersection function table, where a visible-function table is bound, by `interface`'s
    /// slots: plain Metal's answer was not measured (ray tracing's tables are out of scope). A
    /// table bound where the function only reads a buffer reads as an unbound buffer, and a buffer
    /// bound where it reads a visible-function table as a null table (`stage_table_reading`). A
    /// table an offset made null (`null_tables`) where the function reads a buffer is refused too:
    /// not measured.
    pub(super) fn check_function_tables(
        &self,
        call: &'static str,
        stage: Stage,
        interface: &PipelineInterface,
    ) -> Result<(), Error> {
        // No slot is refused where the table binds no visible-function table and no offset made
        // one null: the check below holds for each slot.
        if self.function_tables.is_empty() && self.null_tables.is_empty() {
            return Ok(());
        }
        for (slot, slot_interface) in interface.slots.range(buffer_slots(stage)) {
            let SlotInterface::Buffer(buffer) = slot_interface else {
                continue;
            };
            let index = slot.index;
            let table = self.function_tables.contains_key(&index);
            let unmeasured = match buffer.kind {
                BufferKind::Buffer => buffer.access.write,
                BufferKind::VisibleFunctionTable => false,
                _ => true,
            };
            let nulled = self.null_tables.contains(&index)
                && !matches!(buffer.kind, BufferKind::VisibleFunctionTable);
            if (unmeasured && table) || nulled {
                return Err(invalid(
                    call,
                    Problem::FunctionTable(FunctionTableProblem::TableAtBufferIndex { index }),
                ));
            }
        }
        Ok(())
    }

    /// Refuse a command of `call` whose `stage` function reads a buffer slot bound past its
    /// buffer's length (a compute table holds one, the module docs): Metal's read there is past
    /// the buffer, which memory decides. A slot the function does not read runs as Metal's does.
    pub(super) fn check_offsets(
        &self,
        call: &'static str,
        stage: Stage,
        interface: &PipelineInterface,
    ) -> Result<(), Error> {
        for slot in interface
            .slots
            .range(buffer_slots(stage))
            .map(|(slot, _)| slot)
        {
            if let Some((buffer, offset)) = self.buffers.get(&slot.index) {
                let (offset, length) = (*offset, buffer.length());
                if offset > length {
                    return Err(invalid(call, Problem::OffsetOutOfRange { offset, length }));
                }
            }
        }
        Ok(())
    }

    /// The table as the bind layer reads it. A buffer is stated in its space (`Buffer::space`),
    /// which the descriptor ranges, vertex buffer offsets and device addresses the bind layer
    /// derives count from: a no-copy buffer's bytes may start past its space's start, where its
    /// length and offset are as many bytes longer and its addresses as many lower, so every Metal
    /// address the layer derives is the buffer's. A heap buffer's space starts at its own byte 0,
    /// so the layer states its bytes alone. A buffer bound past its length is left out: a command
    /// whose function reads its slot is refused ([`check_offsets`](Self::check_offsets)), and no
    /// other reads it.
    pub(super) fn stage_table(&self) -> StageTable {
        let mut table = StageTable::default();
        for (&index, (buffer, offset)) in &self.buffers {
            let stride = self.attribute_strides.get(&index).copied();
            if let Some(bound) = bound_buffer(buffer, *offset, stride) {
                table.buffers.insert(index, bound);
            }
        }
        for (&index, bytes) in &self.bytes {
            table.buffers.insert(index, bound_bytes(bytes));
        }
        for (&index, texture) in &self.textures {
            table.textures.insert(index, bound_texture(texture));
        }
        for (&index, (sampler, clamps)) in &self.samplers {
            table
                .samplers
                .insert(index, BoundSampler::new(bound_sampler(sampler, *clamps)));
        }
        table
    }

    /// The table as `stage`'s function of `interface` reads it: `stage_table`, but a buffer bound
    /// where the function reads a visible-function table is left out, so the slot reads its
    /// neutral, a null table of size 0, as plain Metal reads such a buffer where its validation
    /// stops the command (an M3 Max and an M4), and so is a table bound where it reads a buffer,
    /// whose slot then reads zeros, as plain Metal's read (an M4). Its slots that tell the device's
    /// view base alignment apart are those of `views`, the state's, as
    /// [`at_bind_views`](Self::at_bind_views) leaves them.
    pub(super) fn stage_table_reading(
        &self,
        stage: Stage,
        interface: &PipelineInterface,
        views: &AtBindViews,
    ) -> StageTable {
        let mut table = self.stage_table();
        table.at_bind_views = self.at_bind_views(stage, views);
        table
            .buffers
            .retain(|&index, _| !self.reads_unbound((stage, interface), index));
        table
    }

    /// The slots of `stage` whose bind offset class tells the device's view base alignment apart
    /// for a command binding this table (`StageTable::at_bind_views`): the state's `views` of it,
    /// each that binds a buffer here at an offset the alignment does not divide left out, now and
    /// from then on ([`AtBindViews`]). Inline bytes leave none out.
    fn at_bind_views(&self, stage: Stage, views: &AtBindViews) -> u64 {
        let Some(kept) = views.of(stage) else {
            return 0;
        };
        let alignment = views.alignment;
        let viewed = kept.load(Ordering::Relaxed);
        let mut slots = viewed;
        let mut left = viewed;
        while left != 0 {
            let index = left.trailing_zeros();
            left &= left - 1;
            let moved = alignment != 0
                && !self.bytes.contains_key(&index)
                && (self.buffers.get(&index))
                    .is_some_and(|(_, offset)| !offset.is_multiple_of(u64::from(alignment)));
            if moved {
                slots &= !(1 << index);
            }
        }
        if slots != viewed {
            kept.fetch_and(slots, Ordering::Relaxed);
        }
        slots
    }

    /// Whether `stage`'s function of `interface` reads buffer index `index` as unbound
    /// ([`stage_table_reading`](Self::stage_table_reading)): a buffer is bound there and the
    /// function reads a visible-function table, or a table is bound there and the function reads a
    /// buffer it does not write, which plain Metal read as zeros, where its validation stops the
    /// command.
    fn reads_unbound(&self, (stage, interface): (Stage, &PipelineInterface), index: u32) -> bool {
        let slot = SlotRef {
            stage,
            class: SlotClass::Buffer,
            index,
        };
        let Some(SlotInterface::Buffer(buffer)) = interface.slots.get(&slot) else {
            return false;
        };
        let table_bound = self.function_tables.contains_key(&index);
        match buffer.kind {
            BufferKind::VisibleFunctionTable => !table_bound,
            BufferKind::Buffer => table_bound && !buffer.access.write,
            _ => false,
        }
    }

    /// Bring `kept`, the table as a stage's function read it at an earlier draw, to the table as
    /// `stage`'s function of `interface` reads it now
    /// ([`stage_table_reading`](Self::stage_table_reading)), each sampler as `maps` binds it, in
    /// place (`live_bind.rs`): every slot's binding is derived again and stored, so one whose
    /// object changed since with no call (a buffer's pages rebound or detached) is read as it is
    /// now, and `reshape` is told of each change that may change the shape.
    pub(super) fn refresh_reading(
        &self,
        kept: &mut StageTable,
        (stage, interface): (Stage, &PipelineInterface),
        (maps, views): (&SamplerMaps, &AtBindViews),
        reshape: &mut Reshape,
    ) {
        let read = (stage, interface);
        // The slots that tell the view base alignment apart first, which the buffers' shapes
        // read.
        reshape.field(&mut kept.at_bind_views, self.at_bind_views(stage, views));
        let alignment = views.alignment;
        kept.buffers.retain(|&index, _| {
            let bound = !self.reads_unbound(read, index)
                && (self.bytes.contains_key(&index)
                    || self
                        .buffers
                        .get(&index)
                        .is_some_and(|(buffer, offset)| *offset <= buffer.length()));
            if !bound {
                reshape.unbound();
            }
            bound
        });
        for (&index, (buffer, offset)) in &self.buffers {
            if self.bytes.contains_key(&index) || self.reads_unbound(read, index) {
                continue;
            }
            let stride = self.attribute_strides.get(&index).copied();
            if let Some(bound) = bound_buffer(buffer, *offset, stride) {
                let a = kept.class_alignment(index, alignment);
                reshape.slot(&mut kept.buffers, index, bound, |b| b.shape(a));
            }
        }
        for (&index, bytes) in &self.bytes {
            if !self.reads_unbound(read, index) {
                let a = kept.class_alignment(index, alignment);
                reshape.slot(&mut kept.buffers, index, bound_bytes(bytes), |b| b.shape(a));
            }
        }
        kept.textures.retain(|index, _| {
            let bound = self.textures.contains_key(index);
            if !bound {
                reshape.unbound();
            }
            bound
        });
        for (&index, texture) in &self.textures {
            reshape.slot(
                &mut kept.textures,
                index,
                bound_texture(texture),
                BoundTexture::shape,
            );
        }
        kept.samplers.retain(|index, _| {
            let bound = self.samplers.contains_key(index);
            if !bound {
                reshape.unbound();
            }
            bound
        });
        for (&index, (sampler, clamps)) in &self.samplers {
            let slot = SlotRef {
                stage,
                class: SlotClass::Sampler,
                index,
            };
            let desc = maps.sampler(slot, bound_sampler(sampler, *clamps));
            reshape.slot(
                &mut kept.samplers,
                index,
                BoundSampler::new(desc),
                BoundSampler::shape,
            );
        }
        #[cfg(debug_assertions)]
        self.check_reading(kept, read, (maps, views));
    }

    /// The slots the table's calls changed since its stamp was `read`, where it keeps them
    /// ([`Changes`]): those [`refresh_changed`](Self::refresh_changed) brings in.
    pub(super) fn changed_since(
        &self,
        read: Stamp,
    ) -> Option<impl Iterator<Item = (SlotClass, u32)> + '_> {
        Some(
            self.changes
                .since(read)?
                .map(|changed| (changed.class, changed.index)),
        )
    }

    /// [`refresh_reading`](Self::refresh_reading) of `kept`, read last under the stamp `read`
    /// through the same interface and maps: only the slots the table's calls changed since, where
    /// it keeps them ([`Changes`]) and its slots that tell the view base alignment apart are as
    /// `kept` holds them, so every other slot's binding and shape are as read then. Whether it
    /// brought them in; else nothing is changed, and the table is to be read whole.
    pub(super) fn refresh_changed(
        &self,
        (kept, read): (&mut StageTable, Stamp),
        (stage, interface): (Stage, &PipelineInterface),
        (maps, views): (&SamplerMaps, &AtBindViews),
        reshape: &mut Reshape,
    ) -> bool {
        let Some(changed) = self.changes.since(read) else {
            return false;
        };
        // A buffer's shape reads these slots: where they changed, every buffer is read again.
        if self.at_bind_views(stage, views) != kept.at_bind_views {
            return false;
        }
        let alignment = views.alignment;
        // The buffer slots some call since changed otherwise than by its offset (indices below
        // `BUFFER_SLOTS`): each is brought in whole.
        let bit = |index: u32| 1u64.checked_shl(index).unwrap_or(0);
        let whole = (changed.clone())
            .filter(|c| c.class == SlotClass::Buffer && !c.offset)
            .fold(0u64, |whole, c| whole | bit(c.index));
        for Changed {
            class,
            index,
            offset,
            ..
        } in changed
        {
            match class {
                SlotClass::Buffer if offset && bit(index) != 0 && whole & bit(index) == 0 => {
                    let read = (stage, interface);
                    self.refresh_offset((kept, index), read, alignment, reshape)
                }
                SlotClass::Buffer => {
                    self.refresh_buffer((kept, index), (stage, interface), alignment, reshape)
                }
                SlotClass::Texture => match self.textures.get(&index) {
                    Some(texture) => reshape.slot(
                        &mut kept.textures,
                        index,
                        bound_texture(texture),
                        BoundTexture::shape,
                    ),
                    None => {
                        if kept.textures.remove(&index).is_some() {
                            reshape.unbound();
                        }
                    }
                },
                SlotClass::Sampler => match self.samplers.get(&index) {
                    Some((sampler, clamps)) => {
                        let slot = SlotRef {
                            stage,
                            class: SlotClass::Sampler,
                            index,
                        };
                        let desc = maps.sampler(slot, bound_sampler(sampler, *clamps));
                        reshape.slot(
                            &mut kept.samplers,
                            index,
                            BoundSampler::new(desc),
                            BoundSampler::shape,
                        );
                    }
                    None => {
                        if kept.samplers.remove(&index).is_some() {
                            reshape.unbound();
                        }
                    }
                },
            }
        }
        #[cfg(debug_assertions)]
        self.check_reading(kept, (stage, interface), (maps, views));
        true
    }

    /// Buffer slot `index` of `kept` brought in place where every call since `kept` was read moved
    /// the offset (or attribute stride) of the buffer it bound alone ([`Changed`]): `kept`'s
    /// binding of the same buffer, its offset and stride the table's now. The buffer's space and
    /// addresses are as `kept` read them, as no no-copy buffer moved since (the callers' reads
    /// stand only while none did: `live_bind::ReadFrom`), so the buffer is not looked at
    /// (`WorkCounts::bound_offsets_moved`; checked against [`refresh_buffer`](Self::refresh_buffer)'s
    /// binding in builds with debug assertions). Else as `refresh_buffer`.
    fn refresh_offset(
        &self,
        (kept, index): (&mut StageTable, u32),
        read: (Stage, &PipelineInterface),
        alignment: u32,
        reshape: &mut Reshape,
    ) {
        let held = match (self.buffers.get(&index), kept.buffers.get(&index)) {
            (
                Some((buffer, offset)),
                Some(&BoundBuffer::Buffer {
                    length,
                    gpu_address,
                    device_address,
                    ..
                }),
            ) if !self.reads_unbound(read, index) && *offset <= buffer.length() => {
                Some((buffer, *offset, (length, gpu_address, device_address)))
            }
            _ => None,
        };
        let Some((buffer, offset, (length, gpu_address, device_address))) = held else {
            return self.refresh_buffer((kept, index), read, alignment, reshape);
        };
        let stride = self.attribute_strides.get(&index).copied();
        // The space's start before the buffer's byte 0, as `bound_buffer` states it.
        let base = length - buffer.length();
        let bound = BoundBuffer::Buffer {
            length,
            offset: base + offset,
            gpu_address,
            device_address,
            attribute_stride: stride,
        };
        #[cfg(debug_assertions)]
        assert_eq!(
            Some(bound),
            bound_buffer(buffer, offset, stride),
            "a buffer slot whose offset alone moved brought in otherwise than read whole"
        );
        buffer.device().shared().counts.bound_offset_moved();
        let a = kept.class_alignment(index, alignment);
        reshape.slot(&mut kept.buffers, index, bound, |b| b.shape(a));
    }

    /// Buffer slot `index` of `kept` brought in place as [`refresh_reading`](Self::refresh_reading)
    /// brings it: unbound where the function reads it unbound or nothing within its length is
    /// bound there, else its inline bytes or buffer, its shape of the class `alignment` tells it.
    fn refresh_buffer(
        &self,
        (kept, index): (&mut StageTable, u32),
        read: (Stage, &PipelineInterface),
        alignment: u32,
        reshape: &mut Reshape,
    ) {
        let bound = !self.reads_unbound(read, index)
            && (self.bytes.contains_key(&index)
                || (self.buffers.get(&index))
                    .is_some_and(|(buffer, offset)| *offset <= buffer.length()));
        if !bound {
            if kept.buffers.remove(&index).is_some() {
                reshape.unbound();
            }
            return;
        }
        let a = kept.class_alignment(index, alignment);
        if let Some(bytes) = self.bytes.get(&index) {
            reshape.slot(&mut kept.buffers, index, bound_bytes(bytes), |b| b.shape(a));
        } else if let Some((buffer, offset)) = self.buffers.get(&index) {
            let stride = self.attribute_strides.get(&index).copied();
            if let Some(bound) = bound_buffer(buffer, *offset, stride) {
                reshape.slot(&mut kept.buffers, index, bound, |b| b.shape(a));
            }
        }
    }

    /// That `kept`, brought in place by [`refresh_reading`](Self::refresh_reading) (or kept unread
    /// while nothing it reads changed: `live_bind::ReadFrom`), is the table `stage`'s function of
    /// `interface` reads whole ([`stage_table_reading`](Self::stage_table_reading)), each sampler
    /// as `maps` binds it and its slots that tell the view base alignment apart as `views` leaves
    /// them (debug builds): the same slots, each bound alike as its value prints (a float's sign
    /// told apart, every NaN alike: [`same_as_printed`]), and nothing else in the table. Compared
    /// slot by slot, with no table read whole made and nothing printed unless a slot differs, so a
    /// draw allocates as many times with debug assertions as without (`runtime_lane::frame_costs`
    /// holds its bound in every build). A slot that differs is named in the panic.
    #[cfg(debug_assertions)]
    pub(super) fn check_reading(
        &self,
        kept: &StageTable,
        read: (Stage, &PipelineInterface),
        (maps, views): (&SamplerMaps, &AtBindViews),
    ) {
        let (stage, _) = read;
        const DIFFERS: &str = "a stage table brought in place that is not the table read whole";
        // The slots that tell the view base alignment apart, which the buffers' shapes read: each
        // binds no buffer here at an offset the alignment does not divide, and they hold every slot
        // the state's views hold now. A slot leaves those for good, and another encoder of the
        // state may leave one out after this table read them, so `kept` may hold more than a
        // table read whole now would, never fewer.
        let now = views
            .of(stage)
            .map_or(0, |slots| slots.load(Ordering::Relaxed));
        assert!(
            now & !kept.at_bind_views == 0,
            "{DIFFERS}: {stage:?} tells the view base alignment apart at {:#x}, the state's views \
             at {now:#x}",
            kept.at_bind_views
        );
        let mut told = kept.at_bind_views;
        while told != 0 {
            let index = told.trailing_zeros();
            told &= told - 1;
            let moved = !self.bytes.contains_key(&index)
                && (self.buffers.get(&index)).is_some_and(|(_, offset)| {
                    views.alignment != 0 && !offset.is_multiple_of(u64::from(views.alignment))
                });
            assert!(
                !moved,
                "{DIFFERS}: {stage:?} buffer {index} tells the view base alignment {} apart at an \
                 offset it does not divide",
                views.alignment
            );
        }
        // A buffer slot as `stage_table_reading` reads it: inline bytes where they are bound,
        // else the buffer within its length, and nothing where the function reads it unbound.
        let buffer = |index: u32| {
            if self.reads_unbound(read, index) {
                return None;
            }
            match self.bytes.get(&index) {
                Some(bytes) => Some(bound_bytes(bytes)),
                None => self.buffers.get(&index).and_then(|(buffer, offset)| {
                    let stride = self.attribute_strides.get(&index).copied();
                    bound_buffer(buffer, *offset, stride)
                }),
            }
        };
        // Each slot read whole once: a kept slot against its value, then the slots not kept,
        // each of which reads nothing. A read of a no-copy buffer is a look at where it lies
        // (`WorkCounts::no_copy_lies_read`), one a slot here as in the refresh.
        for (&index, bound) in &kept.buffers {
            let whole = buffer(index);
            assert!(
                whole.as_ref() == Some(bound),
                "{DIFFERS}: {stage:?} buffer {index} kept as {bound:?}, read whole as {whole:?}"
            );
        }
        let not_kept = (self.buffers.keys().chain(self.bytes.keys()))
            .filter(|&&index| !kept.buffers.contains_key(&index));
        for &index in not_kept {
            let whole = buffer(index);
            assert!(
                whole.is_none(),
                "{DIFFERS}: {stage:?} buffer {index} not kept, read whole as {whole:?}"
            );
        }
        for (&index, bound) in &kept.textures {
            let whole = self.textures.get(&index).map(bound_texture);
            assert!(
                whole.as_ref() == Some(bound),
                "{DIFFERS}: {stage:?} texture {index} kept as {bound:?}, read whole as {whole:?}"
            );
        }
        for (&index, texture) in &self.textures {
            assert!(
                kept.textures.contains_key(&index),
                "{DIFFERS}: {stage:?} texture {index} not kept, read whole as {:?}",
                bound_texture(texture)
            );
        }
        let sampler = |index: u32| {
            self.samplers.get(&index).map(|(sampler, clamps)| {
                let slot = SlotRef {
                    stage,
                    class: SlotClass::Sampler,
                    index,
                };
                let mut whole = BoundSampler::new(bound_sampler(sampler, *clamps));
                whole.desc = maps.sampler(slot, whole.desc);
                whole
            })
        };
        for (&index, bound) in &kept.samplers {
            let whole = sampler(index);
            assert!(
                whole.is_some_and(|whole| same_as_printed(&whole, bound)),
                "{DIFFERS}: {stage:?} sampler {index} kept as {bound:?}, read whole as {whole:?}"
            );
        }
        for &index in self.samplers.keys() {
            assert!(
                kept.samplers.contains_key(&index),
                "{DIFFERS}: {stage:?} sampler {index} not kept, read whole as {:?}",
                sampler(index)
            );
        }
        // A table read whole binds nothing else (`stage_table` starts from the empty table), but
        // for the object stage's and the kernel's threadgroup lengths, the encoder's, which the
        // draw or the dispatch writes into the kept table after the read
        // (`RenderLowering::refresh_bind_state`, `LiveDispatch::refresh`).
        let encoders = matches!(stage, Stage::Object | Stage::Kernel);
        let lengths = !encoders && !kept.threadgroup_lengths.is_empty();
        assert!(
            !lengths && kept.acceleration_structures.is_empty(),
            "{DIFFERS}: {stage:?} keeps threadgroup lengths {:?} and acceleration structures {:?}, \
             where the table read whole has none",
            kept.threadgroup_lengths,
            kept.acceleration_structures
        );
    }

    /// Hold everything bound until `core`'s command buffer completes: a command uses it, and
    /// the command buffer lists it among its used resources (`used_resources.rs`). Once while it
    /// binds the same objects: where `core` retained the table before under its stamp now,
    /// nothing is retained again, nor where `core` retained it last and each object a call bound
    /// since was one `core` held as it was bound. Else each object `core` holds from an earlier retain
    /// (`retained::HeldBy`) is held and its key kept already, each a function of its object;
    /// `core` keeps the others' keys and takes them, each by another reference to it
    /// (`retained.rs`): its last first, so they go in the table's order, as one hold of them all
    /// went.
    pub(super) fn retain(&self, core: &mut EncoderCore) {
        let mut retained = lock(&self.retained.0);
        let objects = (&self.buffers, &self.textures, &self.samplers);
        retain_into(objects, self.stamp, &mut retained, core);
    }

    /// [`retain`](Self::retain) by its only holder: the borrow is exclusive, so no other thread
    /// can reach the table's record of its last retain, which is read and written without taking
    /// its lock (a render encoder's own tables, at each draw).
    pub(super) fn retain_exclusive(&mut self, core: &mut EncoderCore) {
        let ArgumentTable {
            buffers,
            textures,
            samplers,
            retained,
            stamp,
            ..
        } = self;
        let retained = retained.0.get_mut().unwrap_or_else(PoisonError::into_inner);
        retain_into((buffers, textures, samplers), *stamp, retained, core);
    }

    /// Let go of everything the table binds, as its encoder ends or its state is set back:
    /// [`let_go`] of each object, the last first, so they go in the table's order.
    /// Metal holds what an encoder was handed until its command buffer completes, a binding no
    /// command used included (probe `dealloc-probe2`: a buffer or texture bound after the last
    /// draw or dispatch).
    pub(super) fn let_go_all(&mut self, core: &mut EncoderCore) {
        let ArgumentTable {
            buffers,
            textures,
            samplers,
            ..
        } = self;
        // A stage the encoder bound nothing to (most often its object, mesh and tile stages').
        if buffers.is_empty() && textures.is_empty() && samplers.is_empty() {
            return;
        }
        for object in bound_in(buffers, textures, samplers).rev() {
            if !object.held_by().is(core.id()) {
                core.retain_shared(object.held());
            }
        }
    }
}

/// Let go of `object`, which a table bound until a call replaced or unbound it: `core` holds it
/// until its command buffer completes, as Metal holds every object an encoder was handed, a
/// binding replaced before any command used it included (the M3 Max under macOS 27.0.1, probes
/// `dealloc-probe` and `dealloc-probe2`: a no-copy buffer so bound, then released, ran its
/// deallocator only after the command buffer completed). Nothing is retained where `core` holds
/// it already, from a command that used a table that bound it ([`ArgumentTable::retain`]): a
/// bind before each draw retains nothing here.
fn let_go(core: &mut EncoderCore, object: Bound<'_>) {
    if !object.held_by().is(core.id()) {
        core.retain_shared(object.held());
    }
}

/// [`ArgumentTable::retain`] of a table that binds `objects` (its buffers, textures and sampler
/// states) under `stamp`, whose record of its last retain is `retained`. Where `core` retained the
/// table last and no call since bound an object `core` did not hold as it bound it
/// (`RetainedBy::unheld`), `core` holds every object the table binds, and keeps its key: nothing
/// is looked at. Else every object bound is.
fn retain_into(
    objects: Objects<'_>,
    stamp: Stamp,
    retained: &mut RetainedBy,
    core: &mut EncoderCore,
) {
    let (buffers, textures, samplers) = objects;
    let id = core.id();
    if retained.core == Some(id) {
        // No call changed what it binds since: the same objects, by the stamp.
        if retained.stamp == Some(stamp) {
            return;
        }
        // Calls changed it, binding only objects `core` held.
        if !retained.unheld {
            #[cfg(debug_assertions)]
            retained.check_held(objects);
            retained.stamp = Some(stamp);
            core.device().shared().counts.argument_table_held_at_bind();
            return;
        }
    }
    let bound = || bound_in(buffers, textures, samplers);
    let fresh = |object: &Bound<'_>| !object.held_by().is(id);
    core.touch_keys(bound().filter(fresh).filter_map(Bound::key));
    let mut taken = 0;
    for object in bound().rev().filter(fresh) {
        core.retain_shared(object.held());
        object.held_by().set(id);
        taken += 1;
    }
    retained.core = Some(id);
    retained.stamp = Some(stamp);
    retained.unheld = false;
    #[cfg(debug_assertions)]
    retained.note_retained(objects);
    if taken > 0 {
        core.device().shared().counts.argument_table_retained(taken);
    }
}

/// The buffer slots of `stage`, every index, as an interface's slots order them (by stage, class
/// and index).
fn buffer_slots(stage: Stage) -> std::ops::RangeInclusive<SlotRef> {
    let slot = |index| SlotRef {
        stage,
        class: SlotClass::Buffer,
        index,
    };
    slot(0)..=slot(u32::MAX)
}

/// The `count` indices from `first`, as many as lie below `u32::MAX`.
fn span(first: u32, count: usize) -> Range<u32> {
    first..first.saturating_add(u32::try_from(count).unwrap_or(u32::MAX))
}

/// The objects a table's `buffers`, `textures` and `samplers` bind, in the table's order: its
/// buffers, then its textures, then its sampler states, each by index.
fn bound_in<'a>(
    buffers: &'a BTreeMap<u32, (Buffer, u64)>,
    textures: &'a BTreeMap<u32, Texture>,
    samplers: &'a BTreeMap<u32, (SamplerState, Option<[f32; 2]>)>,
) -> impl DoubleEndedIterator<Item = Bound<'a>> {
    let buffers = buffers.values().map(|(buffer, _)| Bound::Buffer(buffer));
    let textures = textures.values().map(Bound::Texture);
    let samplers = samplers
        .values()
        .map(|(sampler, _)| Bound::Sampler(sampler));
    buffers.chain(textures).chain(samplers)
}

#[cfg(test)]
mod tests {
    use super::{Changes, SlotClass, Stamp, CHANGES_KEPT};

    /// A reader under the stamp before any of the calls the log keeps reads the slots those calls
    /// changed, in order, with their offset marks; it keeps every call after a stamp whose calls
    /// since changed at most `CHANGES_KEPT` slots, and never more than twice as many slots.
    #[test]
    fn the_changes_log_keeps_the_newest_calls() {
        let mut changes = Changes::default();
        // Each call: the stamp before it, and its slots with their offset marks.
        let mut calls: Vec<(Stamp, Vec<(u32, bool)>)> = Vec::new();
        let mut stamp = Stamp::fresh();
        for call in 0..200u32 {
            let first = call % 29;
            let indices = first..first + 1 + call % 3;
            let offset = call % 5 == 0;
            let after = Stamp::fresh();
            changes.note((stamp, after), SlotClass::Buffer, indices.clone(), offset);
            calls.push((stamp, indices.map(|index| (index, offset)).collect()));
            stamp = after;
            assert!(changes.slots.len() <= 2 * CHANGES_KEPT, "call {call}");
            for (at, (read, _)) in calls.iter().enumerate() {
                let want: Vec<(u32, bool)> = calls[at..].iter().flat_map(|c| c.1.clone()).collect();
                match changes.since(*read) {
                    Some(got) => {
                        let got: Vec<(u32, bool)> = got.map(|c| (c.index, c.offset)).collect();
                        assert_eq!(got, want, "call {call}, read before call {at}");
                    }
                    None => assert!(want.len() > CHANGES_KEPT, "call {call}, read before {at}"),
                }
            }
        }
    }
}
