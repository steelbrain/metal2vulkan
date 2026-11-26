//! What a command buffer keeps of the objects its encoders use: each buffer, texture, sampler
//! state and pipeline state once, as Metal's command buffer retains each object once, and each
//! kernel variant past its kernel's cells that it dispatches (`crate::pipeline::Listed`), until
//! the command buffer retires. The first use of an object in a recording takes its one
//! reference, of the object alone ([`Handle::Kept`]): nothing a command buffer keeps counts as a
//! handle of the device, so the consumer's last handle is the device's last whatever work is in
//! flight.
//! Every later use, and every binding slot an encoder fills with it, borrows the object alone
//! ([`Lent`]), one word as the list keeps it, and changes no count. The lists and their address
//! sets keep their capacity from one recording of a queue to the next, up to [`KEPT_BOUND`] objects each, so a recording past
//! the first allocates for its objects only where it uses more than any before it did, or more
//! than the bound. They live in one allocation that moves whole between the queue and its command
//! buffers ([`Objects`]), so a command buffer allocates and frees none of it.

use crate::buffer::Buffer;
use crate::pipeline::{Compiled, ComputePipelineState, Listed};
use crate::render::RenderPipelineState;
use crate::sampler::SamplerState;
use crate::texture::Texture;
use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hasher};
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// A handle to a shared object: its `Arc`, and, where the object is one that holds its device
/// for the consumer (a buffer, a texture, a sampler state), a handle of the device beside it.
pub(crate) trait Handle {
    /// The object alone, which holds no handle of the device: what a command buffer keeps of
    /// it, and what an encoder borrows.
    type Kept: Clone + Uncount;

    /// The object, as the handle holds it.
    fn object(&self) -> &Self::Kept;

    /// `object`'s address: its `Arc`'s pointer, the same for every handle to it, and no other
    /// object's while it lives.
    fn address_of(object: &Self::Kept) -> usize;

    /// The object's address ([`address_of`](Self::address_of)).
    fn address(&self) -> usize {
        Self::address_of(self.object())
    }

    /// The object as a command buffer keeps it: one more count of its `Arc`, none of the device.
    fn kept(&self) -> Self::Kept {
        self.object().clone()
    }
}

/// An object borrowed from what a recording keeps: the object alone, as the list keeps it
/// ([`Handle::Kept`]), which this copy neither retains nor releases. A command buffer keeps the
/// object until it retires, and an encoder's tables, which hold its borrows, end with the encoder,
/// before its command buffer can commit.
pub(crate) struct Lent<T: Handle>(ManuallyDrop<T::Kept>);

impl<T: Handle> Lent<T> {
    /// A borrow of `object`.
    ///
    /// # Safety
    ///
    /// The recording the borrow is used in keeps the object until after the borrow's last use.
    unsafe fn of(object: &T::Kept) -> Lent<T> {
        // SAFETY: the caller keeps the object alive past the borrow's last use.
        Lent(unsafe { object.uncounted() })
    }

    /// The object's address ([`Handle::address_of`]).
    pub(crate) fn address(&self) -> usize {
        T::address_of(&self.0)
    }
}

impl<T: Handle> Deref for Lent<T> {
    type Target = T::Kept;

    fn deref(&self) -> &T::Kept {
        &self.0
    }
}

impl<T: Handle> Clone for Lent<T> {
    fn clone(&self) -> Lent<T> {
        // SAFETY: the same object as `self`, which the recording keeps as long.
        Lent(unsafe { self.0.uncounted() })
    }
}

/// An object of which a copy that takes no count can be made: an `Arc`, or a state that is one.
pub(crate) trait Uncount: Sized {
    /// Another copy of the object, made of the same pointer, that took no count: dropping it
    /// would give back what it never took, so it is never dropped.
    ///
    /// # Safety
    ///
    /// The object lives past the copy's last use.
    unsafe fn uncounted(&self) -> ManuallyDrop<Self>;
}

impl<T> Uncount for Arc<T> {
    unsafe fn uncounted(&self) -> ManuallyDrop<Arc<T>> {
        // SAFETY: the pointer is a live `Arc`'s; the copy is never dropped, and the caller keeps
        // the allocation alive past its last use.
        ManuallyDrop::new(unsafe { Arc::from_raw(Arc::as_ptr(self)) })
    }
}

/// An `Arc` borrowed from an owner that outlives every use of it: the same allocation, which this
/// copy neither retains nor releases, so taking it executes no locked instruction.
pub(crate) struct Uncounted<T>(ManuallyDrop<Arc<T>>);

impl<T> Uncounted<T> {
    /// A borrow of `arc`.
    ///
    /// # Safety
    ///
    /// `arc`'s allocation lives past the borrow's last use.
    pub(crate) unsafe fn of(arc: &Arc<T>) -> Uncounted<T> {
        // SAFETY: the pointer is a live `Arc`'s; the copy is never dropped, so it gives back no
        // count it did not take, and the caller keeps the allocation alive past its last use.
        Uncounted(ManuallyDrop::new(unsafe {
            Arc::from_raw(Arc::as_ptr(arc))
        }))
    }
}

impl<T> Deref for Uncounted<T> {
    type Target = Arc<T>;

    fn deref(&self) -> &Arc<T> {
        &self.0
    }
}

/// The most objects a list of one kind keeps room for past its recording: at the command
/// buffer's retirement a list that grew past it is freed, with its address set, so a rare large
/// recording costs its allocations once and leaves no memory kept. One bound for every kind: past
/// the 99th percentile of the distinct objects of each kind per command buffer in two recorded
/// game streams (at most 128, of textures), and a cap on memory, not a fit to them: a smaller
/// bound for one kind would make every recording that used one more object of it allocate again.
///
/// At most, then, a queue keeps for each command buffer it holds at once ([`Objects`]) six lists
/// at the bound, each 128 handles (1 KiB) and an address set of 256 buckets (2.3 KiB): 19.6 KiB,
/// so 1.2 MiB for a queue of 64 command buffers.
pub(crate) const KEPT_BOUND: usize = 128;

/// The objects of one kind a recording keeps, each once, in the order first used.
pub(crate) struct Kept<T: Handle> {
    list: Vec<T::Kept>,
    addresses: HashSet<usize, BuildHasherDefault<AddressHasher>>,
}

impl<T: Handle> Default for Kept<T> {
    fn default() -> Kept<T> {
        Kept {
            list: Vec::new(),
            addresses: HashSet::default(),
        }
    }
}

impl<T: Handle> Kept<T> {
    /// Keeps `object` where it is not kept yet, taking its one reference, and lends it.
    ///
    /// # Safety
    ///
    /// The borrow is used only while this list keeps its objects, before it is cleared, and while
    /// the recording holds a handle of the object's device: before its commit.
    pub(crate) unsafe fn keep(&mut self, object: &T) -> Lent<T> {
        // SAFETY: the caller's contract.
        unsafe { self.keep_object(object.object()) }
    }

    /// [`keep`](Self::keep) of an object itself, which no handle of the consumer's need hold: a
    /// sampler state's variant (`crate::sampler`).
    ///
    /// # Safety
    ///
    /// As [`keep`](Self::keep).
    pub(crate) unsafe fn keep_object(&mut self, object: &T::Kept) -> Lent<T> {
        if self.addresses.insert(T::address_of(object)) {
            self.list.push(object.clone());
        }
        // SAFETY: the list keeps the object now, until it is cleared, past which the caller uses
        // no borrow.
        unsafe { Lent::of(object) }
    }

    /// Keeps `object` where it is not kept yet: whether it was not.
    pub(crate) fn hold(&mut self, object: &T) -> bool {
        let first = self.addresses.insert(object.address());
        if first {
            self.list.push(object.kept());
        }
        first
    }

    /// The objects kept, in the order first used.
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, T::Kept> {
        self.list.iter()
    }

    /// The objects the list has room for.
    pub(crate) fn capacity(&self) -> usize {
        self.list.capacity()
    }

    /// Releases every object. A list with room for more than [`KEPT_BOUND`] is freed, with its
    /// address set; one within it keeps its room and its set's for the next recording.
    pub(crate) fn clear(&mut self) {
        if self.capacity() > KEPT_BOUND {
            *self = Kept::default();
        } else {
            self.list.clear();
            self.addresses.clear();
        }
    }
}

/// Every object a recording keeps, by kind ([`ByKind`]), in one allocation that moves whole: from
/// its queue to a command buffer made, and back at the command buffer's retirement, for the next
/// one made. The thread that makes a command buffer allocates none of it, so the thread that
/// retires one frees nothing the making thread allocated; only a queue's first command buffers,
/// one per command buffer it holds at once, allocate one.
#[derive(Default)]
pub(crate) struct Objects(Box<ByKind>);

impl Deref for Objects {
    type Target = ByKind;

    fn deref(&self) -> &ByKind {
        &self.0
    }
}

impl DerefMut for Objects {
    fn deref_mut(&mut self) -> &mut ByKind {
        &mut self.0
    }
}

/// The lists of each kind a recording keeps.
#[derive(Default)]
pub(crate) struct ByKind {
    pub(crate) buffers: Kept<Buffer>,
    pub(crate) textures: Kept<Texture>,
    pub(crate) samplers: Kept<SamplerState>,
    pub(crate) pipelines: Kept<ComputePipelineState>,
    pub(crate) render_pipelines: Kept<RenderPipelineState>,
    pub(crate) variants: Kept<Arc<Listed>>,
    /// The sRGB variants of kernels the recording dispatched, which no state holds.
    pub(crate) contents: Kept<Arc<Compiled>>,
}

impl ByKind {
    /// Releases every object, keeping each list's capacity, within [`KEPT_BOUND`], for the next
    /// recording.
    pub(crate) fn clear(&mut self) {
        self.buffers.clear();
        self.textures.clear();
        self.samplers.clear();
        self.pipelines.clear();
        self.render_pipelines.clear();
        self.variants.clear();
        self.contents.clear();
    }
}

/// Hashes an object's address: its bits mixed so that the low bits, which an allocation's
/// alignment leaves zero, are not the set's only index.
#[derive(Default)]
pub(crate) struct AddressHasher(u64);

impl Hasher for AddressHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 << 8 | u64::from(byte)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        }
    }

    fn write_usize(&mut self, address: usize) {
        let mixed = (address as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 = mixed ^ (mixed >> 29);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}
