//! What a command buffer keeps of the objects its encoders use: each buffer, texture, sampler
//! state and pipeline state once, as Metal's command buffer retains each object once, until the
//! command buffer retires. The first use of an object in a recording takes its one reference;
//! every later use, and every binding slot an encoder fills with it, borrows the kept one
//! ([`Lent`]) and changes no count. The lists and their address sets keep their capacity from
//! one recording of a queue to the next, so a recording past the first allocates for its objects
//! only where it uses more than any before it did.

use crate::buffer::Buffer;
use crate::pipeline::ComputePipelineState;
use crate::render::RenderPipelineState;
use crate::sampler::SamplerState;
use crate::texture::Texture;
use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hasher};
use std::mem::ManuallyDrop;
use std::ops::Deref;

/// A handle to a shared object that is one `Arc` and nothing else: a copy of it made bit for bit
/// is the same object, and dropping no copy but one keeps the count right.
///
/// # Safety
///
/// The type holds exactly one `Arc` and no other state, and `address` is that `Arc`'s pointer.
pub(crate) unsafe trait Handle: Clone {
    /// The object's address: the same for every handle to it, and no other object's while it
    /// lives.
    fn address(&self) -> usize;
}

/// A handle borrowed from what a recording keeps: the same object, which this copy neither
/// retains nor releases. A command buffer keeps the object until it retires, and an encoder's
/// tables, which hold its borrows, end with the encoder, before its command buffer can commit.
pub(crate) struct Lent<T: Handle>(ManuallyDrop<T>);

impl<T: Handle> Lent<T> {
    /// A borrow of `object`.
    ///
    /// # Safety
    ///
    /// The recording the borrow is used in keeps `object` until after the borrow's last use.
    unsafe fn of(object: &T) -> Lent<T> {
        // SAFETY: `T: Handle` is one `Arc`, so the copy is the same object; it is never dropped,
        // so the count stays the kept handle's, and the caller keeps that one alive past the
        // copy's last use.
        Lent(ManuallyDrop::new(unsafe { std::ptr::read(object) }))
    }
}

impl<T: Handle> Deref for Lent<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: Handle> Clone for Lent<T> {
    fn clone(&self) -> Lent<T> {
        // SAFETY: the same object as `self`, which the recording keeps as long.
        unsafe { Lent::of(&self.0) }
    }
}

/// The objects of one kind a recording keeps, each once, in the order first used.
pub(crate) struct Kept<T: Handle> {
    list: Vec<T>,
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
    /// The borrow is used only while this list keeps its objects: before it is cleared.
    pub(crate) unsafe fn keep(&mut self, object: &T) -> Lent<T> {
        if self.addresses.insert(object.address()) {
            self.list.push(object.clone());
        }
        // SAFETY: the list keeps `object` now, until it is cleared, past which the caller uses no
        // borrow.
        unsafe { Lent::of(object) }
    }

    /// Keeps `object` where it is not kept yet: whether it was not.
    pub(crate) fn hold(&mut self, object: &T) -> bool {
        let first = self.addresses.insert(object.address());
        if first {
            self.list.push(object.clone());
        }
        first
    }

    /// The objects kept, in the order first used.
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, T> {
        self.list.iter()
    }

    /// The objects kept, as a slice.
    pub(crate) fn as_slice(&self) -> &[T] {
        &self.list
    }

    /// Releases every object, keeping the list's and the set's capacity.
    pub(crate) fn clear(&mut self) {
        self.list.clear();
        self.addresses.clear();
    }
}

/// Every object a recording keeps, by kind.
#[derive(Default)]
pub(crate) struct Objects {
    pub(crate) buffers: Kept<Buffer>,
    pub(crate) textures: Kept<Texture>,
    pub(crate) samplers: Kept<SamplerState>,
    pub(crate) pipelines: Kept<ComputePipelineState>,
    pub(crate) render_pipelines: Kept<RenderPipelineState>,
}

impl Objects {
    /// Releases every object, keeping each list's capacity for the next recording.
    pub(crate) fn clear(&mut self) {
        self.buffers.clear();
        self.textures.clear();
        self.samplers.clear();
        self.pipelines.clear();
        self.render_pipelines.clear();
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
