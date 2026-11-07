//! The Vulkan loader's word at the start of the device handle, kept as the loader wrote it
//! (a copy of validation's `LoaderWord`, which is deleted when validation's executors move onto the
//! runtime).
//!
//! The Loader-Driver interface gives the loader a dispatchable object's first pointer-sized
//! entry: its dispatch table. MoltenVK 1.4.1 writes `ICD_LOADER_MAGIC` back into it every time it
//! names the device in a message it reports to a debug-utils messenger. The loader's own entry
//! points for the device (`vkAllocateCommandBuffers`, `vkGetDeviceQueue`, `vkGetDeviceProcAddr`,
//! `vkDestroyDevice`) then read the magic as their table and fault. A device the runtime creates
//! has such a messenger when its options ask for one, and an adopted device when its consumer
//! registered one, so the runtime records the word the loader wrote and puts it back before each
//! of those calls. On every other driver the word never changes and the restore writes nothing.
//! MoltenVK writes the magic over a queue's word too, where a message names the queue
//! (`device/relay.rs`): the queue's word is put back before each use.
//!
//! The device's word is a [`DeviceWord`]: each of the runtime's calls the loader serves puts it
//! back and runs under its lock ([`DeviceWord::restored`]), and the runtime's own work that runs
//! beside the program's and makes MoltenVK name the device (the submission checks,
//! `submit_check.rs`) holds the lock across its submission calls ([`DeviceWord::held`]), so
//! neither goes between the other's write and the loader's read. MoltenVK names the device from
//! its own threads too, as a submission's Metal command buffer completes, and at other calls than
//! submissions: on a device whose word was found overwritten, by a restore or as a step starts
//! ([`DeviceWord::overwritten`], a fact of the device and never of its driver), the checks hold
//! the lock across each whole step, the wait for their submissions' completion included, so the
//! runtime's loader-served calls on other threads may wait for one step of theirs, once, while
//! they run (the checks run once per driver and runtime, for milliseconds); elsewhere the word
//! never changes and they hold it across their submission calls alone. A narrower hold, around
//! the calls that read the word alone, cannot exclude the writer: MoltenVK writes from its own
//! threads as each of the checks' Metal command buffers completes, at no call of the runtime's,
//! so only keeping the checks' work out of flight while a loader-served call runs does. Seen on MoltenVK 1.4.2 with each activity
//! logged at once, before the lock: the scheduler's `vkAllocateCommandBuffers` faulted reading
//! the magic as its table while the checks submitted beside it (`runtime_lane::device_commands`,
//! which passed before the checks). A driver thread that writes the word on its own (a message
//! from MoltenVK's completion handlers) is not held off.

use crate::device::lock;
use ash::vk::{self, Handle};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, PoisonError};
use std::thread::ThreadId;

/// The loader's word of one live device handle, as it stood when the device reached the runtime.
pub(super) struct LoaderWord {
    entry: *mut usize,
    word: usize,
}

// SAFETY: the pointer names the device's first word, which outlives every use (the device is
// destroyed after the last restore), and every access is a whole-word read or write.
unsafe impl Send for LoaderWord {}
// SAFETY: as above.
unsafe impl Sync for LoaderWord {}

impl LoaderWord {
    /// The word at the start of `device`'s handle now.
    ///
    /// # Safety
    ///
    /// `device` is a live dispatchable handle the Vulkan loader (or a driver loaded directly)
    /// returned, its word is still the loader's, and [`LoaderWord::restore`] is called only while
    /// it lives.
    pub(super) unsafe fn of_device(device: vk::Device) -> LoaderWord {
        // SAFETY: the caller's.
        unsafe { LoaderWord::of_handle(device.as_raw()) }
    }

    /// The word at the start of the dispatchable handle `handle` now (a queue's, which MoltenVK
    /// overwrites as it does the device's, `device/relay.rs`).
    ///
    /// # Safety
    ///
    /// As [`LoaderWord::of_device`]'s, for the object `handle` names.
    pub(super) unsafe fn of_handle(handle: u64) -> LoaderWord {
        let entry: *mut usize = std::ptr::with_exposed_provenance_mut(handle as usize);
        // SAFETY: the caller's contract; a dispatchable handle points at the object's first
        // pointer-sized entry.
        let word = unsafe { entry.read_volatile() };
        LoaderWord { entry, word }
    }

    /// Whether the word is other than the recorded one now.
    ///
    /// # Safety
    ///
    /// The handle is still live.
    unsafe fn changed(&self) -> bool {
        // SAFETY: the caller's contract.
        unsafe { self.entry.read_volatile() != self.word }
    }

    /// Put the recorded word back where the driver overwrote it: whether it did.
    ///
    /// # Safety
    ///
    /// The handle is still live.
    pub(super) unsafe fn restore(&self) -> bool {
        // SAFETY: the caller's contract: the object is live, and its first entry is the loader's.
        unsafe {
            let overwritten = self.entry.read_volatile() != self.word;
            if overwritten {
                self.entry.write_volatile(self.word);
            }
            overwritten
        }
    }
}

/// The device's loader word, with the lock its writers and readers of the runtime's own take (the
/// module docs). A thread may take the lock again while it holds it (a caller's
/// `Device::device_proc_addr` inside its `Device::with_loader_word`).
pub(super) struct DeviceWord {
    word: LoaderWord,
    /// Whether the word was found overwritten: the driver names the device in its messages.
    overwritten: AtomicBool,
    /// The steps of the runtime's own work held whole ([`DeviceWord::step_hold`]), for the tests.
    #[cfg(test)]
    steps_held: std::sync::atomic::AtomicU64,
    /// The thread that holds the lock and how many holds it has.
    holder: Mutex<Option<(ThreadId, usize)>>,
    /// Signalled as the last hold goes.
    free: Condvar,
}

/// A hold of a [`DeviceWord`]'s lock, released when dropped.
pub(super) struct Held<'a>(&'a DeviceWord);

impl Drop for Held<'_> {
    fn drop(&mut self) {
        let mut holder = lock(&self.0.holder);
        if let Some((_, holds)) = holder.as_mut() {
            *holds -= 1;
            if *holds == 0 {
                *holder = None;
                self.0.free.notify_one();
            }
        }
    }
}

impl DeviceWord {
    /// The word at the start of `device`'s handle now.
    ///
    /// # Safety
    ///
    /// As [`LoaderWord::of_device`]'s, with [`DeviceWord::restored`] for its restores.
    pub(super) unsafe fn of_device(device: vk::Device) -> DeviceWord {
        DeviceWord {
            // SAFETY: the caller's.
            word: unsafe { LoaderWord::of_device(device) },
            overwritten: AtomicBool::new(false),
            #[cfg(test)]
            steps_held: std::sync::atomic::AtomicU64::new(0),
            holder: Mutex::new(None),
            free: Condvar::new(),
        }
    }

    /// `call`, one the loader serves (its entry point reads the word first), with the recorded
    /// word put back where the driver overwrote it, under the word's lock.
    ///
    /// # Safety
    ///
    /// The device is still live.
    pub(super) unsafe fn restored<R>(&self, call: impl FnOnce() -> R) -> R {
        let _held = self.held();
        // SAFETY: the caller's contract.
        if unsafe { self.word.restore() } {
            self.overwritten.store(true, Ordering::Relaxed);
        }
        call()
    }

    /// Whether the word was found overwritten, by a restore or now: a fact of this device, never
    /// of its driver (MoltenVK writes it only where it names the device in a message, as with
    /// each activity logged at once). On such a device the runtime's own work beside the
    /// program's holds the lock across its whole steps (the module docs).
    ///
    /// # Safety
    ///
    /// The device is still live.
    pub(super) unsafe fn overwritten(&self) -> bool {
        if self.overwritten.load(Ordering::Relaxed) {
            return true;
        }
        // SAFETY: the caller's contract.
        let now = unsafe { self.word.changed() };
        if now {
            self.overwritten.store(true, Ordering::Relaxed);
        }
        now
    }

    /// A hold across a whole step of the runtime's own work beside the program's, the driver's
    /// completion of the step's submissions included, where the word was found
    /// [`overwritten`](DeviceWord::overwritten): `held`, or a new hold where none is given.
    /// `None`, and `held` let go, on any other device.
    ///
    /// # Safety
    ///
    /// The device is still live.
    pub(super) unsafe fn step_hold<'a>(&'a self, held: Option<Held<'a>>) -> Option<Held<'a>> {
        // SAFETY: the caller's contract.
        if !unsafe { self.overwritten() } {
            return None;
        }
        #[cfg(test)]
        self.steps_held.fetch_add(1, Ordering::Relaxed);
        Some(held.unwrap_or_else(|| self.held()))
    }

    /// The steps held whole so far ([`DeviceWord::step_hold`]).
    #[cfg(test)]
    pub(crate) fn steps_held(&self) -> u64 {
        self.steps_held.load(Ordering::Relaxed)
    }

    /// A hold of the word's lock, for work of the runtime's own that makes the driver name the
    /// device in a message and so write over the word, while it runs beside the program's: none
    /// of the runtime's calls the loader serves is between its restore and its read meanwhile.
    /// Held across calls into the driver, and across a wait for the GPU only on a device whose
    /// word was found [`overwritten`](DeviceWord::overwritten); taken inside a queue's lock or
    /// with none held, never before taking one (a program's use of the queue may call
    /// [`DeviceWord::restored`] under it).
    pub(super) fn held(&self) -> Held<'_> {
        let me = std::thread::current().id();
        let mut holder = lock(&self.holder);
        loop {
            match holder.as_mut() {
                None => {
                    *holder = Some((me, 1));
                    break;
                }
                Some((thread, holds)) if *thread == me => {
                    *holds += 1;
                    break;
                }
                Some(_) => {
                    holder = (self.free.wait(holder)).unwrap_or_else(PoisonError::into_inner);
                }
            }
        }
        Held(self)
    }
}
