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
//! beside the program's on threads of its own and makes MoltenVK name the device holds the lock
//! across its submission calls ([`DeviceWord::held`], taken inside the queue's lock), so neither
//! goes between the other's write and the loader's read. That work is the device's checks: its
//! submission checks (`submit_check.rs`), its queue-switch check (`queue_switch_check.rs`) and its
//! in-place check's internal submissions (`in_place_check.rs`, through
//! `Device::submit_internal_naming`). Their threads make no call the loader serves: the submission
//! checks' command buffers are made as the device is made, before the program holds it, and the
//! queue-switch check's on the program's thread that starts it, so no read of the word on the
//! checks' threads goes between a write of the program's work (its submissions on its own threads,
//! their completions on MoltenVK's) and the loader's read. MoltenVK names the device from its own
//! threads too, as a submission's Metal command buffer completes, and at other calls than
//! submissions: on a device whose word was found overwritten, by a restore or as a step starts
//! ([`DeviceWord::overwritten`], a fact of the device and never of its driver), the checks hold the
//! lock across each whole step of theirs that makes no use of a queue (their objects' making and
//! release) and from each submission through the wait for its completion (each check's through a
//! fence of its own, signalled as its Metal command buffer completes: the queue-switch check's
//! timelines are signalled before) and, where the queue's failures are relayed, for the relay's
//! mark after it ([`DeviceWord::end_step`]: the use's last submission on the queue, whose Metal
//! command buffer completes after the check's own), so the
//! runtime's loader-served calls on other threads may wait for one step of theirs, once, while they
//! run (the checks run once per driver and runtime, for milliseconds); elsewhere the word never
//! changes and they hold it across their submission calls alone. A narrower hold, around the calls
//! that read the word alone, cannot exclude the writer: MoltenVK writes from its own threads as
//! each of the checks' Metal command buffers completes, at no call of the runtime's, so only
//! keeping the checks' work out of flight while a loader-served call runs does. Seen on MoltenVK
//! 1.4.2 with each activity logged at once, before the lock: the scheduler's
//! `vkAllocateCommandBuffers` faulted reading the magic as its table while the submission checks
//! submitted beside it (`runtime_lane::device_commands`, which passed before them), still with the
//! lock across their submission calls alone, and still with it through the wait for a check's own
//! fence alone, MoltenVK's message as the relay's mark completed coming after that wait; and the
//! submission checks' own `vkAllocateCommandBuffers` on their thread faulted while the program's
//! `vkQueueSubmit` beside it logged. A driver thread that writes the word on its own for the
//! program's work (a message from MoltenVK's completion handlers as one of the program's command
//! buffers completes) is not held off.

use crate::device::lock;
use crate::device::relay::Mark;
use ash::vk::{self, Handle};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, PoisonError};
use std::thread::ThreadId;

#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
thread_local! {
    /// Whether this thread is one of the device's checks' ([`on_a_check_s_thread`]).
    static CHECK_THREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Mark this thread as one of the device's checks' (a thread of its own, for good), so the tests
/// count the restores made on it ([`DeviceWord::check_thread_restores`]).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn on_a_check_s_thread() {
    CHECK_THREAD.with(|thread| thread.set(true));
}

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
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    steps_held: std::sync::atomic::AtomicU64,
    /// The relay's marks the held steps waited for ([`DeviceWord::end_step`]), for the tests.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    marks_waited: std::sync::atomic::AtomicU64,
    /// The restores made on a thread of the device's checks ([`on_a_check_s_thread`]), for the
    /// tests: none, as their threads make no call the loader serves (the module docs).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    check_thread_restores: std::sync::atomic::AtomicU64,
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
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            steps_held: std::sync::atomic::AtomicU64::new(0),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            marks_waited: std::sync::atomic::AtomicU64::new(0),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            check_thread_restores: std::sync::atomic::AtomicU64::new(0),
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
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        if CHECK_THREAD.with(std::cell::Cell::get) {
            self.check_thread_restores.fetch_add(1, Ordering::Relaxed);
        }
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
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        self.steps_held.fetch_add(1, Ordering::Relaxed);
        Some(held.unwrap_or_else(|| self.held()))
    }

    /// The steps held whole so far ([`DeviceWord::step_hold`]).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn steps_held(&self) -> u64 {
        self.steps_held.load(Ordering::Relaxed)
    }

    /// The relay's marks the held steps waited for so far ([`DeviceWord::end_step`]).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn marks_waited(&self) -> u64 {
        self.marks_waited.load(Ordering::Relaxed)
    }

    /// The restores made on a thread of the device's checks so far ([`on_a_check_s_thread`]).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn check_thread_restores(&self) -> u64 {
        self.check_thread_restores.load(Ordering::Relaxed)
    }

    /// The end of a step of the runtime's own work, `held` where it is held whole
    /// ([`DeviceWord::step_hold`]), once the step's own waits for its submissions returned: a
    /// held step first waits for `marks`, the relay's marks after its uses of a queue
    /// (`device/relay.rs`), each the last submission of its use, so the lock goes only once
    /// their Metal command buffers completed and MoltenVK named the device for them (the module
    /// docs). The marks are let go either way; a wait that ends (the process exiting, a failed
    /// wait) waits for no other.
    pub(super) fn end_step<'m>(
        &self,
        held: Option<Held<'_>>,
        marks: impl IntoIterator<Item = Mark<'m>>,
    ) {
        if held.is_some() {
            for mark in marks {
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                self.marks_waited.fetch_add(1, Ordering::Relaxed);
                if !mark.wait() {
                    break;
                }
            }
        }
        drop(held);
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
