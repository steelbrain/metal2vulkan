//! Where an indirect command buffer's records lie: placed by their first use, on a device without
//! unified memory that reserves a staging queue (`device/staging.rs`).
//!
//! - **Why.** In system memory, every record a kernel writes and every record an execution reads
//!   crosses the link to the GPU. On an RTX 3090 behind a PCIe x4 link, a kernel writing 8192
//!   render commands moved 32.3 MB in 17.2 ms, and the execution that read them 49.7 MB in
//!   26.0 ms, each near the link's 1.83 GB/s; in device-local memory on an RTX 5080 Laptop GPU
//!   that kernel took about a hundredth of its time in system memory, and the execution's pass
//!   about a sixth (`examples/icb_reads.rs`). The CPU's reads and writes of device-local memory
//!   cross the link each, and on a GPU that sleeps when idle a write wakes it first (about 0.95 s
//!   on the 5080), so the CPU never touches that memory: a buffer the CPU uses keeps its records
//!   in the host's memory.
//! - **The first use decides.** The records start in the host's buffer (`IcbShared::storage`,
//!   its header written and every command reset as it is made), placed nowhere yet. The first use
//!   of the GPU (an execution, `useResource`, a blit's copy, reset or optimize) places them in
//!   device-local memory ([`Home::device`], made then: a fill and an update of its header on the
//!   staging queue, not waited for, which every later submission of the device's waits for on
//!   the GPU). The first host access (a setter, a getter, a CPU reset, a commit or host step that
//!   reads them) places a `Shared` buffer's in the host's buffer for good, counted as a move of
//!   no bytes (`WorkCounts::icb_records_moved_to_host`), and a `Private` buffer's in device-local
//!   memory. A device with unified memory, or without a staging queue, keeps every buffer's in
//!   the host's buffer from its making.
//! - **Holds.** A command buffer that uses the records in device-local memory holds them
//!   ([`RecordsHold`], an encoder's retained object) from the encoding call until it completes or
//!   is dropped. A blit's copy, reset or optimize is GPU work, held as any other.
//! - **The move.** A `Shared` buffer's host access while no command buffer holds the records in
//!   device-local memory moves them into the host's buffer for good: through one copy of the
//!   whole storage on the staging queue, waited for, where the GPU may have written them
//!   (`WorkCounts::icb_record_bytes_moved`), else none, the host's buffer holding what the host
//!   left there. From then on the GPU reaches the host's buffer. The device-local buffer stays
//!   until the indirect command buffer goes.
//! - **Staged.** Any other host access of records in device-local memory (a `Private` buffer's,
//!   which never move, or a `Shared` one's while a command buffer holds them) copies the records
//!   it reads into the host's buffer first and those it writes back after, each on the staging
//!   queue and waited for, under the home's lock, so a move, a hold or another staged access
//!   waits for it; counted (`WorkCounts::icb_records_staged_accesses`) and announced once per
//!   device (`Degradation::StagedIndirectRecords`). The host never reads or writes device memory.
//! - **A recovery** (`carry.rs`) loses records in device-local memory with the lost `VkDevice`:
//!   the host's buffer is written again as a new buffer's, and the records stay there.

use super::{record_offset, storage_length, IndirectCommandBuffer};
use crate::address::Place;
use crate::buffer::{Buffer, Modes};
use crate::degradation::Degradation;
use crate::device::lock;
use crate::encoder::EncoderCore;
use crate::error::{internal, Error};
use crate::exit_guard;
use crate::texture::transfer_barrier;
use crate::{Device, StorageMode};
use ash::vk;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// Where the records lie (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Placed {
    /// In the host's buffer, which no use placed yet.
    Unplaced,
    /// In device-local memory ([`Home::device`]).
    Device,
    /// In the host's buffer, for good.
    Host,
}

/// The home's state, under its lock.
struct State {
    placed: Placed,
    /// The command buffers that hold the records in device-local memory ([`RecordsHold`]).
    held: u32,
    /// The GPU may have written the records in device-local memory.
    written: bool,
    /// The staging queue's value of the submission that set the device-local records' bytes as
    /// they were placed.
    init: u64,
    /// The host saw that submission complete.
    init_seen: bool,
}

/// Where an indirect command buffer's records lie (the module docs).
pub(super) struct Home {
    /// The device-local records, from the use that placed them there.
    device: OnceLock<Buffer>,
    /// The records lie in the host's buffer for good: one load, a host access's way in place.
    on_host: AtomicBool,
    /// The records never move: a `Private` buffer's.
    pinned: bool,
    state: Arc<Mutex<State>>,
}

impl Home {
    /// The home of a buffer of `storage` mode on `device`: in the host's buffer for good where
    /// the device has unified memory or no staging queue, else placed by the first use.
    pub(super) fn new(device: &Device, storage: StorageMode) -> Home {
        let placing = !device.properties().has_unified_memory && device.has_staging_queue();
        let placed = match placing {
            true => Placed::Unplaced,
            false => Placed::Host,
        };
        Home {
            device: OnceLock::new(),
            on_host: AtomicBool::new(!placing),
            pinned: storage == StorageMode::Private,
            state: Arc::new(Mutex::new(State {
                placed,
                held: 0,
                written: false,
                init: 0,
                init_seen: false,
            })),
        }
    }

    /// The device-local records, while the GPU reaches them there.
    pub(super) fn device(&self) -> Option<&Buffer> {
        match self.on_host.load(Ordering::Acquire) {
            true => None,
            false => self.device.get(),
        }
    }

    /// Lie in the host's buffer for good, under `state`, the caller's hold of the lock.
    fn to_host(&self, state: &mut State) {
        state.placed = Placed::Host;
        self.on_host.store(true, Ordering::Release);
    }

    /// Whether the records lie in device-local memory, which a recovery's carry loses: they lie
    /// in the host's buffer for good from now on (`carry.rs`).
    pub(super) fn leave_device(&self) -> bool {
        let mut state = lock(&self.state);
        let left = state.placed == Placed::Device;
        if left {
            self.to_host(&mut state);
        }
        left
    }
}

/// The drops of homes that waited for the submission that placed their records, which the host
/// had not seen complete, on one of the runtime's threads (a command buffer's completion that let
/// go of the buffer last), in the device tests' builds: a test's evidence that the wait ran there
/// (`tests::icb_records_home`).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) static RUNTIME_PLACING_WAITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

impl Drop for Home {
    fn drop(&mut self) {
        // The submission that set the device-local records' bytes holds nothing: the records go
        // once it completed. A section of the exit guard, wherever the last hold lets go (a
        // command buffer's completion, on its queue's waiter or a handler thread): while the
        // process exits, the records stay to its end, as their buffer's release leaves them
        // (`exit_guard`, `release.rs`).
        let state = lock(&self.state);
        if let (Some(records), false) = (self.device.get(), state.init_seen) {
            let Some(_busy) = exit_guard::enter() else {
                return;
            };
            let device = records.generation();
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            let unseen = device.staging_seen() < state.init && exit_guard::on_runtime_thread();
            let _ = device.wait_staging(state.init);
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            if unseen {
                RUNTIME_PLACING_WAITS.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// A command buffer's hold of an indirect command buffer's records in device-local memory (an
/// encoder's retained object), let go of as it completes or is dropped (the module docs).
pub(super) struct RecordsHold(Arc<Mutex<State>>);

impl Drop for RecordsHold {
    fn drop(&mut self) {
        let mut state = lock(&self.0);
        state.held = state.held.saturating_sub(1);
    }
}

/// What a host access does with the bytes of its runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Reach {
    /// Reads them.
    Reads,
    /// Reads them, and writes some.
    Writes,
    /// Writes every byte of them, reading none.
    Overwrites,
}

/// A host access of an indirect command buffer's records ([`IndirectCommandBuffer::host_access`]):
/// in the host's buffer as they lie there, or staged. A staged write's runs are copied back as it
/// goes. One word: every setter's batch holds one (`Batch`), and nearly every one is in place.
#[must_use]
pub(super) struct HostAccess<'a> {
    _staged: Option<Box<Staged<'a>>>,
}

/// A staged host access (the module docs): the home's lock, held until it goes.
struct Staged<'a> {
    icb: &'a IndirectCommandBuffer,
    state: Option<MutexGuard<'a, State>>,
    runs: Vec<Range<u64>>,
    back: bool,
}

impl Drop for Staged<'_> {
    fn drop(&mut self) {
        if self.back {
            let (Some(records), host) = (self.icb.0.home.device.get(), &self.icb.0.storage) else {
                return;
            };
            // A lost device lost the records: there is nothing to copy them back into.
            let _ = copy_runs(self.icb.generation(), records, host, &self.runs, false);
        }
        drop(self.state.take());
        let icb = self.icb;
        icb.generation()
            .degrade(Degradation::StagedIndirectRecords, || {
                let object = format!(
                    "a {:?} indirect command buffer of {} commands",
                    icb.storage_mode(),
                    icb.size()
                );
                let reason = match icb.0.home.pinned {
                    true => {
                        "its records lie in device-local memory for good, which the CPU reaches \
                         only through copies"
                            .to_owned()
                    }
                    false => {
                        "the GPU used its records first, in device-local memory, and a command \
                          buffer that uses them is not complete or dropped"
                            .to_owned()
                    }
                };
                (object, reason)
            });
    }
}

/// Copy `runs` of `from` into the same bytes of `to` on `device`'s staging queue, waited for:
/// from the device-local records into the host's buffer where `down`, else back.
fn copy_runs(
    device: &Device,
    records: &Buffer,
    host: &Buffer,
    runs: &[Range<u64>],
    down: bool,
) -> Result<(), Error> {
    let (records_raw, records_at) = records.at(0);
    let (host_raw, host_at) = host.at(0);
    let (from, from_at, to, to_at) = match down {
        true => (records_raw, records_at, host_raw, host_at),
        false => (host_raw, host_at, records_raw, records_at),
    };
    let regions: Vec<vk::BufferCopy> = (runs.iter())
        .filter(|run| !run.is_empty())
        .map(|run| vk::BufferCopy {
            src_offset: from_at + run.start,
            dst_offset: to_at + run.start,
            size: run.end - run.start,
        })
        .collect();
    if regions.is_empty() {
        return Ok(());
    }
    let mut staging = device.staging_lock()?;
    let value = staging.submit(|raw, command_buffer, _| {
        transfer_barrier(raw, command_buffer, true);
        // SAFETY: the staging command buffer is recording, outside a render pass; both buffers
        // are live, shared with the staging queue's family (`sharing.rs`), and the regions lie
        // inside each (the storage's bytes). No command buffer's work writes the host's buffer
        // meanwhile: the GPU reaches the records in device-local memory.
        unsafe { raw.cmd_copy_buffer(command_buffer, from, to, &regions) };
        transfer_barrier(raw, command_buffer, false);
    })?;
    drop(staging);
    device.wait_staging(value)
}

impl IndirectCommandBuffer {
    /// The records' buffer the GPU reaches now: the device-local one where they lie there, else
    /// the host's (the module docs). A command buffer reads it only once it holds the records
    /// ([`hold_records`](Self::hold_records)).
    pub(crate) fn storage(&self) -> &Buffer {
        self.0.home.device().unwrap_or(&self.0.storage)
    }

    /// Hold the records for the command buffer `core` encodes into, whose work uses them on the
    /// GPU, writing them where `writes`: placed in device-local memory where nothing placed them
    /// yet, and held there until the command buffer completes or is dropped. Nothing where they
    /// lie in the host's buffer (the module docs); there too where device-local memory cannot
    /// hold them.
    pub(crate) fn hold_records(&self, core: &mut EncoderCore, writes: bool) {
        let home = &self.0.home;
        if home.on_host.load(Ordering::Acquire) {
            return;
        }
        let mut state = lock(&home.state);
        match state.placed {
            Placed::Host => return,
            Placed::Unplaced => {
                if self.place_on_device(&mut state).is_err() {
                    home.to_host(&mut state);
                    return;
                }
            }
            Placed::Device => {}
        }
        state.held += 1;
        state.written |= writes;
        drop(state);
        core.retain(RecordsHold(Arc::clone(&home.state)));
    }

    /// Place the records in device-local memory, under `state`, the caller's hold of the home's
    /// lock: a new buffer at the records' address, its header written and every command reset, on
    /// the staging queue (the module docs).
    fn place_on_device(&self, state: &mut State) -> Result<(), Error> {
        const CALL: &str = "new_indirect_command_buffer";
        let device = self.generation();
        let length = storage_length(self.0.size);
        let modes = Modes::of(self.0.options)?;
        let records =
            device.device_records_buffer(CALL, length, modes, Place::Records(self.0.id.held.id))?;
        let header = (self.0.records)
            .encode_header(self.0.size, self.0.inheritance)
            .map_err(|_| internal("an indirect command buffer's header was refused"))?;
        let (raw, at) = records.at(0);
        let mut staging = device.staging_lock()?;
        let value = staging.submit(|commands, command_buffer, _| {
            transfer_barrier(commands, command_buffer, true);
            // SAFETY: the staging command buffer is recording, outside a render pass; the buffer
            // is live (`Home`'s drop waits for this submission) and shared with the staging
            // queue's family; the fill covers the storage's bytes, a multiple of 4 (records of
            // 2248 bytes, a 16-byte header and 4-byte flags), from the records' start, and the
            // header, 16 bytes, lies at its start.
            unsafe { commands.cmd_fill_buffer(command_buffer, raw, at, length, 0) };
            transfer_barrier(commands, command_buffer, true);
            // SAFETY: as the fill's.
            unsafe { commands.cmd_update_buffer(command_buffer, raw, at, &header) };
            transfer_barrier(commands, command_buffer, false);
        })?;
        drop(staging);
        if self.0.home.device.set(records).is_err() {
            return Err(internal(
                "an indirect command buffer's records placed twice",
            ));
        }
        state.placed = Placed::Device;
        state.init = value;
        Ok(())
    }

    /// A host access of the records' `runs` (byte ranges of the storage) that does `reach` with
    /// them, until the access goes: in place where they lie in the host's buffer; else a `Shared`
    /// buffer's records unheld move there, or the access is staged (the module docs). Inside it,
    /// the host's buffer ([`bytes`](Self::bytes)) holds the runs' bytes as they stand.
    pub(super) fn host_access(
        &self,
        runs: &[Range<u64>],
        reach: Reach,
    ) -> Result<HostAccess<'_>, Error> {
        let home = &self.0.home;
        if home.on_host.load(Ordering::Acquire) {
            return Ok(HostAccess { _staged: None });
        }
        let mut state = lock(&home.state);
        let counts = &self.generation().shared().counts;
        match state.placed {
            Placed::Host => return Ok(HostAccess { _staged: None }),
            Placed::Unplaced if !home.pinned => {
                home.to_host(&mut state);
                counts.icb_records_moved_to_host(0);
                return Ok(HostAccess { _staged: None });
            }
            Placed::Unplaced => self.place_on_device(&mut state)?,
            Placed::Device => {}
        }
        let Some(records) = home.device.get() else {
            return Err(internal(
                "an indirect command buffer's records placed nowhere",
            ));
        };
        if state.held == 0 && !home.pinned {
            // Where the GPU wrote nothing, the host's buffer holds what the device-local one does:
            // the header, every command reset, and what staged accesses wrote into both.
            let moved = match state.written {
                true => {
                    let whole = 0..storage_length(self.0.size);
                    let runs = std::slice::from_ref(&whole);
                    copy_runs(self.generation(), records, &self.0.storage, runs, true)?;
                    state.init_seen = true;
                    whole.end
                }
                false => 0,
            };
            home.to_host(&mut state);
            counts.icb_records_moved_to_host(moved);
            return Ok(HostAccess { _staged: None });
        }
        if reach != Reach::Overwrites {
            copy_runs(self.generation(), records, &self.0.storage, runs, true)?;
            state.init_seen = true;
        }
        counts.icb_records_staged_access();
        Ok(HostAccess {
            _staged: Some(Box::new(Staged {
                icb: self,
                state: Some(state),
                runs: runs.to_vec(),
                back: reach != Reach::Reads,
            })),
        })
    }

    /// [`host_access`](Self::host_access) of commands `range`'s records, and their written flags
    /// where `flags`: none of an empty range, which reaches no byte and places nothing.
    #[inline]
    pub(super) fn records_access(
        &self,
        range: Range<u32>,
        flags: bool,
        reach: Reach,
    ) -> Result<HostAccess<'_>, Error> {
        // Where the records lie in the host's buffer for good, as every setter of a buffer the CPU
        // writes finds them, the access is in place: no runs to name.
        if range.is_empty() || self.0.home.on_host.load(Ordering::Acquire) {
            return Ok(HostAccess { _staged: None });
        }
        let records = record_offset(range.start) as u64..record_offset(range.end) as u64;
        let runs = [records, self.flag_bytes(range)];
        let runs = match flags {
            true => &runs[..],
            false => &runs[..1],
        };
        self.host_access(runs, reach)
    }

    /// Whether the records lie in device-local memory now, and how many command buffers hold them
    /// there, for a device test of where they lie.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn records_on_device(&self) -> (bool, u32) {
        let state = lock(&self.0.home.state);
        (state.placed == Placed::Device, state.held)
    }
}
