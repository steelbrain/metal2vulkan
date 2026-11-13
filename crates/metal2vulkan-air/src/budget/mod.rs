//! Per-attempt budget: deadline, cancellation, work units, byte ledger, structural caps.
//!
//! Deterministic caps are typed [`Limit`]s with their values here; exceeding one is
//! `Error::LimitExceeded`, permanent for the input. A [`Meter`] is created per translation attempt
//! and threaded (`&mut`) through every input-scaled loop:
//!
//! - [`Meter::tick`] at poll points: counts deterministic work units, then checks cancellation and,
//!   every 64 polls or 65,536 work units (`CLOCK_POLLS`, `CLOCK_UNITS`), the wall deadline. Only the
//!   deadline outcome (`Budget{Time}`) is transient.
//! - [`Meter::charge`] (or [`Meter::reserve`]) **before** reserving any input-scaled container, so
//!   the ledger is deterministic and refusal happens before allocation.
//! - [`Meter::cap`] / [`Meter::bound`] for structural counts.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::{BudgetExceeded, Error, InternalError, InternalKind, Limit};

/// Instructions after inlining. Frozen by the spike: 1.55x the 1,353,686-instruction corpus
/// maximum.
pub const INLINED_INSTRUCTIONS: u64 = 1 << 21;
/// Irreducible nesting depth.
pub const IRREDUCIBLE_DEPTH: u64 = 8;
/// Emitted SPIR-V words (64 MiB of output). Provisional: 8 words per instruction at
/// [`INLINED_INSTRUCTIONS`], twice the 3.9 words per instruction of the largest analyzed entry's
/// stub estimate, so no entry the inline cap admits is refused on size alone; phase 6 re-sets it to
/// 2x the old translator's maximum output.
pub const SPIRV_WORDS: u64 = 1 << 24;
/// Deterministic work units per attempt. Frozen by the spike: 13.6x the corpus peak, about 5 s of
/// release work at the measured 25-32 M units/s, so it trips well before 20 s on the slowest
/// supported host class.
pub const WORK_UNITS: u64 = 1 << 27;
/// Ledger bytes per attempt: 450 MiB, under the 500 MiB attempt budget. Frozen by the spike at 1.9x
/// the corpus peak; since the ledger charges grown capacity and the SPIR-V builder the corpus peak
/// is 331 MB (the largest row with every function constant zero), 1.43x under the cap.
pub const LEDGER_BYTES: u64 = 450 * 1024 * 1024;
/// Bitcode types per module. Provisional.
pub const TYPE_COUNT: u64 = 1 << 20;
/// Metadata nodes per module. Provisional.
pub const METADATA_NODES: u64 = 1 << 24;
/// Bitstream block nesting depth. Provisional.
pub const BITSTREAM_NESTING: u64 = 64;
/// Decoded instructions per module. Provisional.
pub const INSTRUCTION_COUNT: u64 = 1 << 24;
/// Nesting depth of ABI struct layouts. Provisional: the decoder and every consumer that walks a
/// layout recurse at most this deep.
pub const LAYOUT_NESTING: u64 = 64;
/// Nesting depth of the structured constructs P8 emits. The builder nests constructs by closure
/// scope, so emission recurses once per level; at this cap the recursion fits a 2 MiB thread stack
/// in debug builds too (`structurize::emit`'s tests check it).
pub const CONSTRUCT_NESTING: u64 = 256;
/// Bytes of a function's private arena. A pointer into the arena is a `u32` byte offset, and
/// containment compares it with the arena's size as a `u32` constant, so a larger arena would alias
/// or be checked short.
pub const PRIVATE_ARENA_BYTES: u64 = u32::MAX as u64;
/// Bytes of the call stack a thread has on Metal: a kernel's `maxCallStackDepth` times the frame
/// it reserves may not pass it. Metal refuses a pipeline at 16359 frames of 16 bytes, 909 of 288,
/// and 8180 of 32 ("Compute pipeline exceeds available stack space", an M3 Max and an
/// M4): 261728 bytes, 256 KiB less 416, fits all three.
pub const CALL_STACK_BYTES: u64 = 261_728;
/// Wall-clock ceiling of one attempt (AGENTS.md).
pub const STANDARD_DEADLINE: Duration = Duration::from_secs(20);
/// The polls [`Meter::tick`] lets pass between two reads of the clock for its deadline, after the
/// first poll, which reads it. Reading the clock at every poll was 15% of a render state's
/// creation (an RTX 3080 host): a deadline passed is seen at most this many polls
/// late, or [`CLOCK_UNITS`] work units, whichever comes first.
pub(crate) const CLOCK_POLLS: u32 = 64;
/// The work units [`Meter::tick`] counts at most between two reads of the clock (about 2 ms of
/// release work at the 25-32 M units/s [`WORK_UNITS`] was frozen at), so a poll that counts much
/// work reads it at once.
pub(crate) const CLOCK_UNITS: u64 = 1 << 16;

/// The structurizer output-block cap `4(n+e)+1024` for `n` blocks and `e` edges.
pub fn output_blocks_cap(blocks: u64, edges: u64) -> u64 {
    blocks
        .saturating_add(edges)
        .saturating_mul(4)
        .saturating_add(1024)
}

impl Limit {
    /// The fixed cap of this limit, or `None` when it is relative to the input (`OutputBlocks`:
    /// `4(n+e)+1024` structurizer output blocks for an input of `n` blocks and `e` edges).
    pub const fn fixed_cap(self) -> Option<u64> {
        match self {
            Limit::InlinedInstructions => Some(INLINED_INSTRUCTIONS),
            Limit::OutputBlocks => None,
            Limit::IrreducibleDepth => Some(IRREDUCIBLE_DEPTH),
            Limit::SpirvWords => Some(SPIRV_WORDS),
            Limit::WorkUnits => Some(WORK_UNITS),
            Limit::LedgerBytes => Some(LEDGER_BYTES),
            Limit::TypeCount => Some(TYPE_COUNT),
            Limit::MetadataNodes => Some(METADATA_NODES),
            Limit::BitstreamNesting => Some(BITSTREAM_NESTING),
            Limit::InstructionCount => Some(INSTRUCTION_COUNT),
            Limit::LayoutNesting => Some(LAYOUT_NESTING),
            Limit::ConstructNesting => Some(CONSTRUCT_NESTING),
            Limit::PrivateArenaBytes => Some(PRIVATE_ARENA_BYTES),
            Limit::CallStackBytes => Some(CALL_STACK_BYTES),
        }
    }
}

/// The heap bytes a vector's capacity holds, as the ledger counts a container's footprint.
pub fn vec_bytes<T>(v: &Vec<T>) -> usize {
    v.capacity().saturating_mul(std::mem::size_of::<T>())
}

/// The heap bytes a hash map's capacity holds: one entry and one control byte per slot.
pub fn map_bytes<K, V, S>(m: &std::collections::HashMap<K, V, S>) -> usize {
    m.capacity()
        .saturating_mul(std::mem::size_of::<(K, V)>().saturating_add(1))
}

/// The heap bytes an ordered map or set of `len` entries of `entry` bytes holds, charged at twice
/// the entry size for node slack and links, as the rest of the core charges tree entries.
pub fn tree_bytes(len: usize, entry: usize) -> usize {
    len.saturating_mul(entry.saturating_mul(2))
}

/// The part of a growing container's footprint that is already on the ledger.
///
/// A container that cannot hold a [`Meter`] while it grows (the SPIR-V builder, which the
/// emitters drive through typed constructors) reports its footprint, and its owner charges the
/// growth at each poll point with [`Growth::charge_to`]. The ledger keeps the high-water mark:
/// a shrink is never released, so the charge only lags the container by the growth since the
/// last poll.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Growth {
    charged: usize,
}

impl Growth {
    /// A container whose first `charged` bytes are already on the ledger.
    pub fn from_charged(charged: usize) -> Growth {
        Growth { charged }
    }

    /// Charge the growth of a container whose footprint is now `footprint` bytes.
    pub fn charge_to(&mut self, footprint: usize, m: &mut Meter) -> Result<(), Error> {
        if footprint > self.charged {
            m.charge(footprint - self.charged)?;
            self.charged = footprint;
        }
        Ok(())
    }

    /// The bytes charged so far.
    pub fn charged(&self) -> usize {
        self.charged
    }
}

/// A cooperative cancellation flag shared between the caller and an attempt.
///
/// Clones share one flag: keep one, put a clone in a [`Budget`], and call
/// [`CancelToken::cancel`] from any thread. The attempt ends with `Error::Cancelled` at its next
/// poll point.
#[derive(Clone, Debug, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// A token that is not cancelled.
    pub fn new() -> CancelToken {
        CancelToken::default()
    }

    /// Request cancellation. Attempts observe it at their next poll point.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// The caller's budget for one compile call: an optional wall-clock deadline and an optional
/// cancellation token.
///
/// Every call that takes a `&Budget` (each `Device::compile_*`, and a render pipeline's keys,
/// which decode its vertex function) runs one attempt under it. Past the deadline the attempt
/// ends with `Error::Budget`; once its token is cancelled, with `Error::Cancelled`. Both are
/// transient: never cache them. Deterministic caps (work units, a memory ledger, structural
/// limits) apply whatever the budget; exceeding one is `Error::LimitExceeded`, permanent for the
/// input.
///
/// Build it with [`Budget::standard`] (a 20-second deadline), [`Budget::with_deadline`], or
/// `Default` (no deadline, no cancellation), then set `cancel`: the struct is
/// `#[non_exhaustive]`, so struct literals do not compile outside the crate.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct Budget {
    /// The instant the attempt must finish by; `None` is unbounded.
    pub deadline: Option<Instant>,
    /// The token that cancels the attempt; `None` cannot be cancelled.
    pub cancel: Option<CancelToken>,
}

impl Budget {
    /// The standard budget: a 20 s deadline from now, no cancellation.
    pub fn standard() -> Budget {
        Budget::with_deadline(Instant::now().checked_add(STANDARD_DEADLINE))
    }

    /// A budget with `deadline` (none: unbounded) and no cancellation.
    pub fn with_deadline(deadline: Option<Instant>) -> Budget {
        Budget {
            deadline,
            cancel: None,
        }
    }
}

/// What one piece of work charged a [`Meter`] ([`Meter::measured`]): its work units, the most
/// ledger bytes it held at once above the ledger it began at, and the bytes it still held when it
/// ended. A result kept for later asks charges each of them this ([`Meter::spend`]), so what a
/// meter answers never depends on whether the result was made or found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Spent {
    work: u64,
    peak: u64,
    held: u64,
}

impl Spent {
    /// The ledger bytes the work still held when it ended: what its result holds while it is
    /// kept.
    pub fn held(&self) -> u64 {
        self.held
    }
}

/// Per-attempt meter: deadline, cancellation, work units, and the byte ledger.
#[derive(Debug)]
pub struct Meter {
    deadline: Option<Instant>,
    cancel: Option<CancelToken>,
    work: u64,
    bytes: u64,
    peak_bytes: u64,
    work_cap: u64,
    byte_cap: u64,
    /// The polls until the next read of the clock (0: the next poll reads it).
    clock_in: u32,
    /// The work units counted since the last read of the clock.
    since_clock: u64,
}

impl Meter {
    /// A meter for one attempt under `budget`, with the standard work and ledger caps.
    pub fn new(budget: &Budget) -> Meter {
        Meter::with_caps(budget, WORK_UNITS, LEDGER_BYTES)
    }

    /// A meter with explicit work and ledger caps, for measurement and tests. Product entry points
    /// use [`Meter::new`].
    pub fn with_caps(budget: &Budget, work_cap: u64, byte_cap: u64) -> Meter {
        Meter {
            deadline: budget.deadline,
            cancel: budget.cancel.clone(),
            work: 0,
            bytes: 0,
            peak_bytes: 0,
            work_cap,
            byte_cap,
            clock_in: 0,
            since_clock: 0,
        }
    }

    /// Count `units` of deterministic work, then poll cancellation and the deadline: the clock
    /// at the first poll, then once every 64 polls or 65,536 work units (`CLOCK_POLLS`,
    /// `CLOCK_UNITS`).
    ///
    /// The work cap is checked first so a deterministic refusal never depends on timing. Callers
    /// tick at poll points (per block, per batch of instructions), not per trivial operation.
    pub fn tick(&mut self, units: u32) -> Result<(), Error> {
        self.work = self.work.saturating_add(u64::from(units));
        if self.work > self.work_cap {
            return Err(Error::LimitExceeded(Limit::WorkUnits));
        }
        if self.cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
            return Err(Error::Cancelled);
        }
        if let Some(deadline) = self.deadline {
            self.since_clock = self.since_clock.saturating_add(u64::from(units));
            if self.clock_in == 0 || self.since_clock >= CLOCK_UNITS {
                self.clock_in = CLOCK_POLLS;
                self.since_clock = 0;
                if Instant::now() >= deadline {
                    return Err(Error::Budget(BudgetExceeded::Time));
                }
            }
            self.clock_in -= 1;
        }
        Ok(())
    }

    /// Charge `bytes` to the ledger. Call **before** reserving the memory it accounts for.
    pub fn charge(&mut self, bytes: usize) -> Result<(), Error> {
        let bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
        let total = self.bytes.saturating_add(bytes);
        if total > self.byte_cap {
            return Err(Error::LimitExceeded(Limit::LedgerBytes));
        }
        self.bytes = total;
        self.peak_bytes = self.peak_bytes.max(total);
        Ok(())
    }

    /// Return `bytes` to the ledger when the container that was charged for them is dropped.
    pub fn release(&mut self, bytes: usize) {
        let bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
        self.bytes = self.bytes.saturating_sub(bytes);
    }

    /// Charge for `additional` more elements of `v`, then reserve them.
    pub fn reserve<T>(&mut self, v: &mut Vec<T>, additional: usize) -> Result<(), Error> {
        let bytes = additional
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(Error::LimitExceeded(Limit::LedgerBytes))?;
        self.charge(bytes)?;
        v.try_reserve(additional).map_err(|e| {
            Error::Internal(InternalError::new(
                InternalKind::AllocationFailed,
                format!("reserving {bytes} ledger-admitted bytes: {e}"),
            ))
        })
    }

    /// Make room for exactly `additional` more elements of a vector whose final length the caller
    /// knows, charging the capacity it adds before reserving it: the vector then fills to that
    /// length without growing, so it holds none of [`Meter::grow`]'s doubling slack, and the ledger
    /// carries its whole capacity as `grow` would. Charges nothing while the capacity already holds
    /// `additional` more.
    pub fn reserve_exact<T>(&mut self, v: &mut Vec<T>, additional: usize) -> Result<(), Error> {
        let needed = v
            .len()
            .checked_add(additional)
            .ok_or(Error::LimitExceeded(Limit::LedgerBytes))?;
        if needed <= v.capacity() {
            return Ok(());
        }
        let bytes = (needed - v.capacity())
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(Error::LimitExceeded(Limit::LedgerBytes))?;
        self.charge(bytes)?;
        v.try_reserve_exact(additional).map_err(|e| {
            Error::Internal(InternalError::new(
                InternalKind::AllocationFailed,
                format!("reserving {bytes} ledger-admitted bytes exactly: {e}"),
            ))
        })
    }

    /// Make room for `additional` more elements of a vector that grows one push at a time,
    /// charging the capacity the growth adds.
    ///
    /// [`Meter::reserve`] charges the elements asked for, which is exact for a vector filled to a
    /// known length. A vector grown by repeated pushes holds up to twice its length once it has
    /// doubled, so charging its elements leaves half its storage off the ledger. `grow` decides the
    /// doubled capacity itself (at least twice the current one, at least 4 elements), charges the
    /// bytes it adds before reserving them, and charges nothing while the capacity already holds
    /// `additional` more. The ledger therefore carries the vector's whole capacity, as the
    /// allocator does. A caller never releases per element: it returns `capacity × size` when it
    /// drops the vector, if at all.
    pub fn grow<T>(&mut self, v: &mut Vec<T>, additional: usize) -> Result<(), Error> {
        let needed = v
            .len()
            .checked_add(additional)
            .ok_or(Error::LimitExceeded(Limit::LedgerBytes))?;
        if needed <= v.capacity() {
            return Ok(());
        }
        let target = needed.max(v.capacity().saturating_mul(2)).max(4);
        let bytes = (target - v.capacity())
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(Error::LimitExceeded(Limit::LedgerBytes))?;
        self.charge(bytes)?;
        v.try_reserve_exact(target - v.len()).map_err(|e| {
            Error::Internal(InternalError::new(
                InternalKind::AllocationFailed,
                format!("growing by {bytes} ledger-admitted bytes: {e}"),
            ))
        })
    }

    /// Check a structural count against the fixed cap of `limit`.
    ///
    /// `limit` must have a fixed cap ([`Limit::fixed_cap`]); a relative limit such as
    /// `OutputBlocks` goes through [`Meter::bound`].
    pub fn cap(&self, limit: Limit, value: u64) -> Result<(), Error> {
        let max = match limit {
            Limit::WorkUnits => self.work_cap,
            Limit::LedgerBytes => self.byte_cap,
            _ => limit.fixed_cap().ok_or_else(|| {
                Error::Internal(InternalError::new(
                    InternalKind::Invariant,
                    format!("{limit:?} has no fixed cap; use Meter::bound"),
                ))
            })?,
        };
        self.bound(limit, value, max)
    }

    /// Check `value <= max` for a limit whose cap the caller computed from the input.
    pub fn bound(&self, limit: Limit, value: u64, max: u64) -> Result<(), Error> {
        if value > max {
            Err(Error::LimitExceeded(limit))
        } else {
            Ok(())
        }
    }

    /// Work units counted so far.
    pub fn work(&self) -> u64 {
        self.work
    }

    /// Bytes currently charged.
    pub fn ledger(&self) -> u64 {
        self.bytes
    }

    /// The highest ledger value reached (reported next to the allocator peak, and in a compile's
    /// `CompileReport`).
    pub fn ledger_peak(&self) -> u64 {
        self.peak_bytes
    }

    /// `f` run on this meter, with what it charged ([`Spent`]). A refusal is `f`'s own.
    pub fn measured<T, E>(
        &mut self,
        f: impl FnOnce(&mut Meter) -> Result<T, E>,
    ) -> Result<(T, Spent), E> {
        let (work, bytes, peak) = (self.work, self.bytes, self.peak_bytes);
        // The work's own peak, read from the ledger it begins at; the meter's is restored after.
        self.peak_bytes = bytes;
        let made = f(self);
        let spent = Spent {
            work: self.work.saturating_sub(work),
            peak: self.peak_bytes.saturating_sub(bytes),
            held: self.bytes.saturating_sub(bytes),
        };
        self.peak_bytes = self.peak_bytes.max(peak);
        made.map(|made| (made, spent))
    }

    /// Charge what `spent` states its work charged another meter, or this one before
    /// ([`Meter::measured`]): its work units, then its peak to the ledger, of which the bytes it
    /// held when it ended stay charged. The caps refuse it where they refused the work itself,
    /// counting its work units before its bytes.
    pub fn spend(&mut self, spent: Spent) -> Result<(), Error> {
        let mut work = spent.work;
        while work > 0 {
            let units = u32::try_from(work).unwrap_or(u32::MAX);
            self.tick(units)?;
            work = work.saturating_sub(u64::from(units));
        }
        self.charge(usize::try_from(spent.peak).unwrap_or(usize::MAX))?;
        let gone = spent.peak.saturating_sub(spent.held);
        self.release(usize::try_from(gone).unwrap_or(usize::MAX));
        Ok(())
    }
}

#[cfg(test)]
mod tests;
