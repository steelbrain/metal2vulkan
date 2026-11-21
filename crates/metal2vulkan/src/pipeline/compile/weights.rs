//! The weight of each lowering stage's work against the work cap
//! ([`WORK_UNITS`](crate::budget::WORK_UNITS)): the sixteenths of a work unit one unit it ticks
//! counts ([`Meter::weighted`](crate::budget::Meter::weighted)).
//!
//! A unit ticked takes a different time in each stage: one of the lockstep pass's (sixteen lane
//! steps, or a pair of accesses considered) about 28 ns, one of emission's about 34 ns, one of
//! rolling's under 3 ns. Each weight is its stage's release cost per ticked unit, as sixteenths of
//! 61 ns: a work unit is 61 ns of lowering on a Ryzen 7 7840HS and 119 ns on the slowest CPU
//! measured, an i7-8700B (Windows), which runs the translator 1.95 times slower. The cap's 2^27
//! units are then about 16 s of lowering there whichever stages spend them, under the 20 s ceiling
//! of an attempt.
//!
//! The costs, from the lowerings of one recorded stream's first 217 frames (95 on a Radeon 780M
//! host's 7840HS, 77 on an RTX 2080 host's 7940HS): each stage's average over all of them or over
//! the lowering that ticked the stage most, whichever is higher. Work outside these stages (the
//! front end's decoding, a refusal location's analysis) counts a whole unit a tick.
//!
//! | stage | ns per ticked unit | weight |
//! |---|---|---|
//! | specialize | 25.4 | 7 |
//! | lockstep | 27.8 | 8 |
//! | memory | 21.7 | 6 |
//! | select | 22.1 | 6 |
//! | structurize | 15.3 | 5 |
//! | emit | 33.8 | 9 |
//! | roll | 2.5 | 1 |
//! | hoist | 4.3 | 2 |
//! | the rest of a lowering | 34.4 | 10 |

/// [`specialize_one`](super::specialize_one): decoding, linking and specializing the entry.
pub(crate) const SPECIALIZE: u32 = 7;
/// The lockstep pass ([`lockstep::order`](crate::lower::simd::lockstep::order)).
pub(crate) const LOCKSTEP: u32 = 8;
/// The memory plan ([`analyze_memory`](crate::memory::plan::analyze_memory)).
pub(crate) const MEMORY: u32 = 6;
/// Strategy selection, the structurizer inside it aside.
pub(crate) const SELECT: u32 = 6;
/// The structurizer, inside selection and emission.
pub(crate) const STRUCTURIZE: u32 = 5;
/// Emission ([`emit_module`](crate::lower::emit::emit_module)), the structurizer inside it aside.
pub(crate) const EMIT: u32 = 9;
/// Rolling repeated regions ([`roll_entry`](crate::spirv::roll::roll_entry)), and writing them as
/// kept routines ([`outline_entry`](crate::spirv::outline::outline_entry)): the same spine, keys
/// and exact comparisons, so rolling's weight (the routines' work is not measured apart).
pub(crate) const ROLL: u32 = 1;
/// Hoisting uniform loads ([`hoist_uniform_loads`](crate::spirv::hoist::hoist_uniform_loads)).
pub(crate) const HOIST: u32 = 2;
/// The rest of a lowering: its key, the passes after emission, its artifact.
pub(crate) const REST: u32 = 10;
