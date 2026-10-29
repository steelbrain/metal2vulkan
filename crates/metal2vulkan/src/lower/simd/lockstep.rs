//! Lockstep made explicit (device memory too): barriers that order the threadgroup-memory and
//! device-memory accesses a kernel leaves to the lockstep of Apple's simdgroups.
//!
//! **What Metal guarantees.** The Metal Shading Language specification (4.1, §4.4.1, §5.2.3.6,
//! §6.10) defines a simdgroup as threads that execute concurrently, and orders threadgroup
//! memory between threads only through `threadgroup_barrier` and `simdgroup_barrier` with
//! `mem_threadgroup` (or atomics). It never states that the threads of a simdgroup execute in
//! lockstep, and two threads' conflicting non-atomic accesses between barriers are a data race
//! it does not define.
//!
//! **What Apple GPUs do.** Measured on Apple GPUs, a simdgroup's 32 threads execute in lockstep: an
//! instruction completes for every active thread before the next starts. So two conflicting
//! threadgroup accesses of different threads of one simdgroup between two barriers run in program
//! order there, and kernels rely on it (a lane reads a slot another lane of its simdgroup wrote one
//! instruction earlier). Vulkan gives no such order, not even within a subgroup: lavapipe runs a
//! workgroup in short chunks of invocations, each up to the next barrier, and returns other bytes
//! from the same SPIR-V.
//!
//! **Detection.** Within a barrier interval (the stretch between two barriers that order
//! threadgroup memory), a pair of accesses `a` then `b` conflicts when `b` is reachable from `a`
//! on a barrier-free path, one of them writes, and the IR does not prove that two different
//! threads of one simdgroup touch disjoint bytes (`facts`):
//!
//! - different objects (root sets), or byte ranges of one object that the bounds of every index
//!   keep apart (spans: constants, and values bounded by a threadgroup's 1024 threads, a
//!   simdgroup's 32, masks and shifts, or a loop counter's exit test);
//! - disjoint constant ranges off one anchor (the address up to its last non-constant step,
//!   with a constant that step's index adds folded into the offset where it cannot wrap) that is
//!   either the same for both threads or differs between them by an amount whose bounds keep the
//!   ranges apart;
//! - disjoint ranges into one array row: when the last non-constant index steps through an
//!   array whose earlier steps are all multiples of its size, and its bounds keep the offset
//!   inside the array, two addresses meet only where their offsets into it do, whatever the
//!   earlier indices are (so they need not be uniform, nor settled across iterations);
//! - two extended indices of one GEP shape where one is a sum `u + t` that does not wrap, `u`
//!   uniform in the simdgroup, and the other runs only under a branch that tested it below `u`
//!   (a tree reduction's `if (t < h) a[t] += a[t + h]`), unsigned for zero-extended indices and
//!   signed, with `t` not negative, for sign-extended ones.
//!
//! An anchor or row index differs between threads when it depends on the thread through
//! `thread_index_in_simdgroup` or `thread_index_in_threadgroup` and exact additions, scalings,
//! extensions, truncations, and masks to a power of two, with the bounds a threadgroup's 1024
//! threads and a simdgroup's 32 put on the values (two threads of one simdgroup are 1 to 31
//! apart), or through a loop counter that starts there and steps by more than the starts differ
//! (every value is the start plus a multiple of the step, so two threads' values differ at
//! whatever iterations); a counter whose back edge is never taken is its start. A component of
//! `thread_position_in_grid` or `thread_position_in_threadgroup` does not differ between every two
//! threads (two threads of a simdgroup share its `x` in a threadgroup narrower than 32), but it is
//! the simdgroup's threadgroup origin plus a position below 1024: two threads' are equal or 1 to
//! 1023 apart, so through the same steps it clears a pair whose bytes are apart both at one anchor
//! and at anchors that far apart (`out[12 * gid + k]` against `out[12 * gid + j]`, `j != k`), but
//! never by itself makes one thread at most run an access. Two commutative updates of one class whose
//! results are unused never conflict, and neither do two stores of one value every thread of the
//! simdgroup stores alike (to locations equal or apart by their alignment), nor such a store at
//! one address for the whole simdgroup and a read that follows it in every thread (the read
//! finds that value whichever thread's store landed last), nor a plain store and itself in a later
//! iteration at an address each thread keeps through the loops holding it, in a block every thread
//! of the simdgroup runs as often (in every order the last store to a location is one thread's
//! last there, as under lockstep; a read between is a pair of its own), nor two accesses one thread of a
//! simdgroup at most runs: both run only where one value that differs between any two threads
//! of the simdgroup, as an index above may, equals one value they share (`if (tid == 0)`), both
//! computed outside every loop holding the pair. Any other pair may conflict, which is the sound
//! side: the cost of a false conflict is a barrier. Pairs on a path through a loop's back edge are
//! conflicts across iterations, and only values computed outside every loop holding the pair clear
//! them (for a stepping counter, its start). Two threads executing the same instruction are left
//! alone: lockstep orders instructions, not the threads of one.
//!
//! **Placement.** Each conflicting pair is ordered by a barrier on every barrier-free path from
//! `a` to `b`, with `mem_threadgroup`, at one of four places: right after `a`, right before `b`,
//! at the start of a block `Z` every such path crosses (a post-dominator of `a`'s block that
//! dominates `b`'s but not `a`'s), or, for a pair across iterations, in the header of a loop
//! holding both that no barrier-free path from `a` to `b` avoids (each later iteration starts
//! there). Which barrier depends on who executes the place together, as
//! `uniform::Uniformity::under_barriers` proves it: the uniformity analysis taking Metal's
//! barrier contract, so a block holding a program barrier, and a block in no loop that runs
//! exactly when such a block does, are uniform.
//!
//! - a **threadgroup barrier** at a place uniform across the threadgroup. It orders every
//!   thread of the threadgroup and raises no simd class.
//! - otherwise a **simdgroup barrier** at a place uniform within each simdgroup (where its index
//!   in the threadgroup and the results of its collective operations are uniform too). On a
//!   32-lane subgroup (`Direct`) it is a subgroup control barrier, which orders the invocations
//!   of its tangle, the whole simdgroup there; in the `Emulated` tier it is a workgroup barrier.
//! - otherwise a simdgroup barrier right before `b` that is exact only when every thread walks
//!   it: the kernel is compiled on the `Emulated` tier, whose predicated walk takes every thread
//!   through each barrier ([`Lockstep::emulated`]). On a device that runs a simdgroup's 32 threads
//!   in lockstep itself (`VkProperty::SimdgroupLockstep`: MoltenVK on an Apple GPU, where the
//!   module runs on the hardware and through the compiler the Metal kernel does), no barrier stands
//!   there and the kernel keeps its tier: the device's lockstep orders the pair as Apple's does for
//!   the original, the module requires the property, and [`Lockstep::device`] counts the places
//!   (the user's decision). Where the entry reconverges maximally on 32-lane subgroups
//!   (`shaderMaximalReconvergence` and a `Direct` or `Cluster32` tier for the barrier's class), the
//!   barrier there is a subgroup control barrier in divergent flow, which Vulkan defines for the
//!   invocations that run it together: the tangle maximal reconvergence makes the threads that took
//!   the same path, the threads Apple's simdgroup runs that instruction for. It orders the pair as
//!   lockstep does, so the kernel keeps the tier its class gives.
//!
//! Each pair names an interval of positions in one block where its barrier may stand; one
//! barrier at a position of the most intervals serves them all (interval stabbing, threadgroup
//! barriers first). It stands at the latest position its intervals share, right before the `b`
//! that ends one, unless they all run to the block's terminator (`b` lies in another block): then
//! right after the last `a` that starts one, or at the block's start, not in trailing code the
//! pairs do not reach past. The inserted calls are ordinary `seq_cst` `air.wg.barrier` /
//! `air.simdgroup.barrier` calls through the module's own declaration in whichever form it has (AIR
//! 2.8 or 2.9's, which adds the order operand), so every later stage (uniformity, predication,
//! emission) treats them as the program's own; [`Lockstep::barriers`] names them for the census.
//!
//! **Device memory**. The same holds for a kernel's buffers: a lane may read a device word another
//! lane of its simdgroup stored one instruction earlier, and Apple's lockstep orders the pair where
//! Vulkan orders nothing (NVIDIA's independent thread scheduling and its compiler's reordering of
//! accesses it proves apart within one thread included). The analysis runs a second pass over
//! device memory ([`facts::Space::Device`]): its objects are the buffer parameters, device globals,
//! and the buffer a pointer loaded from memory names (two loads of one argument-buffer member name
//! one), its intervals end at the barriers that order device or threadgroup memory, and every
//! barrier it inserts carries `mem_device` beside `mem_threadgroup` where the threadgroup pass
//! needed one too, so one barrier serves both. Two bindings of one buffer, or a binding and a
//! pointer loaded into it, are taken as distinct objects: a pair that meets only through such an
//! overlap is not ordered. `constant` memory is never written; texture memory (`mem_texture`) is
//! not ordered.
//!
//! **Residue.** Lockstep also fixes which thread's store wins when threads of one simdgroup
//! store to one location in one instruction, and the order in which a simdgroup runs the arms of
//! a branch its threads diverge on; neither is visible in the IR, and Metal defines neither.
//!
//! The analysis is metered: work per instruction, per block reached, and per pair, and its tables
//! are charged while they live.

mod facts;

use std::collections::{BTreeMap, BTreeSet};

use crate::air::abi::Stage as AbiStage;
use crate::air::intrinsics::{AirIntrinsic, BarrierName, MemFlags, MemoryOrder, Sync, ThreadScope};
use crate::air::ir::{
    Block, BlockId, Body, Call, CallConv, Callee, Const, Constant, FastMath, FnAttrs, FuncId,
    Function, Idx, Inst, InstId, InstKind, Linkage, Module, ModuleParts, Param, TailKind, Type,
    TypeLayouts, Value,
};
use crate::air::{Entry, Lockstep};
use crate::budget::{tree_bytes, Meter};
use crate::device::requirements::VkProperty;
use crate::device::subgroup::{tier, OpBits, SimdClass, Tier};
use crate::device::DeviceProfile;
use crate::error::{Error, InternalError, InternalKind};
use crate::graph::{LoopForest, PostDomTree};
use crate::pipeline::sym::Stage;

use super::uniform::{Among, Uniformity};
use facts::{Access, Accesses, Addr, AddrFacts, Row, Space, Spread};

fn invariant(what: &'static str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, what))
}

/// The barrier one position of a block needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Order {
    /// `threadgroup_barrier(mem_threadgroup)`.
    Threadgroup,
    /// `simdgroup_barrier(mem_threadgroup)`, exact where it stands.
    Simdgroup,
    /// `simdgroup_barrier(mem_threadgroup)`, exact on emulated simdgroups only.
    Emulated,
}

/// The positions of one block where a barrier orders a pair: before the instruction at index
/// `lo..=hi`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Want {
    block: u32,
    order: Order,
    hi: u32,
    lo: u32,
}

/// The ways a block reaches another on barrier-free paths.
mod via {
    /// A path that crosses no retreating edge.
    pub const SAME: u8 = 1;
    /// A path that crosses a retreating edge (a later iteration of a loop).
    pub const CROSS: u8 = 2;
}

/// The facts the placement reads.
struct Shape<'a> {
    e: &'a Entry,
    acc: &'a Accesses,
    /// Uniformity among the threads of a workgroup, and of a simdgroup.
    u: &'a Uniformity,
    us: &'a Uniformity,
    pdom: &'a PostDomTree<BlockId>,
    loops: &'a LoopForest<BlockId>,
}

impl Shape<'_> {
    /// Every thread of a simdgroup executes block `b` together.
    fn sg(&self, b: BlockId) -> bool {
        self.wg(b) || self.us.is_uniform(b)
    }

    /// Every thread of the threadgroup executes block `b` together.
    fn wg(&self, b: BlockId) -> bool {
        self.u.is_uniform(b)
    }

    fn first(&self, b: BlockId) -> u32 {
        self.acc.first.get(b.index()).copied().unwrap_or(0)
    }

    fn term(&self, b: BlockId) -> u32 {
        self.acc.term.get(b.index()).copied().unwrap_or(0)
    }

    /// The interval of the pair `a` then `b`, with `b` reached as `flags` say, and whether it
    /// was reached inside `a`'s block (`inb`): after `a`, and before `b` when `b` follows in the
    /// block.
    fn after(&self, a: &Access, b: &Access, inb: bool) -> (u32, u32) {
        let hi = if inb { b.pos } else { self.term(a.block) };
        (a.pos.saturating_add(1), hi)
    }

    /// The interval before `b`.
    fn before(&self, a: &Access, b: &Access, inb: bool) -> (u32, u32) {
        let lo = if inb {
            a.pos.saturating_add(1)
        } else {
            self.first(b.block)
        };
        (lo, b.pos)
    }

    /// Where a barrier orders the conflicting pair `a` then `b`.
    fn place(
        &self,
        (a, b): (&Access, &Access),
        inb: bool,
        avoid: &mut Avoid,
        m: &mut Meter,
    ) -> Result<Want, Error> {
        let (x, y) = (a.block, b.block);
        let want = |block: BlockId, order, (lo, hi): (u32, u32)| Want {
            block: block.raw(),
            order,
            hi,
            lo,
        };
        if self.wg(x) {
            return Ok(want(x, Order::Threadgroup, self.after(a, b, inb)));
        }
        if self.wg(y) {
            return Ok(want(y, Order::Threadgroup, self.before(a, b, inb)));
        }
        // A block every path from `a` to `b` crosses: it post-dominates `a`'s block, dominates
        // `b`'s, and does not dominate `a`'s (so a path reaching `a` without it must take it to
        // reach `b`).
        let mut z = self.pdom.ipdom(x);
        let mut steps = 0u32;
        let mut simd_z = None;
        while let Some(zb) = z {
            m.tick(1)?;
            steps += 1;
            if zb == y || steps > 64 {
                break;
            }
            if self.e.dom.dominates(zb, y) && !self.e.dom.dominates(zb, x) {
                if self.wg(zb) {
                    let at = (self.first(zb), self.term(zb));
                    return Ok(want(zb, Order::Threadgroup, at));
                }
                if self.sg(zb) && simd_z.is_none() {
                    simd_z = Some(zb);
                }
            }
            z = self.pdom.ipdom(zb);
        }
        // The same positions where every thread of the simdgroup runs the barrier together.
        if self.sg(x) {
            return Ok(want(x, Order::Simdgroup, self.after(a, b, inb)));
        }
        if self.sg(y) {
            return Ok(want(y, Order::Simdgroup, self.before(a, b, inb)));
        }
        if let Some(zb) = simd_z {
            let at = (self.first(zb), self.term(zb));
            return Ok(want(zb, Order::Simdgroup, at));
        }
        // Across a back edge: the header of a loop holding both that every barrier-free path
        // from `a` to `b` enters (a later iteration starts there).
        if !inb {
            let mut l = self.loops.innermost(x);
            while let Some(lp) = l {
                m.tick(1)?;
                l = self.loops.parent(lp);
                let h = self.loops.header(lp);
                if h == x || h == y || !self.loops.contains(lp, y) || !(self.wg(h) || self.sg(h)) {
                    continue;
                }
                if !avoid.reaches(self, (x, h), y, m)? {
                    let order = if self.wg(h) {
                        Order::Threadgroup
                    } else {
                        Order::Simdgroup
                    };
                    return Ok(want(h, order, (self.first(h), self.term(h))));
                }
            }
        }
        Ok(want(y, Order::Emulated, self.before(a, b, inb)))
    }
}

impl Shape<'_> {
    /// Whether `a` then `b` may touch one byte from two threads of one simdgroup in an order
    /// that matters, `b` reached across a back edge when `cross`.
    fn conflict(&self, (a, fa): (&Access, &Addr), (b, fb): (&Access, &Addr), cross: bool) -> bool {
        if !(a.write || b.write) {
            return false;
        }
        if a.update.is_some() && a.update == b.update && a.size == b.size {
            return false;
        }
        if let (Some(ra), Some(rb)) = (&fa.roots, &fb.roots) {
            if !ra.iter().any(|r| rb.contains(r)) {
                return false;
            }
        }
        if let (Some(pa), Some(pb), Some(sa), Some(sb)) = (fa.span, fb.span, a.size, b.size) {
            // Every byte one may touch lies before every byte the other may.
            let before = |hi: i64, s: u64, lo: i64| {
                i64::try_from(s)
                    .ok()
                    .and_then(|s| hi.checked_add(s))
                    .is_some_and(|end| end <= lo)
            };
            if pa.root == pb.root && (before(pa.hi, sa, pb.lo) || before(pb.hi, sb, pa.lo)) {
                return false;
            }
        }
        // Across iterations a value must be one value in both: computed outside every loop
        // holding either access.
        let invariant = |h: BlockId| {
            [a.block, b.block].iter().all(|&blk| {
                let mut l = self.loops.innermost(blk);
                while let Some(lp) = l {
                    if self.loops.contains(lp, h) {
                        return false;
                    }
                    l = self.loops.parent(lp);
                }
                true
            })
        };
        let settled = |homes: &[BlockId]| !cross || homes.iter().all(|&h| invariant(h));
        // One plain store against itself in a later iteration, at an address that keeps one value
        // in each thread through the loops holding it, in a block every thread of the simdgroup
        // runs as often: whatever order the threads' iterations take, the last store to a location
        // is one thread's last there, as under lockstep (which thread's is lockstep's residue of
        // one instruction), and a read of it between is a pair of its own.
        if cross
            && a.at == b.at
            && a.stored.is_some()
            && !a.read
            && self.sg(a.block)
            && fa
                .leaves
                .as_ref()
                .is_some_and(|l| l.iter().all(|&h| invariant(h)))
        {
            return false;
        }
        // One thread of a simdgroup at most runs either access: each runs only where one value
        // that differs between the simdgroup's threads equals one value they share, both
        // computed outside every loop holding the pair, so each thread holds one value of each
        // through every dynamic instance of the pair, across iterations too.
        if fa
            .lone
            .iter()
            .any(|l| fb.lone.contains(l) && l.homes.iter().flatten().all(|&h| invariant(h)))
        {
            return false;
        }
        if let (Some(xa), Some(xb), Some(sa), Some(sb)) = (fa.anchor, fb.anchor, a.size, b.size) {
            if xa.id == xb.id
                && apart((xa.spread, xb.spread), (xa.off, sa), (xb.off, sb))
                && settled(&fa.homes)
                && settled(&fb.homes)
            {
                return false;
            }
            // One group, one core at least a uniform `u` (a sum with it that does not wrap)
            // and the other below `u` where its access runs, both in the order the cores'
            // extension gives them: the first core exceeds the second in every pair of
            // threads, by at least one step of `stride` bytes.
            if let (Some(ga), Some(gb)) = (xa.group, xb.group) {
                let above = |fh: &Addr, kl, fl: &Addr| {
                    fh.sum.iter().any(|&(u, home)| {
                        self.us.is_uniform_value(self.e, u)
                            && fl.below.contains(&(kl, u, ga.signed))
                            && (!cross || home.is_none_or(invariant))
                    })
                };
                let past = |o1: i64, o2: i64, s2: u64| {
                    i128::from(o1) + i128::from(ga.stride) >= i128::from(o2) + i128::from(s2)
                };
                if ga.id == gb.id
                    && ga.signed == gb.signed
                    && settled(&fa.group_homes)
                    && settled(&fb.group_homes)
                    && ((above(fa, gb.core, fb) && past(xa.off, xb.off, sb))
                        || (above(fb, ga.core, fa) && past(xb.off, xa.off, sa)))
                {
                    return false;
                }
            }
        }
        if let (Some(ra), Some(rb), Some(sa), Some(sb)) = (&fa.row, &fb.row, a.size, b.size) {
            // Offsets that stay inside the array: differing multiples of its size put the
            // bytes in different arrays, so only the offsets into it can meet.
            let inside = |r: &Row, s: u64| {
                r.lo >= 0 && i128::from(r.hi) + i128::from(s) <= i128::from(r.size)
            };
            if ra.id == rb.id
                && inside(ra, sa)
                && inside(rb, sb)
                && apart((ra.spread, rb.spread), (ra.off, sa), (rb.off, sb))
                && settled(&ra.homes)
                && settled(&rb.homes)
            {
                return false;
            }
        }
        !self.one_value((a, fa), b, cross, &invariant)
    }

    /// Whether the order of `a` then `b` cannot change what memory holds or what `b` reads,
    /// because `a` stores a value every thread of the simdgroup stores alike (uniform among
    /// them, and across a back edge computed outside the loops holding the pair):
    ///
    /// - `b` stores the same value with the same power-of-two size, both at addresses aligned
    ///   to it, so their bytes are one location or apart;
    /// - or `a` stores it at one address for the whole simdgroup (and across a back edge, on
    ///   every iteration), and `b` reads after `a` in every thread that reaches it (`a`
    ///   dominates `b`): the bytes `b` reads there hold the value whichever thread's store
    ///   landed last, its own included.
    fn one_value(
        &self,
        (a, fa): (&Access, &Addr),
        b: &Access,
        cross: bool,
        invariant: &dyn Fn(BlockId) -> bool,
    ) -> bool {
        let Some((va, align_a)) = a.stored else {
            return false;
        };
        if !self.us.is_uniform_value(self.e, va)
            || (cross && fa.stored_home.is_some_and(|h| !invariant(h)))
        {
            return false;
        }
        if let Some((vb, align_b)) = b.stored {
            return a
                .size
                .filter(|&s| s.is_power_of_two() && b.size == Some(s))
                .is_some_and(|s| va == vb && align_a >= s && align_b >= s);
        }
        if b.write || b.ptr.is_none() {
            return false;
        }
        let one_address = fa.span.is_some_and(|p| p.lo == p.hi)
            || (fa.anchor.is_some_and(|x| x.spread == Some(Spread::Same))
                && (!cross || fa.homes.iter().all(|&h| invariant(h))));
        let first = if a.block == b.block {
            a.pos < b.pos
        } else {
            self.e.dom.strictly_dominates(a.block, b.block)
        };
        one_address && first
    }
}

/// Whether the bytes `[ca, ca + sa)` and `[cb, cb + sb)` from two threads' anchors are
/// disjoint for every difference `d` between the anchors their spreads allow: when
/// `d >= ca + sa - cb` or `d <= ca - cb - sb`.
fn apart(
    (xa, xb): (Option<Spread>, Option<Spread>),
    (ca, sa): (i64, u64),
    (cb, sb): (i64, u64),
) -> bool {
    let (ca, cb) = (i128::from(ca), i128::from(cb));
    let (sa, sb) = (i128::from(sa), i128::from(sb));
    let (up, down) = (ca + sa - cb, ca - cb - sb);
    match (xa, xb) {
        (Some(Spread::Same), Some(Spread::Same)) => 0 >= up || 0 <= down,
        (Some(Spread::Apart { min, max, same }), Some(t)) if xa == Some(t) => {
            let (lo, hi) = (i128::from(min), i128::from(max));
            (lo >= up || hi <= down)
                && (-lo <= down || -hi >= up)
                && (!same || 0 >= up || 0 <= down)
        }
        _ => false,
    }
}

/// The blocks reachable from the end of each block on barrier-free paths, with how
/// ([`via`]), computed on demand and kept for the blocks whose accesses reach their end; or,
/// walked back (`back`), the blocks whose end reaches the start of each block so.
struct Reach {
    /// By block: the reached blocks, or `None` before it is computed.
    from_end: Vec<Option<Vec<(BlockId, u8)>>>,
    /// Scratch: by block, the states reached in the current walk.
    seen: Vec<u8>,
    /// The walk follows edges back, from a block's start to the blocks whose end reaches it.
    back: bool,
}

impl Reach {
    fn new(nb: usize, back: bool, m: &mut Meter) -> Result<Reach, Error> {
        let mut from_end = Vec::new();
        m.reserve(&mut from_end, nb)?;
        from_end.resize(nb, None);
        let mut seen = Vec::new();
        m.reserve(&mut seen, nb)?;
        seen.resize(nb, 0u8);
        Ok(Reach {
            from_end,
            seen,
            back,
        })
    }

    /// The blocks reached from the end of `x`; walked back, the blocks whose end reaches the
    /// start of `x`.
    fn of(&mut self, x: BlockId, s: &Shape<'_>, m: &mut Meter) -> Result<&[(BlockId, u8)], Error> {
        let computed = self.from_end.get(x.index()).is_some_and(Option::is_some);
        if !computed {
            let list = self.walk(x, s, m)?;
            if let Some(slot) = self.from_end.get_mut(x.index()) {
                *slot = Some(list);
            }
        }
        Ok(self
            .from_end
            .get(x.index())
            .and_then(Option::as_deref)
            .unwrap_or(&[]))
    }

    fn walk(
        &mut self,
        x: BlockId,
        s: &Shape<'_>,
        m: &mut Meter,
    ) -> Result<Vec<(BlockId, u8)>, Error> {
        let cfg = &s.e.cfg;
        let back = self.back;
        // A state is a block and whether the path to it crossed a retreating edge ([`via`]).
        let flag = |cross: bool| if cross { via::CROSS } else { via::SAME };
        let mut stack: Vec<(BlockId, bool)> = Vec::new();
        let mut touched: Vec<BlockId> = Vec::new();
        let degree = |at: BlockId| {
            if back {
                cfg.predecessors(at).len()
            } else {
                cfg.successors(at).len()
            }
        };
        let edges = |at: BlockId, cross: bool, stack: &mut Vec<_>| {
            if back {
                for from in cfg.predecessors(at) {
                    stack.push((from, cross || cfg.is_retreating(from, at)));
                }
            } else {
                for to in cfg.successors(at) {
                    stack.push((to, cross || cfg.is_retreating(at, to)));
                }
            }
        };
        m.grow(&mut stack, degree(x))?;
        edges(x, false, &mut stack);
        while let Some((y, cross)) = stack.pop() {
            m.tick(1)?;
            let Some(seen) = self.seen.get_mut(y.index()) else {
                continue;
            };
            let f = flag(cross);
            if *seen & f != 0 {
                continue;
            }
            if *seen == 0 {
                m.grow(&mut touched, 1)?;
                touched.push(y);
            }
            *seen |= f;
            if !s.acc.has_barrier(y) {
                m.grow(&mut stack, degree(y))?;
                edges(y, cross, &mut stack);
            }
        }
        let mut out = Vec::new();
        m.reserve(&mut out, touched.len())?;
        for y in touched {
            if let Some(seen) = self.seen.get_mut(y.index()) {
                out.push((y, *seen));
                *seen = 0;
            }
        }
        m.release(crate::budget::vec_bytes(&stack));
        Ok(out)
    }
}

/// By a block and a block to avoid: the blocks reached from the end of the first on
/// barrier-free paths that never enter the second, computed on demand and kept.
struct Avoid {
    memo: BTreeMap<(u32, u32), Vec<BlockId>>,
}

impl Avoid {
    /// Whether `y` is reached from the end of `x` on a barrier-free path that avoids `h`.
    fn reaches(
        &mut self,
        s: &Shape<'_>,
        (x, h): (BlockId, BlockId),
        y: BlockId,
        m: &mut Meter,
    ) -> Result<bool, Error> {
        let key = (x.raw(), h.raw());
        if let std::collections::btree_map::Entry::Vacant(slot) = self.memo.entry(key) {
            let cfg = &s.e.cfg;
            let mut seen: BTreeSet<BlockId> = BTreeSet::new();
            let mut stack: Vec<BlockId> = Vec::new();
            m.grow(&mut stack, cfg.successors(x).len())?;
            stack.extend(cfg.successors(x).filter(|&t| t != h));
            while let Some(z) = stack.pop() {
                m.tick(1)?;
                if !seen.insert(z) {
                    continue;
                }
                m.charge(tree_bytes(1, std::mem::size_of::<BlockId>()))?;
                if !s.acc.has_barrier(z) {
                    m.grow(&mut stack, cfg.successors(z).len())?;
                    stack.extend(cfg.successors(z).filter(|&t| t != h));
                }
            }
            m.release(crate::budget::vec_bytes(&stack));
            m.charge(tree_bytes(
                1,
                std::mem::size_of::<((u32, u32), Vec<BlockId>)>(),
            ))?;
            slot.insert(seen.into_iter().collect());
        }
        Ok(self
            .memo
            .get(&key)
            .is_some_and(|list| list.binary_search(&y).is_ok()))
    }
}

/// What the analysis decided: the barriers to insert, by block and position, whether one needs
/// emulated simdgroups, and how many places the device's lockstep orders instead.
struct Decision {
    inserts: Vec<(BlockId, u32, Order)>,
    /// The memory the inserted barriers order: the spaces whose pairs needed one.
    flags: MemFlags,
    emulated: bool,
    device: u32,
}

/// Order the threadgroup-memory and device-memory accesses of kernel `e` that only Apple's
/// simdgroup lockstep orders, by inserting barriers into its body (see the module documentation), for a device of
/// profile `p`. An entry of another stage, or a kernel without such a pair, is returned
/// unchanged.
pub fn order(mut e: Entry, p: &DeviceProfile, m: &mut Meter) -> Result<Entry, Error> {
    if !matches!(e.ep.stage, AbiStage::Kernel { .. }) {
        return Ok(e);
    }
    let device = p
        .fields()
        .enabled
        .properties
        .contains(&VkProperty::SimdgroupLockstep);
    // A subgroup barrier where the simdgroup's threads diverge orders the threads that run it
    // together (the tangle) where the entry reconverges maximally on 32-lane subgroups, so no place
    // needs the `Emulated` tier there.
    let model = &p.fields().subgroup;
    let tangle = model.maximal_reconvergence
        && matches!(
            tier(&SimdClass::CrossLane(OpBits::BASIC), Stage::Kernel, model),
            Ok(Tier::Direct | Tier::Cluster32)
        );
    let before = m.ledger();
    let decided = decide(&e, (device, tangle), m);
    let spent = usize::try_from(m.ledger().saturating_sub(before)).unwrap_or(usize::MAX);
    m.release(spent);
    let Some(d) = decided? else {
        return Ok(e);
    };
    let barriers = if d.inserts.is_empty() {
        Vec::new()
    } else {
        let mut parts = e.module.into_parts();
        let barriers = insert(&mut parts, e.func, (&d.inserts, d.flags), m)?;
        e.module = Module::new(parts, m)?;
        barriers
    };
    e.lockstep = Lockstep {
        barriers,
        emulated: d.emulated,
        device: d.device,
    };
    Ok(e)
}

/// The barriers kernel `e` needs, or `None` when it needs none; `device`: the device runs a
/// simdgroup in lockstep; `tangle`: a subgroup barrier in divergent flow orders the threads that
/// run it together (maximal reconvergence on 32-lane subgroups).
fn decide(
    e: &Entry,
    (device, tangle): (bool, bool),
    m: &mut Meter,
) -> Result<Option<Decision>, Error> {
    let Some(body) = e.module.func(e.func).and_then(|f| f.body.as_ref()) else {
        return Ok(None);
    };
    let layouts = TypeLayouts::new(&e.layout, e.module.types(), m)?;
    let mut spaces = Vec::new();
    for space in [Space::Threadgroup, Space::Device] {
        m.tick(1)?;
        let acc = Accesses::of(e, body, &layouts, space, m)?;
        if acc.writes {
            spaces.push((space, acc));
        }
    }
    if spaces.is_empty() {
        return Ok(None);
    }
    let loops = LoopForest::compute(&e.cfg, m)?;
    let u = Uniformity::under_barriers(e, body, Among::Workgroup, &loops, m)?;
    let us = Uniformity::under_barriers(e, body, Among::Simdgroup, &loops, m)?;
    let pdom = PostDomTree::compute(&e.cfg, m)?;
    let mut af = AddrFacts::new(e, body, &layouts, &us, m)?;
    let mut wants: BTreeSet<Want> = BTreeSet::new();
    let mut flags = MemFlags::default();
    for (space, acc) in &spaces {
        let mut found = BTreeSet::new();
        pairs(e, acc, (&u, &us), (&pdom, &loops), &mut af, &mut found, m)?;
        if found.is_empty() {
            continue;
        }
        // Every inserted barrier orders each space that needed one, so one barrier serves the
        // intervals of both.
        match space {
            Space::Threadgroup => flags.threadgroup = true,
            Space::Device => flags.device = true,
        }
        wants.extend(found);
    }
    if wants.is_empty() {
        return Ok(None);
    }
    let term = spaces
        .first()
        .map_or(&[][..], |(_, acc)| acc.term.as_slice());
    Ok(Some(stab(&wants, flags, (device, tangle), term, m)?))
}

/// Collect into `wants` where barriers must order the conflicting pairs of `acc`, the accesses of
/// one memory space: each pair is searched within that space's barrier intervals.
fn pairs<'a>(
    e: &'a Entry,
    acc: &'a Accesses,
    (u, us): (&'a Uniformity, &'a Uniformity),
    (pdom, loops): (&'a PostDomTree<BlockId>, &'a LoopForest<BlockId>),
    af: &mut AddrFacts<'a>,
    wants: &mut BTreeSet<Want>,
    m: &mut Meter,
) -> Result<(), Error> {
    let mut addrs: Vec<Addr> = Vec::new();
    m.reserve(&mut addrs, acc.list.len())?;
    for a in &acc.list {
        let addr = af.of(a, m)?;
        m.charge(addr.roots.as_ref().map_or(0, crate::budget::vec_bytes))?;
        addrs.push(addr);
    }
    let shape = Shape {
        e,
        acc,
        u,
        us,
        pdom,
        loops,
    };
    let mut reach = Reach::new(acc.range.len(), false, m)?;
    let mut back = Reach::new(acc.range.len(), true, m)?;
    let mut avoid = Avoid {
        memo: BTreeMap::new(),
    };
    let want_bytes = tree_bytes(1, std::mem::size_of::<Want>());
    let empty = Addr::default();
    let mut consider = |(ai, bi): (usize, usize), inb: bool, flags: u8, m: &mut Meter| {
        m.tick(1)?;
        let (Some(a), Some(b)) = (acc.list.get(ai), acc.list.get(bi)) else {
            return Ok::<(), Error>(());
        };
        let (fa, fb) = (
            addrs.get(ai).unwrap_or(&empty),
            addrs.get(bi).unwrap_or(&empty),
        );
        if !shape.conflict((a, fa), (b, fb), flags & via::CROSS != 0) {
            return Ok(());
        }
        let w = shape.place((a, b), inb, &mut avoid, m)?;
        if wants.insert(w) {
            m.charge(want_bytes)?;
        }
        Ok(())
    };
    // Two reads never conflict, so a pair holds a write: a write's pairs are searched from it
    // forward, and a read's from the writes it reaches, walked back from their blocks. A body of
    // many reads and few writes costs walks from the writes' blocks alone, linear in the body
    // (the contract, which the device pass's reads of every buffer access keep).
    let mut writes: Vec<u32> = Vec::new();
    for (i, a) in acc.list.iter().enumerate() {
        if a.write {
            m.grow(&mut writes, 1)?;
            writes.push(u32::try_from(i).unwrap_or(u32::MAX));
        }
    }
    let writes_of = |b: BlockId| {
        let (s, e) = acc.range.get(b.index()).copied().unwrap_or((0, 0));
        let lo = writes.partition_point(|&i| i < s);
        let hi = writes.partition_point(|&i| i < e);
        writes.get(lo..hi).unwrap_or(&[])
    };
    for (ai, a) in acc.list.iter().enumerate() {
        let next = acc.barrier_after(a.block, a.pos);
        let escapes = next.is_none();
        let start = acc.range.get(a.block.index()).map_or(0, |r| r.0 as usize);
        let first_bar = acc.first_barrier(a.block);
        // Accesses of `a`'s block: later ones before its next barrier, and those reached around
        // a loop into the block's start (`around`: how the block's end reaches its start).
        let in_block = |b: &Access, self_flags: u8| {
            let inb = b.pos > a.pos && next.is_none_or(|p| b.pos < p);
            let around = self_flags != 0 && first_bar.is_none_or(|p| b.pos < p);
            (inb || around).then_some((inb, if around { self_flags } else { 0 }))
        };
        if a.write {
            let reached: Vec<(BlockId, u8)> = if escapes {
                reach.of(a.block, &shape, m)?.to_vec()
            } else {
                Vec::new()
            };
            m.charge(crate::budget::vec_bytes(&reached))?;
            let self_flags = reached
                .iter()
                .find(|&&(y, _)| y == a.block)
                .map_or(0, |&(_, f)| f);
            for (k, b) in acc.in_block(a.block).iter().enumerate() {
                if let Some((inb, flags)) = in_block(b, self_flags) {
                    consider((ai, start + k), inb, flags, m)?;
                }
            }
            for &(y, flags) in &reached {
                if y == a.block {
                    continue;
                }
                let start = acc.range.get(y.index()).map_or(0, |r| r.0 as usize);
                let first_bar = acc.first_barrier(y);
                for (k, b) in acc.in_block(y).iter().enumerate() {
                    if first_bar.is_some_and(|p| b.pos >= p) {
                        break;
                    }
                    consider((ai, start + k), false, flags, m)?;
                }
            }
        } else {
            // A read: the writes of its block; those of other blocks come from the walks back.
            let ws = writes_of(a.block);
            if ws.is_empty() {
                continue;
            }
            let self_flags = if escapes {
                back.of(a.block, &shape, m)?
                    .iter()
                    .find(|&&(x, _)| x == a.block)
                    .map_or(0, |&(_, f)| f)
            } else {
                0
            };
            for &bi in ws {
                let Some(b) = acc.list.get(bi as usize) else {
                    continue;
                };
                if let Some((inb, flags)) = in_block(b, self_flags) {
                    consider((ai, bi as usize), inb, flags, m)?;
                }
            }
        }
    }
    // A read in another block than a write it reaches: from each block's writes before its first
    // barrier, back to the blocks whose end reaches its start, and their reads after their last.
    let mut done: Option<BlockId> = None;
    for &wi in &writes {
        let Some(y) = acc.list.get(wi as usize).map(|w| w.block) else {
            continue;
        };
        if done == Some(y) {
            continue;
        }
        done = Some(y);
        let first_bar = acc.first_barrier(y);
        let ws: Vec<usize> = writes_of(y)
            .iter()
            .map(|&i| i as usize)
            .filter(|&i| {
                acc.list
                    .get(i)
                    .is_some_and(|w| first_bar.is_none_or(|p| w.pos < p))
            })
            .collect();
        m.charge(crate::budget::vec_bytes(&ws))?;
        if ws.is_empty() {
            continue;
        }
        let from = back.of(y, &shape, m)?.to_vec();
        m.charge(crate::budget::vec_bytes(&from))?;
        for &(x, flags) in &from {
            if x == y {
                continue;
            }
            let start = acc.range.get(x.index()).map_or(0, |r| r.0 as usize);
            let last_bar = acc.barriers.get(x.index()).and_then(|v| v.last().copied());
            for (k, a) in acc.in_block(x).iter().enumerate() {
                if a.write || last_bar.is_some_and(|p| a.pos < p) {
                    continue;
                }
                for &bi in &ws {
                    consider((start + k, bi), false, flags, m)?;
                }
            }
        }
    }
    Ok(())
}

/// One barrier at a position of as many intervals as it can serve ([`points`]): threadgroup
/// barriers first, then simdgroup barriers for the intervals no threadgroup barrier serves. On a
/// device that runs a simdgroup in lockstep (`device`), the places only emulated simdgroups
/// would make exact are counted rather than given a barrier; where a subgroup barrier orders the
/// threads that run it together (`tangle`), they take a simdgroup barrier and leave the tier to
/// the kernel's class.
fn stab(
    wants: &BTreeSet<Want>,
    flags: MemFlags,
    (device, tangle): (bool, bool),
    term: &[u32],
    m: &mut Meter,
) -> Result<Decision, Error> {
    let mut placed: BTreeSet<(u32, u32, Order)> = BTreeSet::new();
    let bytes = tree_bytes(1, std::mem::size_of::<(u32, u32, Order)>());
    let threadgroup: Vec<Want> = wants
        .iter()
        .filter(|w| w.order == Order::Threadgroup)
        .copied()
        .collect();
    m.charge(crate::budget::vec_bytes(&threadgroup))?;
    for (w, at) in points(threadgroup, term, m)? {
        m.charge(bytes)?;
        placed.insert((w.block, at, Order::Threadgroup));
    }
    let served = |placed: &BTreeSet<(u32, u32, Order)>, w: &Want| {
        placed
            .range((w.block, w.lo, Order::Threadgroup)..=(w.block, w.hi, Order::Threadgroup))
            .any(|&(_, _, o)| o == Order::Threadgroup)
    };
    let rest: Vec<Want> = wants
        .iter()
        .filter(|w| w.order != Order::Threadgroup && !served(&placed, w))
        .copied()
        .collect();
    m.charge(crate::budget::vec_bytes(&rest))?;
    let emulated = !device && !tangle && rest.iter().any(|w| w.order == Order::Emulated);
    for (w, at) in points(rest, term, m)? {
        m.charge(bytes)?;
        // A block holds places of one kind: a simdgroup barrier's are uniform in the
        // simdgroup, and an emulated one's are not.
        let order = if device && w.order == Order::Emulated {
            Order::Emulated
        } else {
            Order::Simdgroup
        };
        placed.insert((w.block, at, order));
    }
    let device = placed
        .iter()
        .filter(|&&(_, _, o)| o == Order::Emulated)
        .count();
    let mut inserts = Vec::new();
    m.reserve(&mut inserts, placed.len() - device)?;
    inserts.extend(
        placed
            .into_iter()
            .filter(|&(_, _, o)| o != Order::Emulated)
            .map(|(b, p, o)| (BlockId::new(b), p, o)),
    );
    Ok(Decision {
        inserts,
        flags,
        emulated,
        device: u32::try_from(device).unwrap_or(u32::MAX),
    })
}

/// The fewest positions that stab every interval of `list`, each with the interval that placed
/// it: the latest position of an interval that ends before its block's terminator (right before
/// the access that ends it), and for the intervals left that run to the terminator (`term`, by
/// block), the earliest position they all share (right after the last access that starts one,
/// or the block's start), so no barrier stands in trailing code its pairs do not reach past (an
/// inlined function's copy, which rolling compares).
fn points(mut list: Vec<Want>, term: &[u32], m: &mut Meter) -> Result<Vec<(Want, u32)>, Error> {
    list.sort_unstable_by_key(|w| (w.block, w.hi, w.lo));
    let mut out: Vec<(Want, u32)> = Vec::new();
    let mut last: Option<(u32, u32)> = None;
    let mut i = 0;
    while let Some(&w) = list.get(i) {
        m.tick(1)?;
        let served = |w: &Want, last: Option<(u32, u32)>| {
            last.is_some_and(|(b, p)| b == w.block && p >= w.lo)
        };
        if served(&w, last) {
            i += 1;
            continue;
        }
        m.grow(&mut out, 1)?;
        if term.get(w.block as usize) == Some(&w.hi) {
            // Every interval left in the block runs to its end (they are sorted by end): one
            // barrier after the last start of those no earlier barrier serves serves them all.
            let left = list.get(i..).unwrap_or(&[]);
            let n = left.iter().take_while(|x| x.block == w.block).count();
            let at = left
                .iter()
                .take(n)
                .filter(|x| !served(x, last))
                .map(|x| x.lo)
                .max()
                .map_or(w.lo, |lo| lo.min(w.hi));
            out.push((w, at));
            last = Some((w.block, at));
            i += n;
        } else {
            out.push((w, w.hi));
            last = Some((w.block, w.hi));
            i += 1;
        }
    }
    Ok(out)
}

/// A barrier declaration the inserted calls use: its function, and whether it is the AIR 2.9 form
/// `void (i32 flags, i32 order, i32 scope)` rather than AIR 2.8's `void (i32 flags, i32 scope)`.
#[derive(Clone, Copy)]
struct BarrierDecl {
    func: FuncId,
    ordered: bool,
}

/// The declaration of barrier `name` in `parts`: the module's own, in either form, or an AIR 2.8
/// declaration added when the module has none.
fn declaration(parts: &mut ModuleParts, name: &[u8], m: &mut Meter) -> Result<BarrierDecl, Error> {
    let void = parts.types.intern(Type::Void, m)?;
    let i32t = parts.types.intern(Type::Int(32), m)?;
    if let Some(sym) = parts.symbols.lookup(name) {
        if let Some((id, f)) = parts.funcs.iter().find(|(_, f)| f.name == sym) {
            let ordered = match parts.types.get(f.ty) {
                Some(Type::Function {
                    ret,
                    params,
                    vararg: false,
                }) if *ret == void && f.body.is_none() && params.iter().all(|&p| p == i32t) => {
                    match params.len() {
                        2 => Some(false),
                        3 => Some(true),
                        _ => None,
                    }
                }
                _ => None,
            };
            return ordered
                .map(|ordered| BarrierDecl { func: id, ordered })
                .ok_or_else(|| invariant("a barrier declaration of another type"));
        }
    }
    let ty = parts.types.intern(
        Type::Function {
            ret: void,
            params: Box::new([i32t, i32t]),
            vararg: false,
        },
        m,
    )?;
    let name = parts.symbols.intern(name, m)?;
    let mut params = Vec::new();
    m.reserve(&mut params, 2)?;
    params.resize(2, Param::default());
    let f = Function {
        name,
        ty,
        linkage: Linkage::External,
        cconv: CallConv::C,
        attrs: FnAttrs {
            convergent: true,
            ..FnAttrs::default()
        },
        ret_attrs: Default::default(),
        params: params.into_boxed_slice(),
        body: None,
        section: None,
        align: None,
        visibility: Default::default(),
        unnamed_addr: Default::default(),
    };
    Ok(BarrierDecl {
        func: parts.funcs.push(f, m)?,
        ordered: false,
    })
}

/// The `i32` constant `v`.
fn int32(parts: &mut ModuleParts, v: u64, m: &mut Meter) -> Result<Value, Error> {
    let ty = parts.types.intern(Type::Int(32), m)?;
    Ok(Value::Const(parts.consts.intern(
        Constant {
            ty,
            value: Const::Int(v),
        },
        m,
    )?))
}

/// Insert the barriers `inserts` (sorted by block and position) into `func`'s body, each ordering
/// the memory `flags` name. Returns the inserted instructions, sorted.
fn insert(
    parts: &mut ModuleParts,
    func: FuncId,
    (inserts, flags): (&[(BlockId, u32, Order)], MemFlags),
    m: &mut Meter,
) -> Result<Vec<InstId>, Error> {
    let void = parts.types.intern(Type::Void, m)?;
    // `mem_device` (1) and `mem_threadgroup` (2), as `MemFlags` reads the mask.
    let bits = u64::from(flags.device) | (u64::from(flags.threadgroup) << 1);
    let mem = int32(parts, bits, m)?;
    let wg_scope = int32(parts, 1, m)?;
    let sg_scope = int32(parts, 4, m)?;
    let wg = if inserts.iter().any(|&(_, _, o)| o == Order::Threadgroup) {
        Some(declaration(parts, b"air.wg.barrier", m)?)
    } else {
        None
    };
    let sg = if inserts.iter().any(|&(_, _, o)| o != Order::Threadgroup) {
        Some(declaration(parts, b"air.simdgroup.barrier", m)?)
    } else {
        None
    };
    // An AIR 2.9 declaration's order operand: `seq_cst`, the order of every AIR 2.8 barrier.
    let seq_cst = if wg.iter().chain(&sg).any(|d| d.ordered) {
        Some(int32(parts, 5, m)?)
    } else {
        None
    };
    let call = |o: Order| -> Result<Inst, Error> {
        let (decl, name, scope, arg) = match o {
            Order::Threadgroup => (
                wg,
                BarrierName::Threadgroup,
                ThreadScope::Threadgroup,
                wg_scope,
            ),
            Order::Simdgroup | Order::Emulated => {
                (sg, BarrierName::Simdgroup, ThreadScope::Simdgroup, sg_scope)
            }
        };
        let decl = decl.ok_or_else(|| invariant("a barrier without its declaration"))?;
        let (order, args): (_, Box<[Value]>) = match (decl.ordered, seq_cst) {
            (true, Some(o)) => (Some(MemoryOrder::SeqCst), Box::new([mem, o, arg])),
            (true, None) => return Err(invariant("an ordered barrier without its order")),
            (false, _) => (None, Box::new([mem, arg])),
        };
        Ok(Inst {
            ty: void,
            kind: InstKind::Call(Box::new(Call {
                callee: Callee::Air {
                    decl: decl.func,
                    op: AirIntrinsic::Sync(Sync::Barrier {
                        name,
                        flags,
                        order,
                        scope,
                        memory: scope,
                    }),
                },
                args,
                fmf: FastMath::NONE,
                tail: TailKind::None,
                attrs: None,
                range: None,
            })),
        })
    };
    let body = parts
        .funcs
        .get(func)
        .and_then(|f| f.body.as_ref())
        .ok_or_else(|| invariant("the entry has no body"))?;
    let (out, inserted) = rewritten(body, inserts, &call, m)?;
    if let Some(old) = parts
        .funcs
        .get_mut(func)
        .ok_or_else(|| invariant("entry out of range"))?
        .body
        .replace(out)
    {
        // The body the barriers were inserted into is freed.
        old.release(m);
    }
    Ok(inserted)
}

/// `body` with a barrier built by `call` before each instruction `inserts` names.
fn rewritten(
    body: &Body,
    inserts: &[(BlockId, u32, Order)],
    call: &dyn Fn(Order) -> Result<Inst, Error>,
    m: &mut Meter,
) -> Result<(Body, Vec<InstId>), Error> {
    // Each old instruction's id in the output.
    let mut map = Vec::new();
    m.reserve(&mut map, body.insts.len())?;
    let mut shift = 0u32;
    let mut next = inserts.iter().peekable();
    for (b, block) in body.blocks.iter() {
        for (pos, _) in block.inst_ids().enumerate() {
            while next
                .next_if(|&&(ib, ip, _)| ib == b && ip as usize == pos)
                .is_some()
            {
                shift = shift
                    .checked_add(1)
                    .ok_or_else(|| invariant("instruction ids"))?;
            }
            let old = u32::try_from(map.len()).map_err(|_| invariant("instruction ids"))?;
            let new = old
                .checked_add(shift)
                .ok_or_else(|| invariant("instruction ids"))?;
            map.push(InstId::new(new));
        }
    }
    if next.next().is_some() {
        return Err(invariant("a barrier position outside its block"));
    }
    let remap = |v: Value| -> Result<Value, Error> {
        match v {
            Value::Inst(i) => map
                .get(i.index())
                .map(|&n| Value::Inst(n))
                .ok_or_else(|| invariant("an operand out of range")),
            other => Ok(other),
        }
    };
    let mut out = Body::default();
    let mut inserted = Vec::new();
    m.reserve(&mut inserted, inserts.len())?;
    let mut next = inserts.iter().peekable();
    for (b, block) in body.blocks.iter() {
        m.tick(1)?;
        let start = InstId::new(u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?);
        for (pos, i) in block.inst_ids().enumerate() {
            while let Some(&(_, _, o)) = next.next_if(|&&(ib, ip, _)| ib == b && ip as usize == pos)
            {
                inserted.push(out.push_inst(call(o)?, m)?);
            }
            let inst = body
                .insts
                .get(i)
                .ok_or_else(|| invariant("instruction out of range"))?;
            let mut kind = inst.kind.clone();
            let mut failed = None;
            kind.for_each_value_mut(|val| match remap(*val) {
                Ok(n) => *val = n,
                Err(e) => {
                    failed.get_or_insert(e);
                }
            });
            if let Some(e) = failed {
                return Err(e);
            }
            out.push_inst(Inst { ty: inst.ty, kind }, m)?;
        }
        let end = InstId::new(u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?);
        out.blocks.push(
            Block {
                start,
                end,
                name: block.name,
            },
            m,
        )?;
    }
    m.release(crate::budget::vec_bytes(&map));
    Ok((out, inserted))
}
