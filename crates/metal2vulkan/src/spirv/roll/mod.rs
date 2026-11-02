//! Repeated regions of a stage's code written once, as a loop.
//!
//! The AIR inliner (`air::canon::inline`) writes each AIR call's body in line, so a kernel that
//! applies one large function twice holds its body twice, and every driver compiles both copies;
//! a driver's compile time grows with the instructions it compiles. A kernel that applies one
//! large function once per output texture took 7.6 s to create on an M3 Max and 6.2 s on an M4
//! through MoltenVK 1.4.1 (whose SPIRV-Cross also writes every function inlined at each call,
//! `DriverWorkaround::FunctionsInlinedAtEveryCall`), 24.4 to 27.3 s on the RTX 5080 Laptop GPU's
//! driver and 12.4 s on lavapipe, each over the 5-second creation ceiling; written once, about half
//! the time on each.
//!
//! On every profile, when the entry's inlined size ([`entry_weight`]: what a driver that inlines
//! every call compiles, each call counted as its callee's body) is above [`ROLL_ABOVE`], the pass
//! finds a run of `k` (2 to [`MAX_TRIPS`]) consecutive regions of one structured sequence that
//! are the same code, and writes the first once as the body of a loop of `k` trips, deleting the
//! others. A sequence is the top-level code of the entry or of a function it calls (the heaviest
//! [`MAX_FUNCS`]), or an arm of a selection or switch in one, to [`MAX_DEPTH`] constructs deep. The
//! regions are exact: each is compared instruction by instruction with the first, every value one
//! defines mapped to the first's value at the same place, and the loop's trip `t` executes exactly
//! what region `t` executed, in the same order. What differs between regions is carried by the trip
//! index, read from a `Function` variable:
//!
//! - a value defined before the run (a constant, a specialization constant, an earlier result)
//!   in an operand where any value may stand is the trip's own, chosen by `OpSelect` at the top
//!   of the loop body; where every region's is a plain `u32` constant and region `t`'s is
//!   `c + t * d` in wrapping arithmetic (an offset that grows by a stride), it is computed from
//!   the trip index instead, so no select stands for it (a select chain per offset held registers
//!   that cost a native driver occupancy);
//! - a chain, where each region but the first reads, in such an operand, the previous region's
//!   value at one place and the first reads a value defined before the run there, carries the
//!   value in a `Function` variable: stored the first region's operand before the loop, loaded
//!   at the top of every trip, and stored the trip's value at the latch;
//! - a lane read (`OpCompositeExtract`, `OpVectorShuffle`) of values defined before the run whose
//!   lanes or operands differ by region is lifted: each region's read is computed before the
//!   loop, and the trip's is chosen by `OpSelect` at the top of the loop body, where the first
//!   region's read copies it;
//! - integer arithmetic ([`floats`]) on values defined before the run that a region holds where
//!   the first region holds none (an address the first region reads at a base, the others at
//!   the base plus their offset) is computed before the loop, and its value is the region's at
//!   the operand the first region reads the value from before the run at;
//! - a `Function` variable, or a `Private` variable no other function uses, that only its region
//!   uses becomes the first region's, stored its initializer again at the top of every trip (a
//!   variable without one holds an undefined value at the region's start, which the previous
//!   trip's last value is);
//! - an image a region loads (from its own `UniformConstant` variable) and only writes, reads,
//!   fetches, or queries is loaded where it is used, in an `OpSwitch` on the trip index with one
//!   arm per trip, each loading its trip's image;
//! - an operand that must be a constant ([`switched_operand`]: an image instruction's
//!   `ConstOffset`) and is another constant in some region is executed in an `OpSwitch` on the
//!   trip index: the instruction is written once per distinct trip operands, each arm the
//!   region's own instruction with its own constants (with its trip's image, where that differs
//!   too), and its result is the arms' phi;
//! - a value of a region read after the run is the first region's value where it is the last
//!   region's (the loop's last trip leaves it); where it is an earlier region's (of a type a
//!   variable holds), each trip stores the first region's value into a `Function` array at its
//!   trip index, and the loop's merge loads the element of the region read.
//!
//! An `OpCopyObject` is read through: regions compare as if each copy were the value it copies.
//! Anything else refuses the run: a value of one region read by another but the chain's, an
//! operand that must be a constant, any other differing pointer. Nothing is guessed: a run either
//! is exactly the same code or stays as it is. The loop has the builder's canonical shape (an empty
//! header entered from its preheader and its single continue block, the trip count tested in a
//! selection whose arm breaks), and the finalize verifier checks the result like any other
//! module.
//!
//! Candidate runs come from a function's *spine*: the blocks of each of its sequences in layout
//! order, each an item per instruction and one per construct it heads (with the merge's phis).
//! Equal items have equal structural keys (the instruction with its value operands erased, its
//! result type, and a construct's blocks so keyed, but for integer arithmetic on values the
//! sequence does not define, which may float). Heavy constructs that recur anchor a period; the run around a
//! pair of them is extended while keys recur at that period; a few start positions of the
//! heaviest runs are compared exactly. A lane read's key omits its lanes, which a lift lets
//! differ. Work is linear in the entry per comparison, bounded in anchors, pairs, start positions,
//! and comparisons, and metered on the per-attempt [`Meter`].

#[cfg(test)]
mod tests;

use std::collections::BTreeSet;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::ops::Range;

use crate::budget::Meter;
use crate::error::{Error, InternalError, InternalKind};
use crate::spirv::ids::{BlockId, ConstId, FuncId, GlobalId, TypeId, Value, VarId};
use crate::spirv::ir::{
    BinOp, Block, ConstBits, Constant, Contraction, Decl, GlobalKind, Inst, InstNode, LocalVar,
    LoopMeta, MemAccess, Merge, Phi, Shape, StorageClass, Terminator, Type, TypedModule, ValueDef,
    ValueOrigin,
};

/// The entry's inlined size, in instructions ([`entry_weight`]), above which runs are rolled:
/// about what Metal compiles in 2.5 seconds on an M3 Max (the 69a35b6f measurement: about 10
/// instructions a millisecond on M4, 8 on M3 Max). A module below it is written as it was.
pub const ROLL_ABOVE: u64 = 24_576;

/// The smallest region worth a loop, in inlined instructions (1,024, where 2,048 left a run of
/// regions just below it unrolled, or rolled in pairs with bodies twice the size, and NVIDIA's
/// making grows with a loop body's size).
pub const MIN_CHUNK: u64 = 1_024;

/// The most trips a rolled loop takes (one select per trip for each differing value).
pub const MAX_TRIPS: usize = 16;

/// The most runs rolled in one module.
const MAX_LOOPS: u32 = 8;
/// The lightest construct or call that anchors a period.
const ANCHOR_MIN: u64 = 1;
/// The most start positions compared per candidate run.
const MAX_PHASES: usize = 16;
/// The heaviest recurring construct keys tried as anchors per search.
const MAX_ANCHORS: usize = 64;
/// The consecutive occurrences of one anchor key tried as a period.
const MAX_PAIRS: usize = 8;
/// The exact comparisons per search.
const MAX_VERIFIES: usize = 32;
/// [`Item::inst`] of a construct (or the spine's end).
const TAIL: u32 = u32::MAX;
/// The most functions searched per run: the heaviest the entry reaches.
const MAX_FUNCS: usize = 8;
/// The deepest arm searched: a sequence inside this many selections and switches.
const MAX_DEPTH: u32 = 8;

/// The pass's maps and sets of ids, hashed by [`Fnv`]'s word mix rather than SipHash: their keys
/// are the module's own dense ids, and no output depends on their order (each one the pass walks
/// is sorted first, or only read by key).
type HashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<Fnv>>;
type HashSet<K> = std::collections::HashSet<K, BuildHasherDefault<Fnv>>;

fn invariant(detail: impl Into<String>) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

/// Roll `m`'s repeated regions while its entry's inlined size is above [`ROLL_ABOVE`], each
/// region at least [`MIN_CHUNK`], on every profile. Returns the runs written as loops.
pub fn roll_entry(m: &mut TypedModule, meter: &mut Meter) -> Result<u32, Error> {
    roll(m, ROLL_ABOVE, MIN_CHUNK, meter)
}

/// Roll runs of `m`'s entry while its inlined size is above `above`, each region at least
/// `min_chunk` inlined instructions, heaviest saving first. Returns the runs written as loops.
pub fn roll(
    m: &mut TypedModule,
    above: u64,
    min_chunk: u64,
    meter: &mut Meter,
) -> Result<u32, Error> {
    let mut cache = KeyCache::default();
    let r = roll_kept(m, (above, min_chunk), &mut cache, meter);
    meter.release(cache.charged);
    r
}

/// [`roll`] with the instructions' keys kept in `cache` from one search to the next.
fn roll_kept(
    m: &mut TypedModule,
    (above, min_chunk): (u64, u64),
    cache: &mut KeyCache,
    meter: &mut Meter,
) -> Result<u32, Error> {
    let mut weight = entry_weight(m, meter)?;
    let mut loops = 0u32;
    while weight > above && loops < MAX_LOOPS {
        let mut ledger = 0usize;
        let found = find(m, min_chunk, cache, meter, &mut ledger);
        meter.release(ledger);
        let Some(plan) = found? else {
            break;
        };
        apply(m, &plan)?;
        loops += 1;
        weight = entry_weight(m, meter)?;
    }
    if loops > 0 {
        m.rederive()?;
    }
    Ok(loops)
}

/// [`KeyCache::keys`]' entry for a value whose key is not kept.
const UNKEYED: u64 = u64::MAX;

/// Each instruction's [`inst_key`] by its result, kept from one search of [`roll`] to the next,
/// so a search after a run was rolled keys only the instructions the roll made (and those with no
/// result, which are keyed at every search). A key needs no update when a roll rewrites what an
/// instruction reads: its value operands are not part of it. The roll keeps an instruction's
/// result only where it moves the instruction or rewrites its operands, but for a lane read the
/// first region's read turns into a copy of the trip's lane, and a copy's key is never read
/// (every reader skips copies). Everything else the roll writes takes a new result.
#[derive(Default)]
struct KeyCache {
    /// By value: its instruction's key, or [`UNKEYED`].
    keys: Vec<u64>,
    /// The bytes charged for `keys`.
    charged: usize,
}

impl KeyCache {
    /// The key of `node`, kept or taken now.
    fn key(&mut self, m: &TypedModule, node: &InstNode, meter: &mut Meter) -> Result<u64, Error> {
        let Some(r) = node.result else {
            return inst_key(m, node);
        };
        if let Some(&k) = self.keys.get(r.index()).filter(|k| **k != UNKEYED) {
            // Every kept key is the one the instruction has now (the type's documentation).
            #[cfg(test)]
            if inst_key(m, node)? != k {
                return Err(invariant("a kept roll key differs from its instruction's"));
            }
            return Ok(k);
        }
        let k = inst_key(m, node)?;
        if self.keys.len() <= r.index() {
            let len = m.values.len().max(r.index() + 1);
            let bytes = (len - self.keys.len()).saturating_mul(std::mem::size_of::<u64>());
            meter.charge(bytes)?;
            self.charged = self.charged.saturating_add(bytes);
            self.keys.resize(len, UNKEYED);
        }
        if let Some(slot) = self.keys.get_mut(r.index()) {
            *slot = k;
        }
        Ok(k)
    }
}

/// The instructions Metal compiles for `m`'s entry: each block's label and terminator, its
/// phis, and its instructions, a call counting as its callee's whole body (SPIRV-Cross writes
/// every function inlined at each call).
pub fn entry_weight(m: &TypedModule, meter: &mut Meter) -> Result<u64, Error> {
    let w = weights(m, meter)?;
    Ok(w.get(m.entry.func.index()).copied().unwrap_or(0))
}

/// [`entry_weight`] of every function.
fn weights(m: &TypedModule, meter: &mut Meter) -> Result<Vec<u64>, Error> {
    let n = m.functions.len();
    let mut w = vec![0u64; n];
    // 0: not reached, 1: callees pending, 2: weighed.
    let mut state = vec![0u8; n];
    let callees = |f: usize| -> Vec<usize> {
        let mut out = Vec::new();
        for b in m.functions.get(f).map_or(&[][..], |f| &f.blocks[..]) {
            for node in m.blocks.get(b.index()).map_or(&[][..], |b| &b.insts[..]) {
                if let Inst::Call { func, .. } = node.inst {
                    out.push(func.index());
                }
            }
        }
        out
    };
    for root in 0..n {
        let mut stack = vec![root];
        while let Some(&f) = stack.last() {
            match state.get(f).copied() {
                Some(0) => {
                    if let Some(s) = state.get_mut(f) {
                        *s = 1;
                    }
                    for c in callees(f) {
                        if state.get(c).copied() == Some(0) {
                            stack.push(c);
                        }
                    }
                }
                Some(1) => {
                    stack.pop();
                    let func = m.functions.get(f).ok_or_else(|| invariant("function"))?;
                    let mut total = 0u64;
                    for b in &func.blocks {
                        let block = m.blocks.get(b.index()).ok_or_else(|| invariant("block"))?;
                        meter.tick(u32::try_from(block.insts.len()).unwrap_or(u32::MAX))?;
                        total = total
                            .saturating_add(1)
                            .saturating_add(block.phis.len() as u64);
                        for node in &block.insts {
                            total = total.saturating_add(inst_weight(node, &w));
                        }
                    }
                    if let Some(x) = w.get_mut(f) {
                        *x = total;
                    }
                    if let Some(s) = state.get_mut(f) {
                        *s = 2;
                    }
                }
                _ => {
                    stack.pop();
                }
            }
        }
    }
    Ok(w)
}

/// Whether `node` is an `OpCopyObject`: its result is its operand, so a comparison reads through
/// it (the single-store promotion leaves one where each promoted load was, and where the lowering
/// placed those loads differs at a region's edges).
fn is_copy(node: &InstNode) -> bool {
    matches!(
        node.inst,
        Inst::Unary {
            op: crate::spirv::ir::UnOp::CopyObject,
            ..
        }
    )
}

/// One instruction, or a call with its callee's body.
fn inst_weight(node: &InstNode, w: &[u64]) -> u64 {
    match node.inst {
        Inst::Call { func, .. } => w.get(func.index()).copied().unwrap_or(0).saturating_add(1),
        _ => 1,
    }
}

/// The structural keys' hasher, whose keys are the same in every build: each word a key is made
/// of is mixed in whole (a multiply and a shift, each a bijection of the state), where a byte-wise
/// FNV-1a took a multiply per byte. Keys are only compared for equality, never ordered, so the
/// hash function decides nothing but which keys collide.
struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Fnv {
        Fnv::new()
    }
}

impl Fnv {
    fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }

    fn mix(&mut self, x: u64) {
        let h = (self.0 ^ x).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 = h ^ (h >> 32);
    }
}

impl Hasher for Fnv {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for w in words {
            self.mix(u64::from_le_bytes(*w));
        }
        if !rest.is_empty() {
            let mut b = [0u8; 8];
            if let Some(head) = b.get_mut(..rest.len()) {
                head.copy_from_slice(rest);
            }
            self.mix(u64::from_le_bytes(b) ^ ((rest.len() as u64) << 56));
        }
    }

    fn write_u8(&mut self, x: u8) {
        self.mix(u64::from(x));
    }

    fn write_u16(&mut self, x: u16) {
        self.mix(u64::from(x));
    }

    fn write_u32(&mut self, x: u32) {
        self.mix(u64::from(x));
    }

    fn write_u64(&mut self, x: u64) {
        self.mix(x);
    }

    fn write_usize(&mut self, x: usize) {
        self.mix(x as u64);
    }

    fn write_i8(&mut self, x: i8) {
        self.mix(x as u64);
    }

    fn write_i16(&mut self, x: i16) {
        self.mix(x as u64);
    }

    fn write_i32(&mut self, x: i32) {
        self.mix(x as u64);
    }

    fn write_i64(&mut self, x: i64) {
        self.mix(x as u64);
    }

    fn write_isize(&mut self, x: isize) {
        self.mix(x as u64);
    }
}

/// An item of the spine: instruction `inst` of spine block `spine`, or ([`TAIL`]) the construct
/// that block heads, its merge's phis included, or the spine's end.
#[derive(Clone, Copy, Debug)]
struct Item {
    spine: u32,
    inst: u32,
    key: u64,
    weight: u64,
}

/// The entry's top-level code (see the module docs).
struct Spine {
    /// The layout position of each module block in the entry (`u32::MAX`: another function's).
    pos: Vec<u32>,
    /// The entry's blocks in layout order.
    layout: Vec<BlockId>,
    /// The spine's blocks, in order.
    blocks: Vec<BlockId>,
    items: Vec<Item>,
    /// By layout position, each instruction's [`inst_key`] (0 for a copy, which no key reads),
    /// taken once: an item's key, a construct's (which combines its instructions'), and the
    /// alignment of two regions read it.
    keys: Vec<Box<[u64]>>,
}

impl Spine {
    fn pos(&self, b: BlockId) -> Option<usize> {
        self.pos
            .get(b.index())
            .copied()
            .filter(|p| *p != u32::MAX)
            .map(|p| p as usize)
    }

    fn at(&self, lp: usize) -> Result<BlockId, Error> {
        self.layout
            .get(lp)
            .copied()
            .ok_or_else(|| invariant("layout position"))
    }
}

fn block(m: &TypedModule, b: BlockId) -> Result<&Block, Error> {
    m.blocks.get(b.index()).ok_or_else(|| invariant("block"))
}

fn block_mut(m: &mut TypedModule, b: BlockId) -> Result<&mut Block, Error> {
    m.blocks
        .get_mut(b.index())
        .ok_or_else(|| invariant("block"))
}

fn charge<T>(meter: &mut Meter, ledger: &mut usize, n: usize) -> Result<(), Error> {
    let bytes = n.saturating_mul(std::mem::size_of::<T>());
    meter.charge(bytes)?;
    *ledger = ledger.saturating_add(bytes);
    Ok(())
}

/// The structural key of an instruction: the instruction with its value operands erased, and
/// its result's type. The spine takes each instruction's once ([`Spine::keys`]).
fn inst_key(m: &TypedModule, node: &InstNode) -> Result<u64, Error> {
    let mut h = Fnv::new();
    let mut inst = node.inst.clone();
    inst.for_each_value_mut(|v| *v = Value::new(0));
    // A lane read's lanes may differ by region, which a lift carries (the module docs).
    match &mut inst {
        Inst::CompositeExtract { indices: lanes, .. }
        | Inst::VectorShuffle {
            components: lanes, ..
        } => lanes.iter_mut().for_each(|l| *l = 0),
        _ => {}
    }
    inst.hash(&mut h);
    match node.result {
        Some(r) => m.value(r)?.ty.hash(&mut h),
        None => u32::MAX.hash(&mut h),
    }
    Ok(h.finish())
}

/// Hash `b`'s merge and terminator, their targets relative to layout position `at`.
fn tail_shape(sp: &Spine, b: &Block, at: usize, h: &mut Fnv) {
    let rel = |t: BlockId| sp.pos(t).map_or(i64::MIN, |p| p as i64 - at as i64);
    match b.merge {
        Some(Merge::Selection { merge }) => (1u8, rel(merge), 0i64).hash(h),
        Some(Merge::Loop { merge, cont }) => (2u8, rel(merge), rel(cont)).hash(h),
        None => (0u8, 0i64, 0i64).hash(h),
    }
    std::mem::discriminant(&b.term).hash(h);
    let mut term = b.term.clone();
    term.for_each_target_mut(|t| {
        rel(*t).hash(h);
    });
    if let Terminator::Switch { cases, .. } = &b.term {
        for (lit, _) in cases {
            lit.hash(h);
        }
    }
}

/// The phis of `b`: their count and types.
fn phi_shape(m: &TypedModule, b: &Block, h: &mut Fnv) -> Result<(), Error> {
    b.phis.len().hash(h);
    for p in &b.phis {
        m.value(p.result)?.ty.hash(h);
        p.incoming.len().hash(h);
    }
    Ok(())
}

/// Whether `inst` is integer arithmetic a region may hold where another region holds none:
/// `OpIAdd`, `OpISub`, `OpIMul` and the bitwise and, or and xor, whose results are defined for
/// every operand and computed the same wherever they stand.
fn floats(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Binary {
            op: BinOp::IAdd
                | BinOp::ISub
                | BinOp::IMul
                | BinOp::BitwiseAnd
                | BinOp::BitwiseOr
                | BinOp::BitwiseXor,
            ..
        }
    )
}

/// Whether value `v` is the same wherever the sequence from layout position `lo` runs: a
/// constant, variable, or parameter, a value defined before the sequence, or one of `inv`.
fn fixed(m: &TypedModule, sp: &Spine, lo: usize, inv: &HashSet<Value>, v: Value) -> bool {
    match m.values.get(v.index()).map(|d| d.origin) {
        Some(ValueOrigin::Inst(b) | ValueOrigin::Phi(b)) => {
            sp.pos(b).is_some_and(|p| p < lo) || inv.contains(&v)
        }
        Some(_) => true,
        None => false,
    }
}

/// The entry's spine: its top-level sequence, then the arms of its selections and switches,
/// each a sequence of its own, to [`MAX_DEPTH`] constructs deep. Each sequence ends in an item
/// whose key no other item has, so no run crosses from one into another.
fn spine(
    m: &TypedModule,
    (func, w): (FuncId, &[u64]),
    cache: &mut KeyCache,
    meter: &mut Meter,
    ledger: &mut usize,
) -> Result<Spine, Error> {
    let f = m
        .functions
        .get(func.index())
        .ok_or_else(|| invariant("function"))?;
    charge::<BlockId>(meter, ledger, f.blocks.len())?;
    let layout = f.blocks.clone();
    charge::<u32>(meter, ledger, m.blocks.len())?;
    let mut pos = vec![u32::MAX; m.blocks.len()];
    for (i, b) in layout.iter().enumerate() {
        if let Some(p) = pos.get_mut(b.index()) {
            *p = u32::try_from(i).map_err(|_| invariant("layout"))?;
        }
    }
    let mut keys = Vec::with_capacity(layout.len());
    // Every item at most: each instruction, a construct or end per block, and a sequence's end.
    let mut most = 0usize;
    for b in &layout {
        let insts = &block(m, *b)?.insts;
        most = most.saturating_add(insts.len()).saturating_add(3);
        meter.tick(u32::try_from(insts.len()).unwrap_or(u32::MAX))?;
        charge::<u64>(meter, ledger, insts.len())?;
        let mut k = Vec::with_capacity(insts.len());
        for node in insts {
            k.push(if is_copy(node) {
                0
            } else {
                cache.key(m, node, meter)?
            });
        }
        keys.push(k.into_boxed_slice());
    }
    charge::<Item>(meter, ledger, most)?;
    let mut sp = Spine {
        pos,
        layout,
        blocks: Vec::new(),
        items: Vec::with_capacity(most),
        keys,
    };
    let Some(&start) = sp.layout.first() else {
        return Ok(sp);
    };
    // The sequences to walk: the first block, the layout position the sequence stays below
    // (its construct's merge), and its depth.
    charge::<bool>(meter, ledger, sp.layout.len())?;
    let mut begun = vec![false; sp.layout.len()];
    if let Some(x) = begun.first_mut() {
        *x = true;
    }
    let mut seqs: Vec<(BlockId, usize, u32)> = vec![(start, sp.layout.len(), 0)];
    // The values of the construct being keyed that are the same wherever its sequence runs.
    let mut inv: HashSet<Value> = HashSet::default();
    let mut qi = 0usize;
    while let Some(&(first, hi, depth)) = seqs.get(qi) {
        qi += 1;
        let lo = sp.pos(first).ok_or_else(|| invariant("sequence start"))?;
        let (mut cur, mut lp) = (first, lo);
        let last = loop {
            let b = block(m, cur)?;
            meter.tick(u32::try_from(b.insts.len()).unwrap_or(u32::MAX))?;
            let si = u32::try_from(sp.blocks.len()).map_err(|_| invariant("spine"))?;
            sp.blocks.push(cur);
            let keys = sp.keys.get(lp).ok_or_else(|| invariant("spine keys"))?;
            for (i, (node, &key)) in b.insts.iter().zip(keys.iter()).enumerate() {
                if is_copy(node) {
                    continue;
                }
                sp.items.push(Item {
                    spine: si,
                    inst: u32::try_from(i).map_err(|_| invariant("instruction index"))?,
                    key,
                    weight: inst_weight(node, w),
                });
            }
            // The construct `cur` heads, or the plain branch to the next block, and the next
            // spine block, inside the sequence's construct and not another arm's start.
            let next = match (&b.merge, &b.term) {
                (
                    Some(Merge::Selection { merge }),
                    Terminator::CondBranch { .. } | Terminator::Switch { .. },
                ) => Some(*merge),
                (None, Terminator::Branch(t)) if sp.pos(*t) == Some(lp + 1) => {
                    match block(m, *t)?.merge {
                        Some(Merge::Loop { merge, .. }) => Some(merge),
                        _ => Some(*t),
                    }
                }
                _ => None,
            };
            let np = next
                .and_then(|n| sp.pos(n))
                .filter(|p| *p > lp && *p < hi && !begun.get(*p).copied().unwrap_or(true));
            let (Some(next), Some(np)) = (next, np) else {
                let mut h = Fnv::new();
                (0xE4Du16, std::mem::discriminant(&b.term)).hash(&mut h);
                sp.items.push(Item {
                    spine: si,
                    inst: TAIL,
                    key: h.finish(),
                    weight: 1,
                });
                break si;
            };
            // The construct's key leaves out integer arithmetic on values that are the same
            // wherever the sequence runs, which a region may hold where another holds none; its
            // weight counts it.
            let mut h = Fnv::new();
            tail_shape(&sp, b, lp, &mut h);
            let mut weight = 1u64;
            (np - lp).hash(&mut h);
            for q in lp + 1..np {
                let ib = block(m, sp.at(q)?)?;
                meter.tick(u32::try_from(ib.insts.len()).unwrap_or(u32::MAX))?;
                phi_shape(m, ib, &mut h)?;
                let keys = sp.keys.get(q).ok_or_else(|| invariant("spine keys"))?;
                for (node, &key) in ib.insts.iter().zip(keys.iter()) {
                    let copy = is_copy(node);
                    let mut all_fixed = copy || floats(&node.inst);
                    if all_fixed {
                        node.inst
                            .for_each_value(|v| all_fixed &= fixed(m, &sp, lo, &inv, v));
                    }
                    if let (Some(r), true) = (node.result, all_fixed) {
                        charge::<Value>(meter, ledger, 1)?;
                        inv.insert(r);
                    }
                    if copy {
                        continue;
                    }
                    if !all_fixed {
                        h.write_u64(key);
                    }
                    weight = weight.saturating_add(inst_weight(node, w));
                }
                tail_shape(&sp, ib, lp, &mut h);
                weight = weight
                    .saturating_add(1)
                    .saturating_add(ib.phis.len() as u64);
            }
            inv.clear();
            let nb = block(m, next)?;
            phi_shape(m, nb, &mut h)?;
            weight = weight.saturating_add(nb.phis.len() as u64);
            sp.items.push(Item {
                spine: si,
                inst: TAIL,
                key: h.finish(),
                weight,
            });
            // The arms of a selection or switch are sequences of their own.
            if depth < MAX_DEPTH && matches!(b.merge, Some(Merge::Selection { .. })) {
                let mut arms = Vec::new();
                let mut term = b.term.clone();
                term.for_each_target_mut(|t| arms.push(*t));
                for t in arms {
                    let Some(tp) = sp.pos(t).filter(|p| *p > lp && *p < np) else {
                        continue;
                    };
                    if let Some(x) = begun.get_mut(tp).filter(|x| !**x) {
                        *x = true;
                        seqs.push((t, np, depth + 1));
                    }
                }
            }
            cur = next;
            lp = np;
        };
        // No run crosses into the next sequence.
        let mut h = Fnv::new();
        (0x5E9u16, qi).hash(&mut h);
        sp.items.push(Item {
            spine: last,
            inst: TAIL,
            key: h.finish(),
            weight: 0,
        });
    }
    Ok(sp)
}

/// A run of items `[s, end)` whose keys recur at period `p`.
#[derive(Clone, Copy, Debug)]
struct Cand {
    s: usize,
    end: usize,
    p: usize,
    saving: u64,
}

/// The best run of `m`'s entry that rolls, if any.
fn find(
    m: &TypedModule,
    min_chunk: u64,
    cache: &mut KeyCache,
    meter: &mut Meter,
    ledger: &mut usize,
) -> Result<Option<Plan>, Error> {
    let w = weights(m, meter)?;
    // The entry and every function it calls, heaviest first: a stage's code may sit in a
    // function its entry calls (a fragment function's in the routine its entry point calls
    // between reading its inputs and writing its outputs).
    let n = m.functions.len();
    charge::<bool>(meter, ledger, n)?;
    let mut reached = vec![false; n];
    let mut stack = vec![m.entry.func];
    let mut funcs = Vec::new();
    while let Some(f) = stack.pop() {
        match reached.get_mut(f.index()) {
            Some(r) if !*r => *r = true,
            _ => continue,
        }
        funcs.push(f);
        for b in m
            .functions
            .get(f.index())
            .map_or(&[][..], |x| &x.blocks[..])
        {
            for node in &block(m, *b)?.insts {
                if let Inst::Call { func, .. } = node.inst {
                    stack.push(func);
                }
            }
        }
    }
    let weight = |f: FuncId| w.get(f.index()).copied().unwrap_or(0);
    funcs.retain(|f| weight(*f) >= min_chunk.saturating_mul(2));
    funcs.sort_unstable_by(|a, b| weight(*b).cmp(&weight(*a)).then(a.index().cmp(&b.index())));
    funcs.truncate(MAX_FUNCS);
    let mut verifies = 0usize;
    for f in funcs {
        let mut fledger = 0usize;
        let r = find_in(
            m,
            (f, &w, min_chunk),
            cache,
            meter,
            &mut fledger,
            &mut verifies,
        );
        meter.release(fledger);
        if let Some(plan) = r? {
            return Ok(Some(plan));
        }
        if verifies >= MAX_VERIFIES {
            break;
        }
    }
    Ok(None)
}

/// The best run of function `func` that rolls, if any, within the search's `verifies`.
fn find_in(
    m: &TypedModule,
    (func, w, min_chunk): (FuncId, &[u64], u64),
    cache: &mut KeyCache,
    meter: &mut Meter,
    ledger: &mut usize,
    verifies: &mut usize,
) -> Result<Option<Plan>, Error> {
    let sp = spine(m, (func, w), cache, meter, ledger)?;
    let n = sp.items.len();
    // What each value of the function is through its copies.
    charge::<u32>(meter, ledger, m.values.len())?;
    let mut canon: Vec<u32> = (0..m.values.len())
        .map(|i| u32::try_from(i).unwrap_or(u32::MAX))
        .collect();
    for b in &sp.layout {
        for node in &block(m, *b)?.insts {
            if let (Some(r), Inst::Unary { a, .. }) = (node.result, &node.inst) {
                if is_copy(node) {
                    let src = canon.get(a.index()).copied().unwrap_or(a.0);
                    if let Some(c) = canon.get_mut(r.index()) {
                        *c = src;
                    }
                }
            }
        }
    }
    if n < 3 {
        return Ok(None);
    }
    let key = |i: usize| sp.items.get(i).map(|it| it.key);
    charge::<u64>(meter, ledger, n + 1)?;
    let mut pw = Vec::with_capacity(n + 1);
    pw.push(0u64);
    for it in &sp.items {
        let last = pw.last().copied().unwrap_or(0);
        pw.push(last.saturating_add(it.weight));
    }
    let chunk_weight = |a: usize, p: usize| {
        pw.get(a + p)
            .copied()
            .unwrap_or(0)
            .saturating_sub(pw.get(a).copied().unwrap_or(0))
    };
    // Anchors: constructs and calls whose key recurs, heaviest first.
    let mut groups: HashMap<u64, (u64, Vec<usize>)> = HashMap::default();
    for (i, it) in sp.items.iter().enumerate() {
        let call = it.inst != TAIL && it.weight > 1;
        if (it.inst == TAIL || call) && it.weight >= ANCHOR_MIN && i + 1 < n {
            groups
                .entry(it.key)
                .or_insert_with(|| (it.weight, Vec::new()))
                .1
                .push(i);
        }
    }
    let mut groups: Vec<(u64, Vec<usize>)> =
        groups.into_values().filter(|g| g.1.len() >= 2).collect();
    groups.sort_unstable_by(|a, b| {
        b.0.cmp(&a.0)
            .then(a.1.len().cmp(&b.1.len()))
            .then(a.1.first().cmp(&b.1.first()))
    });
    groups.truncate(MAX_ANCHORS);
    // Periods: where `k` regions each hold `n / k` of an anchor's `n` occurrences, the first and
    // the `(n / k)`-th are a period apart; where a region holds one, consecutive ones are.
    let mut cands: Vec<Cand> = Vec::new();
    let mut seen: HashSet<(usize, usize)> = HashSet::default();
    let mut steps = n.saturating_mul(8).saturating_add(1024);
    'groups: for (_, at) in &groups {
        let count = at.len();
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for k in 2..=count.min(MAX_TRIPS) {
            if count % k == 0 {
                if let (Some(&i), Some(&j)) = (at.first(), at.get(count / k)) {
                    pairs.push((i, j));
                }
            }
        }
        pairs.extend(
            at.windows(2)
                .take(MAX_PAIRS)
                .filter_map(|w| Some((*w.first()?, *w.get(1)?))),
        );
        for (i, j) in pairs {
            let p = j - i;
            let mut s = i;
            while s > 0 && steps > 0 && key(s - 1) == key(s - 1 + p) {
                s -= 1;
                steps -= 1;
            }
            if !seen.insert((s, p)) {
                continue;
            }
            let mut e = i;
            while e + p < n && steps > 0 && key(e) == key(e + p) {
                e += 1;
                steps -= 1;
            }
            meter.tick(u32::try_from(e - s).unwrap_or(u32::MAX))?;
            if steps == 0 {
                break 'groups;
            }
            let end = e + p;
            let k = ((end - s) / p).min(MAX_TRIPS);
            if k >= 2 {
                cands.push(Cand {
                    s,
                    end,
                    p,
                    saving: chunk_weight(s, p).saturating_mul(k as u64 - 1),
                });
            }
        }
    }
    cands.sort_unstable_by(|a, b| {
        b.saving
            .cmp(&a.saving)
            .then(a.s.cmp(&b.s))
            .then(a.p.cmp(&b.p))
    });
    // Where the function's variables are used, read once for every candidate that needs it.
    let mut uses: Option<Vec<VarUse>> = None;
    for c in cands {
        for a in (c.s..c.end).take(MAX_PHASES) {
            let k = ((c.end - a) / c.p).min(MAX_TRIPS);
            if k < 2 {
                break;
            }
            if chunk_weight(a, c.p) < min_chunk || a + k * c.p >= n {
                continue;
            }
            *verifies += 1;
            let mut vledger = 0usize;
            let r = verify(
                m,
                (func, &sp, &canon),
                (&mut uses, &mut *ledger),
                (a, c.p, k),
                meter,
                &mut vledger,
            );
            meter.release(vledger);
            if let Some(plan) = r? {
                return Ok(Some(plan));
            }
            if *verifies >= MAX_VERIFIES {
                return Ok(None);
            }
        }
    }
    Ok(None)
}

/// An operand of the first region, named where it is before the rewrite.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Site {
    /// Value operand `operand` (in [`Inst::for_each_value`] order) of instruction `index`.
    Inst {
        block: BlockId,
        index: usize,
        operand: usize,
    },
    /// The value of incoming `incoming` of phi `phi`.
    Phi {
        block: BlockId,
        phi: usize,
        incoming: usize,
    },
    /// Value operand `operand` of the terminator.
    Term { block: BlockId, operand: usize },
}

/// An image the first region loads, which differs by trip.
#[derive(Clone, Debug)]
struct Handle {
    /// The load, its result, and its memory operands.
    load: (BlockId, usize),
    result: Value,
    access: MemAccess,
    /// Each trip's variable.
    globals: Vec<GlobalId>,
    /// Each instruction that uses the image (its only image operand, its first).
    uses: Vec<(BlockId, usize)>,
}

/// An operand of the first region that reads a value defined before the run, where every other
/// region reads the previous region's value at one place (the module docs).
#[derive(Clone, Debug)]
struct Chain {
    site: Site,
    /// The first region's operand, the chain's start.
    init: Value,
    /// The first region's value each next trip reads.
    carried: Value,
    /// The regions (but the first) whose operand is chained.
    trips: Vec<bool>,
}

/// A lane read of the first region, instruction `at`, whose regions read other lanes or other
/// values defined before the run: each trip's instruction (the module docs).
#[derive(Clone, Debug)]
struct Lift {
    at: (BlockId, usize),
    ty: TypeId,
    insts: Vec<Inst>,
}

/// Whether `inst` reads lanes of its operands at literal places: an `OpCompositeExtract` or an
/// `OpVectorShuffle`.
fn lane_read(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::CompositeExtract { .. } | Inst::VectorShuffle { .. }
    )
}

/// A run that rolls, and everything its rewrite needs, in the module's ids before the rewrite.
#[derive(Clone, Debug)]
struct Plan {
    /// The function the run is in.
    func: FuncId,
    k: usize,
    /// The first region's first block and the index of its first instruction there.
    s: BlockId,
    a: usize,
    /// The second region's first block (the first's last), and the index of the second's first
    /// instruction there.
    e: BlockId,
    e_cut: usize,
    /// The block after the run, and the index of its first instruction past the run.
    u: BlockId,
    u_cut: usize,
    /// Layout positions of `s`, `e`, and `u`.
    s_pos: usize,
    e_pos: usize,
    u_pos: usize,
    /// Each differing value operand of the first region, with each trip's value.
    sites: Vec<(Site, Vec<Value>)>,
    /// Each chained operand of the first region ([`Chain`]).
    chains: Vec<Chain>,
    /// Each lifted lane read of the first region ([`Lift`]).
    lifts: Vec<Lift>,
    /// The instructions of other regions computed before the loop, in order, each with its
    /// result there and its operands read through their copies ([`align`]).
    hoists: Vec<(Value, InstNode)>,
    handles: Vec<Handle>,
    /// Each constant operand of the first region that differs by trip, with each trip's.
    switched: Vec<(Site, Vec<Value>)>,
    /// The variables of the other regions, which leave.
    drop_vars: Vec<VarId>,
    /// The first region's variables to store their initializer into at each trip: (pointer,
    /// initializer).
    reinit: Vec<(Value, Value)>,
    /// Values of the last region read after the run, and the first region's values that stand
    /// for them.
    escapes: HashMap<Value, Value>,
    /// Values of other regions read after the run: each first-region value whose trip's value
    /// is kept for after the loop, and the trips that need it.
    carries: Vec<(Value, Vec<usize>)>,
    /// Each value read after the run that one of those stands for: (the value, the first
    /// region's, the trip).
    carried: Vec<(Value, Value, usize)>,
}

/// Where a run's regions are: region `j` spans layout positions `first[j]..=first[j + 1]`, from
/// instruction `cut[j]` of its first block to instruction `cut[j + 1]` (exclusive) of its last,
/// whose phis are the region's too.
struct Geometry {
    k: usize,
    first: Vec<usize>,
    cut: Vec<usize>,
    /// Blocks per region.
    n: usize,
}

/// What part of a block an operand or result is in.
#[derive(Clone, Copy)]
enum Part {
    Phi,
    Inst(usize),
    Tail,
}

impl Geometry {
    /// The region `part` of the block at layout position `lp` belongs to.
    fn chunk_at(&self, lp: usize, part: Part) -> Option<usize> {
        let j = self.first.partition_point(|f| *f <= lp).checked_sub(1)?;
        let at_first = self.first.get(j) == Some(&lp);
        let k = self.k;
        match part {
            Part::Phi => {
                if at_first {
                    j.checked_sub(1).filter(|p| *p < k)
                } else {
                    (j < k).then_some(j)
                }
            }
            Part::Inst(i) => {
                if at_first {
                    let c = self.cut.get(j).copied().unwrap_or(0);
                    if i >= c {
                        (j < k).then_some(j)
                    } else {
                        j.checked_sub(1).filter(|p| *p < k)
                    }
                } else {
                    (j < k).then_some(j)
                }
            }
            Part::Tail => (j < k).then_some(j),
        }
    }

    /// The instructions of region `j` in its block at offset `t`, of `len` instructions.
    fn insts(&self, j: usize, t: usize, len: usize) -> Range<usize> {
        if t == 0 {
            self.cut.get(j).copied().unwrap_or(len).min(len)..len
        } else if t + 1 == self.n {
            0..self.cut.get(j + 1).copied().unwrap_or(0).min(len)
        } else {
            0..len
        }
    }
}

/// Where each variable is used: in one region only, or (`SHARED`) elsewhere too.
const SHARED: usize = usize::MAX;

/// The instructions of `b` in `range` but its copies, with their indices.
fn region_insts(b: &Block, range: Range<usize>) -> Vec<(usize, &InstNode)> {
    let start = range.start;
    b.insts
        .get(range)
        .unwrap_or(&[])
        .iter()
        .enumerate()
        .filter(|(_, n)| !is_copy(n))
        .map(|(i, n)| (start + i, n))
        .collect()
}

/// Whether a value may differ by trip in operand `operand` of `inst`: an operand any value of
/// its type may stand in.
fn varying_ok(inst: &Inst, operand: usize) -> bool {
    match inst {
        Inst::Store { .. } => operand == 1,
        Inst::Unary { .. }
        | Inst::Binary { .. }
        | Inst::Select { .. }
        | Inst::Fma { .. }
        | Inst::Convert { .. }
        | Inst::CompositeConstruct { .. }
        | Inst::CompositeExtract { .. }
        | Inst::CompositeInsert { .. }
        | Inst::VectorExtractDynamic { .. }
        | Inst::VectorInsertDynamic { .. }
        | Inst::VectorShuffle { .. }
        | Inst::BitFieldInsert { .. }
        | Inst::BitFieldExtract { .. }
        | Inst::Glsl { .. }
        | Inst::Call { .. } => true,
        Inst::ImageWrite { .. } => operand == 1 || operand == 2,
        Inst::ImageRead { .. } | Inst::ImageFetch { .. } => operand == 1,
        _ => false,
    }
}

/// Whether a region may hold `inst`: an instruction the stage executes at most once (interlock,
/// mesh output counts) may not be repeated by a loop.
fn rollable(inst: &Inst) -> bool {
    !matches!(
        inst,
        Inst::BeginInterlock | Inst::EndInterlock | Inst::SetMeshOutputs { .. }
    )
}

/// Whether an instruction may use a trip's image, as its first operand.
fn image_use(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::ImageWrite { .. }
            | Inst::ImageRead { .. }
            | Inst::ImageFetch { .. }
            | Inst::ImageQuery { .. }
    )
}

/// The operand of `inst` (in [`Inst::for_each_value`] order) that must be a constant but may
/// differ by trip through an `OpSwitch` on the trip index ([`split_by_trip`]): an image
/// instruction's `ConstOffset`.
fn switched_operand(inst: &Inst) -> Option<usize> {
    let (lead, ops) = match inst {
        Inst::ImageSample { dref, ops, .. } => (2 + usize::from(dref.is_some()), ops),
        Inst::ImageFetch { ops, .. } | Inst::ImageGather { ops, .. } => (2, ops),
        Inst::ImageDrefGather { ops, .. } => (3, ops),
        _ => return None,
    };
    ops.const_offset?;
    Some(
        lead + usize::from(ops.bias.is_some())
            + usize::from(ops.lod.is_some())
            + 2 * usize::from(ops.grad.is_some()),
    )
}

/// The record of an exact comparison of the run's regions.
struct Compare<'m> {
    m: &'m TypedModule,
    k: usize,
    /// What each value is through its copies.
    canon: &'m [u32],
    /// For each value: 0, or 1 + the region that defines it.
    def: Vec<u16>,
    /// For each value of a region but the first: the first region's value at the same place.
    corr: Vec<u32>,
    /// Where each variable (a `Function` variable's or `Private` variable's pointer) is used.
    home: HashMap<Value, usize>,
    sites: Vec<(Site, Vec<Value>)>,
    site_at: HashMap<Site, usize>,
    chains: Vec<Chain>,
    chain_at: HashMap<Site, usize>,
    lifts: Vec<Lift>,
    lift_at: HashMap<(BlockId, usize), usize>,
    handles: Vec<(Site, Vec<GlobalId>)>,
    handle_at: HashMap<Site, usize>,
    /// Each constant operand that differs by trip ([`switched_operand`]), with each trip's.
    switched: Vec<(Site, Vec<Value>)>,
    switched_at: HashMap<Site, usize>,
    /// Each other region's variable and the first region's that replaces it.
    var_to: HashMap<VarId, VarId>,
    var_from: HashMap<(usize, VarId), VarId>,
    global_to: HashMap<GlobalId, GlobalId>,
    global_from: HashMap<(usize, GlobalId), GlobalId>,
    reinit: Vec<(Value, Value)>,
    reinit_seen: HashSet<Value>,
}

impl Compare<'_> {
    fn canon(&self, v: Value) -> Value {
        Value::new(self.canon.get(v.index()).copied().unwrap_or(v.0))
    }

    fn def(&self, v: Value) -> usize {
        usize::from(self.def.get(v.index()).copied().unwrap_or(0))
    }

    fn corr(&self, v: Value) -> Option<Value> {
        self.corr
            .get(v.index())
            .copied()
            .filter(|c| *c != u32::MAX)
            .map(Value::new)
    }

    fn origin(&self, v: Value) -> Option<ValueOrigin> {
        self.m.values.get(v.index()).map(|d| d.origin)
    }

    fn ty(&self, v: Value) -> Option<TypeId> {
        self.m.values.get(v.index()).map(|d| d.ty)
    }

    /// Whether operand `v` of region `j` stands where `w` stands in the first region, recording
    /// what differs. `data_ok`: any value of the type may stand there; `handle_ok`: the operand
    /// is a load's pointer.
    fn operand(
        &mut self,
        j: usize,
        v: Value,
        w: Value,
        site: Site,
        data_ok: bool,
        handle_ok: bool,
    ) -> bool {
        let (v, w) = (self.canon(v), self.canon(w));
        let (dv, dw) = (self.def(v), self.def(w));
        if dv == j + 1 {
            return self.corr(v) == Some(w);
        }
        // The previous region's value where the first region reads one from before the run: a
        // chain, carried from trip to trip.
        if dv == j && dw == 0 && data_ok {
            let carried = if j == 1 { Some(v) } else { self.corr(v) };
            return carried.is_some_and(|c| self.chain(j, site, w, c));
        }
        if dv != 0 || dw != 0 {
            return false;
        }
        if v == w {
            return true;
        }
        match (self.origin(v), self.origin(w)) {
            (Some(ValueOrigin::Var(a)), Some(ValueOrigin::Var(b))) => self.pair_var(j, a, b),
            (Some(ValueOrigin::Global(a)), Some(ValueOrigin::Global(b))) => {
                let (Some(ga), Some(gb)) =
                    (self.m.globals.get(a.index()), self.m.globals.get(b.index()))
                else {
                    return false;
                };
                let private = |g: &crate::spirv::ir::GlobalVar| {
                    g.class == StorageClass::Private && g.kind == GlobalKind::Plain
                };
                if private(ga) && private(gb) {
                    self.pair_global(j, a, b)
                } else if handle_ok
                    && ga.class == StorageClass::UniformConstant
                    && gb.class == StorageClass::UniformConstant
                    && ga.pointee == gb.pointee
                    && matches!(
                        self.m.types.facts(ga.pointee).map(|f| f.shape),
                        Ok(Shape::Opaque)
                    )
                {
                    self.handle(j, site, a, b)
                } else {
                    false
                }
            }
            _ => {
                let (Some(tv), Some(tw)) = (self.ty(v), self.ty(w)) else {
                    return false;
                };
                let data = self.m.types.facts(tv).is_ok_and(|f| {
                    matches!(f.shape, Shape::Plain | Shape::Implicit | Shape::Explicit)
                        && !f.runtime_sized
                });
                data_ok && tv == tw && data && self.data(j, site, v, w)
            }
        }
    }

    fn data(&mut self, j: usize, site: Site, v: Value, w: Value) -> bool {
        if self.chain_at.contains_key(&site) {
            return false;
        }
        let k = self.k;
        let i = match self.site_at.get(&site) {
            Some(i) => *i,
            None => {
                self.sites.push((site, vec![w; k]));
                self.site_at.insert(site, self.sites.len() - 1);
                self.sites.len() - 1
            }
        };
        match self.sites.get_mut(i).and_then(|s| s.1.get_mut(j)) {
            Some(x) => {
                *x = v;
                true
            }
            None => false,
        }
    }

    /// Operand `site` of region `j` reads the previous region's value that the first region's
    /// `carried` is, where the first region reads `init`, defined before the run.
    fn chain(&mut self, j: usize, site: Site, init: Value, carried: Value) -> bool {
        let carries = |v: Value| {
            self.ty(v).is_some_and(|t| {
                self.m.types.facts(t).is_ok_and(|f| {
                    matches!(f.shape, Shape::Plain | Shape::Implicit) && !f.runtime_sized
                })
            }) && !self.m.non_uniform.contains(&v)
        };
        if self.site_at.contains_key(&site)
            || self.ty(init) != self.ty(carried)
            || !carries(init)
            || !carries(carried)
        {
            return false;
        }
        let k = self.k;
        let i = match self.chain_at.get(&site) {
            Some(i) => *i,
            None => {
                self.chains.push(Chain {
                    site,
                    init,
                    carried,
                    trips: vec![false; k],
                });
                self.chain_at.insert(site, self.chains.len() - 1);
                self.chains.len() - 1
            }
        };
        match self.chains.get_mut(i) {
            Some(c) if c.init == init && c.carried == carried => match c.trips.get_mut(j) {
                Some(t) => {
                    *t = true;
                    true
                }
                None => false,
            },
            _ => false,
        }
    }

    /// Whether lane read `x` of region `j` stands for the first region's `y`, at `at`, reading
    /// other lanes or other values defined before the run: `None` where it is not such a read.
    fn lift(&mut self, j: usize, x: &InstNode, y: &InstNode, at: (BlockId, usize)) -> Option<bool> {
        if !lane_read(&y.inst) || std::mem::discriminant(&x.inst) != std::mem::discriminant(&y.inst)
        {
            return None;
        }
        // Each read with its operands as the values they copy, all defined before the run.
        let before = |cmp: &Self, inst: &Inst| -> Option<Inst> {
            let mut inst = inst.clone();
            let mut ok = true;
            inst.for_each_value_mut(|v| {
                *v = cmp.canon(*v);
                ok &= cmp.def(*v) == 0;
            });
            ok.then_some(inst)
        };
        let (xi, yi) = (before(self, &x.inst)?, before(self, &y.inst)?);
        if xi == yi {
            return None;
        }
        let ty = self.ty(y.result?)?;
        let k = self.k;
        let i = match self.lift_at.get(&at) {
            Some(i) => *i,
            None => {
                self.lifts.push(Lift {
                    at,
                    ty,
                    insts: vec![yi; k],
                });
                self.lift_at.insert(at, self.lifts.len() - 1);
                self.lifts.len() - 1
            }
        };
        Some(
            match self.lifts.get_mut(i).and_then(|l| l.insts.get_mut(j)) {
                Some(slot) => {
                    *slot = xi;
                    true
                }
                None => false,
            },
        )
    }

    fn handle(&mut self, j: usize, site: Site, a: GlobalId, b: GlobalId) -> bool {
        let k = self.k;
        let i = match self.handle_at.get(&site) {
            Some(i) => *i,
            None => {
                self.handles.push((site, vec![b; k]));
                self.handle_at.insert(site, self.handles.len() - 1);
                self.handles.len() - 1
            }
        };
        match self.handles.get_mut(i).and_then(|s| s.1.get_mut(j)) {
            Some(x) => {
                *x = a;
                true
            }
            None => false,
        }
    }

    /// Whether constant `v` of region `j` may stand where constant `w` stands in the first
    /// region, at an operand [`switched_operand`] names, recording it.
    fn switched(&mut self, j: usize, site: Site, v: Value, w: Value) -> bool {
        let (v, w) = (self.canon(v), self.canon(w));
        let constant = |x: Value| matches!(self.origin(x), Some(ValueOrigin::Const(_)));
        if !constant(v) || !constant(w) || self.ty(v) != self.ty(w) {
            return false;
        }
        let k = self.k;
        let i = match self.switched_at.get(&site) {
            Some(i) => *i,
            None => {
                self.switched.push((site, vec![w; k]));
                self.switched_at.insert(site, self.switched.len() - 1);
                self.switched.len() - 1
            }
        };
        match self.switched.get_mut(i).and_then(|s| s.1.get_mut(j)) {
            Some(x) => {
                *x = v;
                true
            }
            None => false,
        }
    }

    fn homed(&self, v: Value, j: usize) -> bool {
        self.home.get(&v) == Some(&j)
    }

    /// Variable `a` of region `j` stands for the first region's `b`.
    fn pair_var(&mut self, j: usize, a: VarId, b: VarId) -> bool {
        let (Some(va), Some(vb)) = (self.m.vars.get(a.index()), self.m.vars.get(b.index())) else {
            return false;
        };
        if va.pointee != vb.pointee
            || va.init != vb.init
            || !self.homed(va.value, j)
            || !self.homed(vb.value, 0)
        {
            return false;
        }
        if *self.var_to.entry(a).or_insert(b) != b || *self.var_from.entry((j, b)).or_insert(a) != a
        {
            return false;
        }
        if let Some(c) = vb.init {
            if self.reinit_seen.insert(vb.value) {
                match self.m.const_value(c) {
                    Ok(cv) => self.reinit.push((vb.value, cv)),
                    Err(_) => return false,
                }
            }
        }
        true
    }

    /// `Private` variable `a` of region `j` stands for the first region's `b`.
    fn pair_global(&mut self, j: usize, a: GlobalId, b: GlobalId) -> bool {
        let (Some(ga), Some(gb)) = (self.m.globals.get(a.index()), self.m.globals.get(b.index()))
        else {
            return false;
        };
        if ga.pointee != gb.pointee
            || ga.init != gb.init
            || !self.homed(ga.value, j)
            || !self.homed(gb.value, 0)
        {
            return false;
        }
        if *self.global_to.entry(a).or_insert(b) != b
            || *self.global_from.entry((j, b)).or_insert(a) != a
        {
            return false;
        }
        if let Some(c) = gb.init {
            if self.reinit_seen.insert(gb.value) {
                match self.m.const_value(c) {
                    Ok(cv) => self.reinit.push((gb.value, cv)),
                    Err(_) => return false,
                }
            }
        }
        true
    }

    /// Whether instruction `x` of region `j` is instruction `y` of the first region, at `at`.
    fn inst(&mut self, j: usize, x: &InstNode, y: &InstNode, at: (BlockId, usize)) -> bool {
        if !rollable(&y.inst) {
            return false;
        }
        if let Some(lifted) = self.lift(j, x, y, at) {
            return lifted;
        }
        let mut ops = Vec::new();
        y.inst.for_each_value(|v| ops.push(v));
        let mut xi = x.inst.clone();
        let (mut i, mut ok) = (0usize, true);
        let load = matches!(y.inst, Inst::Load { .. });
        let switched = switched_operand(&y.inst);
        xi.for_each_value_mut(|v| {
            match ops.get(i) {
                Some(&w) => {
                    let site = Site::Inst {
                        block: at.0,
                        index: at.1,
                        operand: i,
                    };
                    let data_ok = varying_ok(&y.inst, i);
                    if self.operand(j, *v, w, site, data_ok, load && i == 0)
                        || (switched == Some(i) && self.switched(j, site, *v, w))
                    {
                        *v = w;
                    } else {
                        ok = false;
                    }
                }
                None => ok = false,
            }
            i += 1;
        });
        ok && i == ops.len() && xi == y.inst
    }
}

/// Pair instructions `xs` of a region but the first with the first region's
/// `ys`, in order: an instruction pairs with the next one of the first region whose structural
/// key is its own, and one that does not and [`floats`] on values defined before the run is
/// computed before the loop instead: its result counts as defined before the run, and it is
/// recorded, its operands read through their copies, in `hoists`. The pairs, as positions in `xs`
/// and `ys`; `None` where the regions differ otherwise.
fn align(
    m: &TypedModule,
    cmp: &mut Compare<'_>,
    (xs, kx): (&[(usize, &InstNode)], &[u64]),
    (ys, ky): (&[(usize, &InstNode)], &[u64]),
    hoists: &mut Vec<(Value, InstNode)>,
) -> Result<Option<Vec<(usize, usize)>>, Error> {
    let key = |keys: &[u64], i: usize| -> Result<u64, Error> {
        keys.get(i)
            .copied()
            .ok_or_else(|| invariant("instruction key"))
    };
    let mut order = Vec::with_capacity(ys.len());
    let mut iy = 0usize;
    for (ix, &(xi, x)) in xs.iter().enumerate() {
        if let Some(&(yi, _)) = ys.get(iy) {
            if key(kx, xi)? == key(ky, yi)? {
                order.push((ix, iy));
                iy += 1;
                continue;
            }
        }
        let mut before = floats(&x.inst);
        if before {
            x.inst
                .for_each_value(|v| before &= cmp.def(cmp.canon(v)) == 0);
        }
        let (true, Some(r)) = (before, x.result) else {
            return Ok(None);
        };
        if m.non_uniform.contains(&r) {
            return Ok(None);
        }
        if let Some(d) = cmp.def.get_mut(r.index()) {
            *d = 0;
        }
        let mut node = (*x).clone();
        node.inst.for_each_value_mut(|v| *v = cmp.canon(*v));
        hoists.push((r, node));
    }
    Ok((iy == ys.len()).then_some(order))
}

/// A use of a variable (a `Function` variable, or a plain `Private` one): the variable, read
/// through its copies, and where it is used in the function searched (its layout position and
/// the part of the block), or `None` in another function.
#[derive(Clone, Copy)]
struct VarUse {
    var: Value,
    at: Option<(usize, Part)>,
}

/// Every use of a variable in `m` ([`VarUse`]), in the function `func` the spine `sp` lays out
/// and in every other function, each operand read through `canon`. A candidate run's regions
/// then take each variable's home from these alone, where each read every operand of the module
/// again.
fn var_uses(
    m: &TypedModule,
    (func, sp, canon): (FuncId, &Spine, &[u32]),
    meter: &mut Meter,
    ledger: &mut usize,
) -> Result<Vec<VarUse>, Error> {
    let is_var = |v: Value| match m.values.get(v.index()).map(|d| d.origin) {
        Some(ValueOrigin::Var(_)) => true,
        Some(ValueOrigin::Global(gid)) => m
            .globals
            .get(gid.index())
            .is_some_and(|gv| gv.class == StorageClass::Private && gv.kind == GlobalKind::Plain),
        _ => false,
    };
    let mut out = Vec::new();
    let mut note = |v: Value, at: Option<(usize, Part)>| {
        let var = Value::new(canon.get(v.index()).copied().unwrap_or(v.0));
        if is_var(var) {
            out.push(VarUse { var, at });
        }
    };
    for (lp, b) in sp.layout.iter().enumerate() {
        let bl = block(m, *b)?;
        meter.tick(u32::try_from(bl.insts.len()).unwrap_or(u32::MAX))?;
        for p in &bl.phis {
            for (v, _) in &p.incoming {
                note(*v, Some((lp, Part::Phi)));
            }
        }
        for (i, node) in bl.insts.iter().enumerate() {
            node.inst
                .for_each_value(|v| note(v, Some((lp, Part::Inst(i)))));
        }
        bl.term.for_each_value(|v| note(v, Some((lp, Part::Tail))));
    }
    for l in &m.loops {
        if let (Some(v), Some(lp)) = (l.exit_sel, sp.pos(l.header)) {
            if let Some(value) = m.vars.get(v.index()).map(|x| x.value) {
                note(value, Some((lp, Part::Tail)));
            }
        }
    }
    for b in &m.blocks {
        if b.func == func {
            continue;
        }
        meter.tick(u32::try_from(b.insts.len()).unwrap_or(u32::MAX))?;
        for node in &b.insts {
            node.inst.for_each_value(|v| note(v, None));
        }
        for p in &b.phis {
            for (v, _) in &p.incoming {
                note(*v, None);
            }
        }
        b.term.for_each_value(|v| note(v, None));
    }
    charge::<VarUse>(meter, ledger, out.len())?;
    Ok(out)
}

/// Compare the run of `k` regions of `p` items from item `a` exactly; the plan that rolls it, if
/// it does.
fn verify(
    m: &TypedModule,
    (func, sp, canon): (FuncId, &Spine, &[u32]),
    (uses, uses_ledger): (&mut Option<Vec<VarUse>>, &mut usize),
    (a, p, k): (usize, usize, usize),
    meter: &mut Meter,
    ledger: &mut usize,
) -> Result<Option<Plan>, Error> {
    // Geometry.
    let mut first = Vec::with_capacity(k + 1);
    let mut cut = Vec::with_capacity(k + 1);
    for j in 0..=k {
        let it = sp
            .items
            .get(a + j * p)
            .ok_or_else(|| invariant("run item"))?;
        let b = *sp
            .blocks
            .get(it.spine as usize)
            .ok_or_else(|| invariant("spine block"))?;
        let lp = sp.pos(b).ok_or_else(|| invariant("spine position"))?;
        let c = if it.inst == TAIL {
            block(m, b)?.insts.len()
        } else {
            it.inst as usize
        };
        first.push(lp);
        cut.push(c);
    }
    let (Some(&f0), Some(&f1)) = (first.first(), first.get(1)) else {
        return Ok(None);
    };
    if f1 <= f0 {
        return Ok(None);
    }
    let n = f1 - f0 + 1;
    if first
        .windows(2)
        .any(|w| w.get(1).zip(w.first()).map(|(b, a)| b - a + 1) != Some(n))
    {
        return Ok(None);
    }
    let g = Geometry { k, first, cut, n };
    let fst = |j: usize| g.first.get(j).copied().unwrap_or(usize::MAX);
    let nvalues = m.values.len();
    charge::<u16>(meter, ledger, nvalues)?;
    charge::<u32>(meter, ledger, nvalues)?;
    let mut cmp = Compare {
        m,
        k,
        canon,
        def: vec![0u16; nvalues],
        corr: vec![u32::MAX; nvalues],
        home: HashMap::default(),
        sites: Vec::new(),
        site_at: HashMap::default(),
        chains: Vec::new(),
        chain_at: HashMap::default(),
        lifts: Vec::new(),
        lift_at: HashMap::default(),
        handles: Vec::new(),
        handle_at: HashMap::default(),
        switched: Vec::new(),
        switched_at: HashMap::default(),
        var_to: HashMap::default(),
        var_from: HashMap::default(),
        global_to: HashMap::default(),
        global_from: HashMap::default(),
        reinit: Vec::new(),
        reinit_seen: HashSet::default(),
    };
    // The values each region defines.
    for j in 0..k {
        let d = u16::try_from(j + 1).map_err(|_| invariant("trips"))?;
        for t in 0..n {
            let b = block(m, sp.at(fst(j) + t)?)?;
            meter.tick(u32::try_from(b.insts.len()).unwrap_or(u32::MAX))?;
            if t > 0 {
                for ph in &b.phis {
                    if let Some(x) = cmp.def.get_mut(ph.result.index()) {
                        *x = d;
                    }
                }
            }
            for node in b.insts.get(g.insts(j, t, b.insts.len())).unwrap_or(&[]) {
                if let Some(r) = node.result {
                    if let Some(x) = cmp.def.get_mut(r.index()) {
                        *x = d;
                    }
                }
            }
        }
    }
    // Each value of the other regions paired with the first region's at the same place; an
    // instruction of another region with no counterpart that floats on values from before the
    // run is computed before the loop.
    let mut aligned: Vec<Vec<(usize, usize)>> = Vec::with_capacity((k - 1) * n);
    let mut hoists: Vec<(Value, InstNode)> = Vec::new();
    for j in 1..k {
        for t in 0..n {
            let (bj, b0) = (block(m, sp.at(fst(j) + t)?)?, block(m, sp.at(fst(0) + t)?)?);
            let mut pairs: Vec<(Value, Value)> = Vec::new();
            if t > 0 {
                if bj.phis.len() != b0.phis.len() {
                    return Ok(None);
                }
                pairs.extend(
                    bj.phis
                        .iter()
                        .zip(&b0.phis)
                        .map(|(x, y)| (x.result, y.result)),
                );
            }
            let (xs, ys) = (
                region_insts(bj, g.insts(j, t, bj.insts.len())),
                region_insts(b0, g.insts(0, t, b0.insts.len())),
            );
            let (kx, ky) = (
                sp.keys
                    .get(fst(j) + t)
                    .ok_or_else(|| invariant("region keys"))?,
                sp.keys
                    .get(fst(0) + t)
                    .ok_or_else(|| invariant("region keys"))?,
            );
            let Some(order) = align(m, &mut cmp, (&xs, kx), (&ys, ky), &mut hoists)? else {
                return Ok(None);
            };
            charge::<(usize, usize)>(meter, ledger, order.len())?;
            for &(xi, yi) in &order {
                let (Some((_, x)), Some((_, y))) = (xs.get(xi), ys.get(yi)) else {
                    return Err(invariant("aligned instruction"));
                };
                match (x.result, y.result) {
                    (Some(rx), Some(ry)) => pairs.push((rx, ry)),
                    (None, None) => {}
                    _ => return Ok(None),
                }
            }
            aligned.push(order);
            for (x, y) in pairs {
                if m.value(x)?.ty != m.value(y)?.ty
                    || m.non_uniform.contains(&x) != m.non_uniform.contains(&y)
                    || m.measured_roundings.contains(&x) != m.measured_roundings.contains(&y)
                    || m.stored_texels.contains(&x) != m.stored_texels.contains(&y)
                {
                    return Ok(None);
                }
                if let Some(c) = cmp.corr.get_mut(x.index()) {
                    *c = y.0;
                }
            }
        }
    }
    // Where each variable is used.
    if uses.is_none() {
        *uses = Some(var_uses(m, (func, sp, canon), meter, uses_ledger)?);
    }
    for u in uses.as_deref().unwrap_or(&[]) {
        let h =
            u.at.and_then(|(lp, part)| g.chunk_at(lp, part))
                .unwrap_or(SHARED);
        let e = cmp.home.entry(u.var).or_insert(h);
        if *e != h {
            *e = SHARED;
        }
    }
    // Each region compared with the first.
    let headers: HashMap<BlockId, &LoopMeta> = m.loops.iter().map(|l| (l.header, l)).collect();
    let at_use: BTreeSet<BlockId> = m.switches_at_use.iter().copied().collect();
    let conts: HashSet<BlockId> = m.loops.iter().map(|l| l.cont).collect();
    for j in 1..k {
        let off = |b: BlockId, jj: usize| -> Option<usize> {
            let lp = sp.pos(b)?;
            let f = fst(jj);
            (lp >= f && lp < f + n).then(|| lp - f)
        };
        let same_target = |x: BlockId, y: BlockId| -> bool {
            matches!((off(x, j), off(y, 0)), (Some(a), Some(b)) if a == b)
        };
        for t in 0..n {
            let (idj, id0) = (sp.at(fst(j) + t)?, sp.at(fst(0) + t)?);
            let (bj, b0) = (block(m, idj)?, block(m, id0)?);
            meter.tick(u32::try_from(b0.insts.len()).unwrap_or(u32::MAX))?;
            if t > 0 {
                for (pi, (x, y)) in bj.phis.iter().zip(&b0.phis).enumerate() {
                    if x.incoming.len() != y.incoming.len() {
                        return Ok(None);
                    }
                    for (ii, ((vx, bx), (vy, by))) in x.incoming.iter().zip(&y.incoming).enumerate()
                    {
                        let site = Site::Phi {
                            block: id0,
                            phi: pi,
                            incoming: ii,
                        };
                        if !same_target(*bx, *by) || !cmp.operand(j, *vx, *vy, site, true, false) {
                            return Ok(None);
                        }
                    }
                }
            }
            let (xs, ys) = (
                region_insts(bj, g.insts(j, t, bj.insts.len())),
                region_insts(b0, g.insts(0, t, b0.insts.len())),
            );
            let order = aligned
                .get((j - 1) * n + t)
                .ok_or_else(|| invariant("alignment"))?;
            for &(xi, yi) in order {
                let (Some((_, x)), Some((iy, y))) = (xs.get(xi), ys.get(yi)) else {
                    return Err(invariant("aligned instruction"));
                };
                if !cmp.inst(j, x, y, (id0, *iy)) {
                    return Ok(None);
                }
            }
            if t + 1 == n {
                continue;
            }
            // The block's construct and terminator are the region's.
            if at_use.contains(&idj) != at_use.contains(&id0) {
                return Ok(None);
            }
            match (bj.merge, b0.merge) {
                (None, None) => {}
                (Some(Merge::Selection { merge: x }), Some(Merge::Selection { merge: y })) => {
                    if !same_target(x, y) {
                        return Ok(None);
                    }
                }
                (
                    Some(Merge::Loop { merge: x, cont: cx }),
                    Some(Merge::Loop { merge: y, cont: cy }),
                ) => {
                    if !same_target(x, y) || !same_target(cx, cy) {
                        return Ok(None);
                    }
                    let (Some(lx), Some(ly)) = (headers.get(&idj), headers.get(&id0)) else {
                        return Ok(None);
                    };
                    if !same_target(lx.preheader, ly.preheader) {
                        return Ok(None);
                    }
                    match (lx.exit_sel, ly.exit_sel) {
                        (None, None) => {}
                        (Some(sx), Some(sy)) => {
                            if sx != sy && !cmp.pair_var(j, sx, sy) {
                                return Ok(None);
                            }
                        }
                        _ => return Ok(None),
                    }
                }
                _ => return Ok(None),
            }
            if std::mem::discriminant(&bj.term) != std::mem::discriminant(&b0.term) {
                return Ok(None);
            }
            if matches!(b0.term, Terminator::EmitMeshTasks { .. }) {
                return Ok(None);
            }
            if let (Terminator::Switch { cases: cx, .. }, Terminator::Switch { cases: cy, .. }) =
                (&bj.term, &b0.term)
            {
                if cx.len() != cy.len() || cx.iter().zip(cy).any(|(a, b)| a.0 != b.0) {
                    return Ok(None);
                }
            }
            let (mut tx, mut ty) = (Vec::new(), Vec::new());
            let mut xt = bj.term.clone();
            xt.for_each_target_mut(|b| tx.push(*b));
            let mut yt = b0.term.clone();
            yt.for_each_target_mut(|b| ty.push(*b));
            if tx.len() != ty.len() || tx.iter().zip(&ty).any(|(x, y)| !same_target(*x, *y)) {
                return Ok(None);
            }
            let (mut vx, mut vy) = (Vec::new(), Vec::new());
            bj.term.for_each_value(|v| vx.push(v));
            b0.term.for_each_value(|v| vy.push(v));
            if vx.len() != vy.len() {
                return Ok(None);
            }
            for (operand, (x, y)) in vx.iter().zip(&vy).enumerate() {
                let site = Site::Term {
                    block: id0,
                    operand,
                };
                if !cmp.operand(j, *x, *y, site, true, false) {
                    return Ok(None);
                }
            }
        }
    }
    // Values of a region read after the run: the last region's are the first's as the loop
    // leaves them; another region's are kept, by trip, in an array the loop stores them to.
    let mut escapes = HashMap::default();
    let mut carries: Vec<(Value, Vec<usize>)> = Vec::new();
    let mut carry_at: HashMap<Value, usize> = HashMap::default();
    let mut carried_reads: Vec<(Value, Value, usize)> = Vec::new();
    let last = fst(k);
    for lp in last..sp.layout.len() {
        let b = block(m, sp.at(lp)?)?;
        let mut vals = Vec::new();
        if lp > last {
            for p in &b.phis {
                vals.extend(p.incoming.iter().map(|(v, _)| *v));
            }
        }
        let from = if lp == last {
            g.cut.get(k).copied().unwrap_or(0)
        } else {
            0
        };
        for node in b.insts.get(from..).unwrap_or(&[]) {
            node.inst.for_each_value(|v| vals.push(v));
        }
        b.term.for_each_value(|v| vals.push(v));
        for v in vals {
            // A copy reads its source; one inside a region that leaves reads it directly.
            let c = cmp.canon(v);
            let d = cmp.def(c);
            if d == 0 {
                if c != v && cmp.def(v) > 1 {
                    escapes.insert(v, c);
                }
                continue;
            }
            let j = d - 1;
            let Some(v0) = (if j == 0 { Some(c) } else { cmp.corr(c) }) else {
                return Ok(None);
            };
            if j + 1 == k {
                escapes.insert(v, v0);
                continue;
            }
            let carried = m.types.facts(m.value(v0)?.ty).is_ok_and(|f| {
                matches!(f.shape, Shape::Plain | Shape::Implicit) && !f.runtime_sized
            });
            if !carried {
                return Ok(None);
            }
            let i = *carry_at.entry(v0).or_insert_with(|| {
                carries.push((v0, Vec::new()));
                carries.len() - 1
            });
            if let Some((_, trips)) = carries.get_mut(i) {
                if !trips.contains(&j) {
                    trips.push(j);
                }
            }
            carried_reads.push((v, v0, j));
        }
    }
    // Each image that differs by trip is loaded once and used only by image instructions.
    let mut handles = Vec::new();
    for (site, globals) in &cmp.handles {
        let Site::Inst {
            block: lb, index, ..
        } = *site
        else {
            return Ok(None);
        };
        let load = block(m, lb)?
            .insts
            .get(index)
            .ok_or_else(|| invariant("handle load"))?;
        let (Some(result), Inst::Load { access, .. }) = (load.result, &load.inst) else {
            return Ok(None);
        };
        if m.non_uniform.contains(&result) {
            return Ok(None);
        }
        let mut uses = Vec::new();
        for t in 0..n {
            let id0 = sp.at(fst(0) + t)?;
            let b0 = block(m, id0)?;
            let mut bad = false;
            if t > 0 {
                for p in &b0.phis {
                    bad |= p.incoming.iter().any(|(v, _)| *v == result);
                }
            }
            if t + 1 < n {
                b0.term.for_each_value(|v| bad |= v == result);
            }
            let r0 = g.insts(0, t, b0.insts.len());
            let start = r0.start;
            for (i, node) in b0.insts.get(r0).unwrap_or(&[]).iter().enumerate() {
                let mut count = (0usize, false);
                let mut operand = 0usize;
                node.inst.for_each_value(|v| {
                    if v == result {
                        count.0 += 1;
                        count.1 |= operand == 0;
                    }
                    operand += 1;
                });
                match count {
                    (0, _) => {}
                    (1, true) if image_use(&node.inst) && !conts.contains(&id0) => {
                        uses.push((id0, start + i));
                    }
                    _ => bad = true,
                }
            }
            if bad {
                return Ok(None);
            }
        }
        handles.push(Handle {
            load: (lb, index),
            result,
            access: *access,
            globals: globals.clone(),
            uses,
        });
    }
    // An instruction with a constant that differs by trip is split as a handle's use is: not
    // in a continue block.
    if cmp.switched.iter().any(|(site, _)| match site {
        Site::Inst { block, .. } => conts.contains(block),
        _ => true,
    }) {
        return Ok(None);
    }
    // A chain reads the previous region's value in every region but the first.
    if cmp
        .chains
        .iter()
        .any(|c| c.trips.iter().skip(1).any(|t| !t))
    {
        return Ok(None);
    }
    let mut drop_vars: Vec<VarId> = cmp.var_to.keys().copied().collect();
    drop_vars.sort_unstable();
    let (s, e, u) = (sp.at(fst(0))?, sp.at(fst(1))?, sp.at(fst(k))?);
    Ok(Some(Plan {
        func,
        k,
        s,
        a: g.cut.first().copied().unwrap_or(0),
        e,
        e_cut: g.cut.get(1).copied().unwrap_or(0),
        u,
        u_cut: g.cut.get(k).copied().unwrap_or(0),
        s_pos: fst(0),
        e_pos: fst(1),
        u_pos: fst(k),
        sites: cmp.sites,
        chains: cmp.chains,
        lifts: cmp.lifts,
        hoists,
        handles,
        switched: cmp.switched,
        drop_vars,
        reinit: cmp.reinit,
        escapes,
        carries,
        carried: carried_reads,
    }))
}

/// Intern type `t`, declaring it when new.
fn ty(m: &mut TypedModule, t: Type) -> Result<TypeId, Error> {
    let (id, fresh) = m.types.intern(t, &m.consts)?;
    if fresh {
        m.decls.push(Decl::Type(id));
    }
    Ok(id)
}

fn new_value(m: &mut TypedModule, ty: TypeId, origin: ValueOrigin) -> Result<Value, Error> {
    let v = Value::new(u32::try_from(m.values.len()).map_err(|_| invariant("value ids"))?);
    m.values.push(ValueDef { ty, origin });
    Ok(v)
}

/// Plain constant `c`, declared when new.
fn constant_id(m: &mut TypedModule, c: Constant) -> Result<ConstId, Error> {
    let ty = c.ty();
    let (id, fresh) = m.consts.intern(c, &m.types)?;
    if fresh {
        if id.index() != m.const_values.len() {
            return Err(invariant("constant ids"));
        }
        let v = new_value(m, ty, ValueOrigin::Const(id))?;
        m.const_values.push(v);
        m.decls.push(Decl::Const(id));
    }
    Ok(id)
}

/// The value of plain constant `c`, declaring it when new.
fn constant(m: &mut TypedModule, c: Constant) -> Result<Value, Error> {
    let id = constant_id(m, c)?;
    Ok(m.const_value(id)?)
}

/// A new `Function` variable of the entry holding `pointee`; its pointer.
fn local(m: &mut TypedModule, entry: FuncId, pointee: TypeId) -> Result<Value, Error> {
    let ptr_ty = ty(
        m,
        Type::Pointer {
            class: StorageClass::Function,
            pointee,
        },
    )?;
    let var = VarId::new(u32::try_from(m.vars.len()).map_err(|_| invariant("variable ids"))?);
    let value = new_value(m, ptr_ty, ValueOrigin::Var(var))?;
    m.vars.push(LocalVar {
        func: entry,
        pointee,
        ptr_ty,
        value,
        init: None,
    });
    m.functions
        .get_mut(entry.index())
        .ok_or_else(|| invariant("function"))?
        .vars
        .push(var);
    Ok(value)
}

fn new_block(m: &mut TypedModule, func: FuncId) -> Result<BlockId, Error> {
    let id = BlockId::new(u32::try_from(m.blocks.len()).map_err(|_| invariant("block ids"))?);
    m.blocks.push(Block {
        func,
        phis: Vec::new(),
        insts: Vec::new(),
        merge: None,
        term: Terminator::Unreachable,
    });
    Ok(id)
}

/// Append an instruction with a result of type `ty` to `b`.
fn push(m: &mut TypedModule, b: BlockId, ty: TypeId, inst: Inst) -> Result<Value, Error> {
    let r = new_value(m, ty, ValueOrigin::Inst(b))?;
    block_mut(m, b)?.insts.push(InstNode {
        result: Some(r),
        inst,
    });
    Ok(r)
}

fn push_void(m: &mut TypedModule, b: BlockId, inst: Inst) -> Result<(), Error> {
    block_mut(m, b)?.insts.push(InstNode { result: None, inst });
    Ok(())
}

/// Mark `results` (of instructions and phis that left their block) removed.
fn removed(m: &mut TypedModule, results: impl Iterator<Item = Value>) {
    for r in results {
        if let Some(d) = m.values.get_mut(r.index()) {
            d.origin = ValueOrigin::Removed;
        }
    }
}

/// Move the merge and terminator of `from` to `to` (a new block that takes over `from`'s
/// edges): the successors' phis, a loop's preheader, and a switch-at-use header follow.
fn move_tail(m: &mut TypedModule, from: BlockId, to: BlockId) -> Result<(), Error> {
    let (merge, term) = {
        let b = block(m, from)?;
        (b.merge, b.term.clone())
    };
    let mut targets = Vec::new();
    let mut t = term.clone();
    t.for_each_target_mut(|x| targets.push(*x));
    {
        let b = block_mut(m, to)?;
        b.merge = merge;
        b.term = term;
    }
    for t in targets {
        for p in &mut block_mut(m, t)?.phis {
            for (_, b) in &mut p.incoming {
                if *b == from {
                    *b = to;
                }
            }
        }
    }
    for l in &mut m.loops {
        if l.preheader == from {
            l.preheader = to;
        }
    }
    for h in &mut m.switches_at_use {
        if *h == from {
            *h = to;
        }
    }
    Ok(())
}

/// The start `c` and step `d` of a site's trip values (`values[t]` is trip `t`'s) when each is
/// a plain `u32` constant and trip `t`'s is `c + t * d` in wrapping arithmetic; `None` when any
/// is another value (a specialization constant among them) or they do not so progress.
fn progression(
    m: &TypedModule,
    values: &[Value],
    u32_: TypeId,
) -> Result<Option<(u32, u32)>, Error> {
    let mut words = Vec::with_capacity(values.len());
    for v in values {
        let def = m.value(*v)?;
        let ValueOrigin::Const(id) = def.origin else {
            return Ok(None);
        };
        let word = match m.consts.scalar_bits(id) {
            Some(ConstBits(b)) if def.ty == u32_ => u32::try_from(b).ok(),
            _ => None,
        };
        let Some(word) = word else {
            return Ok(None);
        };
        words.push(word);
    }
    let (Some(&c0), Some(&c1)) = (words.first(), words.get(1)) else {
        return Ok(None);
    };
    let d = c1.wrapping_sub(c0);
    let mut at = c0;
    for &w in &words {
        if w != at {
            return Ok(None);
        }
        at = at.wrapping_add(d);
    }
    Ok(Some((c0, d)))
}

/// `c + iv * d` (`iv` the trip index), a `u32`, appended to `b`.
fn stride_value(
    m: &mut TypedModule,
    b: BlockId,
    u32_: TypeId,
    iv: Value,
    (c0, d): (u32, u32),
) -> Result<Value, Error> {
    let word = |m: &mut TypedModule, x: u32| {
        constant(
            m,
            Constant::Scalar {
                ty: u32_,
                bits: ConstBits(u64::from(x)),
            },
        )
    };
    let mut at = iv;
    for (op, x, skip) in [(BinOp::IMul, d, 1), (BinOp::IAdd, c0, 0)] {
        if x == skip {
            continue;
        }
        let kx = word(m, x)?;
        at = push(
            m,
            b,
            u32_,
            Inst::Binary {
                op,
                a: at,
                b: kx,
                contract: Contraction::Allowed,
            },
        )?;
    }
    Ok(at)
}

/// Write the run of `plan` as a loop.
fn apply(m: &mut TypedModule, plan: &Plan) -> Result<(), Error> {
    let k = plan.k;
    let entry = plan.func;
    let u32_ = ty(
        m,
        Type::Int {
            width: 32,
            signed: false,
        },
    )?;
    let bool_ = ty(m, Type::Bool)?;
    let mut trip = Vec::with_capacity(k + 1);
    for t in 0..=k {
        trip.push(constant(
            m,
            Constant::Scalar {
                ty: u32_,
                bits: ConstBits(t as u64),
            },
        )?);
    }
    let c = |t: usize| {
        trip.get(t)
            .copied()
            .ok_or_else(|| invariant("trip constant"))
    };
    // The trip index.
    let iptr = local(m, entry, u32_)?;
    let (h, b0, brk, x, cont, mg) = (
        new_block(m, entry)?,
        new_block(m, entry)?,
        new_block(m, entry)?,
        new_block(m, entry)?,
        new_block(m, entry)?,
        new_block(m, entry)?,
    );
    // The loop body's first block: the trip index, each differing value's trip value, the
    // variables' initializers, then the first region's instructions in its first block.
    let iv = push(
        m,
        b0,
        u32_,
        Inst::Load {
            ptr: iptr,
            access: MemAccess::default(),
        },
    )?;
    let mut eq = Vec::new();
    if !plan.sites.is_empty() || !plan.lifts.is_empty() {
        for t in 0..k - 1 {
            eq.push(push(
                m,
                b0,
                bool_,
                Inst::Binary {
                    op: BinOp::IEqual,
                    a: iv,
                    b: c(t)?,
                    contract: Contraction::Allowed,
                },
            )?);
        }
    }
    // Each lifted lane read: every trip's read before the loop (in the preheader, after what
    // precedes the run), its values then chosen as a differing operand's are.
    // Before them, the instructions of other regions that float on values from before the run, in
    // order, each a new result; what read the old one reads the new.
    let mut hoist = Vec::with_capacity(plan.hoists.len());
    let mut moved: HashMap<Value, Value> =
        HashMap::with_capacity_and_hasher(plan.hoists.len(), BuildHasherDefault::default());
    let remap = |moved: &HashMap<Value, Value>, v: &mut Value| {
        if let Some(n) = moved.get(v) {
            *v = *n;
        }
    };
    for (orig, node) in &plan.hoists {
        let mut inst = node.inst.clone();
        inst.for_each_value_mut(|v| remap(&moved, v));
        let ty = m.value(*orig)?.ty;
        let r = new_value(m, ty, ValueOrigin::Inst(plan.s))?;
        hoist.push(InstNode {
            result: Some(r),
            inst,
        });
        moved.insert(*orig, r);
    }
    let mut lifted: Vec<((BlockId, usize), Vec<Value>)> = Vec::with_capacity(plan.lifts.len());
    for lift in &plan.lifts {
        let mut values = Vec::with_capacity(k);
        for inst in &lift.insts {
            let mut inst = inst.clone();
            inst.for_each_value_mut(|v| remap(&moved, v));
            let r = new_value(m, lift.ty, ValueOrigin::Inst(plan.s))?;
            hoist.push(InstNode {
                result: Some(r),
                inst,
            });
            values.push(r);
        }
        lifted.push((lift.at, values));
    }
    let sites: Vec<(Site, Vec<Value>)> = plan
        .sites
        .iter()
        .map(|(site, values)| {
            let mut values = values.clone();
            values.iter_mut().for_each(|v| remap(&moved, v));
            (*site, values)
        })
        .collect();
    let hoisted = hoist.len();
    {
        let sb = block_mut(m, plan.s)?;
        let at = plan.a.min(sb.insts.len());
        sb.insts.splice(at..at, hoist);
    }
    let mut chosen: HashMap<Vec<Value>, Value> = HashMap::default();
    for values in sites
        .iter()
        .map(|(_, values)| values)
        .chain(lifted.iter().map(|(_, values)| values))
    {
        if chosen.contains_key(values) {
            continue;
        }
        let first = *values.first().ok_or_else(|| invariant("site values"))?;
        let vty = m.value(first)?.ty;
        if let Some((c0, d)) = progression(m, values, u32_)? {
            let at = stride_value(m, b0, u32_, iv, (c0, d))?;
            chosen.insert(values.clone(), at);
            continue;
        }
        let mut acc = *values.last().ok_or_else(|| invariant("site values"))?;
        for t in (0..k - 1).rev() {
            let vt = *values.get(t).ok_or_else(|| invariant("site values"))?;
            if vt == acc {
                continue;
            }
            let cond = *eq.get(t).ok_or_else(|| invariant("trip test"))?;
            acc = push(
                m,
                b0,
                vty,
                Inst::Select {
                    cond,
                    t: vt,
                    f: acc,
                },
            )?;
        }
        chosen.insert(values.clone(), acc);
    }
    // Each chain's variable, loaded at the top of every trip.
    let mut chained: HashMap<(Value, Value), (Value, Value)> = HashMap::default();
    for ch in &plan.chains {
        if chained.contains_key(&(ch.init, ch.carried)) {
            continue;
        }
        let vty = m.value(ch.init)?.ty;
        let var = local(m, entry, vty)?;
        let held = push(
            m,
            b0,
            vty,
            Inst::Load {
                ptr: var,
                access: MemAccess::default(),
            },
        )?;
        chained.insert((ch.init, ch.carried), (var, held));
    }
    for (p, init) in &plan.reinit {
        push_void(
            m,
            b0,
            Inst::Store {
                ptr: *p,
                value: *init,
                access: MemAccess::default(),
            },
        )?;
    }
    let pre = block(m, b0)?.insts.len();
    // The preheader keeps what precedes the run, and starts the trip index at 0.
    let tail = block_mut(m, plan.s)?.insts.split_off(plan.a + hoisted);
    for r in tail.iter().filter_map(|n| n.result) {
        if let Some(d) = m.values.get_mut(r.index()) {
            d.origin = ValueOrigin::Inst(b0);
        }
    }
    block_mut(m, b0)?.insts.extend(tail);
    move_tail(m, plan.s, b0)?;
    {
        let zero = c(0)?;
        let s = block_mut(m, plan.s)?;
        s.merge = None;
        s.term = Terminator::Branch(h);
        s.insts.push(InstNode {
            result: None,
            inst: Inst::Store {
                ptr: iptr,
                value: zero,
                access: MemAccess::default(),
            },
        });
    }
    // Each chain starts at the first region's operand.
    let mut starts: Vec<(Value, Value)> = chained
        .iter()
        .map(|((init, _), (var, _))| (*var, *init))
        .collect();
    starts.sort_unstable();
    for (var, init) in starts {
        push_void(
            m,
            plan.s,
            Inst::Store {
                ptr: var,
                value: init,
                access: MemAccess::default(),
            },
        )?;
    }
    {
        let hb = block_mut(m, h)?;
        hb.merge = Some(Merge::Loop { merge: mg, cont });
        hb.term = Terminator::Branch(b0);
    }
    // The first region's last block keeps its part, then counts the trip and leaves after the
    // last.
    let latch_gone = block_mut(m, plan.e)?.insts.split_off(plan.e_cut);
    removed(m, latch_gone.iter().filter_map(|n| n.result));
    // Each chain's next trip reads this trip's value.
    let mut nexts: Vec<(Value, Value)> = chained
        .iter()
        .map(|((_, carried), (var, _))| (*var, *carried))
        .collect();
    nexts.sort_unstable();
    for (var, carried) in nexts {
        push_void(
            m,
            plan.e,
            Inst::Store {
                ptr: var,
                value: carried,
                access: MemAccess::default(),
            },
        )?;
    }
    // Each value another region leaves for after the run is stored, by trip, to an array, and
    // read back after the loop.
    // A hoisted value read after the run reads its new result.
    let mut escapes = plan.escapes.clone();
    for v in escapes.values_mut() {
        remap(&moved, v);
    }
    for (orig, r) in &moved {
        escapes.entry(*orig).or_insert(*r);
    }
    let len = constant_id(
        m,
        Constant::Scalar {
            ty: u32_,
            bits: ConstBits(k as u64),
        },
    )?;
    let mut reads: Vec<(Value, TypeId, Value, Vec<usize>)> = Vec::new();
    for (v0, trips) in &plan.carries {
        let elem = m.value(*v0)?.ty;
        let arr = ty(
            m,
            Type::Array {
                elem,
                len,
                stride: None,
            },
        )?;
        let base = local(m, entry, arr)?;
        let pel = ty(
            m,
            Type::Pointer {
                class: StorageClass::Function,
                pointee: elem,
            },
        )?;
        let at = push(
            m,
            plan.e,
            pel,
            Inst::AccessChain {
                base,
                indices: [iv].into(),
                non_uniform: false,
            },
        )?;
        push_void(
            m,
            plan.e,
            Inst::Store {
                ptr: at,
                value: *v0,
                access: MemAccess::default(),
            },
        )?;
        reads.push((*v0, pel, base, trips.clone()));
    }
    let next = push(
        m,
        plan.e,
        u32_,
        Inst::Binary {
            op: BinOp::IAdd,
            a: iv,
            b: c(1)?,
            contract: Contraction::Allowed,
        },
    )?;
    push_void(
        m,
        plan.e,
        Inst::Store {
            ptr: iptr,
            value: next,
            access: MemAccess::default(),
        },
    )?;
    let done = push(
        m,
        plan.e,
        bool_,
        Inst::Binary {
            op: BinOp::IEqual,
            a: next,
            b: c(k)?,
            contract: Contraction::Allowed,
        },
    )?;
    {
        let eb = block_mut(m, plan.e)?;
        eb.merge = Some(Merge::Selection { merge: x });
        eb.term = Terminator::CondBranch {
            cond: done,
            t: brk,
            f: x,
        };
    }
    block_mut(m, brk)?.term = Terminator::Branch(mg);
    block_mut(m, x)?.term = Terminator::Branch(cont);
    block_mut(m, cont)?.term = Terminator::Branch(h);
    block_mut(m, mg)?.term = Terminator::Branch(plan.u);
    let mut loaded: HashMap<(Value, usize), Value> = HashMap::default();
    for (v0, pel, base, trips) in reads {
        let elem = match m.types.get(pel)? {
            Type::Pointer { pointee, .. } => *pointee,
            _ => return Err(invariant("carry pointer")),
        };
        for j in trips {
            let at = push(
                m,
                mg,
                pel,
                Inst::AccessChain {
                    base,
                    indices: [c(j)?].into(),
                    non_uniform: false,
                },
            )?;
            let read = push(
                m,
                mg,
                elem,
                Inst::Load {
                    ptr: at,
                    access: MemAccess::default(),
                },
            )?;
            loaded.insert((v0, j), read);
        }
    }
    for (orig, v0, j) in &plan.carried {
        let read = *loaded
            .get(&(*v0, *j))
            .ok_or_else(|| invariant("carried value"))?;
        escapes.insert(*orig, read);
    }
    // The block after the run loses the last region's part.
    let gone: Vec<Value> = {
        let ub = block(m, plan.u)?;
        let cut = plan.u_cut.min(ub.insts.len());
        ub.phis
            .iter()
            .map(|p| p.result)
            .chain(
                ub.insts
                    .get(..cut)
                    .unwrap_or(&[])
                    .iter()
                    .filter_map(|n| n.result),
            )
            .collect()
    };
    {
        let ub = block_mut(m, plan.u)?;
        let cut = plan.u_cut.min(ub.insts.len());
        ub.phis.clear();
        ub.insts.drain(..cut);
    }
    removed(m, gone.into_iter());
    m.loops.push(LoopMeta {
        header: h,
        merge: mg,
        cont,
        preheader: plan.s,
        exit_sel: None,
        // Its trips are the regions it compiles once: unrolled, they are written out again.
        dont_unroll: true,
    });
    // Where the first region's operands are now.
    let moved = Moved {
        s: plan.s,
        a: plan.a,
        b0,
        pre,
    };
    for (site, values) in &sites {
        let v = *chosen.get(values).ok_or_else(|| invariant("site value"))?;
        set_operand(m, *site, v, moved)?;
    }
    for ch in &plan.chains {
        let (_, held) = *chained
            .get(&(ch.init, ch.carried))
            .ok_or_else(|| invariant("chain"))?;
        set_operand(m, ch.site, held, moved)?;
    }
    for (at, values) in &lifted {
        let v = *chosen
            .get(values)
            .ok_or_else(|| invariant("lifted value"))?;
        let (b, i) = moved.inst(at.0, at.1);
        let node = block_mut(m, b)?
            .insts
            .get_mut(i)
            .ok_or_else(|| invariant("lifted read"))?;
        node.inst = Inst::Unary {
            op: crate::spirv::ir::UnOp::CopyObject,
            a: v,
        };
    }
    // The last region's values read after the run are the first's.
    let layout = m
        .functions
        .get(entry.index())
        .ok_or_else(|| invariant("function"))?
        .blocks
        .clone();
    if !escapes.is_empty() {
        for b in layout.get(plan.u_pos..).unwrap_or(&[]) {
            let bl = block_mut(m, *b)?;
            for p in &mut bl.phis {
                for (v, _) in &mut p.incoming {
                    if let Some(c) = escapes.get(v) {
                        *v = *c;
                    }
                }
            }
            for node in &mut bl.insts {
                node.inst.for_each_value_mut(|v| {
                    if let Some(c) = escapes.get(v) {
                        *v = *c;
                    }
                });
            }
            bl.term.for_each_value_mut(|v| {
                if let Some(c) = escapes.get(v) {
                    *v = *c;
                }
            });
        }
    }
    // The new layout: the preheader, the header, the first region with its latch, the loop's
    // merge, and what follows the run.
    let mut order: Vec<BlockId> = Vec::with_capacity(layout.len() + 6);
    order.extend(layout.get(..=plan.s_pos).unwrap_or(&[]));
    order.extend([h, b0]);
    order.extend(layout.get(plan.s_pos + 1..=plan.e_pos).unwrap_or(&[]));
    order.extend([brk, x, cont, mg]);
    order.extend(layout.get(plan.u_pos..).unwrap_or(&[]));
    m.functions
        .get_mut(entry.index())
        .ok_or_else(|| invariant("function"))?
        .blocks = order;
    // Each image that differs by trip: the load leaves, and each use loads its trip's; each
    // instruction whose constant operand differs by trip executes its trip's. Both in a switch
    // on the trip index.
    if !plan.handles.is_empty() || !plan.switched.is_empty() {
        let mut loads: Vec<(BlockId, usize)> = Vec::new();
        let mut splits: HashMap<(BlockId, usize), Split<'_>> = HashMap::default();
        for hd in &plan.handles {
            loads.push(moved.inst(hd.load.0, hd.load.1));
            for u in &hd.uses {
                splits.entry(moved.inst(u.0, u.1)).or_default().handle = Some(hd);
            }
        }
        for (site, values) in &plan.switched {
            let Site::Inst {
                block: sb,
                index,
                operand,
            } = *site
            else {
                return Err(invariant("switched operand"));
            };
            splits
                .entry(moved.inst(sb, index))
                .or_default()
                .operands
                .push((operand, values));
        }
        let mut splits: Vec<((BlockId, usize), Split<'_>)> = splits.into_iter().collect();
        loads.sort_unstable();
        for (b, i) in loads.iter().rev() {
            let node = block_mut(m, *b)?.insts.remove(*i);
            removed(m, node.result.into_iter());
            for ((ub, ui), _) in &mut splits {
                if *ub == *b && *ui > *i {
                    *ui -= 1;
                }
            }
        }
        splits.sort_unstable_by_key(|(at, _)| *at);
        let mut sampled = Vec::new();
        for ((b, i), split) in splits.into_iter().rev() {
            sampled.extend(split_by_trip(m, b, i, iv, k, &split)?);
        }
        drop_unused(m, entry, &sampled)?;
    }
    // The other regions' blocks and variables leave.
    let mut keep_block = vec![true; m.blocks.len()];
    for b in layout.get(plan.e_pos + 1..plan.u_pos).unwrap_or(&[]) {
        if let Some(x) = keep_block.get_mut(b.index()) {
            *x = false;
        }
    }
    let mut keep_var = vec![true; m.vars.len()];
    for v in &plan.drop_vars {
        if let Some(x) = keep_var.get_mut(v.index()) {
            *x = false;
        }
    }
    m.drop_ids(&[], &keep_block, &keep_var);
    Ok(())
}

/// Where the first region's code is once its run is rolled: from instruction `a` of block `s`
/// on, the instructions are in the loop's first block `b0` after its `pre` instructions, and
/// `s`'s terminator is `b0`'s.
#[derive(Clone, Copy)]
struct Moved {
    s: BlockId,
    a: usize,
    b0: BlockId,
    pre: usize,
}

impl Moved {
    /// Where instruction `i` of block `b` is now.
    fn inst(self, b: BlockId, i: usize) -> (BlockId, usize) {
        if b == self.s && i >= self.a {
            (self.b0, i - self.a + self.pre)
        } else {
            (b, i)
        }
    }

    /// The block that holds block `b`'s terminator now.
    fn term(self, b: BlockId) -> BlockId {
        if b == self.s {
            self.b0
        } else {
            b
        }
    }
}

/// Set the operand `site` names (where `moved` puts it now) to `v`.
fn set_operand(m: &mut TypedModule, site: Site, v: Value, moved: Moved) -> Result<(), Error> {
    match site {
        Site::Inst {
            block: b,
            index,
            operand,
        } => {
            let (b, index) = moved.inst(b, index);
            let node = block_mut(m, b)?
                .insts
                .get_mut(index)
                .ok_or_else(|| invariant("site instruction"))?;
            let mut i = 0usize;
            node.inst.for_each_value_mut(|x| {
                if i == operand {
                    *x = v;
                }
                i += 1;
            });
        }
        Site::Phi {
            block: b,
            phi,
            incoming,
        } => {
            let slot = block_mut(m, b)?
                .phis
                .get_mut(phi)
                .and_then(|p| p.incoming.get_mut(incoming))
                .ok_or_else(|| invariant("site phi"))?;
            slot.0 = v;
        }
        Site::Term { block: b, operand } => {
            let b = moved.term(b);
            let mut i = 0usize;
            block_mut(m, b)?.term.for_each_value_mut(|x| {
                if i == operand {
                    *x = v;
                }
                i += 1;
            });
        }
    }
    Ok(())
}

/// How the arms of an instruction split by trip differ ([`split_by_trip`]): trip `t`'s loads its
/// image from `handle`'s `globals[t]` as the instruction's operand 0, and sets each switched
/// operand to its `values[t]`.
#[derive(Default)]
struct Split<'p> {
    handle: Option<&'p Handle>,
    operands: Vec<(usize, &'p Vec<Value>)>,
}

/// Split block `b` at instruction `i`, an instruction whose image or constant operands differ by
/// trip ([`Split`]): the block ends in an `OpSwitch` on trip index `iv` with an arm per distinct
/// trip operands (the last trip's the default), each executing the instruction on its own (and
/// loading its image first), and a new block after the arms takes the instructions after it, the
/// block's merge and terminator, and (as a phi) the instruction's result. An `OpSampledImage` of
/// `b` that the instruction or one after it consumes is written again in each arm and in the new
/// block (SPIR-V: a sampled image is consumed in its own block); returns those of `b`, which may
/// now be unused ([`drop_unused`]).
fn split_by_trip(
    m: &mut TypedModule,
    b: BlockId,
    i: usize,
    iv: Value,
    k: usize,
    split: &Split<'_>,
) -> Result<Vec<Value>, Error> {
    // Each trip's arm: trips that load the same image and set the same constants share one.
    let mut arm_keys: Vec<(Option<GlobalId>, Vec<Value>)> = Vec::new();
    let mut arm_of = Vec::with_capacity(k);
    for t in 0..k {
        let global = match split.handle {
            Some(hd) => Some(
                *hd.globals
                    .get(t)
                    .ok_or_else(|| invariant("handle variable"))?,
            ),
            None => None,
        };
        let mut values = Vec::with_capacity(split.operands.len());
        for (_, vs) in &split.operands {
            values.push(*vs.get(t).ok_or_else(|| invariant("switched value"))?);
        }
        let key = (global, values);
        let at = match arm_keys.iter().position(|x| *x == key) {
            Some(at) => at,
            None => {
                arm_keys.push(key);
                arm_keys.len() - 1
            }
        };
        arm_of.push(at);
    }
    let entry = block(m, b)?.func;
    let (node, rest) = {
        let bl = block_mut(m, b)?;
        if i >= bl.insts.len() {
            return Err(invariant("split instruction"));
        }
        let rest = bl.insts.split_off(i + 1);
        let node = bl
            .insts
            .pop()
            .ok_or_else(|| invariant("split instruction"))?;
        (node, rest)
    };
    let uses = |inst: &Inst, r: Value| {
        let mut used = false;
        inst.for_each_value(|v| used |= v == r);
        used
    };
    // The sampled images of `b` the instruction or the ones after it consume.
    let mut sampled: Vec<(Value, TypeId, Inst)> = Vec::new();
    for n in &block(m, b)?.insts {
        if let (Some(r), Inst::SampledImage { .. }) = (n.result, &n.inst) {
            if uses(&node.inst, r) || rest.iter().any(|x| uses(&x.inst, r)) {
                sampled.push((r, m.value(r)?.ty, n.inst.clone()));
            }
        }
    }
    let y = new_block(m, entry)?;
    let mut head = Vec::new();
    let mut again: HashMap<Value, Value> = HashMap::default();
    for (r, ty, inst) in &sampled {
        if rest.iter().any(|x| uses(&x.inst, *r)) {
            let nr = new_value(m, *ty, ValueOrigin::Inst(y))?;
            head.push(InstNode {
                result: Some(nr),
                inst: inst.clone(),
            });
            again.insert(*r, nr);
        }
    }
    for x in rest {
        let mut x = x;
        x.inst.for_each_value_mut(|v| {
            if let Some(n) = again.get(v) {
                *v = *n;
            }
        });
        if let Some(d) = x.result.and_then(|r| m.values.get_mut(r.index())) {
            d.origin = ValueOrigin::Inst(y);
        }
        head.push(x);
    }
    block_mut(m, y)?.insts = head;
    move_tail(m, b, y)?;
    let image_ty = match split.handle {
        Some(hd) => Some(m.value(hd.result)?.ty),
        None => None,
    };
    let mut arms = Vec::with_capacity(arm_keys.len());
    let mut incoming = Vec::new();
    for (global, values) in &arm_keys {
        let arm = new_block(m, entry)?;
        let img = match (global, split.handle, image_ty) {
            (Some(g), Some(hd), Some(image_ty)) => {
                let gv = m
                    .globals
                    .get(g.index())
                    .ok_or_else(|| invariant("handle variable"))?
                    .value;
                Some(push(
                    m,
                    arm,
                    image_ty,
                    Inst::Load {
                        ptr: gv,
                        access: hd.access,
                    },
                )?)
            }
            (None, None, None) => None,
            _ => return Err(invariant("handle arm")),
        };
        let mut inst = node.inst.clone();
        let mut own: HashMap<Value, Value> = HashMap::default();
        for (r, ty, si) in &sampled {
            if uses(&inst, *r) {
                let nr = push(m, arm, *ty, si.clone())?;
                own.insert(*r, nr);
            }
        }
        let mut operand = 0usize;
        inst.for_each_value_mut(|v| {
            if let Some(n) = own.get(v) {
                *v = *n;
            }
            if let (0, Some(img)) = (operand, img) {
                *v = img;
            }
            for ((at, _), value) in split.operands.iter().zip(values) {
                if *at == operand {
                    *v = *value;
                }
            }
            operand += 1;
        });
        match node.result {
            Some(r) => {
                let rty = m.value(r)?.ty;
                let rt = push(m, arm, rty, inst)?;
                incoming.push((rt, arm));
            }
            None => push_void(m, arm, inst)?,
        }
        block_mut(m, arm)?.term = Terminator::Branch(y);
        arms.push(arm);
    }
    if let Some(r) = node.result {
        if let Some(d) = m.values.get_mut(r.index()) {
            d.origin = ValueOrigin::Phi(y);
        }
        block_mut(m, y)?.phis.push(Phi {
            result: r,
            incoming,
        });
    }
    let last = *arm_of.last().ok_or_else(|| invariant("split arms"))?;
    let default = *arms.get(last).ok_or_else(|| invariant("split arms"))?;
    let mut lits: Vec<(u64, BlockId)> = Vec::new();
    for (t, a) in arm_of.iter().enumerate() {
        if *a != last {
            lits.push((
                t as u64,
                *arms.get(*a).ok_or_else(|| invariant("split arms"))?,
            ));
        }
    }
    {
        let bl = block_mut(m, b)?;
        bl.merge = Some(Merge::Selection { merge: y });
        bl.term = Terminator::Switch {
            sel: iv,
            default,
            cases: lits,
        };
    }
    let f = m
        .functions
        .get_mut(entry.index())
        .ok_or_else(|| invariant("function"))?;
    let at = f
        .blocks
        .iter()
        .position(|x| *x == b)
        .ok_or_else(|| invariant("split block"))?;
    let mut tail_order: Vec<BlockId> = arms;
    tail_order.push(y);
    let after = f.blocks.split_off(at + 1);
    f.blocks.extend(tail_order);
    f.blocks.extend(after);
    Ok(sampled.into_iter().map(|(r, _, _)| r).collect())
}

/// Remove each instruction of function `func` whose result is one of `values` and is no longer
/// used: sampled images [`split_by_trip`] wrote again where they are consumed.
fn drop_unused(m: &mut TypedModule, func: FuncId, values: &[Value]) -> Result<(), Error> {
    if values.is_empty() {
        return Ok(());
    }
    let blocks = m
        .functions
        .get(func.index())
        .ok_or_else(|| invariant("function"))?
        .blocks
        .clone();
    let mut used: HashSet<Value> = HashSet::default();
    for b in &blocks {
        let bl = block(m, *b)?;
        for n in &bl.insts {
            n.inst.for_each_value(|v| {
                used.insert(v);
            });
        }
        for p in &bl.phis {
            used.extend(p.incoming.iter().map(|(v, _)| *v));
        }
        bl.term.for_each_value(|v| {
            used.insert(v);
        });
    }
    let dead: HashSet<Value> = values
        .iter()
        .copied()
        .filter(|v| !used.contains(v))
        .collect();
    if dead.is_empty() {
        return Ok(());
    }
    let mut gone = Vec::new();
    for b in &blocks {
        block_mut(m, *b)?.insts.retain(|n| match n.result {
            Some(r) if dead.contains(&r) => {
                gone.push(r);
                false
            }
            _ => true,
        });
    }
    removed(m, gone.into_iter());
    Ok(())
}
