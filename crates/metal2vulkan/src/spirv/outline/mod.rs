//! Repeated regions written once, as a kept routine each region's place calls.
//!
//! The AIR inliner (`air::canon::inline`) writes each AIR call's body in line, so a function a
//! kernel calls from several places is in its entry several times. A driver that honours a call's
//! `DontInline` (NVIDIA's) compiles a kept routine once and calls it; one that inlines every call
//! (Mesa's RADV and ANV) compiles the same code either way. On the RTX 3090 (NVIDIA 620.70.06, job
//! m3-max-64453), a generated kernel calling one function of 1,984 instructions from 50 places
//! was created in 21.5 s with the function written in line at each and in 52 ms with it a kept
//! routine; with 64 values live across every call, 51.8 s and 75 ms. Its dispatches took the same
//! GPU time either way with few values live, and 20.6% less as a routine with 64 live; a function
//! of 497 instructions measured the same (0.1% less, and 19.7% less with 64 live).
//!
//! On every profile, the pass finds a set of regions of one function that are the same code, each
//! of at least [`MIN_REGION`] instructions (`roll::entry_weight`'s count, a call of a kept routine
//! counting one), writes the first as a new kept routine (`Function::kept`, the form of
//! `ModuleBuilder::kept_routine`), and replaces each region with a call of it; then it looks again,
//! the routines' own bodies included, at most [`MAX_ROUTINES`] times. It runs after `spirv::roll`,
//! which writes a run of consecutive regions above its size once as a loop.
//!
//! A region is a run of items of one sequence of the function's spine (`roll::spine`, the bodies of
//! loops included): its first block from an instruction on, the constructs it holds, and its last
//! block up to an instruction. It is entered only at its start and left only at its end: a branch
//! out of it (a loop's break or continue), a return, a kill, or a branch into it from outside keeps
//! it in line. Regions are compared exactly, block by block and instruction by instruction, each
//! value one defines mapped to the first region's at the same place, and each branch target to the
//! first region's block at the same place:
//!
//! - a value defined before the region is a parameter of the routine, one per distinct set of the
//!   regions' values at a place; where every region reads the same constant or module-scope
//!   variable, the routine reads it too. A constant that differs between regions is a parameter
//!   only in an operand any value may take (`roll::varying_ok`, or an index into an array, vector
//!   or matrix of data), never in one that must be a constant;
//! - a pointer parameter is a `Function`, `Private`, `Workgroup` or `UniformConstant` variable
//!   itself, or a `PhysicalStorageBuffer` pointer, and never an atomic's pointer; any other pointer
//!   from before the region (an access chain), an image, a sampler, or a value decorated
//!   `NonUniform` keeps the regions in line;
//! - a `Function` variable only its region uses becomes the routine's own where the region is in
//!   no loop of its function, or where its first use stores the whole variable in a block the
//!   region always runs (so no value of it carries from one run of the region to the next);
//!   otherwise it is a pointer parameter;
//! - a value of a region read after it is returned, several in a struct, and each place reads it
//!   from the call's result; a pointer or an opaque value read after it keeps the regions in line.
//!
//! What differs between regions in any other way keeps them in line. Candidate sets come from the
//! spine's keys: windows of [`WINDOW`] items whose keys hash alike seed a pair of regions, extended
//! both ways while the keys agree against the window's first [`MAX_OCCURRENCES`] repeats, and
//! the set is every occurrence of the window in the function that shares the extent with the
//! largest saving (ranges of keys compared by prefix hashes), so a region repeated a thousand
//! times is one routine from one search; a set whose comparison fails at an operand or a result
//! is compared again without the item that holds it. Work is linear in the function per search,
//! bounded in functions, extents, extension steps and comparisons, and metered on the
//! per-attempt [`Meter`].

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::ops::Range;

use crate::budget::Meter;
use crate::error::{Error, InternalError, InternalKind};
use crate::spirv::ids::{BlockId, FuncId, TypeId, Value, VarId};
use crate::spirv::ir::{
    Block, Function, Inst, InstNode, Merge, Shape, StorageClass, Terminator, Type, TypedModule,
    Unrolling, ValueOrigin,
};
use crate::spirv::roll::{
    block, block_mut, charge, move_tail, new_block, new_value, removed, rollable, spine,
    switched_operand, ty, varying_ok, weights, HashMap, HashSet, Item, KeyCache, Spine, TAIL,
};

/// The smallest region written as a routine, in instructions: the smallest function measured to
/// run no slower as a kept routine than written in line on NVIDIA's driver (the module docs: 497
/// instructions).
pub const MIN_REGION: u64 = 500;

/// The most routines one module is given.
pub const MAX_ROUTINES: u32 = 64;
/// The items a seed window holds.
const WINDOW: usize = 16;
/// The repeats of a seed window whose extents beside its first are tried.
const MAX_OCCURRENCES: usize = 64;
/// The exact comparisons per search.
const MAX_VERIFIES: usize = 32;
/// The times one candidate is compared again without the item where its comparison failed.
const MAX_TRIMS: usize = 4;
/// The most functions searched per search: the heaviest the entry reaches.
const MAX_FUNCS: usize = 8;
/// The most parameters a routine takes (SPIR-V's universal limit on a function's parameters).
const MAX_PARAMS: usize = 255;
/// The most values a routine returns.
const MAX_RESULTS: usize = 255;

fn invariant(detail: impl Into<String>) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

/// Write `m`'s repeated regions of at least [`MIN_REGION`] instructions as kept routines, on
/// every profile. Returns the routines made.
pub fn outline_entry(m: &mut TypedModule, meter: &mut Meter) -> Result<u32, Error> {
    outline(m, MIN_REGION, meter)
}

/// Write `m`'s repeated regions of at least `min_region` instructions as kept routines, largest
/// saving first. Returns the routines made.
pub fn outline(m: &mut TypedModule, min_region: u64, meter: &mut Meter) -> Result<u32, Error> {
    let mut cache = KeyCache::default();
    let r = outline_kept(m, min_region, &mut cache, meter);
    meter.release(cache.charged);
    r
}

/// [`outline`] with the instructions' keys kept in `cache` from one search to the next.
fn outline_kept(
    m: &mut TypedModule,
    min_region: u64,
    cache: &mut KeyCache,
    meter: &mut Meter,
) -> Result<u32, Error> {
    // By function: searched to the end with nothing found, and unchanged since.
    let mut settled: Vec<bool> = Vec::new();
    let mut made = 0u32;
    while made < MAX_ROUTINES {
        settled.resize(m.functions.len(), false);
        let mut ledger = 0usize;
        let found = find(
            m,
            min_region,
            (&mut *cache, settled.as_mut_slice()),
            meter,
            &mut ledger,
        );
        meter.release(ledger);
        let Some(plan) = found? else {
            break;
        };
        if let Some(s) = settled.get_mut(plan.func.index()) {
            *s = false;
        }
        apply(m, &plan)?;
        made += 1;
    }
    if made > 0 {
        m.rederive()?;
    }
    Ok(made)
}

/// The best set of regions of `m` that becomes a routine, if any.
fn find(
    m: &TypedModule,
    min_region: u64,
    (cache, settled): (&mut KeyCache, &mut [bool]),
    meter: &mut Meter,
    ledger: &mut usize,
) -> Result<Option<Plan>, Error> {
    let w = weights(m, meter)?;
    let n = m.functions.len();
    charge::<u64>(meter, ledger, n.saturating_mul(2))?;
    // An item's weight counts a call of a kept routine as one instruction: a driver that keeps the
    // routine compiles it once, whatever calls it.
    let own: Vec<u64> = m
        .functions
        .iter()
        .zip(&w)
        .map(|(f, x)| if f.kept { 0 } else { *x })
        .collect();
    let mut reached = vec![false; n];
    let mut stack = vec![m.entry.func];
    let mut funcs = Vec::new();
    while let Some(f) = stack.pop() {
        match reached.get_mut(f.index()) {
            Some(r) if !*r => *r = true,
            _ => continue,
        }
        funcs.push(f);
        let blocks = m
            .functions
            .get(f.index())
            .map_or(&[][..], |x| &x.blocks[..]);
        for b in blocks {
            for node in &block(m, *b)?.insts {
                if let Inst::Call { func, .. } = node.inst {
                    stack.push(func);
                }
            }
        }
    }
    let weight = |f: FuncId| w.get(f.index()).copied().unwrap_or(0);
    funcs.retain(|f| {
        weight(*f) >= min_region.saturating_mul(2)
            && !settled.get(f.index()).copied().unwrap_or(true)
    });
    funcs.sort_unstable_by(|a, b| weight(*b).cmp(&weight(*a)).then(a.cmp(b)));
    funcs.truncate(MAX_FUNCS);
    let mut verifies = 0usize;
    for f in funcs {
        let mut fledger = 0usize;
        let r = find_in(
            m,
            (f, own.as_slice(), min_region),
            cache,
            meter,
            &mut fledger,
            &mut verifies,
        );
        meter.release(fledger);
        let (plan, whole) = r?;
        if plan.is_some() {
            return Ok(plan);
        }
        if whole {
            if let Some(s) = settled.get_mut(f.index()) {
                *s = true;
            }
        }
        if verifies >= MAX_VERIFIES {
            break;
        }
    }
    Ok(None)
}

/// A candidate set: the regions' first items, ascending, and their length in items.
struct Cand {
    starts: Vec<usize>,
    len: usize,
    saving: u64,
}

/// The best set of regions of function `func` that becomes a routine, within `verifies`, and
/// whether the search reached its end.
fn find_in(
    m: &TypedModule,
    (func, w, min_region): (FuncId, &[u64], u64),
    cache: &mut KeyCache,
    meter: &mut Meter,
    ledger: &mut usize,
    verifies: &mut usize,
) -> Result<(Option<Plan>, bool), Error> {
    let sp = spine(m, (func, w, true), cache, meter, ledger)?;
    let n = sp.items.len();
    if n < 2 * WINDOW {
        return Ok((None, true));
    }
    let key = |i: usize| sp.items.get(i).map(|it| it.key);
    charge::<u64>(meter, ledger, n + 1)?;
    let mut pw = Vec::with_capacity(n + 1);
    pw.push(0u64);
    for it in &sp.items {
        let last = pw.last().copied().unwrap_or(0);
        pw.push(last.saturating_add(it.weight));
    }
    let span = |a: usize, len: usize| {
        let end = pw.get(a + len).copied().unwrap_or(0);
        end.saturating_sub(pw.get(a).copied().unwrap_or(0))
    };
    // Every window's hash (a polynomial over its keys, rolled one item at a time), each window's
    // first item listed under it in order.
    const B: u64 = 0x0000_0100_0000_01B3;
    let pow = (1..WINDOW).fold(1u64, |p, _| p.wrapping_mul(B));
    // The same polynomial over every prefix of the keys, and the powers of `B`: the hash of any
    // range of items in constant time.
    charge::<u64>(meter, ledger, 2 * (n + 1))?;
    let mut ph = Vec::with_capacity(n + 1);
    let mut pp = Vec::with_capacity(n + 1);
    let (mut hp, mut bp) = (0u64, 1u64);
    ph.push(hp);
    pp.push(bp);
    for it in &sp.items {
        hp = hp.wrapping_mul(B).wrapping_add(it.key);
        bp = bp.wrapping_mul(B);
        ph.push(hp);
        pp.push(bp);
    }
    let range = |a: usize, len: usize| {
        let end = ph.get(a + len).copied().unwrap_or(0);
        let start = ph.get(a).copied().unwrap_or(0);
        end.wrapping_sub(start.wrapping_mul(pp.get(len).copied().unwrap_or(0)))
    };
    let mut h = 0u64;
    for i in 0..WINDOW {
        h = h.wrapping_mul(B).wrapping_add(key(i).unwrap_or(0));
    }
    charge::<(u64, u32, u32)>(meter, ledger, n)?;
    let mut buckets: HashMap<u64, Vec<u32>> = HashMap::default();
    meter.tick(u32::try_from(n).unwrap_or(u32::MAX))?;
    for i in 0..=n - WINDOW {
        let at = u32::try_from(i).map_err(|_| invariant("item index"))?;
        buckets.entry(h).or_default().push(at);
        if i + WINDOW < n {
            let out = key(i).unwrap_or(0).wrapping_mul(pow);
            h = h
                .wrapping_sub(out)
                .wrapping_mul(B)
                .wrapping_add(key(i + WINDOW).unwrap_or(0));
        }
    }
    let mut groups: Vec<Vec<u32>> = buckets.into_values().filter(|g| g.len() >= 2).collect();
    groups.sort_unstable_by_key(|g| g.first().copied());
    // Each group's extent with the largest saving around its first window; a window inside an
    // extent taken seeds nothing more.
    charge::<bool>(meter, ledger, n)?;
    let mut covered = vec![false; n];
    let mut cands = Vec::new();
    let mut steps = n.saturating_mul(8).saturating_add(1024);
    let same = |i: usize, j: usize| key(i).is_some() && key(i) == key(j);
    for g in groups {
        let Some(&p0) = g.first() else {
            continue;
        };
        let p0 = p0 as usize;
        if covered.get(p0).copied().unwrap_or(true) {
            continue;
        }
        // Each of the first later occurrences' extent beside the first's: how far the keys agree
        // to the left of the window and from its start.
        let mut ext: Vec<(usize, usize)> = Vec::new();
        for &pj in g.iter().skip(1).take(MAX_OCCURRENCES) {
            let pj = pj as usize;
            if pj < p0 + WINDOW || !(0..WINDOW).all(|d| same(p0 + d, pj + d)) {
                continue;
            }
            let mut l = 0usize;
            while l < p0 && l + WINDOW < pj - p0 && steps > 0 && same(p0 - 1 - l, pj - 1 - l) {
                l += 1;
                steps -= 1;
            }
            let mut r = WINDOW;
            while p0 + r + l < pj && pj + r < n && steps > 0 && same(p0 + r, pj + r) {
                r += 1;
                steps -= 1;
            }
            ext.push((l, r));
        }
        meter.tick(u32::try_from(ext.len()).unwrap_or(u32::MAX))?;
        let mut best: Option<Cand> = None;
        for &(l, r) in &ext {
            let len = l + r;
            let weight = span(p0 - l, len);
            if weight < min_region {
                continue;
            }
            // Every occurrence of the window in the function that holds the extent, none
            // overlapping the one before.
            meter.tick(u32::try_from(g.len()).unwrap_or(u32::MAX))?;
            let h0 = range(p0 - l, len);
            let mut starts = vec![p0 - l];
            for &pk in g.iter().skip(1) {
                let Some(s) = (pk as usize).checked_sub(l) else {
                    continue;
                };
                if starts.last().is_some_and(|q| s >= q + len)
                    && s + len <= n
                    && range(s, len) == h0
                {
                    starts.push(s);
                }
            }
            if starts.len() < 2 {
                continue;
            }
            let saving = weight.saturating_mul(starts.len() as u64 - 1);
            if best.as_ref().is_none_or(|b| saving > b.saving) {
                best = Some(Cand {
                    starts,
                    len,
                    saving,
                });
            }
        }
        if let Some(c) = best {
            if let Some(first) = c.starts.first() {
                for x in covered.iter_mut().skip(*first).take(c.len) {
                    *x = true;
                }
            }
            cands.push(c);
        }
        if steps == 0 {
            break;
        }
    }
    if cands.is_empty() {
        return Ok((None, steps > 0));
    }
    cands.sort_unstable_by(|a, b| {
        b.saving
            .cmp(&a.saving)
            .then(a.starts.first().cmp(&b.starts.first()))
    });
    let mut cledger = 0usize;
    let r = verify_each(
        m,
        (func, &sp, min_region),
        &cands,
        meter,
        &mut cledger,
        verifies,
    );
    meter.release(cledger);
    let (plan, whole) = r?;
    Ok((plan, whole && steps > 0))
}

/// The first of `cands` that verifies, within `verifies`, and whether every one was compared. A
/// candidate whose comparison fails at a place is compared again without the item that holds it
/// (keys alike at a region's edge, as an addition after a call's body that reads the body's result
/// at one place and another value at the other), at most [`MAX_TRIMS`] times.
fn verify_each(
    m: &TypedModule,
    (func, sp, min_region): (FuncId, &Spine, u64),
    cands: &[Cand],
    meter: &mut Meter,
    ledger: &mut usize,
    verifies: &mut usize,
) -> Result<(Option<Plan>, bool), Error> {
    let cx = context(m, func, sp, meter, ledger)?;
    for c in cands {
        let mut cur = Cand {
            starts: c.starts.clone(),
            len: c.len,
            saving: c.saving,
        };
        for _ in 0..=MAX_TRIMS {
            if *verifies >= MAX_VERIFIES {
                return Ok((None, false));
            }
            *verifies += 1;
            let mut vledger = 0usize;
            let r = verify(m, &cx, &cur, meter, &mut vledger);
            meter.release(vledger);
            match r? {
                Ok(plan) => return Ok((Some(plan), false)),
                Err(Some(loc)) => match trimmed(sp, &cur, loc, min_region) {
                    Some(next) => cur = next,
                    None => break,
                },
                Err(None) => break,
            }
        }
    }
    Ok((None, true))
}

/// Candidate `c` without the item of its first region that holds `loc`: the part before the item
/// or the part after it, whichever weighs more, where that part weighs at least `min_region`.
fn trimmed(sp: &Spine, c: &Cand, loc: Loc, min_region: u64) -> Option<Cand> {
    let s0 = *c.starts.first()?;
    // An operand's place, and each item's first place, ordered as the region runs: a block's phis,
    // its instructions, then its construct.
    let place = |b: BlockId, kind: u8, i: u32| sp.pos(b).map(|p| (p, kind, i));
    let target = match loc {
        Loc::Phi { b, .. } => place(b, 0, 0),
        Loc::Inst { b, i, .. } => place(b, 1, i),
        Loc::Term { b } => place(b, 2, 0),
    }?;
    let items = sp.items.get(s0..s0 + c.len)?;
    let mut j = None;
    for (x, it) in items.iter().enumerate() {
        let b = *sp.blocks.get(it.spine as usize)?;
        let key = if it.inst == TAIL {
            place(b, 2, 0)
        } else {
            place(b, 1, it.inst)
        }?;
        if key > target {
            break;
        }
        j = Some(x);
    }
    let j = j?;
    let weight = |part: &[Item]| part.iter().fold(0u64, |a, it| a.saturating_add(it.weight));
    let (wa, wb) = (weight(items.get(..j)?), weight(items.get(j + 1..)?));
    let others = (c.starts.len() as u64).saturating_sub(1);
    if wa >= wb && wa >= min_region {
        Some(Cand {
            starts: c.starts.clone(),
            len: j,
            saving: wa.saturating_mul(others),
        })
    } else if wb >= min_region {
        Some(Cand {
            starts: c.starts.iter().map(|s| s + j + 1).collect(),
            len: c.len - j - 1,
            saving: wb.saturating_mul(others),
        })
    } else {
        None
    }
}

/// What every comparison in one function reads: how many operands read each value, each block's
/// predecessors, and its loops and switch-at-use headers.
struct Cx<'s> {
    func: FuncId,
    sp: &'s Spine,
    /// By value: the operands of the function that read it (a loop's exit selector counting one
    /// at its header).
    uses: Vec<u32>,
    /// By layout position: the positions of the blocks that branch to it or name it as their
    /// merge or continue target.
    preds: Vec<Vec<u32>>,
    /// By layout position: whether a loop of the function holds it (after the loop's header
    /// and before its merge).
    in_loop: Vec<bool>,
    /// Every loop's continue block.
    conts: HashSet<BlockId>,
    /// By header: its loop's index in the module's table.
    header_of: HashMap<BlockId, usize>,
    at_use: HashSet<BlockId>,
}

fn bump(uses: &mut [u32], v: Value) {
    if let Some(u) = uses.get_mut(v.index()) {
        *u = u.saturating_add(1);
    }
}

/// Every block `b`'s terminator branches to, then its merge and continue targets.
fn successors(b: &Block) -> Vec<BlockId> {
    let mut out = Vec::new();
    let mut term = b.term.clone();
    term.for_each_target_mut(|t| out.push(*t));
    match b.merge {
        Some(Merge::Selection { merge }) => out.push(merge),
        Some(Merge::Loop { merge, cont }) => {
            out.push(merge);
            out.push(cont);
        }
        None => {}
    }
    out
}

fn context<'s>(
    m: &TypedModule,
    func: FuncId,
    sp: &'s Spine,
    meter: &mut Meter,
    ledger: &mut usize,
) -> Result<Cx<'s>, Error> {
    charge::<u32>(meter, ledger, m.values.len())?;
    let mut uses = vec![0u32; m.values.len()];
    let nl = sp.layout.len();
    charge::<(Vec<u32>, [u32; 3])>(meter, ledger, nl)?;
    let mut preds: Vec<Vec<u32>> = vec![Vec::new(); nl];
    for (lp, b) in sp.layout.iter().enumerate() {
        let bl = block(m, *b)?;
        meter.tick(u32::try_from(bl.insts.len()).unwrap_or(u32::MAX))?;
        for p in &bl.phis {
            for (v, _) in &p.incoming {
                bump(&mut uses, *v);
            }
        }
        for node in &bl.insts {
            node.inst.for_each_value(|v| bump(&mut uses, v));
        }
        bl.term.for_each_value(|v| bump(&mut uses, v));
        let from = u32::try_from(lp).map_err(|_| invariant("layout"))?;
        for t in successors(bl) {
            if let Some(q) = sp.pos(t).and_then(|q| preds.get_mut(q)) {
                q.push(from);
            }
        }
    }
    charge::<(i32, bool)>(meter, ledger, nl + 1)?;
    charge::<BlockId>(meter, ledger, m.loops.len())?;
    let mut depth = vec![0i32; nl + 1];
    let mut conts = HashSet::default();
    let mut header_of = HashMap::default();
    for (i, l) in m.loops.iter().enumerate() {
        conts.insert(l.cont);
        let (Some(h), Some(mg)) = (sp.pos(l.header), sp.pos(l.merge)) else {
            continue;
        };
        if h + 1 < mg {
            if let Some(d) = depth.get_mut(h + 1) {
                *d += 1;
            }
            if let Some(d) = depth.get_mut(mg) {
                *d -= 1;
            }
        }
        header_of.insert(l.header, i);
        if let Some(var) = l.exit_sel.and_then(|v| m.vars.get(v.index())) {
            bump(&mut uses, var.value);
        }
    }
    let mut open = 0i32;
    let in_loop = depth
        .iter()
        .take(nl)
        .map(|d| {
            open += d;
            open > 0
        })
        .collect();
    charge::<(BlockId, usize)>(meter, ledger, header_of.len())?;
    charge::<BlockId>(meter, ledger, m.switches_at_use.len())?;
    let at_use = m.switches_at_use.iter().copied().collect();
    Ok(Cx {
        func,
        sp,
        uses,
        preds,
        in_loop,
        conts,
        header_of,
        at_use,
    })
}

/// One region of a candidate set: its first block and the first instruction of it the region
/// holds, its last block and the instruction of it the region stops before, and their layout
/// positions.
struct Bounds {
    first: BlockId,
    a: usize,
    last: BlockId,
    idx: usize,
    ps: usize,
    pe: usize,
}

/// The region of items `[s, s + len)` of spine `sp`.
fn bounds(m: &TypedModule, sp: &Spine, s: usize, len: usize) -> Result<Option<Bounds>, Error> {
    let (Some(start), Some(end)) = (sp.items.get(s), sp.items.get(s + len)) else {
        return Ok(None);
    };
    let at = |i: u32| {
        sp.blocks
            .get(i as usize)
            .copied()
            .ok_or_else(|| invariant("spine block"))
    };
    let (first, last) = (at(start.spine)?, at(end.spine)?);
    let a = match start.inst {
        TAIL => block(m, first)?.insts.len(),
        i => i as usize,
    };
    let idx = match end.inst {
        TAIL => block(m, last)?.insts.len(),
        i => i as usize,
    };
    let (Some(ps), Some(pe)) = (sp.pos(first), sp.pos(last)) else {
        return Ok(None);
    };
    if pe < ps || (pe == ps && idx <= a) {
        return Ok(None);
    }
    Ok(Some(Bounds {
        first,
        a,
        last,
        idx,
        ps,
        pe,
    }))
}

/// The parts of the region's block `k` of `nb` (of `len` instructions) it holds: whether its phis,
/// the range of its instructions, and whether its merge and terminator.
fn parts(bd: &Bounds, k: usize, nb: usize, len: usize) -> (bool, Range<usize>, bool) {
    if nb == 1 {
        return (false, bd.a..bd.idx, false);
    }
    if k == 0 {
        return (false, bd.a..len, true);
    }
    if k + 1 == nb {
        return (true, 0..bd.idx, false);
    }
    (true, 0..len, true)
}

/// An operand may be a constant that differs between regions, as a parameter.
const CONST_OK: u8 = 1;
/// An operand is an atomic's pointer, which must be a variable or an access chain's result.
const ATOMIC_PTR: u8 = 2;
/// An operand is the pointer a store writes whole.
const WHOLE_STORE: u8 = 4;

/// An operand of a region, in walk order: its value, its block's offset in the region, and its
/// flags.
#[derive(Clone, Copy)]
struct Op {
    v: Value,
    k: u32,
    flags: u8,
}

/// Where an operand of the first region is: incoming value `j` of phi `p` of block `b`, operand
/// `o` (in [`Inst::for_each_value`] order) of instruction `i`, or the value operand of `b`'s
/// terminator.
#[derive(Clone, Copy, Debug)]
enum Loc {
    Phi { b: BlockId, p: u32, j: u32 },
    Inst { b: BlockId, i: u32, o: u32 },
    Term { b: BlockId },
}

/// What two regions must agree on, part by part: a phi's type and incoming count, an instruction
/// with its value operands erased and its result's type, a block's merge kind and terminator
/// with its operands and targets erased (each target compared by its place in the region).
#[derive(Clone, Debug, PartialEq, Eq)]
enum Part {
    Phi { ty: TypeId, incoming: usize },
    Inst { inst: Inst, ty: Option<TypeId> },
    Tail { merge: u8, term: Terminator },
}

/// One region read in layout order.
#[derive(Default)]
struct Walk {
    defs: Vec<Value>,
    ops: Vec<Op>,
    /// Each operand's place: the first region's only.
    locs: Vec<Loc>,
    /// Each definition's place: the first region's only.
    def_locs: Vec<Loc>,
    /// The first region's parts; another region's are compared as it is read.
    parts: Vec<Part>,
    /// Each part's place: the first region's only.
    part_locs: Vec<Loc>,
    /// The region offset of each branch, merge or continue target and of each phi's incoming
    /// block, in order.
    edges: Vec<u32>,
    /// Each loop it holds: its unrolling and its exit selector's variable.
    loops: Vec<(Unrolling, Option<Value>)>,
    /// The offsets of the switch-at-use headers it holds.
    at_use: Vec<u32>,
    refused: bool,
    /// The part where the walk was first refused, where one was.
    failed: Option<usize>,
}

impl Walk {
    /// Refuse the region at part `at`.
    fn fail(&mut self, at: usize) {
        self.refused = true;
        self.failed.get_or_insert(at);
    }
}

/// Record `part` at `loc`, or compare it with `reference`'s at the same place; returns its index.
fn note(w: &mut Walk, reference: Option<&Walk>, at: &mut usize, (part, loc): (Part, Loc)) -> usize {
    let here = *at;
    match reference {
        Some(r) => {
            if r.parts.get(here) != Some(&part) {
                w.fail(here);
            }
        }
        None => {
            w.parts.push(part);
            w.part_locs.push(loc);
        }
    }
    *at += 1;
    here
}

/// Whether operand `operand` of `inst` (in [`Inst::for_each_value`] order) is an access chain's
/// index into an array, vector or matrix of data, which any value may be.
fn array_index(m: &TypedModule, inst: &Inst, operand: usize) -> Result<bool, Error> {
    let Inst::AccessChain { base, indices, .. } = inst else {
        return Ok(false);
    };
    if operand == 0 || operand > indices.len() {
        return Ok(false);
    }
    let Type::Pointer { pointee, .. } = m.types.get(m.value(*base)?.ty)? else {
        return Ok(false);
    };
    let mut t = *pointee;
    for i in indices.iter().take(operand - 1) {
        t = match m.types.get(t)? {
            Type::Array { elem, .. }
            | Type::RuntimeArray { elem, .. }
            | Type::Vector { elem, .. } => *elem,
            Type::Matrix { col, .. } => *col,
            Type::Struct { members, .. } => {
                let ValueOrigin::Const(c) = m.value(*i)?.origin else {
                    return Ok(false);
                };
                let member = m
                    .consts
                    .scalar_bits(c)
                    .and_then(|b| usize::try_from(b.0).ok())
                    .and_then(|x| members.get(x));
                match member {
                    Some(x) => *x,
                    None => return Ok(false),
                }
            }
            _ => return Ok(false),
        };
    }
    let indexed = matches!(
        m.types.get(t)?,
        Type::Array { .. } | Type::RuntimeArray { .. } | Type::Vector { .. } | Type::Matrix { .. }
    );
    Ok(indexed && !matches!(m.types.facts(t)?.shape, Shape::DescriptorArray { .. }))
}

/// Walk region `bd` of `cx`'s function, comparing its parts with `reference`'s where given.
fn walk(
    m: &TypedModule,
    cx: &Cx<'_>,
    bd: &Bounds,
    reference: Option<&Walk>,
) -> Result<Walk, Error> {
    let mut w = Walk::default();
    let nb = bd.pe - bd.ps + 1;
    let mut part_at = 0usize;
    let offset = |t: BlockId| {
        cx.sp
            .pos(t)
            .filter(|p| (bd.ps..=bd.pe).contains(p))
            .and_then(|p| u32::try_from(p - bd.ps).ok())
    };
    for k in 0..nb {
        let b = *cx
            .sp
            .layout
            .get(bd.ps + k)
            .ok_or_else(|| invariant("layout"))?;
        let bl = block(m, b)?;
        let k32 = u32::try_from(k).map_err(|_| invariant("region block"))?;
        let (phis, range, tail) = parts(bd, k, nb, bl.insts.len());
        if phis {
            for (pi, p) in bl.phis.iter().enumerate() {
                let pi = u32::try_from(pi).map_err(|_| invariant("phi index"))?;
                w.defs.push(p.result);
                if reference.is_none() {
                    w.def_locs.push(Loc::Phi { b, p: pi, j: 0 });
                }
                let part = Part::Phi {
                    ty: m.value(p.result)?.ty,
                    incoming: p.incoming.len(),
                };
                let loc = Loc::Phi { b, p: pi, j: 0 };
                let here = note(&mut w, reference, &mut part_at, (part, loc));
                for (j, (v, from)) in p.incoming.iter().enumerate() {
                    match offset(*from) {
                        Some(o) if (o as usize) + 1 < nb => w.edges.push(o),
                        _ => w.fail(here),
                    }
                    w.ops.push(Op {
                        v: *v,
                        k: k32,
                        flags: CONST_OK,
                    });
                    if reference.is_none() {
                        w.locs.push(Loc::Phi {
                            b,
                            p: pi,
                            j: u32::try_from(j).map_err(|_| invariant("phi operand"))?,
                        });
                    }
                }
            }
        }
        let start = range.start;
        let insts = bl
            .insts
            .get(range)
            .ok_or_else(|| invariant("region instructions"))?;
        for (ii, node) in insts.iter().enumerate() {
            let i = u32::try_from(start + ii).map_err(|_| invariant("instruction index"))?;
            if let Some(r) = node.result {
                w.defs.push(r);
                if reference.is_none() {
                    w.def_locs.push(Loc::Inst { b, i, o: 0 });
                }
            }
            let mut erased = node.inst.clone();
            erased.for_each_value_mut(|v| *v = Value::new(0));
            let ty = match node.result {
                Some(r) => Some(m.value(r)?.ty),
                None => None,
            };
            let part = Part::Inst { inst: erased, ty };
            let here = note(
                &mut w,
                reference,
                &mut part_at,
                (part, Loc::Inst { b, i, o: 0 }),
            );
            if !rollable(&node.inst) {
                w.fail(here);
            }
            let switched = switched_operand(&node.inst);
            let atomic = node.inst.atomic_pointer();
            let store = matches!(node.inst, Inst::Store { .. });
            let mut vals = Vec::new();
            node.inst.for_each_value(|v| vals.push(v));
            for (o, v) in vals.into_iter().enumerate() {
                let mut flags = 0u8;
                if switched != Some(o)
                    && (varying_ok(&node.inst, o) || array_index(m, &node.inst, o)?)
                {
                    flags |= CONST_OK;
                }
                if atomic == Some(v) {
                    flags |= ATOMIC_PTR;
                }
                if store && o == 0 {
                    flags |= WHOLE_STORE;
                }
                w.ops.push(Op { v, k: k32, flags });
                if reference.is_none() {
                    w.locs.push(Loc::Inst {
                        b,
                        i,
                        o: u32::try_from(o).map_err(|_| invariant("operand index"))?,
                    });
                }
            }
        }
        if tail {
            let merge = match bl.merge {
                None => 0u8,
                Some(Merge::Selection { .. }) => 1,
                Some(Merge::Loop { .. }) => 2,
            };
            let mut term = bl.term.clone();
            term.for_each_target_mut(|t| *t = BlockId::new(0));
            term.for_each_value_mut(|v| *v = Value::new(0));
            let part = Part::Tail { merge, term };
            let here = note(&mut w, reference, &mut part_at, (part, Loc::Term { b }));
            match bl.term {
                Terminator::Branch(_)
                | Terminator::CondBranch { .. }
                | Terminator::Switch { .. }
                | Terminator::Unreachable => {}
                _ => w.fail(here),
            }
            // Every target is a later block of the region.
            for t in successors(bl) {
                match offset(t) {
                    Some(o) if o >= 1 => w.edges.push(o),
                    _ => w.fail(here),
                }
            }
            let mut vals = Vec::new();
            bl.term.for_each_value(|v| vals.push(v));
            for v in vals {
                w.ops.push(Op {
                    v,
                    k: k32,
                    flags: CONST_OK,
                });
                if reference.is_none() {
                    w.locs.push(Loc::Term { b });
                }
            }
        }
        if let Some(&li) = cx.header_of.get(&b) {
            let l = m.loops.get(li).ok_or_else(|| invariant("loop"))?;
            let sel = match l.exit_sel {
                Some(v) => Some(
                    m.vars
                        .get(v.index())
                        .ok_or_else(|| invariant("variable"))?
                        .value,
                ),
                None => None,
            };
            w.loops.push((l.unrolling, sel));
        }
        if cx.at_use.contains(&b) {
            w.at_use.push(k32);
        }
        if w.refused {
            return Ok(w);
        }
    }
    if reference.is_some_and(|r| r.parts.len() != part_at) {
        w.fail(part_at);
    }
    Ok(w)
}

/// Whether a value of type `t` may be a routine's parameter or result as data.
fn data(m: &TypedModule, t: TypeId) -> Result<bool, Error> {
    Ok(matches!(
        m.types.facts(t)?.shape,
        Shape::Plain | Shape::Implicit
    ))
}

/// Whether value `v` may be the argument of a parameter of pointer type `t`: a variable of a class
/// a call may pass, as the builder's calls take them.
fn pointer_arg(m: &TypedModule, t: TypeId, v: Value) -> Result<bool, Error> {
    let Type::Pointer { class, .. } = m.types.get(t)? else {
        return Ok(false);
    };
    Ok(match class {
        StorageClass::PhysicalStorageBuffer => true,
        StorageClass::UniformConstant
        | StorageClass::Function
        | StorageClass::Private
        | StorageClass::Workgroup => matches!(
            m.value(v)?.origin,
            ValueOrigin::Global(_) | ValueOrigin::Var(_)
        ),
        _ => false,
    })
}

/// How often a region reads a value, and its first read's flags and block offset.
#[derive(Clone, Copy)]
struct Seen {
    count: u32,
    flags: u8,
    k: u32,
}

/// One region of a plan, where it is before the rewrite.
#[derive(Clone, Debug)]
struct Occurrence {
    first: BlockId,
    a: usize,
    last: BlockId,
    idx: usize,
    /// Its layout position and first instruction, which order the rewrite.
    at: (usize, usize),
    /// Its blocks in layout order, the first and last included.
    blocks: Vec<BlockId>,
    /// Its arguments, one per parameter.
    args: Vec<Value>,
    /// Its values read after it, in the routine's result order.
    escapes: Vec<Value>,
    /// The `Function` variables that become the routine's (the first region's) or leave with
    /// their region (another's), in the same order.
    locals: Vec<VarId>,
}

/// A set of regions that are the same code, and the routine they become.
#[derive(Clone, Debug)]
struct Plan {
    func: FuncId,
    /// The first region (whose code the routine takes), then the others.
    occ: Vec<Occurrence>,
    /// Each parameter's type.
    params: Vec<TypeId>,
    /// Each operand of the first region that reads a parameter, and which.
    subs: Vec<(Loc, usize)>,
}

/// A candidate's plan, or where its first region's comparison failed, where it can say.
type Verdict = Result<Plan, Option<Loc>>;

/// The plan for candidate `c`, where its regions are exactly the same code and each can become a
/// call; otherwise the place of the first region whose operand or result kept them in line, where
/// there is one.
fn verify(
    m: &TypedModule,
    cx: &Cx<'_>,
    c: &Cand,
    meter: &mut Meter,
    ledger: &mut usize,
) -> Result<Verdict, Error> {
    let sp = cx.sp;
    let mut occ: Vec<Bounds> = Vec::with_capacity(c.starts.len());
    for &s in &c.starts {
        match bounds(m, sp, s, c.len)? {
            Some(b) => occ.push(b),
            None => return Ok(Err(None)),
        }
    }
    let nb = occ.first().map_or(0, |b| b.pe - b.ps + 1);
    if nb == 0 || occ.iter().any(|b| b.pe - b.ps + 1 != nb) {
        return Ok(Err(None));
    }
    // No two regions share a block but where one ends and the next starts.
    let mut order: Vec<usize> = (0..occ.len()).collect();
    order.sort_unstable_by_key(|&t| occ.get(t).map(|b| (b.ps, b.a)));
    for pair in order.windows(2) {
        let (Some(x), Some(y)) = (
            pair.first().and_then(|&t| occ.get(t)),
            pair.get(1).and_then(|&t| occ.get(t)),
        ) else {
            return Ok(Err(None));
        };
        if x.pe > y.ps || (x.pe == y.ps && x.idx > y.a) {
            return Ok(Err(None));
        }
    }
    // Each region is entered only at its start and left only at its end: nothing outside it
    // branches to, merges at or continues at its blocks, and nothing inside it branches back to
    // its first block. Its last block, whose terminator stays outside, continues no loop.
    for bd in &occ {
        for k in 0..nb {
            for &src in cx.preds.get(bd.ps + k).map_or(&[][..], |v| &v[..]) {
                let src = src as usize;
                let bad = if k == 0 {
                    (bd.ps..=bd.pe).contains(&src)
                } else {
                    !(bd.ps..bd.pe).contains(&src)
                };
                if bad {
                    return Ok(Err(None));
                }
            }
        }
        if nb > 1 && cx.conts.contains(&bd.last) {
            return Ok(Err(None));
        }
    }
    // Each region's walk, compared with the first's.
    let mut walks: Vec<Walk> = Vec::with_capacity(occ.len());
    for bd in &occ {
        meter.tick(u32::try_from(bd.pe - bd.ps + 1).unwrap_or(u32::MAX))?;
        let w = walk(m, cx, bd, walks.first())?;
        if w.refused {
            // Where the first region's walk was refused, or the part of it another's differs at.
            let first = walks.first().unwrap_or(&w);
            return Ok(Err(w
                .failed
                .and_then(|at| first.part_locs.get(at))
                .copied()));
        }
        charge::<Op>(meter, ledger, w.ops.len())?;
        charge::<(Value, Value, u32)>(meter, ledger, w.defs.len())?;
        charge::<(Loc, Part)>(meter, ledger, w.parts.len())?;
        if let Some(w0) = walks.first() {
            let differ = w.ops.len() != w0.ops.len()
                || w.defs.len() != w0.defs.len()
                || w.edges != w0.edges
                || w.at_use != w0.at_use
                || w.loops.len() != w0.loops.len()
                || w.loops
                    .iter()
                    .zip(&w0.loops)
                    .any(|(x, y)| x.0 != y.0 || x.1.is_some() != y.1.is_some());
            if differ {
                return Ok(Err(None));
            }
        }
        walks.push(w);
    }
    let Some(w0) = walks.first() else {
        return Ok(Err(None));
    };
    // Each region's definitions by place, and what it reads.
    let mut def_at: Vec<HashMap<Value, usize>> = Vec::with_capacity(walks.len());
    let mut seen: Vec<HashMap<Value, Seen>> = Vec::with_capacity(walks.len());
    for w in &walks {
        def_at.push(w.defs.iter().enumerate().map(|(i, v)| (*v, i)).collect());
        let mut s: HashMap<Value, Seen> = HashMap::default();
        for o in &w.ops {
            s.entry(o.v)
                .and_modify(|x| x.count = x.count.saturating_add(1))
                .or_insert(Seen {
                    count: 1,
                    flags: o.flags,
                    k: o.k,
                });
        }
        for v in w.loops.iter().filter_map(|l| l.1) {
            s.entry(v)
                .and_modify(|x| x.count = x.count.saturating_add(1))
                .or_insert(Seen {
                    count: 1,
                    flags: 0,
                    k: u32::MAX,
                });
        }
        seen.push(s);
    }
    // Whether each region is inside a loop of its function, and which of its blocks it always
    // runs: the blocks of its own sequence.
    let in_loop: Vec<bool> = occ
        .iter()
        .map(|bd| cx.in_loop.get(bd.ps).copied().unwrap_or(false))
        .collect();
    let mut always: Vec<Vec<bool>> = Vec::with_capacity(occ.len());
    for (bd, &s) in occ.iter().zip(&c.starts) {
        let mut f = vec![false; nb];
        for it in sp.items.get(s..=s + c.len).unwrap_or(&[]) {
            let b = sp.blocks.get(it.spine as usize).copied();
            if let Some(x) = b
                .and_then(|b| sp.pos(b))
                .and_then(|p| p.checked_sub(bd.ps))
                .and_then(|k| f.get_mut(k))
            {
                *x = true;
            }
        }
        always.push(f);
    }
    // A `Function` variable becomes the routine's where only its region reads it and no value of
    // it carries from one run of the region to the next.
    let local = |t: usize, v: Value| -> Result<Option<VarId>, Error> {
        let ValueOrigin::Var(var) = m.value(v)?.origin else {
            return Ok(None);
        };
        let Some(x) = seen.get(t).and_then(|s| s.get(&v)) else {
            return Ok(None);
        };
        if cx.uses.get(v.index()).copied() != Some(x.count) {
            return Ok(None);
        }
        let stored_first = x.flags & WHOLE_STORE != 0
            && always
                .get(t)
                .and_then(|f| f.get(x.k as usize))
                .copied()
                .unwrap_or(false);
        let fresh = !in_loop.get(t).copied().unwrap_or(true) || stored_first;
        Ok(fresh.then_some(var))
    };
    // Every operand of every region against the first's: a definition at the same place, a local
    // variable paired one to one, or a value from before it (a site).
    let n_occ = occ.len();
    let mut site_vals: Vec<Vec<Value>> = vec![Vec::new(); n_occ];
    let mut site_ops: Vec<usize> = Vec::new();
    let mut locals: Vec<Vec<VarId>> = vec![Vec::new(); n_occ];
    let mut var_map: Vec<HashMap<VarId, VarId>> = vec![HashMap::default(); n_occ];
    let mut var_taken: Vec<HashSet<VarId>> = vec![HashSet::default(); n_occ];
    for (t, w) in walks.iter().enumerate() {
        let (Some(defs0), Some(defst)) = (def_at.first(), def_at.get(t)) else {
            return Ok(Err(None));
        };
        meter.tick(u32::try_from(w.ops.len()).unwrap_or(u32::MAX))?;
        for (i, (o0, ot)) in w0.ops.iter().zip(&w.ops).enumerate() {
            match (defs0.get(&o0.v), defst.get(&ot.v)) {
                (Some(x), Some(y)) if x == y => continue,
                (None, None) => {}
                _ => return Ok(Err(w0.locs.get(i).copied())),
            }
            match (local(0, o0.v)?, local(t, ot.v)?) {
                (Some(x), Some(y)) => {
                    let map = var_map.get_mut(t).ok_or_else(|| invariant("variables"))?;
                    match map.get(&x) {
                        Some(z) if *z == y => {}
                        Some(_) => return Ok(Err(w0.locs.get(i).copied())),
                        None => {
                            let taken =
                                var_taken.get_mut(t).ok_or_else(|| invariant("variables"))?;
                            let (Some(vx), Some(vy)) =
                                (m.vars.get(x.index()), m.vars.get(y.index()))
                            else {
                                return Ok(Err(w0.locs.get(i).copied()));
                            };
                            if !taken.insert(y) || vx.pointee != vy.pointee || vx.init != vy.init {
                                return Ok(Err(w0.locs.get(i).copied()));
                            }
                            map.insert(x, y);
                            locals
                                .get_mut(t)
                                .ok_or_else(|| invariant("locals"))?
                                .push(y);
                        }
                    }
                    continue;
                }
                (None, None) => {}
                _ => return Ok(Err(w0.locs.get(i).copied())),
            }
            if m.value(o0.v)?.ty != m.value(ot.v)?.ty {
                return Ok(Err(w0.locs.get(i).copied()));
            }
            site_vals
                .get_mut(t)
                .ok_or_else(|| invariant("sites"))?
                .push(ot.v);
            if t == 0 {
                site_ops.push(i);
            }
        }
        // Each loop's exit selector is a local variable, paired as the region's others are.
        for (x, y) in w0.loops.iter().zip(&w.loops) {
            match (x.1, y.1) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    let (ValueOrigin::Var(va), ValueOrigin::Var(vb)) =
                        (m.value(a)?.origin, m.value(b)?.origin)
                    else {
                        return Ok(Err(None));
                    };
                    if var_map.get(t).and_then(|map| map.get(&va)) != Some(&vb) {
                        return Ok(Err(None));
                    }
                }
                _ => return Ok(Err(None)),
            }
        }
    }
    // Parameters: one per distinct set of the regions' values at a site.
    let mut params: Vec<TypeId> = Vec::new();
    let mut subs: Vec<(Loc, usize)> = Vec::new();
    let mut by_tuple: BTreeMap<Vec<Value>, usize> = BTreeMap::new();
    let mut args: Vec<Vec<Value>> = vec![Vec::new(); n_occ];
    for (s, &oi) in site_ops.iter().enumerate() {
        let tuple: Vec<Value> = site_vals
            .iter()
            .map(|x| x.get(s).copied())
            .collect::<Option<_>>()
            .ok_or_else(|| invariant("site"))?;
        let (Some(&v0), Some(op0)) = (tuple.first(), w0.ops.get(oi)) else {
            return Ok(Err(None));
        };
        let same = tuple.iter().all(|v| *v == v0);
        if same
            && matches!(
                m.value(v0)?.origin,
                ValueOrigin::Const(_) | ValueOrigin::Global(_)
            )
        {
            continue;
        }
        let mut any_const = false;
        for v in &tuple {
            any_const |= matches!(m.value(*v)?.origin, ValueOrigin::Const(_));
            if m.non_uniform.contains(v) {
                return Ok(Err(w0.locs.get(oi).copied()));
            }
        }
        if (any_const && !same && op0.flags & CONST_OK == 0) || op0.flags & ATOMIC_PTR != 0 {
            return Ok(Err(w0.locs.get(oi).copied()));
        }
        let t0 = m.value(v0)?.ty;
        if !data(m, t0)? {
            for v in &tuple {
                if !pointer_arg(m, t0, *v)? {
                    return Ok(Err(w0.locs.get(oi).copied()));
                }
            }
        }
        let p = match by_tuple.get(&tuple) {
            Some(p) => *p,
            None => {
                let p = params.len();
                if p >= MAX_PARAMS {
                    return Ok(Err(None));
                }
                params.push(t0);
                for (t, v) in tuple.iter().enumerate() {
                    args.get_mut(t)
                        .ok_or_else(|| invariant("arguments"))?
                        .push(*v);
                }
                by_tuple.insert(tuple, p);
                p
            }
        };
        let loc = *w0.locs.get(oi).ok_or_else(|| invariant("site place"))?;
        subs.push((loc, p));
    }
    // Results: each place whose definition some region's code reads after it.
    let mut out_idx: Vec<usize> = Vec::new();
    for (i, v0) in w0.defs.iter().enumerate() {
        let mut escapes = false;
        for (t, w) in walks.iter().enumerate() {
            let v = *w.defs.get(i).ok_or_else(|| invariant("definition"))?;
            let inside = seen.get(t).and_then(|s| s.get(&v)).map_or(0, |x| x.count);
            escapes |= cx.uses.get(v.index()).copied().unwrap_or(0) > inside;
        }
        if !escapes {
            continue;
        }
        if !data(m, m.value(*v0)?.ty)?
            || walks
                .iter()
                .any(|w| w.defs.get(i).is_some_and(|v| m.non_uniform.contains(v)))
        {
            return Ok(Err(w0.def_locs.get(i).copied()));
        }
        out_idx.push(i);
    }
    if out_idx.len() > MAX_RESULTS {
        return Ok(Err(None));
    }
    let mut plan_occ = Vec::with_capacity(n_occ);
    for (t, (bd, w)) in occ.iter().zip(&walks).enumerate() {
        let blocks = sp
            .layout
            .get(bd.ps..=bd.pe)
            .ok_or_else(|| invariant("layout"))?
            .to_vec();
        let escapes = out_idx
            .iter()
            .map(|i| w.defs.get(*i).copied().ok_or_else(|| invariant("result")))
            .collect::<Result<Vec<_>, _>>()?;
        plan_occ.push(Occurrence {
            first: bd.first,
            a: bd.a,
            last: bd.last,
            idx: bd.idx,
            at: (bd.ps, bd.a),
            blocks,
            args: args.get(t).cloned().unwrap_or_default(),
            escapes,
            locals: locals.get(t).cloned().unwrap_or_default(),
        });
    }
    Ok(Ok(Plan {
        func: cx.func,
        occ: plan_occ,
        params,
        subs,
    }))
}

/// Set the operand at `loc` to `v`.
fn set_operand(m: &mut TypedModule, loc: Loc, v: Value) -> Result<(), Error> {
    match loc {
        Loc::Phi { b, p, j } => {
            let slot = block_mut(m, b)?
                .phis
                .get_mut(p as usize)
                .and_then(|p| p.incoming.get_mut(j as usize))
                .ok_or_else(|| invariant("phi operand"))?;
            slot.0 = v;
        }
        Loc::Inst { b, i, o } => {
            let node = block_mut(m, b)?
                .insts
                .get_mut(i as usize)
                .ok_or_else(|| invariant("instruction"))?;
            let mut at = 0u32;
            node.inst.for_each_value_mut(|x| {
                if at == o {
                    *x = v;
                }
                at += 1;
            });
        }
        Loc::Term { b } => block_mut(m, b)?.term.for_each_value_mut(|x| *x = v),
    }
    Ok(())
}

/// Make each of `nodes`' results defined in block `b`.
fn set_origins(m: &mut TypedModule, nodes: &[InstNode], b: BlockId) {
    for r in nodes.iter().filter_map(|n| n.result) {
        if let Some(d) = m.values.get_mut(r.index()) {
            d.origin = ValueOrigin::Inst(b);
        }
    }
}

/// End block `b` of a routine returning `escapes` (several as a struct of type `ret`).
fn finish_return(
    m: &mut TypedModule,
    b: BlockId,
    escapes: &[Value],
    ret: TypeId,
) -> Result<(), Error> {
    let term = match escapes {
        [] => Terminator::Return,
        [one] => Terminator::ReturnValue(*one),
        _ => {
            let r = new_value(m, ret, ValueOrigin::Inst(b))?;
            block_mut(m, b)?.insts.push(InstNode {
                result: Some(r),
                inst: Inst::CompositeConstruct {
                    parts: escapes.into(),
                },
            });
            Terminator::ReturnValue(r)
        }
    };
    let bl = block_mut(m, b)?;
    bl.merge = None;
    bl.term = term;
    Ok(())
}

/// The call of routine `f` (returning `ret`, whose results have types `results`) that stands for
/// region `o`, in `o`'s first block, and the value each of its results is read from.
fn call_nodes(
    m: &mut TypedModule,
    o: &Occurrence,
    f: FuncId,
    ret: TypeId,
    results: &[TypeId],
) -> Result<(Vec<InstNode>, Vec<Value>), Error> {
    let b = o.first;
    let call = Inst::Call {
        func: f,
        args: o.args.clone().into_boxed_slice(),
    };
    let mut nodes = Vec::with_capacity(results.len() + 1);
    let mut outs = Vec::with_capacity(results.len());
    if results.is_empty() {
        nodes.push(InstNode {
            result: None,
            inst: call,
        });
        return Ok((nodes, outs));
    }
    let r = new_value(m, ret, ValueOrigin::Inst(b))?;
    nodes.push(InstNode {
        result: Some(r),
        inst: call,
    });
    if results.len() == 1 {
        outs.push(r);
        return Ok((nodes, outs));
    }
    for (i, t) in results.iter().enumerate() {
        let x = new_value(m, *t, ValueOrigin::Inst(b))?;
        let lane = u32::try_from(i).map_err(|_| invariant("result index"))?;
        nodes.push(InstNode {
            result: Some(x),
            inst: Inst::CompositeExtract {
                composite: r,
                indices: [lane].into(),
            },
        });
        outs.push(x);
    }
    Ok((nodes, outs))
}

/// Write `plan`'s first region as a new kept routine, and every region as a call of it.
fn apply(m: &mut TypedModule, plan: &Plan) -> Result<(), Error> {
    let caller = plan.func;
    let tmpl = plan.occ.first().ok_or_else(|| invariant("no region"))?;
    let f = FuncId::new(u32::try_from(m.functions.len()).map_err(|_| invariant("function ids"))?);
    // The routine: its parameters, and what it returns.
    let mut results = Vec::with_capacity(tmpl.escapes.len());
    for v in &tmpl.escapes {
        results.push(m.value(*v)?.ty);
    }
    let ret = match results.as_slice() {
        [] => ty(m, Type::Void)?,
        [one] => *one,
        _ => ty(
            m,
            Type::Struct {
                members: results.clone().into_boxed_slice(),
                layout: None,
            },
        )?,
    };
    let fn_ty = ty(
        m,
        Type::Function {
            ret,
            params: plan.params.clone().into_boxed_slice(),
        },
    )?;
    let mut params = Vec::with_capacity(plan.params.len());
    for (i, t) in plan.params.iter().enumerate() {
        let index = u32::try_from(i).map_err(|_| invariant("parameter index"))?;
        params.push(new_value(m, *t, ValueOrigin::Param { func: f, index })?);
    }
    m.functions.push(Function {
        ret,
        fn_ty,
        params: params.clone(),
        blocks: Vec::new(),
        vars: tmpl.locals.clone(),
        kept: true,
    });
    // The first region reads the parameters where it read values from before it, and its
    // variables are the routine's.
    for (loc, p) in &plan.subs {
        let v = *params.get(*p).ok_or_else(|| invariant("parameter"))?;
        set_operand(m, *loc, v)?;
    }
    let moved_vars: HashSet<VarId> = tmpl.locals.iter().copied().collect();
    for var in &tmpl.locals {
        m.vars
            .get_mut(var.index())
            .ok_or_else(|| invariant("variable"))?
            .func = f;
    }
    m.functions
        .get_mut(caller.index())
        .ok_or_else(|| invariant("function"))?
        .vars
        .retain(|v| !moved_vars.contains(v));
    // Each region in turn from the last in layout, so a block where one region ends and the next
    // starts is cut at the later region first.
    let mut order: Vec<usize> = (0..plan.occ.len()).collect();
    order.sort_unstable_by_key(|&t| std::cmp::Reverse(plan.occ.get(t).map(|o| o.at)));
    let mut rename: HashMap<Value, Value> = HashMap::default();
    let mut gone: HashSet<BlockId> = HashSet::default();
    let mut body: Vec<BlockId> = Vec::new();
    let mut cut: HashSet<BlockId> = HashSet::default();
    for t in order {
        let o = plan.occ.get(t).ok_or_else(|| invariant("region"))?;
        let (call, outs) = call_nodes(m, o, f, ret, &results)?;
        for (v, e) in o.escapes.iter().zip(&outs) {
            rename.insert(*v, *e);
        }
        let rest = if o.blocks.len() == 1 {
            // One block: its instructions from `a` to `idx` leave it.
            let rest = block_mut(m, o.first)?.insts.split_off(o.idx);
            let taken = block_mut(m, o.first)?.insts.split_off(o.a);
            if t == 0 {
                let e = new_block(m, f)?;
                set_origins(m, &taken, e);
                block_mut(m, e)?.insts.extend(taken);
                finish_return(m, e, &o.escapes, ret)?;
                body.push(e);
            } else {
                removed(m, taken.iter().filter_map(|n| n.result));
            }
            rest
        } else {
            // The first block's instructions from `a`, and its merge and terminator, leave it; the
            // last block's after `idx`, and its merge and terminator, are the first block's.
            let taken = block_mut(m, o.first)?.insts.split_off(o.a);
            if t == 0 {
                let e = new_block(m, f)?;
                set_origins(m, &taken, e);
                block_mut(m, e)?.insts.extend(taken);
                move_tail(m, o.first, e)?;
                body.push(e);
                body.extend(o.blocks.iter().skip(1).copied());
            } else {
                removed(m, taken.iter().filter_map(|n| n.result));
                cut.insert(o.first);
                gone.extend(o.blocks.iter().skip(1).copied());
            }
            let rest = block_mut(m, o.last)?.insts.split_off(o.idx);
            set_origins(m, &rest, o.first);
            move_tail(m, o.last, o.first)?;
            if t == 0 {
                for b in o.blocks.iter().skip(1) {
                    block_mut(m, *b)?.func = f;
                }
                finish_return(m, o.last, &o.escapes, ret)?;
            }
            rest
        };
        let bs = block_mut(m, o.first)?;
        bs.insts.extend(call);
        bs.insts.extend(rest);
    }
    m.switches_at_use.retain(|h| !cut.contains(h));
    let moved: HashSet<BlockId> = body.iter().copied().collect();
    m.functions
        .get_mut(f.index())
        .ok_or_else(|| invariant("routine"))?
        .blocks = body;
    let cf = m
        .functions
        .get_mut(caller.index())
        .ok_or_else(|| invariant("function"))?;
    cf.blocks.retain(|b| !moved.contains(b));
    let caller_blocks = cf.blocks.clone();
    // Each value a region leaves for after it is read from its call.
    let swap = |v: &mut Value| {
        if let Some(r) = rename.get(&*v) {
            *v = *r;
        }
    };
    for b in caller_blocks {
        if gone.contains(&b) {
            continue;
        }
        let bl = block_mut(m, b)?;
        for p in &mut bl.phis {
            for (v, _) in &mut p.incoming {
                swap(v);
            }
        }
        for n in &mut bl.insts {
            n.inst.for_each_value_mut(swap);
        }
        bl.term.for_each_value_mut(swap);
    }
    let mut keep_block = vec![true; m.blocks.len()];
    for b in &gone {
        if let Some(k) = keep_block.get_mut(b.index()) {
            *k = false;
        }
    }
    let mut keep_var = vec![true; m.vars.len()];
    for v in plan.occ.iter().skip(1).flat_map(|o| &o.locals) {
        if let Some(k) = keep_var.get_mut(v.index()) {
            *k = false;
        }
    }
    m.drop_ids(&[], &keep_block, &keep_var);
    Ok(())
}
