//! The register pressure of a kernel's entry, estimated from its structure: the most 32-bit words
//! its values hold at once, which [`super::widest_pin`] bounds.
//!
//! **Live intervals in layout order.** The body's instructions are numbered block after block
//! (the `Body` arena's order). A value is live from its definition (a parameter from the start)
//! through its last use; a phi's incoming value is used at the end of the block it comes from, and
//! a value defined before a loop and used in it is live to the loop's last instruction (a later
//! iteration reads it again). Numbering in layout order also keeps a value live across an arm of
//! a branch it is not used in, which is what a SIMT device does: it runs both arms of a divergent
//! branch, one after the other, with the value held. The estimate is the largest sum of the
//! widths of the values live at one instruction.
//!
//! **Widths are the lowered ones.** A value takes its type's 32-bit words (a narrow integer or
//! float one, a 64-bit one two, a vector its lanes', an aggregate its members'), and a pointer
//! two. A pointer loaded from memory or made from an integer, and every pointer derived from one,
//! is a wide pointer in the lowered module (`memory::roots`: four `u64`s, [`WIDE_POINTER_WORDS`]),
//! and the resident-set search that makes it holds [`SEARCH_WORDS`] more while it runs, at the
//! instruction that makes it.

use crate::air::ir::{BlockId, Body, CastOp, Idx, InstKind, Type, TypeId, TypeTable, Value};
use crate::air::Entry;
use crate::budget::{vec_bytes, Meter};
use crate::error::Error;
use crate::graph::LoopForest;

/// The 32-bit words of a wide pointer: `(root, at, host, size)`, four `u64`s.
pub const WIDE_POINTER_WORDS: u64 = 8;

/// The 32-bit words the resident-set search holds beside what is live at its call (`memory::roots`'s
/// `search`): the table's and the set's addresses, the address searched and the entry's Metal
/// address (two words each), the first root and the count (two each), and the candidate and the
/// count left (one each).
pub const SEARCH_WORDS: u64 = 14;

/// The passes over the body that find the wide pointers a phi carries around a loop: a phi whose
/// wide incoming value is defined later in layout order takes another pass, one per such phi in a
/// chain of them. A chain longer than this leaves its last phis counted at a pointer's two words.
const WIDE_PASSES: u32 = 8;

/// The largest number of 32-bit words the entry's values hold at once (the module docs). Its
/// work is linear in the body's instructions and operands, and in its values for each loop; its
/// tables are charged while they live.
pub fn peak_words(entry: &Entry, body: &Body, m: &mut Meter) -> Result<u64, Error> {
    let types = entry.module.types();
    let n = body.insts.len();
    let params: Vec<TypeId> = match entry.module.func(entry.func).map(|f| f.ty) {
        Some(ty) => match types.get(ty) {
            Some(Type::Function { params, .. }) => params.to_vec(),
            _ => Vec::new(),
        },
        None => Vec::new(),
    };
    // Per instruction: the last position reading it, and whether it is a wide pointer.
    let mut last: Vec<u32> = Vec::new();
    let mut wide: Vec<bool> = Vec::new();
    let mut arg_last: Vec<Option<u32>> = vec![None; params.len()];
    m.reserve_exact(&mut last, n)?;
    m.reserve_exact(&mut wide, n)?;
    let held = vec_bytes(&last) + vec_bytes(&wide) + vec_bytes(&arg_last) + vec_bytes(&params);
    m.charge(vec_bytes(&arg_last) + vec_bytes(&params))?;
    let r = (|| {
        last.extend((0..n).map(|i| u32::try_from(i).unwrap_or(u32::MAX)));
        wide.resize(n, false);
        let use_at = |v: Value, at: u32, last: &mut [u32], arg_last: &mut [Option<u32>]| match v {
            Value::Inst(i) => {
                if let Some(l) = last.get_mut(i.index()) {
                    *l = (*l).max(at);
                }
            }
            Value::Arg(a) => {
                if let Some(l) = arg_last.get_mut(a.index()) {
                    *l = Some(l.map_or(at, |x| x.max(at)));
                }
            }
            Value::Const(_) => {}
        };
        for (_, block) in body.blocks.iter() {
            for id in block.inst_ids() {
                m.tick(1)?;
                let Some(inst) = body.insts.get(id) else {
                    continue;
                };
                let at = u32::try_from(id.index()).unwrap_or(u32::MAX);
                match &inst.kind {
                    InstKind::Phi { incoming, .. } => {
                        for &(v, pred) in incoming.iter() {
                            let end = body
                                .blocks
                                .get(pred)
                                .map_or(at, |b| b.end.raw().saturating_sub(1));
                            use_at(v, end, &mut last, &mut arg_last);
                        }
                    }
                    kind => kind.for_each_value(|v| use_at(v, at, &mut last, &mut arg_last)),
                }
            }
        }
        // Wide pointers, to a fixpoint over phis whose incoming value comes later in layout (a
        // loop's back edge), at most `WIDE_PASSES` passes.
        let is_wide = |v: Value, wide: &[bool]| match v {
            Value::Inst(i) => wide.get(i.index()).copied().unwrap_or(false),
            _ => false,
        };
        let mut changed = true;
        let mut passes = 0;
        while changed && passes < WIDE_PASSES {
            changed = false;
            passes += 1;
            for (id, inst) in body.insts.iter() {
                m.tick(1)?;
                if wide.get(id.index()).copied().unwrap_or(true) {
                    continue;
                }
                let pointer = matches!(types.get(inst.ty), Some(Type::Ptr { .. }));
                let made = pointer
                    && match &inst.kind {
                        InstKind::Load { .. }
                        | InstKind::Cast {
                            op: CastOp::IntToPtr,
                            ..
                        } => true,
                        InstKind::Gep { base, .. } => is_wide(*base, &wide),
                        InstKind::Cast { v, .. } => is_wide(*v, &wide),
                        InstKind::Select { t, f, .. } => is_wide(*t, &wide) || is_wide(*f, &wide),
                        InstKind::Phi { incoming, .. } => {
                            incoming.iter().any(|&(v, _)| is_wide(v, &wide))
                        }
                        _ => false,
                    };
                if made {
                    wide[id.index()] = true;
                    changed = true;
                }
            }
        }
        // A value defined before a loop and read in it is live to the loop's end: each loop that
        // holds its last read, from the innermost out, whose first instruction follows its
        // definition.
        let loops = LoopForest::compute(&entry.cfg, m)?;
        let r = (|| {
            let mut spans = vec![(u32::MAX, 0u32); 0];
            m.reserve_exact(&mut spans, loops.len())?;
            spans.resize(loops.len(), (u32::MAX, 0));
            let mut block_of: Vec<BlockId> = Vec::new();
            m.reserve_exact(&mut block_of, n)?;
            for (b, block) in body.blocks.iter() {
                block_of.extend(block.inst_ids().map(|_| b));
            }
            for l in loops.loops() {
                for b in loops.blocks(l) {
                    m.tick(1)?;
                    if let Some(block) = body.blocks.get(b) {
                        let span = &mut spans[l.index()];
                        span.0 = span.0.min(block.start.raw());
                        span.1 = span.1.max(block.end.raw().saturating_sub(1));
                    }
                }
            }
            let extend = |def: u32, l: &mut u32, m: &mut Meter| -> Result<(), Error> {
                let mut lp = block_of.get(*l as usize).and_then(|&b| loops.innermost(b));
                while let Some(id) = lp {
                    m.tick(1)?;
                    let (lo, hi) = spans[id.index()];
                    if def < lo {
                        *l = (*l).max(hi);
                    }
                    lp = loops.parent(id);
                }
                Ok(())
            };
            for (def, l) in last.iter_mut().enumerate() {
                extend(u32::try_from(def).unwrap_or(u32::MAX), l, m)?;
            }
            for l in arg_last.iter_mut().flatten() {
                // A parameter is defined before the body's first instruction.
                let mut at = *l;
                let mut lp = block_of.get(at as usize).and_then(|&b| loops.innermost(b));
                while let Some(id) = lp {
                    m.tick(1)?;
                    at = at.max(spans[id.index()].1);
                    lp = loops.parent(id);
                }
                *l = at;
            }
            let bytes = vec_bytes(&spans) + vec_bytes(&block_of);
            m.release(bytes);
            Ok::<(), Error>(())
        })();
        m.release(loops.ledger_bytes());
        r?;
        // The words live at each position: a difference array, summed in order.
        let mut delta: Vec<i64> = Vec::new();
        m.reserve_exact(&mut delta, n + 1)?;
        delta.resize(n + 1, 0);
        let mut search = vec![false; 0];
        m.reserve_exact(&mut search, n)?;
        search.resize(n, false);
        for (id, inst) in body.insts.iter() {
            let i = id.index();
            let w = if wide[i] {
                WIDE_POINTER_WORDS
            } else {
                words(types, inst.ty, 0)
            };
            if w == 0 {
                continue;
            }
            let made_here = wide[i]
                && matches!(
                    inst.kind,
                    InstKind::Load { .. }
                        | InstKind::Cast {
                            op: CastOp::IntToPtr,
                            ..
                        }
                );
            search[i] = made_here;
            delta[i] += w as i64;
            delta[(last[i] as usize + 1).min(n)] -= w as i64;
        }
        for (a, l) in arg_last.iter().enumerate() {
            if let Some(l) = l {
                let w = params.get(a).map_or(0, |&t| words(types, t, 0)) as i64;
                delta[0] += w;
                delta[(*l as usize + 1).min(n)] -= w;
            }
        }
        let (mut live, mut peak) = (0i64, 0i64);
        for (i, d) in delta.iter().take(n).enumerate() {
            live += d;
            let transient = if search[i] { SEARCH_WORDS as i64 } else { 0 };
            peak = peak.max(live + transient);
        }
        m.release(vec_bytes(&delta) + vec_bytes(&search));
        Ok(u64::try_from(peak).unwrap_or(0))
    })();
    m.release(held);
    r
}

/// The 32-bit words a value of `ty` holds in registers (the module docs); 0 for `void`, a label
/// and the like. `depth` bounds the walk into nested aggregates.
fn words(types: &TypeTable, ty: TypeId, depth: u32) -> u64 {
    match types.get(ty) {
        Some(Type::Int(bits)) => u64::from(bits.div_ceil(32).max(1)),
        Some(Type::Half | Type::BFloat | Type::Float) => 1,
        Some(Type::Double) => 2,
        Some(Type::Ptr { .. }) => 2,
        Some(Type::Vector { elem, len }) => {
            u64::from(*len).saturating_mul(words(types, *elem, depth))
        }
        Some(Type::Array { elem, len }) if depth < 4 => {
            len.saturating_mul(words(types, *elem, depth + 1))
        }
        Some(Type::Struct { fields, .. }) if depth < 4 => fields
            .iter()
            .map(|&t| words(types, t, depth + 1))
            .fold(0u64, u64::saturating_add),
        _ => 0,
    }
}
