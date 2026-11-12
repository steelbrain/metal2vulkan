//! SCCP's incremental PHI meet against the full meet it replaced: both settle every value, block,
//! and edge alike and rebuild the same body, and the incremental meet's work is linear in a PHI's
//! entries where the full meet's is quadratic.

use std::collections::HashMap;

use super::*;
use crate::budget::Budget;
use crate::ir::build::ModuleBuilder;
use crate::ir::{
    AddrSpace, Align, BinOp, Block, FastMath, Inst, IntFlags, IntPredicate, Linkage, ModuleParts,
    Section, TypeId,
};

/// The PHI entries past which [`agrees_with_the_full_meet`] does not solve a body again: the
/// full meet is quadratic in a PHI's entries, and the tests here compare larger PHIs directly.
const FULL_MEET_ENTRIES: usize = 1 << 12;

/// The meets this thread has made so far ([`meet`] counts them under `cfg(test)`).
fn meets() -> u64 {
    MEETS.with(std::cell::Cell::get)
}

impl Solver<'_> {
    /// The meet of every entry of a PHI of block `b` whose edge is executable.
    fn full_value(&self, incoming: &[(Value, BlockId)], b: BlockId) -> Lat {
        let mut acc = Lat::Top;
        for &(v, p) in incoming {
            if self.edges.contains(&(p, b)) {
                acc = meet(acc, merged(self.lat(v)));
            }
        }
        acc
    }

    /// A PHI's value as SCCP once computed it: the meet of every entry whose edge is executable, at
    /// every visit, here also a work unit an entry.
    pub(super) fn full_phi(
        &self,
        incoming: &[(Value, BlockId)],
        b: BlockId,
        m: &mut Meter,
    ) -> Result<Lat, Error> {
        m.tick(units(incoming.len()))?;
        Ok(self.full_value(incoming, b))
    }

    /// With `check_slots`, assert that the value PHI `i` of block `b` reads from its slot is the
    /// full meet of its executable entries at this visit, without counting the check's meets.
    pub(super) fn check_slot(&self, i: InstId, b: BlockId, value: Lat) {
        if !self.check_slots {
            return;
        }
        let Some(InstKind::Phi { incoming, .. }) = self.body.insts.get(i).map(|x| &x.kind) else {
            panic!("{i:?} is not a PHI");
        };
        let counted = meets();
        let full = self.full_value(incoming, b);
        MEETS.with(|n| n.set(counted));
        assert_eq!(
            value, full,
            "PHI {i:?} of {b:?}: its slot is not the full meet"
        );
    }
}

/// What a solve settles: every instruction's value, every block's executability, and the
/// executable edges.
#[derive(Debug, PartialEq)]
struct Settled {
    val: Vec<Lat>,
    exec: Vec<bool>,
    edges: Vec<(BlockId, BlockId)>,
}

/// One SCCP run: what it settled (`None` when the solver was not built), its result, and its
/// work units.
struct Solved {
    settled: Option<Result<Settled, Error>>,
    out: Result<Rebuilt, Error>,
    work: u64,
}

/// Run SCCP over `body` with the incremental or (`full`) the full meet, on a meter of its own
/// without caps; with `check`, every visit of a PHI checks its slot against the full meet.
fn solve_with(body: &Body, cx: &mut Context<'_>, full: bool, check: bool) -> Solved {
    let mut m = Meter::with_caps(&Budget::default(), u64::MAX, u64::MAX);
    let mut settled = None;
    let out = scoped(&mut m, |m, held| {
        let mut s = Solver::new(body, cx, m, held)?;
        s.full_meet = full;
        s.check_slots = check;
        let solved = s.solve(cx, m, held);
        let mut edges: Vec<(BlockId, BlockId)> = s.edges.iter().copied().collect();
        edges.sort_unstable();
        settled = Some(solved.clone().map(|()| Settled {
            val: s.val.clone(),
            exec: s.exec.clone(),
            edges,
        }));
        solved?;
        s.check(cx)?;
        let (edit, reads) = s.edit(cx, m, held)?;
        Ok(Rebuilt {
            body: apply(body, &edit, &|c| c, m)?,
            reads,
        })
    });
    Solved {
        settled,
        out,
        work: m.work(),
    }
}

fn entries(body: &Body) -> usize {
    body.insts
        .values()
        .map(|inst| match &inst.kind {
            InstKind::Phi { incoming, .. } => incoming.len(),
            _ => 0,
        })
        .sum()
}

/// Solve `body` with both meets and assert they settle alike and rebuild `out`, SCCP's result,
/// unless `out` is a budget refusal (the solves here run uncapped). [`run`] calls this for every
/// SCCP a unit test runs, so every canonicalization test is also this check.
pub(super) fn agrees_with_the_full_meet(
    body: &Body,
    cx: &mut Context<'_>,
    out: &Result<Rebuilt, Error>,
) {
    if matches!(
        out,
        Err(Error::LimitExceeded(_) | Error::Budget(_) | Error::Cancelled)
    ) || entries(body) > FULL_MEET_ENTRIES
    {
        return;
    }
    let incremental = solve_with(body, cx, false, true);
    let full = solve_with(body, cx, true, false);
    assert_eq!(
        incremental.settled, full.settled,
        "the incremental and the full meet settle differently"
    );
    assert_eq!(&incremental.out, out, "a second incremental solve differs");
    assert_eq!(&full.out, out, "the full meet rebuilds a different body");
}

/// The types, constants, and globals the test bodies use, and the function constants.
struct Fixture {
    parts: ModuleParts,
    fcs: HashMap<GlobalId, FcState>,
    void: TypeId,
    i1: TypeId,
    i32t: TypeId,
    i64t: TypeId,
    dev: TypeId,
    /// `i32` constants 0 to 3.
    ints: Vec<Value>,
    /// `i1` false, true, and `undef`.
    bools: Vec<Value>,
    /// Addresses of two unsupplied function constants (indices 3 and 1, so a meet of both
    /// keeps the lower) and of a constant global holding 2.
    addrs: Vec<Value>,
}

impl Fixture {
    fn new() -> Fixture {
        let mut m = ModuleBuilder::new();
        let void = m.void();
        let i1 = m.int(1);
        let i32t = m.int(32);
        let i64t = m.int(64);
        let dev = m.ptr(AddrSpace::Device);
        let ints = (0..4).map(|v| m.const_int(i32t, v)).collect();
        let undef = Value::Const(m.constant(i1, Const::Undef));
        let bools = vec![m.const_int(i1, 0), m.const_int(i1, 1), undef];
        let mut fcs = HashMap::new();
        let mut addrs = Vec::new();
        for (name, index) in [("a", 3), ("b", 1)] {
            let undef = m.constant(i32t, Const::Undef);
            let name = m.sym(name);
            let g = m.global(Global {
                name,
                value_ty: i32t,
                space: AddrSpace::Constant,
                constant: true,
                init: Some(undef),
                externally_initialized: true,
                linkage: Linkage::LinkOnceOdr,
                align: None,
                section: Some(Section::FcInitializer),
                visibility: Default::default(),
                unnamed_addr: Default::default(),
            });
            fcs.insert(
                g,
                FcState {
                    index: Undefined::Constant(index),
                    supplied: false,
                    decl: true,
                },
            );
            addrs.push(m.global_addr(g));
        }
        let two = m.constant(i32t, Const::Int(2));
        let name = m.sym("k");
        let k = m.global(Global {
            name,
            value_ty: i32t,
            space: AddrSpace::Constant,
            constant: true,
            init: Some(two),
            externally_initialized: false,
            linkage: Linkage::Internal,
            align: None,
            section: None,
            visibility: Default::default(),
            unnamed_addr: Default::default(),
        });
        addrs.push(m.global_addr(k));
        Fixture {
            parts: m.finish_parts().expect("fixture"),
            fcs,
            void,
            i1,
            i32t,
            i64t,
            dev,
            ints,
            bools,
            addrs,
        }
    }

    fn context(&mut self) -> Context<'_> {
        Context {
            folder: Folder {
                types: &self.parts.types,
                consts: &mut self.parts.consts,
                held: 0,
            },
            globals: &self.parts.globals,
            fcs: &self.fcs,
            name: DiagName::new("k"),
        }
    }

    /// Solve `body` with both meets, assert they agree, and return the incremental run.
    fn agree(&mut self, body: &Body) -> Solved {
        let mut cx = self.context();
        let incremental = solve_with(body, &mut cx, false, true);
        let full = solve_with(body, &mut cx, true, false);
        assert_eq!(incremental.settled, full.settled);
        assert_eq!(incremental.out, full.out);
        incremental
    }
}

fn load(ty: TypeId, ptr: Value) -> Inst {
    Inst {
        ty,
        kind: InstKind::Load {
            ptr,
            align: Align::from_bytes(4),
            volatile: false,
            atomic: None,
            range: None,
        },
    }
}

fn binop(ty: TypeId, op: BinOp, a: Value, b: Value) -> Inst {
    Inst {
        ty,
        kind: InstKind::BinOp {
            op,
            flags: IntFlags::NONE,
            fmf: FastMath::NONE,
            a,
            b,
        },
    }
}

/// Lay `blocks` (each a list of instructions, the terminator last) out as a body.
fn lay_out(blocks: Vec<Vec<Inst>>) -> Body {
    let mut m = Meter::with_caps(&Budget::default(), u64::MAX, u64::MAX);
    let mut body = Body::default();
    for insts in blocks {
        let start = InstId::new(u32::try_from(body.insts.len()).expect("ids"));
        for inst in insts {
            body.insts.push(inst, &mut m).expect("instruction");
        }
        let end = InstId::new(u32::try_from(body.insts.len()).expect("ids"));
        body.blocks
            .push(
                Block {
                    start,
                    end,
                    name: None,
                },
                &mut m,
            )
            .expect("block");
    }
    body
}

/// The body of `src/tests/engine_metered_walks.rs`'s `phi_kernel(incoming, uses)`, laid out as
/// the decoder lays it out: a switch on a word loaded from `out` (`%a0`) to `incoming` blocks,
/// each a GEP into `out`; one PHI of the `incoming` pointers; then `uses` blocks, each reading
/// through the PHI and adding to a sum that the last stores to `out`. `entry` is the value each
/// arm's PHI entry carries in place of its GEP, if given.
fn phi_kernel(f: &Fixture, incoming: u32, uses: u32, entry: Option<&dyn Fn(u32) -> Value>) -> Body {
    let n = incoming.max(1);
    let out = Value::Arg(ArgIdx::new(0));
    let t = Value::Arg(ArgIdx::new(1));
    let block = |k: u32| BlockId::new(k);
    // Block 0: the switch; 1..=n: the arms; n + 1: the join; then the uses; then the store.
    let join = block(n + 1);
    let mut blocks = vec![vec![
        load(f.i32t, out),
        Inst {
            ty: f.void,
            kind: InstKind::Switch {
                v: Value::Inst(InstId::new(0)),
                default: block(1),
                cases: (1..n).map(|i| (u64::from(i), block(i + 1))).collect(),
                hints: None,
            },
        },
    ]];
    let mut next = 2u32;
    let mut arms = Vec::new();
    for i in 0..n {
        let index = Value::Const(
            f.parts
                .consts
                .lookup(&Constant {
                    ty: f.i64t,
                    value: Const::Int(u64::from(i)),
                })
                .unwrap_or(ConstId::new(0)),
        );
        let gep = Inst {
            ty: f.dev,
            kind: InstKind::Gep {
                src_ty: f.i32t,
                base: out,
                indices: Box::new([index]),
                inbounds: true,
            },
        };
        let value = entry.map_or(Value::Inst(InstId::new(next)), |e| e(i));
        arms.push((value, block(i + 1)));
        blocks.push(vec![
            gep,
            Inst {
                ty: f.void,
                kind: InstKind::Br {
                    dest: join,
                    hints: None,
                },
            },
        ]);
        next += 2;
    }
    let phi = InstId::new(next);
    let mut sum = InstId::new(next + 1);
    blocks.push(vec![
        Inst {
            ty: f.dev,
            kind: InstKind::Phi {
                incoming: arms.into_boxed_slice(),
                fmf: FastMath::NONE,
            },
        },
        binop(f.i32t, BinOp::Add, t, f.ints[0]),
        Inst {
            ty: f.void,
            kind: InstKind::Br {
                dest: block(n + 2),
                hints: None,
            },
        },
    ]);
    next += 3;
    for j in 0..uses {
        let gep = Inst {
            ty: f.dev,
            kind: InstKind::Gep {
                src_ty: f.i32t,
                base: Value::Inst(phi),
                indices: Box::new([f.ints[(j % 4) as usize]]),
                inbounds: true,
            },
        };
        blocks.push(vec![
            gep,
            load(f.i32t, Value::Inst(InstId::new(next))),
            binop(
                f.i32t,
                BinOp::Add,
                Value::Inst(sum),
                Value::Inst(InstId::new(next + 1)),
            ),
            Inst {
                ty: f.void,
                kind: InstKind::Br {
                    dest: block(n + 3 + j),
                    hints: None,
                },
            },
        ]);
        sum = InstId::new(next + 2);
        next += 4;
    }
    blocks.push(vec![
        Inst {
            ty: f.void,
            kind: InstKind::Store {
                val: Value::Inst(sum),
                ptr: out,
                align: Align::from_bytes(4),
                volatile: false,
                atomic: None,
            },
        },
        Inst {
            ty: f.void,
            kind: InstKind::Ret(None),
        },
    ]);
    lay_out(blocks)
}

/// A fixture whose `i64` constants 0 to `n - 1` exist, for [`phi_kernel`]'s GEPs.
fn kernel_fixture(n: u32) -> Fixture {
    let mut f = Fixture::new();
    let mut m = Meter::with_caps(&Budget::default(), u64::MAX, u64::MAX);
    for i in 0..n.max(1) {
        f.parts
            .consts
            .intern(
                Constant {
                    ty: f.i64t,
                    value: Const::Int(u64::from(i)),
                },
                &mut m,
            )
            .expect("constant");
    }
    f
}

/// SCCP meets a PHI's entries once each, as their edges open and their values change, so its work
/// grows linearly in a PHI's entries, and every meet is metered. The work is counted twice: by the
/// meter, and by the meets themselves, which [`meet`] counts under `cfg(test)` whoever calls it,
/// metered or not. Twice the entries make under 2.2 times the meets and cost under 2.2 times the
/// units, and this kernel makes no more meets than it is charged units. The full meet SCCP ran
/// previously (every executable entry at every one of the PHI's visits) makes over 3.5 times the
/// meets. With that unmetered arm put back in `Solver::visit` in place of the slot's read, this
/// test fails on the meets.
#[test]
fn a_phi_costs_work_linear_in_its_entries() {
    let work = |n: u32, full: bool| {
        let mut f = kernel_fixture(n);
        let body = phi_kernel(&f, n, 16, None);
        let mut cx = f.context();
        let before = meets();
        let s = solve_with(&body, &mut cx, full, false);
        assert!(s.out.is_ok(), "phi_kernel({n}): {:?}", s.out.err());
        (s.work, meets() - before)
    };
    let ((small, small_meets), (large, large_meets)) = (work(1 << 10, false), work(1 << 11, false));
    assert!(
        large_meets * 10 < small_meets * 22,
        "incremental: {large_meets} meets for 2^11 entries, {small_meets} for 2^10"
    );
    assert!(
        large * 10 < small * 22,
        "incremental: {large} units for 2^11 entries, {small} for 2^10"
    );
    assert!(
        large_meets <= large && small_meets <= small,
        "incremental: {large_meets} meets on {large} units, {small_meets} on {small}"
    );
    let ((_, small_meets), (_, large_meets)) = (work(1 << 10, true), work(1 << 11, true));
    assert!(
        large_meets * 10 > small_meets * 35,
        "full: {large_meets} meets for 2^11 entries, {small_meets} for 2^10"
    );
}

/// A value that changes after its edge opened is met into the PHI again. A loop counter `%i = phi
/// [0, entry], [%j, latch]` exits on an argument, so the exit edge opens while `%i` is still 0 and
/// the exit's `%r = phi [%i, header]` meets 0; `%i` then becomes overdefined when the latch edge
/// opens, and `%r` must meet that through `Phis::changed`: every visit's slot is the full meet, and
/// `%r` ends overdefined, not 0.
#[test]
fn a_value_that_changes_after_its_edge_opened_is_met_again() {
    let mut f = Fixture::new();
    let a1 = Value::Arg(ArgIdx::new(1));
    // entry 0: br header; header 1: %i (1), br a1, latch, exit; latch 2: %j (3), br header;
    // exit 3: %r (5), ret.
    let (i, j, r) = (InstId::new(1), InstId::new(3), InstId::new(5));
    let br = |dest: u32| Inst {
        ty: f.void,
        kind: InstKind::Br {
            dest: BlockId::new(dest),
            hints: None,
        },
    };
    let body = lay_out(vec![
        vec![br(1)],
        vec![
            Inst {
                ty: f.i32t,
                kind: InstKind::Phi {
                    incoming: Box::new([
                        (f.ints[0], BlockId::new(0)),
                        (Value::Inst(j), BlockId::new(2)),
                    ]),
                    fmf: FastMath::NONE,
                },
            },
            Inst {
                ty: f.void,
                kind: InstKind::CondBr {
                    c: a1,
                    t: BlockId::new(2),
                    f: BlockId::new(3),
                    hints: None,
                },
            },
        ],
        vec![binop(f.i32t, BinOp::Add, Value::Inst(i), f.ints[1]), br(1)],
        vec![
            Inst {
                ty: f.i32t,
                kind: InstKind::Phi {
                    incoming: Box::new([(Value::Inst(i), BlockId::new(1))]),
                    fmf: FastMath::NONE,
                },
            },
            Inst {
                ty: f.void,
                kind: InstKind::Ret(None),
            },
        ],
    ]);
    let solved = f.agree(&body);
    let Some(Ok(settled)) = solved.settled else {
        panic!("not solved: {:?}", solved.out);
    };
    assert_eq!(settled.val.get(r.index()), Some(&Lat::Bottom));
    assert_eq!(settled.val.get(i.index()), Some(&Lat::Bottom));
}

/// The phi_kernel family settles and rebuilds alike under both meets: pointer entries
/// (overdefined), one constant in every entry (folded), different constants, unsupplied
/// function constants (the lower index wins), a mix of a constant and a tainted value, and a
/// PHI some entries of which never execute (a switch on a constant).
#[test]
fn the_phi_kernel_family_settles_alike_under_both_meets() {
    for n in [1u32, 2, 3, 17, 64, 300] {
        let f = kernel_fixture(n);
        let ints = f.ints.clone();
        let addrs = f.addrs.clone();
        let one = move |_: u32| ints[1];
        let ints = f.ints.clone();
        let mixed = move |i: u32| ints[(i % 4) as usize];
        let fc = move |i: u32| addrs[(i % 2) as usize];
        let ints = f.ints.clone();
        let addrs = f.addrs.clone();
        let tainted_last = move |i: u32| if i + 1 == n { addrs[0] } else { ints[2] };
        let entries: [Option<&dyn Fn(u32) -> Value>; 5] = [
            None,
            Some(&one),
            Some(&mixed),
            Some(&fc),
            Some(&tainted_last),
        ];
        for entry in entries {
            for uses in [0u32, 1, 3] {
                let mut f = kernel_fixture(n);
                let body = phi_kernel(&f, n, uses, entry);
                f.agree(&body);
            }
        }
        // The switch on a constant: one arm's entry executes.
        let mut f = kernel_fixture(n);
        let mut body = phi_kernel(&f, n, 2, None);
        if let Some(InstKind::Switch { v, .. }) =
            body.insts.get_mut(InstId::new(1)).map(|i| &mut i.kind)
        {
            *v = f.ints[(n % 4) as usize];
        }
        f.agree(&body);
    }
}

/// A small deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A value in `0..n` (`n > 0`).
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len())]
    }
}

/// What a generated block's terminator does.
enum Term {
    Ret,
    Br(u32),
    CondBr(u32, u32),
    Switch(u32, Vec<(u64, u32)>),
}

/// A random function body over `f`'s constants and globals and the arguments `(i32, i1)`, with
/// leading PHIs whose entries follow the predecessor edges (one per edge, parallel edges
/// alike), loops, switches with parallel cases, branches on constants, `undef`, arguments, and
/// computed values, and values tainted by unsupplied function constants. Operands follow
/// dominance loosely (a block reads its own earlier values, the entry block's, and its PHIs; a
/// PHI entry reads its predecessor's), so most bodies rebuild and some are refused.
fn random_body(f: &Fixture, rng: &mut Rng) -> Body {
    let nblocks = 2 + rng.below(9) as u32;
    let mut terms = Vec::new();
    for b in 0..nblocks {
        let target = |rng: &mut Rng| 1 + rng.below(nblocks as usize - 1) as u32;
        terms.push(match rng.below(10) {
            0 if b > 0 => Term::Ret,
            _ if b + 1 == nblocks && rng.below(2) == 0 => Term::Ret,
            0..=2 => Term::Br(target(rng)),
            3..=6 => Term::CondBr(target(rng), target(rng)),
            _ => {
                let cases = (0..1 + rng.below(5))
                    .map(|_| (rng.below(4) as u64, target(rng)))
                    .collect::<Vec<_>>();
                // Case values are distinct.
                let mut seen = Vec::new();
                let cases = cases
                    .into_iter()
                    .filter(|&(v, _)| {
                        let fresh = !seen.contains(&v);
                        seen.push(v);
                        fresh
                    })
                    .collect();
                Term::Switch(target(rng), cases)
            }
        });
    }
    // Each block's predecessor edges, in order (parallel edges repeated).
    let mut preds: Vec<Vec<u32>> = vec![Vec::new(); nblocks as usize];
    for (b, t) in terms.iter().enumerate() {
        let succs: Vec<u32> = match t {
            Term::Ret => Vec::new(),
            Term::Br(d) => vec![*d],
            Term::CondBr(x, y) => vec![*x, *y],
            Term::Switch(d, cases) => std::iter::once(*d)
                .chain(cases.iter().map(|&(_, s)| s))
                .collect(),
        };
        for s in succs {
            preds[s as usize].push(b as u32);
        }
    }
    // Shapes: per block its PHI count and its other instructions' kinds (0 arithmetic, 1
    // compare, 2 select, 3 load), then ids in layout order.
    let mut nphi = Vec::new();
    let mut shapes: Vec<Vec<usize>> = Vec::new();
    let mut start = Vec::new();
    let mut next = 0u32;
    for b in 0..nblocks as usize {
        nphi.push(if preds[b].is_empty() {
            0
        } else {
            rng.below(4) as u32
        });
        shapes.push((0..rng.below(5)).map(|_| rng.below(4)).collect());
        start.push(next);
        next += nphi[b] + shapes[b].len() as u32 + 1;
    }
    let inst = |b: usize, k: u32| Value::Inst(InstId::new(start[b] + k));
    // The block's values of each type (`i32`: its PHIs and arithmetic, selects, and loads;
    // `i1`: its compares), the first `upto` other instructions only.
    let values = |b: usize, upto: usize, wide: bool| -> Vec<Value> {
        let mut out = Vec::new();
        if wide {
            out.extend((0..nphi[b]).map(|k| inst(b, k)));
        }
        for (k, &s) in shapes[b].iter().enumerate().take(upto) {
            if (s == 1) != wide {
                out.push(inst(b, nphi[b] + k as u32));
            }
        }
        out
    };
    let a0 = Value::Arg(ArgIdx::new(0));
    let a1 = Value::Arg(ArgIdx::new(1));
    let mut blocks = Vec::new();
    for b in 0..nblocks as usize {
        let mut insts = Vec::new();
        let entry_wide = if b == 0 {
            Vec::new()
        } else {
            values(0, usize::MAX, true)
        };
        let entry_narrow = if b == 0 {
            Vec::new()
        } else {
            values(0, usize::MAX, false)
        };
        // PHIs: per predecessor block one value, repeated on its parallel edges.
        for _ in 0..nphi[b] {
            let mut chosen: Vec<(u32, Value)> = Vec::new();
            let mut incoming = Vec::new();
            for &p in &preds[b] {
                let v = match chosen.iter().find(|&&(q, _)| q == p) {
                    Some(&(_, v)) => v,
                    None => {
                        let mut pool = f.ints.clone();
                        pool.push(a0);
                        pool.extend(
                            f.addrs
                                .iter()
                                .take(2)
                                .copied()
                                .filter(|_| rng.below(3) == 0),
                        );
                        pool.extend(values(p as usize, usize::MAX, true));
                        pool.extend(entry_wide.iter().copied());
                        let v = rng.pick(&pool);
                        chosen.push((p, v));
                        v
                    }
                };
                incoming.push((v, BlockId::new(p)));
            }
            insts.push(Inst {
                ty: f.i32t,
                kind: InstKind::Phi {
                    incoming: incoming.into_boxed_slice(),
                    fmf: FastMath::NONE,
                },
            });
        }
        for (k, &shape) in shapes[b].iter().enumerate() {
            let mut wide = f.ints.clone();
            wide.push(a0);
            wide.extend(values(b, k, true));
            wide.extend(entry_wide.iter().copied());
            let mut narrow = f.bools[..2].to_vec();
            narrow.push(a1);
            narrow.extend(values(b, k, false));
            narrow.extend(entry_narrow.iter().copied());
            insts.push(match shape {
                0 => {
                    let op = rng.pick(&[BinOp::Add, BinOp::Mul, BinOp::Xor]);
                    binop(f.i32t, op, rng.pick(&wide), rng.pick(&wide))
                }
                1 => Inst {
                    ty: f.i1,
                    kind: InstKind::ICmp {
                        pred: rng.pick(&[IntPredicate::Eq, IntPredicate::Ult]),
                        a: rng.pick(&wide),
                        b: rng.pick(&wide),
                    },
                },
                2 => Inst {
                    ty: f.i32t,
                    kind: InstKind::Select {
                        c: rng.pick(&narrow),
                        t: rng.pick(&wide),
                        f: rng.pick(&wide),
                        fmf: FastMath::NONE,
                    },
                },
                _ => load(f.i32t, rng.pick(&f.addrs)),
            });
        }
        let mut wide = f.ints.clone();
        wide.push(a0);
        wide.extend(values(b, usize::MAX, true));
        wide.extend(entry_wide.iter().copied());
        let mut narrow = f.bools[..2].to_vec();
        narrow.push(a1);
        if rng.below(12) == 0 {
            narrow.push(f.bools[2]);
        }
        narrow.extend(values(b, usize::MAX, false));
        narrow.extend(entry_narrow.iter().copied());
        let block = BlockId::new;
        insts.push(Inst {
            ty: f.void,
            kind: match &terms[b] {
                Term::Ret => InstKind::Ret(None),
                Term::Br(d) => InstKind::Br {
                    dest: block(*d),
                    hints: None,
                },
                Term::CondBr(x, y) => InstKind::CondBr {
                    c: rng.pick(&narrow),
                    t: block(*x),
                    f: block(*y),
                    hints: None,
                },
                Term::Switch(d, cases) => InstKind::Switch {
                    v: rng.pick(&wide),
                    default: block(*d),
                    cases: cases.iter().map(|&(v, s)| (v, block(s))).collect(),
                    hints: None,
                },
            },
        });
        blocks.push(insts);
    }
    lay_out(blocks)
}

/// Random bodies (loops, switches with parallel edges, constant and `undef` branches,
/// function-constant taint through PHIs) settle every value, block, and edge alike and rebuild
/// alike under both meets; most rebuild, some are refused alike.
#[test]
fn random_bodies_settle_alike_under_both_meets() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut rebuilt = 0;
    let mut phis = 0;
    for _ in 0..600 {
        let mut f = Fixture::new();
        let body = random_body(&f, &mut rng);
        phis += entries(&body);
        if f.agree(&body).out.is_ok() {
            rebuilt += 1;
        }
    }
    assert!(rebuilt > 150, "only {rebuilt} of 600 bodies rebuilt");
    assert!(phis > 1000, "only {phis} PHI entries generated");
}

/// A PHI after its block's first other instruction is not a verified body's; SCCP, which gives a
/// block's leading PHIs their slots, refuses it as an invariant violation.
#[test]
fn a_phi_after_another_instruction_is_an_invariant_violation() {
    let mut f = Fixture::new();
    let body = lay_out(vec![
        vec![Inst {
            ty: f.void,
            kind: InstKind::Br {
                dest: BlockId::new(1),
                hints: None,
            },
        }],
        vec![
            binop(f.i32t, BinOp::Add, f.ints[1], f.ints[2]),
            Inst {
                ty: f.i32t,
                kind: InstKind::Phi {
                    incoming: Box::new([(f.ints[1], BlockId::new(0))]),
                    fmf: FastMath::NONE,
                },
            },
            Inst {
                ty: f.void,
                kind: InstKind::Ret(None),
            },
        ],
    ]);
    let mut cx = f.context();
    let s = solve_with(&body, &mut cx, false, true);
    assert!(
        matches!(&s.out, Err(Error::Internal(e)) if e.to_string().contains("PHI after")),
        "{:?}",
        s.out
    );
}

/// SCCP's worklists are on the ledger. One block computes `%x` from an argument, uses it in `n`
/// instructions, and switches on it over `n` cases to one block: visiting `%x` pushes its `n + 1`
/// users on top of the block's `n + 3` instructions, and the switch pushes `n + 1` edges. Their
/// capacity, at least `(2n + 4)` instructions and `(n + 1)` edges, is charged while SCCP solves
/// (previously, only the block's instructions were).
#[test]
fn the_worklists_are_charged() {
    let n = 1usize << 10;
    let mut f = Fixture::new();
    let a0 = Value::Arg(ArgIdx::new(0));
    let x = Value::Inst(InstId::new(0));
    let mut insts = vec![binop(f.i32t, BinOp::Add, a0, f.ints[1])];
    insts.extend((0..n).map(|_| binop(f.i32t, BinOp::Add, x, f.ints[1])));
    insts.push(Inst {
        ty: f.void,
        kind: InstKind::Switch {
            v: x,
            default: BlockId::new(1),
            cases: (0..n as u64).map(|v| (v, BlockId::new(1))).collect(),
            hints: None,
        },
    });
    let body = lay_out(vec![
        insts,
        vec![Inst {
            ty: f.void,
            kind: InstKind::Ret(None),
        }],
    ]);
    let mut cx = f.context();
    let mut m = Meter::with_caps(&Budget::default(), u64::MAX, u64::MAX);
    let mut held = 0usize;
    let mut s = Solver::new(&body, &cx, &mut m, &mut held).expect("solver");
    let before = m.ledger();
    s.solve(&mut cx, &mut m, &mut held).expect("solve");
    let charged = m.ledger_peak() - before;
    let insts = (2 * n + 4) * std::mem::size_of::<InstId>();
    let edges = (n + 1) * std::mem::size_of::<(BlockId, BlockId)>();
    assert!(
        charged >= (insts + edges) as u64,
        "{charged} bytes charged, {insts} + {edges} expected"
    );
}

/// SCCP's rebuild does not hold the solver: [`run`] ends the solver's scope with its edit, so the
/// tables it solved with are back in the ledger before the new body is built. On a body that
/// folds nothing (a chain of adds of two arguments), the run's ledger peak stays below what the
/// solver holds after solving plus what the edit and its rebuild hold alone, the peak it reached
/// when the rebuild ran inside the solver's scope (a 1.9M-instruction body passed the ledger
/// there).
#[test]
fn the_rebuild_does_not_hold_the_solver() {
    let n = 1usize << 12;
    let mut f = Fixture::new();
    let (a0, a1) = (Value::Arg(ArgIdx::new(0)), Value::Arg(ArgIdx::new(1)));
    let mut insts = vec![binop(f.i32t, BinOp::Add, a0, a1)];
    for k in 1..n {
        let prev = Value::Inst(InstId::new(u32::try_from(k - 1).expect("ids")));
        insts.push(binop(f.i32t, BinOp::Add, prev, a1));
    }
    insts.push(Inst {
        ty: f.void,
        kind: InstKind::Ret(None),
    });
    let body = lay_out(vec![insts]);
    let uncapped = || Meter::with_caps(&Budget::default(), u64::MAX, u64::MAX);
    let mut cx = f.context();

    let mut m = uncapped();
    let mut held = 0usize;
    let mut s = Solver::new(&body, &cx, &mut m, &mut held).expect("solver");
    s.solve(&mut cx, &mut m, &mut held).expect("solve");
    s.check(&cx).expect("check");
    let solver = m.ledger();

    let mut m = uncapped();
    let mut held = 0usize;
    let edit = BodyEdit::keep_all(&body, &mut m, &mut held).expect("edit");
    apply(&body, &edit, &|c| c, &mut m).expect("rebuild");
    let rebuild = m.ledger_peak();

    let mut m = uncapped();
    let out = run(&body, &mut cx, &mut m).expect("sccp");
    assert_eq!(out.body.insts.len(), body.insts.len(), "nothing folds");
    let peak = m.ledger_peak();
    assert!(
        peak < solver + rebuild,
        "SCCP peaked at {peak} bytes: the solver holds {solver}, the edit and rebuild {rebuild}"
    );
}
