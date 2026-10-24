//! Invocations, call frames, SIMT subgroups with maximal reconvergence, and the barrier
//! scheduler.
//!
//! **Structured order.** Each function's blocks are numbered by a depth-first reverse postorder
//! that explores a header's merge block (and a loop's continue block) before the construct's
//! other successors. Every forward edge then goes to a larger number, and each construct's merge
//! is numbered after every block of the construct, so the lanes at the smallest position are
//! always the ones that are behind: running them first makes diverged lanes meet again at the
//! merge, and keeps every lane of a loop in the same iteration (only continue blocks branch back).
//!
//! **Position.** A lane's position is its call stack, outermost first, as (structured block
//! number, instruction index) per frame; positions compare lexicographically, so a lane about to
//! execute a call sorts before the lanes already inside it. The active tangle of a subgroup step
//! is every running lane at the smallest position; they execute one instruction (or terminator)
//! together.

use std::borrow::Cow;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;

use crate::pipeline::sym::{SpecKey, VaryingKey};
use crate::spirv::ids::{BlockId, ConstId, FuncId, GlobalId, TypeId, Value, VarId};
use crate::spirv::ir::{
    ExecMode, ExecModel, GlobalKind, Inst, InstNode, Scope, StorageClass, Terminator, TypedModule,
    ValueOrigin,
};

use super::eval::{self, Flow};
use super::layout;
use super::memory::{MemoryKind, Region, Shared};
use super::value::{Ptr, Root, Tys, Val};
use super::{deriv, group};
use super::{
    malformed, BarrierRule, Config, IResult, InterpError, InvocationStats, Order, Outcome,
    Resources, Site, TrapKind, Unmodelled,
};

/// Where a module-scope variable lives.
#[derive(Clone, Debug)]
enum GRoot {
    /// In each invocation (`Private`, `Input`): see [`Lane::globals`].
    Lane,
    /// `Workgroup` region `n`.
    Wg(u32),
    /// Resource buffer `n`.
    Buf(u32),
    /// A descriptor array of buffers whose element 0 is resource buffer `n`.
    BufArray(u32),
    /// Bound image `n`.
    Image(u32),
    Push,
    /// Not reachable in the model; using it is this error.
    Missing(InterpError),
}

/// The module and everything derived from it once per run.
pub(super) struct Prog<'m> {
    pub m: &'m TypedModule,
    pub t: Tys<'m>,
    pub consts: Vec<Val>,
    /// Structured order number of each block within its function (`u32::MAX` if unreachable).
    order: Vec<u32>,
    /// The loop index of each block that heads a loop, by block.
    loop_of: Vec<Option<usize>>,
    globals: Vec<GRoot>,
    /// The bound images (by index) whose variable is decorated `Volatile`: a load of one never
    /// returns the value of an earlier out-of-bounds store ([`Lane::oob_stores`]).
    volatile: HashSet<u32>,
    /// The length of each runtime descriptor array of images (a set-1 heap array), by its first
    /// bound image: its consecutive images of the slot.
    image_arrays: HashMap<u32, u64>,
    /// The length of each descriptor array of buffers (a set-1 heap array of storage buffers), by
    /// its first bound buffer: its consecutive buffers of the slot.
    buffer_arrays: HashMap<u32, u64>,
    layout: FrameLayout,
    /// The encoded initializer of each `Private` global, by global, filled by the first
    /// invocation that allocates it ([`new_lane`]). The encoding depends on the module and the
    /// run's constants alone, so every other invocation copies these bytes instead of encoding
    /// the constant again (a lookup table's initializer was a quarter of a dispatch's time).
    private_init: RefCell<Vec<Option<Vec<u8>>>>,
    /// The bytes of each user interface input the run supplies ([`super::run_with_inputs`]), by
    /// relocation key; every invocation reads the same bytes.
    inputs: BTreeMap<VaryingKey, Vec<u8>>,
}

/// Where a frame keeps each function's values and variables, computed once per run: each value
/// an instruction or phi defines is numbered densely within its function, so a frame holds its
/// values in a vector indexed by that slot, and finds each variable's allocation at its position
/// past the frame's arena mark, instead of hashing a value or variable id at every use.
struct FrameLayout {
    /// Each value's function and slot in that function's frame (`Frame::vals`); `NO_SLOT`
    /// for a value no frame holds.
    slot: Vec<(u32, u32)>,
    /// The number of slots of each function's frame.
    slots: Vec<u32>,
    /// Each `Function` variable's function and its position among that function's variables,
    /// which is its allocation's offset from the frame's arena mark (`Lane::call`).
    var_at: Vec<(u32, u32)>,
}

/// The slot of a value that no frame holds (a constant, global, variable, or parameter).
const NO_SLOT: (u32, u32) = (u32::MAX, u32::MAX);

impl<'m> Prog<'m> {
    fn new(
        m: &'m TypedModule,
        res: &Resources,
        images: &[super::Image],
        spec: &'m BTreeMap<SpecKey, u64>,
        inputs: BTreeMap<VaryingKey, Vec<u8>>,
    ) -> IResult<Prog<'m>> {
        let t = Tys { m, spec };
        let consts = t.constants(&res.spec)?;
        let mut order = vec![u32::MAX; m.blocks().len()];
        for f in m.functions() {
            structured_order(m, f.blocks.first().copied(), &mut order)?;
        }
        let mut loop_of = vec![None; m.blocks().len()];
        for (i, l) in m.loops().iter().enumerate() {
            if let Some(x) = loop_of.get_mut(l.header.index()) {
                *x = Some(i);
            }
        }
        let layout = FrameLayout::of(m);
        let mut globals = Vec::with_capacity(m.globals().len());
        let mut volatile = HashSet::new();
        let mut image_arrays = HashMap::new();
        let mut buffer_arrays = HashMap::new();
        let mut wg = 0u32;
        for (i, g) in m.globals().iter().enumerate() {
            let gid = GlobalId::new(i as u32);
            let root = match g.class {
                StorageClass::Private | StorageClass::Input | StorageClass::Output => GRoot::Lane,
                StorageClass::Workgroup => {
                    wg += 1;
                    GRoot::Wg(wg - 1)
                }
                StorageClass::PushConstant => GRoot::Push,
                StorageClass::StorageBuffer | StorageClass::Uniform => match &g.kind {
                    // A descriptor array of buffers: its elements are the slot's consecutive
                    // buffers, element `e` the `e`th; `OpAccessChain` selects one and traps past
                    // them (and a sized array needs all of its elements bound).
                    GlobalKind::Resource(slot) => {
                        let first = res.buffers.iter().position(|b| b.slot == Some(*slot));
                        let bound = |k: usize| {
                            res.buffers
                                .iter()
                                .skip(k)
                                .take_while(|b| b.slot == Some(*slot))
                                .count() as u64
                        };
                        let missing = || {
                            GRoot::Missing(InterpError::Config(format!(
                                "no buffer bound for {slot:?}"
                            )))
                        };
                        match (t.get(g.pointee)?, first) {
                            (crate::spirv::ir::Type::Array { len, .. }, Some(k)) => {
                                let n = t.array_len(*len)?;
                                match bound(k) >= n {
                                    true => {
                                        buffer_arrays.insert(k as u32, n);
                                        GRoot::BufArray(k as u32)
                                    }
                                    false => GRoot::Missing(InterpError::Config(format!(
                                        "no {n} consecutive buffers bound for {slot:?}"
                                    ))),
                                }
                            }
                            (crate::spirv::ir::Type::RuntimeArray { .. }, Some(k)) => {
                                buffer_arrays.insert(k as u32, bound(k));
                                GRoot::BufArray(k as u32)
                            }
                            (_, Some(k)) => GRoot::Buf(k as u32),
                            (_, None) => missing(),
                        }
                    }
                    _ => GRoot::Missing(malformed("a buffer variable that is not a resource")),
                },
                StorageClass::UniformConstant => match (&g.kind, t.get(g.pointee)?) {
                    // A combined image sampler binds its image half: the image model has no
                    // sampler state, so only `OpImage` of it is modelled (a fetch through it).
                    (
                        GlobalKind::Resource(slot),
                        crate::spirv::ir::Type::Image(_)
                        | crate::spirv::ir::Type::SampledImage { .. },
                    ) => match images.iter().position(|i| i.slot == *slot) {
                        Some(k) => {
                            if m.volatile.contains(&gid) {
                                volatile.insert(k as u32);
                            }
                            GRoot::Image(k as u32)
                        }
                        None => GRoot::Missing(InterpError::Config(format!(
                            "no image bound for {slot:?}"
                        ))),
                    },
                    // A descriptor array of images (a storage slot's level views): its elements are
                    // consecutive images of the slot, `OpAccessChain` selects one.
                    (
                        GlobalKind::Resource(slot),
                        crate::spirv::ir::Type::Array { elem, len, .. },
                    ) if matches!(t.get(*elem)?, crate::spirv::ir::Type::Image(_)) => {
                        let n = t.array_len(*len)?;
                        let first = images.iter().position(|i| i.slot == *slot);
                        let complete = first.is_some_and(|k| {
                            (0..n).all(|e| {
                                images
                                    .get(k + e as usize)
                                    .is_some_and(|i| i.slot == *slot && u64::from(i.element) == e)
                            })
                        });
                        match first {
                            Some(k) if complete => {
                                if m.volatile.contains(&gid) {
                                    volatile.extend((0..n).map(|e| k as u32 + e as u32));
                                }
                                GRoot::Image(k as u32)
                            }
                            _ => GRoot::Missing(InterpError::Config(format!(
                                "no {n} consecutive images bound for {slot:?}"
                            ))),
                        }
                    }
                    // A runtime descriptor array of images (a heap array): its elements are the
                    // slot's consecutive images, element `e` bound to element `e`, as many as are
                    // bound; `OpAccessChain` selects one and traps past them.
                    (
                        GlobalKind::Resource(slot),
                        crate::spirv::ir::Type::RuntimeArray { elem, .. },
                    ) if matches!(t.get(*elem)?, crate::spirv::ir::Type::Image(_)) => {
                        match images.iter().position(|i| i.slot == *slot) {
                            Some(k) => {
                                let n = images
                                    .iter()
                                    .skip(k)
                                    .zip(0u32..)
                                    .take_while(|(i, e)| i.slot == *slot && i.element == *e)
                                    .count() as u64;
                                image_arrays.insert(k as u32, n);
                                if m.volatile.contains(&gid) {
                                    volatile.extend((0..n).map(|e| k as u32 + e as u32));
                                }
                                GRoot::Image(k as u32)
                            }
                            None => GRoot::Missing(InterpError::Config(format!(
                                "no image bound for {slot:?}"
                            ))),
                        }
                    }
                    _ => GRoot::Missing(Unmodelled::StorageClass(g.class).err()),
                },
                other => GRoot::Missing(Unmodelled::StorageClass(other).err()),
            };
            globals.push(root);
        }
        Ok(Prog {
            m,
            t,
            consts,
            order,
            loop_of,
            globals,
            volatile,
            image_arrays,
            buffer_arrays,
            layout,
            private_init: RefCell::new(vec![None; m.globals().len()]),
            inputs,
        })
    }

    /// The initial bytes of `Private` global `i`, of type `ty`, initialized by constant `c`:
    /// encoded once per run, then copied.
    fn private_init(&self, i: usize, ty: TypeId, c: ConstId) -> IResult<Vec<u8>> {
        if let Some(Some(bytes)) = self.private_init.borrow().get(i) {
            return Ok(bytes.clone());
        }
        let v = self
            .consts
            .get(c.index())
            .ok_or_else(|| malformed("init"))?;
        let bytes = layout::encode(&self.t, ty, v)?;
        if let Some(slot) = self.private_init.borrow_mut().get_mut(i) {
            *slot = Some(bytes.clone());
        }
        Ok(bytes)
    }

    /// Whether bound image `k`'s variable is decorated `Volatile`.
    pub(super) fn volatile(&self, k: u32) -> bool {
        self.volatile.contains(&k)
    }

    /// The bound length of the runtime descriptor array of images whose first image is `k`.
    pub(super) fn image_array_len(&self, k: u32) -> Option<u64> {
        self.image_arrays.get(&k).copied()
    }

    /// The bound length of the descriptor array of buffers whose first buffer is `k`.
    pub(super) fn buffer_array_len(&self, k: u32) -> Option<u64> {
        self.buffer_arrays.get(&k).copied()
    }

    fn block(&self, b: BlockId) -> IResult<&'m crate::spirv::ir::Block> {
        self.m
            .blocks()
            .get(b.index())
            .ok_or_else(|| malformed(format!("block {b:?}")))
    }

    fn order_of(&self, b: BlockId) -> u32 {
        self.order.get(b.index()).copied().unwrap_or(u32::MAX)
    }
}

impl FrameLayout {
    fn of(m: &TypedModule) -> FrameLayout {
        let mut func_of = vec![u32::MAX; m.blocks().len()];
        let mut var_at = vec![NO_SLOT; m.vars().len()];
        for (fi, f) in m.functions().iter().enumerate() {
            for b in &f.blocks {
                if let Some(x) = func_of.get_mut(b.index()) {
                    *x = fi as u32;
                }
            }
            for (k, var) in f.vars.iter().enumerate() {
                if let Some(x) = var_at.get_mut(var.index()) {
                    *x = (fi as u32, k as u32);
                }
            }
        }
        let mut slots = vec![0u32; m.functions().len()];
        let slot = m
            .values()
            .iter()
            .map(|d| match d.origin {
                ValueOrigin::Inst(b) | ValueOrigin::Phi(b) => {
                    let f = func_of.get(b.index()).copied().unwrap_or(u32::MAX);
                    match slots.get_mut(f as usize) {
                        Some(n) => {
                            *n += 1;
                            (f, *n - 1)
                        }
                        None => NO_SLOT,
                    }
                }
                _ => NO_SLOT,
            })
            .collect();
        FrameLayout {
            slot,
            slots,
            var_at,
        }
    }

    /// Value `v`'s slot, when an instruction or phi of function `func` defines it.
    fn slot_in(&self, func: FuncId, v: Value) -> Option<usize> {
        match self.slot.get(v.index()) {
            Some((f, k)) if *f as usize == func.index() => Some(*k as usize),
            _ => None,
        }
    }

    /// Variable `var`'s position among function `func`'s variables, when it is one of them.
    fn var_in(&self, func: FuncId, var: VarId) -> Option<usize> {
        match self.var_at.get(var.index()) {
            Some((f, k)) if *f as usize == func.index() => Some(*k as usize),
            _ => None,
        }
    }

    /// The number of slots of function `func`'s frame.
    fn slots_of(&self, func: FuncId) -> usize {
        self.slots.get(func.index()).map_or(0, |n| *n as usize)
    }
}

/// Number the blocks reachable from `entry` in structured order (see the module docs).
fn structured_order(m: &TypedModule, entry: Option<BlockId>, order: &mut [u32]) -> IResult<()> {
    let Some(entry) = entry else {
        return Ok(());
    };
    let succs = |b: BlockId| -> IResult<Vec<BlockId>> {
        let blk = m
            .blocks()
            .get(b.index())
            .ok_or_else(|| malformed(format!("block {b:?}")))?;
        let mut s = Vec::new();
        match blk.merge {
            Some(crate::spirv::ir::Merge::Loop { merge, cont }) => {
                s.push(merge);
                s.push(cont);
            }
            Some(crate::spirv::ir::Merge::Selection { merge }) => s.push(merge),
            None => {}
        }
        match &blk.term {
            Terminator::Branch(t) => s.push(*t),
            Terminator::CondBranch { t, f, .. } => {
                s.push(*t);
                s.push(*f);
            }
            Terminator::Switch { default, cases, .. } => {
                s.push(*default);
                s.extend(cases.iter().map(|(_, b)| *b));
            }
            _ => {}
        }
        Ok(s)
    };
    // Iterative DFS; `post` collects the postorder.
    let mut seen: HashSet<BlockId> = HashSet::new();
    let mut post = Vec::new();
    let mut stack: Vec<(BlockId, Vec<BlockId>)> = vec![(entry, succs(entry)?)];
    seen.insert(entry);
    while let Some((b, rest)) = stack.last_mut() {
        if rest.is_empty() {
            post.push(*b);
            stack.pop();
            continue;
        }
        let next = rest.remove(0);
        if seen.insert(next) {
            let s = succs(next)?;
            stack.push((next, s));
        }
    }
    let n = post.len() as u32;
    for (i, b) in post.iter().enumerate() {
        if let Some(slot) = order.get_mut(b.index()) {
            *slot = n - 1 - i as u32;
        }
    }
    Ok(())
}

/// One call frame.
#[derive(Debug)]
pub(super) struct Frame {
    /// The function this frame runs.
    func: FuncId,
    pub block: BlockId,
    pub inst: usize,
    /// The values its instructions and phis have defined, by slot (`FrameLayout`).
    vals: Vec<Option<Val>>,
    pub params: Vec<Val>,
    /// The lane arena length before this frame's variables, which follow it in the function's
    /// order.
    arena_mark: usize,
    /// The caller's result value for this call.
    ret: Option<Value>,
}

impl Frame {
    /// Define value `v` of this frame's function as `val`.
    fn define(&mut self, p: &Prog<'_>, v: Value, val: Val) -> IResult<()> {
        let slot = p
            .layout
            .slot_in(self.func, v)
            .and_then(|k| self.vals.get_mut(k))
            .ok_or_else(|| malformed(format!("{v:?} defined outside its function")))?;
        *slot = Some(val);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LaneState {
    Run,
    /// Waiting at a workgroup barrier.
    Barrier,
    Done,
}

/// One invocation.
#[derive(Debug)]
pub(super) struct Lane {
    pub global: [u32; 3],
    /// The lane's index in its subgroup.
    pub sg_lane: u32,
    pub frames: Vec<Frame>,
    /// `Function`, `Private`, and `Input` memory.
    pub arena: Vec<Region>,
    /// The arena allocation of each `Private`/`Input` global.
    globals: Vec<Option<u32>>,
    pub state: LaneState,
    pub stats: InvocationStats,
    /// The texel of the last store this invocation made to each out-of-bounds location of an
    /// image that models the device's robustness (`Image::robust`), keyed by image, coordinate,
    /// level, and sample. Robustness discards the store, but Vulkan lets a later load of that
    /// location in the same invocation return it, unless the load is volatile (the image's
    /// `Volatile` decoration or the read's `VolatileTexel`): the model takes that permission.
    pub oob_stores: BTreeMap<(u32, Vec<u64>, u64, u64), [u32; 4]>,
}

impl Lane {
    pub(super) fn top(&self) -> IResult<&Frame> {
        self.frames
            .last()
            .ok_or_else(|| malformed("lane without a frame"))
    }

    pub(super) fn top_mut(&mut self) -> IResult<&mut Frame> {
        self.frames
            .last_mut()
            .ok_or_else(|| malformed("lane without a frame"))
    }

    /// The value of operand `v` in the current frame.
    pub(super) fn get(&self, p: &Prog<'_>, v: Value) -> IResult<Val> {
        self.get_ref(p, v).map(Cow::into_owned)
    }

    /// The value of operand `v` in the current frame, borrowed where the frame or the program
    /// holds it, so reading a composite operand copies nothing.
    pub(super) fn get_ref<'a>(&'a self, p: &'a Prog<'_>, v: Value) -> IResult<Cow<'a, Val>> {
        let def =
            p.m.value(v)
                .map_err(|e| malformed(format!("value {v:?}: {e:?}")))?;
        let f = self.top()?;
        match def.origin {
            ValueOrigin::Const(c) => p
                .consts
                .get(c.index())
                .map(Cow::Borrowed)
                .ok_or_else(|| malformed(format!("constant {c:?}"))),
            ValueOrigin::Global(g) => self.global_ptr(p, g).map(|q| Cow::Owned(Val::P(q))),
            ValueOrigin::Var(var) => p
                .layout
                .var_in(f.func, var)
                .map(|k| {
                    Cow::Owned(Val::P(Ptr {
                        root: Root::Lane((f.arena_mark + k) as u32),
                        off: 0,
                        ms: None,
                    }))
                })
                .ok_or_else(|| malformed(format!("variable {var:?} of another function"))),
            ValueOrigin::Param { index, .. } => f
                .params
                .get(index as usize)
                .map(Cow::Borrowed)
                .ok_or_else(|| malformed("parameter out of range")),
            ValueOrigin::Inst(_) | ValueOrigin::Phi(_) => p
                .layout
                .slot_in(f.func, v)
                .and_then(|k| f.vals.get(k))
                .and_then(Option::as_ref)
                .map(Cow::Borrowed)
                .ok_or_else(|| malformed(format!("{v:?} used before its definition"))),
            ValueOrigin::Removed => Err(malformed(format!("{v:?} of a removed function"))),
        }
    }

    fn global_ptr(&self, p: &Prog<'_>, g: GlobalId) -> IResult<Ptr> {
        let root = match p.globals.get(g.index()) {
            Some(GRoot::Lane) => match self.globals.get(g.index()).copied().flatten() {
                Some(a) => Root::Lane(a),
                None => {
                    let gv =
                        p.m.globals()
                            .get(g.index())
                            .ok_or_else(|| malformed("global"))?;
                    return Err(match gv.kind {
                        GlobalKind::BuiltIn { builtin, .. } => Unmodelled::BuiltIn(builtin).err(),
                        _ => Unmodelled::StorageClass(gv.class).err(),
                    });
                }
            },
            Some(GRoot::Wg(i)) => Root::Workgroup(*i),
            Some(GRoot::Buf(i)) => Root::Buffer(*i),
            Some(GRoot::BufArray(i)) => Root::BufferArray(*i),
            Some(GRoot::Image(i)) => Root::Image(*i),
            Some(GRoot::Push) => Root::Push,
            Some(GRoot::Missing(e)) => return Err(e.clone()),
            None => return Err(malformed(format!("global {g:?}"))),
        };
        Ok(Ptr {
            root,
            off: 0,
            ms: None,
        })
    }

    /// Set the current frame's value `v`.
    pub(super) fn set(&mut self, p: &Prog<'_>, v: Value, val: Val) -> IResult<()> {
        self.top_mut()?.define(p, v, val)
    }

    /// This lane's position against `other`'s (see the module docs), compared frame by frame
    /// without collecting either: lexicographically, a stack that is a prefix of the other's
    /// before it. Every step compares each running lane with the subgroup's lead, so this is a
    /// plain loop rather than an iterator chain, which the test profile's opt-level 0 runs slowly,
    /// and frames at the same block and instruction (every frame of converged lanes) skip the
    /// structured-order lookup.
    fn cmp_position(&self, p: &Prog<'_>, other: &Lane) -> Ordering {
        for (a, b) in self.frames.iter().zip(&other.frames) {
            if a.block == b.block && a.inst == b.inst {
                continue;
            }
            let order = (p.order_of(a.block), a.inst).cmp(&(p.order_of(b.block), b.inst));
            if order != Ordering::Equal {
                return order;
            }
        }
        self.frames.len().cmp(&other.frames.len())
    }

    fn site(&self) -> Site {
        match self.frames.last() {
            Some(f) => Site {
                block: f.block,
                inst: Some(f.inst),
            },
            None => Site {
                block: BlockId::new(u32::MAX),
                inst: None,
            },
        }
    }

    /// Push a frame for `func` with `args`, allocating its variables.
    pub(super) fn call(
        &mut self,
        p: &Prog<'_>,
        func: FuncId,
        args: Vec<Val>,
        ret: Option<Value>,
    ) -> IResult<()> {
        let f =
            p.m.functions()
                .get(func.index())
                .ok_or_else(|| malformed(format!("function {func:?}")))?;
        let entry = *f
            .blocks
            .first()
            .ok_or_else(|| malformed("function without blocks"))?;
        let arena_mark = self.arena.len();
        for var in &f.vars {
            let lv =
                p.m.vars()
                    .get(var.index())
                    .ok_or_else(|| malformed(format!("variable {var:?}")))?;
            let bytes = match lv.init {
                Some(c) => {
                    let v = p
                        .consts
                        .get(c.index())
                        .ok_or_else(|| malformed("initializer"))?;
                    layout::encode(&p.t, lv.pointee, v)?
                }
                // Undefined `Function` memory is modelled as zero (only `Workgroup` is poisoned).
                None => zeroed(&p.t, lv.pointee)?,
            };
            self.arena
                .push(Region::new(MemoryKind::Function, bytes, true));
        }
        self.frames.push(Frame {
            func,
            block: entry,
            inst: 0,
            vals: vec![None; p.layout.slots_of(func)],
            params: args,
            arena_mark,
            ret,
        });
        Ok(())
    }

    /// Return from the current frame with `v`.
    fn ret(&mut self, p: &Prog<'_>, v: Option<Val>) -> IResult<()> {
        let f = self
            .frames
            .pop()
            .ok_or_else(|| malformed("return without a frame"))?;
        self.arena.truncate(f.arena_mark);
        match self.frames.last_mut() {
            None => self.state = LaneState::Done,
            Some(caller) => {
                if let (Some(dst), Some(v)) = (f.ret, v) {
                    caller.define(p, dst, v)?;
                }
                caller.inst += 1;
            }
        }
        Ok(())
    }

    /// Branch from the current block to `target`, evaluating its phis for this edge.
    fn enter(&mut self, p: &Prog<'_>, target: BlockId) -> IResult<()> {
        let from = self.top()?.block;
        let blk = p.block(target)?;
        let mut phis = Vec::with_capacity(blk.phis.len());
        for phi in &blk.phis {
            let (v, _) = phi
                .incoming
                .iter()
                .find(|(_, b)| *b == from)
                .ok_or_else(|| malformed(format!("phi in {target:?} has no edge from {from:?}")))?;
            phis.push((phi.result, self.get(p, *v)?));
        }
        let f = self.top_mut()?;
        f.block = target;
        f.inst = 0;
        for (v, val) in phis {
            f.define(p, v, val)?;
        }
        if let Some(i) = p.loop_of.get(target.index()).copied().flatten() {
            if let Some(c) = self.stats.header_visits.get_mut(i) {
                *c += 1;
            }
        }
        Ok(())
    }
}

fn zeroed(t: &Tys<'_>, ty: TypeId) -> IResult<Vec<u8>> {
    let n = layout::size(t, ty, None)?;
    let n = usize::try_from(n).map_err(|_| Unmodelled::Type(ty).err())?;
    Ok(vec![0; n])
}

/// The workgroup size: the `BuiltIn WorkgroupSize` composite when the module declares one (it
/// takes precedence over any local-size mode, as in Vulkan), else the entry's execution modes.
fn local_size(p: &Prog<'_>) -> IResult<[u32; 3]> {
    if let Some(c) = p.m.consts().workgroup_size_const() {
        let v = p.consts.get(c.index()).ok_or_else(|| malformed("size"))?;
        return match v.parts()? {
            [x, y, z] => Ok([x.s()? as u32, y.s()? as u32, z.s()? as u32]),
            _ => Err(malformed("workgroup size is not three components")),
        };
    }
    p.m.entry()
        .modes
        .iter()
        .find_map(|mode| match mode {
            ExecMode::LocalSize { x, y, z } => Some([*x, *y, *z]),
            _ => None,
        })
        .ok_or_else(|| malformed("compute entry without a local size"))
}

/// The dispatch-wide facts a lane's builtins derive from.
struct Grid {
    local: [u32; 3],
    groups: [u32; 3],
    sg_size: u32,
    /// A vertex run's first vertex and first instance ([`Config::first`]).
    first: [u32; 2],
}

impl Grid {
    fn subgroups(&self) -> u32 {
        let n = self.local[0] * self.local[1] * self.local[2];
        n.div_ceil(self.sg_size)
    }
}

/// A new invocation of workgroup `wg` with local linear index `li`.
fn new_lane(p: &Prog<'_>, grid: &Grid, wg: [u32; 3], li: u32) -> IResult<Lane> {
    let [lx, ly, _] = grid.local;
    let lid = [li % lx, (li / lx) % ly, li / (lx * ly)];
    let gid = [
        wg[0] * lx + lid[0],
        wg[1] * ly + lid[1],
        wg[2] * grid.local[2] + lid[2],
    ];
    let mut lane = Lane {
        global: gid,
        sg_lane: li % grid.sg_size,
        frames: Vec::new(),
        arena: Vec::new(),
        globals: vec![None; p.m.globals().len()],
        state: LaneState::Run,
        stats: InvocationStats {
            global_id: gid,
            header_visits: vec![0; p.m.loops().len()],
            barriers: 0,
        },
        oob_stores: BTreeMap::new(),
    };
    for (i, g) in p.m.globals().iter().enumerate() {
        let region = match (g.class, &g.kind) {
            (StorageClass::Private, _) => {
                let bytes = match g.init {
                    Some(c) => p.private_init(i, g.pointee, c)?,
                    None => zeroed(&p.t, g.pointee)?,
                };
                Region::new(MemoryKind::Private, bytes, true)
            }
            // A user interface input the run supplies: the same bytes in every invocation, as
            // fixed-function vertex input or the rasterizer would deliver them. One the run does
            // not supply stays unmodelled, an error only if the module reads it.
            (StorageClass::Input, GlobalKind::Io { key, .. }) => match p.inputs.get(key) {
                Some(bytes) => Region::new(MemoryKind::Input, bytes.clone(), false),
                None => continue,
            },
            (StorageClass::Input, GlobalKind::BuiltIn { builtin, .. }) => {
                use crate::spv::BuiltIn as B;
                let words: Vec<u32> = match builtin {
                    B::GlobalInvocationId => gid.to_vec(),
                    B::LocalInvocationId => lid.to_vec(),
                    B::WorkgroupId => wg.to_vec(),
                    B::NumWorkgroups => grid.groups.to_vec(),
                    B::LocalInvocationIndex => vec![li],
                    B::SubgroupSize => vec![grid.sg_size],
                    B::SubgroupLocalInvocationId => vec![li % grid.sg_size],
                    B::SubgroupId => vec![li / grid.sg_size],
                    B::NumSubgroups => vec![grid.subgroups()],
                    // A vertex run draws `workgroups[0]` vertices of `workgroups[1]` instances
                    // from its first vertex and instance (`Config::first`, the draw's bases) in
                    // `workgroups[2]` views (a multiview draw whose view mask is its first
                    // `workgroups[2]` views), one invocation per workgroup.
                    B::VertexIndex => vec![grid.first[0].wrapping_add(wg[0])],
                    B::InstanceIndex => vec![grid.first[1].wrapping_add(wg[1])],
                    B::ViewIndex => vec![wg[2]],
                    B::BaseVertex => vec![grid.first[0]],
                    B::BaseInstance => vec![grid.first[1]],
                    // Declared but unmodelled: an error only if the module reads it.
                    _ => continue,
                };
                let bytes = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                Region::new(MemoryKind::Input, bytes, false)
            }
            // A vertex stage's outputs: written, never reported.
            (StorageClass::Output, _) => {
                Region::new(MemoryKind::Output, zeroed(&p.t, g.pointee)?, true)
            }
            _ => continue,
        };
        if let Some(slot) = lane.globals.get_mut(i) {
            *slot = Some(lane.arena.len() as u32);
        }
        lane.arena.push(region);
    }
    let entry = p.m.entry().func;
    lane.call(p, entry, Vec::new(), None)?;
    Ok(lane)
}

/// The invocations of one workgroup and its memory.
struct Workgroup<'a, 'm> {
    p: &'a Prog<'m>,
    cfg: &'a Config,
    lanes: Vec<Lane>,
    /// The lane indices of each subgroup, a run of consecutive lanes.
    sgs: Vec<Range<usize>>,
    steps: &'a mut u64,
}

impl Workgroup<'_, '_> {
    fn ordered<T: Copy>(&self, v: &[T]) -> Vec<T> {
        match self.cfg.order {
            Order::Forward => v.to_vec(),
            Order::Reverse => v.iter().rev().copied().collect(),
        }
    }

    fn count(&mut self, n: usize) -> IResult<()> {
        *self.steps = self.steps.saturating_add(n as u64);
        if *self.steps > self.cfg.step_limit {
            return Err(TrapKind::StepLimit(self.cfg.step_limit).err());
        }
        Ok(())
    }

    /// Run until no lane is runnable, then release a uniform barrier; repeat until every lane
    /// has returned.
    fn run(&mut self, sh: &mut Shared) -> IResult<()> {
        let sg_order: Vec<usize> = self.ordered(&(0..self.sgs.len()).collect::<Vec<_>>());
        // The tangle of each step, its allocation kept across steps.
        let mut tangle = Vec::new();
        loop {
            for sg in &sg_order {
                while self.step(*sg, sh, &mut tangle)? {}
            }
            let waiting: Vec<usize> = (0..self.lanes.len())
                .filter(|i| {
                    self.lanes
                        .get(*i)
                        .is_some_and(|l| l.state != LaneState::Done)
                })
                .collect();
            let Some(first) = waiting.first() else {
                return Ok(());
            };
            let first_lane = self.lanes.get(*first).ok_or_else(|| malformed("lane"))?;
            if self.cfg.barriers == BarrierRule::Vulkan && waiting.len() != self.lanes.len() {
                // An invocation returned while the others wait at a barrier.
                return Err(TrapKind::NonUniformBarrier
                    .err()
                    .locate(first_lane.global, first_lane.site()));
            }
            for i in &waiting {
                let l = self.lanes.get(*i).ok_or_else(|| malformed("lane"))?;
                if l.state != LaneState::Barrier
                    || l.cmp_position(self.p, first_lane) != Ordering::Equal
                {
                    return Err(TrapKind::NonUniformBarrier.err().locate(l.global, l.site()));
                }
            }
            for i in &waiting {
                if let Some(l) = self.lanes.get_mut(*i) {
                    l.state = LaneState::Run;
                    l.stats.barriers += 1;
                    if let Some(f) = l.frames.last_mut() {
                        f.inst += 1;
                    }
                }
            }
        }
    }

    /// One SIMT step of subgroup `sg`, its tangle gathered in `tangle`. Returns whether a lane
    /// ran.
    fn step(&mut self, sg: usize, sh: &mut Shared, tangle: &mut Vec<usize>) -> IResult<bool> {
        let ids = self.sgs.get(sg).cloned().unwrap_or(0..0);
        let mut lead: Option<usize> = None;
        tangle.clear();
        for i in ids.clone() {
            let Some(l) = self.lanes.get(i) else {
                continue;
            };
            if l.state != LaneState::Run {
                continue;
            }
            let order = lead
                .and_then(|b| self.lanes.get(b))
                .map(|b| l.cmp_position(self.p, b));
            match order {
                None | Some(Ordering::Less) => {
                    lead = Some(i);
                    tangle.clear();
                    tangle.push(i);
                }
                Some(Ordering::Equal) => tangle.push(i),
                Some(Ordering::Greater) => {}
            }
        }
        let Some(lead) = lead else {
            return Ok(false);
        };
        if self.cfg.order == Order::Reverse {
            tangle.reverse();
        }
        let tangle = &tangle[..];
        self.count(tangle.len())?;
        let (block, inst) = {
            let f = self
                .lanes
                .get(lead)
                .ok_or_else(|| malformed("lane"))?
                .top()?;
            (f.block, f.inst)
        };
        let blk = self.p.block(block)?;
        match blk.insts.get(inst) {
            Some(node) => self.exec(ids, tangle, node, sh)?,
            None => {
                for i in tangle {
                    let l = self.lanes.get_mut(*i).ok_or_else(|| malformed("lane"))?;
                    let site = Site { block, inst: None };
                    terminate(self.p, l, &blk.term).map_err(|e| e.locate(l.global, site))?;
                }
            }
        }
        Ok(true)
    }

    /// Execute `node` for the tangle (lanes of the subgroup `ids`).
    fn exec(
        &mut self,
        ids: Range<usize>,
        tangle: &[usize],
        node: &InstNode,
        sh: &mut Shared,
    ) -> IResult<()> {
        match &node.inst {
            Inst::Group { op, .. } => {
                let res = node
                    .result
                    .ok_or_else(|| malformed("subgroup operation without a result"))?;
                let mut lanes = Vec::with_capacity(tangle.len());
                for i in tangle {
                    lanes.push(self.lanes.get(*i).ok_or_else(|| malformed("lane"))?);
                }
                let (lead_gid, lead_site) = lanes
                    .first()
                    .map(|l| (l.global, l.site()))
                    .ok_or_else(|| malformed("empty tangle"))?;
                let size = self.cfg.subgroup.lanes();
                let out = group::exec(self.p, op, size, &lanes)
                    .map_err(|e| e.locate(lead_gid, lead_site))?;
                for (i, v) in tangle.iter().zip(out) {
                    let l = self.lanes.get_mut(*i).ok_or_else(|| malformed("lane"))?;
                    l.set(self.p, res, v)?;
                    l.top_mut()?.inst += 1;
                }
                Ok(())
            }
            Inst::Derivative { op, a } => {
                let res = node
                    .result
                    .ok_or_else(|| malformed("a derivative without a result"))?;
                let mut lanes = Vec::with_capacity(tangle.len());
                for i in tangle {
                    lanes.push(self.lanes.get(*i).ok_or_else(|| malformed("lane"))?);
                }
                let (lead_gid, lead_site) = lanes
                    .first()
                    .map(|l| (l.global, l.site()))
                    .ok_or_else(|| malformed("empty tangle"))?;
                let out = deriv::exec(self.p, *op, *a, &lanes)
                    .map_err(|e| e.locate(lead_gid, lead_site))?;
                for (i, v) in tangle.iter().zip(out) {
                    let l = self.lanes.get_mut(*i).ok_or_else(|| malformed("lane"))?;
                    l.set(self.p, res, v)?;
                    l.top_mut()?.inst += 1;
                }
                Ok(())
            }
            Inst::ControlBarrier { exec, .. } => match exec.scope {
                Scope::Workgroup => {
                    for i in tangle {
                        if let Some(l) = self.lanes.get_mut(*i) {
                            l.state = LaneState::Barrier;
                        }
                    }
                    Ok(())
                }
                Scope::Subgroup => {
                    // SPIR-V leaves a barrier below `Workgroup` scope to the invocations of its
                    // tangle; under `MaximallyReconvergesKHR` the tangle is defined (the lanes
                    // this step runs), and the barrier orders those alone. Elsewhere which lanes
                    // share it is the device's: every lane of the subgroup that has not returned
                    // must be in the tangle.
                    let maximal = self
                        .p
                        .m
                        .entry()
                        .modes
                        .contains(&ExecMode::MaximallyReconverges);
                    let outside = if maximal {
                        None
                    } else {
                        ids.into_iter().find(|&i| {
                            self.lanes
                                .get(i)
                                .is_some_and(|l| l.state != LaneState::Done && !tangle.contains(&i))
                        })
                    };
                    if let Some(i) = outside {
                        let l = self.lanes.get(i).ok_or_else(|| malformed("lane"))?;
                        let (g, s) = match tangle.first().and_then(|t| self.lanes.get(*t)) {
                            Some(t) => (t.global, t.site()),
                            None => (l.global, l.site()),
                        };
                        return Err(TrapKind::NonUniformBarrier.err().locate(g, s));
                    }
                    for i in tangle {
                        if let Some(l) = self.lanes.get_mut(*i) {
                            l.top_mut()?.inst += 1;
                        }
                    }
                    Ok(())
                }
                other => Err(Unmodelled::Scope(other).err()),
            },
            _ => {
                for i in tangle {
                    let l = self.lanes.get_mut(*i).ok_or_else(|| malformed("lane"))?;
                    let site = l.site();
                    let flow =
                        eval::exec(self.p, l, sh, node).map_err(|e| e.locate(l.global, site))?;
                    match flow {
                        Flow::Next => l.top_mut()?.inst += 1,
                        Flow::Call { func, args } => {
                            l.call(self.p, func, args, node.result)
                                .map_err(|e| e.locate(l.global, site))?;
                        }
                    }
                }
                Ok(())
            }
        }
    }
}

/// Execute a block terminator for one lane.
fn terminate(p: &Prog<'_>, l: &mut Lane, term: &Terminator) -> IResult<()> {
    match term {
        Terminator::Branch(t) => l.enter(p, *t),
        Terminator::CondBranch { cond, t, f } => {
            let c = l.get(p, *cond)?.s()?;
            l.enter(p, if c != 0 { *t } else { *f })
        }
        Terminator::Switch {
            sel,
            default,
            cases,
        } => {
            let v = l.get(p, *sel)?.s()?;
            let target = cases
                .iter()
                .find(|(lit, _)| *lit == v)
                .map_or(*default, |(_, b)| *b);
            l.enter(p, target)
        }
        Terminator::Return => l.ret(p, None),
        Terminator::ReturnValue(v) => {
            let v = l.get(p, *v)?;
            l.ret(p, Some(v))
        }
        Terminator::Kill => Err(Unmodelled::Opcode(crate::spv::Op::Kill).err()),
        Terminator::TerminateInvocation => {
            Err(Unmodelled::Opcode(crate::spv::Op::TerminateInvocation).err())
        }
        Terminator::Unreachable => Err(TrapKind::Unreachable.err()),
        Terminator::EmitMeshTasks { .. } => {
            Err(Unmodelled::Opcode(crate::spv::Op::EmitMeshTasksEXT).err())
        }
    }
}

/// See [`super::run`].
pub(super) fn run(
    m: &TypedModule,
    cfg: &Config,
    res: Resources,
    images: Vec<super::Image>,
    inputs: BTreeMap<VaryingKey, Vec<u8>>,
) -> IResult<(Outcome, Vec<super::Image>)> {
    let model = m.entry().model;
    if !matches!(model, ExecModel::GLCompute | ExecModel::Vertex) {
        return Err(Unmodelled::ExecModel(model).err());
    }
    // Kept apart from `res`, whose buffers the run consumes: specialization-length arrays (the
    // threadgroup arena) are sized by these values for the whole run.
    let spec = res.spec.clone();
    let p = Prog::new(m, &res, &images, &spec, inputs)?;
    // A vertex invocation shares no workgroup: each is a workgroup of one (see `super::run`).
    let local = match model {
        ExecModel::Vertex => [1, 1, 1],
        _ => local_size(&p)?,
    };
    let per_group = local
        .iter()
        .try_fold(1u32, |a, b| a.checked_mul(*b))
        .filter(|n| *n > 0)
        .ok_or_else(|| InterpError::Config(format!("workgroup size {local:?}")))?;
    let groups = cfg.workgroups;
    let total = groups
        .iter()
        .try_fold(u64::from(per_group), |a, b| a.checked_mul(u64::from(*b)))
        .filter(|n| *n > 0)
        .ok_or_else(|| InterpError::Config(format!("dispatch {groups:?}")))?;
    if total > cfg.step_limit {
        return Err(InterpError::Config(format!(
            "{total} invocations exceed the step limit"
        )));
    }
    let grid = Grid {
        local,
        groups,
        sg_size: cfg.subgroup.lanes(),
        first: cfg.first,
    };
    let mut bufs = Vec::with_capacity(res.buffers.len());
    for b in res.buffers {
        let mut r = Region::new(MemoryKind::Buffer, b.bytes, true);
        r.address = b.address;
        bufs.push(r);
    }
    let mut sh = Shared {
        wg: Vec::new(),
        bufs,
        push: Region::new(
            MemoryKind::PushConstant,
            super::specialized_push(m, &res.spec, res.push),
            false,
        ),
        images,
    };
    let mut ids = Vec::new();
    for z in 0..groups[2] {
        for y in 0..groups[1] {
            for x in 0..groups[0] {
                ids.push([x, y, z]);
            }
        }
    }
    if cfg.order == Order::Reverse {
        ids.reverse();
    }
    let mut steps = 0u64;
    let mut stats: Vec<(u64, InvocationStats)> = Vec::with_capacity(total as usize);
    for wg in ids {
        sh.wg.clear();
        for g in m.globals() {
            if g.class != StorageClass::Workgroup {
                continue;
            }
            let size = layout::size(&p.t, g.pointee, None)?;
            let size = usize::try_from(size).map_err(|_| Unmodelled::Type(g.pointee).err())?;
            let region = match g.init {
                Some(c) => {
                    let v = p.consts.get(c.index()).ok_or_else(|| malformed("init"))?;
                    Region::new(
                        MemoryKind::Workgroup,
                        layout::encode(&p.t, g.pointee, v)?,
                        true,
                    )
                }
                None => Region::poisoned(MemoryKind::Workgroup, size),
            };
            sh.wg.push(region);
        }
        let mut lanes = Vec::with_capacity(per_group as usize);
        for li in 0..per_group {
            lanes.push(new_lane(&p, &grid, wg, li)?);
        }
        let (n, size) = (per_group as usize, grid.sg_size as usize);
        let sgs: Vec<Range<usize>> = (0..n).step_by(size).map(|s| s..(s + size).min(n)).collect();
        let mut w = Workgroup {
            p: &p,
            cfg,
            lanes,
            sgs,
            steps: &mut steps,
        };
        w.run(&mut sh)?;
        let base = (u64::from(wg[2]) * u64::from(groups[1]) + u64::from(wg[1]))
            * u64::from(groups[0])
            + u64::from(wg[0]);
        for (li, l) in w.lanes.into_iter().enumerate() {
            stats.push((base * u64::from(per_group) + li as u64, l.stats));
        }
    }
    stats.sort_by_key(|(k, _)| *k);
    let outcome = Outcome {
        buffers: sh.bufs.into_iter().map(|r| r.bytes).collect(),
        steps,
        invocations: stats.into_iter().map(|(_, s)| s).collect(),
    };
    Ok((outcome, sh.images))
}
