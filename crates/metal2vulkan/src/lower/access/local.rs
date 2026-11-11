//! Arena and typed-object pointers and accesses.
//!
//! A threadgroup pointer (`Repr::TgPtr`), a pointer into the private arena
//! (`Repr::PrivPtr(PrivForm::Arena)`), and a pointer into a typed object (a typed alloca,
//! `PrivForm::Typed`, a closed constant global, `ConstForm::Private`, or a typed threadgroup
//! object, `Repr::TgTyped`, a `Workgroup` variable; `memory::alloca`, `memory::plan::SharedPlan`)
//! are `u32` byte offsets from their [`Home`]'s start, demoted as `DemoteTy::ArenaOffset`:
//!
//! - an alloca placed in the private arena is its plan offset; a typed alloca's, a constant
//!   global's and a typed threadgroup object's address are 0; a static threadgroup global's
//!   address is its plan offset (a
//!   constant `getelementptr` of one adds the plan's exact displacement); a `[[threadgroup(n)]]`
//!   parameter is its region's offset, an `OpSpecConstantOp` of the specialized lengths before it;
//! - a GEP adds its datalayout offsets in wrapping `u32` arithmetic, with the buffer family's
//!   index rule (`access::gep_offset`); `bitcast`, `addrspacecast`, and `freeze` copy the offset,
//!   and `select` selects it (a constant side is its global's offset);
//! - a constant `getelementptr` of a threadgroup global or of a closed constant global is the
//!   global's offset plus the plan's exact displacement;
//! - `icmp` of two pointers into one arena or one typed object compares their offsets, which are
//!   exact within an object and order an arena's objects by the plan's layout (an order MSL
//!   leaves unspecified); a pointer whose roots are objects only (no null, no `Dynamic`) is
//!   never null, so its unsigned or equality compare with the null constant is a constant;
//! - a non-atomic load or store of a scalar or 2–4-lane vector is one word-composed leaf
//!   (`memory::leaf`) in the arena's word memory (`memory::arena::ArenaMem`), or, in a typed
//!   object, an `OpLoad`/`OpStore` of the held form through an access chain whose dynamic indices
//!   come from the offset (`memory::alloca::TypedPath`), a typed threadgroup object's with the
//!   memory model's operands the arena's words take; a volatile one's words carry `Volatile`,
//!   except a load of a read-only typed constant object.
//!
//! **Containment** of an arena access is on unless the profile's containment is `UnsafeOff` (no
//! robustness feature covers arenas): with `inb` the normative check against the arena's exact byte
//! size, a load reads at `select(inb, off, 0)` and yields `select(inb, v, 0)`, and a store sits in
//! a builder-closed `if_(inb)`. A typed access's dynamic indices are **contained** under
//! `GuestIsolation::On` (`memory::alloca`), so the access never leaves its object: a thread's own
//! object's (an alloca's) are checked against their levels' lengths, a load past a level reading
//! zero and a store past one sitting in a builder-closed `if_` that drops it; a threadgroup or
//! constant object's are each clamped to the level's last element. A constant index is in range by
//! construction. Each dynamic index is the defining `getelementptr`'s own operand where the GEP's
//! dynamic indices line up one to one with the path's dynamic levels (`lined_up`), and is
//! recovered from the pointer's offset otherwise.
//!
//! The offset shape of a typed access is read from its pointer's definitions: an alloca or a global
//! address is `Exact(0)`, a GEP adds its constant and a dynamic multiple of each of its strides
//! (one stride per nested array level, `alloca::MAX_DYNAMIC` at most), and casts, `freeze`, and
//! joins of equal shapes keep it. A pointer cycle (a loop-carried pointer phi) is not implemented.
//! Each definition's shape is read once per function, and every definition the walk reaches is a
//! unit of work.
//!
//! Selecting a threadgroup access or parameter fixes the arena's shape once per function: the
//! static part is checked against `maxComputeSharedMemorySize` there (`TgShape::check`), and the
//! function depends on `ThreadgroupMemoryLength` of every region (the arena's layout does).
//! A fragment function's threadgroup memory is its tile's, which persists across the render pass
//! (`access::tile_memory`): its arena is that memory in the pass's buffer, no workgroup array, and
//! a function's arguments make a tile function's and a fragment function's arena alike depend on
//! `TileThreadgroupMemory` of each. No other stage has threadgroup memory.
//!
//! Atomics on the threadgroup arena and memory intrinsics on either arena are `access::atomic`'s
//! and `access::copy`'s, which reach the arenas through this module.
//!
//! A pointer stored into or loaded from a forwarding private cell is `access::cell`'s.
//!
//! Not lowered here (`Unsupported{Lowering(Access), NotImplemented}`): pointers into these homes
//! stored anywhere else or converted to integers, a compare of pointers into different homes or
//! of a pointer that may be null, a signed compare with null, a phi with a constant pointer
//! operand, a `select` on a constant condition, and aggregate accesses.
//!
//! **An integer threadgroup address** (`inttoptr` of an integer literal) names a byte of Metal's
//! threadgroup layout, which the arena is (`memory::plan` places the static objects as Metal does).
//! A load, store, or atomic whose address operand is one, and whose bytes lie inside the static
//! objects, accesses those arena bytes (`access_ptr`). Any other use of one (outside the static
//! objects, displaced by a `getelementptr`, as a value, in a module that declares function
//! constants, or where Metal's order of the objects is not known) is
//! `Unsupported{DereferenceLoadedPointer, NotRepresentable}`: Metal's answer there is not measured.

use std::collections::BTreeMap;

use crate::air::abi::{ParamBinding, ParamRole};
use crate::air::ir::{
    AddrSpace, CastOp, Const, ConstExpr, ConstId, GlobalId as AirGlobalId, Inst, InstId, InstKind,
    IntPredicate, Type, TypeId, Value, ValueId,
};
use crate::budget::{tree_bytes, Meter, PRIVATE_ARENA_BYTES};
use crate::device::profile::{Containment, GuestIsolation};
use crate::error::{Error, Feature, Limit, LowerFamily, Unsupported};
use crate::lower::emit::EmitCx;
use crate::lower::select::{Plan, SelectCx};
use crate::memory::alloca::{self, Init, Shape, Step, Strides, TypedPath, TypedTy};
use crate::memory::arena::{self, Arena, ArenaDecl, ArenaMem, TgDecl, TgShape};
use crate::memory::contain;
use crate::memory::fragment;
use crate::memory::leaf::{self, k32, Leaf, WordOps};
use crate::memory::plan::ArenaObject;
use crate::memory::provenance::{map_entry_bytes, ConstForm, OffsetRange, PrivForm, Repr, Root};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::Stage;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, GlobalId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, GlslOp, MemAccess, StorageClass, Type as SType,
};

use super::emit::index_u32;
use super::tile::{KeptCells, KeptMem};
use super::{is_ptr_ty, not_lowered, operand, refuse, GepTerms, IndexForm, Indexed};

/// Where a pointer points: an arena, or a typed object (by its index in the memory plan's
/// `typed_allocas`, `const_globals` or `typed_shared`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Home {
    Arena(Arena),
    Alloca(u32),
    Constant(u32),
    /// A typed threadgroup object: a `Workgroup` variable (`memory::plan::SharedPlan`).
    Shared(u32),
}

/// A pointer operand: a value, or a constant offset (a global's address).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LocalPtr {
    Value(ValueId),
    Offset(u32),
}

/// A leaf load or store through an arena pointer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalAccess {
    pub arena: Arena,
    pub ptr: LocalPtr,
    /// The stored value (stores only).
    pub value: Option<ValueId>,
    pub leaf: Leaf,
    /// The in-shader containment check (off only for `Containment::UnsafeOff`).
    pub contained: bool,
    /// A volatile access: every word it loads or stores carries `Volatile`.
    pub volatile: bool,
}

impl LocalAccess {
    pub(super) fn words(&self) -> u32 {
        self.leaf
            .cost()
            .saturating_add(if self.contained { 64 } else { 0 })
            .saturating_add(16)
    }
}

/// A load or store of a typed object's subobject (`memory::alloca`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypedAccess {
    pub home: Home,
    pub ptr: LocalPtr,
    /// The stored value (stores only).
    pub value: Option<ValueId>,
    pub path: TypedPath,
    /// Each dynamic level's index as the defining `getelementptr`'s operand, in level order,
    /// where its dynamic indices line up one to one with the path's dynamic levels
    /// (`lined_up`); `None` recovers each from the pointer's offset.
    pub indices: Option<Vec<(ValueId, IndexForm)>>,
    /// Each dynamic index is contained (`GuestIsolation::On`): checked against its level's length
    /// where the access is [`checked`](Self::checked), clamped to its level's last element
    /// otherwise.
    pub clamped: bool,
    /// A volatile access: its load or store carries `Volatile`.
    pub volatile: bool,
    /// A simdgroup matrix held as its fragment (`memory::fragment`): the held form of the whole
    /// `<64 x T>` value (its chunked vector), which a load rebuilds from the fragment's two
    /// elements and a store keeps those of.
    pub fragment: Option<DemoteTy>,
    /// An access of a held matrix's elements 0 and 1, or of one of them: the path ends at the
    /// fragment, which the access loads or stores as is, or one lane of.
    pub part: Option<alloca::FragmentPart>,
}

impl TypedAccess {
    /// A contained access of a thread's own typed object (an alloca) is checked: a load past a
    /// level reads zero and a store past one is dropped, so a store never changes an element it
    /// does not address, as Metal's store past the array changes none of the array's. A threadgroup
    /// or constant object's contained access clamps each index to its level's last element
    /// (`memory::alloca`).
    pub fn checked(&self) -> bool {
        self.clamped && matches!(self.home, Home::Alloca(_))
    }

    fn words(&self) -> u32 {
        let chain = 8u32.saturating_add(self.path.steps.len() as u32);
        let levels = self.path.dynamic().count() as u32;
        let dynamic = 24 * levels;
        // A checked access's value select or store guard, beside its per-level checks.
        let clamp = match (self.clamped, self.checked()) {
            (false, _) => 0,
            (true, false) => 32 * levels,
            (true, true) => (32 * levels).saturating_add(32),
        };
        // A fragment's shuffle, and the rebuilt value's chunks.
        let fragment = match &self.fragment {
            Some(DemoteTy::Composite(chunks)) => 12u32.saturating_add(chunks.len() as u32),
            _ => 0,
        };
        // A held element's lane index.
        let lane = u32::from(matches!(self.part, Some(alloca::FragmentPart::Lane(_)))) * 4;
        chain + dynamic + clamp + fragment + lane + 16
    }
}

/// The arena and typed-object strategies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocalStrategy {
    /// An alloca: its offset in the private arena, or 0 in its own typed object.
    Alloca {
        home: Home,
        offset: u32,
    },
    /// `getelementptr` on an offset: `base + konst + Σ index · stride` (wrapping `u32`).
    Gep {
        home: Home,
        base: LocalPtr,
        konst: u32,
        terms: GepTerms,
    },
    /// `bitcast`, `addrspacecast`, or `freeze` of a pointer.
    Copy {
        home: Home,
        of: LocalPtr,
    },
    /// `select c, t, f` of two pointers into one home (a side may be a constant: a threadgroup
    /// global's address).
    Select {
        home: Home,
        c: ValueId,
        t: LocalPtr,
        f: LocalPtr,
    },
    /// `icmp` of two pointers into one arena or typed object: their offsets compared.
    Compare {
        home: Home,
        pred: IntPredicate,
        a: LocalPtr,
        b: LocalPtr,
    },
    /// `icmp` of a pointer that is never null (its roots are objects) with the null constant:
    /// the constant result.
    NullCompare {
        home: Home,
        result: bool,
    },
    Load(LocalAccess),
    Store(LocalAccess),
    /// A load from a typed object.
    TypedLoad(TypedAccess),
    /// A store into a typed alloca.
    TypedStore(TypedAccess),
}

/// The census name of an operation by home kind: threadgroup arena or typed threadgroup object,
/// private arena, typed alloca, constant global, payload.
fn by_home(home: Home, names: [&'static str; 5]) -> &'static str {
    let [tg, private, typed, constant, payload] = names;
    match home {
        Home::Arena(Arena::Threadgroup) | Home::Shared(_) => tg,
        Home::Arena(Arena::Private) => private,
        Home::Alloca(_) => typed,
        Home::Constant(_) => constant,
        Home::Arena(Arena::Payload) => payload,
    }
}

impl LocalStrategy {
    /// The volatile form of a load or store (`access::volatile_form`): its words carry `Volatile`,
    /// except a typed constant object's, which is read-only; the refusal of any other strategy.
    pub(super) fn volatile(self) -> Result<LocalStrategy, Error> {
        Ok(match self {
            // A payload word under the Vulkan model is an atomic, which carries no `Volatile`.
            LocalStrategy::Load(a) | LocalStrategy::Store(a) if a.arena == Arena::Payload => {
                return Err(refuse())
            }
            LocalStrategy::Load(a) => LocalStrategy::Load(LocalAccess {
                volatile: true,
                ..a
            }),
            LocalStrategy::Store(a) => LocalStrategy::Store(LocalAccess {
                volatile: true,
                ..a
            }),
            LocalStrategy::TypedLoad(a) if matches!(a.home, Home::Constant(_)) => {
                LocalStrategy::TypedLoad(a)
            }
            LocalStrategy::TypedLoad(a) => LocalStrategy::TypedLoad(TypedAccess {
                volatile: true,
                ..a
            }),
            LocalStrategy::TypedStore(a) => LocalStrategy::TypedStore(TypedAccess {
                volatile: true,
                ..a
            }),
            _ => return Err(refuse()),
        })
    }

    /// The census name (`access.<home>.<op>`).
    pub fn name(&self) -> &'static str {
        use LocalStrategy as L;
        match self {
            L::Alloca { home, .. } => by_home(
                *home,
                [
                    "access.threadgroup.alloca",
                    "access.private.alloca",
                    "access.typed.alloca",
                    "access.constant.alloca",
                    "access.payload.alloca",
                ],
            ),
            L::Gep { home, .. } => by_home(
                *home,
                [
                    "access.threadgroup.gep",
                    "access.private.gep",
                    "access.typed.gep",
                    "access.constant.gep",
                    "access.payload.gep",
                ],
            ),
            L::Copy { home, .. } => by_home(
                *home,
                [
                    "access.threadgroup.pointer_copy",
                    "access.private.pointer_copy",
                    "access.typed.pointer_copy",
                    "access.constant.pointer_copy",
                    "access.payload.pointer_copy",
                ],
            ),
            L::Select { home, .. } => by_home(
                *home,
                [
                    "access.threadgroup.pointer_select",
                    "access.private.pointer_select",
                    "access.typed.pointer_select",
                    "access.constant.pointer_select",
                    "access.payload.pointer_select",
                ],
            ),
            L::Load(a) => match (a.arena, a.contained) {
                (Arena::Threadgroup, true) => "access.threadgroup.load.checked",
                (Arena::Threadgroup, false) => "access.threadgroup.load",
                (Arena::Private, true) => "access.private.load.checked",
                (Arena::Private, false) => "access.private.load",
                (Arena::Payload, true) => "access.payload.load.checked",
                (Arena::Payload, false) => "access.payload.load",
            },
            L::Store(a) => match (a.arena, a.contained) {
                (Arena::Threadgroup, true) => "access.threadgroup.store.checked",
                (Arena::Threadgroup, false) => "access.threadgroup.store",
                (Arena::Private, true) => "access.private.store.checked",
                (Arena::Private, false) => "access.private.store",
                (Arena::Payload, true) => "access.payload.store.checked",
                (Arena::Payload, false) => "access.payload.store",
            },
            L::Compare { home, .. } => by_home(
                *home,
                [
                    "access.threadgroup.pointer_compare",
                    "access.private.pointer_compare",
                    "access.typed.pointer_compare",
                    "access.constant.pointer_compare",
                    "access.payload.pointer_compare",
                ],
            ),
            L::NullCompare { home, .. } => by_home(
                *home,
                [
                    "access.threadgroup.null_compare",
                    "access.private.null_compare",
                    "access.typed.null_compare",
                    "access.constant.null_compare",
                    "access.payload.null_compare",
                ],
            ),
            L::TypedLoad(a) => match (a.home, a.clamped) {
                (Home::Constant(_), true) => "access.constant.load.clamped",
                (Home::Constant(_), false) => "access.constant.load",
                (Home::Shared(_), true) => "access.threadgroup.typed_load.clamped",
                (Home::Shared(_), false) => "access.threadgroup.typed_load",
                (_, true) => "access.typed.load.clamped",
                (_, false) => "access.typed.load",
            },
            L::TypedStore(a) => match (a.home, a.clamped) {
                (Home::Shared(_), true) => "access.threadgroup.typed_store.clamped",
                (Home::Shared(_), false) => "access.threadgroup.typed_store",
                (_, true) => "access.typed.store.clamped",
                (_, false) => "access.typed.store",
            },
        }
    }

    /// An upper estimate of the SPIR-V words one use emits.
    pub fn words(&self) -> u32 {
        match self {
            LocalStrategy::Alloca { .. } | LocalStrategy::Copy { .. } => 4,
            LocalStrategy::Gep { terms, .. } => {
                12u32.saturating_add(16u32.saturating_mul(terms.len() as u32))
            }
            LocalStrategy::Select { .. } | LocalStrategy::Compare { .. } => 6,
            LocalStrategy::NullCompare { .. } => 2,
            LocalStrategy::Load(a) | LocalStrategy::Store(a) => a.words(),
            LocalStrategy::TypedLoad(a) | LocalStrategy::TypedStore(a) => a.words(),
        }
    }
}

/// A typed object a function uses: its home, its type (an array of `count` elements of `ty`),
/// and a constant global's initializer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypedObject {
    pub home: Home,
    pub ty: TypedTy,
    pub count: u32,
    pub init: Option<Init>,
}

/// The arenas and typed objects a function uses, fixed while selecting.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LocalPlan {
    /// The threadgroup arena's shape, when a threadgroup pointer is used.
    pub tg: Option<TgShape>,
    /// The private arena's byte size, when a pointer into it is used.
    pub private: Option<u64>,
    /// The payload's byte size (its parameter's `air.arg_type_size`), when a pointer into it is
    /// used.
    pub payload: Option<u64>,
    /// The typed objects the function uses, in order of first use.
    pub typed: Vec<TypedObject>,
    /// Each typed object's position in `typed`, so a use finds it in one search, not a scan.
    pub positions: BTreeMap<Home, u32>,
}

impl LocalPlan {
    /// The position of typed object `home` in `typed` (and in `LocalDecl::typed`), if the
    /// function uses it.
    pub fn position(&self, home: Home) -> Option<usize> {
        self.positions.get(&home).map(|&i| i as usize)
    }
}

/// The declared arenas and typed objects.
#[derive(Clone, Debug, Default)]
pub struct LocalDecl {
    pub tg: Option<TgDecl>,
    pub private: Option<ArenaDecl>,
    /// The payload arena.
    pub payload: Option<ArenaDecl>,
    /// Each typed object's variable: `Private`, or a typed threadgroup object's `Workgroup`.
    pub typed: Vec<(Home, GlobalId)>,
}

/// The home of pointer value `v`, if it points into an arena or a typed object.
pub(super) fn home_of(cx: &SelectCx<'_>, v: Value) -> Option<Home> {
    match cx.memory.repr(v)? {
        Repr::TgPtr => Some(Home::Arena(Arena::Threadgroup)),
        Repr::TgTyped(i) => Some(Home::Shared(i)),
        Repr::PayloadPtr => Some(Home::Arena(Arena::Payload)),
        Repr::PrivPtr(PrivForm::Arena) => Some(Home::Arena(Arena::Private)),
        Repr::PrivPtr(PrivForm::Typed(i)) => Some(Home::Alloca(i)),
        Repr::ConstPtr(ConstForm::Private(i)) => Some(Home::Constant(i)),
        _ => None,
    }
}

/// The AIR global of constant home `i`.
fn const_global<'a>(cx: &'a SelectCx<'_>, i: u32) -> Result<&'a crate::air::ir::Global, Error> {
    cx.memory
        .const_globals
        .get(i as usize)
        .and_then(|c| cx.entry.module.global(c.global))
        .ok_or_else(|| cx.invariant("a constant global out of range"))
}

/// The element type and element count of typed home `home`.
fn object(cx: &SelectCx<'_>, home: Home) -> Result<(TypeId, u64), Error> {
    match home {
        Home::Alloca(i) => {
            let a = cx
                .memory
                .typed_allocas
                .get(i as usize)
                .ok_or_else(|| cx.invariant("a typed alloca out of range"))?;
            Ok((a.ty, a.count))
        }
        Home::Constant(i) => Ok((const_global(cx, i)?.value_ty, 1)),
        Home::Shared(i) => Ok((shared_global(cx, i)?.value_ty, 1)),
        Home::Arena(_) => Err(cx.invariant("an arena is not a typed object")),
    }
}

/// The AIR global of typed threadgroup object `i`.
fn shared_global<'a>(cx: &'a SelectCx<'_>, i: u32) -> Result<&'a crate::air::ir::Global, Error> {
    cx.memory
        .typed_shared
        .get(i as usize)
        .and_then(|s| cx.entry.module.global(s.global))
        .ok_or_else(|| cx.invariant("a typed threadgroup object out of range"))
}

/// The bytes of workgroup memory the function's static threadgroup objects take: the arena's
/// static part where the function declares the arena, and each typed threadgroup object's
/// (`memory::plan::SharedPlan::bytes`). `None` when it takes none.
pub fn static_workgroup_bytes(cx: &SelectCx<'_>) -> Option<u32> {
    let arena = cx
        .families
        .access
        .local
        .tg
        .as_ref()
        .map(|shape| shape.static_bytes().unwrap_or(u64::MAX));
    let typed = cx.memory.typed_shared_bytes();
    let total = match (arena, typed) {
        (None, 0) => return None,
        (arena, typed) => arena.unwrap_or(0).saturating_add(typed),
    };
    Some(u32::try_from(total).unwrap_or(u32::MAX))
}

/// Whether typed home `home` holds its simdgroup matrices as fragments.
fn fragments(cx: &SelectCx<'_>, home: Home) -> bool {
    match home {
        Home::Alloca(i) => cx
            .memory
            .typed_allocas
            .get(i as usize)
            .is_some_and(|a| a.fragments),
        Home::Constant(_) | Home::Shared(_) | Home::Arena(_) => false,
    }
}

/// Record that the function uses `home`: an arena ([`use_arena`]), or a typed object whose type
/// (and initializer) is rebuilt once, in work and ledger bytes linear in its distinct types and
/// constants. The rebuilt type and initializer stay on the ledger: the plan holds them.
fn use_home(cx: &mut SelectCx<'_>, home: Home, m: &mut Meter) -> Result<(), Error> {
    if let Home::Arena(a) = home {
        return use_arena(cx, a);
    }
    if cx.families.access.local.position(home).is_some() {
        return Ok(());
    }
    let (ty, count) = object(cx, home)?;
    let tt = alloca::typed_ty(cx.entry, cx.profile, ty, fragments(cx, home), m)??;
    // A refusal past the shape releases it.
    let (count, init) = match count_and_init(cx, home, count, &tt, m) {
        Ok(rest) => rest,
        Err(err) => {
            tt.release(m);
            return Err(err);
        }
    };
    let local = &mut cx.families.access.local;
    let at = u32::try_from(local.typed.len()).map_err(|_| refuse());
    // The object and its position are charged while the plan holds them.
    let charged = at.and_then(|at| {
        m.grow(&mut local.typed, 1)?;
        m.charge(map_entry_bytes::<Home, u32>())?;
        Ok(at)
    });
    let at = match charged {
        Ok(at) => at,
        Err(err) => {
            tt.release(m);
            return Err(err);
        }
    };
    local.typed.push(TypedObject {
        home,
        ty: tt,
        count,
        init,
    });
    local.positions.insert(home, at);
    Ok(())
}

/// The element count of typed home `home`, and a constant's initializer of shape `tt`.
fn count_and_init(
    cx: &SelectCx<'_>,
    home: Home,
    count: u64,
    tt: &TypedTy,
    m: &mut Meter,
) -> Result<(u32, Option<Init>), Error> {
    let count = u32::try_from(count)
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(refuse)?;
    let init = match home {
        Home::Constant(i) => {
            let c = const_global(cx, i)?.init.ok_or_else(refuse)?;
            Some(alloca::init_of(cx.entry, c, tt, m)?)
        }
        _ => None,
    };
    Ok((count, init))
}

/// Record that the function uses `arena`. The threadgroup arena's shape is fixed and checked
/// against `maxComputeSharedMemorySize` once, and every region's length becomes a runtime fact of
/// the function. A private arena past `budget::PRIVATE_ARENA_BYTES` is
/// `LimitExceeded(PrivateArenaBytes)`.
pub(super) fn use_arena(cx: &mut SelectCx<'_>, arena: Arena) -> Result<(), Error> {
    match arena {
        Arena::Threadgroup => {
            if cx.families.access.local.tg.is_some() {
                return Ok(());
            }
            // A fragment function's threadgroup memory is its tile's: one allocation per tile,
            // shared by every fragment of the tile and kept across the render pass's draws and
            // tile dispatches (MSL §4.4), as a tile function's is (`tile_memory`). An object or
            // mesh function's is its workgroup's, as a kernel's.
            let fragment = match cx.stage {
                Stage::Kernel | Stage::Object | Stage::Mesh => false,
                Stage::Fragment => true,
                Stage::Vertex => return Err(refuse()),
            };
            // A tile function declares no threadgroup variables, as Metal refuses its pipeline
            // whatever their size; Metal's compiler refuses a fragment function's.
            if (fragment || super::tile::is_tile(cx)) && cx.memory.has_static_threadgroup_objects()
            {
                return Err(super::tile::threadgroup_variable());
            }
            let mut shape = TgShape::of(&cx.memory.tg_arena);
            if fragment {
                // Its arguments are its tile's memory in the pass's buffer, no workgroup array.
                if shape.regions.is_empty() {
                    return Err(refuse());
                }
                shape.fragment = Some(super::tile_memory::plan(cx));
            } else {
                shape.check(crate::lower::mesh::workgroup_limit(cx))?;
                // An object function's arguments, whose lengths the render encoder sets: the
                // binder checks them beside its arena's static part and its payload.
                if cx.stage == Stage::Object && !shape.regions.is_empty() {
                    let word = |bytes: u64| u32::try_from(bytes).unwrap_or(u32::MAX);
                    cx.facts.insert(RuntimeFact::ObjectThreadgroupArena {
                        static_bytes: word(shape.static_bytes().unwrap_or(u64::MAX)),
                        payload: crate::lower::mesh::payload_extent(cx).map(word),
                    });
                }
            }
            // A tile function's arguments are its tile's threadgroup memory, which the render
            // pass keeps across its tile dispatches (`tile::wrap_entry`), and so are a fragment
            // function's.
            shape.tile = (fragment || super::tile::is_tile(cx)) && !shape.regions.is_empty();
            for &(_, index) in &shape.regions {
                if shape.tile {
                    cx.facts
                        .insert(RuntimeFact::TileThreadgroupMemory { index });
                } else {
                    crate::lower::dispatch::note_threadgroup_length(cx, index);
                }
            }
            cx.families.access.local.tg = Some(shape);
        }
        Arena::Private => {
            if cx.families.access.local.private.is_none() {
                // A pointer into the arena is a `u32` byte offset, and containment compares it
                // with the arena's size as a `u32` constant: a larger arena would alias, or be
                // checked short.
                let size = cx.memory.priv_arena.static_size;
                if size > PRIVATE_ARENA_BYTES {
                    return Err(Error::LimitExceeded(Limit::PrivateArenaBytes));
                }
                cx.families.access.local.private = Some(size);
            }
        }
        // An object or mesh function's payload, its parameter's size (a mesh function's at least
        // its pipeline's object function's), within the device's payload limits
        // (`lower::mesh::check_payload`).
        Arena::Payload => {
            if cx.families.access.local.payload.is_none() {
                let Some(size) = crate::lower::mesh::payload_extent(cx) else {
                    return Err(cx.invariant("a payload pointer without a payload parameter"));
                };
                crate::lower::mesh::check_payload(cx, size)?;
                cx.families.access.local.payload = Some(size);
            }
        }
    }
    Ok(())
}

/// The refusal of a store into `arena` the function may not write: a mesh function's payload is
/// its object function's, which it reads only (`TaskPayloadWorkgroupEXT` is read-only in a
/// `MeshEXT` entry; Metal's mesh function takes it `const`).
pub(super) fn writable(cx: &SelectCx<'_>, arena: Arena) -> Result<(), Error> {
    if arena == Arena::Payload && cx.stage != Stage::Object {
        return Err(refuse());
    }
    Ok(())
}

/// An arena offset as a `u32`; an offset past 2³² is not representable in the arena's pointer.
fn offset32(off: u64) -> Result<u32, Error> {
    u32::try_from(off).map_err(|_| refuse())
}

/// The most constant expressions [`addressed_global`] steps through (the constant graph is
/// acyclic; the bound only keeps the walk finite on any input).
const CONST_WALK: usize = 64;

/// The global constant pointer `c` addresses: a global's address, or a constant
/// `getelementptr` or same-space cast of one (a bounded walk of their base operands).
fn addressed_global(cx: &SelectCx<'_>, c: ConstId) -> Option<AirGlobalId> {
    match constant_base(cx, c)? {
        Const::GlobalAddr(g) => Some(*g),
        _ => None,
    }
}

/// The constant at the base of constant pointer `c`: `c` itself, or the base operand of the
/// constant `getelementptr`s and same-space casts it is made of (a bounded walk).
fn constant_base<'m>(cx: &SelectCx<'m>, mut c: ConstId) -> Option<&'m Const> {
    for _ in 0..CONST_WALK {
        let value = &cx.entry.module.consts().get(c)?.value;
        match value {
            Const::Expr(ConstExpr::Gep { ops, .. }) => c = *ops.first()?,
            Const::Expr(ConstExpr::Cast {
                op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                ops: [o],
            }) => c = *o,
            _ => return Some(value),
        }
    }
    None
}

/// The integer threadgroup address constant pointer `c` is: an `inttoptr` of an integer literal,
/// through same-space casts (no `getelementptr`, whose displacement of a literal is not
/// measured). `None` for any other constant.
fn literal_address(cx: &SelectCx<'_>, mut c: ConstId) -> Option<u64> {
    let consts = cx.entry.module.consts();
    for _ in 0..CONST_WALK {
        match &consts.get(c)?.value {
            Const::Expr(ConstExpr::Cast {
                op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                ops: [o],
            }) => c = *o,
            Const::Expr(ConstExpr::Cast {
                op: CastOp::IntToPtr,
                ops: [o],
            }) => {
                return match consts.get(*o)?.value {
                    Const::Int(k) => Some(k),
                    _ => None,
                };
            }
            _ => return None,
        }
    }
    None
}

/// The operand form of arena pointer `v` that a load, store, or atomic of `bytes` bytes accesses:
/// `local_ptr`'s, but for an integer threadgroup address (`literal_address`), the arena offset
/// of its byte where all `bytes` lie inside the static objects.
///
/// Metal lays a kernel's static threadgroup objects out in the reverse of the order of their
/// last references, each aligned, from 0, and an integer address reaches that byte (measured on
/// an M3 Max and an M4: an atomic subtract at byte 1024 and a store at bytes 16, 1024 and 2064,
/// each the byte of the object the layout puts there); the threadgroup arena is that layout
/// (`memory::plan`), so the byte is the arena's own. Where the planner could not fix Metal's
/// order (an instruction naming two objects, `ArenaLayout::metal_layout`) the address is
/// refused. Past the static objects Metal pads its allocation to 16 bytes where a regionless
/// arena ends, and past the allocation its answer was measured only for one threadgroup, so
/// such an address stays refused, as does one a `getelementptr` displaces. So does every one in
/// a module that declares function constants: Metal lays out the specialized function, whose
/// references a constant may remove (`Entry::function_constants`).
pub(super) fn access_ptr(
    cx: &SelectCx<'_>,
    v: Value,
    arena: Arena,
    bytes: u64,
) -> Result<LocalPtr, Error> {
    let literal = match (arena, v) {
        (Arena::Threadgroup, Value::Const(c)) => literal_address(cx, c),
        _ => None,
    };
    let Some(k) = literal else {
        return local_ptr(cx, v, Home::Arena(arena));
    };
    let inside = k
        .checked_add(bytes)
        .is_some_and(|end| end <= cx.memory.tg_arena.static_size);
    if !inside || !cx.memory.tg_arena.metal_layout || cx.entry.function_constants {
        return Err(Unsupported::not_representable(Feature::DereferenceLoadedPointer).into());
    }
    offset32(k).map(LocalPtr::Offset)
}

/// The refusal of a constant threadgroup pointer that addresses no global where `access_ptr`
/// does not take it: an `inttoptr` of an integer literal (or a constant `getelementptr` or cast
/// of one) is `NotRepresentable` `DereferenceLoadedPointer`; any other such constant stays the
/// access family's `NotImplemented`. Such an address names a byte of Metal's threadgroup layout,
/// which a load, store, or atomic of it inside the static objects reaches; a pointer value of it,
/// or a `getelementptr` of it, can reach any byte, Metal's answer for which is measured only inside
/// its allocation.
fn threadgroup_constant_refusal(cx: &SelectCx<'_>, c: ConstId) -> Error {
    let consts = cx.entry.module.consts();
    let literal = match constant_base(cx, c) {
        Some(Const::Expr(ConstExpr::Cast {
            op: CastOp::IntToPtr,
            ops: [o],
        })) => matches!(consts.get(*o).map(|k| &k.value), Some(Const::Int(_))),
        _ => false,
    };
    if literal {
        Unsupported::not_representable(Feature::DereferenceLoadedPointer).into()
    } else {
        refuse()
    }
}

/// The operand form of pointer `v` into `home`: a constant global's address is 0; a threadgroup
/// global's address is its arena offset, and a constant `getelementptr` of one that offset plus
/// the memory plan's exact displacement from the global, in the wrapping `u32` arithmetic of a
/// GEP. Any other constant pointer is refused.
pub(super) fn local_ptr(cx: &SelectCx<'_>, v: Value, home: Home) -> Result<LocalPtr, Error> {
    let Value::Const(c) = v else {
        return Ok(LocalPtr::Value(operand(cx, v)?));
    };
    let k = cx.entry.module.consts().get(c).ok_or_else(refuse)?;
    match (&k.value, home) {
        (Const::GlobalAddr(g), Home::Constant(i))
            if cx
                .memory
                .const_globals
                .get(i as usize)
                .is_some_and(|c| c.global == *g) =>
        {
            Ok(LocalPtr::Offset(0))
        }
        // A constant `getelementptr` or cast of the global: its exact displacement.
        (Const::Expr(_), Home::Constant(i))
            if addressed_global(cx, c).is_some_and(|g| {
                cx.memory
                    .const_globals
                    .get(i as usize)
                    .is_some_and(|c| c.global == g)
            }) =>
        {
            let displacement = cx
                .memory
                .offset(v)
                .and_then(OffsetRange::exact)
                .ok_or_else(refuse)?;
            Ok(LocalPtr::Offset(offset32(
                u64::try_from(displacement).map_err(|_| refuse())?,
            )?))
        }
        (Const::GlobalAddr(g), Home::Shared(i))
            if cx
                .memory
                .typed_shared
                .get(i as usize)
                .is_some_and(|s| s.global == *g) =>
        {
            Ok(LocalPtr::Offset(0))
        }
        // A constant `getelementptr` or cast of the object: its exact displacement.
        (Const::Expr(_), Home::Shared(i))
            if addressed_global(cx, c).is_some_and(|g| {
                cx.memory
                    .typed_shared
                    .get(i as usize)
                    .is_some_and(|s| s.global == g)
            }) =>
        {
            let displacement = cx
                .memory
                .offset(v)
                .and_then(OffsetRange::exact)
                .ok_or_else(refuse)?;
            Ok(LocalPtr::Offset(offset32(
                u64::try_from(displacement).map_err(|_| refuse())?,
            )?))
        }
        (Const::GlobalAddr(_) | Const::Expr(_), Home::Arena(Arena::Threadgroup)) => {
            let g = addressed_global(cx, c).ok_or_else(|| threadgroup_constant_refusal(cx, c))?;
            let displacement = cx
                .memory
                .offset(v)
                .and_then(OffsetRange::exact)
                .ok_or_else(refuse)?;
            let e = cx
                .memory
                .tg_arena
                .entry(ArenaObject::Global(g))
                .ok_or_else(|| cx.invariant("a threadgroup global outside the arena"))?;
            // Modulo 2³², like a GEP's constant (the truncation is the wrap).
            Ok(LocalPtr::Offset(
                offset32(e.offset)?.wrapping_add(displacement as u32),
            ))
        }
        _ => Err(refuse()),
    }
}

/// The lowered representation of an arena or typed-object pointer `v`
/// (`DemoteTy::ArenaOffset`), `None` for any other pointer. A phi with a constant incoming
/// pointer is refused (its copy would need the constant materialized in the incoming block).
pub fn lowered_pointer(cx: &SelectCx<'_>, v: Value) -> Result<Option<DemoteTy>, Error> {
    if home_of(cx, v).is_none() {
        return Ok(None);
    }
    if super::phis::incoming(cx, v)?.is_some_and(|i| i.constant) {
        return Err(refuse());
    }
    Ok(Some(DemoteTy::ArenaOffset))
}

/// Plan a `[[threadgroup(n)]]` parameter: its value is its region's offset in the threadgroup
/// arena. A payload parameter's is 0 in the payload arena.
pub fn select_param(cx: &mut SelectCx<'_>, b: &ParamBinding) -> Result<(), Error> {
    if let ParamRole::Payload { .. } = b.role {
        return use_arena(cx, Arena::Payload);
    }
    use_arena(cx, Arena::Threadgroup)?;
    let known = cx
        .families
        .access
        .local
        .tg
        .as_ref()
        .is_some_and(|s| s.regions.iter().any(|(p, _)| *p == b.param));
    if !known {
        return Err(cx.invariant("a threadgroup parameter without an arena region"));
    }
    Ok(())
}

/// The value of `[[threadgroup(n)]]` parameter `b`: its region's offset; of a payload parameter, 0.
pub fn param_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: &ParamBinding,
) -> BResult<SValue> {
    if let ParamRole::Payload { .. } = b.role {
        return k32(fb, 0);
    }
    let off = cx
        .decl
        .families
        .access
        .local
        .tg
        .as_ref()
        .and_then(|t| t.regions.iter().find(|(p, _)| *p == b.param))
        .map(|(_, c)| *c);
    match (off, &b.role) {
        (Some(c), ParamRole::Threadgroup(_)) => fb.module().const_value(c),
        _ => Err(cx.fail(format!("parameter {} has no threadgroup region", b.param))),
    }
}

/// Declare a typed object's variable: an array of `count` elements, `Private` with a constant
/// global's initializer, or a typed threadgroup object's `Workgroup` one.
fn declare_typed(mb: &mut ModuleBuilder, o: &TypedObject) -> BResult<GlobalId> {
    let (elem, types) = alloca::spirv_types(mb, &o.ty)?;
    let u32_ = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let len = mb.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(o.count)),
    })?;
    let ty = mb.ty(SType::Array {
        elem,
        len,
        stride: None,
    })?;
    let init = match &o.init {
        Some(i) => {
            let one = alloca::declare_init(mb, &o.ty, &types, i)?;
            let parts = vec![one; o.count as usize];
            Some(mb.constant(Constant::Composite {
                ty,
                parts: parts.into(),
            })?)
        }
        None => None,
    };
    let class = match o.home {
        Home::Shared(_) => StorageClass::Workgroup,
        _ => StorageClass::Private,
    };
    mb.plain_global(class, ty, init)
}

/// Declare the arenas and typed objects the plan uses.
pub fn declare(mb: &mut ModuleBuilder, plan: &Plan) -> BResult<LocalDecl> {
    let p = &plan.families.access.local;
    let mut typed = Vec::with_capacity(p.typed.len());
    for o in &p.typed {
        typed.push((o.home, declare_typed(mb, o)?));
    }
    Ok(LocalDecl {
        tg: match &p.tg {
            Some(shape) => Some(arena::declare_threadgroup(mb, shape)?),
            None => None,
        },
        private: match p.private {
            Some(size) => Some(arena::declare_private(mb, size)?),
            None => None,
        },
        payload: match p.payload {
            Some(size) => Some(arena::declare_payload(
                mb,
                size,
                plan.families.mesh.emulated,
            )?),
            None => None,
        },
        typed,
    })
}

/// The value id of a stored value: a constant must be one the arith family materializes.
fn stored_value(cx: &SelectCx<'_>, value: Option<Value>) -> Result<Option<ValueId>, Error> {
    Ok(match value {
        None => None,
        Some(v) if matches!(v, Value::Const(_)) => {
            if cx.constant(v).is_none() {
                return Err(not_lowered(LowerFamily::Arith));
            }
            Some(
                cx.values
                    .id(v)
                    .ok_or_else(|| cx.invariant("a stored value out of range"))?,
            )
        }
        Some(v) => Some(operand(cx, v)?),
    })
}

/// A leaf load (or store of `value`) of AIR type `ty` through arena pointer `ptr`.
pub(crate) fn access(
    cx: &mut SelectCx<'_>,
    arena: Arena,
    ptr: Value,
    value: Option<Value>,
    ty: TypeId,
    align: Option<crate::air::ir::Align>,
) -> Result<LocalAccess, Error> {
    use_arena(cx, arena)?;
    if value.is_some() {
        writable(cx, arena)?;
    }
    if value.is_some() && arena == Arena::Threadgroup {
        super::tile_memory::written(cx)?;
    }
    let align = align.map_or(1, |a| a.bytes());
    let leaf = leaf::leaf_of(cx.entry, cx.profile, &cx.memory.layouts, ty, align)?;
    let ptr = access_ptr(cx, ptr, arena, u64::from(leaf.bytes))?;
    let value = stored_value(cx, value)?;
    Ok(LocalAccess {
        arena,
        ptr,
        value,
        leaf,
        contained: cx.profile.fields().containment != Containment::UnsafeOff,
        volatile: false,
    })
}

/// A pointer's offset shape while reading its definitions: `Empty` is a join's identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sh {
    Empty,
    Is(Shape),
}

fn join(a: Sh, b: Sh) -> Result<Sh, Error> {
    match (a, b) {
        (Sh::Empty, x) | (x, Sh::Empty) => Ok(x),
        (x, y) if x == y => Ok(x),
        _ => Err(refuse()),
    }
}

/// The most definitions a shape walk holds open at once (the walk's bound, kept though the walk no
/// longer recurses, so no access's outcome changed with it).
const SHAPE_WALK: usize = 256;

/// A definition's offset shape as [`shape_of`] read it, and its height: the most definitions on
/// one path of operands from it, itself included (a constant's is 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Read {
    sh: Sh,
    height: usize,
}

/// The offset shapes [`shape_of`] has read in the function being selected, by definition. A read
/// shape is a function of its definition alone, never of the walk that reached it: a definition is
/// recorded only once its own walk returns a shape, and no definition an enclosing walk holds open
/// can be below it (with that one it would form a pointer cycle, which is refused); what the walk
/// reads besides (types, constants, the memory plan, the index extensions and words) is fixed
/// before the first instruction is selected. So each definition is read once per function: a walk
/// visits each definition it reaches once (a DAG of selects, phis, or GEPs sharing operands cost a
/// walk per path before, exponential in its depth), and a later access through a read pointer costs
/// one lookup. A read's height keeps [`SHAPE_WALK`] exact: reaching a read definition with `n`
/// definitions open refuses, as reading it again would, when `n` plus its height passes the bound.
#[derive(Clone, Debug, Default)]
pub(crate) struct Shapes {
    read: BTreeMap<InstId, Read>,
    /// The ledger bytes `read` holds.
    charged: usize,
}

impl Shapes {
    /// Forget every read shape and release its ledger bytes: selection is over.
    pub(crate) fn release(&mut self, m: &mut Meter) {
        self.read.clear();
        m.release(self.charged);
        self.charged = 0;
    }
}

/// A definition [`shape_of`] holds open: the join of the shapes of its operands read so far
/// (for a GEP, its base's), the most height among them, and its next operand.
struct Open<'b> {
    at: InstId,
    inst: &'b Inst,
    next: usize,
    joined: Sh,
    below: usize,
}

/// Operand `k` of a definition the shape walk reads: a GEP's base, a cast's or `freeze`'s
/// operand, a select's arms, a phi's incoming values.
fn shape_operand(kind: &InstKind, k: usize) -> Option<Value> {
    match kind {
        InstKind::Gep { base, .. } => (k == 0).then_some(*base),
        InstKind::Cast { v, .. } | InstKind::Freeze(v) => (k == 0).then_some(*v),
        InstKind::Select { t, f, .. } => [*t, *f].get(k).copied(),
        InstKind::Phi { incoming, .. } => incoming.get(k).map(|(x, _)| *x),
        _ => None,
    }
}

/// The offset shape of constant pointer `c` (`v`): a global's address is `Exact(0)`, a constant
/// `getelementptr` or cast of a global its exact displacement.
fn constant_shape(cx: &SelectCx<'_>, c: ConstId, v: Value) -> Result<Sh, Error> {
    match cx.entry.module.consts().get(c).map(|k| &k.value) {
        Some(Const::GlobalAddr(_)) => Ok(Sh::Is(Shape::Exact(0))),
        Some(Const::Expr(_)) if addressed_global(cx, c).is_some() => cx
            .memory
            .offset(v)
            .and_then(OffsetRange::exact)
            .map(|d| Sh::Is(Shape::Exact(d)))
            .ok_or_else(refuse),
        _ => Err(refuse()),
    }
}

/// The offset shape of `getelementptr src_ty, base, indices` from its base's shape `base`: its
/// constant added, and a dynamic multiple of each of its strides (one per nested array level).
fn gep_shape(
    cx: &mut SelectCx<'_>,
    base: Sh,
    src_ty: TypeId,
    indices: &[Value],
) -> Result<Sh, Error> {
    let (konst, terms) = super::gep_offset(cx, src_ty, indices)?;
    // A typed object is below 2³¹ bytes: the wrapped constant is its signed value.
    let k = i64::from(konst as i32);
    let mut added: Option<Strides> = None;
    for (_, s, _) in &terms {
        added = Some(
            match added {
                None => Strides::one(u64::from(*s)),
                Some(acc) => acc.with(u64::from(*s)),
            }
            .ok_or_else(refuse)?,
        );
    }
    Ok(match (base, added) {
        (Sh::Empty, _) => Sh::Empty,
        (Sh::Is(Shape::Exact(c)), None) => Sh::Is(Shape::Exact(c.saturating_add(k))),
        (Sh::Is(Shape::Exact(c)), Some(strides)) => Sh::Is(Shape::Strided {
            base: c.saturating_add(k),
            strides,
        }),
        (Sh::Is(Shape::Strided { base, strides }), more) => Sh::Is(Shape::Strided {
            base: base.saturating_add(k),
            strides: match more {
                None => strides,
                Some(more) => strides.union(more).ok_or_else(refuse)?,
            },
        }),
    })
}

/// The offset shape of pointer `v` into a typed object (module doc). The walk reads `v`'s
/// definitions depth first without recursing, holding at most [`SHAPE_WALK`] open; each value
/// it reads is a unit of work, and each definition is read once per function ([`Shapes`]). A
/// definition is joined from its operands in order, so the first refusal is the one a recursive
/// walk meets.
fn shape_of(cx: &mut SelectCx<'_>, v: Value, m: &mut Meter) -> Result<Sh, Error> {
    let body = cx.body;
    let mut open: Vec<Open<'_>> = Vec::new();
    let mut next = v;
    loop {
        m.tick(1)?;
        // `next`'s shape when it is known at once: a constant, or a definition read before.
        let mut read = match next {
            Value::Const(c) => Some(Read {
                sh: constant_shape(cx, c, next)?,
                height: 0,
            }),
            Value::Inst(i) => match cx.shapes.read.get(&i) {
                Some(r) => {
                    // Reading it again would open `height` more definitions on its deepest path.
                    if open.len().saturating_add(r.height) > SHAPE_WALK {
                        return Err(refuse());
                    }
                    Some(*r)
                }
                None => {
                    // A pointer cycle is not implemented: its shape needs a fixpoint.
                    if open.iter().any(|o| o.at == i) || open.len() >= SHAPE_WALK {
                        return Err(refuse());
                    }
                    let inst = body
                        .insts
                        .get(i)
                        .ok_or_else(|| cx.invariant("a pointer without a definition"))?;
                    match &inst.kind {
                        InstKind::Alloca { .. }
                        | InstKind::Gep { .. }
                        | InstKind::Cast {
                            op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                            ..
                        }
                        | InstKind::Freeze(_)
                        | InstKind::Select { .. }
                        | InstKind::Phi { .. } => {}
                        _ => return Err(refuse()),
                    }
                    open.push(Open {
                        at: i,
                        inst,
                        next: 0,
                        joined: Sh::Empty,
                        below: 0,
                    });
                    None
                }
            },
            _ => return Err(refuse()),
        };
        // Give the read shape to the definition that asked for it, and finish each definition
        // whose operands are all read, until one has an operand left to read.
        loop {
            let Some(mut done) = open.pop() else {
                return read
                    .map(|r| r.sh)
                    .ok_or_else(|| cx.invariant("a shape walk that read nothing"));
            };
            if let Some(r) = read {
                done.joined = join(done.joined, r.sh)?;
                done.below = done.below.max(r.height);
                done.next += 1;
            }
            if let Some(x) = shape_operand(&done.inst.kind, done.next) {
                open.push(done);
                next = x;
                break;
            }
            let sh = match &done.inst.kind {
                InstKind::Alloca { .. } => Sh::Is(Shape::Exact(0)),
                InstKind::Gep {
                    src_ty, indices, ..
                } => gep_shape(cx, done.joined, *src_ty, indices)?,
                _ => done.joined,
            };
            let r = Read {
                sh,
                height: done.below.saturating_add(1),
            };
            let bytes = tree_bytes(1, std::mem::size_of::<(InstId, Read)>());
            m.charge(bytes)?;
            cx.shapes.charged = cx.shapes.charged.saturating_add(bytes);
            cx.shapes.read.insert(done.at, r);
            read = Some(r);
        }
    }
}

/// The dynamic indices of the `getelementptr` that defines typed access pointer `ptr`, one per
/// dynamic level of `path` in order, where they line up one to one with those levels; `None` where
/// they do not (or `ptr` is no `getelementptr`), and the access recovers each index from its
/// offset. Only `ptr`'s own definition is read, so each access costs the same however its pointer
/// was made.
///
/// They line up where the GEP's base is at a constant offset into the object (no dynamic part
/// before the GEP), the GEP has exactly one dynamic index per dynamic level, and each, in order,
/// has the level's stride and steps over the same elements: an array index over an array of the
/// level's length, or the GEP's first (pointer) index over the object's outermost level from the
/// object's start, which spans the whole object (`count` elements of `obj`). Within its level, each
/// such index is the one the offset recovers; a vector index, or a level the GEP does not index,
/// falls back.
///
/// Where they line up, `at` (the access) reads the operands where it stands
/// (`AccessPlan::typed_indices`), so each is carried to its block as the access's other operands
/// are.
fn lined_up(
    cx: &mut SelectCx<'_>,
    at: InstId,
    ptr: Value,
    path: &TypedPath,
    (obj, count): (TypeId, u64),
    m: &mut Meter,
) -> Result<Option<Vec<(ValueId, IndexForm)>>, Error> {
    let levels: Vec<alloca::DynLevel> = path.dynamic().collect();
    if levels.is_empty() {
        return Ok(None);
    }
    m.tick(1)?;
    let Value::Inst(i) = ptr else {
        return Ok(None);
    };
    let Some(inst) = cx.body.insts.get(i) else {
        return Err(cx.invariant("a pointer without a definition"));
    };
    let InstKind::Gep {
        base,
        src_ty,
        indices,
        ..
    } = &inst.kind
    else {
        return Ok(None);
    };
    let (base, src_ty) = (*base, *src_ty);
    let Sh::Is(Shape::Exact(start)) = shape_of(cx, base, m)? else {
        return Ok(None);
    };
    let (_, terms) = super::gep_offset_indexed(cx, src_ty, indices)?;
    if terms.len() != levels.len() {
        return Ok(None);
    }
    let whole = cx
        .memory
        .layouts
        .alloc_size(obj)
        .and_then(|s| s.checked_mul(count));
    let mut out = Vec::with_capacity(terms.len());
    for (j, (t, d)) in terms.iter().zip(&levels).enumerate() {
        let steps_over = match t.over {
            Indexed::Array(len) => len == u64::from(d.len),
            Indexed::Pointer => {
                j == 0
                    && start == 0
                    && d.start == 0
                    && whole == Some(u64::from(d.stride) * u64::from(d.len))
            }
            Indexed::Vector => false,
        };
        if t.stride != u64::from(d.stride) || !steps_over {
            return Ok(None);
        }
        out.push((t.index, t.form));
    }
    let reads: Vec<Value> = terms.iter().map(|t| t.operand).collect();
    cx.families.access.typed_indices.insert(at, reads);
    Ok(Some(out))
}

/// A load or store of type `ty` through `ptr` into typed object `home`, by instruction `reader`
/// where it is one (an intrinsic's store has none, and recovers its indices from the offset).
fn typed_access(
    cx: &mut SelectCx<'_>,
    home: Home,
    (ptr, reader): (Value, Option<InstId>),
    value: Option<Value>,
    ty: TypeId,
    m: &mut Meter,
) -> Result<TypedAccess, Error> {
    use_home(cx, home, m)?;
    // Only scalar and vector accesses (an aggregate access is not lowered).
    let leaf = matches!(
        cx.entry.module.types().get(ty),
        Some(
            Type::Int(_)
                | Type::Half
                | Type::BFloat
                | Type::Float
                | Type::Double
                | Type::Vector { .. }
        )
    );
    if !leaf {
        return Err(refuse());
    }
    let held = crate::lower::arith::lowered_type(cx.entry, cx.profile, ty)?;
    let (obj, count) = object(cx, home)?;
    let shape = match shape_of(cx, ptr, m)? {
        Sh::Is(s) => s,
        Sh::Empty => return Err(refuse()),
    };
    // An access of a held matrix's elements 0 and 1, or of one of them, is the fragment's, or one
    // lane of it: its held form must be the fragment's, or the lane's.
    let part = match fragments(cx, home) {
        true => alloca::fragment_path(cx.entry, &cx.memory.layouts, obj, count, shape, ty),
        false => None,
    };
    let (path, fragment, part) = match part {
        Some((path, part, matrix)) => {
            let whole = crate::lower::arith::lowered_type(cx.entry, cx.profile, matrix)?;
            let pair = fragment::fragment_form(&whole)
                .ok_or_else(|| cx.invariant("a held matrix without a fragment form"))?;
            let fits = match (part, &pair) {
                (alloca::FragmentPart::Pair, _) => held == pair,
                (alloca::FragmentPart::Lane(_), DemoteTy::Vector { elem, .. }) => {
                    held == DemoteTy::Scalar(*elem)
                }
                _ => false,
            };
            if !fits {
                return Err(refuse());
            }
            (path, None, Some(part))
        }
        None => {
            let path = alloca::path(cx.entry, &cx.memory.layouts, obj, count, shape, ty)
                .ok_or_else(refuse)?;
            // A typed object holds a simdgroup matrix only as its fragment, which is accessed
            // whole or by its held elements: the memory plan typed no object whose matrices
            // hold more (`memory::fragment`).
            let types = cx.entry.module.types();
            let fragment = match (fragment::is_matrix(types, ty), fragments(cx, home)) {
                (false, false) => None,
                (false, true) if !fragment::enters_matrix(types, obj, &path.steps) => None,
                (true, true) => Some(held),
                _ => return Err(cx.invariant("an access into a simdgroup matrix's fragment")),
            };
            (path, fragment, None)
        }
    };
    let clamped =
        path.dynamic().next().is_some() && cx.profile.fields().isolation == GuestIsolation::On;
    let indices = match reader {
        Some(at) => lined_up(cx, at, ptr, &path, (obj, count), m)?,
        None => None,
    };
    let ptr = local_ptr(cx, ptr, home)?;
    let value = stored_value(cx, value)?;
    Ok(TypedAccess {
        home,
        ptr,
        value,
        path,
        indices,
        clamped,
        volatile: false,
        fragment,
        part,
    })
}

/// A store of one scalar or vector through a thread pointer that an intrinsic writes (a
/// multi-level intersector's instance ids), into the private arena or a typed alloca: the pointer's
/// access with no stored value of its own, the intrinsic's emission supplying the value
/// ([`emit_intrinsic_store`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntrinsicStore {
    Arena(LocalAccess),
    Typed(TypedAccess),
}

impl IntrinsicStore {
    /// An upper estimate of the store's words.
    pub fn words(&self) -> u32 {
        match self {
            IntrinsicStore::Arena(a) => a.words(),
            IntrinsicStore::Typed(a) => a.words(),
        }
    }
}

/// The store of a value of AIR type `ty` through thread pointer `ptr`, which an intrinsic writes:
/// the arena or typed-object access a plain store through `ptr` would take. A pointer into another
/// home (a constant global, a buffer, null) is refused.
pub fn intrinsic_store(
    cx: &mut SelectCx<'_>,
    ptr: Value,
    ty: TypeId,
    m: &mut Meter,
) -> Result<IntrinsicStore, Error> {
    let align = cx
        .memory
        .layouts
        .abi_align(ty)
        .and_then(crate::air::ir::Align::from_bytes);
    match home_of(cx, ptr) {
        Some(Home::Arena(arena)) => Ok(IntrinsicStore::Arena(access(
            cx, arena, ptr, None, ty, align,
        )?)),
        Some(home @ Home::Alloca(_)) => Ok(IntrinsicStore::Typed(typed_access(
            cx,
            home,
            (ptr, None),
            None,
            ty,
            m,
        )?)),
        Some(Home::Constant(_) | Home::Shared(_)) | None => Err(refuse()),
    }
}

/// Store `v` through intrinsic store `s`, contained.
pub fn emit_intrinsic_store(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &IntrinsicStore,
    v: SValue,
) -> BResult<()> {
    match s {
        IntrinsicStore::Arena(a) => emit_access(cx, fb, a, Some(v)).map(|_| ()),
        IntrinsicStore::Typed(a) => emit_typed(cx, fb, a, Some(v)).map(|_| ()),
    }
}

/// Select the arena or typed-object strategy of `inst`, or `None` when it is neither (the buffer
/// family's `select` then handles or refuses it).
pub fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    m: &mut Meter,
) -> Result<Option<LocalStrategy>, Error> {
    let here = Value::Inst(at);
    let same = |cx: &SelectCx<'_>, v: Value, home: Home| -> Result<(), Error> {
        if home_of(cx, v) == Some(home) {
            Ok(())
        } else {
            Err(refuse())
        }
    };
    Ok(Some(match &inst.kind {
        InstKind::Alloca { .. } => match home_of(cx, here) {
            Some(Home::Arena(Arena::Private)) => {
                let e = cx
                    .memory
                    .priv_arena
                    .entry(ArenaObject::Alloca(at))
                    .ok_or_else(|| cx.invariant("an arena alloca outside the private arena"))?;
                let offset = offset32(e.offset)?;
                use_arena(cx, Arena::Private)?;
                LocalStrategy::Alloca {
                    home: Home::Arena(Arena::Private),
                    offset,
                }
            }
            Some(home @ Home::Alloca(i)) => {
                if cx.memory.typed_allocas.get(i as usize).map(|a| a.inst) != Some(at) {
                    return Err(cx.invariant("a typed alloca of another instruction"));
                }
                use_home(cx, home, m)?;
                LocalStrategy::Alloca { home, offset: 0 }
            }
            _ => return Ok(None),
        },
        InstKind::Gep {
            src_ty,
            base,
            indices,
            ..
        } => {
            let Some(home) = home_of(cx, here) else {
                return Ok(None);
            };
            same(cx, *base, home)?;
            use_home(cx, home, m)?;
            let base = local_ptr(cx, *base, home)?;
            let (konst, terms) = super::gep_offset(cx, *src_ty, indices)?;
            LocalStrategy::Gep {
                home,
                base,
                konst,
                terms,
            }
        }
        InstKind::Cast {
            op: CastOp::Bitcast | CastOp::AddrSpaceCast,
            v,
        }
        | InstKind::Freeze(v)
            if is_ptr_ty(cx, inst.ty) =>
        {
            let Some(home) = home_of(cx, here) else {
                return Ok(None);
            };
            same(cx, *v, home)?;
            use_home(cx, home, m)?;
            LocalStrategy::Copy {
                home,
                of: local_ptr(cx, *v, home)?,
            }
        }
        InstKind::Select { c, t, f, .. } if is_ptr_ty(cx, inst.ty) => {
            let Some(home) = home_of(cx, here) else {
                return Ok(None);
            };
            same(cx, *t, home)?;
            same(cx, *f, home)?;
            use_home(cx, home, m)?;
            let c = match c {
                Value::Const(_) => return Err(refuse()),
                c => operand(cx, *c)?,
            };
            LocalStrategy::Select {
                home,
                c,
                t: local_ptr(cx, *t, home)?,
                f: local_ptr(cx, *f, home)?,
            }
        }
        InstKind::ICmp { pred, a, b } => return compare(cx, *pred, *a, *b, m),
        InstKind::Load {
            ptr,
            align,
            volatile: false,
            atomic: None,
            ..
        } if !is_ptr_ty(cx, inst.ty) => match home_of(cx, *ptr) {
            None => return Ok(None),
            Some(Home::Arena(arena)) => {
                LocalStrategy::Load(access(cx, arena, *ptr, None, inst.ty, *align)?)
            }
            Some(home) => LocalStrategy::TypedLoad(typed_access(
                cx,
                home,
                (*ptr, Some(at)),
                None,
                inst.ty,
                m,
            )?),
        },
        InstKind::Store {
            val,
            ptr,
            align,
            volatile: false,
            atomic: None,
        } => {
            let Some(home) = home_of(cx, *ptr) else {
                return Ok(None);
            };
            let ty = cx
                .value_ty(*val)
                .ok_or_else(|| cx.invariant("a stored value without a type"))?;
            if is_ptr_ty(cx, ty) {
                return Err(refuse());
            }
            match home {
                Home::Arena(arena) => {
                    LocalStrategy::Store(access(cx, arena, *ptr, Some(*val), ty, *align)?)
                }
                // A constant global in `Private` form is never written.
                Home::Constant(_) => return Err(refuse()),
                Home::Alloca(_) | Home::Shared(_) => LocalStrategy::TypedStore(typed_access(
                    cx,
                    home,
                    (*ptr, Some(at)),
                    Some(*val),
                    ty,
                    m,
                )?),
            }
        }
        _ => return Ok(None),
    }))
}

/// Select `icmp pred a, b` of two thread or threadgroup pointers: two pointers into one arena, or
/// into one typed object, compare their offsets, which are exact within one object and order the
/// objects of an arena by the plan's layout (MSL leaves their order unspecified). A pointer whose
/// roots are objects only (no null, no `Dynamic`) is never null, so an unsigned compare with the
/// null constant is a constant. `None` for pointers of other spaces (the buffer and constant
/// families').
fn compare(
    cx: &mut SelectCx<'_>,
    pred: IntPredicate,
    a: Value,
    b: Value,
    m: &mut Meter,
) -> Result<Option<LocalStrategy>, Error> {
    let space = cx
        .value_ty(a)
        .and_then(|t| cx.entry.module.types().ptr_space(t));
    if !matches!(space, Some(AddrSpace::Thread | AddrSpace::Threadgroup)) {
        return Ok(None);
    }
    let null = |cx: &SelectCx<'_>, v: Value| match v {
        Value::Const(c) => matches!(
            cx.entry.module.consts().get(c).map(|k| &k.value),
            Some(Const::Null)
        ),
        _ => false,
    };
    // A pointer that may be null or may be any arena offset.
    let nullable = |cx: &SelectCx<'_>, v: Value| {
        cx.memory
            .roots_of(v)
            .is_none_or(|r| r.iter().any(|r| matches!(r, Root::Null | Root::Dynamic)))
    };
    let (ptr, against_null) = match (null(cx, a), null(cx, b)) {
        (false, true) => (a, Some(pred)),
        (true, false) => (b, Some(swapped(pred))),
        (true, true) => return Err(refuse()),
        (false, false) => (a, None),
    };
    let home = home_of(cx, ptr).ok_or_else(refuse)?;
    let Some(against_null) = against_null else {
        if home_of(cx, b) != Some(home) || nullable(cx, a) || nullable(cx, b) {
            return Err(refuse());
        }
        use_home(cx, home, m)?;
        return Ok(Some(LocalStrategy::Compare {
            home,
            pred,
            a: local_ptr(cx, a, home)?,
            b: local_ptr(cx, b, home)?,
        }));
    };
    if nullable(cx, ptr) {
        return Err(refuse());
    }
    // `ptr` against null, whose address is below every object's.
    let result = match against_null {
        IntPredicate::Eq | IntPredicate::Ult | IntPredicate::Ule => false,
        IntPredicate::Ne | IntPredicate::Ugt | IntPredicate::Uge => true,
        IntPredicate::Sgt | IntPredicate::Sge | IntPredicate::Slt | IntPredicate::Sle => {
            return Err(refuse())
        }
    };
    Ok(Some(LocalStrategy::NullCompare { home, result }))
}

/// `pred` with its operands exchanged.
fn swapped(pred: IntPredicate) -> IntPredicate {
    use IntPredicate as P;
    match pred {
        P::Eq => P::Eq,
        P::Ne => P::Ne,
        P::Ugt => P::Ult,
        P::Uge => P::Ule,
        P::Ult => P::Ugt,
        P::Ule => P::Uge,
        P::Sgt => P::Slt,
        P::Sge => P::Sle,
        P::Slt => P::Sgt,
        P::Sle => P::Sge,
    }
}

/// The SPIR-V comparison of an LLVM integer predicate.
fn int_compare(pred: IntPredicate) -> BinOp {
    match pred {
        IntPredicate::Eq => BinOp::IEqual,
        IntPredicate::Ne => BinOp::INotEqual,
        IntPredicate::Ugt => BinOp::UGreaterThan,
        IntPredicate::Uge => BinOp::UGreaterThanEqual,
        IntPredicate::Ult => BinOp::ULessThan,
        IntPredicate::Ule => BinOp::ULessThanEqual,
        IntPredicate::Sgt => BinOp::SGreaterThan,
        IntPredicate::Sge => BinOp::SGreaterThanEqual,
        IntPredicate::Slt => BinOp::SLessThan,
        IntPredicate::Sle => BinOp::SLessThanEqual,
    }
}

/// The declaration of `arena` as the open function reaches it: for a fragment function's tile
/// memory, with the fragment's tile's first word in the pass's buffer.
pub(super) fn decl_of(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    arena: Arena,
) -> BResult<ArenaDecl> {
    let d = &cx.decl.families.access.local;
    let (found, fragment) = match arena {
        Arena::Threadgroup => (
            d.tg.as_ref().map(|t| t.decl),
            d.tg.as_ref().and_then(|t| t.fragment),
        ),
        Arena::Private => (d.private, None),
        Arena::Payload => (d.payload, None),
    };
    let Some(mut decl) = found else {
        return Err(cx.fail(format!("the {arena:?} arena is not declared")));
    };
    if let Some(tile) = fragment {
        decl.tile_base = Some(cx.tile_base(fb, tile)?);
    }
    Ok(decl)
}

fn ptr_value(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, p: LocalPtr) -> BResult<SValue> {
    match p {
        LocalPtr::Value(v) => cx.operand(fb, v),
        LocalPtr::Offset(o) => k32(fb, o),
    }
}

/// Load or store (`stored`) `a`, contained.
pub(crate) fn emit_access(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    a: &LocalAccess,
    stored: Option<SValue>,
) -> BResult<Option<SValue>> {
    let off = ptr_value(cx, fb, a.ptr)?;
    emit_access_at(cx, fb, a, off, stored)
}

/// Load or store (`stored`) `a`'s leaf at arena byte offset `off` (`a.ptr`'s value, or that
/// pointer displaced by a memory intrinsic), contained.
pub(super) fn emit_access_at(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    a: &LocalAccess,
    off: SValue,
    stored: Option<SValue>,
) -> BResult<Option<SValue>> {
    let decl = decl_of(cx, fb, a.arena)?;
    let leaf = a.leaf;
    let ops = WordOps {
        coherence: crate::lower::sync::memory_model::coherence(cx.plan),
        volatile: a.volatile,
    };
    // A tile function's cells the pass keeps in part read their other words as zero and drop
    // their writes (`access::tile::KeptCells`).
    let kept = match a.arena {
        Arena::Threadgroup => KeptCells::of(cx, fb)?,
        Arena::Private | Arena::Payload => None,
    };
    let mut mem = KeptMem {
        mem: ArenaMem(decl, ops),
        kept,
    };
    if !a.contained {
        return match stored {
            None => leaf::load(fb, &mut mem, &leaf, off).map(Some),
            Some(v) => leaf::store(fb, &mut mem, &leaf, off, v).map(|_| None),
        };
    }
    let size = fb.module().const_value(decl.size)?;
    let inb = contain::in_bounds(fb, off, size, u64::from(leaf.bytes))?;
    // A constant offset's check is decided at translation time: the access runs unchecked when it
    // is in bounds, and is a zero load or no store when it is not.
    if let Some(inside) = fb.const_bool(inb)? {
        return match (stored, inside) {
            (None, true) => leaf::load(fb, &mut mem, &leaf, off).map(Some),
            (None, false) => leaf::zero(fb, &leaf).map(Some),
            (Some(v), true) => leaf::store(fb, &mut mem, &leaf, off, v).map(|_| None),
            (Some(_), false) => Ok(None),
        };
    }
    match stored {
        None => {
            let zero = k32(fb, 0)?;
            let at = fb.select(inb, off, zero)?;
            let v = leaf::load(fb, &mut mem, &leaf, at)?;
            let z = leaf::zero(fb, &leaf)?;
            fb.select(inb, v, z).map(Some)
        }
        Some(v) => {
            let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
                let mut mem = KeptMem {
                    mem: ArenaMem(decl, ops),
                    kept,
                };
                leaf::store(fb, &mut mem, &leaf, off, v)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            fb.if_(&mut (), inb, then, None)?;
            Ok(None)
        }
    }
}

/// The access-chain indices of `a`'s path, and the check's condition where `a` is checked
/// ([`TypedAccess::checked`]). Each dynamic index is the defining GEP's operand where they line up
/// (`TypedAccess::indices`), or is recovered from the pointer's offset level by level (`(rem -
/// start) / stride`, the next level's `rem` being `(rem - start) % stride`, `memory::alloca`).
/// When `a.clamped`, a checked access's condition is every level's `idx < len`, a load reading
/// each level at `select(idx < len, idx, 0)` (a store's chain is built under the condition); any
/// other access clamps each index to its level's last element, `min(idx, len - 1)`. Either way the
/// chain never leaves the object.
fn chain_indices(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    a: &TypedAccess,
    load: bool,
) -> BResult<(Vec<SValue>, Option<SValue>)> {
    let levels = a.path.dynamic().count();
    let mut rem = match (levels, &a.indices) {
        (0, _) | (_, Some(_)) => None,
        _ => Some(ptr_value(cx, fb, a.ptr)?),
    };
    let mut seen = 0;
    let checked = a.checked();
    let mut ok: Option<SValue> = None;
    let mut out = Vec::with_capacity(a.path.steps.len());
    for step in &a.path.steps {
        let d = match step {
            Step::Const(i) => {
                out.push(k32(fb, *i)?);
                continue;
            }
            Step::Dyn(d) => *d,
        };
        let idx = match &a.indices {
            Some(indices) => {
                let Some(&(v, form)) = indices.get(seen) else {
                    return Err(cx.fail("a dynamic level without its GEP index"));
                };
                seen += 1;
                let x = cx.operand(fb, v)?;
                index_u32(fb, x, form)?
            }
            None => {
                let Some(off) = rem else {
                    return Err(cx.fail("a dynamic level without an offset"));
                };
                seen += 1;
                let rel = if d.start == 0 {
                    off
                } else {
                    let s = k32(fb, d.start)?;
                    fb.binary(BinOp::ISub, off, s)?
                };
                let idx = match (d.stride.is_power_of_two(), d.stride.trailing_zeros()) {
                    (true, 0) => rel,
                    (true, sh) => {
                        let sh = k32(fb, sh)?;
                        fb.binary(BinOp::ShiftRightLogical, rel, sh)?
                    }
                    (false, _) => {
                        let s = k32(fb, d.stride)?;
                        fb.binary(BinOp::UDiv, rel, s)?
                    }
                };
                // The offset inside the element, for the next level.
                rem = if seen == levels {
                    None
                } else if d.stride.is_power_of_two() {
                    let m = k32(fb, d.stride - 1)?;
                    Some(fb.binary(BinOp::BitwiseAnd, rel, m)?)
                } else {
                    let s = k32(fb, d.stride)?;
                    Some(fb.binary(BinOp::UMod, rel, s)?)
                };
                idx
            }
        };
        if !a.clamped {
            out.push(idx);
            continue;
        }
        if !checked {
            let last = k32(fb, d.len.saturating_sub(1))?;
            out.push(fb.glsl(GlslOp::UMin, &[idx, last])?);
            continue;
        }
        let len = k32(fb, d.len)?;
        let in_range = fb.binary(BinOp::ULessThan, idx, len)?;
        ok = Some(match ok {
            None => in_range,
            Some(all) => fb.binary(BinOp::LogicalAnd, all, in_range)?,
        });
        out.push(if load {
            let zero = k32(fb, 0)?;
            fb.select(in_range, idx, zero)?
        } else {
            idx
        });
    }
    Ok((out, ok))
}

/// Load or store (`stored`) typed access `a` through an access chain of its path, contained when
/// `a.clamped` ([`chain_indices`]): a checked load past a level yields zero and a checked store
/// past one is dropped.
fn emit_typed(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    a: &TypedAccess,
    stored: Option<SValue>,
) -> BResult<Option<SValue>> {
    let found = cx
        .plan
        .families
        .access
        .local
        .position(a.home)
        .and_then(|i| cx.decl.families.access.local.typed.get(i))
        .filter(|(h, _)| *h == a.home)
        .map(|(_, g)| *g);
    let Some(g) = found else {
        return Err(cx.fail(format!("typed object {:?} is not declared", a.home)));
    };
    let (mut idx, ok) = chain_indices(cx, fb, a, stored.is_none())?;
    // One held element of a matrix's fragment is that lane of it.
    if let Some(alloca::FragmentPart::Lane(i)) = a.part {
        idx.push(k32(fb, i)?);
    }
    let base = fb.module().global_value(g)?;
    let plain = MemAccess {
        volatile: a.volatile,
        ..MemAccess::default()
    };
    // A typed threadgroup object's loads and stores take the memory model's operands, as the
    // threadgroup arena's words do (`memory::arena::ArenaMem`).
    let (load_access, store_access) = match a.home {
        Home::Shared(_) => {
            let ops = WordOps {
                coherence: crate::lower::sync::memory_model::coherence(cx.plan),
                volatile: a.volatile,
            };
            (
                ops.access(fb, MemAccess::default(), false)?,
                ops.access(fb, MemAccess::default(), true)?,
            )
        }
        _ => (plain, plain),
    };
    // A matrix held as its fragment stores elements 0 and 1.
    let stored = match (stored, &a.fragment) {
        (Some(v), Some(_)) => {
            let first = fb.composite_extract(v, &[0])?;
            Some(fb.vector_shuffle(first, first, &[0, 1])?)
        }
        (stored, _) => stored,
    };
    match (stored, ok) {
        (None, ok) => {
            let p = fb.access_chain(base, &idx)?;
            let v = fb.load(p, load_access)?;
            let v = match ok {
                None => v,
                Some(ok) => {
                    let ty = fb.value_ty(v)?;
                    let z = fb.module().constant(Constant::Null { ty })?;
                    let z = fb.module().const_value(z)?;
                    fb.select(ok, v, z)?
                }
            };
            match &a.fragment {
                Some(d) => rebuild(cx, fb, d, v).map(Some),
                None => Ok(Some(v)),
            }
        }
        (Some(v), None) => {
            let p = fb.access_chain(base, &idx)?;
            fb.store(p, v, store_access)?;
            Ok(None)
        }
        (Some(v), Some(ok)) => {
            let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
                let p = fb.access_chain(base, &idx)?;
                fb.store(p, v, store_access)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            fb.if_(&mut (), ok, then, None)?;
            Ok(None)
        }
    }
}

/// The simdgroup matrix held as `d` (its chunked vector) whose fragment, elements 0 and 1, is
/// the 2-lane vector `v`: zero in elements 2..63 (`memory::fragment`).
fn rebuild(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    d: &DemoteTy,
    v: SValue,
) -> BResult<SValue> {
    let DemoteTy::Composite(chunks) = d else {
        return Err(cx.fail("a matrix fragment's value is not chunked"));
    };
    let Some(ty) = crate::lower::arith::spirv_type(fb.module(), d)? else {
        return Err(cx.fail("a matrix fragment's value does not type"));
    };
    let pair = fb.value_ty(v)?;
    let zero = fb.module().constant(Constant::Null { ty: pair })?;
    let zero = fb.module().const_value(zero)?;
    let mut members = Vec::with_capacity(chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        let lanes = match chunk {
            DemoteTy::Vector { len, .. } => usize::from(*len),
            _ => return Err(cx.fail("a matrix fragment's chunk is not a vector")),
        };
        if i == 0 {
            let Some(order) = [0, 1, 2, 3].get(..lanes) else {
                return Err(cx.fail("a matrix fragment's chunk is past four lanes"));
            };
            members.push(fb.vector_shuffle(v, zero, order)?);
            continue;
        }
        let Some(cty) = crate::lower::arith::spirv_type(fb.module(), chunk)? else {
            return Err(cx.fail("a matrix fragment's chunk does not type"));
        };
        let c = fb.module().constant(Constant::Null { ty: cty })?;
        members.push(fb.module().const_value(c)?);
    }
    fb.composite_construct(ty, &members)
}

/// Emit arena or typed-object strategy `s`, defining value `vid`.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    vid: ValueId,
    s: &LocalStrategy,
) -> BResult<()> {
    let r = match s {
        LocalStrategy::Alloca { offset, .. } => k32(fb, *offset)?,
        LocalStrategy::Gep {
            base, konst, terms, ..
        } => {
            let mut off = ptr_value(cx, fb, *base)?;
            if *konst != 0 {
                let c = k32(fb, *konst)?;
                off = fb.binary(BinOp::IAdd, off, c)?;
            }
            for (v, stride, form) in terms {
                let x = cx.operand(fb, *v)?;
                let mut x = index_u32(fb, x, *form)?;
                if *stride != 1 {
                    let s = k32(fb, *stride)?;
                    x = fb.binary(BinOp::IMul, x, s)?;
                }
                off = fb.binary(BinOp::IAdd, off, x)?;
            }
            off
        }
        LocalStrategy::Copy { of, .. } => ptr_value(cx, fb, *of)?,
        LocalStrategy::Select { c, t, f, .. } => {
            let c = cx.operand(fb, *c)?;
            let t = ptr_value(cx, fb, *t)?;
            let f = ptr_value(cx, fb, *f)?;
            fb.select(c, t, f)?
        }
        LocalStrategy::Compare { pred, a, b, .. } => {
            let a = ptr_value(cx, fb, *a)?;
            let b = ptr_value(cx, fb, *b)?;
            fb.binary(int_compare(*pred), a, b)?
        }
        LocalStrategy::NullCompare { result, .. } => {
            let ty = fb.module().ty(SType::Bool)?;
            let c = fb.module().constant(Constant::Scalar {
                ty,
                bits: ConstBits(u64::from(*result)),
            })?;
            fb.module().const_value(c)?
        }
        LocalStrategy::Load(a) => match emit_access(cx, fb, a, None)? {
            Some(v) => v,
            None => return Err(cx.fail("an arena load without a value")),
        },
        LocalStrategy::Store(a) => {
            let Some(val) = a.value else {
                return Err(cx.fail("an arena store without a value"));
            };
            let v = cx.operand(fb, val)?;
            emit_access(cx, fb, a, Some(v))?;
            return Ok(());
        }
        LocalStrategy::TypedLoad(a) => match emit_typed(cx, fb, a, None)? {
            Some(v) => v,
            None => return Err(cx.fail("a typed load without a value")),
        },
        LocalStrategy::TypedStore(a) => {
            let Some(val) = a.value else {
                return Err(cx.fail("a typed store without a value"));
            };
            let v = cx.operand(fb, val)?;
            emit_typed(cx, fb, a, Some(v))?;
            return Ok(());
        }
    };
    cx.define(vid, r);
    Ok(())
}
