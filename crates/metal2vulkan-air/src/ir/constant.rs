//! Constants and the constant interner.
//!
//! Constants are module-level and interned: equal constants have equal [`ConstId`]s. As with
//! types, a constant may refer only to constants that already exist, so the constant graph is
//! acyclic and child ids precede parents. Globals and functions appear as operands only through
//! [`Const::GlobalAddr`] / [`Const::FuncAddr`], so every non-local operand is one `ConstId`.

use super::id::{Arena, ConstId, FuncId, GlobalId, TypeId};
use super::inst::{BinOp, CastOp, IntFlags, Value};
use super::intern::FxHashMap;
use crate::budget::Meter;
use crate::error::Error;
use crate::error::{InputError, InputErrorKind, Location, VerifyError};

/// A typed constant.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Constant {
    /// The constant's type.
    pub ty: TypeId,
    /// The constant's value.
    pub value: Const,
}

/// The value of a [`Constant`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Const {
    /// An integer, zero-extended to 64 bits (wider integers are refused at decode).
    Int(u64),
    /// A floating-point value's bit pattern, zero-extended to 64 bits.
    Float(u64),
    /// The null pointer or the all-zero value of any sized type (`zeroinitializer`).
    Null,
    /// `undef`: an unspecified value of the type, which may differ at each use.
    Undef,
    /// `poison`: the result of an operation whose preconditions failed.
    Poison,
    /// A struct, array, or vector; `DATA`/`STRING` records expand here.
    Aggregate(Box<[ConstId]>),
    /// The address of a global (typed `Ptr` in the global's space).
    GlobalAddr(GlobalId),
    /// The address of a function (typed `Ptr` in the thread space).
    FuncAddr(FuncId),
    /// A constant expression.
    Expr(ConstExpr),
    /// An inline assembly expression (`asm`), typed `ptr`: only ever the callee of a call, at the
    /// call's function type. The IR keeps its strings and flags as written and gives it no
    /// meaning; canonicalization refuses a call to it that an entry reaches.
    InlineAsm(Box<InlineAsm>),
}

/// The text and flags of an inline assembly expression ([`Const::InlineAsm`]), as LLVM's
/// `CST_CODE_INLINEASM` records spell them.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InlineAsm {
    /// The assembly template.
    pub template: Box<[u8]>,
    /// The constraint string: the operands, results and clobbers.
    pub constraints: Box<[u8]>,
    /// `sideeffect`: the expression has effects its constraints do not list.
    pub side_effects: bool,
    /// `alignstack`: the stack is aligned for it.
    pub align_stack: bool,
    /// `inteldialect`: the template is in Intel syntax rather than AT&T.
    pub intel_dialect: bool,
    /// `unwind`: it may throw.
    pub can_throw: bool,
}

impl InlineAsm {
    /// The bytes it holds outside its box.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.template.len().saturating_add(self.constraints.len())
    }

    /// Its LLVM spelling: `asm`, the flags, then the quoted template and constraints, with LLVM's
    /// `\XX` escapes (`asm sideeffect "", ""`).
    pub fn spelling(&self) -> String {
        let mut out = String::from("asm");
        for (on, word) in [
            (self.side_effects, " sideeffect"),
            (self.align_stack, " alignstack"),
            (self.intel_dialect, " inteldialect"),
            (self.can_throw, " unwind"),
        ] {
            if on {
                out.push_str(word);
            }
        }
        for (i, text) in [&self.template, &self.constraints].into_iter().enumerate() {
            out.push_str(if i == 0 { " \"" } else { ", \"" });
            for &b in text.iter() {
                if (b.is_ascii_graphic() || b == b' ') && b != b'"' && b != b'\\' {
                    out.push(char::from(b));
                } else {
                    out.push_str(&format!("\\{b:02X}"));
                }
            }
            out.push('"');
        }
        out
    }
}

impl Const {
    /// The constants this constant refers to directly.
    pub fn children(&self) -> &[ConstId] {
        match self {
            Const::Aggregate(elems) => elems,
            Const::Expr(e) => e.operands(),
            Const::Int(_)
            | Const::Float(_)
            | Const::Null
            | Const::Undef
            | Const::Poison
            | Const::GlobalAddr(_)
            | Const::FuncAddr(_)
            | Const::InlineAsm(_) => &[],
        }
    }

    /// The bytes it holds outside its slot: its children's ids, or an inline assembly
    /// expression's box and strings.
    pub(crate) fn heap_bytes(&self) -> usize {
        match self {
            Const::InlineAsm(a) => std::mem::size_of::<InlineAsm>().saturating_add(a.heap_bytes()),
            other => std::mem::size_of::<ConstId>().saturating_mul(other.children().len()),
        }
    }
}

/// A constant expression (`CE_CAST`, `CE_BINOP`, `CE_GEP`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConstExpr {
    /// A cast of `ops[0]` to the constant's type.
    Cast {
        /// The cast operator.
        op: CastOp,
        /// The constant cast.
        ops: [ConstId; 1],
    },
    /// An integer binary operation over `ops[0]` and `ops[1]`.
    BinOp {
        /// The operator.
        op: BinOp,
        /// The integer wrap/exactness flags.
        flags: IntFlags,
        /// The left and right operands.
        ops: [ConstId; 2],
    },
    /// `getelementptr` over `src_ty`; `ops[0]` is the base, the rest are indices.
    Gep {
        /// The source element type: the first index steps over it, the rest index into it.
        src_ty: TypeId,
        /// `inbounds`: the address stays within the object the base points into.
        inbounds: bool,
        /// The base pointer, then the indices.
        ops: Box<[ConstId]>,
    },
}

impl ConstExpr {
    /// Every operand constant, in order.
    pub fn operands(&self) -> &[ConstId] {
        match self {
            ConstExpr::Cast { ops, .. } => ops,
            ConstExpr::BinOp { ops, .. } => ops,
            ConstExpr::Gep { ops, .. } => ops,
        }
    }
}

/// A malformed-constant error at module scope.
pub(crate) fn malformed_constant() -> Error {
    Error::InvalidInput(InputError::new(
        InputErrorKind::Verify(VerifyError::MalformedConstant),
        Location::module(),
    ))
}

/// The module's constant interner.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConstTable {
    consts: Arena<ConstId, Constant>,
    index: FxHashMap<Constant, ConstId>,
}

impl ConstTable {
    /// An empty table.
    pub fn new() -> ConstTable {
        ConstTable::default()
    }

    /// Intern `c`. Every constant it refers to must already exist; typing is checked by
    /// [`super::Module::new`], which also sees the types, globals, and functions.
    pub fn intern(&mut self, c: Constant, meter: &mut Meter) -> Result<ConstId, Error> {
        if let Some(&id) = self.index.get(&c) {
            return Ok(id);
        }
        if !c.value.children().iter().all(|&e| self.consts.contains(e)) {
            return Err(malformed_constant());
        }
        // Stored twice: once in the arena, once as the lookup key.
        meter.charge(c.value.heap_bytes().saturating_mul(2))?;
        meter.charge(std::mem::size_of::<Constant>())?;
        let id = self.consts.push(c.clone(), meter)?;
        self.index.insert(c, id);
        Ok(id)
    }

    /// The id of `c` if it is interned.
    #[cfg(test)]
    pub fn lookup(&self, c: &Constant) -> Option<ConstId> {
        self.index.get(c).copied()
    }

    /// The constant at `id`.
    pub fn get(&self, id: ConstId) -> Option<&Constant> {
        self.consts.get(id)
    }

    /// The inline assembly expression `v` is, if it is one.
    pub(crate) fn inline_asm(&self, v: Value) -> Option<&InlineAsm> {
        match v {
            Value::Const(c) => match &self.get(c)?.value {
                Const::InlineAsm(asm) => Some(asm),
                _ => None,
            },
            Value::Arg(_) | Value::Inst(_) => None,
        }
    }

    /// The type of the constant at `id`.
    pub fn ty(&self, id: ConstId) -> Option<TypeId> {
        self.get(id).map(|c| c.ty)
    }

    /// Whether `id` is in range.
    pub fn contains(&self, id: ConstId) -> bool {
        self.consts.contains(id)
    }

    /// The number of constants.
    pub fn len(&self) -> usize {
        self.consts.len()
    }

    /// Whether there is no constant.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Every `(id, constant)` in id order; children precede parents.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (ConstId, &Constant)> + '_ {
        self.consts.iter()
    }
}
