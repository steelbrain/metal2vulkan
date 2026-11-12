//! The function-constant specializations the specialization audit translates a row under.
//!
//! The plain translation census compiles every row with no function-constant values, Metal's
//! "undefined", so a row whose code uses a constant's value stops at `NeedsInput(FunctionConstant)`
//! and everything behind that use goes unmeasured. The corpus holds no values a real pipeline used:
//! harvest reads metallibs only, the PVG host never receives values (external facts Q1), and AIR
//! declares no default (each initializer is `externally_initialized undef`; an MSL "default" is
//! code, `is_function_constant_defined(c) ? c : d`, which the undefined compile already runs). The
//! only real values are the authored cases' (qualified on Metal). Every other value set here is a
//! deterministic measurement value, never a guess at what an application sends.
//!
//! A row's plan ([`plan`]) is built from its declared constants ([`declared_constants`]): each is
//! **optional** when the module names its initializer in an `air.is_function_constant_defined`
//! call (the code handles it undefined), else **required**. The sets, each a [`Specialization`]
//! carrying every [`SpecKind`] that produced it:
//! - [`SpecKind::Authored`]: each authored case's constants;
//! - [`SpecKind::Zero`] / [`SpecKind::One`]: the required constants 0 / 1, the optional ones
//!   undefined;
//! - [`SpecKind::ZeroDefined`] / [`SpecKind::OneDefined`]: every constant 0 / 1;
//! - [`SpecKind::Covering`]: a strength-2 binary covering array over every constant ([`covering`]):
//!   each set defines all of them, each 0 or 1, and every pair of constants takes all four
//!   combinations somewhere (every combination for at most three constants). A row's first
//!   refusal hides the rest, so a set where one gate is on and another off reaches code no
//!   all-on or all-off set does.
//!
//! Equal value sets collapse into one that lists each kind. The empty set is never planned: it is
//! the plain census's compile. [`MAX_SPECIALIZATIONS`] is the plan size the design states; a plan
//! above it is kept whole and reported, never truncated.
//!
//! A plan also keeps the functions the row's authored cases link ([`AuthoredLinks`]): a real
//! pipeline's linked functions, exact inputs an attempt links under every set.
//!
//! A constant can feed an argument's binding: its slot (`[[texture(c)]]`) or a table's length.
//! Metal binds one resource per slot of an argument table, so a set under which two live
//! arguments of the row's entry share a slot ([`shared_slot`]; a zero puts `[[texture(c)]]` at
//! another texture's 0) is one Metal refuses when it makes the function ("argument has invalid
//! location"; the translation's `OverlappingArgument`). A constant can also gate an output, and a
//! set that leaves two color outputs at one attachment and blend index ([`shared_color`]) is one
//! Metal refuses when it makes the function (as the translation does, `DuplicateColorOutput`). The plan keeps such a synthetic set apart ([`AliasedSet`]) and
//! the audit does not attempt it: its refusal would measure the planner's value, not the
//! translation. An authored set, which Metal ran, is always attempted.

pub mod attempt;
pub mod audit;

use crate::case::AuthoredCase;
use crate::check::interface::{constant_int, AirInterface, Output, Param};
use crate::hash::sha256_bytes;
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::DataType;
use metal2vulkan_air::abi::{FunctionConstantDecl, OutputRole, ParamRole};
use metal2vulkan_air::canon::ConstValue;
use metal2vulkan_air::intrinsics::{AirIntrinsic, FunctionConstant as FcIntrinsic};
use metal2vulkan_air::ir::{Callee, Const, GlobalId, InstKind, Module, Type, Value};
use metal2vulkan_air::Decoded;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// The plan size the design states: every plan of a row with at most 126 constants and no authored
/// case fits. A plan above it is kept whole and counted as over the cap in the report.
pub const MAX_SPECIALIZATIONS: usize = 16;

/// Which rule produced a specialization (a set may come from several).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecKind {
    /// An authored case's constants, qualified on Metal: the corpus's only real values.
    Authored,
    /// The required constants 0, the optional ones undefined.
    Zero,
    /// The required constants 1, the optional ones undefined.
    One,
    /// Every constant 0.
    ZeroDefined,
    /// Every constant 1.
    OneDefined,
    /// A row of the strength-2 binary covering array over every constant.
    Covering,
    /// No constants: a row whose plain compile needed another pipeline input, compiled with the
    /// inputs the audit supplies (`specialization::inputs`).
    Inputs,
}

impl SpecKind {
    pub const ALL: [SpecKind; 7] = [
        SpecKind::Authored,
        SpecKind::Zero,
        SpecKind::One,
        SpecKind::ZeroDefined,
        SpecKind::OneDefined,
        SpecKind::Covering,
        SpecKind::Inputs,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            SpecKind::Authored => "authored",
            SpecKind::Zero => "zero",
            SpecKind::One => "one",
            SpecKind::ZeroDefined => "zero_defined",
            SpecKind::OneDefined => "one_defined",
            SpecKind::Covering => "covering",
            SpecKind::Inputs => "inputs",
        }
    }

    pub fn parse(key: &str) -> Option<SpecKind> {
        SpecKind::ALL.into_iter().find(|kind| kind.as_str() == key)
    }
}

/// The element of a constant's stored form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Elem {
    /// An integer of this many bytes (`bool` is stored as `i8`).
    Int(usize),
    Half,
    BFloat,
    Float,
}

impl Elem {
    fn bytes(self) -> usize {
        match self {
            Elem::Int(bytes) => bytes,
            Elem::Half | Elem::BFloat => 2,
            Elem::Float => 4,
        }
    }

    /// One lane of 1 in this element's little-endian encoding.
    fn one(self) -> Vec<u8> {
        match self {
            Elem::Int(bytes) => {
                let mut out = vec![0; bytes];
                out[0] = 1;
                out
            }
            Elem::Half => 0x3c00u16.to_le_bytes().to_vec(),
            Elem::BFloat => 0x3f80u16.to_le_bytes().to_vec(),
            Elem::Float => 1.0f32.to_le_bytes().to_vec(),
        }
    }
}

/// One declared function constant as the planner reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeclaredConstant {
    pub index: u32,
    /// The `MTLDataType` its values are given as: the container's when a metallib states it,
    /// else the unsigned (or floating) type of its stored form, which the product's check
    /// accepts for every MSL type of that form (`air::abi::data_type_is`).
    pub data_type: DataType,
    elem: Elem,
    lanes: usize,
    /// The module names its initializer in an `air.is_function_constant_defined` call.
    pub optional: bool,
    /// The fifth operand of its `!air.function_constants` entry, when present.
    pub flag: Option<bool>,
}

impl DeclaredConstant {
    /// Its value with every lane 0 (`false`, `+0.0`) or 1 (`true`, `1.0`).
    fn value(&self, one: bool) -> FunctionConstantValue {
        let lane = if one {
            self.elem.one()
        } else {
            vec![0; self.elem.bytes()]
        };
        FunctionConstantValue::new(self.data_type, lane.repeat(self.lanes))
    }
}

/// What a row's declarations say, summed by the report: how many constants it declares, how
/// many are optional, and whether the fifth `!air.function_constants` operand agrees with
/// "required" (true exactly when no `air.is_function_constant_defined` call names the constant).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstantFacts {
    pub declared: u32,
    pub optional: u32,
    pub flag_agrees: u32,
    pub flag_disagrees: u32,
    pub flag_absent: u32,
}

impl ConstantFacts {
    fn of(constants: &[DeclaredConstant]) -> ConstantFacts {
        let mut facts = ConstantFacts::default();
        for constant in constants {
            facts.declared += 1;
            facts.optional += u32::from(constant.optional);
            match constant.flag {
                Some(flag) if flag == !constant.optional => facts.flag_agrees += 1,
                Some(_) => facts.flag_disagrees += 1,
                None => facts.flag_absent += 1,
            }
        }
        facts
    }

    pub fn add(&mut self, other: &ConstantFacts) {
        self.declared += other.declared;
        self.optional += other.optional;
        self.flag_agrees += other.flag_agrees;
        self.flag_disagrees += other.flag_disagrees;
        self.flag_absent += other.flag_absent;
    }
}

/// The declared function constants of `decoded`, in index order, or why one has no value the
/// planner can give (a stored form outside the `MTLDataType`s).
pub fn declared_constants(decoded: &Decoded) -> Result<Vec<DeclaredConstant>, String> {
    let named = defined_queries(&decoded.module);
    let mut out = decoded
        .abi
        .function_constants
        .iter()
        .map(|decl| declared(&decoded.module, decl, named.contains(&decl.global)))
        .collect::<Result<Vec<_>, _>>()?;
    out.sort_by_key(|constant| constant.index);
    Ok(out)
}

fn declared(
    module: &Module,
    decl: &FunctionConstantDecl,
    optional: bool,
) -> Result<DeclaredConstant, String> {
    let types = module.types();
    let unsupported = || {
        format!(
            "function constant {} has IR type {:?}, which no MTLDataType stores",
            decl.index,
            types.get(decl.ty)
        )
    };
    let (elem_ty, lanes) = match types.get(decl.ty) {
        Some(Type::Vector { elem, len }) if (2..=4).contains(len) => (*elem, *len as usize),
        Some(Type::Vector { .. }) => return Err(unsupported()),
        _ => (decl.ty, 1),
    };
    let elem = match types.get(elem_ty) {
        Some(Type::Int(bits)) if matches!(bits, 8 | 16 | 32 | 64) => Elem::Int(*bits as usize / 8),
        Some(Type::Half) => Elem::Half,
        Some(Type::BFloat) => Elem::BFloat,
        Some(Type::Float) => Elem::Float,
        _ => return Err(unsupported()),
    };
    let data_type = match decl.data_type {
        Some(data_type) => data_type,
        None => stored_data_type(elem, lanes).ok_or_else(unsupported)?,
    };
    Ok(DeclaredConstant {
        index: decl.index,
        data_type,
        elem,
        lanes,
        optional,
        flag: decl.flag,
    })
}

/// The `MTLDataType` this planner gives a stored form when the container states none.
fn stored_data_type(elem: Elem, lanes: usize) -> Option<DataType> {
    use DataType as D;
    let row: [D; 4] = match elem {
        Elem::Int(1) => [D::UChar, D::UChar2, D::UChar3, D::UChar4],
        Elem::Int(2) => [D::UShort, D::UShort2, D::UShort3, D::UShort4],
        Elem::Int(4) => [D::UInt, D::UInt2, D::UInt3, D::UInt4],
        Elem::Int(8) => [D::ULong, D::ULong2, D::ULong3, D::ULong4],
        Elem::Half => [D::Half, D::Half2, D::Half3, D::Half4],
        Elem::BFloat => [D::BFloat, D::BFloat2, D::BFloat3, D::BFloat4],
        Elem::Float => [D::Float, D::Float2, D::Float3, D::Float4],
        Elem::Int(_) => return None,
    };
    row.get(lanes.checked_sub(1)?).copied()
}

/// The globals some `air.is_function_constant_defined` call of `module` names.
fn defined_queries(module: &Module) -> HashSet<GlobalId> {
    let consts = module.consts();
    let mut named = HashSet::new();
    for function in module.parts().funcs.values() {
        let Some(body) = &function.body else {
            continue;
        };
        for inst in body.insts.values() {
            let InstKind::Call(call) = &inst.kind else {
                continue;
            };
            let Callee::Air {
                op: AirIntrinsic::FunctionConstant(FcIntrinsic::IsDefined),
                ..
            } = call.callee
            else {
                continue;
            };
            if let Some(Value::Const(c)) = call.args.first() {
                if let Some(Const::GlobalAddr(g)) = consts.get(*c).map(|k| &k.value) {
                    named.insert(*g);
                }
            }
        }
    }
    named
}

/// One function-constant value set and every kind that produced it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Specialization {
    pub kinds: BTreeSet<SpecKind>,
    /// The supplied values by constant index; a constant absent here is undefined.
    pub constants: BTreeMap<u32, FunctionConstantValue>,
}

impl Specialization {
    /// The set's identity: SHA-256 over each supplied constant's index, `MTLDataType`, and
    /// bytes, in index order (the kinds are not part of it).
    pub fn id(&self) -> String {
        id_of(&self.constants)
    }

    /// The values as the product takes them.
    pub fn values(&self) -> FunctionConstantValues {
        values_of(&self.constants)
    }

    /// The kinds, comma-joined in their order (the index and report spelling).
    pub fn kinds_key(&self) -> String {
        self.kinds
            .iter()
            .map(|kind| kind.as_str())
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// `constants` as the product takes them; a constant absent from the map is undefined.
pub fn values_of(constants: &BTreeMap<u32, FunctionConstantValue>) -> FunctionConstantValues {
    constants.iter().fold(
        FunctionConstantValues::defaults(),
        |values, (index, value)| values.with_index(*index, value.clone()),
    )
}

/// [`Specialization::id`] of `constants`.
pub fn id_of(constants: &BTreeMap<u32, FunctionConstantValue>) -> String {
    let mut bytes = b"metal2vulkan-specialization".to_vec();
    for (index, value) in constants {
        bytes.extend_from_slice(&index.to_le_bytes());
        bytes.extend_from_slice(&value.ty.as_raw().to_le_bytes());
        bytes.extend_from_slice(&(value.bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&value.bytes);
    }
    sha256_bytes(&bytes)
}

/// The rows of a strength-2 binary covering array over `k` factors: every combination for at
/// most three, else Kleitman and Spencer's optimal construction. With `N` the least row count
/// where `C(N-1, ceil(N/2)) >= k`, factor `j` is the `j`-th `ceil(N/2)`-subset of `0..N-1` in
/// lexicographic order and is 1 in exactly the rows its subset holds. Two distinct subsets of
/// that size intersect (a row of 1 1), neither holds the other (rows of 1 0 and 0 1), and row
/// `N-1` is in none (0 0), so every pair of factors takes all four combinations. Rows of 4
/// factors: 5; up to 10: 6; up to 15: 7; up to 35: 8; up to 56: 9; up to 126: 10.
pub fn covering(k: usize) -> Vec<Vec<bool>> {
    if k == 0 {
        return Vec::new();
    }
    if k <= 3 {
        return (0..1usize << k)
            .map(|row| (0..k).map(|j| (row >> j) & 1 == 1).collect())
            .collect();
    }
    let mut n = 4usize;
    while binomial(n - 1, n.div_ceil(2)) < k {
        n += 1;
    }
    let size = n.div_ceil(2);
    let mut columns = Vec::with_capacity(k);
    let mut subset = (0..size).collect::<Vec<_>>();
    while columns.len() < k {
        columns.push(subset.clone());
        if !next_subset(&mut subset, n - 1) {
            break;
        }
    }
    (0..n)
        .map(|row| columns.iter().map(|column| column.contains(&row)).collect())
        .collect()
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    (0..k).fold(1usize, |acc, i| acc.saturating_mul(n - i) / (i + 1))
}

/// Advance `subset` (sorted, of `0..n`) to the next subset of its size in lexicographic order;
/// false after the last.
fn next_subset(subset: &mut [usize], n: usize) -> bool {
    let size = subset.len();
    for i in (0..size).rev() {
        if subset[i] < n - size + i {
            subset[i] += 1;
            for j in i + 1..size {
                subset[j] = subset[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

/// The plan of a row declaring `constants`, with the authored cases' value sets `authored` (see
/// the module docs). Sets are ordered by their first kind, then by identity.
pub fn plan(
    constants: &[DeclaredConstant],
    authored: &[BTreeMap<u32, FunctionConstantValue>],
) -> Vec<Specialization> {
    let mut sets = BTreeMap::<String, Specialization>::new();
    let mut add = |kind: SpecKind, values: BTreeMap<u32, FunctionConstantValue>| {
        if values.is_empty() {
            return;
        }
        sets.entry(id_of(&values))
            .or_insert_with(|| Specialization {
                kinds: BTreeSet::new(),
                constants: values,
            })
            .kinds
            .insert(kind);
    };
    for values in authored {
        add(SpecKind::Authored, values.clone());
    }
    let required = |one: bool| {
        constants
            .iter()
            .filter(|constant| !constant.optional)
            .map(|constant| (constant.index, constant.value(one)))
            .collect::<BTreeMap<_, _>>()
    };
    let every = |one: bool| {
        constants
            .iter()
            .map(|constant| (constant.index, constant.value(one)))
            .collect::<BTreeMap<_, _>>()
    };
    add(SpecKind::Zero, required(false));
    add(SpecKind::One, required(true));
    add(SpecKind::ZeroDefined, every(false));
    add(SpecKind::OneDefined, every(true));
    for row in covering(constants.len()) {
        add(
            SpecKind::Covering,
            constants
                .iter()
                .zip(row)
                .map(|(constant, one)| (constant.index, constant.value(one)))
                .collect(),
        );
    }
    let mut out = sets.into_values().collect::<Vec<_>>();
    out.sort_by(|a, b| {
        a.kinds
            .first()
            .cmp(&b.kinds.first())
            .then_with(|| a.id().cmp(&b.id()))
    });
    out
}

/// The functions a row's authored cases link, each Metal name with the retained module that
/// defines it: every case's `visible_function_references` and its visible-function tables'
/// entries. A name the cases give two modules is linked by none and kept as a conflict.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthoredLinks {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub functions: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub conflicts: BTreeSet<String>,
}

impl AuthoredLinks {
    /// The links of `cases`.
    pub fn of(cases: &[AuthoredCase]) -> AuthoredLinks {
        let mut modules = BTreeMap::<&str, BTreeSet<&str>>::new();
        for case in cases {
            let references = case
                .visible_function_references
                .iter()
                .map(|r| (r.function.as_str(), r.module_sha256.as_str()));
            let tables = case
                .visible_function_tables
                .iter()
                .flat_map(|table| &table.entries)
                .map(|e| (e.function.as_str(), e.module_sha256.as_str()));
            for (function, module) in references.chain(tables) {
                modules.entry(function).or_default().insert(module);
            }
        }
        let mut links = AuthoredLinks::default();
        for (function, modules) in modules {
            match modules.into_iter().collect::<Vec<_>>().as_slice() {
                [module] => {
                    links
                        .functions
                        .insert(function.to_string(), module.to_string());
                }
                _ => {
                    links.conflicts.insert(function.to_string());
                }
            }
        }
        links
    }

    pub fn is_empty(&self) -> bool {
        self.functions.is_empty() && self.conflicts.is_empty()
    }
}

/// A planned set under which two of the entry's bindings share what Metal gives one ([`Shared`]).
/// It is kept apart and not attempted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AliasedSet {
    pub set: Specialization,
    pub shared: Shared,
}

/// What an [`AliasedSet`] gives two of the entry's bindings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Shared {
    /// One slot of one argument table (`buffer`, `texture`, `sampler`, `threadgroup`), bound by
    /// two live arguments: their LLVM parameter indices ([`shared_slot`]).
    Slot {
        table: String,
        slot: u32,
        params: (u32, u32),
    },
    /// One color attachment at one blend index (`[[color(color), index(index)]]`), written by
    /// two outputs the set leaves: their fields of the returned struct ([`shared_color`]).
    Color {
        color: u32,
        index: u32,
        fields: (u32, u32),
    },
}

impl Shared {
    /// What the plan's summary counts it by: its argument table, or `color`.
    pub fn kind(&self) -> &str {
        match self {
            Shared::Slot { table, .. } => table,
            Shared::Color { .. } => "color",
        }
    }
}

/// The argument table a role binds in: buffers (with argument buffers, acceleration structures,
/// and function tables, which Metal binds at buffer indices), textures, samplers, or threadgroup
/// memory. `None` for a role bound by no table slot.
fn argument_table(role: &ParamRole) -> Option<&'static str> {
    match role {
        ParamRole::Buffer(_)
        | ParamRole::ArgumentBuffer(_)
        | ParamRole::AccelerationStructure { .. }
        | ParamRole::FunctionTable(_)
        | ParamRole::IntersectionTable(_) => Some("buffer"),
        ParamRole::Texture(_) => Some("texture"),
        ParamRole::Sampler { .. } => Some("sampler"),
        ParamRole::Threadgroup(_) => Some("threadgroup"),
        _ => None,
    }
}

/// The first slot two live arguments of `params` share in one argument table: the table, the
/// slot, and the two arguments' parameter indices. An argument's slots are its first slot and
/// count (a table of count 0, whose length the pipeline chooses, holds at least its first).
pub fn shared_slot(params: &[Param]) -> Option<(&'static str, u32, (u32, u32))> {
    let mut ranges = params
        .iter()
        .filter(|param| param.live)
        .filter_map(|param| {
            let table = argument_table(&param.binding.role)?;
            let (first, count) = param.slots()?;
            let end = u64::from(first) + u64::from(count.max(1));
            Some((table, first, end, param.binding.param))
        })
        .collect::<Vec<_>>();
    ranges.sort();
    ranges.windows(2).find_map(|pair| {
        let [(table, _, end, a), (next, first, _, b)] = pair else {
            return None;
        };
        (table == next && u64::from(*first) < *end).then_some((*table, *first, (*a, *b)))
    })
}

/// The first color attachment and blend index two live outputs of `outputs` write: the color,
/// the index, and the two outputs' fields. MSL refuses such a pair outright, so only constants
/// that leave both outputs of a gated pair make one, and Metal then refuses the function when it
/// makes it ("duplicated indices pair"; the translation's `DuplicateColorOutput`). An output whose
/// attachment does not evaluate is passed over.
pub fn shared_color(outputs: &[Output]) -> Option<(u32, u32, (u32, u32))> {
    let mut seen = BTreeMap::new();
    for output in outputs.iter().filter(|output| output.live) {
        let OutputRole::RenderTarget {
            attachment, index, ..
        } = output.binding.role
        else {
            continue;
        };
        let Some(color) = constant_int(attachment) else {
            continue;
        };
        let field = output.binding.field;
        if let Some(first) = seen.insert((color, index), field) {
            return Some((color, index, (first, field)));
        }
    }
    None
}

/// What two bindings of `entry` share under `set`: the slot two live arguments share
/// ([`shared_slot`]), else the color two live outputs write ([`shared_color`]); `None` also when
/// the bindings do not evaluate under the set (the attempt then says what it can).
fn aliasing(decoded: &Decoded, entry: &str, set: &Specialization) -> Option<AliasedSet> {
    let values = set
        .constants
        .iter()
        .map(|(&index, value)| {
            (
                index,
                ConstValue {
                    ty: value.ty,
                    bytes: value.bytes.clone(),
                },
            )
        })
        .collect();
    let (params, outputs) =
        AirInterface::bindings_for(&decoded.module, &decoded.abi, entry, &values).ok()?;
    let shared = match shared_slot(&params) {
        Some((table, slot, params)) => Shared::Slot {
            table: table.to_string(),
            slot,
            params,
        },
        None => {
            let (color, index, fields) = shared_color(&outputs)?;
            Shared::Color {
                color,
                index,
                fields,
            }
        }
    };
    Some(AliasedSet {
        set: set.clone(),
        shared,
    })
}

/// A row's plan as the index keeps it: the declarations' facts and the sets, or why the row has
/// none (it did not decode, or a constant has no value the planner can give), the sets kept apart
/// because they alias two bindings ([`AliasedSet`]), and the functions its authored cases link.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowPlan {
    pub facts: ConstantFacts,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub specializations: Vec<Specialization>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliased: Vec<AliasedSet>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refused: Option<String>,
    #[serde(default, skip_serializing_if = "AuthoredLinks::is_empty")]
    pub links: AuthoredLinks,
}

impl RowPlan {
    /// The plan of entry `entry` of the row `decoded` with the authored cases' value sets
    /// `authored`; a set that aliases two bindings is kept apart.
    pub fn of(
        decoded: &Decoded,
        entry: &str,
        authored: &[BTreeMap<u32, FunctionConstantValue>],
    ) -> RowPlan {
        match declared_constants(decoded) {
            Ok(constants) => {
                let mut row = RowPlan {
                    facts: ConstantFacts::of(&constants),
                    ..RowPlan::default()
                };
                for set in plan(&constants, authored) {
                    // An authored set is a pipeline Metal ran: it is attempted as it is.
                    let aliased = (!set.kinds.contains(&SpecKind::Authored))
                        .then(|| aliasing(decoded, entry, &set))
                        .flatten();
                    match aliased {
                        Some(aliased) => row.aliased.push(aliased),
                        None => row.specializations.push(set),
                    }
                }
                row
            }
            Err(why) => RowPlan {
                refused: Some(why),
                ..RowPlan::default()
            },
        }
    }

    /// Whether the plan is larger than the design's bound ([`MAX_SPECIALIZATIONS`]).
    pub fn over_cap(&self) -> bool {
        self.specializations.len() > MAX_SPECIALIZATIONS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant(index: u32, elem: Elem, lanes: usize, optional: bool) -> DeclaredConstant {
        DeclaredConstant {
            index,
            data_type: stored_data_type(elem, lanes).unwrap(),
            elem,
            lanes,
            optional,
            flag: Some(!optional),
        }
    }

    /// Every pair of factors takes all four combinations, the row count is the stated one, and
    /// the array is the same on every call.
    #[test]
    fn the_covering_array_covers_every_pair_in_the_stated_rows() {
        let expected_rows = |k: usize| match k {
            0 => 0,
            1..=3 => 1 << k,
            4 => 5,
            5..=10 => 6,
            11..=15 => 7,
            16..=35 => 8,
            36..=56 => 9,
            57..=126 => 10,
            127..=210 => 11,
            _ => 12,
        };
        for k in (0..=40).chain([56, 57, 96, 126, 127, 181, 210, 211, 300]) {
            let rows = covering(k);
            assert_eq!(rows.len(), expected_rows(k), "k={k}");
            assert!(rows.iter().all(|row| row.len() == k), "k={k}");
            for a in 0..k {
                for b in a + 1..k {
                    let seen = rows
                        .iter()
                        .map(|row| (row[a], row[b]))
                        .collect::<BTreeSet<_>>();
                    assert_eq!(
                        seen.len(),
                        4,
                        "k={k}: factors {a} and {b} miss a combination"
                    );
                }
            }
            assert_eq!(covering(k), rows, "k={k}: not deterministic");
        }
    }

    #[test]
    fn values_are_zero_and_one_in_each_stored_form() {
        let cases = [
            (Elem::Int(1), 1, vec![1u8], DataType::UChar),
            (Elem::Int(2), 1, vec![1, 0], DataType::UShort),
            (Elem::Int(4), 4, [1, 0, 0, 0].repeat(4), DataType::UInt4),
            (
                Elem::Int(8),
                1,
                vec![1, 0, 0, 0, 0, 0, 0, 0],
                DataType::ULong,
            ),
            (Elem::Half, 2, [0x00, 0x3c].repeat(2), DataType::Half2),
            (Elem::BFloat, 1, vec![0x80, 0x3f], DataType::BFloat),
            (
                Elem::Float,
                3,
                [0, 0, 0x80, 0x3f].repeat(3),
                DataType::Float3,
            ),
        ];
        for (elem, lanes, one, ty) in cases {
            let c = constant(0, elem, lanes, false);
            assert_eq!(c.data_type, ty);
            assert_eq!(c.value(true), FunctionConstantValue::new(ty, one.clone()));
            assert_eq!(
                c.value(false),
                FunctionConstantValue::new(ty, vec![0; one.len()])
            );
        }
        assert_eq!(stored_data_type(Elem::Int(1), 5), None);
        assert_eq!(stored_data_type(Elem::Int(3), 1), None);
    }

    /// A row of required constants only: the zero and one sets are the defined ones, and the
    /// covering rows add the mixed sets. The empty set is never planned.
    #[test]
    fn required_constants_plan_their_covering_array_and_the_constant_sets() {
        let constants = [
            constant(0, Elem::Int(1), 1, false),
            constant(3, Elem::Int(4), 1, false),
        ];
        let sets = plan(&constants, &[]);
        // 2 constants: every combination (4 sets); zero and one coincide with two of them.
        assert_eq!(sets.len(), 4, "{sets:#?}");
        let zero = sets
            .iter()
            .find(|s| s.kinds.contains(&SpecKind::Zero))
            .unwrap();
        assert_eq!(
            zero.kinds,
            BTreeSet::from([SpecKind::Zero, SpecKind::ZeroDefined, SpecKind::Covering])
        );
        assert!(sets.iter().all(|s| s.constants.len() == 2));
        assert!(plan(&[], &[]).is_empty());
        // Ids are the value sets'.
        let ids = sets.iter().map(Specialization::id).collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), 4);
    }

    /// Optional constants stay undefined in the zero and one sets and are defined in every
    /// other; a row of optional constants alone plans no set without them (that is the plain
    /// compile).
    #[test]
    fn optional_constants_are_undefined_only_in_the_zero_and_one_sets() {
        let constants = [
            constant(0, Elem::Int(1), 1, false),
            constant(1, Elem::Int(1), 1, true),
        ];
        let sets = plan(&constants, &[]);
        let zero = sets
            .iter()
            .find(|s| s.kinds.contains(&SpecKind::Zero))
            .unwrap();
        assert_eq!(zero.kinds, BTreeSet::from([SpecKind::Zero]));
        assert_eq!(zero.constants.keys().copied().collect::<Vec<_>>(), [0]);
        // 4 covering sets (which hold zero_defined and one_defined) plus zero and one.
        assert_eq!(sets.len(), 6, "{sets:#?}");
        let optional_only = [constant(7, Elem::Float, 1, true)];
        let sets = plan(&optional_only, &[]);
        assert!(sets.iter().all(|s| s.constants.contains_key(&7)));
        assert!(!sets.iter().any(|s| s.kinds.contains(&SpecKind::Zero)));
    }

    /// An authored set is planned as it is, and a synthetic set of the same values only adds its
    /// kind.
    #[test]
    fn authored_sets_lead_and_merge_with_equal_synthetic_sets() {
        let constants = [constant(2, Elem::Int(2), 1, false)];
        let real = BTreeMap::from([(2, FunctionConstantValue::new(DataType::UShort, [7, 0]))]);
        let same_as_one = BTreeMap::from([(2, constants[0].value(true))]);
        let sets = plan(&constants, &[real.clone(), same_as_one]);
        assert_eq!(sets[0].kinds.first(), Some(&SpecKind::Authored));
        assert_eq!(sets.len(), 3, "{sets:#?}");
        assert!(sets.iter().any(|s| s.constants == real));
        let one = sets
            .iter()
            .find(|s| s.kinds.contains(&SpecKind::One))
            .unwrap();
        assert!(one.kinds.contains(&SpecKind::Authored));
    }

    /// The facts count optional constants and the fifth operand's agreement with "required".
    #[test]
    fn facts_count_the_flag_against_the_defined_queries() {
        let mut constants = vec![
            constant(0, Elem::Int(1), 1, false),
            constant(1, Elem::Int(1), 1, true),
            constant(2, Elem::Int(4), 1, true),
        ];
        constants[2].flag = Some(true);
        constants.push(DeclaredConstant {
            flag: None,
            ..constant(3, Elem::Float, 1, false)
        });
        let facts = ConstantFacts::of(&constants);
        assert_eq!(
            facts,
            ConstantFacts {
                declared: 4,
                optional: 2,
                flag_agrees: 2,
                flag_disagrees: 1,
                flag_absent: 1,
            }
        );
    }

    /// A plan of many constants stays within the design's bound; with authored sets beside it the
    /// plan is kept whole and reports itself over the cap.
    #[test]
    fn a_plan_above_the_bound_is_kept_whole_and_reported() {
        let constants = (0..126)
            .map(|i| constant(i, Elem::Int(1), 1, false))
            .collect::<Vec<_>>();
        let sets = plan(&constants, &[]);
        assert!(sets.len() <= MAX_SPECIALIZATIONS, "{}", sets.len());
        let authored = (0..8u8)
            .map(|v| BTreeMap::from([(0, FunctionConstantValue::new(DataType::UChar, [v + 2]))]))
            .collect::<Vec<_>>();
        let row = RowPlan {
            facts: ConstantFacts::of(&constants),
            specializations: plan(&constants, &authored),
            ..RowPlan::default()
        };
        assert_eq!(row.specializations.len(), sets.len() + 8);
        assert!(row.over_cap());
    }

    /// Every function a case links by reference or places in a table is linked from its one
    /// module; a name the cases give two modules is a conflict, linked by none.
    #[test]
    fn authored_links_take_references_and_table_entries() {
        use crate::case::{FunctionTableEntry, FunctionTableResource, LinkedFunctionResource};
        let case = |references: &[(&str, &str)], entries: &[(&str, &str)]| {
            let mut case =
                crate::case::attachmentless_fragment_test_case("aa".into(), "main".into());
            case.visible_function_references = references
                .iter()
                .map(|(function, module)| LinkedFunctionResource {
                    module_sha256: module.to_string(),
                    function: function.to_string(),
                })
                .collect();
            case.visible_function_tables = vec![FunctionTableResource {
                binding: 0,
                size: 4,
                entries: entries
                    .iter()
                    .zip(0..)
                    .map(|((function, module), index)| FunctionTableEntry {
                        index,
                        module_sha256: module.to_string(),
                        function: function.to_string(),
                    })
                    .collect(),
            }];
            case
        };
        let links = AuthoredLinks::of(&[
            case(&[("f", "m1")], &[("g", "m2"), ("f", "m1")]),
            case(&[("h", "m1")], &[("h", "m3")]),
        ]);
        assert_eq!(
            links.functions,
            BTreeMap::from([("f".into(), "m1".into()), ("g".into(), "m2".into())])
        );
        assert_eq!(links.conflicts, BTreeSet::from(["h".to_string()]));
        let plan = RowPlan {
            links: links.clone(),
            ..RowPlan::default()
        };
        let json = serde_json::to_string(&plan).unwrap();
        assert_eq!(serde_json::from_str::<RowPlan>(&json).unwrap(), plan);
        assert!(!serde_json::to_string(&RowPlan::default())
            .unwrap()
            .contains("links"));
    }

    /// A kernel with `dest [[texture(0)]]` and `src [[texture(c)]]`, `c` a required `uint`
    /// constant whose value the static initializer stores into `src`'s location.
    const SLOT_FROM_CONSTANT: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@c.MTL_FC_INIT_0_j = internal unnamed_addr addrspace(2) externally_initialized constant i32 undef, section "air.fc_initializer", align 4
@expr_0 = internal addrspace(2) global i32 0, align 4

define void @k(ptr addrspace(1) %0, ptr addrspace(1) %1) {
  ret void
}

define internal void @init() section "air.static_init" {
  %1 = load i32, ptr addrspace(2) @c.MTL_FC_INIT_0_j, align 4
  store i32 %1, ptr addrspace(2) @expr_0, align 4
  ret void
}

!air.kernel = !{!0}
!air.function_constants = !{!5}
!air.version = !{!6}
!air.language_version = !{!7}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"dest"}
!4 = !{i32 1, !"air.texture", !"air.location_index", ptr addrspace(2) @expr_0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"src"}
!5 = !{ptr addrspace(2) @c.MTL_FC_INIT_0_j, !"uint", !"c", i32 0, i1 true}
!6 = !{i32 2, i32 8, i32 0}
!7 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

    /// A set under which two live textures share a slot is kept apart, and not attempted: here
    /// `c = 0` puts `src` at `dest`'s slot 0, and `c = 1` binds them apart.
    #[test]
    fn a_set_that_puts_two_arguments_in_one_slot_is_kept_apart() {
        let bitcode = match crate::engine::RowSource::of(None, SLOT_FROM_CONSTANT.into()).bitcode()
        {
            Ok(bitcode) => bitcode,
            Err(crate::engine::BitcodeError::Tooling(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(other) => panic!("{other}"),
        };
        let mut meter = metal2vulkan_air::budget::Meter::new(&metal2vulkan::Budget::standard());
        let decoded = metal2vulkan_air::decode(&bitcode, &mut meter).unwrap();
        let row = RowPlan::of(&decoded, "k", &[]);
        let value = |set: &Specialization| set.constants[&0].bytes.to_vec();
        assert_eq!(
            row.specializations.iter().map(value).collect::<Vec<_>>(),
            vec![1u32.to_le_bytes().to_vec()],
            "{row:#?}"
        );
        let [aliased] = row.aliased.as_slice() else {
            panic!("{row:#?}");
        };
        assert_eq!(value(&aliased.set), 0u32.to_le_bytes().to_vec());
        assert_eq!(
            aliased.shared,
            Shared::Slot {
                table: "texture".into(),
                slot: 0,
                params: (0, 1)
            }
        );
        assert_eq!(aliased.shared.kind(), "texture");
        let json = serde_json::to_string(&row).unwrap();
        assert_eq!(serde_json::from_str::<RowPlan>(&json).unwrap(), row);
    }

    /// A fragment function with two `float4` outputs at `[[color(0)]]` and one at
    /// `[[color(0), index(1)]]`, the first two gated by the boolean constants `use_a` (0) and
    /// `use_b` (1).
    const COLOR_FROM_CONSTANTS: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@use_a.MTL_FC_INIT_0_b = internal unnamed_addr addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@use_b.MTL_FC_INIT_1_b = internal unnamed_addr addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@pred_0 = internal addrspace(2) global i8 0, align 1
@pred_1 = internal addrspace(2) global i8 0, align 1

define <{ <4 x float>, <4 x float>, <4 x float> }> @f() {
  ret <{ <4 x float>, <4 x float>, <4 x float> }> zeroinitializer
}

declare i8 @air.normalize_function_constant_predicate.i8(i8)

define internal void @init() section "air.static_init" {
  %1 = load i8, ptr addrspace(2) @use_a.MTL_FC_INIT_0_b, align 1
  %2 = load i8, ptr addrspace(2) @use_b.MTL_FC_INIT_1_b, align 1
  %3 = tail call i8 @air.normalize_function_constant_predicate.i8(i8 %1)
  store i8 %3, ptr addrspace(2) @pred_0, align 1
  %4 = tail call i8 @air.normalize_function_constant_predicate.i8(i8 %2)
  store i8 %4, ptr addrspace(2) @pred_1, align 1
  ret void
}

!air.fragment = !{!0}
!air.function_constants = !{!10, !11}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @f, !1, !8}
!1 = !{!2, !4, !6}
!2 = !{!"air.function_constant", !3, !"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"a"}
!3 = !{ptr addrspace(2) @pred_0, !"bool", !"use_a"}
!4 = !{!"air.function_constant", !5, !"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"b"}
!5 = !{ptr addrspace(2) @pred_1, !"bool", !"use_b"}
!6 = !{!"air.render_target", i32 0, i32 1, !"air.arg_type_name", !"float4", !"air.arg_name", !"second"}
!8 = !{}
!10 = !{ptr addrspace(2) @use_a.MTL_FC_INIT_0_b, !"bool", !"use_a", i32 0}
!11 = !{ptr addrspace(2) @use_b.MTL_FC_INIT_1_b, !"bool", !"use_b", i32 1}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

    /// Every set that leaves both gated `[[color(0)]]` outputs is kept apart, naming the pair and
    /// the two fields; every other set is attempted. The output at `index(1)` shares the color,
    /// not the blend index, so it aliases neither.
    #[test]
    fn a_set_that_leaves_two_outputs_at_one_color_is_kept_apart() {
        let bitcode =
            match crate::engine::RowSource::of(None, COLOR_FROM_CONSTANTS.into()).bitcode() {
                Ok(bitcode) => bitcode,
                Err(crate::engine::BitcodeError::Tooling(why)) => {
                    eprintln!("SKIPPED: {why}");
                    return;
                }
                Err(other) => panic!("{other}"),
            };
        let mut meter = metal2vulkan_air::budget::Meter::new(&metal2vulkan::Budget::standard());
        let decoded = metal2vulkan_air::decode(&bitcode, &mut meter).unwrap();
        let row = RowPlan::of(&decoded, "f", &[]);
        let both = |set: &Specialization| {
            [0, 1].iter().all(|i| {
                set.constants
                    .get(i)
                    .is_some_and(|v| v.bytes.as_ref() == [1])
            })
        };
        assert!(!row.aliased.is_empty(), "{row:#?}");
        for aliased in &row.aliased {
            assert!(both(&aliased.set), "{row:#?}");
            assert_eq!(
                aliased.shared,
                Shared::Color {
                    color: 0,
                    index: 0,
                    fields: (0, 1)
                }
            );
            assert_eq!(aliased.shared.kind(), "color");
        }
        assert!(!row.specializations.is_empty(), "{row:#?}");
        assert!(row.specializations.iter().all(|set| !both(set)), "{row:#?}");
        let json = serde_json::to_string(&row).unwrap();
        assert_eq!(serde_json::from_str::<RowPlan>(&json).unwrap(), row);
    }
}
