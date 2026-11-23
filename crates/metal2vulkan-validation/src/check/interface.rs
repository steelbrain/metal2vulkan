//! The Metal-ABI facts of one checked entry, from the AIR input layer.
//!
//! The checker compares an authored manifest against what Metal itself binds: argument-table
//! slots and their counts, argument-buffer members with their byte offsets and `[[id]]`s, the
//! declared access of each argument, the stage interface, and the function-constant declarations.
//! Every one of those is a fact of the AIR the Metal oracle loads, so it is read from the decoded
//! module's typed [`AirAbi`] and IR, never from a translation: a product change cannot move a
//! Metal slot under an installed case, and the Metal observations recorded against these facts
//! survive the rewrite.
//!
//! - **Resolution.** An index, count, attachment, or predicate the metadata states as a global
//!   resolves to the value the static initializer leaves there for the case's function constants
//!   ([`canon::static_values`]). A parameter whose predicate is false is not *live*: Metal drops
//!   it from the specialized function, so the case need not bind it, but may (a function-constant
//!   alternative sharing its slot). Only the constants the AIR declares are bound; Metal ignores
//!   the rest. A value the AIR layer does not evaluate (a static initializer computes it through
//!   an operation outside the evaluated subset) stays unknown and never becomes a guess: an
//!   unknown predicate makes its argument optional, an unknown count leaves the range unsized, and
//!   an unknown slot is an error for a live argument and drops an optional one. Each is named in
//!   [`AirInterface::unresolved`]. Any other failure to evaluate is the checker's error.
//! - **Access.** A buffer is written when AIR declares it writable or when the IR stores through
//!   a pointer derived from it: the compiler annotates some written buffers `air.read`, and the
//!   question the checker asks is whether the shader can write the resource, not what its
//!   qualifier spells. Any pointer an intrinsic, an unknown, or an indirect call receives counts
//!   as written, which only ever widens the answer.
//! - **Texture shapes.** AIR metadata spells a texture's type only as a name. Its shape and texel
//!   class come from the `air.*` texture operations that use it, whose typed operand positions
//!   state both ([`Texture`](metal2vulkan_air::intrinsics::Texture)); a texture no operation uses
//!   has no shape.
//! - **Types.** Attributes, outputs, and constants are compared by the IR type of the value (float
//!   or integer, width, lanes). The IR does not carry an integer's sign, so neither does this
//!   comparison; the executors pass the authored format to Metal, which checks it.
//!
//! Origins are followed through casts, GEPs, `select`, `phi`, and calls to defined functions,
//! and one level of memory: a pointer or handle loaded from a parameter's memory at a known byte
//! offset is that parameter's member or table element there. A pointer loaded from any other
//! memory has no origin, and a write through it is not attributed.

use crate::case::{ResourceRole, ScalarType, TextureFormat, TextureType};
use metal2vulkan::library::FunctionConstantValues;
use metal2vulkan::metal::TextureUsage;
use metal2vulkan::Budget;
use metal2vulkan_air::abi::{
    AbiInt, Access, AirAbi, EntryPoint, MemberKind, OutputBinding, OutputRole, ParamBinding,
    ParamRole, PatchInput, Slot, Stage, StructInfo, StructRef,
};
use metal2vulkan_air::budget::Meter;
use metal2vulkan_air::canon::{self, ConstValue};
use metal2vulkan_air::intrinsics::{
    AirIntrinsic, ImageDim, ImageShape, NumTy, Sign, TexOp, TexRole,
};
use metal2vulkan_air::ir::{
    AddrSpace, Callee, Const, FuncId, GlobalId, Idx, InstKind, Module, Type, TypeId, TypeLayouts,
    Value,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Nesting depth the argument-buffer flattening follows (the ABI decoder's own bound is the
/// metadata's; this only keeps a malformed self-reference finite).
const LAYOUT_NESTING: usize = 8;

/// A value's numeric shape as the IR types it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NumShape {
    pub float: bool,
    pub bits: u32,
    pub lanes: u32,
}

impl NumShape {
    /// The shape of a value of IR type `ty` in `module`: a number or a vector of numbers.
    pub fn of(module: &Module, ty: TypeId) -> Option<NumShape> {
        let types = module.types();
        let scalar = |ty: TypeId| match types.get(ty)? {
            Type::Int(bits) => Some((false, u32::from(*bits))),
            Type::Half | Type::BFloat => Some((true, 16)),
            Type::Float => Some((true, 32)),
            Type::Double => Some((true, 64)),
            _ => None,
        };
        match types.get(ty)? {
            Type::Vector { elem, len } => {
                let (float, bits) = scalar(*elem)?;
                Some(NumShape {
                    float,
                    bits,
                    lanes: *len,
                })
            }
            _ => {
                let (float, bits) = scalar(ty)?;
                Some(NumShape {
                    float,
                    bits,
                    lanes: 1,
                })
            }
        }
    }

    /// The shape an authored scalar type with `lanes` lanes stores as. `bool` is a byte, as the
    /// frontend stores a function constant and an attribute.
    pub fn of_scalar(scalar: ScalarType, lanes: u32) -> NumShape {
        let (float, bits) = match scalar {
            ScalarType::Bool | ScalarType::I8 | ScalarType::U8 => (false, 8),
            ScalarType::I16 | ScalarType::U16 => (false, 16),
            ScalarType::F16 => (true, 16),
            ScalarType::I32 | ScalarType::U32 => (false, 32),
            ScalarType::F32 => (true, 32),
            ScalarType::I64 | ScalarType::U64 => (false, 64),
            ScalarType::F64 => (true, 64),
        };
        NumShape { float, bits, lanes }
    }

    /// Whether an `i1` value of this many lanes is a `bool` of the authored shape.
    fn matches(self, other: NumShape) -> bool {
        self == other
            || (self.bits == 1 && other.bits == 8 && !other.float && self.lanes == other.lanes)
            || (other.bits == 1 && self.bits == 8 && !self.float && self.lanes == other.lanes)
    }

    /// Whether this is the IR shape of a value of `other`'s authored shape.
    pub fn agrees(self, other: NumShape) -> bool {
        self.matches(other)
    }
}

/// The numeric class of a texture's texels (Metal's `T` in `texture2d<T>`), as the operations
/// that use the texture spell it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TexelClass {
    Float,
    Sint,
    Uint,
    /// An integer texel whose operation spells no sign token.
    Int,
    /// A 64-bit unsigned texel, Metal's `ulong`, which only `Rg32Uint` backs.
    Ulong,
}

/// One use of a texture by an `air.*` operation: the image shape and texel class it spells.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureUse {
    pub dim: ImageDimKey,
    pub array: bool,
    pub ms: bool,
    pub depth: bool,
    pub texel: Option<TexelClass>,
}

/// [`ImageDim`] with an order, so uses can be collected in sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ImageDimKey {
    D1,
    D2,
    D3,
    Cube,
    Buffer,
}

impl TextureUse {
    fn new(shape: ImageShape, texel: Option<TexelClass>) -> TextureUse {
        TextureUse {
            dim: match shape.dim {
                ImageDim::D1 => ImageDimKey::D1,
                ImageDim::D2 => ImageDimKey::D2,
                ImageDim::D3 => ImageDimKey::D3,
                ImageDim::Cube => ImageDimKey::Cube,
                ImageDim::Buffer => ImageDimKey::Buffer,
            },
            array: shape.array,
            ms: shape.ms,
            depth: shape.depth,
            texel,
        }
    }

    /// The authored texture type this use's shape is, `None` for a shape the schema has no
    /// type for.
    pub fn texture_type(self) -> Option<TextureType> {
        Some(match (self.dim, self.array, self.ms) {
            (ImageDimKey::Buffer, false, false) => TextureType::Buffer,
            (ImageDimKey::D1, false, false) => TextureType::D1,
            (ImageDimKey::D1, true, false) => TextureType::D1Array,
            (ImageDimKey::D2, false, false) => TextureType::D2,
            (ImageDimKey::D2, true, false) => TextureType::D2Array,
            (ImageDimKey::D2, false, true) => TextureType::D2Multisample,
            (ImageDimKey::D2, true, true) => TextureType::D2MultisampleArray,
            (ImageDimKey::D3, false, false) => TextureType::D3,
            (ImageDimKey::Cube, false, false) => TextureType::Cube,
            (ImageDimKey::Cube, true, false) => TextureType::CubeArray,
            _ => return None,
        })
    }

    /// Whether an authored format can back this use: its component class is the texel class
    /// the operation spells (a use that spells none, a query, accepts any). Metal binds a `ulong`
    /// texture only to `Rg32Uint`, and a case binds `Rg32Uint` only to a `ulong` texture
    /// ([`TextureFormat::is_ulong`]).
    pub fn accepts_format(self, format: TextureFormat) -> bool {
        use crate::air::reflect::TextureComponent;
        let Some(texel) = self.texel else {
            return true;
        };
        if texel == TexelClass::Ulong || format.is_ulong() {
            return texel == TexelClass::Ulong && format.is_ulong();
        }
        let class = match format.component_class() {
            TextureComponent::Float => TexelClass::Float,
            TextureComponent::Sint => TexelClass::Sint,
            TextureComponent::Uint => TexelClass::Uint,
        };
        class == texel || (texel == TexelClass::Int && class != TexelClass::Float)
    }
}

/// Where a handle or a written pointer comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Root {
    /// The entry parameter itself.
    Param(u32),
    /// A value loaded from the parameter's memory at this byte offset (`None`: not a constant):
    /// an argument-buffer member or a table element.
    Loaded { param: u32, offset: Option<u64> },
}

/// A pointer: its root and its constant byte offset from it (`None`: not a constant).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Origin {
    root: Root,
    offset: Option<u64>,
}

/// One entry parameter with its integers resolved for the case's constants.
#[derive(Clone, Debug)]
pub struct Param {
    /// The binding; every [`AbiInt`] of its role is [`AbiInt::Const`].
    pub binding: ParamBinding,
    /// Present in the pipeline: its predicate, if any, holds for the case's constants. `false`
    /// also when the predicate does not evaluate: the argument is then optional.
    pub live: bool,
    /// The LLVM parameter is a thread-space pointer: the argument is a table (`array_ref`) of
    /// buffers, textures, or samplers, one per slot.
    pub table: bool,
    /// The IR shape of a value parameter (an attribute or a builtin).
    pub value: Option<NumShape>,
}

impl Param {
    /// The argument-table slot range `(first, count)` of a slot-bound role.
    pub fn slots(&self) -> Option<(u32, u32)> {
        let slot = role_slot(&self.binding.role)?;
        Some((constant_int(slot.index)?, constant_int(slot.count)?))
    }

    /// Whether Metal requires the pipeline to bind this argument: it is live and AIR does not
    /// mark it `air.arg_unused`.
    pub fn required(&self) -> bool {
        self.live && !self.binding.unused
    }
}

/// One stage output with its integers resolved.
#[derive(Clone, Debug)]
pub struct Output {
    pub binding: OutputBinding,
    /// Present in the pipeline.
    pub live: bool,
    /// The IR shape of the value the entry returns for it.
    pub value: Option<NumShape>,
}

/// One argument-buffer member that binds an argument: its byte offset in the buffer, its
/// `[[id]]`, and its role. A member array is one entry per element.
#[derive(Clone, Debug)]
pub struct Member {
    pub offset: u32,
    /// The argument id Metal's argument encoder takes (`[[id(n)]]`): the member's own
    /// `air.location_index` plus the ids of the `air.indirect_argument` structs enclosing it (they
    /// add down the nesting path), plus the element of an array. `None` when the location does
    /// not evaluate for the case's constants. It is not the member node's leading integer,
    /// [`IndirectArg::ordinal`](metal2vulkan_air::abi::IndirectArg::ordinal).
    pub id: Option<u32>,
    pub role: ParamRole,
    /// The element of its member this entry is, and how many the member binds (1 for a member
    /// that is no array).
    pub element: u32,
    pub elements: u32,
}

/// The Metal-ABI facts of one entry; see the module documentation.
#[derive(Clone, Debug)]
pub struct AirInterface {
    pub stage: Stage,
    pub params: Vec<Param>,
    pub outputs: Vec<Output>,
    /// The layouts the parameters' [`StructRef`]s index, with their integers resolved.
    pub structs: Vec<StructInfo>,
    /// Each declared function constant's IR shape, by index.
    pub function_constants: BTreeMap<u32, Option<NumShape>>,
    /// The metadata values the AIR layer does not evaluate for the case's constants, each with
    /// the check it leaves out. Empty when every check is complete.
    pub unresolved: Vec<UnresolvedValue>,
    /// What each texture operation used, by the handle it reached.
    texture_uses: BTreeMap<Root, BTreeSet<TextureUse>>,
    /// Every handle a texture atomic (`air.atomic_*_explicit_texture_*`) reached.
    atomic_textures: BTreeSet<Root>,
    /// Every root a store, an atomic, or a call may write through.
    written: BTreeSet<Root>,
    /// Every root an atomic operation (an `air.atomic.*` call, `atomicrmw`, or `cmpxchg`)
    /// accesses through.
    atomics: BTreeSet<Root>,
    /// Every root a `load` reads through.
    read: BTreeSet<Root>,
    /// Each `alloca` of every function the scan reached: whether its allocated type holds an
    /// array (a thread-local object a dynamic index can reach past).
    pub thread_objects: Vec<ThreadObject>,
    /// How the entry reads its `[[vertex_id]]` argument ([`VertexIdUse`]).
    pub vertex_id: VertexIdUse,
    /// How far a kernel entry's every thread accesses each buffer at the element its grid
    /// position names ([`GridReach`](super::grid_reach::GridReach)); empty for other stages.
    pub grid_reaches: Vec<super::grid_reach::GridReach>,
    /// The simdgroup float sums and products the entry and the functions it calls in its module
    /// call ([`float_reductions`]).
    pub float_reductions: BTreeSet<&'static str>,
}

/// How an entry reads its `[[vertex_id]]` argument. An indexed draw's index read past its buffer is
/// the oracle's shadow, a vertex id far past every buffer, so a case that declares an index buffer
/// target must keep that id out of its accesses: every read of it clamps it to a constant first.
/// Which access the clamped value reaches, and whether the constant lies inside it, is not part of
/// this fact (`out[min(vid, 1u << 30)]` is `Clamped`): the case's author keeps the bound inside the
/// buffer it indexes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexIdUse {
    /// The entry declares no vertex id, or never reads it.
    Unused,
    /// Every read is an operand of an unsigned minimum (`air.min.u.*`, `llvm.umin.*`) whose other
    /// operand is an integer constant: every value the entry derives from it is at most that
    /// constant.
    Clamped,
    /// Some read is anything else: a cast, an arithmetic or comparison operand, an index, an
    /// argument of another call, or a minimum with a dynamic bound.
    Unclamped,
}

/// One thread-local allocation (`alloca`) of the entry or a function it reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThreadObject {
    /// Its allocated type is or contains an array, at any nesting.
    pub holds_array: bool,
}

/// The function-constant values `values` supplies by index, as the AIR layer binds them (the
/// `constants` of [`AirInterface::new`]); a constant `values` does not supply stays undefined.
pub fn bound_constants(values: &FunctionConstantValues) -> BTreeMap<u32, ConstValue> {
    values
        .by_index
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
        .collect()
}

impl AirInterface {
    /// The facts of `entry` in `module` for the case's function `constants`.
    ///
    /// The AIR layer runs under the deterministic work and ledger caps, with no deadline: a
    /// checker verdict must not depend on host load.
    pub fn new(
        module: &Module,
        abi: &AirAbi,
        entry: &str,
        constants: &BTreeMap<u32, ConstValue>,
    ) -> Result<AirInterface, String> {
        Self::metered(
            module,
            abi,
            entry,
            constants,
            &mut Meter::new(&Budget::default()),
        )
    }

    /// The parameters and outputs of `entry` in `module` with their integers resolved for
    /// `constants`, and whether each is live: the bindings alone, without the scan of the entry's
    /// uses that [`AirInterface::new`] adds.
    pub fn bindings_for(
        module: &Module,
        abi: &AirAbi,
        entry: &str,
        constants: &BTreeMap<u32, ConstValue>,
    ) -> Result<(Vec<Param>, Vec<Output>), String> {
        let ep = entry_point(module, abi, entry)?;
        let mut meter = Meter::new(&Budget::default());
        let values = Values::evaluate(module, abi, ep, constants, &mut meter)?;
        let Resolved {
            params, outputs, ..
        } = values.resolve(module, abi, ep)?;
        Ok((params, outputs))
    }

    fn metered(
        module: &Module,
        abi: &AirAbi,
        entry: &str,
        constants: &BTreeMap<u32, ConstValue>,
        meter: &mut Meter,
    ) -> Result<AirInterface, String> {
        let ep = entry_point(module, abi, entry)?;
        let values = Values::evaluate(module, abi, ep, constants, meter)?;
        let Resolved {
            params,
            outputs,
            structs,
            unresolved,
        } = values.resolve(module, abi, ep)?;
        let mut function_constants = BTreeMap::new();
        for decl in &abi.function_constants {
            function_constants.insert(decl.index, NumShape::of(module, decl.ty));
        }
        let grid_reaches = match ep.stage {
            Stage::Kernel { .. } => super::grid_reach::scan(module, ep.func, &params),
            _ => Vec::new(),
        };
        let mut interface = AirInterface {
            stage: ep.stage,
            params,
            outputs,
            structs,
            function_constants,
            unresolved,
            texture_uses: BTreeMap::new(),
            atomic_textures: BTreeSet::new(),
            written: BTreeSet::new(),
            atomics: BTreeSet::new(),
            read: BTreeSet::new(),
            thread_objects: Vec::new(),
            vertex_id: VertexIdUse::Unused,
            grid_reaches,
            float_reductions: float_reductions(module, [ep.func]),
        };
        interface.vertex_id = vertex_id_use(module, ep.func, &interface.params);
        Uses::scan(module, ep.func)?.record(&mut interface);
        Ok(interface)
    }

    /// Every parameter whose role `select` picks, with the role.
    pub fn params_of<'a, R: 'a>(
        &'a self,
        select: impl Fn(&'a ParamRole) -> Option<R> + 'a,
    ) -> impl Iterator<Item = (&'a Param, R)> + 'a {
        self.params
            .iter()
            .filter_map(move |param| select(&param.binding.role).map(|role| (param, role)))
    }

    /// The flattened members of the argument buffer at parameter `param`: every member that binds
    /// an argument, at its byte offset in the buffer.
    pub fn members(&self, param: &Param) -> Vec<Member> {
        let ParamRole::ArgumentBuffer(buffer) = &param.binding.role else {
            return Vec::new();
        };
        let mut out = Vec::new();
        if let Some(layout) = buffer.layout {
            self.flatten(layout, 0, 0, 0, &mut out);
        }
        out
    }

    /// The argument buffers whose layouts bind, each with its slot, in declaration order.
    /// Several declarations may share a slot (function-constant alternatives). Where a live one
    /// occupies the slot, only live ones bind there: a gated-off alternative's layout is not
    /// what the pipeline's encoder writes, so its members are neither declared nor encodable.
    /// Where none is live, every declaration binds (the manifest may author either).
    pub fn bound_argument_buffers(&self) -> Vec<(u32, &Param)> {
        let at = |param: &Param| match param.binding.role {
            ParamRole::ArgumentBuffer(_) => param.slots().map(|(slot, _)| slot),
            _ => None,
        };
        let live = self
            .params
            .iter()
            .filter(|param| param.live)
            .filter_map(at)
            .collect::<BTreeSet<_>>();
        self.params
            .iter()
            .filter_map(|param| at(param).map(|slot| (slot, param)))
            .filter(|(slot, param)| param.live || !live.contains(slot))
            .collect()
    }

    /// The argument-buffer member at byte `offset` of the argument buffer bound at `slot`, with
    /// its parameter, among [`Self::bound_argument_buffers`] (the first declaration that has
    /// one).
    pub fn member_at(&self, slot: u32, offset: u32) -> Option<(&Param, Member)> {
        self.bound_argument_buffers()
            .into_iter()
            .filter(|(at, _)| *at == slot)
            .find_map(|(_, param)| {
                self.members(param)
                    .into_iter()
                    .find(|member| member.offset == offset)
                    .map(|member| (param, member))
            })
    }

    fn flatten(
        &self,
        layout: StructRef,
        base: u32,
        base_id: u32,
        depth: usize,
        out: &mut Vec<Member>,
    ) {
        if depth == LAYOUT_NESTING {
            return;
        }
        let Some(info) = self.structs.get(layout.0 as usize) else {
            return;
        };
        for member in &info.members {
            let offset = base.saturating_add(member.offset);
            match &member.kind {
                MemberKind::Indirect(arg) => {
                    let slot = role_slot(&arg.role);
                    let location = slot.and_then(|slot| constant_int(slot.index));
                    let (elements, stride) = member_elements(member, slot);
                    for element in 0..elements {
                        out.push(Member {
                            offset: offset.saturating_add(element.saturating_mul(stride)),
                            id: location.map(|location| {
                                base_id.saturating_add(location).saturating_add(element)
                            }),
                            role: arg.role.clone(),
                            element,
                            elements,
                        });
                    }
                }
                // A struct array's later elements repeat the layout at ids the metadata does not
                // state; only its first element is flattened.
                MemberKind::IndirectStruct { first_id } => {
                    if let Some(inner) = member.layout {
                        let base_id = base_id.saturating_add(*first_id);
                        self.flatten(inner, offset, base_id, depth + 1, out);
                    }
                }
                MemberKind::Plain => {
                    if let Some(inner) = member.layout {
                        self.flatten(inner, offset, base_id, depth + 1, out);
                    }
                }
            }
        }
    }

    /// Every shape the operations reaching `root` spell.
    pub fn texture_uses(&self, root: Root) -> impl Iterator<Item = TextureUse> + '_ {
        self.texture_uses
            .get(&root)
            .into_iter()
            .flat_map(|uses| uses.iter().copied())
    }

    /// Every shape the operations reaching the handle loaded from parameter `param`'s memory at
    /// byte `offset` spell: an element of a texture table, or an argument-buffer texture member.
    /// A use at a dynamic offset reaches every offset, since its origin keeps only the parameter.
    pub fn loaded_uses(&self, param: u32, offset: u64) -> impl Iterator<Item = TextureUse> + '_ {
        self.texture_uses
            .iter()
            .filter(move |(root, _)| {
                matches!(root, Root::Loaded { param: p, offset: o }
                    if *p == param && o.is_none_or(|o| o == offset))
            })
            .flat_map(|(_, uses)| uses.iter().copied())
    }

    /// Whether the shader may write through `root`; a dynamically offset write through the same
    /// parameter's memory reaches every member.
    pub fn writes(&self, root: Root) -> bool {
        reaches(&self.written, root)
    }

    /// Whether the shader may load or store through `root`, with [`Self::writes`]'s reach.
    pub fn accesses(&self, root: Root) -> bool {
        reaches(&self.read, root) || reaches(&self.written, root)
    }

    /// Whether a texture atomic reaches the texture handle `root`: Metal needs
    /// `MTLTextureUsageShaderAtomic` on the texture there.
    pub fn texture_atomic(&self, root: Root) -> bool {
        self.atomic_textures.contains(&root)
    }

    /// Whether an atomic operation accesses parameter `param`'s memory directly (not through a
    /// pointer it holds).
    pub fn atomic_through(&self, param: u32) -> bool {
        self.atomics.contains(&Root::Param(param))
    }

    /// Whether the shader writes anything through parameter `param`'s memory or the pointers
    /// it holds.
    pub fn writes_through(&self, param: u32) -> bool {
        self.written.iter().any(|root| match root {
            Root::Param(p) | Root::Loaded { param: p, .. } => *p == param,
        })
    }
}

/// Whether `roots` holds `root`; a dynamically offset root of the same parameter's memory reaches
/// every member.
fn reaches(roots: &BTreeSet<Root>, root: Root) -> bool {
    match root {
        Root::Param(_) => roots.contains(&root),
        Root::Loaded { param, offset } => roots.iter().any(|held| {
            matches!(held, Root::Loaded { param: p, offset: o }
                if *p == param && (o.is_none() || offset.is_none() || *o == offset))
        }),
    }
}

/// The constant byte offset a GEP over `src_ty` with `indices` adds in `module` (its type layouts
/// `layouts`), `None` when an index is not a constant.
pub(super) fn gep_offset(
    module: &Module,
    layouts: &TypeLayouts,
    src_ty: TypeId,
    indices: &[Value],
) -> Option<u64> {
    let index = |value: Value| -> Option<i64> {
        let Value::Const(id) = value else {
            return None;
        };
        let constant = module.consts().get(id)?;
        let bits = match module.types().get(constant.ty)? {
            Type::Int(bits) => u32::from(*bits),
            _ => return None,
        };
        match constant.value {
            Const::Int(raw) if bits < 64 => {
                let shift = 64 - bits;
                Some(((raw << shift) as i64) >> shift)
            }
            Const::Int(raw) => Some(raw as i64),
            Const::Null => Some(0),
            _ => None,
        }
    };
    let mut offset: i64 = 0;
    let mut ty = src_ty;
    for (position, value) in indices.iter().enumerate() {
        let step = index(*value)?;
        if position == 0 {
            let size = layouts.alloc_size(ty)? as i64;
            offset = offset.checked_add(step.checked_mul(size)?)?;
            continue;
        }
        match module.types().get(ty)? {
            Type::Struct { fields, .. } => {
                let field = usize::try_from(step).ok()?;
                offset = offset.checked_add(*layouts.field_offsets(ty)?.get(field)? as i64)?;
                ty = *fields.get(field)?;
            }
            Type::Array { elem, .. } | Type::Vector { elem, .. } => {
                let size = layouts.alloc_size(*elem)? as i64;
                offset = offset.checked_add(step.checked_mul(size)?)?;
                ty = *elem;
            }
            _ => return None,
        }
    }
    u64::try_from(offset).ok()
}

/// Whether type `ty` of `module` is or contains an array, through struct fields (bounded by the
/// layout nesting).
fn holds_array(module: &Module, ty: TypeId, depth: usize) -> bool {
    if depth == LAYOUT_NESTING {
        return false;
    }
    match module.types().get(ty) {
        Some(Type::Array { .. }) => true,
        Some(Type::Struct { fields, .. }) => fields
            .iter()
            .any(|field| holds_array(module, *field, depth + 1)),
        _ => false,
    }
}

/// The value of a resolved ABI integer.
pub fn constant_int(value: AbiInt) -> Option<u32> {
    match value {
        AbiInt::Const(value) => Some(value),
        AbiInt::Global(_) => None,
    }
}

/// Whether an argument declared with `access` may be written.
pub fn declared_writable(access: Access) -> bool {
    matches!(access, Access::Write | Access::ReadWrite)
}

/// The Metal usage a texture argument needs.
///
/// A case's `role` says which bytes are authored and which resource is read back. It does not
/// say how the *shader* touches the texture, and Metal refuses a shader that reads a texture
/// whose usage omits `ShaderRead`: `reads texture (splatX[0]) whose usage (0x02) doesn't
/// specify MTLTextureUsageShaderRead (0x01)`. That access is an AIR fact, so take it from the
/// AIR declaration and union it with the role rather than deriving one from the other. The
/// access arrives here only as `writable`, which does not separate `write` from `read_write`;
/// granting a write-only texture `ShaderRead` as well costs nothing, while withholding it from
/// a `read_write` one is undefined behaviour.
pub fn texture_shader_usage(role: ResourceRole, writable: Option<bool>) -> TextureUsage {
    let by_role = match role {
        ResourceRole::Input => TextureUsage::SHADER_READ,
        ResourceRole::Output => TextureUsage::SHADER_WRITE,
        ResourceRole::InOut => TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
    };
    match writable {
        Some(true) => by_role | TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        Some(false) => by_role | TextureUsage::SHADER_READ,
        None => by_role,
    }
}

/// Whether AIR declares the texture argument at slot `slot` with a `write` or `read_write`
/// access qualifier. `None` when the entry declares no texture there at all. Of several
/// declarations of the slot (function-constant alternatives) a live one answers first.
pub fn declared_texture_writable(interface: &AirInterface, slot: u32) -> Option<bool> {
    let mut textures = interface
        .params_of(|role| match role {
            ParamRole::Texture(texture) => Some(texture.access),
            _ => None,
        })
        .filter(|(param, _)| param.slots().map(|(first, _)| first) == Some(slot))
        .collect::<Vec<_>>();
    textures.sort_by_key(|(param, _)| !param.live);
    textures
        .first()
        .map(|(_, access)| declared_writable(*access))
}

/// Whether a texture atomic reaches the texture argument at slot `slot` (any declaration of the
/// slot).
pub fn declared_texture_atomic(interface: &AirInterface, slot: u32) -> bool {
    interface
        .params_of(|role| match role {
            ParamRole::Texture(_) => Some(()),
            _ => None,
        })
        .filter(|(param, _)| param.slots().map(|(first, _)| first) == Some(slot))
        .any(|(param, _)| interface.texture_atomic(Root::Param(param.binding.param)))
}

/// A metadata value of the entry that the AIR layer does not evaluate for the case's constants:
/// a static initializer computes it through an operation outside the evaluated subset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unresolved {
    /// A parameter's predicate. The argument is optional: neither required nor refused.
    Predicate { param: u32 },
    /// A parameter's slot count. The range is unsized and its length unchecked.
    Count { param: u32 },
    /// The slot of a parameter whose predicate does not evaluate either. It binds nothing the
    /// manifest can match.
    Slot { param: u32 },
    /// An output's predicate. The output is not required.
    OutputPredicate { position: usize },
    /// The attachment of an output whose predicate does not evaluate either. It binds nothing
    /// the manifest can match.
    Attachment { position: usize },
    /// The slot of an argument-buffer member (layout `layout`, member `member`) that a live or
    /// optional argument buffer reaches. The member has no argument id.
    Member { layout: usize, member: usize },
    /// The slot count of such a member. Only its first element has an argument id.
    MemberCount { layout: usize, member: usize },
}

/// An [`Unresolved`] value with the AIR layer's reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnresolvedValue {
    pub value: Unresolved,
    pub reason: String,
}

impl std::fmt::Display for UnresolvedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (value, leaves) = match self.value {
            Unresolved::Predicate { param } => (
                format!("the predicate of entry parameter {param}"),
                "the argument is optional",
            ),
            Unresolved::Count { param } => (
                format!("the slot count of entry parameter {param}"),
                "its length is unchecked",
            ),
            Unresolved::Slot { param } => (
                format!("the slot of entry parameter {param}, whose predicate is unknown too"),
                "it binds nothing",
            ),
            Unresolved::OutputPredicate { position } => (
                format!("the predicate of output {position}"),
                "the output is not required",
            ),
            Unresolved::Attachment { position } => (
                format!("the attachment of output {position}, whose predicate is unknown too"),
                "it binds nothing",
            ),
            Unresolved::Member { layout, member } => (
                format!("the slot of member {member} of argument-buffer layout {layout}"),
                "it has no argument id",
            ),
            Unresolved::MemberCount { layout, member } => (
                format!("the slot count of member {member} of argument-buffer layout {layout}"),
                "only its first element has an argument id",
            ),
        };
        write!(f, "{value} does not evaluate ({}); {leaves}", self.reason)
    }
}

/// Why an ABI integer has no value.
#[derive(Clone, Debug)]
enum Missing {
    /// The AIR layer does not evaluate the static initializer that computes it: unknown, and
    /// recorded as [`Unresolved`] where it is used.
    NotEvaluated(String),
    /// Anything else (a constant the case omits, a value that does not fit): the case's error.
    Invalid(String),
}

impl Missing {
    fn of(error: metal2vulkan::error::Error) -> Missing {
        match error {
            metal2vulkan::error::Error::Unsupported(u)
                if u.feature == metal2vulkan::error::Feature::StaticInitializer =>
            {
                Missing::NotEvaluated(metal2vulkan::error::Error::Unsupported(u).to_string())
            }
            other => Missing::Invalid(other.to_string()),
        }
    }
}

/// The integers the static initializer leaves in the ABI's globals.
struct Values {
    ints: HashMap<GlobalId, Result<u64, Missing>>,
}

impl Values {
    fn evaluate(
        module: &Module,
        abi: &AirAbi,
        ep: &EntryPoint,
        constants: &BTreeMap<u32, ConstValue>,
        meter: &mut Meter,
    ) -> Result<Values, String> {
        let mut globals = BTreeSet::new();
        let roles = ep.params.iter().map(|param| &param.role).chain(
            abi.structs
                .iter()
                .flat_map(|info| info.members.iter())
                .filter_map(|member| match &member.kind {
                    MemberKind::Indirect(arg) => Some(&arg.role),
                    MemberKind::Plain | MemberKind::IndirectStruct { .. } => None,
                }),
        );
        let attachments = ep.outputs.iter().filter_map(|output| match &output.role {
            OutputRole::RenderTarget { attachment, .. } => Some(*attachment),
            _ => None,
        });
        for value in roles.flat_map(role_ints).chain(attachments) {
            if let AbiInt::Global(global) = value {
                globals.insert(global);
            }
        }
        let predicates = ep
            .params
            .iter()
            .filter_map(|param| param.predicate.as_ref())
            .chain(
                ep.outputs
                    .iter()
                    .filter_map(|output| output.predicate.as_ref()),
            )
            .map(|predicate| predicate.global);
        globals.extend(predicates);
        // Metal ignores a value for an index the function declares no constant at, so only the
        // declared ones bind.
        let constants = constants
            .iter()
            .filter(|(index, _)| {
                abi.function_constants
                    .iter()
                    .any(|decl| decl.index == **index)
            })
            .map(|(index, value)| (*index, value.clone()))
            .collect::<BTreeMap<_, _>>();
        if globals.is_empty() && constants.is_empty() {
            return Ok(Values {
                ints: HashMap::new(),
            });
        }
        let globals = globals.into_iter().collect::<Vec<_>>();
        // The evaluation consumes a copy of the module, which the checker makes outside the AIR
        // layer's ledger.
        let results = match canon::static_values(module.clone(), abi, &constants, &globals, meter)
            .map_err(metal2vulkan::error::Error::from)
        {
            Ok(results) => results
                .into_iter()
                .map(|r| r.map_err(|e| Missing::of(e.into())))
                .collect(),
            // A value of the wrong type, or for no declared constant, is the case's error.
            Err(error @ metal2vulkan::error::Error::InvalidDescriptor(_)) => {
                return Err(format!(
                    "function constants do not bind to the AIR: {error}"
                ));
            }
            // The initializers do not evaluate as a whole (a branch on, or an address from,
            // what the AIR layer does not evaluate): every global they compute is unknown.
            Err(error) => match Missing::of(error) {
                missing @ Missing::NotEvaluated(_) => vec![Err(missing); globals.len()],
                // A limit, the deadline, cancellation, or an internal failure is no fact
                // about the case; it must not make the verdict depend on the host.
                Missing::Invalid(error) => {
                    return Err(format!(
                        "the AIR layer does not evaluate the metadata globals: {error}"
                    ))
                }
            },
        };
        Ok(Values {
            ints: globals.into_iter().zip(results).collect(),
        })
    }

    /// The value of `value`.
    fn int(&self, value: AbiInt) -> Result<u32, Missing> {
        match value {
            AbiInt::Const(value) => Ok(value),
            AbiInt::Global(global) => {
                let resolved = self
                    .ints
                    .get(&global)
                    .cloned()
                    .unwrap_or_else(|| Err(Missing::Invalid("not evaluated".into())))?;
                u32::try_from(resolved)
                    .map_err(|_| Missing::Invalid(format!("{resolved} does not fit 32 bits")))
            }
        }
    }

    /// Whether a predicate holds: `Err(reason)` when the AIR layer does not evaluate it, and a
    /// checker error when it has no value for any other reason.
    fn present(
        &self,
        predicate: &Option<metal2vulkan_air::abi::Predicate>,
        what: impl Fn() -> String,
    ) -> Result<Result<bool, String>, String> {
        let Some(predicate) = predicate else {
            return Ok(Ok(true));
        };
        match self.int(AbiInt::Global(predicate.global)) {
            Ok(value) => Ok(Ok(value != 0)),
            Err(Missing::NotEvaluated(reason)) => Ok(Err(reason)),
            Err(Missing::Invalid(error)) => Err(format!(
                "the predicate of {} does not evaluate: {error}",
                what()
            )),
        }
    }

    /// Resolve every integer of `role`. A count the AIR layer does not evaluate becomes 0, a
    /// length the checker does not check (its reason is returned); an index that does not
    /// evaluate, or a count that is the case's error, is `Err`.
    fn role(&self, role: &mut ParamRole) -> Result<Option<String>, Missing> {
        let mut unevaluated_count = None;
        for slot in role_slots_mut(role) {
            slot.index = AbiInt::Const(self.int(slot.index)?);
            slot.count = AbiInt::Const(match self.int(slot.count) {
                Ok(count) => count,
                Err(Missing::NotEvaluated(reason)) => {
                    unevaluated_count = Some(reason);
                    0
                }
                Err(invalid) => return Err(invalid),
            });
        }
        if let ParamRole::ColorInput { index, .. } = role {
            *index = AbiInt::Const(self.int(*index)?);
        }
        Ok(unevaluated_count)
    }

    fn resolve(&self, module: &Module, abi: &AirAbi, ep: &EntryPoint) -> Result<Resolved, String> {
        let function = module
            .func(ep.func)
            .ok_or_else(|| "the entry is not a function of the module".to_string())?;
        let param_types = match module.types().get(function.ty) {
            Some(Type::Function { params, .. }) => params.clone(),
            _ => return Err("the entry has no function type".into()),
        };
        let ret = match module.types().get(function.ty) {
            Some(Type::Function { ret, .. }) => *ret,
            _ => return Err("the entry has no function type".into()),
        };
        let mut unresolved = Vec::new();
        let mut params = Vec::with_capacity(ep.params.len());
        let mut optional = BTreeSet::new();
        for binding in &ep.params {
            let param = binding.param;
            // An argument whose predicate the AIR layer does not evaluate is optional: not
            // required, but still a declaration a manifest may bind.
            let what = || format!("entry parameter {param}");
            let present = match self.present(&binding.predicate, what)? {
                Ok(present) => Some(present),
                Err(reason) => {
                    unresolved.push(UnresolvedValue {
                        value: Unresolved::Predicate { param },
                        reason,
                    });
                    None
                }
            };
            let mut binding = binding.clone();
            match (self.role(&mut binding.role), present) {
                (Ok(None), _) => {}
                (Ok(Some(reason)), _) => unresolved.push(UnresolvedValue {
                    value: Unresolved::Count { param },
                    reason,
                }),
                // Metal binds nothing for an argument its predicate removes, so a slot computed
                // from a constant the case omits is no error there.
                (Err(_), Some(false)) => continue,
                // An optional argument whose slot the AIR layer does not evaluate binds nothing
                // the manifest can match.
                (Err(Missing::NotEvaluated(reason)), None) => {
                    unresolved.push(UnresolvedValue {
                        value: Unresolved::Slot { param },
                        reason,
                    });
                    continue;
                }
                (Err(Missing::NotEvaluated(error) | Missing::Invalid(error)), _) => {
                    return Err(format!(
                        "the slot of entry parameter {param} does not evaluate: {error}"
                    ))
                }
            }
            let live = present == Some(true);
            if present.is_none() {
                optional.insert(param);
            }
            binding.predicate = None;
            let ty = param_types.get(param as usize).copied();
            let table = ty.is_some_and(|ty| {
                matches!(
                    module.types().get(ty),
                    Some(Type::Ptr {
                        space: AddrSpace::Thread
                    })
                )
            });
            params.push(Param {
                value: ty.and_then(|ty| NumShape::of(module, ty)),
                binding,
                live,
                table,
            });
        }
        // Each program-scope binding (`!air.global_bindings`) binds as a parameter of its role
        // does: Metal fills its global from the stage's argument table at the node's slot. No
        // LLVM parameter carries it, so it is numbered after the entry's own and what the code
        // does through it is not traced; every binding of the module is required.
        for (k, global) in abi.global_bindings.iter().enumerate() {
            let param = u32::try_from(param_types.len() + k)
                .map_err(|_| "too many program-scope bindings".to_string())?;
            let mut binding = ParamBinding {
                param,
                role: global.role.clone(),
                unused: false,
                predicate: None,
                ty: global.ty.clone(),
                name: global.name.clone(),
            };
            if let Err(Missing::NotEvaluated(error) | Missing::Invalid(error)) =
                self.role(&mut binding.role)
            {
                return Err(format!(
                    "the slot of program-scope binding {k} does not evaluate: {error}"
                ));
            }
            params.push(Param {
                value: None,
                binding,
                live: true,
                table: false,
            });
        }
        let returned = match module.types().get(ret) {
            Some(Type::Struct { fields, .. }) => fields.to_vec(),
            Some(Type::Void) => Vec::new(),
            _ => vec![ret],
        };
        let mut outputs = Vec::with_capacity(ep.outputs.len());
        for (position, binding) in ep.outputs.iter().enumerate() {
            let present = match self.present(&binding.predicate, || format!("output {position}"))? {
                Ok(present) => Some(present),
                Err(reason) => {
                    unresolved.push(UnresolvedValue {
                        value: Unresolved::OutputPredicate { position },
                        reason,
                    });
                    None
                }
            };
            let mut binding = binding.clone();
            if let OutputRole::RenderTarget { attachment, .. } = &mut binding.role {
                match (self.int(*attachment), present) {
                    (Ok(value), _) => *attachment = AbiInt::Const(value),
                    (Err(_), Some(false)) => continue,
                    (Err(Missing::NotEvaluated(reason)), None) => {
                        unresolved.push(UnresolvedValue {
                            value: Unresolved::Attachment { position },
                            reason,
                        });
                        continue;
                    }
                    (Err(Missing::NotEvaluated(error) | Missing::Invalid(error)), _) => {
                        return Err(format!(
                            "the attachment of output {position} does not evaluate: {error}"
                        ))
                    }
                }
            }
            let live = present == Some(true);
            binding.predicate = None;
            outputs.push(Output {
                value: returned
                    .get(position)
                    .and_then(|ty| NumShape::of(module, *ty)),
                binding,
                live,
            });
        }
        // The layouts a live argument buffer reaches bind; those only an optional one reaches may.
        let live = reached_layouts(&abi.structs, params.iter().filter(|param| param.live));
        let may_bind = reached_layouts(
            &abi.structs,
            params
                .iter()
                .filter(|param| param.live || optional.contains(&param.binding.param)),
        );
        let mut structs = abi.structs.clone();
        for (layout, info) in structs.iter_mut().enumerate() {
            for (member, entry) in info.members.iter_mut().enumerate() {
                let MemberKind::Indirect(arg) = &mut entry.kind else {
                    continue;
                };
                // A member whose slot does not evaluate keeps its global: nothing binds it by
                // argument id.
                let (value, reason) = match self.role(&mut arg.role) {
                    Ok(None) => continue,
                    Ok(Some(reason)) => (Unresolved::MemberCount { layout, member }, reason),
                    Err(Missing::Invalid(error)) if live.contains(&layout) => {
                        return Err(format!(
                            "the slot of member {member} of argument-buffer layout {layout} does \
                             not evaluate: {error}"
                        ))
                    }
                    Err(Missing::NotEvaluated(reason) | Missing::Invalid(reason)) => {
                        (Unresolved::Member { layout, member }, reason)
                    }
                };
                if may_bind.contains(&layout) {
                    unresolved.push(UnresolvedValue { value, reason });
                }
            }
        }
        Ok(Resolved {
            params,
            outputs,
            structs,
            unresolved,
        })
    }
}

/// The layouts the argument buffers among `params` reach, nested ones included.
fn reached_layouts<'a>(
    structs: &[StructInfo],
    params: impl Iterator<Item = &'a Param>,
) -> BTreeSet<usize> {
    let mut reached = BTreeSet::new();
    let mut work = params
        .filter_map(|param| match &param.binding.role {
            ParamRole::ArgumentBuffer(buffer) => buffer.layout,
            _ => None,
        })
        .collect::<Vec<_>>();
    while let Some(layout) = work.pop() {
        let index = layout.0 as usize;
        if !reached.insert(index) {
            continue;
        }
        let Some(info) = structs.get(index) else {
            continue;
        };
        work.extend(info.members.iter().filter_map(|member| member.layout));
    }
    reached
}

/// The entry's parameters, outputs, and layouts with their integers resolved, and what did not
/// resolve.
/// The entry point `entry` of `module`, declared once.
fn entry_point<'a>(
    module: &Module,
    abi: &'a AirAbi,
    entry: &str,
) -> Result<&'a EntryPoint, String> {
    let mut found = abi.entries.iter().filter(|ep| {
        module
            .func(ep.func)
            .and_then(|f| module.symbols().get(f.name))
            == Some(entry.as_bytes())
    });
    match (found.next(), found.next()) {
        (Some(ep), None) => Ok(ep),
        (Some(_), Some(_)) => Err(format!("AIR declares entry {entry:?} twice")),
        (None, _) => Err(format!("AIR declares no entry point {entry:?}")),
    }
}

struct Resolved {
    params: Vec<Param>,
    outputs: Vec<Output>,
    structs: Vec<StructInfo>,
    unresolved: Vec<UnresolvedValue>,
}

/// Every integer a role states.
fn role_ints(role: &ParamRole) -> Vec<AbiInt> {
    let mut role = role.clone();
    let mut ints = role_slots_mut(&mut role)
        .into_iter()
        .flat_map(|slot| [slot.index, slot.count])
        .collect::<Vec<_>>();
    if let ParamRole::ColorInput { index, .. } = role {
        ints.push(index);
    }
    ints
}

/// How many elements an argument-buffer member binds, and the byte stride between them. A C array
/// (`texture2d<float> t[N]`) states its length as `array_len` and `size` per element; the
/// `metal::array<T, N>` spelling states `array_len` 0, the whole array's `size`, and `N` as its
/// location count. Anything else is one element.
fn member_elements(member: &metal2vulkan_air::abi::Member, slot: Option<Slot>) -> (u32, u32) {
    let count = slot.and_then(|slot| constant_int(slot.count));
    match (member.array_len, count) {
        (0, Some(count)) if count > 1 && member.size.is_multiple_of(count) => {
            (count, member.size / count)
        }
        (0, _) => (1, member.size),
        (len, _) => (len, member.size),
    }
}

/// The one argument-table location of a slot-bound role: an entry parameter's slot range, or an
/// argument-buffer member's `air.location_index`.
fn role_slot(role: &ParamRole) -> Option<Slot> {
    match role {
        ParamRole::Buffer(b) | ParamRole::ArgumentBuffer(b) | ParamRole::Threadgroup(b) => {
            Some(b.location)
        }
        ParamRole::Texture(t) => Some(t.location),
        ParamRole::Sampler { location }
        | ParamRole::StageIn { location }
        | ParamRole::VertexInput { location }
        | ParamRole::IndirectConstant { location } => Some(*location),
        ParamRole::AccelerationStructure { arg, .. }
        | ParamRole::FunctionTable(arg)
        | ParamRole::IntersectionTable(arg)
        | ParamRole::CommandBuffer(arg)
        | ParamRole::ComputePipelineState(arg)
        | ParamRole::RenderPipelineState(arg)
        | ParamRole::DepthStencilState(arg) => Some(arg.location),
        ParamRole::Builtin(_)
        | ParamRole::FragmentInput(_)
        | ParamRole::ColorInput { .. }
        | ParamRole::PatchInput(_)
        | ParamRole::Imageblock(_)
        | ParamRole::Payload { .. }
        | ParamRole::MeshGridProperties
        | ParamRole::Mesh(_) => None,
    }
}

fn role_slots_mut(role: &mut ParamRole) -> Vec<&mut Slot> {
    match role {
        ParamRole::Buffer(b) | ParamRole::ArgumentBuffer(b) | ParamRole::Threadgroup(b) => {
            vec![&mut b.location]
        }
        ParamRole::Texture(t) => vec![&mut t.location],
        ParamRole::Sampler { location }
        | ParamRole::StageIn { location }
        | ParamRole::VertexInput { location }
        | ParamRole::IndirectConstant { location }
        | ParamRole::PatchInput(PatchInput::Patch { location }) => vec![location],
        ParamRole::PatchInput(PatchInput::ControlPoints { fields, .. }) => {
            fields.iter_mut().map(|field| &mut field.location).collect()
        }
        ParamRole::AccelerationStructure { arg, .. }
        | ParamRole::FunctionTable(arg)
        | ParamRole::IntersectionTable(arg)
        | ParamRole::CommandBuffer(arg)
        | ParamRole::ComputePipelineState(arg)
        | ParamRole::RenderPipelineState(arg)
        | ParamRole::DepthStencilState(arg) => vec![&mut arg.location],
        ParamRole::Builtin(_)
        | ParamRole::FragmentInput(_)
        | ParamRole::ColorInput { .. }
        | ParamRole::Imageblock(_)
        | ParamRole::Payload { .. }
        | ParamRole::MeshGridProperties
        | ParamRole::Mesh(_) => Vec::new(),
    }
}

/// The origins every value of every reachable function can have, to a fixpoint.
struct Uses<'m> {
    module: &'m Module,
    layouts: TypeLayouts,
    /// Per function, per value.
    origins: HashMap<(FuncId, Value), BTreeSet<Origin>>,
}

/// Constant offsets one value keeps per root before it reaches the root at a dynamic offset.
const MAX_OFFSETS: usize = 64;

impl<'m> Uses<'m> {
    fn scan(module: &'m Module, entry: FuncId) -> Result<Uses<'m>, String> {
        let mut meter = Meter::new(&Budget::default());
        let layouts = TypeLayouts::new(module.data_layout(), module.types(), &mut meter)
            .map_err(|error| format!("AIR type layouts: {error}"))?;
        let mut uses = Uses {
            module,
            layouts,
            origins: HashMap::new(),
        };
        let arity = match module.func(entry).and_then(|f| module.types().get(f.ty)) {
            Some(Type::Function { params, .. }) => params.len(),
            _ => 0,
        };
        for param in 0..arity {
            let Some(arg) = metal2vulkan_air::ir::ArgIdx::from_index(param) else {
                continue;
            };
            uses.origins.insert(
                (entry, Value::Arg(arg)),
                BTreeSet::from([Origin {
                    root: Root::Param(param as u32),
                    offset: Some(0),
                }]),
            );
        }
        // Each function is revisited only when one of its parameters gained an origin; the
        // origin lattice has finite height, so the worklist drains.
        let mut pending = vec![entry];
        while let Some(func) = pending.pop() {
            for callee in uses.propagate(func) {
                if !pending.contains(&callee) {
                    pending.push(callee);
                }
            }
        }
        Ok(uses)
    }

    fn get(&self, func: FuncId, value: Value) -> BTreeSet<Origin> {
        self.origins
            .get(&(func, value))
            .cloned()
            .unwrap_or_default()
    }

    /// Add `add` to the origins of `value`; returns whether they grew. Per root, a dynamic offset
    /// absorbs every constant one, and more than [`MAX_OFFSETS`] constant offsets become the
    /// dynamic one, so each value's set has finite height.
    fn join(&mut self, func: FuncId, value: Value, add: BTreeSet<Origin>) -> bool {
        if add.is_empty() {
            return false;
        }
        let entry = self.origins.entry((func, value)).or_default();
        let before = entry.clone();
        entry.extend(add);
        let roots = entry
            .iter()
            .map(|origin| origin.root)
            .collect::<BTreeSet<_>>();
        for root in roots {
            let offsets = entry.iter().filter(|origin| origin.root == root).count();
            let dynamic = Origin { root, offset: None };
            if entry.contains(&dynamic) || offsets > MAX_OFFSETS {
                entry.retain(|origin| origin.root != root);
                entry.insert(dynamic);
            }
        }
        *entry != before
    }

    /// Propagate origins through `func` to a fixpoint; returns the defined callees whose
    /// parameters gained an origin.
    fn propagate(&mut self, func: FuncId) -> BTreeSet<FuncId> {
        let module = self.module;
        let Some(body) = module.func(func).and_then(|f| f.body.as_ref()) else {
            return BTreeSet::new();
        };
        let mut callees = BTreeSet::new();
        loop {
            let mut changed = false;
            for (block, _) in body.blocks.iter() {
                for (id, inst) in body.block_insts(block) {
                    let result = Value::Inst(id);
                    let add = match &inst.kind {
                        InstKind::Cast { v, .. } | InstKind::Freeze(v) => self.get(func, *v),
                        InstKind::Select { t, f, .. } => {
                            let mut both = self.get(func, *t);
                            both.extend(self.get(func, *f));
                            both
                        }
                        InstKind::Phi { incoming, .. } => incoming
                            .iter()
                            .flat_map(|(value, _)| self.get(func, *value))
                            .collect(),
                        InstKind::Gep {
                            src_ty,
                            base,
                            indices,
                            ..
                        } => {
                            let delta = self.gep_offset(*src_ty, indices);
                            self.get(func, *base)
                                .into_iter()
                                .map(|origin| Origin {
                                    root: origin.root,
                                    offset: origin
                                        .offset
                                        .zip(delta)
                                        .map(|(a, b)| a.wrapping_add(b)),
                                })
                                .collect()
                        }
                        InstKind::Load { ptr, .. }
                            if matches!(module.types().get(inst.ty), Some(Type::Ptr { .. })) =>
                        {
                            self.get(func, *ptr)
                                .into_iter()
                                .filter_map(|origin| match origin.root {
                                    Root::Param(param) => Some(Origin {
                                        root: Root::Loaded {
                                            param,
                                            offset: origin.offset,
                                        },
                                        offset: Some(0),
                                    }),
                                    Root::Loaded { .. } => None,
                                })
                                .collect()
                        }
                        InstKind::Call(call) => {
                            if let Callee::Direct(callee) = call.callee {
                                let defined = module.func(callee).is_some_and(|f| f.body.is_some());
                                if defined {
                                    for (position, arg) in call.args.iter().enumerate() {
                                        let Some(param) =
                                            metal2vulkan_air::ir::ArgIdx::from_index(position)
                                        else {
                                            continue;
                                        };
                                        let origins = self.get(func, *arg);
                                        if self.join(callee, Value::Arg(param), origins) {
                                            callees.insert(callee);
                                        }
                                    }
                                }
                            }
                            BTreeSet::new()
                        }
                        _ => BTreeSet::new(),
                    };
                    changed |= self.join(func, result, add);
                }
            }
            if !changed {
                return callees;
            }
        }
    }

    /// The constant byte offset a GEP adds, `None` when an index is not a constant.
    fn gep_offset(&self, src_ty: TypeId, indices: &[Value]) -> Option<u64> {
        gep_offset(self.module, &self.layouts, src_ty, indices)
    }

    /// Record the texture uses and written roots of every function the scan reached.
    fn record(self, interface: &mut AirInterface) {
        let module = self.module;
        let functions = self
            .origins
            .keys()
            .map(|(func, _)| *func)
            .collect::<BTreeSet<_>>();
        for func in functions {
            let Some(body) = module.func(func).and_then(|f| f.body.as_ref()) else {
                continue;
            };
            for (block, _) in body.blocks.iter() {
                for (_, inst) in body.block_insts(block) {
                    if let InstKind::AtomicRmw { ptr, .. } | InstKind::CmpXchg { ptr, .. } =
                        &inst.kind
                    {
                        let origins = self.get(func, *ptr);
                        interface
                            .atomics
                            .extend(origins.iter().map(|origin| origin.root));
                    }
                    if let InstKind::Call(call) = &inst.kind {
                        if let Callee::Air {
                            op: AirIntrinsic::Atomic(_),
                            ..
                        } = &call.callee
                        {
                            for arg in call.args.iter() {
                                let origins = self.get(func, *arg);
                                interface
                                    .atomics
                                    .extend(origins.iter().map(|origin| origin.root));
                            }
                        }
                    }
                    let mut written = |origins: BTreeSet<Origin>| {
                        interface
                            .written
                            .extend(origins.into_iter().map(|origin| origin.root));
                    };
                    match &inst.kind {
                        InstKind::Store { ptr, .. }
                        | InstKind::AtomicRmw { ptr, .. }
                        | InstKind::CmpXchg { ptr, .. } => written(self.get(func, *ptr)),
                        InstKind::Load { ptr, .. } => interface
                            .read
                            .extend(self.get(func, *ptr).into_iter().map(|origin| origin.root)),
                        InstKind::Alloca { ty, .. } => {
                            interface.thread_objects.push(ThreadObject {
                                holds_array: holds_array(module, *ty, 0),
                            })
                        }
                        InstKind::Call(call) => {
                            let defined = matches!(call.callee, Callee::Direct(callee)
                                if module.func(callee).is_some_and(|f| f.body.is_some()));
                            if let Callee::Air {
                                op: AirIntrinsic::Texture(texture),
                                ..
                            } = &call.callee
                            {
                                let texel =
                                    texture.texel.map(|texel| texel_class(texel.ty, texel.sign));
                                let usage = TextureUse::new(texture.image, texel);
                                let atomic = matches!(texture.op, TexOp::Atomic(_));
                                if let Some(position) = texture.layout().index(TexRole::Texture) {
                                    if let Some(arg) = call.args.get(position) {
                                        for origin in self.get(func, *arg) {
                                            if origin.offset == Some(0) {
                                                interface
                                                    .texture_uses
                                                    .entry(origin.root)
                                                    .or_default()
                                                    .insert(usage);
                                                if atomic {
                                                    interface.atomic_textures.insert(origin.root);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // A texture or sampler operation writes no memory a pointer
                            // argument names: texture writes are the declared access's.
                            let handle_op = matches!(
                                call.callee,
                                Callee::Air {
                                    op: AirIntrinsic::Texture(_) | AirIntrinsic::Sampler(_),
                                    ..
                                }
                            );
                            if !defined && !handle_op {
                                for arg in call.args.iter() {
                                    written(self.get(func, *arg));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

/// The simdgroup float sums and products (`air.simd_sum`, `air.simd_product`, and their prefix
/// forms, of a `half` or `float` type) that `roots` and every function they call directly in
/// `module`, transitively, call: each by the name the translator's strategy census gives the
/// operation (`simd.reduce.fadd.float`, `simd.scan.fmul.half`, ...), the names
/// [`REDUCTIONS`](crate::observation_contract::REDUCTIONS) bounds. A float sum's or
/// product's association order is the device's (`PrecisionNote::ReductionOrder`), so these are
/// the operations that note is recorded for: a comparison of a run whose modules the executor
/// cannot read (a run through the runtime) bounds each by its own operation, read from the AIR.
/// A quad's operations, which run in Metal's order, are not among them.
pub fn float_reductions(
    module: &Module,
    roots: impl IntoIterator<Item = FuncId>,
) -> BTreeSet<&'static str> {
    use metal2vulkan_air::intrinsics::{Elem, GroupOp as G, GroupScope};
    let mut found = BTreeSet::new();
    let mut pending: Vec<FuncId> = roots.into_iter().collect();
    let mut seen: BTreeSet<FuncId> = pending.iter().copied().collect();
    while let Some(func) = pending.pop() {
        let Some(body) = module.func(func).and_then(|f| f.body.as_ref()) else {
            continue;
        };
        for (block, _) in body.blocks.iter() {
            for (_, inst) in body.block_insts(block) {
                let InstKind::Call(call) = &inst.kind else {
                    continue;
                };
                let group = match call.callee {
                    Callee::Direct(callee) => {
                        if seen.insert(callee) {
                            pending.push(callee);
                        }
                        continue;
                    }
                    Callee::Air {
                        op: AirIntrinsic::Group(group),
                        ..
                    } => group,
                    _ => continue,
                };
                if group.scope != GroupScope::Simdgroup {
                    continue;
                }
                let format = match group.ty.map(|ty| ty.elem) {
                    Some(Elem::F16) => "half",
                    Some(Elem::F32) => "float",
                    _ => continue,
                };
                let operation = match group.op {
                    G::Sum => "reduce.fadd",
                    G::Product => "reduce.fmul",
                    G::PrefixInclusiveSum | G::PrefixExclusiveSum => "scan.fadd",
                    G::PrefixInclusiveProduct | G::PrefixExclusiveProduct => "scan.fmul",
                    _ => continue,
                };
                if let Some(&(name, ..)) = crate::observation_contract::REDUCTIONS
                    .iter()
                    .find(|(name, ..)| *name == format!("simd.{operation}.{format}"))
                {
                    found.insert(name);
                }
            }
        }
    }
    found
}

/// The functions of `module` whose names are `names`, in `module`'s order: the roots of
/// [`float_reductions`] in a module a case links functions from.
pub fn functions_named<'m>(
    module: &'m Module,
    names: &'m [String],
) -> impl Iterator<Item = FuncId> + 'm {
    module
        .parts()
        .funcs
        .iter()
        .filter_map(move |(id, function)| {
            let name = module.symbols().as_str(function.name)?;
            names.iter().any(|wanted| wanted == name).then_some(id)
        })
}

/// How `entry` reads the arguments of `params` whose role is `[[vertex_id]]` ([`VertexIdUse`]).
fn vertex_id_use(module: &Module, entry: FuncId, params: &[Param]) -> VertexIdUse {
    use metal2vulkan_air::abi::Builtin;
    use metal2vulkan_air::intrinsics::{LlvmIntOp, LlvmIntrinsic, Math, MathFn};
    let ids = params
        .iter()
        .filter(|param| matches!(param.binding.role, ParamRole::Builtin(Builtin::VertexId)))
        .filter_map(|param| metal2vulkan_air::ir::ArgIdx::from_index(param.binding.param as usize))
        .map(Value::Arg)
        .collect::<Vec<_>>();
    let Some(body) = module.func(entry).and_then(|f| f.body.as_ref()) else {
        return VertexIdUse::Unused;
    };
    let constant = |value: Value| {
        let Value::Const(id) = value else {
            return false;
        };
        module.consts().get(id).is_some_and(|constant| {
            matches!(module.types().get(constant.ty), Some(Type::Int(_)))
                && matches!(constant.value, Const::Int(_))
        })
    };
    let mut reads = false;
    for (block, _) in body.blocks.iter() {
        for (_, inst) in body.block_insts(block) {
            let mut uses = false;
            inst.kind
                .for_each_value(|value| uses |= ids.contains(&value));
            if !uses {
                continue;
            }
            reads = true;
            let clamp = match &inst.kind {
                InstKind::Call(call) => {
                    let unsigned_min = matches!(
                        call.callee,
                        Callee::Air {
                            op: AirIntrinsic::Math(Math {
                                f: MathFn::IMin(Sign::Unsigned),
                                ..
                            }),
                            ..
                        } | Callee::Llvm {
                            op: LlvmIntrinsic::Int {
                                op: LlvmIntOp::UMin,
                                ..
                            },
                            ..
                        }
                    );
                    unsigned_min
                        && matches!(
                            call.args.as_ref(),
                            [a, b] if (ids.contains(a) && constant(*b))
                                || (ids.contains(b) && constant(*a))
                        )
                }
                _ => false,
            };
            if !clamp {
                return VertexIdUse::Unclamped;
            }
        }
    }
    if reads {
        VertexIdUse::Clamped
    } else {
        VertexIdUse::Unused
    }
}

fn texel_class(ty: NumTy, sign: Option<Sign>) -> TexelClass {
    use metal2vulkan_air::intrinsics::Elem;
    match (ty.elem, sign) {
        (Elem::F16 | Elem::BF16 | Elem::F32, _) => TexelClass::Float,
        // The product decodes a 64-bit texel only as Metal's unsigned `ulong`.
        (Elem::I64, _) => TexelClass::Ulong,
        (_, Some(Sign::Unsigned)) => TexelClass::Uint,
        (_, Some(Sign::Signed)) => TexelClass::Sint,
        (_, None) => TexelClass::Int,
    }
}

#[cfg(test)]
impl AirInterface {
    /// An interface with these declarations and no uses or writes.
    pub(crate) fn synthetic(
        stage: Stage,
        params: Vec<Param>,
        outputs: Vec<Output>,
        structs: Vec<StructInfo>,
    ) -> AirInterface {
        AirInterface {
            stage,
            params,
            outputs,
            structs,
            function_constants: BTreeMap::new(),
            unresolved: Vec::new(),
            texture_uses: BTreeMap::new(),
            atomic_textures: BTreeSet::new(),
            written: BTreeSet::new(),
            atomics: BTreeSet::new(),
            read: BTreeSet::new(),
            thread_objects: Vec::new(),
            vertex_id: VertexIdUse::Unused,
            grid_reaches: Vec::new(),
            float_reductions: BTreeSet::new(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
