//! `RelocatableModule` serializer and `relocate`.
//!
//! [`serialize`] consumes a finalized [`TypedModule`] and writes SPIR-V 1.5 words in the logical
//! layout of the SPIR-V spec §2.4: capabilities, extensions, the `GLSL.std.450` import, the memory
//! model, the entry point, execution modes, annotations, types/constants/globals in declaration
//! order, then functions. Result ids are assigned here, densely, in that same order, so equal
//! modules serialize to equal words. An `OpCopyObject` (P9's promoted loads) is written as its
//! operand: its result takes the operand's id and the instruction is left out, which SSA dominance
//! allows (the operand dominates the copy, which dominates its uses). Every `OpUndef` of one type
//! is one module-level `OpUndef`, written after the declarations: an undefined value may be any
//! value, so one of them serves every use. A tangled value ([`Inst::is_tangled`]) that reaches a
//! lazily evaluated operand is written with its own id and an `OpCopyObject` that carries the
//! value's id (`lazy_values`), so SPIRV-Cross evaluates it where the module does, never inside a
//! `?:` arm or a `&&`/`||` operand. A value SPIRV-Cross would forward into its one reader without
//! the dependencies of its operands is written the same way (`forwards_without_dependencies`), so
//! it is evaluated where the module does, before any later write overwrites what it reads. A kept
//! routine (a texel routine's sum or a sampling routine's emulation) is declared `DontInline`.
//!
//! Every number link assigns (descriptor set and binding, `Location`/`Component`, `SpecId`) is a
//! literal operand of an `OpDecorate`. The serializer writes 0 there and records a relocation
//! `(Symbol, word offset)`; [`relocate`] writes only those literals, so the words of a module with
//! its relocation slots zeroed are the same for every link.

use crate::spv::{Decoration, Op};

use crate::device::requirements::Requirements;
use crate::pipeline::sym::{ResourceKind, Symbol};
use crate::spirv::builder::mesh_per_primitive;
use crate::spirv::error::{BResult, BuildError, ControlRule, Dangling, RelocRule};
use crate::spirv::ids::{BlockId, ConstId, FuncId, TypeId, Value};
use crate::spirv::ir::{
    BinOp, Block, ConstKind, Constant, Contraction, Decl, Dim, ExecMode, ExecModel, GlobalKind,
    GlslOp, GroupInst, GroupOperation, ImageOperands, ImageQuery, ImageUsage, Inst, Interpolation,
    IoDir, MemAccess, Merge, RayQueryInst, Sampling, SpecOp, StorageClass, Terminator, TexelMemory,
    Type, TypedModule, UnOp, Unrolling, ValueOrigin,
};
use crate::spirv::requirements::{
    spirv_extension_name, uses_physical_storage_buffer, uses_vulkan_memory_model,
};

mod non_writable;
#[cfg(test)]
mod tests;

/// The SPIR-V version every profile emits: 1.5.
pub const SPIRV_VERSION: u32 = 0x0001_0500;

// A loop's `PartialCount` (`Unrolling::PartialCount`) is SPIR-V 1.4's.
const _: () = assert!(SPIRV_VERSION >= 0x0001_0400);

/// The entry point's name in every module (modules are compiled per function).
pub const ENTRY_NAME: &str = "main";

/// A serialized module whose link-assigned literals are still symbolic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelocatableModule {
    /// SPIR-V words; every relocation slot holds 0.
    pub words: Vec<u32>,
    /// Each relocatable literal: its symbol and word offset, in word order.
    pub relocs: Vec<(Symbol, u32)>,
    /// The requirements the module was serialized with (its declared capabilities and
    /// extensions are exactly their SPIR-V half).
    pub requirements: Requirements,
}

impl RelocatableModule {
    /// The heap bytes of the words and relocations, for the byte ledger.
    pub fn footprint(&self) -> usize {
        crate::budget::vec_bytes(&self.words) + crate::budget::vec_bytes(&self.relocs)
    }
}

/// What link assigned: the number of each symbol. `pipeline::link`'s `LinkTable` implements it;
/// any map or closure from symbols to numbers does too.
pub trait SymbolTable {
    /// The number assigned to `sym`, if any.
    fn resolve(&self, sym: &Symbol) -> Option<u32>;
}

impl<F: Fn(&Symbol) -> Option<u32>> SymbolTable for F {
    fn resolve(&self, sym: &Symbol) -> Option<u32> {
        self(sym)
    }
}

impl SymbolTable for std::collections::BTreeMap<Symbol, u32> {
    fn resolve(&self, sym: &Symbol) -> Option<u32> {
        self.get(sym).copied()
    }
}

/// Write the link-assigned literals of `r`. A symbol the table does not know is a `BuildError`;
/// nothing but the relocation slots changes.
pub fn relocate(r: &RelocatableModule, link: &impl SymbolTable) -> BResult<Vec<u32>> {
    let mut words = r.words.clone();
    for (sym, at) in &r.relocs {
        let n = link
            .resolve(sym)
            .ok_or_else(|| BuildError::Reloc(RelocRule::MissingSymbol(sym.clone())))?;
        let slot = words
            .get_mut(*at as usize)
            .ok_or(BuildError::Reloc(RelocRule::Offset(*at)))?;
        *slot = n;
    }
    Ok(words)
}

/// Serialize a finalized module (consumes it; only the words are retained). Each kept routine
/// (`ModuleBuilder::kept_routine`) is declared `DontInline`, the standard SPIR-V statement that its
/// calls stay calls: a driver that honours it (NVIDIA's) compiles the routine once, and one that
/// inlines every function whatever its control (Mesa's) is unchanged. No other function takes a
/// control: NVIDIA's driver ran a kernel of many plain routines 8.1 times as long with it on the
/// RTX 3090.
pub fn serialize(m: TypedModule) -> BResult<RelocatableModule> {
    let ids = Ids::assign(&m)?;
    let mut w = Writer::default();
    // Header; the bound is patched below.
    w.words
        .extend_from_slice(&[crate::spv::MAGIC_NUMBER, SPIRV_VERSION, 0, 0, 0]);
    let mut controls: Vec<(BlockId, Unrolling)> = m
        .loops
        .iter()
        .filter(|l| l.unrolling != Unrolling::Driver)
        .map(|l| (l.header, l.unrolling))
        .collect();
    controls.sort_unstable_by_key(|&(header, _)| header);
    let s = Ser {
        m: &m,
        ids: &ids,
        controls,
    };
    s.preamble(&mut w)?;
    s.annotations(&mut w)?;
    s.declarations(&mut w)?;
    for (fi, _) in m.functions.iter().enumerate() {
        s.function(&mut w, FuncId::new(fi as u32))?;
    }
    if let Some(b) = w.words.get_mut(3) {
        *b = ids.bound;
    }
    Ok(RelocatableModule {
        words: w.words,
        relocs: w.relocs,
        requirements: m.requirements,
    })
}

/// The heap bytes of the id tables [`serialize`] holds while it writes `m`, for the byte ledger: one
/// `u32` per value, block, and function, two per type (its id and its module-level `OpUndef`'s),
/// one `bool` per value (which values SPIRV-Cross may evaluate lazily), one byte per value (how
/// often SPIRV-Cross reads it and whether it may carry an index) and one `u32` per value while
/// those are found (each copy's root), and the maps of void calls' ids and of materialized values'
/// own ids, each at up to twice its bound (void calls; tangled instructions and instructions whose
/// result SPIRV-Cross may forward without its operands' dependencies), rounded to a power of two as
/// the map grows. One `BlockId` and `Unrolling` per loop that asks something of the driver's
/// unrolling (the sorted headers the loop merges look up).
pub fn id_table_bytes(m: &TypedModule) -> usize {
    let dense = m
        .types
        .len()
        .saturating_mul(2)
        .saturating_add(m.values.len())
        .saturating_add(m.blocks.len())
        .saturating_add(m.functions.len());
    let insts = || m.blocks.iter().flat_map(|b| b.insts.iter());
    let void_calls = insts()
        .filter(|n| n.result.is_none() && matches!(n.inst, Inst::Call { .. }))
        .count();
    let materializable = insts()
        .filter(|n| {
            n.result.is_some() && (n.inst.is_tangled() || may_forward_without_dependencies(&n.inst))
        })
        .count();
    let slots = |n: usize| match n {
        0 => 0,
        n => n.saturating_mul(2).next_power_of_two(),
    };
    dense
        .saturating_mul(std::mem::size_of::<u32>())
        .saturating_add(
            m.values
                .len()
                .saturating_mul(2 + std::mem::size_of::<u32>()),
        )
        .saturating_add(
            slots(void_calls).saturating_mul(std::mem::size_of::<((u32, u32), u32)>() + 1),
        )
        .saturating_add(
            slots(materializable).saturating_mul(std::mem::size_of::<(Value, u32)>() + 1),
        )
        .saturating_add(
            m.loops
                .iter()
                .filter(|l| l.unrolling != Unrolling::Driver)
                .count()
                .saturating_mul(std::mem::size_of::<(BlockId, Unrolling)>()),
        )
}

/// The ids beyond its own that the copy of a materialized value of type `ty` writes
/// (`Ser::copy`): none for a value but an array of a known length, and for such an array two per
/// element (the element extracted and its copy) and its element's own.
fn copy_parts(m: &TypedModule, ty: TypeId) -> BResult<u32> {
    let Type::Array { elem, len, .. } = m.types().get(ty)? else {
        return Ok(0);
    };
    let Some(n) = m.consts().array_len(*len, m.types())? else {
        return Ok(0);
    };
    copy_parts(m, *elem)?
        .checked_add(2)
        .and_then(|each| each.checked_mul(n))
        .ok_or(BuildError::IdSpaceExhausted)
}

/// Which values SPIRV-Cross may evaluate lazily, by value.
///
/// SPIRV-Cross (MoltenVK's MSL conversion) writes a value it forwards as an expression into the
/// text of the instruction that reads it, and writes `OpSelect` on a scalar condition as
/// `c ? t : f` and `OpLogicalAnd`/`OpLogicalOr` as `a && b`/`a || b` (per component for
/// vectors), whose `t`/`f` and `b` run only in the invocations that take them. A tangled value
/// ([`Inst::is_tangled`]: a subgroup operation, a derivative, implicit-LOD sampling) moved there
/// is computed by a subset of the invocations that compute it in the module, which changes its
/// result. A value is lazy when it is `OpSelect`'s `t` or `f` (whatever the condition's width,
/// since SPIRV-Cross's choice of `select()` for a vector condition depends on its options),
/// `OpLogicalAnd`'s or `OpLogicalOr`'s second operand, or an operand of a lazy value's
/// instruction that is not itself tangled (an expression inlined into a lazy one is inlined
/// with its operands; a materialized tangled value's operands are evaluated at its copy). Uses
/// through phis are not followed: SPIRV-Cross assigns a phi's incoming value in its predecessor
/// as a statement. The scan is conservative across blocks (SPIRV-Cross declares a value used
/// across a structured construct, but forwards across a plain `OpBranch`). One reverse pass
/// suffices: blocks follow their dominators, so every non-phi use of a value is visited before
/// its definition.
fn lazy_values(m: &TypedModule) -> BResult<Vec<bool>> {
    fn mark(lazy: &mut [bool], v: Value) -> BResult<()> {
        *lazy
            .get_mut(v.index())
            .ok_or(BuildError::Dangling(Dangling::Value(v)))? = true;
        Ok(())
    }
    let mut lazy = vec![false; m.values.len()];
    for f in &m.functions {
        for b in f.blocks.iter().rev() {
            let bd = m
                .blocks
                .get(b.index())
                .ok_or(BuildError::Dangling(Dangling::Block(*b)))?;
            for n in bd.insts.iter().rev() {
                let is_lazy = match n.result {
                    Some(r) => *lazy
                        .get(r.index())
                        .ok_or(BuildError::Dangling(Dangling::Value(r)))?,
                    None => false,
                };
                if is_lazy && !n.inst.is_tangled() {
                    let mut dangling = None;
                    n.inst.for_each_value(|v| match lazy.get_mut(v.index()) {
                        Some(l) => *l = true,
                        None => dangling = Some(v),
                    });
                    if let Some(v) = dangling {
                        return Err(BuildError::Dangling(Dangling::Value(v)));
                    }
                }
                match &n.inst {
                    Inst::Select { t, f, .. } => {
                        mark(&mut lazy, *t)?;
                        mark(&mut lazy, *f)?;
                    }
                    Inst::Binary {
                        op: BinOp::LogicalAnd | BinOp::LogicalOr,
                        b,
                        ..
                    } => mark(&mut lazy, *b)?,
                    _ => {}
                }
            }
        }
    }
    Ok(lazy)
}

/// Whether `inst` is one of the instructions whose result SPIRV-Cross may forward without the
/// dependencies of its operands, whatever its operands are: `OpQuantizeToF16`, an image query,
/// `OpUMulExtended`, `OpSMulExtended`, `InterpolateAtSample`, `InterpolateAtOffset`, an image
/// sample, fetch, gather, or read, `OpArrayLength`, and `OpFunctionCall`. An upper bound of
/// [`forwards_without_dependencies`], for the byte ledger.
fn may_forward_without_dependencies(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Unary {
            op: UnOp::QuantizeToF16,
            ..
        } | Inst::ImageQuery { .. }
            | Inst::Binary {
                op: BinOp::UMulExtended | BinOp::SMulExtended,
                ..
            }
            | Inst::Glsl {
                op: GlslOp::InterpolateAtSample | GlslOp::InterpolateAtOffset,
                ..
            }
            | Inst::ImageSample { .. }
            | Inst::ImageFetch { .. }
            | Inst::ImageGather { .. }
            | Inst::ImageDrefGather { .. }
            | Inst::ImageRead { .. }
            | Inst::ArrayLength { .. }
            | Inst::Call { .. }
    )
}

/// Whether SPIRV-Cross forwards the result of `inst` into the instruction that reads it without
/// the dependencies of an operand that a later instruction may overwrite.
///
/// SPIRV-Cross (MoltenVK's MSL conversion) writes a value it forwards into the text of its
/// reader, so the value is evaluated where the reader is written, and it keeps that correct by
/// recording, on each forwarded expression, the expressions it is built from: a store to a
/// variable invalidates the expressions loaded from it, an `OpCompositeInsert` it performs in
/// place invalidates the composite, and a read of an expression that depends on an invalidated
/// one makes it declare the value where it stands. These forwarded results record no
/// dependencies of an operand (`adec7acb`, MoltenVK 1.4.1's pin, and upstream through
/// 2026-09-22), in `CompilerMSL`: `OpQuantizeToF16`; `OpImageQuerySizeLod`, of its level;
/// `OpUMulExtended` and `OpSMulExtended`, of either factor; `InterpolateAtSample` and
/// `InterpolateAtOffset`, of the sample or offset (the interpolant is a stage input, never
/// overwritten). In `CompilerGLSL` and `CompilerMSL` alike, no image query, sample, fetch,
/// gather, or read records the dependencies of its image or sampled image, and `OpArrayLength`
/// none of its buffer: an image, sampler, or buffer chosen from an array by an index that is not a
/// constant ([`Reads::indexed`]) is written with that index, which a later write may overwrite.
/// In `CompilerGLSL`: a call to a function SPIRV-Cross finds pure, of an argument it does not
/// load itself. Such a value read after a write that overwrites one of those operands reads the
/// new operand: authored case 29d314e8 read three half taps one step to the right on MoltenVK.
/// A constant operand is never overwritten.
fn forwards_without_dependencies(m: &TypedModule, inst: &Inst, reads: &Reads) -> BResult<bool> {
    let overwritable =
        |v: Value| -> BResult<bool> { Ok(!matches!(m.value(v)?.origin, ValueOrigin::Const(_))) };
    let any_overwritable = |vs: &[Value]| -> BResult<bool> {
        for v in vs {
            if overwritable(*v)? {
                return Ok(true);
            }
        }
        Ok(false)
    };
    match inst {
        Inst::Unary {
            op: UnOp::QuantizeToF16,
            a,
        } => overwritable(*a),
        Inst::ImageQuery { query, image } => Ok(reads.indexed(*image)
            || match query {
                ImageQuery::SizeLod(lod) => overwritable(*lod)?,
                _ => false,
            }),
        Inst::ImageSample { sampled_image, .. }
        | Inst::ImageGather { sampled_image, .. }
        | Inst::ImageDrefGather { sampled_image, .. } => Ok(reads.indexed(*sampled_image)),
        Inst::ImageFetch { image, .. } | Inst::ImageRead { image, .. } => Ok(reads.indexed(*image)),
        Inst::ArrayLength { ptr, .. } => Ok(reads.indexed(*ptr)),
        Inst::Binary {
            op: BinOp::UMulExtended | BinOp::SMulExtended,
            a,
            b,
            ..
        } => any_overwritable(&[*a, *b]),
        Inst::Glsl {
            op: GlslOp::InterpolateAtSample | GlslOp::InterpolateAtOffset,
            args,
            ..
        } => any_overwritable(args.get(1..).unwrap_or_default()),
        Inst::Call { args, .. } => any_overwritable(args),
        _ => Ok(false),
    }
}

/// What the forwarding analysis knows of each value, by value: a lower bound of how many times
/// SPIRV-Cross reads it (saturating at 2), and whether it is an image, sampler, or pointer that may
/// carry an index that is not a constant.
struct Reads {
    flags: Vec<u8>,
}

impl Reads {
    /// The read count, at most 2.
    const COUNT: u8 = 0b0000_0011;
    /// The value has been read by a scalar extract of a vector (counted once, see [`reads`]).
    const EXTRACTED: u8 = 0b1000_0000;
    /// The value may carry an index that is not a constant.
    const INDEXED: u8 = 0b0100_0000;
    /// While [`reads`] counts one instruction's reads: the value has been counted for it.
    const SEEN: u8 = 0b0010_0000;

    /// `f` with one more read counted, saturating at 2.
    fn counted(f: u8) -> u8 {
        (f & !Self::COUNT) | (f & Self::COUNT).saturating_add(1).min(2)
    }

    /// How many times SPIRV-Cross reads `v`, at least (saturating at 2).
    fn count(&self, v: Value) -> u8 {
        self.flags.get(v.index()).map_or(0, |f| f & Self::COUNT)
    }

    /// Whether `v`, an image, sampler, sampled image, or pointer, may be written by SPIRV-Cross
    /// as an expression that contains an index that is not a constant: it is a phi, or the result
    /// of an instruction other than a load, access chain, sampled image, image, or copy, or one of
    /// those reached through such a value or through an access chain index that is not a constant.
    /// A global, a `Function` variable, and a parameter carry no index.
    fn indexed(&self, v: Value) -> bool {
        self.flags
            .get(v.index())
            .is_none_or(|f| f & Self::INDEXED != 0)
    }
}

/// A lower bound of how many times SPIRV-Cross reads each value, saturating at 2, and which values
/// may carry an index that is not a constant.
///
/// SPIRV-Cross forwards a value it reads once into its reader and declares a value it reads twice
/// where it stands (the second read forces a temporary, for an expression that tracks its reads,
/// which every result [`forwards_without_dependencies`] admits does). A value is copied only when
/// it is read once, so the count must never exceed SPIRV-Cross's: a count of one where
/// SPIRV-Cross reads twice adds a copy it did not need, while a count of two where it reads once
/// drops one it did. A count of zero is exact: SPIRV-Cross never reads the value. So:
/// - A copy is its operand: a read of a copy is a read of the value whose id it takes, and the copy
///   reads nothing itself.
/// - An instruction, a phi, or a terminator reads each distinct value once, however often it names
///   it (SPIRV-Cross writes a splat of one scalar as `float4(q)`).
/// - An `OpVectorShuffle` reads an operand only when a selected component comes from it (with
///   every component from the first, SPIRV-Cross writes a swizzle of it and never reads the
///   second).
/// - Every scalar extract of one vector together is one read of the vector, made only if one of
///   them is read: SPIRV-Cross writes such an extract as a swizzle of the vector it reads where
///   the extract is read (an extract nothing reads never reads the vector), and merges adjacent
///   extracts of one vector in a constructor into one swizzle (`float4(q.xy, 0, 1)`).
/// - A switch with no case reads nothing: SPIRV-Cross writes it as a block and never reads its
///   selector. Every other terminator reads its operand: an `OpBranchConditional` the builder
///   makes (`if_`) sends its true edge to an arm block of its own, never to the merge, so
///   SPIRV-Cross always writes `if (c)`.
///
/// One forward pass in layout order (definitions before their uses, except a phi's incoming
/// values) finds each copy's root and each value that may carry an index; a second counts the
/// reads of every instruction, phi, and terminator but those extracts; a third counts the
/// extracts that are read. Counting keeps no list: a per-value mark (`Reads::SEEN`), set at an
/// instruction's first read of a value and cleared after it, makes each value count once.
fn reads(m: &TypedModule) -> BResult<Reads> {
    let block = |b: &BlockId| {
        m.blocks
            .get(b.index())
            .ok_or(BuildError::Dangling(Dangling::Block(*b)))
    };
    let dangling = |v: Value| BuildError::Dangling(Dangling::Value(v));
    let mut flags = vec![0u8; m.values.len()];
    let mut root: Vec<u32> = (0..m.values.len())
        .map(|i| u32::try_from(i).unwrap_or(u32::MAX))
        .collect();
    for f in &m.functions {
        for b in &f.blocks {
            let bd = block(b)?;
            for p in &bd.phis {
                *flags.get_mut(p.result.index()).ok_or(dangling(p.result))? |= Reads::INDEXED;
            }
            for n in &bd.insts {
                let Some(r) = n.result else { continue };
                let indexed = |v: Value| -> BResult<bool> {
                    Ok(flags.get(v.index()).ok_or(dangling(v))? & Reads::INDEXED != 0)
                };
                let carries = match &n.inst {
                    Inst::Unary {
                        op: UnOp::CopyObject,
                        a,
                    } => {
                        let to = *root.get(a.index()).ok_or(dangling(*a))?;
                        *root.get_mut(r.index()).ok_or(dangling(r))? = to;
                        indexed(*a)?
                    }
                    Inst::Load { ptr, .. } => indexed(*ptr)?,
                    Inst::AccessChain { base, indices, .. } => {
                        // No module the builder accepts reaches an image, a sampler, or a block
                        // through a chain on another chain (descriptor arrays do not nest, and
                        // neither can be a member), so the base is a global today; it is checked
                        // so the rule stays conservative if that changes.
                        let mut carries = indexed(*base)?;
                        for i in indices.iter() {
                            carries |= !matches!(m.value(*i)?.origin, ValueOrigin::Const(_));
                        }
                        carries
                    }
                    Inst::SampledImage { image, sampler } => indexed(*image)? || indexed(*sampler)?,
                    Inst::Image(a) => indexed(*a)?,
                    _ => true,
                };
                if carries {
                    *flags.get_mut(r.index()).ok_or(dangling(r))? |= Reads::INDEXED;
                }
            }
        }
    }
    let root_of = |v: Value| root.get(v.index()).copied().ok_or(dangling(v));
    let vector_len = |v: Value| -> BResult<Option<u32>> {
        Ok(match m.types().get(m.value(v)?.ty)? {
            Type::Vector { len, .. } => Some(u32::from(*len)),
            _ => None,
        })
    };
    // The vector a scalar extract of a vector reads, which SPIRV-Cross reads where the extract
    // is read.
    let scalar_extract_of_vector = |inst: &Inst| -> BResult<Option<Value>> {
        Ok(match inst {
            Inst::CompositeExtract { composite, indices }
                if indices.len() == 1 && vector_len(*composite)?.is_some() =>
            {
                Some(*composite)
            }
            _ => None,
        })
    };
    // Count one read of each distinct value (by root) that `each` visits, the reads of one
    // instruction, phi, or terminator: the first visit of a root marks it `SEEN` and counts it,
    // and a second walk clears the marks, so no list of the values read is kept.
    let once = |flags: &mut [u8], each: &dyn Fn(&mut dyn FnMut(Value))| -> BResult<()> {
        let mut missing = None;
        each(
            &mut |v| match root.get(v.index()).and_then(|r| flags.get_mut(*r as usize)) {
                Some(f) if *f & Reads::SEEN == 0 => *f = Reads::counted(*f | Reads::SEEN),
                Some(_) => {}
                None => missing = Some(v),
            },
        );
        each(&mut |v| {
            if let Some(f) = root.get(v.index()).and_then(|r| flags.get_mut(*r as usize)) {
                *f &= !Reads::SEEN;
            }
        });
        missing.map_or(Ok(()), |v| Err(dangling(v)))
    };
    for f in &m.functions {
        for b in &f.blocks {
            let bd = block(b)?;
            for p in &bd.phis {
                once(&mut flags, &|visit| {
                    for (v, _) in &p.incoming {
                        visit(*v);
                    }
                })?;
            }
            for n in &bd.insts {
                match &n.inst {
                    Inst::Unary {
                        op: UnOp::CopyObject,
                        ..
                    } => {}
                    Inst::VectorShuffle { a, b, components } => {
                        let first = vector_len(*a)?;
                        let from_a = components.iter().any(|c| first.is_none_or(|len| *c < len));
                        let from_b = components.iter().any(|c| first.is_none_or(|len| *c >= len));
                        once(&mut flags, &|visit| {
                            if from_a {
                                visit(*a);
                            }
                            if from_b {
                                visit(*b);
                            }
                        })?;
                    }
                    // Counted below, once it is known whether the extract is read.
                    inst if scalar_extract_of_vector(inst)?.is_some() => {}
                    inst => once(&mut flags, &|visit| inst.for_each_value(visit))?,
                }
            }
            match &bd.term {
                // A switch with no case is a block to SPIRV-Cross (`do { … } while (false)`, or
                // nothing when its default is its merge): it never reads the selector.
                Terminator::Switch { cases, .. } if cases.is_empty() => {}
                term => once(&mut flags, &|visit| term.for_each_value(visit))?,
            }
        }
    }
    for f in &m.functions {
        for b in &f.blocks {
            for n in &block(b)?.insts {
                let (Some(r), Some(composite)) = (n.result, scalar_extract_of_vector(&n.inst)?)
                else {
                    continue;
                };
                if flags.get(r.index()).ok_or(dangling(r))? & Reads::COUNT != 0 {
                    let v = root_of(composite)?;
                    let f = flags.get_mut(v as usize).ok_or(dangling(Value::new(v)))?;
                    if *f & Reads::EXTRACTED == 0 {
                        *f = Reads::counted(*f | Reads::EXTRACTED);
                    }
                }
            }
        }
    }
    Ok(Reads { flags })
}

// ---- Ids ----

/// SPIR-V result ids, assigned in logical-layout order.
struct Ids {
    glsl: Option<u32>,
    types: Vec<u32>,
    values: Vec<u32>,
    blocks: Vec<u32>,
    funcs: Vec<u32>,
    /// The module-level `OpUndef` of each type some `OpUndef` has (0 for none), by type.
    undefs: Vec<u32>,
    /// Result ids of void `OpFunctionCall`s (the typed IR gives them no value), by
    /// `(block, instruction index)`; a lookup table, never iterated.
    void_calls: std::collections::HashMap<(u32, u32), u32>,
    /// The instruction's own result id of each value SPIRV-Cross could evaluate where the module
    /// does not: a tangled value it could evaluate lazily, and a value it would forward into its
    /// one reader without the dependencies of its operands. The value's id is its
    /// `OpCopyObject`'s, written right after it. A lookup table, never iterated.
    materialized: std::collections::HashMap<Value, u32>,
    bound: u32,
}

impl Ids {
    fn assign(m: &TypedModule) -> BResult<Ids> {
        let mut next = 1u32;
        let mut take = || -> BResult<u32> {
            let id = next;
            next = next.checked_add(1).ok_or(BuildError::IdSpaceExhausted)?;
            Ok(id)
        };
        let uses_glsl = m
            .blocks
            .iter()
            .any(|b| b.insts.iter().any(|n| matches!(n.inst, Inst::Glsl { .. })));
        let glsl = if uses_glsl { Some(take()?) } else { None };
        let mut types = vec![0u32; m.types.len()];
        let mut values = vec![0u32; m.values.len()];
        let mut blocks = vec![0u32; m.blocks.len()];
        let mut funcs = vec![0u32; m.functions.len()];
        let mut void_calls = std::collections::HashMap::new();
        let mut materialized = std::collections::HashMap::new();
        let lazy = lazy_values(m)?;
        let reads = reads(m)?;
        fn set(values: &mut [u32], v: Value, id: u32) -> BResult<()> {
            *values
                .get_mut(v.index())
                .ok_or(BuildError::Dangling(Dangling::Value(v)))? = id;
            Ok(())
        }
        for d in &m.decls {
            match d {
                Decl::Type(t) => {
                    *types
                        .get_mut(t.index())
                        .ok_or(BuildError::Dangling(Dangling::Type(*t)))? = take()?;
                }
                Decl::Const(c) => {
                    let v = m.const_value(*c)?;
                    set(&mut values, v, take()?)?;
                }
                Decl::Global(g) => {
                    let v = m
                        .globals
                        .get(g.index())
                        .ok_or(BuildError::Dangling(Dangling::Global(*g)))?
                        .value;
                    set(&mut values, v, take()?)?;
                }
            }
        }
        // One module-level `OpUndef` per type, written after the declarations.
        let mut undefs = vec![0u32; m.types.len()];
        for n in m.blocks.iter().flat_map(|b| b.insts.iter()) {
            if let (Some(r), Inst::Undef) = (n.result, &n.inst) {
                let t = m.value(r)?.ty;
                *undefs
                    .get_mut(t.index())
                    .ok_or(BuildError::Dangling(Dangling::Type(t)))? = 1;
            }
        }
        for u in undefs.iter_mut().filter(|u| **u != 0) {
            *u = take()?;
        }
        for f in funcs.iter_mut() {
            *f = take()?;
        }
        for f in &m.functions {
            for p in &f.params {
                set(&mut values, *p, take()?)?;
            }
            for (i, b) in f.blocks.iter().enumerate() {
                *blocks
                    .get_mut(b.index())
                    .ok_or(BuildError::Dangling(Dangling::Block(*b)))? = take()?;
                if i == 0 {
                    for var in &f.vars {
                        let v = m
                            .vars
                            .get(var.index())
                            .ok_or(BuildError::Dangling(Dangling::Var(*var)))?
                            .value;
                        set(&mut values, v, take()?)?;
                    }
                }
                let bd = m
                    .blocks
                    .get(b.index())
                    .ok_or(BuildError::Dangling(Dangling::Block(*b)))?;
                for p in &bd.phis {
                    set(&mut values, p.result, take()?)?;
                }
                for (i, n) in bd.insts.iter().enumerate() {
                    // A value SPIRV-Cross could evaluate where the module does not is written with
                    // its own id and a copy that carries the value's: a tangled value it could
                    // evaluate lazily, and a value it would forward into its one reader without
                    // the dependencies of its operands.
                    let materialize = match n.result {
                        Some(r) => {
                            (n.inst.is_tangled() && lazy.get(r.index()).copied().unwrap_or(false))
                                || (reads.count(r) == 1
                                    && forwards_without_dependencies(m, &n.inst, &reads)?)
                        }
                        None => false,
                    };
                    match (n.result, &n.inst) {
                        // A copy is its operand: the operand's definition dominates the copy, which
                        // dominates every use, and blocks are laid out after their dominators, so
                        // the operand's id is assigned by now.
                        (
                            Some(r),
                            Inst::Unary {
                                op: UnOp::CopyObject,
                                a,
                            },
                        ) => {
                            let id = values.get(a.index()).copied().unwrap_or(0);
                            let id = if id == 0 { take()? } else { id };
                            set(&mut values, r, id)?
                        }
                        (Some(r), Inst::Undef) => {
                            let t = m.value(r)?.ty;
                            let id = undefs.get(t.index()).copied().unwrap_or(0);
                            set(&mut values, r, id)?
                        }
                        (Some(r), _) if materialize => {
                            materialized.insert(r, take()?);
                            set(&mut values, r, take()?)?;
                            // An array's copy is written element by element (`Ser::copy`), under
                            // the ids that follow the value's.
                            for _ in 0..copy_parts(m, m.value(r)?.ty)? {
                                take()?;
                            }
                        }
                        (Some(r), _) => set(&mut values, r, take()?)?,
                        (None, Inst::Call { .. }) => {
                            void_calls.insert((b.0, i as u32), take()?);
                        }
                        (None, _) => {}
                    }
                }
            }
        }
        Ok(Ids {
            glsl,
            types,
            values,
            blocks,
            funcs,
            undefs,
            void_calls,
            materialized,
            bound: next,
        })
    }

    fn ty(&self, t: TypeId) -> BResult<u32> {
        nonzero(
            self.types.get(t.index()),
            BuildError::Dangling(Dangling::Type(t)),
        )
    }

    fn val(&self, v: Value) -> BResult<u32> {
        nonzero(
            self.values.get(v.index()),
            BuildError::Dangling(Dangling::Value(v)),
        )
    }

    fn block(&self, b: BlockId) -> BResult<u32> {
        nonzero(
            self.blocks.get(b.index()),
            BuildError::Dangling(Dangling::Block(b)),
        )
    }

    fn func(&self, f: FuncId) -> BResult<u32> {
        nonzero(
            self.funcs.get(f.index()),
            BuildError::Dangling(Dangling::Func(f)),
        )
    }
}

fn nonzero(id: Option<&u32>, e: BuildError) -> BResult<u32> {
    match id {
        Some(id) if *id != 0 => Ok(*id),
        _ => Err(e),
    }
}

// ---- Words ----

#[derive(Default)]
struct Writer {
    words: Vec<u32>,
    relocs: Vec<(Symbol, u32)>,
}

impl Writer {
    fn inst(&mut self, op: Op, operands: &[u32]) -> BResult<()> {
        let words = operands.len().saturating_add(1);
        let count = u32::try_from(words).ok().filter(|c| *c <= 0xFFFF).ok_or(
            BuildError::InstructionTooLong {
                op,
                words: u64::try_from(words).unwrap_or(u64::MAX),
            },
        )?;
        self.words.push((count << 16) | op as u32);
        self.words.extend_from_slice(operands);
        Ok(())
    }

    /// `OpDecorate target deco <literal>` whose literal link assigns.
    fn reloc_decorate(&mut self, target: u32, deco: Decoration, sym: Symbol) -> BResult<()> {
        self.inst(Op::Decorate, &[target, deco as u32, 0])?;
        let at = u32::try_from(self.words.len() - 1).map_err(|_| BuildError::IdSpaceExhausted)?;
        self.relocs.push((sym, at));
        Ok(())
    }
}

/// A literal string: UTF-8, NUL-terminated, padded to a word.
fn string(s: &str) -> Vec<u32> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| c.iter().rev().fold(0u32, |w, b| (w << 8) | u32::from(*b)))
        .collect()
}

struct Ser<'a> {
    m: &'a TypedModule,
    ids: &'a Ids,
    /// The headers of the loops whose `OpLoopMerge` asks something of the driver's unrolling,
    /// with what they ask, sorted ([`crate::spirv::ir::LoopMeta::unrolling`]).
    controls: Vec<(BlockId, Unrolling)>,
}

impl Ser<'_> {
    fn ty(&self, t: TypeId) -> BResult<u32> {
        self.ids.ty(t)
    }

    fn val(&self, v: Value) -> BResult<u32> {
        self.ids.val(v)
    }

    fn cid(&self, c: ConstId) -> BResult<u32> {
        self.ids.val(self.m.const_value(c)?)
    }

    fn value_ty(&self, v: Value) -> BResult<TypeId> {
        self.m.value(v).map(|d| d.ty)
    }

    fn result_ty(&self, v: Value) -> BResult<u32> {
        self.ty(self.value_ty(v)?)
    }

    /// Write the copy SPIRV-Cross declares as a temporary where it stands: `dst`, of type `ty`, a
    /// copy of `src`. A scalar, vector, matrix or struct is one `OpCopyObject`. An array of a
    /// known length is copied element by element, each element extracted and copied under the
    /// next two ids of `parts` (an array element recursively), then constructed under `dst`
    /// ([`copy_parts`] counts those ids): SPIRV-Cross's MSL declares an array temporary as a
    /// `spvUnsafeArray` and initializes the `OpCopyObject` of a loaded array with the load's
    /// expression, which Metal's compiler refuses where the array lies in a buffer block (a plain
    /// C array there, `no viable conversion from 'const constant uint4[2]'`), while each element's
    /// copy and the construct of the copies are values it converts for every array.
    fn copy(&self, w: &mut Writer, ty: TypeId, dst: u32, src: u32, parts: &mut u32) -> BResult<()> {
        fn next(parts: &mut u32) -> BResult<u32> {
            let id = *parts;
            *parts = id.checked_add(1).ok_or(BuildError::IdSpaceExhausted)?;
            Ok(id)
        }
        let array = match self.m.types().get(ty)? {
            Type::Array { elem, len, .. } => self
                .m
                .consts()
                .array_len(*len, self.m.types())?
                .map(|n| (*elem, n)),
            _ => None,
        };
        let Some((elem, n)) = array else {
            return w.inst(Op::CopyObject, &[self.ty(ty)?, dst, src]);
        };
        let mut construct = vec![self.ty(ty)?, dst];
        for i in 0..n {
            let extracted = next(parts)?;
            let part = next(parts)?;
            w.inst(Op::CompositeExtract, &[self.ty(elem)?, extracted, src, i])?;
            self.copy(w, elem, part, extracted, parts)?;
            construct.push(part);
        }
        w.inst(Op::CompositeConstruct, &construct)
    }

    // ---- Preamble ----

    fn preamble(&self, w: &mut Writer) -> BResult<()> {
        let r = &self.m.requirements;
        for c in &r.spirv_capabilities {
            w.inst(Op::Capability, &[c.as_raw()])?;
        }
        for e in &r.spirv_extensions {
            w.inst(Op::Extension, &string(spirv_extension_name(*e)))?;
        }
        if let Some(id) = self.ids.glsl {
            let mut ops = vec![id];
            ops.extend(string("GLSL.std.450"));
            w.inst(Op::ExtInstImport, &ops)?;
        }
        let addressing = if uses_physical_storage_buffer(r) {
            crate::spv::AddressingModel::PhysicalStorageBuffer64
        } else {
            crate::spv::AddressingModel::Logical
        };
        let memory = if uses_vulkan_memory_model(r) {
            crate::spv::MemoryModel::Vulkan
        } else {
            crate::spv::MemoryModel::GLSL450
        };
        w.inst(Op::MemoryModel, &[addressing as u32, memory as u32])?;
        let e = &self.m.entry;
        let model = match e.model {
            ExecModel::Vertex => crate::spv::ExecutionModel::Vertex,
            ExecModel::Fragment => crate::spv::ExecutionModel::Fragment,
            ExecModel::GLCompute => crate::spv::ExecutionModel::GLCompute,
            ExecModel::TessellationEvaluation => crate::spv::ExecutionModel::TessellationEvaluation,
            ExecModel::TessellationControl => crate::spv::ExecutionModel::TessellationControl,
            ExecModel::TaskEXT => crate::spv::ExecutionModel::TaskEXT,
            ExecModel::MeshEXT => crate::spv::ExecutionModel::MeshEXT,
        };
        let f = self.ids.func(e.func)?;
        let mut ops = vec![model as u32, f];
        ops.extend(string(ENTRY_NAME));
        for g in &self.m.interface {
            let gv = self
                .m
                .globals
                .get(g.index())
                .ok_or(BuildError::Dangling(Dangling::Global(*g)))?;
            ops.push(self.val(gv.value)?);
        }
        w.inst(Op::EntryPoint, &ops)?;
        for mode in &e.modes {
            self.mode(w, f, *mode)?;
        }
        Ok(())
    }

    fn mode(&self, w: &mut Writer, f: u32, mode: ExecMode) -> BResult<()> {
        use crate::spv::ExecutionMode as M;
        let plain = |w: &mut Writer, m: M, lits: &[u32]| {
            let mut ops = vec![f, m as u32];
            ops.extend_from_slice(lits);
            w.inst(Op::ExecutionMode, &ops)
        };
        match mode {
            ExecMode::OriginUpperLeft => plain(w, M::OriginUpperLeft, &[]),
            ExecMode::EarlyFragmentTests => plain(w, M::EarlyFragmentTests, &[]),
            ExecMode::DepthReplacing => plain(w, M::DepthReplacing, &[]),
            ExecMode::DepthGreater => plain(w, M::DepthGreater, &[]),
            ExecMode::DepthLess => plain(w, M::DepthLess, &[]),
            ExecMode::StencilRefReplacing => plain(w, M::StencilRefReplacingEXT, &[]),
            ExecMode::PixelInterlockOrdered => plain(w, M::PixelInterlockOrderedEXT, &[]),
            ExecMode::SampleInterlockOrdered => plain(w, M::SampleInterlockOrderedEXT, &[]),
            ExecMode::LocalSize { x, y, z } => plain(w, M::LocalSize, &[x, y, z]),
            ExecMode::DerivativeGroupLinear => plain(w, M::DerivativeGroupLinearKHR, &[]),
            ExecMode::SignedZeroInfNanPreserve { width } => {
                plain(w, M::SignedZeroInfNanPreserve, &[u32::from(width)])
            }
            ExecMode::DenormPreserve { width } => plain(w, M::DenormPreserve, &[u32::from(width)]),
            ExecMode::RoundingModeRte { width } => {
                plain(w, M::RoundingModeRTE, &[u32::from(width)])
            }
            ExecMode::SubgroupUniformControlFlow => plain(w, M::SubgroupUniformControlFlowKHR, &[]),
            ExecMode::MaximallyReconverges => plain(w, M::MaximallyReconvergesKHR, &[]),
            ExecMode::RequireFullQuads => plain(w, M::RequireFullQuadsKHR, &[]),
            ExecMode::PostDepthCoverage => plain(w, M::PostDepthCoverage, &[]),
            ExecMode::Triangles => plain(w, M::Triangles, &[]),
            ExecMode::Quads => plain(w, M::Quads, &[]),
            ExecMode::SpacingEqual => plain(w, M::SpacingEqual, &[]),
            ExecMode::SpacingFractionalEven => plain(w, M::SpacingFractionalEven, &[]),
            ExecMode::SpacingFractionalOdd => plain(w, M::SpacingFractionalOdd, &[]),
            ExecMode::VertexOrderCw => plain(w, M::VertexOrderCw, &[]),
            ExecMode::VertexOrderCcw => plain(w, M::VertexOrderCcw, &[]),
            ExecMode::OutputVertices { n } => plain(w, M::OutputVertices, &[n]),
            ExecMode::OutputPrimitives { n } => plain(w, M::OutputPrimitivesEXT, &[n]),
            ExecMode::OutputPoints => plain(w, M::OutputPoints, &[]),
            ExecMode::OutputLines => plain(w, M::OutputLinesEXT, &[]),
            ExecMode::OutputTriangles => plain(w, M::OutputTrianglesEXT, &[]),
        }
    }

    // ---- Annotations ----

    fn decorate(&self, w: &mut Writer, target: u32, d: Decoration, lits: &[u32]) -> BResult<()> {
        let mut ops = vec![target, d as u32];
        ops.extend_from_slice(lits);
        w.inst(Op::Decorate, &ops)
    }

    fn annotations(&self, w: &mut Writer) -> BResult<()> {
        let m = self.m;
        // Explicitly laid-out `Workgroup` blocks share one allocation; when an entry declares
        // more than one, each must say it aliases the others.
        let mut workgroup_blocks = 0usize;
        // Read-only storage resources of a graphics stage, unwritten buffers of a compute stage.
        let read_only = non_writable::read_only_resources(m)?;
        for g in &m.interface {
            let gv = m
                .globals
                .get(g.index())
                .ok_or(BuildError::Dangling(Dangling::Global(*g)))?;
            if gv.class == StorageClass::Workgroup && m.types.facts(gv.pointee)?.block {
                workgroup_blocks += 1;
            }
        }
        for d in &m.decls {
            match d {
                Decl::Type(t) => self.type_decorations(w, *t)?,
                Decl::Const(c) => match m.consts.get(*c)?.kind {
                    ConstKind::Spec { key, .. } => {
                        w.reloc_decorate(self.cid(*c)?, Decoration::SpecId, Symbol::SpecId(key))?;
                    }
                    ConstKind::WorkgroupSize { .. } => {
                        let builtin = crate::spv::BuiltIn::WorkgroupSize as u32;
                        self.decorate(w, self.cid(*c)?, Decoration::BuiltIn, &[builtin])?;
                    }
                    ConstKind::Plain(_) | ConstKind::SpecOp { .. } => {}
                },
                Decl::Global(g) => {
                    let gv = m
                        .globals
                        .get(g.index())
                        .ok_or(BuildError::Dangling(Dangling::Global(*g)))?;
                    let id = self.val(gv.value)?;
                    match &gv.kind {
                        GlobalKind::Resource(slot) => {
                            w.reloc_decorate(
                                id,
                                Decoration::DescriptorSet,
                                Symbol::DescriptorSet(*slot),
                            )?;
                            w.reloc_decorate(id, Decoration::Binding, Symbol::Binding(*slot))?;
                            if let ResourceKind::InputAttachment(color) = slot.kind {
                                self.decorate(w, id, Decoration::InputAttachmentIndex, &[color])?;
                            }
                            // Two Metal buffer slots may be bound to the same memory, so every
                            // writable view of a Metal buffer is `Aliased`.
                            if slot.kind == ResourceKind::Buffer
                                && gv.class == StorageClass::StorageBuffer
                            {
                                self.decorate(w, id, Decoration::Aliased, &[])?;
                            }
                            if read_only.contains(g) {
                                self.decorate(w, id, Decoration::NonWritable, &[])?;
                            }
                            if m.coherent.contains(g) {
                                self.decorate(w, id, Decoration::Coherent, &[])?;
                            }
                            if m.volatile.contains(g) {
                                self.decorate(w, id, Decoration::Volatile, &[])?;
                            }
                        }
                        GlobalKind::Io { key, dir, deco } => {
                            w.reloc_decorate(
                                id,
                                Decoration::Location,
                                Symbol::Location(key.clone()),
                            )?;
                            if deco.packed {
                                w.reloc_decorate(
                                    id,
                                    Decoration::Component,
                                    Symbol::Component(key.clone()),
                                )?;
                            }
                            // A mesh entry's primitive output, or a fragment input reading one; an
                            // integer one is still `Flat` (VUID-StandaloneSpirv-Flat-04744).
                            if m.per_primitive.contains(g) {
                                self.decorate(w, id, Decoration::PerPrimitiveEXT, &[])?;
                            }
                            // Interpolation qualifiers exist only across the rasterizer.
                            let interpolated = matches!(
                                (m.entry.model, dir),
                                (
                                    ExecModel::Vertex
                                        | ExecModel::TessellationEvaluation
                                        | ExecModel::MeshEXT,
                                    IoDir::Output
                                ) | (ExecModel::Fragment, IoDir::Input)
                            );
                            if interpolated {
                                match deco.interpolation {
                                    Interpolation::Smooth => {}
                                    Interpolation::Flat => {
                                        self.decorate(w, id, Decoration::Flat, &[])?
                                    }
                                    Interpolation::NoPerspective => {
                                        self.decorate(w, id, Decoration::NoPerspective, &[])?
                                    }
                                }
                                match deco.sampling {
                                    Sampling::Center => {}
                                    Sampling::Centroid => {
                                        self.decorate(w, id, Decoration::Centroid, &[])?
                                    }
                                    Sampling::Sample => {
                                        self.decorate(w, id, Decoration::Sample, &[])?
                                    }
                                }
                            }
                            if let Some(i) = deco.index {
                                self.decorate(w, id, Decoration::Index, &[i])?;
                            }
                        }
                        GlobalKind::BuiltIn { builtin, dir } => {
                            self.decorate(w, id, Decoration::BuiltIn, &[*builtin as u32])?;
                            if m.invariant.contains(g) {
                                self.decorate(w, id, Decoration::Invariant, &[])?;
                            }
                            // A mesh entry's per-primitive builtin outputs.
                            if m.entry.model == ExecModel::MeshEXT
                                && *dir == IoDir::Output
                                && mesh_per_primitive(*builtin)
                            {
                                self.decorate(w, id, Decoration::PerPrimitiveEXT, &[])?;
                            }
                            // The tessellation levels are per patch
                            // (VUID-StandaloneSpirv-TessLevelInner-10880).
                            if matches!(
                                builtin,
                                crate::spv::BuiltIn::TessLevelOuter
                                    | crate::spv::BuiltIn::TessLevelInner
                            ) {
                                self.decorate(w, id, Decoration::Patch, &[])?;
                            }
                            // Integer and double fragment inputs are never interpolated
                            // (VUID-StandaloneSpirv-Flat-04744), builtins included.
                            if m.entry.model == ExecModel::Fragment
                                && *dir == IoDir::Input
                                && self.needs_flat(gv.pointee)?
                            {
                                self.decorate(w, id, Decoration::Flat, &[])?;
                            }
                        }
                        GlobalKind::FixedIo {
                            location, patch, ..
                        } => {
                            self.decorate(w, id, Decoration::Location, &[*location])?;
                            if *patch {
                                self.decorate(w, id, Decoration::Patch, &[])?;
                            }
                        }
                        GlobalKind::PushBlock | GlobalKind::Plain => {}
                    }
                    if self.holds_psb_pointer(gv.pointee)? {
                        self.decorate(w, id, Decoration::AliasedPointer, &[])?;
                    }
                    if workgroup_blocks > 1
                        && gv.class == StorageClass::Workgroup
                        && m.types.facts(gv.pointee)?.block
                    {
                        self.decorate(w, id, Decoration::Aliased, &[])?;
                    }
                }
            }
        }
        // A copy shares its operand's id, so one id may carry several values: each decoration is
        // written once per id.
        let mut non_uniform = std::collections::BTreeSet::new();
        for f in &m.functions {
            for p in &f.params {
                if self.is_psb_pointer(self.value_ty(*p)?)? {
                    self.decorate(w, self.val(*p)?, Decoration::Aliased, &[])?;
                }
            }
            for var in &f.vars {
                let lv = m
                    .vars
                    .get(var.index())
                    .ok_or(BuildError::Dangling(Dangling::Var(*var)))?;
                if self.holds_psb_pointer(lv.pointee)? {
                    self.decorate(w, self.val(lv.value)?, Decoration::AliasedPointer, &[])?;
                }
            }
            for b in &f.blocks {
                let bd = m
                    .blocks
                    .get(b.index())
                    .ok_or(BuildError::Dangling(Dangling::Block(*b)))?;
                for n in &bd.insts {
                    let Some(r) = n.result else { continue };
                    if let Inst::Binary {
                        contract: Contraction::Forbidden,
                        ..
                    }
                    | Inst::Glsl {
                        contract: Contraction::Forbidden,
                        ..
                    }
                    | Inst::Convert {
                        contract: Contraction::Forbidden,
                        ..
                    } = &n.inst
                    {
                        self.decorate(w, self.val(r)?, Decoration::NoContraction, &[])?;
                    }
                    // A value carrying a non-uniform descriptor element to its consumer
                    // (VUID-RuntimeSpirv-None-10148 and VUID-RuntimeSpirv-subgroupSize-10149).
                    if m.non_uniform.contains(&r) && non_uniform.insert(self.val(r)?) {
                        self.decorate(w, self.val(r)?, Decoration::NonUniform, &[])?;
                    }
                }
            }
        }
        Ok(())
    }

    fn is_psb_pointer(&self, t: TypeId) -> BResult<bool> {
        Ok(matches!(
            self.m.types.get(t)?,
            Type::Pointer {
                class: StorageClass::PhysicalStorageBuffer,
                ..
            }
        ))
    }

    /// Whether a fragment input of type `t` must be `Flat`: its components are integers or
    /// 64-bit floats.
    fn needs_flat(&self, t: TypeId) -> BResult<bool> {
        let mut t = t;
        loop {
            match self.m.types.get(t)? {
                Type::Array { elem, .. }
                | Type::RuntimeArray { elem, .. }
                | Type::Vector { elem, .. } => t = *elem,
                Type::Int { .. } | Type::Float { width: 64 } => return Ok(true),
                _ => return Ok(false),
            }
        }
    }

    /// A variable whose data is a `PhysicalStorageBuffer` pointer (or array of them) must say
    /// whether it aliases (`AliasedPointer`, the conservative choice).
    fn holds_psb_pointer(&self, t: TypeId) -> BResult<bool> {
        let mut t = t;
        loop {
            match self.m.types.get(t)? {
                Type::Array { elem, .. } | Type::RuntimeArray { elem, .. } => t = *elem,
                _ => return self.is_psb_pointer(t),
            }
        }
    }

    fn type_decorations(&self, w: &mut Writer, t: TypeId) -> BResult<()> {
        let id = self.ty(t)?;
        match self.m.types.get(t)? {
            Type::Array {
                stride: Some(s), ..
            }
            | Type::RuntimeArray {
                stride: Some(s), ..
            } => self.decorate(w, id, Decoration::ArrayStride, &[*s]),
            Type::Struct {
                layout: Some(l), ..
            } => {
                if l.block {
                    self.decorate(w, id, Decoration::Block, &[])?;
                }
                for (i, ml) in l.members.iter().enumerate() {
                    let i = i as u32;
                    w.inst(
                        Op::MemberDecorate,
                        &[id, i, Decoration::Offset as u32, ml.offset],
                    )?;
                    if l.non_writable {
                        w.inst(Op::MemberDecorate, &[id, i, Decoration::NonWritable as u32])?;
                    }
                    if let Some(ms) = ml.matrix_stride {
                        w.inst(Op::MemberDecorate, &[id, i, Decoration::ColMajor as u32])?;
                        w.inst(
                            Op::MemberDecorate,
                            &[id, i, Decoration::MatrixStride as u32, ms],
                        )?;
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    // ---- Types, constants, globals ----

    fn declarations(&self, w: &mut Writer) -> BResult<()> {
        let m = self.m;
        for d in &m.decls {
            match d {
                Decl::Type(t) => self.type_decl(w, *t)?,
                Decl::Const(c) => self.const_decl(w, *c)?,
                Decl::Global(g) => {
                    let gv = m
                        .globals
                        .get(g.index())
                        .ok_or(BuildError::Dangling(Dangling::Global(*g)))?;
                    let mut ops = vec![
                        self.ty(gv.ptr_ty)?,
                        self.val(gv.value)?,
                        gv.class.to_spirv() as u32,
                    ];
                    if let Some(c) = gv.init {
                        ops.push(self.cid(c)?);
                    }
                    w.inst(Op::Variable, &ops)?;
                }
            }
        }
        for (i, id) in self.ids.undefs.iter().enumerate() {
            if *id != 0 {
                w.inst(Op::Undef, &[self.ty(TypeId::new(i as u32))?, *id])?;
            }
        }
        Ok(())
    }

    fn type_decl(&self, w: &mut Writer, t: TypeId) -> BResult<()> {
        let id = self.ty(t)?;
        match self.m.types.get(t)? {
            Type::Void => w.inst(Op::TypeVoid, &[id]),
            Type::Bool => w.inst(Op::TypeBool, &[id]),
            Type::Int { width, signed } => {
                w.inst(Op::TypeInt, &[id, u32::from(*width), u32::from(*signed)])
            }
            Type::Float { width } => w.inst(Op::TypeFloat, &[id, u32::from(*width)]),
            Type::Float8 { encoding } => w.inst(Op::TypeFloat, &[id, 8, *encoding as u32]),
            Type::Vector { elem, len } => {
                w.inst(Op::TypeVector, &[id, self.ty(*elem)?, u32::from(*len)])
            }
            Type::Matrix { col, cols } => {
                w.inst(Op::TypeMatrix, &[id, self.ty(*col)?, u32::from(*cols)])
            }
            Type::Array { elem, len, .. } => {
                w.inst(Op::TypeArray, &[id, self.ty(*elem)?, self.cid(*len)?])
            }
            Type::RuntimeArray { elem, .. } => w.inst(Op::TypeRuntimeArray, &[id, self.ty(*elem)?]),
            Type::Struct { members, .. } => {
                let mut ops = vec![id];
                for m in members.iter() {
                    ops.push(self.ty(*m)?);
                }
                w.inst(Op::TypeStruct, &ops)
            }
            Type::Pointer { class, pointee } => w.inst(
                Op::TypePointer,
                &[id, class.to_spirv() as u32, self.ty(*pointee)?],
            ),
            Type::Function { ret, params } => {
                let mut ops = vec![id, self.ty(*ret)?];
                for p in params.iter() {
                    ops.push(self.ty(*p)?);
                }
                w.inst(Op::TypeFunction, &ops)
            }
            Type::Image(img) => {
                let dim = match img.dim {
                    Dim::D1 => crate::spv::Dim::D1,
                    Dim::D2 => crate::spv::Dim::D2,
                    Dim::D3 => crate::spv::Dim::D3,
                    Dim::Cube => crate::spv::Dim::Cube,
                    Dim::Buffer => crate::spv::Dim::Buffer,
                    Dim::SubpassData => crate::spv::Dim::SubpassData,
                };
                let sampled = match img.usage {
                    ImageUsage::Sampled => 1,
                    ImageUsage::Storage => 2,
                };
                w.inst(
                    Op::TypeImage,
                    &[
                        id,
                        self.ty(img.sampled_type)?,
                        dim as u32,
                        u32::from(img.depth),
                        u32::from(img.arrayed),
                        u32::from(img.multisampled),
                        sampled,
                        img.format as u32,
                    ],
                )
            }
            Type::Sampler => w.inst(Op::TypeSampler, &[id]),
            Type::SampledImage { image } => w.inst(Op::TypeSampledImage, &[id, self.ty(*image)?]),
            Type::AccelerationStructure => w.inst(Op::TypeAccelerationStructureKHR, &[id]),
            Type::RayQuery => w.inst(Op::TypeRayQueryKHR, &[id]),
        }
    }

    /// The literal words of a scalar of type `ty` with raw bits `bits`.
    fn literal(&self, ty: TypeId, bits: u64) -> BResult<Vec<u32>> {
        Ok(match self.m.types.get(ty)? {
            Type::Int { width: 64, .. } | Type::Float { width: 64 } => {
                vec![bits as u32, (bits >> 32) as u32]
            }
            // Narrow signed integers are sign-extended into the word; floats and unsigned
            // integers are zero-extended (SPIR-V §2.2.1 "Literal").
            Type::Int {
                width,
                signed: true,
            } if *width < 32 => {
                let shift = 64 - u32::from(*width);
                vec![(((bits << shift) as i64) >> shift) as u32]
            }
            _ => vec![bits as u32],
        })
    }

    fn const_decl(&self, w: &mut Writer, c: ConstId) -> BResult<()> {
        let def = self.m.consts.get(c)?;
        let id = self.cid(c)?;
        let ty = self.ty(def.ty)?;
        let is_bool = matches!(self.m.types.get(def.ty)?, Type::Bool);
        match &def.kind {
            ConstKind::Plain(Constant::Scalar { bits, .. }) => {
                if is_bool {
                    let op = if bits.0 != 0 {
                        Op::ConstantTrue
                    } else {
                        Op::ConstantFalse
                    };
                    w.inst(op, &[ty, id])
                } else {
                    let mut ops = vec![ty, id];
                    ops.extend(self.literal(def.ty, bits.0)?);
                    w.inst(Op::Constant, &ops)
                }
            }
            ConstKind::Plain(Constant::Composite { parts, .. }) => {
                let mut ops = vec![ty, id];
                for p in parts.iter() {
                    ops.push(self.cid(*p)?);
                }
                w.inst(Op::ConstantComposite, &ops)
            }
            ConstKind::Plain(Constant::Null { .. }) => w.inst(Op::ConstantNull, &[ty, id]),
            ConstKind::WorkgroupSize { parts } => {
                let mut ops = vec![ty, id];
                for p in parts {
                    ops.push(self.cid(*p)?);
                }
                w.inst(Op::SpecConstantComposite, &ops)
            }
            ConstKind::SpecOp { op, operands } => {
                let opcode = match op {
                    SpecOp::IAdd => Op::IAdd,
                    SpecOp::IMul => Op::IMul,
                    SpecOp::BitwiseAnd => Op::BitwiseAnd,
                    SpecOp::ShiftRightLogical => Op::ShiftRightLogical,
                    SpecOp::ULessThan => Op::ULessThan,
                    SpecOp::Select => Op::Select,
                };
                let mut ops = vec![ty, id, opcode as u32];
                for p in operands.iter() {
                    ops.push(self.cid(*p)?);
                }
                w.inst(Op::SpecConstantOperation, &ops)
            }
            ConstKind::Spec { default, .. } => {
                if is_bool {
                    let op = if default.0 != 0 {
                        Op::SpecConstantTrue
                    } else {
                        Op::SpecConstantFalse
                    };
                    w.inst(op, &[ty, id])
                } else {
                    let mut ops = vec![ty, id];
                    ops.extend(self.literal(def.ty, default.0)?);
                    w.inst(Op::SpecConstant, &ops)
                }
            }
        }
    }

    // ---- Functions ----

    fn function(&self, w: &mut Writer, f: FuncId) -> BResult<()> {
        let m = self.m;
        let fd = m
            .functions
            .get(f.index())
            .ok_or(BuildError::Dangling(Dangling::Func(f)))?;
        w.inst(
            Op::Function,
            &[
                self.ty(fd.ret)?,
                self.ids.func(f)?,
                // FunctionControl: `DontInline` (0x2) for a kept routine, else `None`.
                if fd.kept { 0x2 } else { 0 },
                self.ty(fd.fn_ty)?,
            ],
        )?;
        for p in &fd.params {
            w.inst(Op::FunctionParameter, &[self.result_ty(*p)?, self.val(*p)?])?;
        }
        for (i, b) in fd.blocks.iter().enumerate() {
            w.inst(Op::Label, &[self.ids.block(*b)?])?;
            if i == 0 {
                for var in &fd.vars {
                    let lv = m
                        .vars
                        .get(var.index())
                        .ok_or(BuildError::Dangling(Dangling::Var(*var)))?;
                    let mut ops = vec![
                        self.ty(lv.ptr_ty)?,
                        self.val(lv.value)?,
                        StorageClass::Function.to_spirv() as u32,
                    ];
                    if let Some(c) = lv.init {
                        ops.push(self.cid(c)?);
                    }
                    w.inst(Op::Variable, &ops)?;
                }
            }
            let bd = m
                .blocks
                .get(b.index())
                .ok_or(BuildError::Dangling(Dangling::Block(*b)))?;
            self.block(w, *b, bd)?;
        }
        w.inst(Op::FunctionEnd, &[])
    }

    fn block(&self, w: &mut Writer, id: BlockId, b: &Block) -> BResult<()> {
        for p in &b.phis {
            let mut ops = vec![self.result_ty(p.result)?, self.val(p.result)?];
            for (v, from) in &p.incoming {
                ops.push(self.val(*v)?);
                ops.push(self.ids.block(*from)?);
            }
            w.inst(Op::Phi, &ops)?;
        }
        for (i, n) in b.insts.iter().enumerate() {
            let own = n
                .result
                .and_then(|r| self.ids.materialized.get(&r).map(|id| (r, *id)));
            match own {
                // A materialized value: the instruction under its own id, then the copy SPIRV-Cross
                // declares as a temporary where it stands.
                Some((r, own)) => {
                    self.inst(w, (id, i as u32), Some((r, own)), &n.inst)?;
                    let at = self.val(r)?;
                    let mut parts = at.checked_add(1).ok_or(BuildError::IdSpaceExhausted)?;
                    self.copy(w, self.value_ty(r)?, at, own, &mut parts)?;
                }
                None => {
                    let result = n
                        .result
                        .map(|r| self.val(r).map(|id| (r, id)))
                        .transpose()?;
                    self.inst(w, (id, i as u32), result, &n.inst)?;
                }
            }
        }
        match b.merge {
            Some(Merge::Selection { merge }) => {
                w.inst(Op::SelectionMerge, &[self.ids.block(merge)?, 0])?
            }
            Some(Merge::Loop { merge, cont }) => {
                // What the loop asks of the driver's unrolling (`Unrolling`).
                let unrolling = self
                    .controls
                    .binary_search_by_key(&id, |&(h, _)| h)
                    .ok()
                    .and_then(|at| self.controls.get(at))
                    .map_or(Unrolling::Driver, |&(_, u)| u);
                let (merge, cont) = (self.ids.block(merge)?, self.ids.block(cont)?);
                match unrolling.literal() {
                    Some(n) => w.inst(Op::LoopMerge, &[merge, cont, unrolling.control(), n])?,
                    None => w.inst(Op::LoopMerge, &[merge, cont, unrolling.control()])?,
                }
            }
            None => {}
        }
        self.terminator(w, &b.term)
    }

    fn terminator(&self, w: &mut Writer, t: &Terminator) -> BResult<()> {
        match t {
            Terminator::Branch(b) => w.inst(Op::Branch, &[self.ids.block(*b)?]),
            Terminator::CondBranch { cond, t, f } => w.inst(
                Op::BranchConditional,
                &[self.val(*cond)?, self.ids.block(*t)?, self.ids.block(*f)?],
            ),
            Terminator::Switch {
                sel,
                default,
                cases,
            } => {
                // Each case literal is one word per 32 bits of the selector's width; the builder
                // admits only 32-bit selectors, so a literal above `u32::MAX` is refused rather
                // than truncated.
                let wide = matches!(
                    self.m.types.get(self.value_ty(*sel)?)?,
                    Type::Int { width: 64, .. }
                );
                let mut ops = vec![self.val(*sel)?, self.ids.block(*default)?];
                for (lit, b) in cases {
                    if wide {
                        ops.push(*lit as u32);
                        ops.push((*lit >> 32) as u32);
                    } else {
                        let lit = u32::try_from(*lit)
                            .map_err(|_| BuildError::Control(ControlRule::CaseLiterals))?;
                        ops.push(lit);
                    }
                    ops.push(self.ids.block(*b)?);
                }
                w.inst(Op::Switch, &ops)
            }
            Terminator::Return => w.inst(Op::Return, &[]),
            Terminator::ReturnValue(v) => w.inst(Op::ReturnValue, &[self.val(*v)?]),
            Terminator::Kill => w.inst(Op::Kill, &[]),
            Terminator::TerminateInvocation => w.inst(Op::TerminateInvocation, &[]),
            Terminator::Unreachable => w.inst(Op::Unreachable, &[]),
            Terminator::EmitMeshTasks { x, y, z, payload } => {
                let mut o = vec![self.val(*x)?, self.val(*y)?, self.val(*z)?];
                if let Some(p) = payload {
                    o.push(self.val(*p)?);
                }
                w.inst(Op::EmitMeshTasksEXT, &o)
            }
        }
    }

    /// `MemoryAccess` operands: the mask, then the operands in mask-bit order (the `Aligned`
    /// literal, then the `MakePointerAvailable` and `MakePointerVisible` scope ids).
    fn access(&self, a: MemAccess) -> BResult<Vec<u32>> {
        let flag = |on: bool, bit: u32| if on { bit } else { 0 };
        let mask = flag(a.volatile, 0x1)
            | flag(a.aligned.is_some(), 0x2)
            | flag(a.nontemporal, 0x4)
            | flag(a.make_available.is_some(), 0x8)
            | flag(a.make_visible.is_some(), 0x10)
            | flag(a.non_private, 0x20);
        if mask == 0 {
            return Ok(Vec::new());
        }
        let mut out = vec![mask];
        out.extend(a.aligned);
        if let Some(s) = a.make_available {
            out.push(self.cid(s.id)?);
        }
        if let Some(s) = a.make_visible {
            out.push(self.cid(s.id)?);
        }
        Ok(out)
    }

    /// Image operands: the mask, then the operands in mask-bit order.
    fn image_ops(&self, o: &ImageOperands) -> BResult<Vec<u32>> {
        self.storage_ops(o, TexelMemory::PRIVATE, false)
    }

    /// The image operands of a storage read or (`write`) write: [`Ser::image_ops`] with the
    /// texel operands `mem`, whose `Made` scope is `MakeTexelAvailable` on a write and
    /// `MakeTexelVisible` on a read.
    fn storage_ops(&self, o: &ImageOperands, mem: TexelMemory, write: bool) -> BResult<Vec<u32>> {
        let mut mask = 0u32;
        let mut rest = Vec::new();
        if let Some(v) = o.bias {
            mask |= 0x1;
            rest.push(self.val(v)?);
        }
        if let Some(v) = o.lod {
            mask |= 0x2;
            rest.push(self.val(v)?);
        }
        if let Some((dx, dy)) = o.grad {
            mask |= 0x4;
            rest.push(self.val(dx)?);
            rest.push(self.val(dy)?);
        }
        if let Some(v) = o.const_offset {
            mask |= 0x8;
            rest.push(self.val(v)?);
        }
        if let Some(v) = o.offset {
            mask |= 0x10;
            rest.push(self.val(v)?);
        }
        if let Some(v) = o.sample {
            mask |= 0x40;
            rest.push(self.val(v)?);
        }
        if let Some(v) = o.min_lod {
            mask |= 0x80;
            rest.push(self.val(v)?);
        }
        if let Some(s) = mem.made_scope() {
            mask |= if write { 0x100 } else { 0x200 };
            rest.push(self.cid(s.id)?);
        }
        if mem.non_private() {
            mask |= 0x400;
        }
        if mem.volatile() {
            mask |= 0x800;
        }
        if mask == 0 {
            return Ok(Vec::new());
        }
        let mut out = vec![mask];
        out.extend(rest);
        Ok(out)
    }

    /// Write `inst`; `result` is its value and the id the instruction writes it under (the
    /// value's own id, or a materialized value's instruction id).
    fn inst(
        &self,
        w: &mut Writer,
        site: (BlockId, u32),
        result: Option<(Value, u32)>,
        inst: &Inst,
    ) -> BResult<()> {
        // `[result type, result id]` of an instruction with a result.
        let head = || -> BResult<Vec<u32>> {
            let (r, id) = result.ok_or(BuildError::IdSpaceExhausted)?;
            Ok(vec![self.result_ty(r)?, id])
        };
        let v = |x: Value| self.val(x);
        let with = |mut h: Vec<u32>, rest: &[u32]| {
            h.extend_from_slice(rest);
            h
        };
        match inst {
            Inst::Load { ptr, access } => {
                let mut ops = with(head()?, &[v(*ptr)?]);
                ops.extend(self.access(*access)?);
                w.inst(Op::Load, &ops)
            }
            Inst::Store { ptr, value, access } => {
                let mut ops = vec![v(*ptr)?, v(*value)?];
                ops.extend(self.access(*access)?);
                w.inst(Op::Store, &ops)
            }
            // A copy that shares its operand's id is not written.
            Inst::Unary {
                op: UnOp::CopyObject,
                a,
            } if result.map(|(_, id)| id) == Some(v(*a)?) => Ok(()),
            Inst::Unary { op, a } => w.inst(op.to_spirv(), &with(head()?, &[v(*a)?])),
            Inst::Binary { op, a, b, .. } => {
                w.inst(op.to_spirv(), &with(head()?, &[v(*a)?, v(*b)?]))
            }
            Inst::Select { cond, t, f } => {
                w.inst(Op::Select, &with(head()?, &[v(*cond)?, v(*t)?, v(*f)?]))
            }
            Inst::BitFieldInsert {
                base,
                insert,
                offset,
                count,
            } => w.inst(
                Op::BitFieldInsert,
                &with(head()?, &[v(*base)?, v(*insert)?, v(*offset)?, v(*count)?]),
            ),
            Inst::BitFieldExtract {
                signed,
                base,
                offset,
                count,
            } => w.inst(
                if *signed {
                    Op::BitFieldSExtract
                } else {
                    Op::BitFieldUExtract
                },
                &with(head()?, &[v(*base)?, v(*offset)?, v(*count)?]),
            ),
            Inst::Convert { op, a, .. } => w.inst(op.to_spirv(), &with(head()?, &[v(*a)?])),
            Inst::CompositeConstruct { parts } => {
                let mut ops = head()?;
                for p in parts.iter() {
                    ops.push(v(*p)?);
                }
                w.inst(Op::CompositeConstruct, &ops)
            }
            Inst::CompositeExtract { composite, indices } => {
                let mut ops = with(head()?, &[v(*composite)?]);
                ops.extend_from_slice(indices);
                w.inst(Op::CompositeExtract, &ops)
            }
            Inst::CompositeInsert {
                object,
                composite,
                indices,
            } => {
                let mut ops = with(head()?, &[v(*object)?, v(*composite)?]);
                ops.extend_from_slice(indices);
                w.inst(Op::CompositeInsert, &ops)
            }
            Inst::VectorExtractDynamic { vector, index } => w.inst(
                Op::VectorExtractDynamic,
                &with(head()?, &[v(*vector)?, v(*index)?]),
            ),
            Inst::VectorInsertDynamic {
                vector,
                component,
                index,
            } => w.inst(
                Op::VectorInsertDynamic,
                &with(head()?, &[v(*vector)?, v(*component)?, v(*index)?]),
            ),
            Inst::VectorShuffle { a, b, components } => {
                let mut ops = with(head()?, &[v(*a)?, v(*b)?]);
                ops.extend_from_slice(components);
                w.inst(Op::VectorShuffle, &ops)
            }
            // Written once per type among the declarations.
            Inst::Undef => Ok(()),
            Inst::AccessChain { base, indices, .. } => {
                let mut ops = with(head()?, &[v(*base)?]);
                for i in indices.iter() {
                    ops.push(v(*i)?);
                }
                w.inst(Op::AccessChain, &ops)
            }
            Inst::ArrayLength { ptr, member } => {
                w.inst(Op::ArrayLength, &with(head()?, &[v(*ptr)?, *member]))
            }
            Inst::Call { func, args } => {
                // A void call still has a result id in SPIR-V; the typed IR gives it no value,
                // so its id comes from `Ids::void_calls` and its type is the callee's `void`.
                let mut ops = match result {
                    Some((r, id)) => vec![self.result_ty(r)?, id],
                    None => {
                        let fd = self
                            .m
                            .functions
                            .get(func.index())
                            .ok_or(BuildError::Dangling(Dangling::Func(*func)))?;
                        let id = self
                            .ids
                            .void_calls
                            .get(&(site.0 .0, site.1))
                            .copied()
                            .ok_or(BuildError::Dangling(Dangling::Func(*func)))?;
                        vec![self.ty(fd.ret)?, id]
                    }
                };
                ops.push(self.ids.func(*func)?);
                for a in args.iter() {
                    ops.push(v(*a)?);
                }
                w.inst(Op::FunctionCall, &ops)
            }
            Inst::AtomicLoad { ptr, scope, sem } => w.inst(
                Op::AtomicLoad,
                &with(head()?, &[v(*ptr)?, self.cid(scope.id)?, self.cid(sem.id)?]),
            ),
            Inst::AtomicStore {
                ptr,
                scope,
                sem,
                value,
            } => w.inst(
                Op::AtomicStore,
                &[v(*ptr)?, self.cid(scope.id)?, self.cid(sem.id)?, v(*value)?],
            ),
            Inst::Atomic {
                op,
                ptr,
                scope,
                sem,
                value,
            } => w.inst(
                op.to_spirv(),
                &with(
                    head()?,
                    &[v(*ptr)?, self.cid(scope.id)?, self.cid(sem.id)?, v(*value)?],
                ),
            ),
            Inst::AtomicCompareExchange {
                ptr,
                scope,
                equal,
                unequal,
                value,
                comparator,
            } => w.inst(
                Op::AtomicCompareExchange,
                &with(
                    head()?,
                    &[
                        v(*ptr)?,
                        self.cid(scope.id)?,
                        self.cid(equal.id)?,
                        self.cid(unequal.id)?,
                        v(*value)?,
                        v(*comparator)?,
                    ],
                ),
            ),
            Inst::ImageTexelPointer {
                image,
                coord,
                sample,
            } => w.inst(
                Op::ImageTexelPointer,
                &with(head()?, &[v(*image)?, v(*coord)?, v(*sample)?]),
            ),
            Inst::ControlBarrier { exec, mem, sem } => w.inst(
                Op::ControlBarrier,
                &[self.cid(exec.id)?, self.cid(mem.id)?, self.cid(sem.id)?],
            ),
            Inst::MemoryBarrier { mem, sem } => {
                w.inst(Op::MemoryBarrier, &[self.cid(mem.id)?, self.cid(sem.id)?])
            }
            Inst::SampledImage { image, sampler } => w.inst(
                Op::SampledImage,
                &with(head()?, &[v(*image)?, v(*sampler)?]),
            ),
            Inst::Image(a) => w.inst(Op::Image, &with(head()?, &[v(*a)?])),
            Inst::ImageSample {
                form,
                sampled_image,
                coord,
                dref,
                ops,
            } => {
                let mut o = with(head()?, &[v(*sampled_image)?, v(*coord)?]);
                if let Some(d) = dref {
                    o.push(v(*d)?);
                }
                o.extend(self.image_ops(ops)?);
                w.inst(form.to_spirv(), &o)
            }
            Inst::ImageFetch {
                image,
                coord,
                ops,
                sparse,
            } => {
                let mut o = with(head()?, &[v(*image)?, v(*coord)?]);
                o.extend(self.image_ops(ops)?);
                let op = if *sparse {
                    Op::ImageSparseFetch
                } else {
                    Op::ImageFetch
                };
                w.inst(op, &o)
            }
            Inst::ImageGather {
                sampled_image,
                coord,
                component_id,
                ops,
                sparse,
                ..
            } => {
                let mut o = with(
                    head()?,
                    &[v(*sampled_image)?, v(*coord)?, self.cid(*component_id)?],
                );
                o.extend(self.image_ops(ops)?);
                let op = if *sparse {
                    Op::ImageSparseGather
                } else {
                    Op::ImageGather
                };
                w.inst(op, &o)
            }
            Inst::ImageDrefGather {
                sampled_image,
                coord,
                dref,
                ops,
                sparse,
            } => {
                let mut o = with(head()?, &[v(*sampled_image)?, v(*coord)?, v(*dref)?]);
                o.extend(self.image_ops(ops)?);
                let op = if *sparse {
                    Op::ImageSparseDrefGather
                } else {
                    Op::ImageDrefGather
                };
                w.inst(op, &o)
            }
            Inst::ImageRead {
                image,
                coord,
                ops,
                sparse,
                mem,
            } => {
                let mut o = with(head()?, &[v(*image)?, v(*coord)?]);
                o.extend(self.storage_ops(ops, *mem, false)?);
                let op = if *sparse {
                    Op::ImageSparseRead
                } else {
                    Op::ImageRead
                };
                w.inst(op, &o)
            }
            Inst::ImageSparseTexelsResident(code) => {
                w.inst(Op::ImageSparseTexelsResident, &with(head()?, &[v(*code)?]))
            }
            Inst::ImageWrite {
                image,
                coord,
                texel,
                ops,
                mem,
            } => {
                let mut o = vec![v(*image)?, v(*coord)?, v(*texel)?];
                o.extend(self.storage_ops(ops, *mem, true)?);
                w.inst(Op::ImageWrite, &o)
            }
            Inst::ImageQuery { query, image } => {
                let mut o = with(head()?, &[v(*image)?]);
                if let ImageQuery::SizeLod(lod) = query {
                    o.push(v(*lod)?);
                }
                w.inst(query.to_spirv(), &o)
            }
            Inst::ImageQueryLod {
                sampled_image,
                coord,
            } => w.inst(
                Op::ImageQueryLod,
                &with(head()?, &[v(*sampled_image)?, v(*coord)?]),
            ),
            Inst::Derivative { op, a } => w.inst(op.to_spirv(), &with(head()?, &[v(*a)?])),
            Inst::Group { scope, op } => {
                let mut o = with(head()?, &[self.cid(scope.id)?]);
                self.group_operands(&mut o, op)?;
                w.inst(op.to_spirv(), &o)
            }
            Inst::Glsl { op, args, .. } => {
                let set = self.ids.glsl.ok_or(BuildError::IdSpaceExhausted)?;
                let mut o = with(head()?, &[set, op.to_spirv() as u32]);
                for a in args.iter() {
                    o.push(v(*a)?);
                }
                w.inst(Op::ExtInst, &o)
            }
            Inst::BeginInterlock => w.inst(Op::BeginInvocationInterlockEXT, &[]),
            Inst::EndInterlock => w.inst(Op::EndInvocationInterlockEXT, &[]),
            Inst::DemoteToHelperInvocation => w.inst(Op::DemoteToHelperInvocation, &[]),
            Inst::IsHelperInvocation => w.inst(Op::IsHelperInvocationEXT, &head()?),
            Inst::SetMeshOutputs {
                vertices,
                primitives,
            } => w.inst(Op::SetMeshOutputsEXT, &[v(*vertices)?, v(*primitives)?]),
            Inst::RayQuery(op) => {
                let o = match op {
                    RayQueryInst::Initialize { .. } | RayQueryInst::Confirm { .. } => {
                        let mut o = Vec::with_capacity(8);
                        op.for_each_value(|x| o.push(x));
                        o.iter().map(|x| v(*x)).collect::<BResult<Vec<u32>>>()?
                    }
                    RayQueryInst::Proceed { query } => with(head()?, &[v(*query)?]),
                    RayQueryInst::Get { query, which, .. } => {
                        with(head()?, &[v(*query)?, self.cid(*which)?])
                    }
                };
                w.inst(op.to_spirv(), &o)
            }
        }
    }

    fn group_operands(&self, o: &mut Vec<u32>, op: &GroupInst) -> BResult<()> {
        let v = |x: Value| self.val(x);
        match op {
            GroupInst::Elect => {}
            GroupInst::All(a)
            | GroupInst::Any(a)
            | GroupInst::BroadcastFirst(a)
            | GroupInst::Ballot(a) => o.push(v(*a)?),
            GroupInst::BallotBitExtract { ballot, index } => {
                o.push(v(*ballot)?);
                o.push(v(*index)?);
            }
            GroupInst::Shuffle { value, x, .. } => {
                o.push(v(*value)?);
                o.push(v(*x)?);
            }
            GroupInst::Arith { group, value, .. } => {
                o.push(group.to_spirv() as u32);
                o.push(v(*value)?);
                if let GroupOperation::ClusteredReduce { id, .. } = group {
                    o.push(self.cid(*id)?);
                }
            }
            GroupInst::QuadBroadcast { value, index } => {
                o.push(v(*value)?);
                o.push(v(*index)?);
            }
            GroupInst::QuadSwap { value, dir_id, .. } => {
                o.push(v(*value)?);
                o.push(self.cid(*dir_id)?);
            }
        }
        Ok(())
    }
}
