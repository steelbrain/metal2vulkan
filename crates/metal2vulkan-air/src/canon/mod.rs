//! Canonicalization: entry selection, linking, function-constant baking, SCCP, inlining.
//!
//! [`specialize_linked`] turns a decoded module, its ABI, a [`SpecRequest`], and the pipeline's
//! linked functions into one [`Entry`] ([`specialize`] links none): a
//! module with exactly one defined function, constants baked, pointer casts erased, conditional
//! constants propagated, and everything the entry does not reach pruned. Each step maps valid IR
//! to valid IR and the result is verified again. In order:
//!
//! 1. **Entry selection** by name among the ABI's entry points.
//! 2. **Linking**: each visible-function reference stub whose name one of the pipeline's linked
//!    functions has is resolved. That function is imported with everything it reaches (its own
//!    references resolved in turn) and the stub's calls are redirected to it, so inlining takes it
//!    like any other callee ([`LinkedModule`]). An entry that takes a visible-function table also
//!    imports every function a table can hold.
//! 3. **Function-constant binding**: supplied values become their globals' initializers; unsupplied
//!    constants stay undefined.
//! 4. **Static-initializer evaluation**: the `air.static_init` functions run over constants and
//!    their stores become constant initializers; a global computed from an unsupplied constant is
//!    as undefined as the constant.
//! 5. **Pointer erasure**: casts between identical erased pointer types disappear, so typed- and
//!    opaque-pointer bitcode of one program canonicalize identically. Each defined function is
//!    erased in place before inlining, so the inlined body is built erased.
//! 6. **Inlining**: every direct call to a defined function is inlined into the entry, after a
//!    memoized size pre-pass checked against `Limit::InlinedInstructions`; a call cycle is
//!    `Unsupported{Recursion}`. Between the pre-pass and building the inlined body, a value use of
//!    an unsupplied constant that the entry's own entry block makes certain is refused as SCCP
//!    would refuse it. **Devirtualization** then makes the inlined body's calls through constant
//!    function addresses direct, reads visible-function tables as buffers of words, and makes each
//!    call through a table a switch over the functions it may call; the direct calls this makes are
//!    inlined in turn, round after round, until none is made.
//! 7. **SCCP** over the inlined entry, including `air.is_function_constant_defined` and loads of
//!    bound constants; unreachable blocks are deleted. A value use of an unsupplied constant that
//!    survives is `Error::NeedsInput(FunctionConstant)`.
//! 8. **Post-check**: no unknown intrinsic, indirect call, or call to an undefined non-intrinsic
//!    function survives, and no reachable branch tests `undef`. A table call's refused candidate,
//!    a table call no linked function answers, and a function pointer used as a value are refused
//!    here, as stubs are. Indirect calls, calls to functions of a dynamic library, and ray
//!    intersection are refused as their typed features, not implemented yet, naming the callee; a
//!    call through a function's constant address at another type is classified by that function. A
//!    call to a visible-function reference linking did not resolve is `NeedsInput(LinkedFunction)`,
//!    or the reason the linked function of its name cannot resolve it. The one call to a
//!    declaration that stays is a post-tessellation entry's control-point fetch
//!    (`air.patch_control_point_function`, whose body Metal synthesizes from the vertex descriptor)
//!    with the entry's control-point input as its pointer operand; `lower::tessellation` lowers it.
//! 9. **Arm speculation**: a branch whose short arms Metal's backend runs in every lane of the
//!    branch's block, and whose arms hold a group operation that answers differently there,
//!    becomes a jump into its arms with a `select` for each value they join with (`speculate`).
//!    **Rasterization-rate maps**: each `float2` `air.map_*_coordinates` call becomes Apple's GPU
//!    decoder of Metal's parameter data (`rate_map_data`) in plain AIR: word loads through the
//!    call's map pointer and Apple's arithmetic. **AIR 2.7 simdgroup-matrix block loads and
//!    stores** take their AIR 2.8 form, their extent, stride and origin built from the elements per
//!    row, the origin and the constant transpose flag (`legacy_matrix`). **Global built-in
//!    variables**: each `air.get_*` read of a program-scope built-in becomes a read of a built-in
//!    parameter appended to the entry (`global_builtin`).
//! 10. **ABI resolution and pruning**: predicate-gated parameters and outputs that are off
//!     disappear from the entry point, a read of an absent argument takes its absent value (the
//!     null reference of a resource argument) or is refused, global-valued ABI integers become
//!     constants, what nothing observes is deleted (a value no side effect, branch, or output
//!     depends on, a write into a stack allocation no load reaches), and the module keeps only what
//!     the entry reaches, renumbered.
//!
//! Inlining runs before SCCP: SCCP over the one inlined body sees each call's constant arguments,
//! which per-function SCCP could not, and deletes the callees' dead paths with the caller's.

mod async_copy;
mod command_writes;
mod const_loads;
mod devirt;
#[cfg(test)]
mod devirt_tests;
pub mod discard_sink;
#[cfg(test)]
mod discard_sink_tests;
#[cfg(test)]
mod dynamic_tests;
mod erase;
mod fc;
mod fold;
pub mod function_table;
mod global_builtin;
mod inline;
mod interface;
mod legacy_matrix;
mod link;
#[cfg(test)]
mod link_tests;
mod matmul2d;
mod out;
mod pieces;
mod prune;
mod rate_map;
pub mod rate_map_data;
mod reads;
mod recursion;
mod renumber;
mod rewrite;
mod routine;
mod sccp;
mod speculate;
#[cfg(test)]
mod speculate_tests;
mod static_init;
mod table_copy;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::abi::{AbiInt, Access, AirAbi, BufferArg, CompileOptions, EntryPoint};
use crate::abi::{LanguageVersion, ParamBinding, ParamRole, PatchInput, SamplerState, Stage};
use crate::abi::{StaticSampler, StructInfo};
use crate::budget::Meter;
use crate::error::{CanonError, DiagName, InputError, InputErrorKind, IntrinsicError, Location};
use crate::error::{Error, Feature, MissingInput, RequestError, Unsupported};
use crate::graph::{Cfg, DomTree};
use crate::intrinsics::{self, AirIntrinsic};
use crate::ir::{AddrSpace, BlockId, Body, Callee, DataLayout, FuncId, GlobalId, Idx};
use crate::ir::{Const, ConstTable, ModuleParts, Type, TypeId, TypeTable, Value};
use crate::ir::{InstKind, Module};
use crate::metal::DataType;

pub use command_writes::CommandWrites;
pub use link::{imports_dynamically, DynamicLinks, DynamicModule};
use link::{Item, Stub, Stubs};
pub use link::{LinkedCode, LinkedModule, LinkedName};
pub use reads::{constants_read, ConstantRead};
use rewrite::invariant;

/// One function-constant value: its Metal data type and its packed little-endian bytes (a
/// `float3` is 12 bytes), as `MTLFunctionConstantValues` takes them.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConstValue {
    /// The value's Metal data type, which must be the declared constant's.
    pub ty: DataType,
    /// The value's packed little-endian bytes.
    pub bytes: Box<[u8]>,
}

/// What to specialize: an entry point by name and the function-constant values supplied for it
/// (by index). A declared constant without a value is undefined.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpecRequest {
    /// The entry point's function name.
    pub entry: String,
    /// The supplied function-constant values, by each constant's index
    /// (`[[function_constant(n)]]`).
    pub constants: BTreeMap<u32, ConstValue>,
    /// The pipeline's `maxCallStackDepth`: it sizes the explicit call stack of a recursion through
    /// visible functions. 0 is taken as 1, Metal's default.
    pub call_stack_depth: u32,
}

/// The constexpr samplers of one linked function that a pipeline state numbers
/// ([`Entry::linked_sampler_groups`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkedSamplerGroup {
    /// How many of [`Entry::static_samplers`]' last entries are this group's, in order (those of
    /// its samplers the entry reaches; none for a function linking did not import).
    pub kept: usize,
    /// The sampler states the pipeline state numbers for the function, in its module's order
    /// (`!air.sampler_states`, the order of their first use).
    pub states: Vec<SamplerState>,
    /// A dynamic library symbol's, whose numbering is not measured.
    pub library: bool,
}

/// One specialized entry point.
#[derive(Clone, Debug)]
pub struct Entry {
    /// Exactly one defined function (`func`), the declarations it calls, and the globals and
    /// constants it reaches. Pointers are `Ptr{space}` only.
    pub module: Module,
    /// The entry point's function in `module`.
    pub func: FuncId,
    /// The entry point, with gated-off parameters and outputs removed, predicates resolved, and
    /// every ABI integer a constant. The body reads no gated-off parameter but those of
    /// [`Entry::absent_handles`].
    pub ep: EntryPoint,
    /// The texture and sampler parameters (texture tables included) a function constant removed
    /// that the body still reads other than as a texture call's own operand: it stores one, puts
    /// it in an aggregate, loads an element of a table, or merges one with another handle. Each
    /// binds no slot, and its integers stay unresolved; lowering follows its handle as it follows a
    /// present one's, and each texture call the handle reaches is an absent read.
    pub absent_handles: Vec<ParamBinding>,
    /// The `air.struct_type_info` layouts `ep`'s [`crate::abi::StructRef`]s index.
    pub structs: Vec<StructInfo>,
    /// `None` when the module does not state it.
    pub language: Option<LanguageVersion>,
    /// The module's `!air.compile_options`: whether it was compiled with fast math, which decides
    /// how Metal rounds an extended-range blend.
    pub compile: CompileOptions,
    /// The module's data layout.
    pub layout: DataLayout,
    /// The control-flow graph of `func`'s body.
    pub cfg: Cfg<BlockId>,
    /// The dominator tree of `cfg`.
    pub dom: DomTree<BlockId>,
    /// The constexpr samplers the entry reaches, by their global in `module`: the module's own,
    /// in its order (`!air.sampler_states`), then those linking brought
    /// ([`Entry::linked_sampler_groups`]).
    pub static_samplers: Vec<(GlobalId, SamplerState)>,
    /// The constexpr samplers of each linked function a pipeline state numbers after the
    /// module's own: one group per linking import that brought any, in the order of the imports,
    /// then one per function the pipeline can place in a visible-function table that linking did
    /// not import (nothing names it, and the entry reaches no table). Metal numbers every constexpr
    /// sampler of a function a table can hold, called or not.
    pub linked_sampler_groups: Vec<LinkedSamplerGroup>,
    /// Whether a linked function may have constexpr samplers that Metal's pipeline numbers and
    /// [`Entry::linked_sampler_groups`] lacks: a function a table can hold that linking did not
    /// import could not be planned. A function of `privateFunctions` is numbered as far as the
    /// entry reaches it, as Metal numbers it.
    pub unnumbered_linked_samplers: bool,
    /// Whether a function the pipeline can place in a table (a linked function with a handle) may
    /// be an intersection function: one whose module did not decode. A module that holds an
    /// intersection function does not decode (`!air.intersection` is not read), so a decoded
    /// module's are visible functions, which an intersection function table cannot hold.
    pub intersection_candidates: bool,
    /// Whether the module declares function constants (`air.function_constants`). Metal lays
    /// out the threadgroup objects of the specialized function, whose references a constant may
    /// remove, so an integer threadgroup address is not translated in such a module.
    pub function_constants: bool,
    /// The Metal buffer indices of the visible-function-table parameters read as buffers of
    /// words: `ep` states each as a read-only device `Buffer`, and the interface says the slot
    /// holds a table.
    pub function_tables: BTreeSet<u32>,
    /// The barriers that make the order of a simdgroup's lockstep explicit: empty from
    /// canonicalization; `lower::simd::lockstep` fills it when it inserts them.
    pub lockstep: Lockstep,
    /// The call stack of the entry's recursion through visible functions: `None` when the entry has
    /// none.
    pub call_stack: Option<CallStack>,
    /// Whether the body holds a branch whose arms hold a quad or simdgroup operation that Metal's
    /// backend makes answer over the lanes of the branch's block when it speculates the arms, where
    /// no measurement says whether it speculates them (an arm of an instruction, size or shape no
    /// probe covers). Such a branch keeps the AIR's lanes; Metal's may be the block's.
    pub unmeasured_speculation: bool,
}

/// An entry's explicit call stack: what its recursion through visible functions runs on. Its stacks
/// are allocas of the entry, whose bytes the memory plan counts with its other private memory
/// (`MemoryPlan::private_bytes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallStack {
    /// The `maxCallStackDepth` the stacks are sized for, at least 1.
    pub depth: u32,
}

/// The threadgroup and simdgroup barriers `lower::simd::lockstep` inserted into an entry's body.
///
/// Apple GPUs run a simdgroup's 32 threads in lockstep, so two conflicting threadgroup accesses
/// of different threads of one simdgroup between two barriers run in program order there; Vulkan
/// orders nothing between barriers. Each inserted barrier orders such a pair where the program
/// left it to lockstep.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Lockstep {
    /// The inserted barrier calls, sorted.
    pub barriers: Vec<crate::ir::InstId>,
    /// Some inserted simdgroup barrier stands where the lanes of a subgroup are not known to
    /// reach it together, so it orders its pair only on emulated simdgroups, whose predicated
    /// walk takes every thread of the workgroup through it: the kernel takes the `Emulated` tier.
    /// Never set for a device whose lockstep orders those pairs (`device`), nor where a subgroup
    /// barrier in divergent flow orders the threads that run it together (maximal reconvergence on
    /// 32-lane subgroups).
    pub emulated: bool,
    /// On a device that runs a simdgroup's 32 threads in lockstep
    /// (`VkProperty::SimdgroupLockstep`): the barrier positions that only emulated simdgroups would
    /// make exact, left out because the device's lockstep orders their pairs as Apple's does. The
    /// module then requires the property. 0 elsewhere.
    pub device: u32,
}

impl Lockstep {
    /// Whether instruction `i` is an inserted barrier.
    pub fn inserted(&self, i: crate::ir::InstId) -> bool {
        self.barriers.binary_search(&i).is_ok()
    }
}

impl Entry {
    /// Whether the entry may encode, reset or copy indirect commands: `false` only where its
    /// metadata names no indirect command buffer it takes and its body proves it touches none.
    ///
    /// It is `true` where the metadata names one (`EntryPoint::reaches_command_buffers`, whatever
    /// the body does with it), and where the body calls an indirect-command operation (an
    /// `air.*_command` of [`intrinsics::IndirectCommand`]) on any command operand: AIR's command
    /// buffer is a device pointer, so a pointer the body casts from another argument's bytes, loads
    /// from a buffer no layout describes, or makes from an integer reaches the operation as the
    /// metadata's would, with no cast left after pointer erasure to tell them apart. Metal gives a
    /// command buffer's records no address, so an operation is the only way to them. A call the
    /// walk cannot see into counts too: an indirect call, an intrinsic outside the closed set, or a
    /// call to a declaration other than the entry's control-point functions. Canonicalization
    /// refuses each of those (its post-check), so the entry it returns holds none, but the answer
    /// does not rest on that. Linear in the module's instructions, each visited once.
    pub fn may_encode_indirect_commands(&self) -> bool {
        self.ep.reaches_command_buffers(&self.structs) || calls_commands(&self.module, &self.ep)
    }

    /// What the entry's indirect-command operations may write into the records they reach
    /// ([`CommandWrites`]): the operations its bodies call and the primitive types their draws
    /// pass, or [`CommandWrites::ANY`] where a body makes a call the walk cannot see into, as
    /// [`Entry::may_encode_indirect_commands`] counts one. A command buffer the metadata names
    /// and no operation reaches adds nothing: an operation is the only way to its records. So
    /// it is empty wherever `may_encode_indirect_commands` answers `false`. Linear in the
    /// module's instructions, each visited once.
    pub fn command_writes(&self) -> CommandWrites {
        command_writes::command_writes(&self.module, &self.ep)
    }

    /// Whether the entry may write a texture: `false` only where its metadata names no texture
    /// it takes with write access and its body proves it writes none.
    ///
    /// It is `true` where the metadata names a texture it takes declared `access::write` or
    /// `access::read_write`, as a parameter or as a member of a buffer it takes at any depth
    /// (`EntryPoint::takes_writable_textures`, whatever the body does with it), and where the
    /// body calls a texture operation that stores a texel (a `write`, an imageblock slice
    /// written to a texture, or an atomic other than a load) on any handle: a parameter, an
    /// argument buffer's or a heap's handle, a resource ID, or one the body selects or loads,
    /// since the operation is the only way to a texture's texels whatever the handle's origin. A call the walk cannot see into counts too, as for
    /// [`Entry::may_encode_indirect_commands`]: an indirect call, an intrinsic outside the closed
    /// set, or a call to a declaration other than the entry's control-point functions. Writes to
    /// the bytes of a buffer a texture views, color attachment outputs, and imageblock or tile
    /// stores are no texture operation and are not counted. Linear in the module's
    /// instructions, each visited once.
    pub fn may_write_textures(&self) -> bool {
        self.ep.takes_writable_textures(&self.structs)
            || calls(&self.module, &self.ep, writes_texture)
    }
}

/// Whether a function body of `module` makes a call that may reach an indirect command buffer
/// ([`Entry::may_encode_indirect_commands`]): an indirect-command operation, or a call
/// [`calls`] cannot see into.
fn calls_commands(module: &Module, ep: &EntryPoint) -> bool {
    calls(module, ep, |op| {
        matches!(op, AirIntrinsic::IndirectCommand(_))
    })
}

/// Whether `op` stores a texel into a texture ([`Entry::may_write_textures`]): a `write`, an
/// imageblock slice written to a texture, or a texture atomic other than a load (an atomic load
/// only reads). Every texture operation is named, so a new one takes a decision here.
fn writes_texture(op: &AirIntrinsic) -> bool {
    use intrinsics::{AtomicOp, TexOp};
    let AirIntrinsic::Texture(t) = op else {
        return false;
    };
    match t.op {
        TexOp::Write | TexOp::WriteImageblockSlice => true,
        TexOp::Atomic(a) => a != AtomicOp::Load,
        TexOp::Sample { .. }
        | TexOp::SampleGrad { .. }
        | TexOp::Gather { .. }
        | TexOp::Read
        | TexOp::Fence
        | TexOp::Query(_)
        | TexOp::IsNull
        | TexOp::GetNull
        | TexOp::CalculateLod { .. } => false,
    }
}

/// Whether a function body of `module` calls an `air.*` intrinsic `wanted` accepts, or makes a
/// call the walk cannot see into: an indirect call, an intrinsic outside the closed set, or a
/// call to a declaration that is not one of `ep`'s control-point functions. A defined callee's
/// body is a body of `module`, walked as the entry's is. Every instruction is visited once.
fn calls(module: &Module, ep: &EntryPoint, wanted: impl Fn(&AirIntrinsic) -> bool) -> bool {
    let declared = |f: FuncId| module.func(f).is_none_or(|f| f.body.is_none());
    module
        .parts()
        .funcs
        .iter()
        .filter_map(|(_, f)| f.body.as_ref())
        .flat_map(|body| body.insts.values())
        .any(|inst| match &inst.kind {
            InstKind::Call(call) => match call.callee {
                Callee::Air { op, .. } => wanted(&op),
                Callee::Indirect { .. } | Callee::UnknownIntrinsic(_) => true,
                Callee::Llvm { .. } => false,
                Callee::Direct(f) => declared(f) && !ep.is_control_point_function(f),
            },
            _ => false,
        })
}

/// Specialize `module` (with its decoded `abi`) for `req`, linking no function.
pub fn specialize(
    module: Module,
    abi: &AirAbi,
    req: &SpecRequest,
    m: &mut Meter,
) -> Result<Entry, Error> {
    specialize_linked(module, abi, req, &[], m)
}

/// Specialize `module` (with its decoded `abi`) for `req`, resolving its visible-function
/// references against the pipeline's `linked` functions.
pub fn specialize_linked(
    module: Module,
    abi: &AirAbi,
    req: &SpecRequest,
    linked: &[LinkedModule],
    m: &mut Meter,
) -> Result<Entry, Error> {
    specialize_dynamic(module, abi, req, (linked, &DynamicLinks::default()), m)
}

/// [`specialize_linked`], resolving the module's dynamic-library imports against the stage's
/// `dynamic` libraries too.
pub fn specialize_dynamic(
    module: Module,
    abi: &AirAbi,
    req: &SpecRequest,
    (linked, dynamic): (&[LinkedModule], &DynamicLinks),
    m: &mut Meter,
) -> Result<Entry, Error> {
    // A linkage signature describes how a closed module was produced: a stitched module defines
    // every stitched function, and a linked-constants module was specialized with the constants it
    // lists and declares none. What the module still needs from outside is what any module needs
    // (constants, linked functions), refused or bound as usual. A linked-constants module that
    // still declares constants has no reading.
    let linked_constants = abi
        .linkage_signature
        .is_some_and(|s| s.kind == crate::abi::LinkageSignatureKind::LinkedConstants);
    if linked_constants && !abi.function_constants.is_empty() {
        let at = Location::metadata(None, None, Some("air.signature"));
        return Err(Unsupported::not_implemented(Feature::Linking).at(at).into());
    }
    let ep = select(&module, abi, &req.entry)?;
    let func = ep.func;
    let name = diag_name(&module, func);
    let (pending, mut fcs) = fc::check(&module, abi, &req.constants, m)?;

    // Linking: each stub a linked function resolves calls its imported definition, and an entry
    // that can call through a visible-function table, one it takes or one an argument buffer it
    // takes holds, imports what a table can hold. Each dynamic-library import a library exports is
    // resolved to the library's definition.
    let tables = table_params(ep);
    let intersection_candidates = linked
        .iter()
        .any(|l| l.code.is_err() && l.functions.iter().any(|f| f.handle.is_some()));
    let (mut parts, mut stubs, linked_samplers, candidates) = link::link(
        module.into_parts(),
        abi,
        (linked, dynamic),
        abi.reaches_function_tables(ep),
        m,
    )?;
    fc::bind(
        &parts.types,
        &mut parts.consts,
        &mut parts.globals,
        &pending,
        m,
    )?;

    // Static initializers: their stores become constant initializers; a global computed from an
    // unsupplied constant is tainted like the constant itself.
    let evaluated = static_init::evaluate(&mut parts, &fcs, static_init::Unevaluable::Refuse, m)?;
    for (g, index) in evaluated.derived {
        m.charge(std::mem::size_of::<(GlobalId, sccp::FcState)>() * 2)?;
        fcs.insert(
            g,
            sccp::FcState {
                index,
                supplied: false,
                decl: false,
            },
        );
    }

    // Recursion through visible functions: each recursive part of the call graph becomes an
    // explicit call stack, so inlining and devirtualization take none of its calls. Each function
    // whose address is taken gets the word devirtualization carries it as: a table's handle, or a
    // word of the stage's own past every handle.
    let first_own = linked
        .iter()
        .flat_map(|l| &l.functions)
        .filter_map(|f| f.handle)
        .max()
        .map_or(Some(function_table::FIRST_FUNCTION_WORD), |h| {
            h.checked_add(1)
        })
        .ok_or_else(|| invariant("function handles past the word space"))?;
    let recursion::Prepared {
        mut recursion,
        words,
    } = recursion::prepare(
        &mut parts,
        &stubs,
        func,
        (&candidates, first_own),
        req.call_stack_depth,
        m,
    )?;
    let groups = recursion
        .as_ref()
        .map(|r| r.groups().clone())
        .unwrap_or_default();
    let enters = recursion
        .as_ref()
        .map(|r| r.enters().clone())
        .unwrap_or_default();

    // Pointer erasure, then inlining: every defined function is erased in place, one at a time, so
    // the inliner copies erased bodies and the inlined body, as large as the entry's whole call
    // tree, is never rebuilt to erase it (that copy beside the inlined body was a translation's
    // ledger peak).
    let emap = erase::erase_consts(&mut parts.consts, m)?;
    erase::erase_globals(parts.globals.values_mut(), &emap);
    erase_functions(&mut parts, &emap, m)?;
    // The entry block always runs: a value use there of an unsupplied constant is SCCP's first
    // refusal, known before the inlined body is built.
    let entry_refusal = |consts: &ConstTable, m: &mut Meter| {
        let body = parts
            .funcs
            .get(func)
            .and_then(|f| f.body.as_ref())
            .ok_or_else(|| invariant("the entry has no body"))?;
        let inlined = |k: &InstKind| inline::inlined_callee(&parts.funcs, k).is_some();
        match sccp::entry_block_refusal(body, consts, &fcs, &inlined, m)? {
            Some(index) => Err(Error::NeedsInput(MissingInput::FunctionConstant { index })),
            None => Ok(()),
        }
    };
    let inlined = inline::inline(
        &parts.funcs,
        (&parts.types, &parts.symbols),
        &mut parts.consts,
        func,
        m,
        entry_refusal,
    );
    let body = inlined.and_then(|inlined| erased_entry(&parts, func, inlined, &emap, m));
    // Devirtualization: calls through constant function addresses and visible-function tables
    // become direct calls, which inlining then takes, until none is made.
    let body = body.and_then(|body| {
        let round = Round {
            func,
            emap: &emap,
            tables: &tables,
            candidates: &candidates,
            linked,
            name: &name,
            groups: &groups,
            enters: &enters,
            words: &words,
            fcs: &fcs,
        };
        let (mut body, mut rewritten) = inline_rounds(body, &mut parts, &mut stubs, &round, m)?;
        // Each root of a recursive part gets its copies, inlined and devirtualized in turn, until
        // no root is new; then each becomes its machine.
        if let Some(rec) = recursion.as_mut() {
            let params: Vec<TypeId> =
                match parts.funcs.get(func).and_then(|f| parts.types.get(f.ty)) {
                    Some(Type::Function { params, .. }) => params.to_vec(),
                    _ => return Err(invariant("the entry's type is not a function")),
                };
            while let Some(next) = rec.islands(&body, &mut parts, (&params, &emap), m)? {
                body.release(m);
                (body, rewritten) = reinline(next, &mut parts, &mut stubs, &round, m)?;
            }
            body = rec.assemble(body, &mut parts, m)?;
        }
        Ok((body, rewritten))
    });
    m.release(emap.len() * std::mem::size_of::<crate::ir::ConstId>());
    let (body, rewritten) = body?;
    let call_stack = recursion
        .as_ref()
        .and_then(recursion::Recursion::call_stack);
    // `matmul2d` and tensor objects: each call modeled gets a body of its own, inlined, before SCCP
    // folds the sizes of the storage it allocates.
    let (body, lanes) = matmul2d::expand(body, &mut parts, ep, m)?;
    // A kernel's simdgroup asynchronous copies complete when their calls return, shared by the
    // simdgroup's lanes, and each wait is a simdgroup barrier.
    let body = async_copy::expand(body, &mut parts, ep, m)?;

    // SCCP. The body it read is freed as soon as it has run.
    let solved = {
        let mut cx = sccp::Context {
            folder: fold::Folder {
                types: &parts.types,
                consts: &mut parts.consts,
                held: 0,
            },
            globals: &parts.globals,
            fcs: &fcs,
            name: name.clone(),
        };
        sccp::run(&body, &mut cx, m)?
    };
    body.release(m);
    let body = solved;
    // A pointer loaded from a field of a constant global is the field's constant: a `constexpr
    // sampler` that SCCP's folded select picked is its sampler state. SCCP runs again over a body
    // that changed, to fold what those constants decide.
    let (body, folded) = const_loads::fold(body, &mut parts, m)?;
    let body = if folded {
        let solved = {
            let mut cx = sccp::Context {
                folder: fold::Folder {
                    types: &parts.types,
                    consts: &mut parts.consts,
                    held: 0,
                },
                globals: &parts.globals,
                fcs: &fcs,
                name: name.clone(),
            };
            sccp::run(&body, &mut cx, m)?
        };
        body.release(m);
        solved
    } else {
        body
    };
    matmul2d::refuse_left(&body, &parts, &ep.stage, &name, m)?;
    post_check(&body, &parts, &stubs, &name, &control_points(ep), m)?;
    // A branch's short arms that Metal's backend speculates, where a group operation in them
    // answers over the lanes of the branch's block.
    let (body, unmeasured_speculation) = {
        let f = parts
            .funcs
            .get(func)
            .ok_or_else(|| invariant("entry out of range"))?;
        let Some(Type::Function { params: tys, .. }) = parts.types.get(f.ty) else {
            return Err(invariant("the entry's type is not a function"));
        };
        speculate::expand(body, (&parts.types, &parts.consts), (tys, &f.params), m)?
    };
    // Rasterization-rate maps: each `float2` map call becomes Apple's GPU decoder of the parameter
    // data, word loads through its own map pointer.
    let body = rate_map::expand(body, &mut parts, m)?;
    // The AIR 2.7 simdgroup-matrix block loads and stores in their AIR 2.8 form.
    let body = legacy_matrix::expand(body, &mut parts, m)?;
    // Metal runtime routines: each `frexp` and `ilogb` call becomes the routine's measured answer
    // in plain AIR.
    let body = routine::expand(body, &mut parts, m)?;
    // A simdgroup reduction before a discard that Metal's compiler moves past it is placed after
    // the discard, where the fragment lowering gives Metal's lanes.
    let body = if matches!(ep.stage, Stage::Fragment { .. }) {
        discard_sink::expand(body, m)?
    } else {
        body
    };
    // Global built-in variables: each read becomes a read of a built-in parameter added to the
    // entry, but one Metal refuses in the entry's stage.
    let (body, global_builtins) = global_builtin::expand(body, &mut parts, func, &ep.stage, m)?;
    // The entry's body it replaces (decoded, or devirtualized before inlining) is freed.
    if let Some(replaced) = parts
        .funcs
        .get_mut(func)
        .ok_or_else(|| invariant("entry out of range"))?
        .body
        .replace(body)
    {
        replaced.release(m);
    }

    // ABI resolution, the absent arguments' values, then pruning.
    let values = interface::Values {
        globals: &parts.globals,
        consts: &parts.consts,
        fcs: &fcs,
    };
    let (mut ep, structs, absent) = values.entry(abi, ep)?;
    // The lane parameters `matmul2d`'s lowering added.
    if let Some(lanes) = lanes {
        ep.params.extend(lanes.bindings());
    }
    // The built-in parameters the global built-ins' reads became.
    ep.params.extend(global_builtins);
    // A table whose every use devirtualization rewrote is read as a buffer of words.
    let mut function_tables = BTreeSet::new();
    for p in &mut ep.params {
        if let ParamRole::FunctionTable(arg) = p.role {
            if rewritten.contains(&p.param) {
                if let AbiInt::Const(index) = arg.location.index {
                    function_tables.insert(index);
                }
                p.role = ParamRole::Buffer(BufferArg {
                    location: arg.location,
                    access: Access::Read,
                    space: AddrSpace::Device,
                    buffer_size: None,
                    layout: None,
                    raster_order_group: None,
                });
            }
        }
    }
    // A texture or sampler table copied into thread memory is copied handle by handle, so lowering
    // follows each handle it stores; a removed table's handles included.
    table_copy::expand(&mut parts, func, ep.params.iter().chain(&absent), m)?;
    let absent_handles = interface::absent_arguments(&mut parts, func, &ep, &absent, m)?;
    // What nothing observes goes before pruning, so nothing it alone reaches is kept.
    interface::unobserved_code(&mut parts, func, &ep, m)?;
    let pruned = prune::prune(parts, func, m)?;
    ep.func = pruned.func;
    remap_control_points(&mut ep, &pruned.fmap);
    // The module's constexpr samplers and those of the linked functions it imported.
    let kept = |&StaticSampler { global, state }: &StaticSampler| {
        Some((pruned.gmap.get(global.index()).copied().flatten()?, state))
    };
    let mut static_samplers: Vec<_> = abi.static_samplers.iter().filter_map(kept).collect();
    // A function a table can hold is numbered whole, what the entry reaches of it or not; a
    // private function's or a library symbol's as far as the entry reaches it, as the stage's
    // own (Metal numbers none of a private function whose call a function constant removed).
    let mut linked_sampler_groups = Vec::new();
    let unnumbered_linked_samplers = linked_samplers.unknown;
    for (group, origin) in &linked_samplers.groups {
        let at = static_samplers.len();
        static_samplers.extend(group.iter().filter_map(kept));
        let reached = static_samplers.len() - at;
        let states = match origin {
            link::SamplerOrigin::Table => group.iter().map(|s| s.state).collect(),
            _ => static_samplers
                .get(at..)
                .unwrap_or_default()
                .iter()
                .map(|&(_, state)| state)
                .collect(),
        };
        let library = *origin == link::SamplerOrigin::Library;
        if reached > 0 || *origin == link::SamplerOrigin::Table {
            linked_sampler_groups.push(LinkedSamplerGroup {
                kept: reached,
                states,
                library,
            });
        }
    }
    for states in &linked_samplers.unimported {
        linked_sampler_groups.push(LinkedSamplerGroup {
            kept: 0,
            states: states.clone(),
            library: false,
        });
    }
    let module = Module::new(pruned.parts, m)?;
    let body = module
        .func(pruned.func)
        .and_then(|f| f.body.as_ref())
        .ok_or_else(|| invariant("the entry lost its body"))?;
    let cfg = Cfg::build(body, m)?;
    let dom = DomTree::compute(&cfg, m)?;
    Ok(Entry {
        func: pruned.func,
        ep,
        absent_handles,
        structs,
        language: abi.language,
        compile: abi.compile,
        layout: module.data_layout().clone(),
        cfg,
        dom,
        static_samplers,
        linked_sampler_groups,
        unnumbered_linked_samplers,
        intersection_candidates,
        function_constants: !abi.function_constants.is_empty(),
        function_tables,
        lockstep: Lockstep::default(),
        call_stack,
        unmeasured_speculation,
        module,
    })
}

/// The entry's visible-function-table parameters.
fn table_params(ep: &EntryPoint) -> Vec<devirt::TableParam> {
    ep.params
        .iter()
        .filter_map(|p| match &p.role {
            ParamRole::FunctionTable(_) => Some(devirt::TableParam { arg: p.param }),
            _ => None,
        })
        .collect()
}

/// The entry `func`'s working body once inlining ran: `inlined`, or a copy of its own body when
/// it inlined nothing, with its no-op pointer casts erased and its constants renamed through
/// `emap`. It is rebuilt only where erasure changes something: the functions inlined were erased
/// before, so only what a later round added (a devirtualization stub's code) can be. A body
/// replaced is freed.
fn erased_entry(
    parts: &ModuleParts,
    func: FuncId,
    inlined: Option<Body>,
    emap: &[crate::ir::ConstId],
    m: &mut Meter,
) -> Result<Body, Error> {
    match inlined {
        Some(inlined) => match erased(parts, func, Some(&inlined), emap, m)? {
            Some(body) => {
                inlined.release(m);
                Ok(body)
            }
            None => Ok(inlined),
        },
        None => match erased(parts, func, None, emap, m)? {
            Some(body) => Ok(body),
            None => {
                let own = parts
                    .funcs
                    .get(func)
                    .and_then(|f| f.body.as_ref())
                    .ok_or_else(|| invariant("the entry has no body"))?;
                charged_copy(own, m)
            }
        },
    }
}

/// Erase every defined function's body in place, one at a time, each body replaced freed.
fn erase_functions(
    parts: &mut ModuleParts,
    emap: &[crate::ir::ConstId],
    m: &mut Meter,
) -> Result<(), Error> {
    for index in 0..parts.funcs.len() {
        let f = FuncId::from_index(index).ok_or_else(|| invariant("function id space"))?;
        if parts.funcs.get(f).is_some_and(|f| f.body.is_some()) {
            if let Some(body) = erased(parts, f, None, emap, m)? {
                let function = parts
                    .funcs
                    .get_mut(f)
                    .ok_or_else(|| invariant("function out of range"))?;
                if let Some(replaced) = function.body.replace(body) {
                    replaced.release(m);
                }
            }
        }
    }
    Ok(())
}

/// A copy of `body`, charged to the ledger before it is made: its arenas hold exactly its
/// instructions and blocks, as a clone's do, and each instruction its payload.
fn charged_copy(body: &Body, m: &mut Meter) -> Result<Body, Error> {
    let slots = body
        .insts
        .len()
        .saturating_mul(std::mem::size_of::<crate::ir::Inst>())
        .saturating_add(
            body.blocks
                .len()
                .saturating_mul(std::mem::size_of::<crate::ir::Block>()),
        );
    let payload = body
        .insts
        .values()
        .fold(0usize, |sum, i| sum.saturating_add(i.kind.heap_bytes()));
    m.tick(u32::try_from(body.insts.len()).unwrap_or(u32::MAX))?;
    m.charge(slots.saturating_add(payload))?;
    Ok(body.clone())
}

/// Function `func`'s body `inlined` (its own body when `None`) with its no-op pointer casts
/// erased and its constants renamed through `emap`; `None` when there is nothing to erase or
/// rename.
fn erased(
    parts: &ModuleParts,
    func: FuncId,
    inlined: Option<&Body>,
    emap: &[crate::ir::ConstId],
    m: &mut Meter,
) -> Result<Option<Body>, Error> {
    let f = parts
        .funcs
        .get(func)
        .ok_or_else(|| invariant("entry out of range"))?;
    let body = match inlined {
        Some(b) => b,
        None => f
            .body
            .as_ref()
            .ok_or_else(|| invariant("the entry has no body"))?,
    };
    let params: &[TypeId] = match parts.types.get(f.ty) {
        Some(Type::Function { params, .. }) => params,
        _ => &[],
    };
    let value_ty = |v: Value| match v {
        Value::Arg(a) => params.get(a.index()).copied(),
        Value::Inst(i) => body.insts.get(i).map(|i| i.ty),
        Value::Const(c) => parts.consts.ty(c),
    };
    erase::erase_body(body, emap, &parts.consts, &value_ty, m)
}

/// What the rounds of devirtualization and inlining work on.
struct Round<'a> {
    func: FuncId,
    emap: &'a [crate::ir::ConstId],
    tables: &'a [devirt::TableParam],
    candidates: &'a [link::Candidate],
    linked: &'a [LinkedModule],
    name: &'a DiagName,
    /// The dispatch markers of recursive functions a table can hold.
    groups: &'a BTreeMap<FuncId, FuncId>,
    /// The enter markers of recursive functions.
    enters: &'a BTreeMap<FuncId, FuncId>,
    /// The words of the functions whose address is taken.
    words: &'a devirt::FunctionWords,
    /// The function constants as SCCP sees them (devirtualization asks SCCP what they decide of a
    /// call through a pointer).
    fcs: &'a HashMap<GlobalId, sccp::FcState>,
}

/// Inline and devirtualize `body`, the entry's body with new copies of recursive functions, as the
/// first rounds did the entry's own.
fn reinline(
    body: Body,
    parts: &mut ModuleParts,
    stubs: &mut Stubs,
    r: &Round<'_>,
    m: &mut Meter,
) -> Result<(Body, Vec<u32>), Error> {
    if let Some(replaced) = parts
        .funcs
        .get_mut(r.func)
        .ok_or_else(|| invariant("entry out of range"))?
        .body
        .replace(body)
    {
        replaced.release(m);
    }
    let inlined = inline::inline(
        &parts.funcs,
        (&parts.types, &parts.symbols),
        &mut parts.consts,
        r.func,
        m,
        |_, _| Ok(()),
    )?;
    let body = erased_entry(parts, r.func, inlined, r.emap, m)?;
    inline_rounds(body, parts, stubs, r, m)
}

/// Devirtualize the inlined, erased entry `body`, inline the direct calls that makes, and repeat
/// until a round makes none. Returns the body and the table parameters every use of which was
/// rewritten into word loads.
///
/// Each round's new calls lie in code the round before inlined, so a chain of rounds longer than
/// the module's defined functions calls some function from within its own inlined body: a round
/// past that count is `Recursion`, located at a call it made, naming the callee.
fn inline_rounds(
    body: Body,
    parts: &mut ModuleParts,
    stubs: &mut Stubs,
    r: &Round<'_>,
    m: &mut Meter,
) -> Result<(Body, Vec<u32>), Error> {
    let defined = parts.funcs.values().filter(|f| f.body.is_some()).count();
    let params: Vec<TypeId> = match parts.funcs.get(r.func).and_then(|f| parts.types.get(f.ty)) {
        Some(Type::Function { params, .. }) => params.to_vec(),
        _ => return Err(invariant("the entry's type is not a function")),
    };
    let cx = devirt::Cx {
        tables: r.tables,
        candidates: r.candidates,
        linked: r.linked,
        params: &params,
        groups: r.groups,
        enters: r.enters,
        words: r.words,
        fcs: r.fcs,
        name: r.name,
    };
    let mut body = body;
    let mut rounds = 0usize;
    while let Some(pass) = devirt::pass(&body, parts, stubs, &cx, m)? {
        // Each body a round replaces is freed as the round makes its successor.
        body.release(m);
        body = pass.body;
        if !pass.calls {
            break;
        }
        rounds += 1;
        if rounds > defined {
            return Err(recursion(&body, parts, r.name));
        }
        if let Some(replaced) = parts
            .funcs
            .get_mut(r.func)
            .ok_or_else(|| invariant("entry out of range"))?
            .body
            .replace(body)
        {
            replaced.release(m);
        }
        let inlined = inline::inline(
            &parts.funcs,
            (&parts.types, &parts.symbols),
            &mut parts.consts,
            r.func,
            m,
            |_, _| Ok(()),
        )?;
        body = erased_entry(parts, r.func, inlined, r.emap, m)?;
    }
    // A table parameter keeps its role when anything but a word load or its address uses it.
    let mut rewritten = Vec::new();
    for t in r.tables {
        let arg = Value::Arg(crate::ir::ArgIdx::new(t.arg));
        let mut words_only = true;
        for inst in body.insts.values() {
            m.tick(1)?;
            let uses = |v: Value| v == arg;
            let word_use = match &inst.kind {
                InstKind::Load { ptr, .. } => *ptr == arg,
                InstKind::Gep { base, indices, .. } => {
                    *base == arg && !indices.iter().copied().any(uses)
                }
                _ => false,
            };
            let mut named = false;
            inst.kind.for_each_value(|v| named |= uses(v));
            if named && !word_use {
                words_only = false;
                break;
            }
        }
        if words_only {
            rewritten.push(t.arg);
        }
    }
    Ok((body, rewritten))
}

/// `Recursion` at the first direct call to a defined function left in `body`, naming its callee.
fn recursion(body: &Body, parts: &ModuleParts, name: &DiagName) -> Error {
    for (b, block) in body.blocks.iter() {
        for i in block.inst_ids() {
            let Some(InstKind::Call(call)) = body.insts.get(i).map(|i| &i.kind) else {
                continue;
            };
            let Callee::Direct(f) = call.callee else {
                continue;
            };
            let Some(func) = parts.funcs.get(f).filter(|f| f.body.is_some()) else {
                continue;
            };
            let callee = parts.symbols.get(func.name).unwrap_or(b"?");
            return Unsupported::not_implemented(Feature::Recursion)
                .naming(DiagName::new(String::from_utf8_lossy(callee)))
                .at(Location::function(
                    name.clone(),
                    Some(b.raw()),
                    Some(i.raw()),
                ))
                .into();
        }
    }
    Unsupported::not_implemented(Feature::Recursion)
        .at(Location::function(name.clone(), None, None))
        .into()
}

/// The integer each of `globals` holds once `constants` are bound and the static initializers
/// have run, without canonicalizing any function body: how a metadata integer or predicate that the
/// ABI states as a global ([`crate::abi::AbiInt::Global`], [`crate::abi::Predicate`]) resolves for
/// one pipeline, as step 8 resolves it.
///
/// `module` is consumed: the initializers' stores are folded into it, as [`specialize`] folds
/// them. `constants` are checked against the declarations as [`specialize`] checks them (an
/// undeclared index or a value of another type is `InvalidRequest`). With no `globals` the
/// static initializers are not run: only the constants are checked.
///
/// Each entry of the result answers the global at the same position: `NeedsInput
/// (FunctionConstant)` when it is computed from an unsupplied constant, and
/// `Unsupported{StaticInitializer}` when it does not hold an integer or is computed through a
/// pure operation the interpreter does not evaluate (float arithmetic outside the operations it
/// models). Such an operation taints only the globals it reaches: a global computed without it
/// still resolves. A branch on its result, an address computed from it, or any other construct the
/// interpreter refuses still fails the whole call.
pub fn static_values(
    module: Module,
    abi: &AirAbi,
    constants: &BTreeMap<u32, ConstValue>,
    globals: &[GlobalId],
    m: &mut Meter,
) -> Result<Vec<Result<u64, Error>>, Error> {
    let (pending, mut fcs) = fc::check(&module, abi, constants, m)?;
    if globals.is_empty() {
        return Ok(Vec::new());
    }
    let mut parts = module.into_parts();
    fc::bind(
        &parts.types,
        &mut parts.consts,
        &mut parts.globals,
        &pending,
        m,
    )?;
    let evaluated = static_init::evaluate(&mut parts, &fcs, static_init::Unevaluable::Opaque, m)?;
    for (g, index) in evaluated.derived {
        m.charge(std::mem::size_of::<(GlobalId, sccp::FcState)>() * 2)?;
        fcs.insert(
            g,
            sccp::FcState {
                index,
                supplied: false,
                decl: false,
            },
        );
    }
    let values = interface::Values {
        globals: &parts.globals,
        consts: &parts.consts,
        fcs: &fcs,
    };
    Ok(globals
        .iter()
        .map(|&g| match evaluated.opaque.contains(&g) {
            true => Err(interface::not_evaluable()),
            false => values.int(g),
        })
        .collect())
}

/// A post-tessellation entry's control-point inputs: each one's parameter and control-point
/// function.
fn control_points(ep: &EntryPoint) -> Vec<(u32, FuncId)> {
    if !matches!(ep.stage, Stage::Vertex { patch: Some(_) }) {
        return Vec::new();
    }
    ep.params
        .iter()
        .filter_map(|p| match &p.role {
            ParamRole::PatchInput(PatchInput::ControlPoints { function, .. }) => {
                Some((p.param, *function))
            }
            _ => None,
        })
        .collect()
}

/// Renumber each control-point input's function to the pruned module. A function pruning dropped
/// is one the body never calls, and the post-check lets the input reach nothing but its
/// function's calls, so no instruction reads the input: its binding goes with the function, as a
/// gated-off parameter's does (the parameter stays, known unused).
fn remap_control_points(ep: &mut EntryPoint, fmap: &[Option<FuncId>]) {
    ep.params.retain_mut(|p| {
        let ParamRole::PatchInput(PatchInput::ControlPoints { function, .. }) = &mut p.role else {
            return true;
        };
        match fmap.get(function.index()).copied().flatten() {
            Some(f) => {
                *function = f;
                true
            }
            None => false,
        }
    });
}

/// The control-point input (a parameter of `control_points`) that instruction `kind` uses other
/// than as the pointer of a call to that input's own function.
fn stray_control_point_use(kind: &InstKind, control_points: &[(u32, FuncId)]) -> Option<u32> {
    let input = |v: Value| match v {
        Value::Arg(a) if control_points.iter().any(|(p, _)| *p == a.raw()) => Some(a.raw()),
        _ => None,
    };
    if let InstKind::Call(call) = kind {
        if let (Callee::Direct(f), [index, Value::Arg(a)]) = (&call.callee, call.args.as_ref()) {
            if control_points.contains(&(a.raw(), *f)) {
                return input(*index);
            }
        }
    }
    let mut found = None;
    kind.for_each_value(|v| {
        if found.is_none() {
            found = input(v);
        }
    });
    found
}

fn diag_name(module: &Module, f: FuncId) -> DiagName {
    let bytes = module
        .func(f)
        .and_then(|f| module.symbols().get(f.name))
        .unwrap_or(b"?");
    DiagName::new(String::from_utf8_lossy(bytes))
}

/// The entry point named `entry`.
pub fn select<'a>(module: &Module, abi: &'a AirAbi, entry: &str) -> Result<&'a EntryPoint, Error> {
    let mut found = abi.entries.iter().filter(|e| {
        module
            .func(e.func)
            .and_then(|f| module.symbols().get(f.name))
            == Some(entry.as_bytes())
    });
    let first = found.next();
    match (first, found.next()) {
        (Some(ep), None) => Ok(ep),
        (Some(_), Some(_)) => Err(Error::InvalidRequest(RequestError::AmbiguousEntry {
            requested: DiagName::new(entry),
        })),
        (None, _) => Err(Error::InvalidRequest(RequestError::EntryNotFound {
            requested: DiagName::new(entry),
            available: u32::try_from(abi.entries.len()).unwrap_or(u32::MAX),
        })),
    }
}

/// Each refusal carries the call's location in the inlined body (the entry's function, block, and
/// instruction) and names its callee's symbol where it has one. Indirect calls devirtualization
/// left (a function pointer loaded from memory or computed, a table that is no entry parameter),
/// calls to functions this module does not define (linking), and unknown intrinsics of a typed
/// family (`intrinsics::UNDECODED_FEATURES`) are refused as their features, not implemented yet;
/// any other unknown intrinsic is `UnknownIntrinsic`. A call to a visible-function reference that
/// linking did not resolve is refused by the stub's state: `NeedsInput(LinkedFunction)` when no
/// linked function has its name, else the reason the linked function of that name cannot resolve it
/// (a typed `Unsupported` is refused at the call). The declarations devirtualization calls in place
/// of a table candidate that cannot be imported, or of a table call no candidate answers, or before
/// a function pointer's use as a value, are refused the same way. Any other declaration is a
/// function of a dynamic library: `Linking`, not implemented. A call through a function's constant
/// address at another function type is classified by that function: defined (or a stub linking
/// resolved), `InvalidInput(Canon(CallTypeMismatch))` when its arguments differ in count or shape
/// from the function's parameters, since nothing binds them, else a `DirectCall` the inliner cannot
/// take; an intrinsic, a signature mismatch; any other declaration, as a direct call to it. A
/// post-tessellation entry's control-point input may be only the pointer of a call to its own
/// function, the one call to a declaration kept; any other use is `Tessellation` at the input.
fn post_check(
    body: &Body,
    parts: &ModuleParts,
    stubs: &Stubs,
    name: &DiagName,
    control_points: &[(u32, FuncId)],
    m: &mut Meter,
) -> Result<(), Error> {
    if stubs.imports_variables() {
        rewrite::scoped(m, |m, held| declared_imports(body, parts, stubs, m, held))?;
    }
    for (b, block) in body.blocks.iter() {
        for i in block.inst_ids() {
            m.tick(1)?;
            let Some(kind) = body.insts.get(i).map(|i| &i.kind) else {
                continue;
            };
            // Metal reads a control-point input only through its function: any other use, whether
            // or not the body also calls the function, is refused at the input.
            if let Some(arg) = stray_control_point_use(kind, control_points) {
                let at = Location::metadata(Some(name.clone()), Some(arg), None);
                return Err(Unsupported::not_implemented(Feature::Tessellation)
                    .at(at)
                    .into());
            }
            let InstKind::Call(call) = kind else {
                continue;
            };
            let at = Location::function(name.clone(), Some(b.raw()), Some(i.raw()));
            let symbol = |f: FuncId| {
                parts
                    .funcs
                    .get(f)
                    .and_then(|f| parts.symbols.get(f.name))
                    .unwrap_or(b"?")
            };
            let diag = |bytes: &[u8]| DiagName::new(String::from_utf8_lossy(bytes));
            // A call to the declaration `f` this module does not define.
            let undefined =
                |f: FuncId| -> Result<Unsupported, Error> {
                    match stubs.get(f) {
                        Some(Stub::Missing(s)) => {
                            Err(Error::NeedsInput(MissingInput::LinkedFunction {
                                symbol: diag(&s.0),
                            }))
                        }
                        Some(Stub::Refused(Error::Unsupported(u))) => Ok(u.clone()),
                        Some(Stub::Refused(e)) => Err(e.clone()),
                        Some(Stub::Resolved(_)) => {
                            Err(invariant("a call to a resolved stub survived linking"))
                        }
                        // A dynamic-library import, or a declaration no library can define.
                        None => match stubs.import_refusal(Item::Func(f), symbol(f)) {
                            Some(e) => Err(e),
                            None => Ok(Unsupported::not_implemented(Feature::Linking)
                                .naming(diag(symbol(f)))),
                        },
                    }
                };
            let refusal = match call.callee {
                Callee::Air { .. } | Callee::Llvm { .. } => continue,
                // The decoder keeps a call through a function's own constant address indirect when
                // the call's function type is not the function's (`bitcode::function`): its callee
                // is still that one function, so it is classified by the function, not as dynamic.
                Callee::Indirect { callee, fn_ty } => {
                    match constant_callee(callee, &parts.consts) {
                        Some(f) => {
                            let target = stubs.target(f);
                            match parts.funcs.get(target) {
                                // A defined function called at another type whose arguments the
                                // parameters cannot receive: no rule binds them
                                // (`CanonError::CallTypeMismatch`).
                                Some(func)
                                    if func.body.is_some()
                                        && arguments_mismatch(&parts.types, fn_ty, func.ty, m)? =>
                                {
                                    let kind = InputErrorKind::Canon(CanonError::CallTypeMismatch);
                                    return Err(InputError::new(kind, at).into());
                                }
                                // Otherwise (another return type, identically shaped structs
                                // of other names, a variadic type) the inliner cannot take it,
                                // as linking does not resolve a reference of another isomorphic
                                // type.
                                Some(func) if func.body.is_some() => {
                                    Unsupported::not_implemented(Feature::DirectCall)
                                        .naming(diag(symbol(target)))
                                }
                                // An intrinsic called at a type other than its declaration's.
                                _ if intrinsics::is_intrinsic_name(symbol(f)) => {
                                    let kind = InputErrorKind::Intrinsic(
                                        IntrinsicError::SignatureMismatch,
                                    );
                                    return Err(InputError::new(kind, at).into());
                                }
                                _ => undefined(f)?,
                            }
                        }
                        None => Unsupported::not_implemented(Feature::DynamicCall),
                    }
                }
                Callee::Direct(f) => match parts.funcs.get(f) {
                    Some(callee) if callee.body.is_some() => {
                        return Err(invariant("a call to a defined function survived inlining"));
                    }
                    // A control-point fetch: its pointer operand must be the entry's control-point
                    // input of that function (a pointer the body computes is not a patch).
                    _ if control_points.iter().any(|(_, cp)| *cp == f) => {
                        let fetch = matches!(call.args.as_ref(), [_, Value::Arg(a)]
                            if control_points.contains(&(a.raw(), f)));
                        if fetch {
                            continue;
                        }
                        Unsupported::not_implemented(Feature::Tessellation).naming(diag(symbol(f)))
                    }
                    _ => undefined(f)?,
                },
                Callee::UnknownIntrinsic(f) => {
                    let intrinsic = symbol(f);
                    match intrinsics::undecoded_feature(intrinsic) {
                        Some(feature) => {
                            Unsupported::not_implemented(feature).naming(diag(intrinsic))
                        }
                        None => {
                            let kind =
                                InputErrorKind::Intrinsic(IntrinsicError::UnknownIntrinsic {
                                    name: diag(intrinsic),
                                });
                            return Err(InputError::new(kind, at).into());
                        }
                    }
                }
            };
            return Err(refusal.at(at).into());
        }
    }
    Ok(())
}

/// Refuse the first use in `body` of a variable import that is still a declaration: no dynamic
/// library supplied it.
fn declared_imports(
    body: &Body,
    parts: &ModuleParts,
    stubs: &Stubs,
    m: &mut Meter,
    held: &mut usize,
) -> Result<(), Error> {
    let mut seen = BTreeSet::new();
    let mut stack = Vec::new();
    for inst in body.insts.values() {
        m.tick(1)?;
        inst.kind.for_each_value(|v| {
            if let Value::Const(c) = v {
                stack.push(c);
            }
        });
        while let Some(c) = stack.pop() {
            m.tick(1)?;
            if !seen.insert(c) {
                continue;
            }
            let Some(k) = parts.consts.get(c) else {
                continue;
            };
            let bytes = std::mem::size_of::<crate::ir::ConstId>() * (k.value.children().len() + 3);
            m.charge(bytes)?;
            *held += bytes;
            if let Const::GlobalAddr(g) = k.value {
                let declared = parts.globals.get(g).filter(|g| g.init.is_none());
                if let Some(global) = declared {
                    let name = parts.symbols.get(global.name).unwrap_or(b"?");
                    if let Some(e) = stubs.import_refusal(Item::Global(g), name) {
                        return Err(e);
                    }
                }
            }
            stack.extend_from_slice(k.value.children());
        }
    }
    Ok(())
}

/// Whether a call at function type `call` passes arguments that a function of type `def` cannot
/// receive: another parameter count, or a parameter whose type differs in shape (identified structs
/// matched by body, as linking matches them, `renumber::isomorphism`). Such a call binds no
/// argument to any parameter (`CanonError::CallTypeMismatch`). A call that differs only in its
/// return type or in the names of identically shaped structs passes every argument as a parameter
/// receives it, and a variadic type on either side is a variadic call: none of these is a mismatch.
fn arguments_mismatch(
    types: &TypeTable,
    call: TypeId,
    def: TypeId,
    m: &mut Meter,
) -> Result<bool, Error> {
    let params = |t: TypeId| match types.get(t) {
        Some(Type::Function { params, vararg, .. }) => Ok((params, *vararg)),
        _ => Err(invariant("a call's function type is not a function type")),
    };
    let ((call, call_vararg), (def, def_vararg)) = (params(call)?, params(def)?);
    if call_vararg || def_vararg {
        return Ok(false);
    }
    if call.len() != def.len() {
        return Ok(true);
    }
    for (&a, &b) in call.iter().zip(def.iter()) {
        if renumber::isomorphism((types, types), (a, b), |_| None, m)?.is_none() {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The function whose constant address `callee` is, if it is one.
fn constant_callee(callee: Value, consts: &ConstTable) -> Option<FuncId> {
    match callee {
        Value::Const(c) => match consts.get(c).map(|k| &k.value) {
            Some(Const::FuncAddr(f)) => Some(*f),
            _ => None,
        },
        Value::Arg(_) | Value::Inst(_) => None,
    }
}
