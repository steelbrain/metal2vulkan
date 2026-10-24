//! CPU SPIR-V interpreter for L2 tests; never on a product path.
//!
//! [`run`] executes the compute entry point of a finished [`TypedModule`] (before serialization)
//! over a dispatch grid, with byte-array memory for every storage class the new core uses in
//! compute: `Function`, `Private`, `Input` (the compute builtins), `Workgroup`, `StorageBuffer`,
//! `Uniform`, `PushConstant`, and `PhysicalStorageBuffer`. It is a test oracle, so it is strict
//! where GPUs are silent: every behavior Vulkan leaves undefined that a test could depend on is a
//! typed [`Trap`] rather than a plausible value.
//!
//! **Vertex entries** run as a draw without vertex input or index buffer: [`Config::workgroups`]
//! is `[vertices, instances, views]` from [`Config::first`]'s vertex and instance, each invocation
//! a workgroup of one that reads `VertexIndex`, `InstanceIndex`, `BaseVertex`, and `BaseInstance`
//! (the draw's first vertex and first instance) and `ViewIndex`
//! (views `0..views`, a multiview draw's view mask of its first views; 1 without multiview) and
//! writes its `Output` variables into memory of its own, which the run does not report. Interface
//! inputs (`Location` variables) hold the bytes [`run_with_inputs`] supplies, and are unmodelled
//! without it.
//!
//! **Traps.** Out-of-bounds accesses (offsets are computed without wrapping, so a wrapped or huge
//! offset traps instead of aliasing), misaligned `Aligned` physical accesses, reads of
//! uninitialized `Workgroup` bytes (the arena is poison-tracked per bit), non-uniform barriers,
//! out-of-range composite and access-chain indices, integer division by zero and signed overflow,
//! over-wide shifts and bit fields, out-of-range float-to-integer conversions, subgroup reads of
//! inactive lanes, `OpUnreachable`, and exhausting the step bound (so non-terminating output is a
//! failed test, not a hang).
//!
//! **Arithmetic** is exact: integer, bit, and composite operations bit for bit, and IEEE-754 float
//! operations with round-to-nearest-even per operation (via `f64`, where double rounding is
//! innocuous for `+ - * /` and `sqrt`); `f16` values are rounded explicitly. Transcendental
//! built-ins and fragment operations are [`Unmodelled`].
//!
//! **Compute derivatives** (`deriv`) run under `DerivativeGroupLinearKHR` over groups of four
//! consecutive local indices, which must all be in the tangle. Fine derivatives are exact;
//! coarse and unqualified ones, whose row or column Vulkan lets the implementation choose, are
//! exact when every choice agrees and a [`TrapKind::ImplementationChoice`] otherwise.
//!
//! **Images** (`image`, [`run_with_images`]): bound images hold their texels as the components of
//! their sampled type; fetches, reads, writes, and size/level/sample queries are exact and an
//! access outside the image traps. Sampling and gathers are [`Unmodelled`].
//!
//! **Scheduling** (`sched`). Each subgroup (32 or 64 consecutive linear local invocations) executes
//! in SIMT lockstep with maximal reconvergence: at every step the active tangle is the set of lanes
//! at the earliest position in the structured block order, so lanes that diverged reconverge at the
//! construct's merge and subgroup operations see exactly the lanes a maximally reconverging device
//! would activate. A workgroup barrier is a rendezvous of every invocation that has not returned
//! (the Q3 model: returned invocations are absent), all at the same dynamic position, or a
//! [`TrapKind::NonUniformBarrier`]; under [`BarrierRule::Vulkan`] a returned invocation traps too,
//! since Vulkan defines a workgroup barrier only where every invocation of the workgroup executes
//! it. A subgroup barrier orders its tangle: under `MaximallyReconvergesKHR` the lanes of its step,
//! wherever they diverged; without that mode every lane of the subgroup that has not returned must
//! be in it, or it is a [`TrapKind::NonUniformBarrier`]. Between barriers the subgroups run
//! round-robin, each to its next barrier, in [`Order::Forward`] or [`Order::Reverse`] (lane order
//! within a step and workgroup order follow it too), so running both orders and comparing outputs
//! detects races.

mod deriv;
mod eval;
mod float;
mod glsl;
mod group;
pub mod harness;
mod image;
mod layout;
mod memory;
mod sched;
mod value;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::fmt;

use crate::pipeline::sym::{ResourceSlot, SpecKey, VaryingKey};
use crate::spirv::ids::{BlockId, TypeId};
use crate::spirv::ir::{ExecModel, GlslOp, Scope, StorageClass, TypedModule};

pub use image::Image;
pub use memory::MemoryKind;

/// The interpreter's result type.
pub type IResult<T> = Result<T, InterpError>;

/// Why a run did not complete.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InterpError {
    /// The module did something undefined, or exceeded the step bound.
    Trap(Trap),
    /// The module uses something outside the interpreter's model.
    Unmodelled(Unmodelled),
    /// The dispatch configuration or resources do not fit the module.
    Config(String),
    /// The module violates an invariant every built module holds (an interpreter or builder bug).
    Malformed(String),
}

impl fmt::Display for InterpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterpError::Trap(t) => write!(f, "trap: {t:?}"),
            InterpError::Unmodelled(u) => write!(f, "unmodelled: {u:?}"),
            InterpError::Config(s) => write!(f, "config: {s}"),
            InterpError::Malformed(s) => write!(f, "malformed module: {s}"),
        }
    }
}

impl std::error::Error for InterpError {}

impl InterpError {
    /// The trap kind, when this is a trap.
    pub fn trap_kind(&self) -> Option<&TrapKind> {
        match self {
            InterpError::Trap(t) => Some(&t.kind),
            _ => None,
        }
    }

    /// Attach the invocation and site to a trap that has none yet.
    fn locate(self, invocation: [u32; 3], site: Site) -> InterpError {
        match self {
            InterpError::Trap(mut t) => {
                t.invocation.get_or_insert(invocation);
                t.site.get_or_insert(site);
                InterpError::Trap(t)
            }
            other => other,
        }
    }
}

/// A trap: what went wrong, and where, when known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trap {
    pub kind: TrapKind,
    /// The global invocation id of the trapping invocation.
    pub invocation: Option<[u32; 3]>,
    pub site: Option<Site>,
}

/// A position in the module: an instruction of a block, or its terminator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Site {
    pub block: BlockId,
    /// The instruction index, or `None` for the terminator.
    pub inst: Option<usize>,
}

/// What a trap caught.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrapKind {
    /// `[offset, offset + len)` is not inside the memory object (`size` bytes; 0 for an unmapped
    /// physical address). Offsets never wrap.
    OutOfBounds {
        memory: MemoryKind,
        offset: u64,
        len: u64,
        size: u64,
    },
    /// A physical access whose address is not a multiple of its `Aligned` operand.
    Misaligned {
        address: u64,
        align: u32,
    },
    /// A read of `Workgroup` bytes no invocation has written.
    UninitializedRead {
        offset: u64,
    },
    /// A write to read-only memory (push constants, builtins).
    ReadOnlyWrite {
        memory: MemoryKind,
    },
    /// A dynamic index into a vector, matrix, or sized array (access chain or dynamic extract) is
    /// out of range.
    IndexOutOfRange {
        index: i64,
        len: u64,
    },
    /// A negative (signed) access-chain index into a runtime array.
    NegativeIndex {
        index: i64,
    },
    /// An address computation that would pass `u64::MAX` (offsets never wrap).
    AddressOverflow,
    /// Invocations wait at different barriers, or a barrier is reached by part of a subgroup
    /// whose entry does not reconverge maximally.
    NonUniformBarrier,
    /// The run exceeded [`Config::step_limit`] lane-steps.
    StepLimit(u64),
    DivideByZero,
    /// `SDiv`/`SRem`/`SMod` of the minimum value by -1.
    SignedOverflow,
    /// A shift by at least the base's width.
    ShiftOutOfRange {
        shift: u64,
        width: u8,
    },
    /// A bit field with `offset + count` above the width.
    BitFieldOutOfRange,
    /// A float-to-integer conversion of NaN or of a value outside the result type.
    ConversionOutOfRange,
    /// A subgroup or derivative operation read a lane that is not in the tangle (or does not
    /// exist).
    InactiveLane {
        lane: i64,
    },
    /// An operation whose result Vulkan leaves undefined (named), e.g. `FindLSB` of an empty
    /// ballot.
    Undefined(&'static str),
    /// An operation whose result Vulkan lets the implementation choose among several values, and
    /// the choices differ here (named), e.g. a coarse derivative over a group whose rows differ.
    ImplementationChoice(&'static str),
    /// `OpUnreachable` executed.
    Unreachable,
    /// An image access whose coordinate, layer, level, or sample index is outside bound image
    /// `image` (by its index in the run's images), or a size query at a level it lacks.
    ImageOutOfBounds {
        image: u32,
    },
}

impl TrapKind {
    fn err(self) -> InterpError {
        InterpError::Trap(Trap {
            kind: self,
            invocation: None,
            site: None,
        })
    }
}

/// Something the interpreter does not model.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unmodelled {
    /// Only `GLCompute` and `Vertex` entry points run.
    ExecModel(ExecModel),
    /// An instruction outside the model (sampling, derivatives under `DerivativeGroupQuadsKHR`,
    /// fragment operations, kills).
    Opcode(crate::spv::Op),
    /// A `GLSL.std.450` instruction outside the exact subset (transcendentals and friends).
    Glsl(GlslOp),
    /// A variable of this storage class.
    StorageClass(StorageClass),
    /// An input builtin other than the compute ones and the vertex indices.
    BuiltIn(crate::spv::BuiltIn),
    /// A barrier or atomic scope Vulkan does not allow in compute.
    Scope(Scope),
    /// A type the memory model cannot lay out or operate on.
    Type(TypeId),
}

impl Unmodelled {
    fn err(self) -> InterpError {
        InterpError::Unmodelled(self)
    }
}

fn malformed(what: impl Into<String>) -> InterpError {
    InterpError::Malformed(what.into())
}

/// Scheduling order. Two orders over the same input must produce the same output, or the module
/// has a race.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Order {
    /// Workgroups, subgroups, and lanes within a step in increasing order.
    #[default]
    Forward,
    /// Everything in decreasing order.
    Reverse,
}

/// The modelled subgroup width.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SubgroupSize {
    /// 8 lanes (a narrow device, such as the emulated simdgroups run on).
    W8,
    /// 16 lanes.
    W16,
    #[default]
    W32,
    W64,
}

impl SubgroupSize {
    /// The width in lanes.
    pub fn lanes(self) -> u32 {
        match self {
            SubgroupSize::W8 => 8,
            SubgroupSize::W16 => 16,
            SubgroupSize::W32 => 32,
            SubgroupSize::W64 => 64,
        }
    }
}

/// A dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// The number of workgroups in each dimension (for a vertex entry, `[vertices, instances,
    /// views]`).
    pub workgroups: [u32; 3],
    pub subgroup: SubgroupSize,
    pub order: Order,
    /// The most lane-steps (one instruction or terminator for one invocation) the whole dispatch
    /// may take before [`TrapKind::StepLimit`].
    pub step_limit: u64,
    /// Which invocations a workgroup barrier waits for.
    pub barriers: BarrierRule,
    /// A vertex run's first vertex and first instance (Vulkan's `firstVertex` and
    /// `firstInstance`; 0 and 0 by default): `VertexIndex` and `InstanceIndex` count from them,
    /// and `BaseVertex` and `BaseInstance` read them.
    pub first: [u32; 2],
}

/// Which invocations a workgroup barrier waits for.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BarrierRule {
    /// Control's Q3 model: every invocation that has not returned, all at the same dynamic
    /// position; returned invocations are absent.
    #[default]
    Q3,
    /// Vulkan's rule (SPIR-V `OpControlBarrier`: with `Workgroup` execution scope, behavior is
    /// undefined unless every invocation of the workgroup executes the same dynamic instance):
    /// an invocation that returned while others wait at a workgroup barrier is a
    /// [`TrapKind::NonUniformBarrier`] too.
    Vulkan,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            workgroups: [1, 1, 1],
            subgroup: SubgroupSize::W32,
            order: Order::Forward,
            step_limit: 1 << 24,
            barriers: BarrierRule::Q3,
            first: [0, 0],
        }
    }
}

/// One buffer: bound to a descriptor slot, reachable at a device address, or both.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Buffer {
    /// The `StorageBuffer`/`Uniform` binding this buffer backs. The elements of a descriptor
    /// array of buffers are the slot's consecutive buffers, in element order.
    pub slot: Option<ResourceSlot>,
    /// The device address of byte 0, for `PhysicalStorageBuffer` accesses.
    pub address: Option<u64>,
    pub bytes: Vec<u8>,
}

/// Everything the dispatch reads and writes besides its own variables.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Resources {
    pub buffers: Vec<Buffer>,
    /// The push-constant bytes.
    pub push: Vec<u8>,
    /// Specialization values by key (`VkSpecializationInfo`): the bits of each specialization
    /// constant the module declares under that key, in its type's width (`bool` 0 or 1). A
    /// constant without an entry takes its default; an entry for a key the module does not
    /// declare is ignored, as in Vulkan.
    pub spec: BTreeMap<SpecKey, u64>,
}

/// The push bytes `push`, with each push member that carries a specialization key's value as
/// data (`PushField::Specialized`) holding the value `spec` gives the key, else the default the
/// lowering declares for the key's constant, as a binder writes the value it would have
/// specialized the constant to: a test that specializes a key in [`Resources::spec`] reads the same
/// value in either form. A field placed in the stage's `ArgumentTable` is the test's to bind.
fn specialized_push(m: &TypedModule, spec: &BTreeMap<SpecKey, u64>, mut push: Vec<u8>) -> Vec<u8> {
    use crate::pipeline::sym::PushField;
    for member in m.push_members() {
        let PushField::Specialized(key) = member.field else {
            continue;
        };
        let value = spec.get(&key).copied().unwrap_or(match key {
            SpecKey::Presence(_) => 1,
            SpecKey::TexelFormat(_) => u64::from(crate::metal::texel_format::TexelFormat::OTHER.0),
            SpecKey::TextureSwizzle(_) => {
                u64::from(crate::metal::TextureSwizzleChannels::DEFAULT_WORD)
            }
            SpecKey::BicubicTexels(_) => u64::from(crate::metal::bicubic::BicubicTexels::NEUTRAL.0),
            _ => 0,
        });
        let at = member.offset as usize;
        if push.len() < at + 4 {
            push.resize(at + 4, 0);
        }
        push[at..at + 4].copy_from_slice(&(value as u32).to_le_bytes());
    }
    push
}

/// What one invocation did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvocationStats {
    pub global_id: [u32; 3],
    /// Visits of each loop header, indexed like [`TypedModule::loops`].
    pub header_visits: Vec<u64>,
    /// Workgroup barriers this invocation passed.
    pub barriers: u64,
}

/// A completed dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    /// Every buffer's final bytes, in [`Resources::buffers`] order.
    pub buffers: Vec<Vec<u8>>,
    /// Lane-steps taken.
    pub steps: u64,
    /// Per invocation, in (workgroup, local linear index) order whatever the schedule.
    pub invocations: Vec<InvocationStats>,
}

/// Run `m`'s compute entry point over `cfg.workgroups` with `res` (or its vertex entry point over
/// `cfg.workgroups[0]` vertices of `cfg.workgroups[1]` instances; see the module docs).
pub fn run(m: &TypedModule, cfg: &Config, res: Resources) -> IResult<Outcome> {
    sched::run(m, cfg, res, Vec::new(), BTreeMap::new()).map(|(o, _)| o)
}

/// [`run`] with `inputs` held in the module's user interface input variables, by relocation key
/// (`VaryingKey::attribute` for a vertex attribute Vulkan's fixed-function input fetches): each
/// entry's bytes are the variable's value in every invocation, the little-endian encoding of its
/// declared type. A variable the map omits stays unmodelled.
pub fn run_with_inputs(
    m: &TypedModule,
    cfg: &Config,
    res: Resources,
    inputs: BTreeMap<VaryingKey, Vec<u8>>,
) -> IResult<Outcome> {
    sched::run(m, cfg, res, Vec::new(), inputs).map(|(o, _)| o)
}

/// [`run`] with `images` bound to the module's image bindings (each by its `slot`); returns the
/// images' final texels too.
pub fn run_with_images(
    m: &TypedModule,
    cfg: &Config,
    res: Resources,
    images: Vec<Image>,
) -> IResult<(Outcome, Vec<Image>)> {
    sched::run(m, cfg, res, images, BTreeMap::new())
}
