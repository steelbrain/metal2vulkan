//! The route counts of the draws a rate-mapped pass routes on the GPU (the runtime's
//! `render/gpu_routes.rs`): from each draw's probe flag, the draw counts of its two routes, its
//! fan-out variant's and its region-copy variant's, which its indirect-count draws read.
//!
//! **The words.** A routed draw `i` of a dispatch takes [`PROBE_DRAW_WORDS`] words of the flags
//! from word `PROBE_DRAW_WORDS·i`, the probe's (its flag first, [`PROBE_DRAW_WORDS`]), and
//! [`ROUTE_WORDS`] words of the routes from word `ROUTE_WORDS·i`: its mark, its fan-out route's
//! draw count, and its region-copy route's. The consumer writes each draw's routes before the
//! dispatch: a mark of 0 and the counts 0 and [`MAX_SEGMENTS`], the region-copy route's alone,
//! which no dispatch changes; or a mark of [`ROUTE_ROUTED`] where it recorded the draw's probe and
//! both its routes.
//!
//! **The module.** One `GLCompute` entry `main` with local size ([`RouteCountsModule::LOCAL_SIZE`],
//! 1, 1), [`RouteCountsModule::PUSH_BYTES`] of `COMPUTE` push constants at offset 0 (the draws `n`
//! of the dispatch), and in set 0 two storage buffers: the flags
//! ([`RouteCountsModule::FLAGS_BINDING`]) and the routes ([`RouteCountsModule::ROUTES_BINDING`]).
//! Invocation `i` below `n` whose mark is [`ROUTE_ROUTED`] writes [`ROUTE_DECIDED`] over it, and
//! the counts: 1 and 0 where the draw's flag is 0 (the probe flagged none of its triangles), else
//! 0 and [`MAX_SEGMENTS`]. Every other invocation writes nothing. So exactly one of a draw's two
//! counts is nonzero: the fan-out route draws once, at most one draw, and the region-copy route
//! draws each of a recording's regions, at most [`MAX_SEGMENTS`] (a row holds no more), whatever
//! its recording's draw count.
//!
//! **Synchronization** (consumer): the probe's writes of the flags → `SHADER_STORAGE_READ` in
//! `COMPUTE_SHADER` before the dispatch; its writes of the routes → `INDIRECT_COMMAND_READ` in
//! `DRAW_INDIRECT`, and `HOST_READ` where the host reads the marks after the command buffer
//! completed.

use std::sync::Arc;

use super::{MAX_SEGMENTS, PROBE_DRAW_WORDS};
use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass, Type,
    TypedModule,
};

/// The words of a routed draw's routes (the module docs): its mark, its fan-out route's draw
/// count, and its region-copy route's.
pub const ROUTE_WORDS: u32 = 3;

/// The mark of a draw whose probe and both routes the consumer recorded: the dispatch decides its
/// counts.
pub const ROUTE_ROUTED: u32 = 1;

/// The mark the dispatch leaves on a draw whose counts it wrote.
pub const ROUTE_DECIDED: u32 = 2;

/// The crate-owned route-counts module on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteCountsModule {
    /// SPIR-V 1.5 words, entry point `main`, its bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl RouteCountsModule {
    /// Descriptor set of the bindings.
    pub const SET: u32 = 0;
    /// The probe's flags, a storage buffer the dispatch reads.
    pub const FLAGS_BINDING: u32 = 0;
    /// The draws' routes, a storage buffer the dispatch reads and writes.
    pub const ROUTES_BINDING: u32 = 1;
    /// Invocations of one workgroup: one draw each.
    pub const LOCAL_SIZE: u32 = 64;
    /// Bytes of the `COMPUTE` push-constant range at offset 0: the draws of the dispatch.
    pub const PUSH_BYTES: u32 = 4;

    /// The workgroups of a dispatch over `draws` draws.
    pub fn groups(draws: u32) -> u32 {
        draws.div_ceil(Self::LOCAL_SIZE)
    }
}

/// The binding symbol of the flags in the typed module.
const FLAGS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};

/// The binding symbol of the routes in the typed module.
const ROUTES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};

/// The push field the draws are declared under: a crate-owned module's own word, which no
/// selection places.
const DRAWS: PushField = PushField::DispatchRegion;

/// A `Block { u32 words[]; }` storage buffer bound at `sym`.
fn words_buffer(b: &mut ModuleBuilder, sym: ResourceSlot, u32_: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(sym, StorageClass::StorageBuffer, block)
}

fn u32_value(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// The route-counts module as a typed module, before serialization: the module
/// [`route_counts_module`] serializes, which the CPU interpreter runs. Its bindings are the
/// symbols `FLAGS` and `ROUTES`, and its push field the draws, a `u32` (the module docs).
fn typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let m_draws = b.push_member(DRAWS, u32_, 0)?;
    let push = b.push_global()?;
    let flags = words_buffer(&mut b, FLAGS, u32_)?;
    let routes = words_buffer(&mut b, ROUTES, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let one = u32_value(&mut f, 1)?;
    let two = u32_value(&mut f, 2)?;
    let routed = u32_value(&mut f, ROUTE_ROUTED)?;
    let decided = u32_value(&mut f, ROUTE_DECIDED)?;
    let most = u32_value(&mut f, MAX_SEGMENTS)?;
    let draw_words = u32_value(&mut f, PROBE_DRAW_WORDS)?;
    let route_words = u32_value(&mut f, ROUTE_WORDS)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let i = f.composite_extract(invocation, &[0])?;
    let push = f.module().global_value(push)?;
    let member = u32_value(&mut f, m_draws.index)?;
    let draws = f.access_chain(push, &[member])?;
    let draws = f.load(draws, MemAccess::default())?;
    let flags = f.module().global_value(flags)?;
    let routes = f.module().global_value(routes)?;
    let inside = f.binary(BinOp::ULessThan, i, draws)?;
    let decide: ArmFn<'_, ()> = Box::new(move |f, _| {
        let at = f.binary(BinOp::IMul, i, route_words)?;
        let mark_ptr = f.access_chain(routes, &[zero, at])?;
        let mark = f.load(mark_ptr, MemAccess::default())?;
        let marked = f.binary(BinOp::IEqual, mark, routed)?;
        let write: ArmFn<'_, ()> = Box::new(move |f, _| {
            let flag_at = f.binary(BinOp::IMul, i, draw_words)?;
            let flag_ptr = f.access_chain(flags, &[zero, flag_at])?;
            let flag = f.load(flag_ptr, MemAccess::default())?;
            let fanned = f.binary(BinOp::IEqual, flag, zero)?;
            let fan = f.select(fanned, one, zero)?;
            let copies = f.select(fanned, zero, most)?;
            f.store(mark_ptr, decided, MemAccess::default())?;
            for (offset, count) in [(one, fan), (two, copies)] {
                let word = f.binary(BinOp::IAdd, at, offset)?;
                let ptr = f.access_chain(routes, &[zero, word])?;
                f.store(ptr, count, MemAccess::default())?;
            }
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), marked, write, None)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, decide, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: RouteCountsModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(RouteCountsModule::SET),
        Symbol::Binding(s) if *s == FLAGS => Some(RouteCountsModule::FLAGS_BINDING),
        Symbol::Binding(s) if *s == ROUTES => Some(RouteCountsModule::ROUTES_BINDING),
        _ => None,
    }
}

/// The route-counts module of `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed bindings of [`RouteCountsModule`].
pub fn route_counts_module(device: &Device) -> Result<RouteCountsModule, Error> {
    let profile = device.profile();
    let m = typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the route counts need {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(RouteCountsModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        typed_module, RouteCountsModule, FLAGS, ROUTES, ROUTE_DECIDED, ROUTE_ROUTED, ROUTE_WORDS,
    };
    use crate::bind::fan_out::{MAX_SEGMENTS, PROBE_DRAW_WORDS};
    use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    #[test]
    fn the_route_counts_kernel_is_valid_on_every_committed_profile() {
        for (_, p) in committed_profiles().expect("profiles") {
            assert_valid(typed_module().expect("module"), &p);
        }
    }

    fn bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn words(bytes: &[u8]) -> Vec<u32> {
        let chunks = bytes.as_chunks::<4>().0;
        chunks.iter().map(|c| u32::from_le_bytes(*c)).collect()
    }

    /// The routes after one dispatch over the first `draws` of `flags` (each draw's flag word)
    /// and `marks`, in either invocation order, each draw's routes written as the consumer writes
    /// them, its flags' other words 0.
    fn decide(flags: &[u32], marks: &[u32], draws: u32) -> Vec<[u32; 3]> {
        let mut flag_words = vec![0u32; flags.len() * PROBE_DRAW_WORDS as usize];
        for (i, &flag) in flags.iter().enumerate() {
            flag_words[i * PROBE_DRAW_WORDS as usize] = flag;
        }
        let route_words: Vec<u32> = (marks.iter())
            .flat_map(|&mark| [mark, 0, MAX_SEGMENTS])
            .collect();
        let m = typed_module().expect("module");
        let mut out = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let res = Resources {
                buffers: vec![
                    Buffer {
                        slot: Some(FLAGS),
                        address: None,
                        bytes: bytes(&flag_words),
                    },
                    Buffer {
                        slot: Some(ROUTES),
                        address: None,
                        bytes: bytes(&route_words),
                    },
                ],
                push: draws.to_le_bytes().to_vec(),
                spec: Default::default(),
            };
            let cfg = Config {
                order,
                workgroups: [RouteCountsModule::groups(draws), 1, 1],
                ..Config::default()
            };
            out.push(run(&m, &cfg, res).expect("run").buffers);
        }
        assert_eq!(out[0], out[1], "order independent");
        let flags_after = words(&out[0][0]);
        assert_eq!(flags_after, flag_words, "the flags are only read");
        let routes = words(&out[0][1]);
        assert_eq!(routes.len(), marks.len() * ROUTE_WORDS as usize);
        (routes.as_chunks::<3>().0).to_vec()
    }

    /// A routed draw the probe flagged nothing of takes its fan-out route alone, and a flagged
    /// one its region-copy route alone, each marked decided; a draw not routed keeps the
    /// region-copy route the consumer wrote, unmarked.
    #[test]
    fn a_routed_draw_takes_exactly_one_route_by_its_flag() {
        let routed = ROUTE_ROUTED;
        let got = decide(&[0, 1, 0, 1], &[routed, routed, 0, 0], 4);
        assert_eq!(
            got,
            vec![
                [ROUTE_DECIDED, 1, 0],
                [ROUTE_DECIDED, 0, MAX_SEGMENTS],
                [0, 0, MAX_SEGMENTS],
                [0, 0, MAX_SEGMENTS],
            ]
        );
    }

    /// The draws past the dispatch's count keep their words, marked or not, over more than one
    /// workgroup.
    #[test]
    fn draws_past_the_count_are_left_as_written() {
        let n = RouteCountsModule::LOCAL_SIZE as usize + 3;
        let flags = vec![0u32; n + 2];
        let marks = vec![ROUTE_ROUTED; n + 2];
        let got = decide(&flags, &marks, n as u32);
        for (i, routes) in got.iter().enumerate() {
            let want = match i < n {
                true => [ROUTE_DECIDED, 1, 0],
                false => [ROUTE_ROUTED, 0, MAX_SEGMENTS],
            };
            assert_eq!(*routes, want, "draw {i}");
        }
    }
}
