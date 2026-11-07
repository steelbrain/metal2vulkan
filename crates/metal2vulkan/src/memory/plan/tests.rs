//! L0 tests of the memory analysis on test-builder IR: the provenance lattice, `Repr`, windows
//! against hand-computed ranges, arenas, typed allocas, constant globals and the blob, handle
//! classification, the Profile-F refusal, and ledger charging.

use super::*;
use crate::air::abi::{
    ArgTypeInfo, BufferArg, EntryPoint, IndirectArg, Member, MemberKind, ParamBinding, Slot,
    Stage as AbiStage, StructInfo, StructRef, TextureArg,
};
use crate::air::ir::build::{FunctionBuilder, ModuleBuilder as AirBuilder};
use crate::air::ir::{
    Align, BlockId, CastOp, DataLayout, FastMath, Global, IntPredicate, Linkage, Module,
};
use crate::device::requirements::VkFeature;
use crate::device::DevicePolicy;
use crate::error::{Capability, DiagName, Needs};
use crate::graph::{Cfg, DomTree};
use crate::memory::provenance::{Class, PrivForm, Repr, Root};
use crate::pipeline::testing::{meter, profile};

const AIR_DL: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

fn addressed() -> DeviceProfile {
    profile("moltenvk-m3", DevicePolicy::default()).expect("moltenvk-m3")
}

fn floor() -> DeviceProfile {
    profile("vk12-floor", DevicePolicy::default()).expect("vk12-floor")
}

fn slot(index: u32) -> Slot {
    Slot {
        index: AbiInt::Const(index),
        count: AbiInt::Const(1),
    }
}

fn buffer(index: u32, space: AddrSpace, buffer_size: Option<u32>) -> ParamRole {
    ParamRole::Buffer(BufferArg {
        location: slot(index),
        access: Access::ReadWrite,
        space,
        buffer_size,
        layout: None,
        raster_order_group: None,
    })
}

fn texture(index: u32) -> ParamRole {
    ParamRole::Texture(TextureArg {
        location: slot(index),
        access: Access::Read,
        raster_order_group: None,
    })
}

/// A kernel `k` whose parameters have `spaces` (pointers) and roles `roles`, with the body
/// `body` builds; the entry is what `specialize` would hand memory analysis.
struct Kernel {
    mb: AirBuilder,
    params: Vec<TypeId>,
    roles: Vec<ParamRole>,
    structs: Vec<StructInfo>,
    samplers: Vec<GlobalId>,
}

impl Kernel {
    fn new() -> Kernel {
        let mut mb = AirBuilder::new();
        mb.set_data_layout(DataLayout::parse(AIR_DL).expect("datalayout"));
        Kernel {
            mb,
            params: Vec::new(),
            roles: Vec::new(),
            structs: Vec::new(),
            samplers: Vec::new(),
        }
    }

    fn param(&mut self, space: AddrSpace, role: ParamRole) -> &mut Kernel {
        let p = self.mb.ptr(space);
        self.params.push(p);
        self.roles.push(role);
        self
    }

    fn build(mut self, body: impl FnOnce(&mut FunctionBuilder<'_>)) -> Entry {
        let void = self.mb.void();
        let fty = self.mb.fn_ty(void, &self.params);
        let mut fb = self.mb.define("k", fty);
        body(&mut fb);
        let func = fb.finish();
        let module = self.mb.finish().expect("module");
        entry(module, func, self.roles, self.structs, self.samplers)
    }
}

fn entry(
    module: Module,
    func: crate::air::ir::FuncId,
    roles: Vec<ParamRole>,
    structs: Vec<StructInfo>,
    samplers: Vec<GlobalId>,
) -> Entry {
    let mut m = meter();
    let body = module
        .func(func)
        .and_then(|f| f.body.as_ref())
        .expect("body");
    let cfg = Cfg::build(body, &mut m).expect("cfg");
    let dom = DomTree::compute(&cfg, &mut m).expect("dom");
    let params = roles
        .into_iter()
        .enumerate()
        .map(|(i, role)| ParamBinding {
            param: i as u32,
            role,
            unused: false,
            predicate: None,
            ty: ArgTypeInfo::default(),
            name: None,
        })
        .collect();
    let static_samplers = samplers
        .into_iter()
        .map(|g| {
            (
                g,
                crate::air::abi::SamplerState::from_words([0, 0]).expect("sampler"),
            )
        })
        .collect();
    Entry {
        func,
        ep: EntryPoint {
            func,
            stage: AbiStage::Kernel {
                max_total_threads: None,
            },
            params,
            outputs: Vec::new(),
            name: DiagName::new("k"),
            user_annotation: None,
        },
        absent_handles: Vec::new(),
        structs,
        language: None,
        compile: Default::default(),
        layout: module.data_layout().clone(),
        cfg,
        dom,
        static_samplers,
        linked_sampler_groups: Vec::new(),
        unnumbered_linked_samplers: false,
        intersection_candidates: false,
        function_constants: false,
        undefined_reads: Vec::new(),
        function_tables: Default::default(),
        lockstep: Default::default(),
        call_stack: None,
        unmeasured_speculation: false,
        module,
    }
}

fn align(n: u64) -> Option<Align> {
    Align::from_bytes(n)
}

fn load(fb: &mut FunctionBuilder<'_>, ty: TypeId, ptr: Value) -> Value {
    fb.push(
        ty,
        InstKind::Load {
            ptr,
            align: align(4),
            volatile: false,
            atomic: None,
            range: None,
        },
    )
}

fn store(fb: &mut FunctionBuilder<'_>, val: Value, ptr: Value) {
    fb.push_void(InstKind::Store {
        val,
        ptr,
        align: align(4),
        volatile: false,
        atomic: None,
    });
}

fn gep(fb: &mut FunctionBuilder<'_>, src_ty: TypeId, base: Value, indices: &[Value]) -> Value {
    let ty = fb.value_ty(base).expect("base type");
    fb.push(
        ty,
        InstKind::Gep {
            src_ty,
            base,
            indices: indices.into(),
            inbounds: true,
        },
    )
}

fn plan(e: &Entry, p: &DeviceProfile) -> Result<MemoryPlan, Error> {
    analyze_memory(e, p, &mut meter())
}

fn ok(e: &Entry) -> MemoryPlan {
    plan(e, &addressed()).expect("analyze")
}

fn window_of(p: &MemoryPlan, param: u32) -> Option<Window> {
    p.slots
        .iter()
        .find(|s| s.param == param)
        .and_then(|s| s.window)
}

fn repr_roots(p: &MemoryPlan, v: Value) -> Vec<Root> {
    match p.repr(v) {
        Some(Repr::BufPtr { roots, .. } | Repr::Handle { cands: roots, .. }) => {
            p.roots(roots).to_vec()
        }
        other => panic!("not a root-set repr: {other:?}"),
    }
}

fn refused_feature(r: Result<MemoryPlan, Error>) -> Feature {
    match r {
        Err(Error::Unsupported(u)) => u.feature,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

// ---- Buffer roots, offsets, windows ----

/// A store at a constant offset: one slot, its window the stored bytes, the pointer a
/// static-slot fat pointer (the walking skeleton's shape).
#[test]
fn a_constant_offset_store_is_a_static_slot_with_its_bytes_as_window() {
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, buffer(0, AddrSpace::Device, None));
    let mut p8 = Value::Arg(crate::air::ir::ArgIdx::new(0));
    let e = k.build(|fb| {
        let i32t = fb.module().int(32);
        let i64t = fb.module().int(64);
        let three = fb.module().const_int(i64t, 3);
        let v = fb.module().const_int(i32t, 7);
        p8 = gep(fb, i32t, fb.arg(0), &[three]);
        store(fb, v, p8);
        fb.ret(None);
    });
    let plan = ok(&e);
    assert_eq!(plan.slots.len(), 1);
    let s = plan.slots.first().copied().expect("slot");
    assert_eq!(s.access, AccessSet::WRITE);
    assert_eq!(s.window, Some(Window::Bounded { start: 12, len: 4 }));
    assert!(!s.guest_base);
    assert_eq!(plan.static_slot(p8), Some((0, 12)));
    assert!(matches!(
        plan.repr(p8),
        Some(Repr::BufPtr { wide: false, .. })
    ));
    assert_eq!(repr_roots(&plan, p8), vec![Root::Param(0)]);
    assert!(!plan.needs.dynamic_roots);
}

/// GEPs through a struct and an array compose to datalayout offsets, and a negative GEP gives a
/// negative window start (hand-computed: `{ i32, [4 x <3 x float>] }` puts element 2 at
/// 16 + 2·16 = 48; `float3` is 16 bytes by `v96:128`).
#[test]
fn struct_and_array_geps_and_negative_offsets_give_hand_computed_windows() {
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, buffer(0, AddrSpace::Device, None));
    let e = k.build(|fb| {
        let i32t = fb.module().int(32);
        let i64t = fb.module().int(64);
        let f = fb.module().float();
        let f3 = fb.module().vector(f, 3);
        let arr = fb.module().array(f3, 4);
        let st = fb.module().struct_ty(&[i32t, arr]);
        let (z64, one32, two64) = (
            fb.module().const_int(i64t, 0),
            fb.module().const_int(i32t, 1),
            fb.module().const_int(i64t, 2),
        );
        let p = gep(fb, st, fb.arg(0), &[z64, one32, two64]);
        let v = load(fb, f3, p);
        let neg = fb.module().const_int(i64t, (-4i64) as u64);
        let q = gep(fb, i32t, fb.arg(0), &[neg]);
        let x = load(fb, i32t, q);
        let _ = (v, x);
        fb.ret(None);
    });
    let plan = ok(&e);
    // Loads cover [48, 60) (a float3 stores 12 bytes) and [-16, -12) (index -4 of i32).
    assert_eq!(
        window_of(&plan, 0),
        Some(Window::Bounded {
            start: -16,
            len: 76
        })
    );
    assert_eq!(plan.slots.first().map(|s| s.access), Some(AccessSet::READ));
}

/// A dynamic index makes the offset unknown: `Whole` for an unsized pointer, `[0, buffer_size)`
/// for a reference parameter (`air.buffer_size`).
#[test]
fn a_dynamic_index_is_whole_unless_the_parameter_is_a_sized_reference() {
    for (size, want) in [
        (None, Window::Whole),
        (Some(64), Window::Bounded { start: 0, len: 64 }),
    ] {
        let mut k = Kernel::new();
        k.param(AddrSpace::Device, buffer(0, AddrSpace::Device, size));
        k.param(AddrSpace::Device, buffer(1, AddrSpace::Device, None));
        let e = k.build(|fb| {
            let i32t = fb.module().int(32);
            let i = load(fb, i32t, fb.arg(1));
            let p = gep(fb, i32t, fb.arg(0), &[i]);
            let v = fb.module().const_int(i32t, 1);
            store(fb, v, p);
            fb.ret(None);
        });
        let plan = ok(&e);
        assert_eq!(window_of(&plan, 0), Some(want));
        assert_eq!(
            window_of(&plan, 1),
            Some(Window::Bounded { start: 0, len: 4 })
        );
    }
}

/// A select of two parameters has both roots; each slot's window covers the access; the
/// roots are merged.
#[test]
fn a_select_of_two_parameters_has_both_roots() {
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, buffer(0, AddrSpace::Device, None));
    k.param(AddrSpace::Device, buffer(1, AddrSpace::Device, None));
    let mut sel = Value::Arg(crate::air::ir::ArgIdx::new(0));
    let e = k.build(|fb| {
        let i1 = fb.module().int(1);
        let i32t = fb.module().int(32);
        let ptr = fb.module().ptr(AddrSpace::Device);
        let c = fb.module().const_int(i1, 1);
        sel = fb.push(
            ptr,
            InstKind::Select {
                c,
                t: fb.arg(0),
                f: fb.arg(1),
                fmf: FastMath::NONE,
            },
        );
        let _ = load(fb, i32t, sel);
        fb.ret(None);
    });
    let plan = ok(&e);
    assert_eq!(repr_roots(&plan, sel), vec![Root::Param(0), Root::Param(1)]);
    for p in 0..2 {
        assert_eq!(
            window_of(&plan, p),
            Some(Window::Bounded { start: 0, len: 4 })
        );
    }
    assert_eq!(plan.static_slot(sel), None);
}

/// A loop that advances a pointer terminates by widening: the phi's offset becomes unknown and
/// the window `Whole`, with the root set unchanged.
#[test]
fn a_pointer_advanced_in_a_loop_widens_and_terminates() {
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, buffer(0, AddrSpace::Device, None));
    let mut phi = Value::Arg(crate::air::ir::ArgIdx::new(0));
    let e = k.build(|fb| {
        let i1 = fb.module().int(1);
        let i32t = fb.module().int(32);
        let i64t = fb.module().int(64);
        let ptr = fb.module().ptr(AddrSpace::Device);
        let one = fb.module().const_int(i64t, 1);
        let entry = BlockId::new(0);
        let (head, exit) = (fb.block(), fb.block());
        fb.br(head);
        fb.switch_to(head);
        // `phi` and `next` are numbered in this block; `next` is the phi's back-edge value.
        let next_id = Value::Inst(crate::air::ir::InstId::new(3));
        phi = fb.push(
            ptr,
            InstKind::Phi {
                incoming: Box::new([(fb.arg(0), entry), (next_id, head)]),
                fmf: FastMath::NONE,
            },
        );
        let v = fb.module().const_int(i32t, 5);
        store(fb, v, phi);
        let next = gep(fb, i32t, phi, &[one]);
        assert_eq!(next, next_id);
        let c = fb.push(
            i1,
            InstKind::ICmp {
                pred: IntPredicate::Eq,
                a: next,
                b: fb.arg(0),
            },
        );
        fb.cond_br(c, exit, head);
        fb.switch_to(exit);
        fb.ret(None);
    });
    let plan = ok(&e);
    assert_eq!(repr_roots(&plan, phi), vec![Root::Param(0)]);
    assert_eq!(plan.offset(phi), Some(OffsetRange::Top));
    assert_eq!(window_of(&plan, 0), Some(Window::Whole));
    // A same-root compare observes no guest base.
    assert!(!plan.slots.first().expect("slot").guest_base);
}

/// `ptrtoint` and a cross-root compare observe the slot's guest base.
#[test]
fn ptrtoint_and_cross_root_compares_observe_the_guest_base() {
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, buffer(0, AddrSpace::Device, None));
    k.param(AddrSpace::Device, buffer(1, AddrSpace::Device, None));
    k.param(AddrSpace::Device, buffer(2, AddrSpace::Device, None));
    let e = k.build(|fb| {
        let i1 = fb.module().int(1);
        let i64t = fb.module().int(64);
        let _ = fb.push(
            i64t,
            InstKind::Cast {
                op: CastOp::PtrToInt,
                v: fb.arg(0),
            },
        );
        let _ = fb.push(
            i1,
            InstKind::ICmp {
                pred: IntPredicate::Ult,
                a: fb.arg(1),
                b: fb.arg(2),
            },
        );
        fb.ret(None);
    });
    let plan = ok(&e);
    assert_eq!(plan.slots.len(), 3);
    assert!(plan.slots.iter().all(|s| s.guest_base));
    // The escaping slot is reached by the pointers recovered from its address; the compared ones
    // are not accessed.
    let windows: Vec<_> = plan.slots.iter().map(|s| s.window).collect();
    assert_eq!(windows, vec![Some(Window::Whole), None, None]);
    // Its pointer is wide: it joins the dynamic component.
    assert!(matches!(
        plan.repr(Value::Arg(crate::air::ir::ArgIdx::new(0))),
        Some(Repr::BufPtr { wide: true, .. })
    ));
}

// ---- Dynamic roots and Profile F ----

fn loaded_pointer_kernel() -> (Entry, Value) {
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, buffer(0, AddrSpace::Device, Some(8)));
    k.param(AddrSpace::Device, buffer(1, AddrSpace::Device, None));
    let mut q = Value::Arg(crate::air::ir::ArgIdx::new(0));
    let e = k.build(|fb| {
        let i32t = fb.module().int(32);
        let ptr = fb.module().ptr(AddrSpace::Device);
        q = load(fb, ptr, fb.arg(0));
        let v = fb.module().const_int(i32t, 9);
        store(fb, v, q);
        store(fb, v, fb.arg(1));
        fb.ret(None);
    });
    (e, q)
}

/// A pointer loaded from memory has the dynamic root: a wide fat pointer, every window `Whole`
/// (even a sized reference's), and Profile A accepts it.
#[test]
fn a_loaded_pointer_is_dynamic_wide_and_makes_every_window_whole() {
    let (e, q) = loaded_pointer_kernel();
    let plan = ok(&e);
    assert!(plan.needs.dynamic_roots);
    assert!(matches!(
        plan.repr(q),
        Some(Repr::BufPtr { wide: true, .. })
    ));
    assert_eq!(repr_roots(&plan, q), vec![Root::Dynamic]);
    assert_eq!(window_of(&plan, 0), Some(Window::Whole));
    assert_eq!(window_of(&plan, 1), Some(Window::Whole));
}

/// Profile F refuses the dereference of a dynamic root, naming `bufferDeviceAddress` and
/// `shaderInt64` (both missing on `vk12-floor`), at the dereferencing instruction.
#[test]
fn profile_f_refuses_a_dynamic_dereference_naming_buffer_device_address() {
    let (e, _) = loaded_pointer_kernel();
    let f = floor();
    assert_eq!(f.fields().buffer_access, BufferAccess::Floor);
    match plan(&e, &f) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::DereferenceLoadedPointer);
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![vec![
                    Capability::Feature(VkFeature::BufferDeviceAddress),
                    Capability::Feature(VkFeature::ShaderInt64),
                ]])
            );
            assert!(u.location.is_some());
        }
        other => panic!("{other:?}"),
    }
}

/// Profile F accepts a loaded pointer that is never dereferenced (only converted back to an
/// integer): no device address is needed.
#[test]
fn profile_f_accepts_a_loaded_pointer_that_is_never_dereferenced() {
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, buffer(0, AddrSpace::Device, None));
    let e = k.build(|fb| {
        let i64t = fb.module().int(64);
        let ptr = fb.module().ptr(AddrSpace::Device);
        let q = load(fb, ptr, fb.arg(0));
        let _ = fb.push(
            i64t,
            InstKind::Cast {
                op: CastOp::PtrToInt,
                v: q,
            },
        );
        fb.ret(None);
    });
    let plan = plan(&e, &floor()).expect("analyze");
    assert!(plan.needs.dynamic_roots);
}

// ---- Allocas and the private arena ----

/// A private array read and written through constant and dynamic element indices of its own
/// element type is typed; a punned alloca and two allocas merged by a select go to the private
/// arena at aligned offsets.
#[test]
fn typed_allocas_and_the_private_arena() {
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, buffer(0, AddrSpace::Device, None));
    let mut ptrs = Vec::new();
    let e = k.build(|fb| {
        let i1 = fb.module().int(1);
        let i32t = fb.module().int(32);
        let i64t = fb.module().int(64);
        let f = fb.module().float();
        let arr = fb.module().array(f, 4);
        let p0 = fb.module().ptr(AddrSpace::Thread);
        let one = fb.module().const_int(i32t, 1);
        let z = fb.module().const_int(i64t, 0);
        let two = fb.module().const_int(i64t, 2);
        let alloca = |fb: &mut FunctionBuilder<'_>, ty| {
            fb.push(
                p0,
                InstKind::Alloca {
                    ty,
                    count: one,
                    align: align(4),
                },
            )
        };
        let typed = alloca(fb, arr);
        let punned = alloca(fb, i32t);
        let m1 = alloca(fb, i64t);
        let m2 = alloca(fb, i32t);
        let i = load(fb, i64t, fb.arg(0));
        let at2 = gep(fb, arr, typed, &[z, two]);
        let fv = fb.module().constant(f, Const::Float(0x3f80_0000));
        store(fb, Value::Const(fv), at2);
        let ati = gep(fb, arr, typed, &[z, i]);
        let _ = load(fb, f, ati);
        let iv = fb.module().const_int(i32t, 3);
        store(fb, iv, punned);
        let _ = load(fb, f, punned);
        let c = fb.module().const_int(i1, 0);
        let s = fb.push(
            p0,
            InstKind::Select {
                c,
                t: m1,
                f: m2,
                fmf: FastMath::NONE,
            },
        );
        let _ = load(fb, i32t, s);
        ptrs.extend([typed, punned, m1, m2, s, ati]);
        fb.ret(None);
    });
    let plan = ok(&e);
    let [typed, punned, m1, m2, s, ati] = ptrs[..] else {
        unreachable!()
    };
    assert_eq!(plan.typed_allocas.len(), 1);
    assert_eq!(plan.repr(typed), Some(Repr::PrivPtr(PrivForm::Typed(0))));
    assert_eq!(plan.repr(ati), Some(Repr::PrivPtr(PrivForm::Typed(0))));
    for v in [punned, m1, m2, s] {
        assert_eq!(plan.repr(v), Some(Repr::PrivPtr(PrivForm::Arena)));
    }
    // punned (i32 at 0), m1 (i64 at 8), m2 (i32 at 16): exact sizes, natural alignment.
    let offsets: Vec<(u64, u64)> = plan
        .priv_arena
        .entries
        .iter()
        .map(|e| (e.offset, e.size))
        .collect();
    assert_eq!(offsets, vec![(0, 4), (8, 8), (16, 4)]);
    assert_eq!(plan.priv_arena.static_size, 20);
}

// ---- Threadgroup arena ----

/// Static threadgroup globals sit in the reverse of the order of their last references, each at
/// its alignment and at least 4, as Metal places them; the arena's size is exactly the allocations
/// plus padding, and threadgroup parameters follow as dynamic regions. Previously `a` sat at 0 and
/// `b` at 16.
#[test]
fn the_threadgroup_arena_is_exact() {
    let mut k = Kernel::new();
    k.param(
        AddrSpace::Threadgroup,
        ParamRole::Threadgroup(BufferArg {
            location: slot(0),
            access: Access::ReadWrite,
            space: AddrSpace::Threadgroup,
            buffer_size: None,
            layout: None,
            raster_order_group: None,
        }),
    );
    let i8t = k.mb.int(8);
    let f = k.mb.float();
    let f4 = k.mb.vector(f, 4);
    let bytes = k.mb.array(i8t, 3);
    let big = k.mb.array(f4, 2);
    let g0 = k.mb.simple_global("a", bytes, AddrSpace::Threadgroup, None);
    let g1 = k.mb.simple_global("b", big, AddrSpace::Threadgroup, None);
    let (a0, a1) = (k.mb.global_addr(g0), k.mb.global_addr(g1));
    let e = k.build(|fb| {
        let i8 = fb.module().int(8);
        let _ = load(fb, i8, a0);
        let _ = load(fb, f4, a1);
        let _ = load(fb, f4, fb.arg(0));
        fb.ret(None);
    });
    let plan = ok(&e);
    let entries: Vec<(u64, u64, u64)> = plan
        .tg_arena
        .entries
        .iter()
        .map(|e| (e.offset, e.size, e.align))
        .collect();
    // In object order: `a` (3 bytes, aligned to 4) after `b` (32 bytes at 0), named last.
    assert_eq!(entries, vec![(32, 3, 4), (0, 32, 16)]);
    assert_eq!(plan.tg_arena.static_size, 35);
    assert!(plan.tg_arena.metal_layout);
    assert_eq!(
        plan.tg_arena
            .entry(ArenaObject::Global(g0))
            .map(|e| e.offset),
        Some(32)
    );
    assert_eq!(plan.tg_arena.dynamic.len(), 1);
    assert_eq!(plan.repr(a1), Some(Repr::TgPtr));
}

/// The order is the last references', not the module's or the first references': `a`, `b`, `c`
/// named `c`, `a`, `b`, then `b`, `a`, `c` lie `c`, `a`, `b` from 0. An instruction naming two of
/// them (a `select` of two addresses) orders them by no measured rule, so the layout is not Metal's
/// ([`ArenaLayout::metal_layout`]) and no integer address reads it. Byte loads of their words keep
/// them in the arena.
#[test]
fn threadgroup_globals_lie_in_the_reverse_of_their_last_references() {
    let layout = |merged: bool| {
        let mut k = Kernel::new();
        let i32t = k.mb.int(32);
        let i8t = k.mb.int(8);
        let words = k.mb.array(i32t, 4);
        let g: Vec<GlobalId> = ["a", "b", "c"]
            .iter()
            .map(|n| k.mb.simple_global(n, words, AddrSpace::Threadgroup, None))
            .collect();
        let addr: Vec<Value> = g.iter().map(|&x| k.mb.global_addr(x)).collect();
        let e = k.build(|fb| {
            let i1 = fb.module().int(1);
            let ptr = fb.module().ptr(AddrSpace::Threadgroup);
            for i in [2, 0, 1, 1, 0, 2] {
                let _ = load(fb, i8t, addr[i]);
            }
            if merged {
                let c = fb.module().const_int(i1, 1);
                let sel = fb.push(
                    ptr,
                    InstKind::Select {
                        c,
                        t: addr[0],
                        f: addr[1],
                        fmf: FastMath::NONE,
                    },
                );
                let _ = load(fb, i32t, sel);
            }
            fb.ret(None);
        });
        let plan = ok(&e);
        let offsets: Vec<u64> = g
            .iter()
            .map(|&x| {
                plan.tg_arena
                    .entry(ArenaObject::Global(x))
                    .map(|e| e.offset)
                    .expect("placed")
            })
            .collect();
        (offsets, plan.tg_arena.metal_layout)
    };
    assert_eq!(layout(false), (vec![16, 32, 0], true));
    assert!(!layout(true).1);
}

// ---- Constant globals and the blob ----

/// A closed constant table read through a dynamic element index is a typed `Private`; one whose
/// address escapes through `ptrtoint` is placed in the blob with its initializer's bytes.
#[test]
fn closed_constants_are_private_and_escaping_ones_go_to_the_blob() {
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, buffer(0, AddrSpace::Device, None));
    let i32t = k.mb.int(32);
    let i64t = k.mb.int(64);
    let arr = k.mb.array(i32t, 2);
    let (c1, c2) = (
        k.mb.constant(i32t, Const::Int(0x0102_0304)),
        k.mb.constant(i32t, Const::Int(0xAABB_CCDD)),
    );
    let init = k.mb.constant(arr, Const::Aggregate(Box::new([c1, c2])));
    let name = k.mb.sym("table");
    let table = k.mb.global(Global {
        name,
        value_ty: arr,
        space: AddrSpace::Constant,
        constant: true,
        init: Some(init),
        externally_initialized: false,
        visibility: Default::default(),
        unnamed_addr: Default::default(),
        linkage: Linkage::Internal,
        align: None,
        section: None,
    });
    let escaping =
        k.mb.simple_global("escaping", arr, AddrSpace::Constant, Some(init));
    let (ta, ea) = (k.mb.global_addr(table), k.mb.global_addr(escaping));
    let e = k.build(|fb| {
        let z = fb.module().const_int(i64t, 0);
        let i = load(fb, i64t, fb.arg(0));
        let p = gep(fb, arr, ta, &[z, i]);
        let _ = load(fb, i32t, p);
        let _ = fb.push(
            i64t,
            InstKind::Cast {
                op: CastOp::PtrToInt,
                v: ea,
            },
        );
        fb.ret(None);
    });
    let plan = ok(&e);
    assert_eq!(
        plan.const_globals,
        vec![
            ConstGlobalPlan {
                global: table,
                form: ConstGlobalForm::Private
            },
            ConstGlobalPlan {
                global: escaping,
                form: ConstGlobalForm::Blob { offset: 0 }
            },
        ]
    );
    assert_eq!(plan.repr(ta), Some(Repr::ConstPtr(ConstForm::Private(0))));
    assert!(matches!(
        plan.repr(ea),
        Some(Repr::BufPtr { wide: false, .. })
    ));
    assert_eq!(
        plan.blob.as_ref().map(|b| b.bytes.to_vec()),
        Some(vec![0x04, 0x03, 0x02, 0x01, 0xDD, 0xCC, 0xBB, 0xAA])
    );
}

// ---- Typed shapes ----

/// A kernel whose module holds the struct DAG `%e{k} = { %e{k-1}, %e{k-1} }` over `%e0 = i32`,
/// `depth` levels (`2^depth` leaves through `depth + 1` types), and the matching constant DAG
/// `c{k} = { c{k-1}, c{k-1} }` over `c0 = i32 1` (constants are interned, so each level's two
/// parts are one constant): the entry, the top type, and the top constant.
fn struct_dag(depth: u32) -> (Entry, TypeId, ConstId) {
    let mut k = Kernel::new();
    let mut t = k.mb.int(32);
    let mut c = k.mb.constant(t, Const::Int(1));
    for _ in 0..depth {
        let s = k.mb.struct_ty(&[t, t]);
        c = k.mb.constant(s, Const::Aggregate(Box::new([c, c])));
        t = s;
    }
    let e = k.build(|fb| fb.ret(None));
    (e, t, c)
}

/// A typed object's shape and initializer hold one node per distinct type (and constant), built in
/// one work unit per type reached: for the 40-level struct DAG, 41 nodes and 81 units (the type and
/// both members of each of its 40 structs) each, and 41 SPIR-V types and 41 constants. The nodes
/// stay on the ledger until the shape is released, and the record of built types does not. Built
/// per path, the shape would be `2^41 - 1` nodes.
#[test]
fn a_typed_shape_and_its_initializer_are_built_once_per_type() {
    use crate::memory::alloca;
    let (e, ty, c) = struct_dag(40);
    let mut m = meter();
    let shape = alloca::typed_ty(&e, &addressed(), ty, false, &mut m)
        .expect("no budget stop")
        .expect("a typed shape");
    assert_eq!(shape.nodes().len(), 41);
    assert_eq!(m.work(), 81);
    // Level k's tree is `2^(k+1) - 1` nodes (the `i32` is level 0): `2^42 - 2 - 41` in all.
    assert_eq!(shape.unshared(), (1 << 42) - 2 - 41);
    let held = m.ledger();
    let nodes = 41 * std::mem::size_of::<alloca::TypedNode>() + 40 * 2 * 4;
    assert!(
        held >= nodes as u64,
        "{held} ledger bytes for {nodes} bytes of nodes"
    );
    assert!(
        m.ledger_peak() > held,
        "the record of built types was not charged"
    );

    let init = alloca::init_of(&e, c, &shape, &mut m).expect("an initializer");
    assert_eq!(init.nodes().len(), 41);
    assert_eq!(m.work(), 162);

    let mut mb = crate::spirv::builder::ModuleBuilder::new();
    let (root, types) = alloca::spirv_types(&mut mb, &shape).expect("types");
    assert_eq!(types.len(), 41);
    assert_eq!(types.last(), Some(&root));
    assert_eq!(mb.types().len(), 41);
    alloca::declare_init(&mut mb, &shape, &types, &init).expect("constant");
    assert_eq!(mb.consts().len(), 41);

    let mut m = meter();
    let shape = alloca::typed_ty(&e, &addressed(), ty, false, &mut m)
        .expect("no budget stop")
        .expect("a typed shape");
    shape.release(&mut m);
    assert_eq!(
        m.ledger(),
        0,
        "the shape's bytes are all it leaves on the ledger"
    );
}

/// The typed depth bound (`MAX_DEPTH`, 64 types below the object's) exactly where the tree walk had
/// it. `x` is a chain of 60 single-member structs over `i32`, and `{ x, w }` reaches `x` again
/// through `w`, `j` more single-member structs: its deepest `i32` is `j + 60` types below. The
/// second visit finds `x` built at depth 1; at `j = 4` it is accepted, at `j = 5` refused, as
/// rebuilding `x` there would be.
#[test]
fn the_typed_depth_bound_is_unchanged_by_building_each_type_once() {
    use crate::memory::alloca;
    let shaped = |j: u32| {
        let mut k = Kernel::new();
        let mut x = k.mb.int(32);
        for _ in 0..59 {
            x = k.mb.struct_ty(&[x]);
        }
        let mut w = x;
        for _ in 0..j {
            w = k.mb.struct_ty(&[w]);
        }
        let top = k.mb.struct_ty(&[x, w]);
        let e = k.build(|fb| fb.ret(None));
        alloca::typed_ty(&e, &addressed(), top, false, &mut meter()).expect("no budget stop")
    };
    assert!(shaped(4).is_ok(), "{:?}", shaped(4));
    assert_eq!(shaped(5).err(), Some(not_lowered()));
}

/// A typed constant's vector lanes are held with its initializer and charged, 8 bytes a lane,
/// before they are built: a `<4 x i32>` constant's initializer holds exactly 32 bytes more than an
/// `i32`'s. An initializer refused at a lane (a `ptrtoint` constant expression) returns the ledger
/// to what it held before.
#[test]
fn a_typed_constant_s_vector_lanes_are_charged() {
    use crate::memory::alloca;
    let mut k = Kernel::new();
    let i32t = k.mb.int(32);
    let v4 = k.mb.vector(i32t, 4);
    let one = k.mb.constant(i32t, Const::Int(1));
    let lanes = k.mb.constant(v4, Const::Aggregate(Box::new([one; 4])));
    let g =
        k.mb.simple_global("lane", i32t, AddrSpace::Constant, Some(one));
    let Value::Const(addr) = k.mb.global_addr(g) else {
        panic!("a global's address is a constant");
    };
    let cast = k.mb.constant(
        i32t,
        Const::Expr(crate::air::ir::ConstExpr::Cast {
            op: CastOp::PtrToInt,
            ops: [addr],
        }),
    );
    let bad =
        k.mb.constant(v4, Const::Aggregate(Box::new([one, one, one, cast])));
    let e = k.build(|fb| fb.ret(None));
    let init_bytes = |ty, c| {
        let mut m = meter();
        let shape = alloca::typed_ty(&e, &addressed(), ty, false, &mut m)
            .expect("no budget stop")
            .expect("a typed shape");
        let held = m.ledger();
        let init = alloca::init_of(&e, c, &shape, &mut m).map(|_| ());
        (init, m.ledger() - held)
    };
    let (scalar, s) = init_bytes(i32t, one);
    let (vector, v) = init_bytes(v4, lanes);
    assert_eq!((scalar, vector), (Ok(()), Ok(())));
    assert_eq!(
        v - s,
        32,
        "{v} bytes for the vector's initializer, {s} for the scalar's"
    );
    let (refused, r) = init_bytes(v4, bad);
    assert_eq!(refused, Err(not_lowered()));
    assert_eq!(r, 0, "a refused initializer left {r} bytes on the ledger");
}

/// The typed objects of one function declare at most `MAX_UNSHARED` of unshared type size,
/// constants first, then allocas in instruction order; an object past what is left is held
/// untyped. A 13-level struct DAG is 32,752 (`2^15 - 2 - 14`), so two such allocas are typed and
/// the third has its place in the private arena.
#[test]
fn typed_objects_past_the_function_s_room_are_held_untyped() {
    let mut k = Kernel::new();
    let mut t = k.mb.int(32);
    for _ in 0..13 {
        t = k.mb.struct_ty(&[t, t]);
    }
    let e = k.build(|fb| {
        let i32t = fb.module().int(32);
        let one = fb.module().const_int(i32t, 1);
        let p0 = fb.module().ptr(AddrSpace::Thread);
        for _ in 0..3 {
            fb.push(
                p0,
                InstKind::Alloca {
                    ty: t,
                    count: one,
                    align: align(4),
                },
            );
        }
        fb.ret(None);
    });
    let plan = ok(&e);
    assert_eq!(plan.typed_allocas.len(), 2);
    assert_eq!(plan.priv_arena.entries.len(), 1);
    assert_eq!(plan.priv_arena.static_size, 4 << 13);
}

/// A private arena whose size passes `u64`, by one alloca's count or by placing the next alloca, is
/// past the arena's `u32` offsets: `LimitExceeded(PrivateArenaBytes)`, as selection refuses an
/// arena past `u32::MAX` bytes. A `[2^61 x i32]` (`2^63` bytes) is longer than a typed object's
/// array, so it has its place in the arena.
#[test]
fn a_private_arena_past_u64_is_past_its_bound() {
    for (allocas, count) in [(2, 1u64), (1, 4)] {
        let mut k = Kernel::new();
        let i32t = k.mb.int(32);
        let big = k.mb.array(i32t, 1 << 61);
        let e = k.build(|fb| {
            let i64t = fb.module().int(64);
            let n = fb.module().const_int(i64t, count);
            let p0 = fb.module().ptr(AddrSpace::Thread);
            for _ in 0..allocas {
                fb.push(
                    p0,
                    InstKind::Alloca {
                        ty: big,
                        count: n,
                        align: align(4),
                    },
                );
            }
            fb.ret(None);
        });
        assert_eq!(
            plan(&e, &floor()).err(),
            Some(Error::LimitExceeded(Limit::PrivateArenaBytes)),
            "{allocas} allocas of {count}"
        );
    }
}

// ---- Handles ----

/// Texture parameters merged by a select are one handle over both candidates, demoted to a
/// handle index.
#[test]
fn merged_textures_are_one_handle_over_both_candidates() {
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, texture(0));
    k.param(AddrSpace::Device, texture(1));
    let mut sel = Value::Arg(crate::air::ir::ArgIdx::new(0));
    let e = k.build(|fb| {
        let i1 = fb.module().int(1);
        let ptr = fb.module().ptr(AddrSpace::Device);
        let c = fb.module().const_int(i1, 1);
        sel = fb.push(
            ptr,
            InstKind::Select {
                c,
                t: fb.arg(0),
                f: fb.arg(1),
                fmf: FastMath::NONE,
            },
        );
        let _ = fb.push(ptr, InstKind::Freeze(sel));
        fb.ret(None);
    });
    let plan = ok(&e);
    assert!(matches!(
        plan.repr(sel),
        Some(Repr::Handle {
            class: HandleClass::Texture,
            ..
        })
    ));
    assert_eq!(repr_roots(&plan, sel), vec![Root::Param(0), Root::Param(1)]);
    assert!(plan.slots.is_empty());
    assert_eq!(plan.needs.handle_table, None);
}

/// A handle reaching a GEP, or merged with a buffer pointer, is `HandleAsPointer`.
#[test]
fn a_handle_used_as_bytes_or_merged_with_bytes_is_refused() {
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, texture(0));
    let e = k.build(|fb| {
        let i8t = fb.module().int(8);
        let i64t = fb.module().int(64);
        let one = fb.module().const_int(i64t, 1);
        let _ = gep(fb, i8t, fb.arg(0), &[one]);
        fb.ret(None);
    });
    assert_eq!(
        refused_feature(plan(&e, &addressed())),
        Feature::HandleAsPointer
    );

    let mut k = Kernel::new();
    k.param(AddrSpace::Device, texture(0));
    k.param(AddrSpace::Device, buffer(1, AddrSpace::Device, None));
    let e = k.build(|fb| {
        let i1 = fb.module().int(1);
        let ptr = fb.module().ptr(AddrSpace::Device);
        let c = fb.module().const_int(i1, 1);
        let _ = fb.push(
            ptr,
            InstKind::Select {
                c,
                t: fb.arg(0),
                f: fb.arg(1),
                fmf: FastMath::NONE,
            },
        );
        fb.ret(None);
    });
    assert_eq!(
        refused_feature(plan(&e, &addressed())),
        Feature::HandleAsPointer
    );
}

/// A texture loaded from an argument-buffer member (`air.struct_type_info` kind) is a handle
/// with the dynamic candidate, so the function needs the handle table for textures; the
/// argument buffer itself is a read slot with the member's bytes as its window.
#[test]
fn an_argument_buffer_texture_member_seeds_a_dynamic_handle() {
    let mut k = Kernel::new();
    k.structs.push(StructInfo {
        members: vec![
            Member {
                offset: 0,
                size: 4,
                array_len: 0,
                layout: None,
                kind: MemberKind::Plain,
                render_target: None,
                raster_order_group: None,
                type_name: DiagName::new("uint"),
                name: DiagName::new("n"),
                key: crate::air::abi::UserKey(b"n".as_slice().into()),
            },
            Member {
                offset: 8,
                size: 8,
                array_len: 0,
                layout: None,
                kind: MemberKind::Indirect(IndirectArg {
                    ordinal: 1,
                    role: texture(0),
                    ty: ArgTypeInfo::default(),
                    name: None,
                }),
                render_target: None,
                raster_order_group: None,
                type_name: DiagName::new("texture2d<float>"),
                name: DiagName::new("t"),
                key: crate::air::abi::UserKey(b"t".as_slice().into()),
            },
        ],
    });
    k.param(
        AddrSpace::Constant,
        ParamRole::ArgumentBuffer(BufferArg {
            location: slot(0),
            access: Access::Read,
            space: AddrSpace::Constant,
            buffer_size: Some(16),
            layout: Some(StructRef(0)),
            raster_order_group: None,
        }),
    );
    let mut t = Value::Arg(crate::air::ir::ArgIdx::new(0));
    let e = k.build(|fb| {
        let i8t = fb.module().int(8);
        let i64t = fb.module().int(64);
        let ptr = fb.module().ptr(AddrSpace::Device);
        let eight = fb.module().const_int(i64t, 8);
        let at = gep(fb, i8t, fb.arg(0), &[eight]);
        t = fb.push(
            ptr,
            InstKind::Load {
                ptr: at,
                align: align(8),
                volatile: false,
                atomic: None,
                range: None,
            },
        );
        let _ = fb.push(ptr, InstKind::Freeze(t));
        fb.ret(None);
    });
    let plan = ok(&e);
    assert!(matches!(
        plan.repr(t),
        Some(Repr::Handle {
            class: HandleClass::Texture,
            ..
        })
    ));
    assert_eq!(plan.needs.handle_table, Some(vec![HandleClass::Texture]));
    // A dynamic handle is no dynamic buffer root.
    assert!(!plan.needs.dynamic_roots);
    let s = plan.slots.first().copied().expect("slot");
    assert!(s.argument_buffer);
    assert_eq!(s.access, AccessSet::READ);
    assert_eq!(s.window, Some(Window::Bounded { start: 8, len: 8 }));
}

/// A constexpr sampler's address is a sampler handle (seeded from the entry's static samplers).
#[test]
fn a_constexpr_sampler_address_is_a_sampler_handle() {
    let mut k = Kernel::new();
    let i8t = k.mb.int(8);
    let g = k.mb.simple_global("s", i8t, AddrSpace::Constant, None);
    k.samplers.push(g);
    let addr = k.mb.global_addr(g);
    let mut phi = addr;
    let e = k.build(|fb| {
        let ptr = fb.module().ptr(AddrSpace::Constant);
        phi = fb.push(ptr, InstKind::Freeze(addr));
        fb.ret(None);
    });
    let plan = ok(&e);
    assert!(matches!(
        plan.repr(phi),
        Some(Repr::Handle {
            class: HandleClass::Sampler,
            ..
        })
    ));
    assert!(plan.const_globals.is_empty());
}

// ---- Totality, ledger ----

/// The ledger holds exactly the plan's retained tables after planning, and nothing after a
/// refusal.
#[test]
fn the_ledger_holds_the_plan_after_success_and_nothing_after_a_refusal() {
    let (e, _) = loaded_pointer_kernel();
    let mut m = meter();
    let plan = analyze_memory(&e, &addressed(), &mut m).expect("analyze");
    assert_eq!(m.ledger(), plan.retained_bytes() as u64);
    assert!(m.ledger_peak() >= m.ledger());

    let mut m = meter();
    assert!(analyze_memory(&e, &floor(), &mut m).is_err());
    assert_eq!(m.ledger(), 0);

    // A buffer array (count 2) is refused after the ledger was charged.
    let mut arrayed = e.clone();
    for b in &mut arrayed.ep.params {
        if let ParamRole::Buffer(arg) = &mut b.role {
            arg.location.count = AbiInt::Const(2);
        }
    }
    let mut m = meter();
    assert_eq!(
        analyze_memory(&arrayed, &addressed(), &mut m).err(),
        Some(not_lowered())
    );
    assert_eq!(m.ledger(), 0);
}

/// A pointer cell whose bytes at offset 8 are read as each of `widths` (in bits): data accesses
/// only the cells' verification records.
fn cell_reads_kernel(widths: &[u8]) -> Entry {
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, buffer(0, AddrSpace::Device, None));
    k.build(|fb| {
        let i8t = fb.module().int(8);
        let i32t = fb.module().int(32);
        let i64t = fb.module().int(64);
        let arr = fb.module().array(i64t, 4);
        let p0 = fb.module().ptr(AddrSpace::Thread);
        let pd = fb.module().ptr(AddrSpace::Device);
        let one = fb.module().const_int(i32t, 1);
        let eight = fb.module().const_int(i64t, 8);
        let cell = fb.push(
            p0,
            InstKind::Alloca {
                ty: arr,
                count: one,
                align: align(8),
            },
        );
        store(fb, fb.arg(0), cell);
        let q = load(fb, pd, cell);
        let at8 = gep(fb, i8t, cell, &[eight]);
        for &w in widths {
            let t = fb.module().int(w);
            let _ = load(fb, t, at8);
        }
        let v = fb.module().const_int(i32t, 9);
        store(fb, v, q);
        fb.ret(None);
    })
}

/// The cells' verification returns its tables to the ledger when it returns: reading a cell's bytes
/// as four widths rather than one four times, which only those tables record, leaves the ledger
/// after provenance unchanged.
#[test]
fn verifying_the_cells_returns_its_tables_to_the_ledger() {
    let held = |widths: &[u8]| {
        let e = cell_reads_kernel(widths);
        let mut m = meter();
        let layouts = TypeLayouts::new(&e.layout, e.module.types(), &mut m).expect("layouts");
        let before = m.ledger();
        let prov = provenance::analyze(&e, &layouts, &mut m).expect("provenance");
        assert!(
            prov.dynamic_deref.is_none(),
            "the cell forwards its pointer"
        );
        m.ledger() - before
    };
    assert_eq!(held(&[8, 8, 8, 8]), held(&[8, 16, 32, 64]));
}

/// The index finds each parameter's binding, and what it is, as a scan of the bindings did, with
/// the bindings in any order: a buffer, a buffer table, a texture, and a texture array.
#[test]
fn the_parameter_index_finds_each_binding_as_the_scan_did() {
    use crate::memory::abi_index::AbiIndex;
    use crate::memory::provenance::{buffer_table_of, handle_array_of};
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, buffer(0, AddrSpace::Device, None));
    k.param(AddrSpace::Thread, buffer(1, AddrSpace::Device, None));
    k.param(AddrSpace::Device, texture(0));
    k.param(AddrSpace::Thread, texture(1));
    let mut e = k.build(|fb| fb.ret(None));
    e.ep.params.reverse();
    let mut m = meter();
    let index = AbiIndex::of(&e, &mut m).expect("index");
    for p in 0..4u32 {
        let scanned = e.ep.params.iter().find(|b| b.param == p).expect("binding");
        let found = index.binding(&e, p).expect("indexed binding");
        assert!(std::ptr::eq(found, scanned), "parameter {p}");
        assert_eq!(index.handle_array(p), handle_array_of(&e, scanned));
        assert_eq!(index.buffer_table(p), buffer_table_of(&e, scanned));
    }
    assert!(index.buffer_table(1).is_some() && index.handle_array(3).is_some());
    assert!(index.buffer_table(0).is_none() && index.handle_array(2).is_none());
    assert!(index.binding(&e, 4).is_none());
}

/// A tight ledger cap refuses the analysis deterministically with `LedgerBytes`.
#[test]
fn a_tight_ledger_refuses_with_ledger_bytes() {
    let (e, _) = loaded_pointer_kernel();
    let mut m = Meter::with_caps(&crate::budget::Budget::default(), u64::MAX, 64);
    assert_eq!(
        analyze_memory(&e, &addressed(), &mut m).err(),
        Some(Error::LimitExceeded(crate::error::Limit::LedgerBytes))
    );
}

/// The class of an unconstrained pointer (a null constant) never conflicts.
#[test]
fn class_join_is_a_lattice() {
    use Class::*;
    let t = Handle(HandleClass::Texture);
    assert_eq!(Unknown.join(Byte), Byte);
    assert_eq!(t.join(t), t);
    assert_eq!(t.join(Byte), Conflict);
    assert_eq!(Handle(HandleClass::Sampler).join(t), Conflict);
    assert_eq!(Conflict.join(Unknown), Conflict);
}

// ---- Fixtures ----

/// Every entry of every public fixture (top level and `engine/`) that decodes and
/// specializes is analyzed on an Addressed and a Floor profile without an internal error or a
/// panic; refusals are typed.
#[test]
fn every_public_fixture_analyzes_without_an_internal_error() {
    use crate::pipeline::testing::{engine_dir, fixtures_dir, llvm_assemble, AssembleError};
    let profiles = [addressed(), floor()];
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    for dir in [fixtures_dir(), engine_dir()] {
        let mut here: Vec<_> = std::fs::read_dir(&dir)
            .expect("fixtures")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension().is_some_and(|x| x == "ll")
                    && !p.to_string_lossy().ends_with(".air.ll")
            })
            .collect();
        here.sort();
        files.extend(here);
    }
    let (mut analyzed, mut refused) = (0usize, 0usize);
    for path in files {
        let bitcode = match llvm_assemble(&path) {
            Ok(b) => b,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            // Fixtures the installed llvm-as rejects are the assembler's, not memory's.
            Err(AssembleError::Rejected(_)) => continue,
        };
        let Ok(decoded) = crate::air::decode(&bitcode, &mut meter()) else {
            continue;
        };
        let names: Vec<String> = decoded
            .abi
            .entries
            .iter()
            .filter_map(|ep| {
                let f = decoded.module.func(ep.func)?;
                let n = decoded.module.symbols().get(f.name)?;
                Some(String::from_utf8_lossy(n).into_owned())
            })
            .collect();
        for name in names {
            let d = crate::air::decode(&bitcode, &mut meter()).expect("decode");
            let req = crate::air::SpecRequest {
                call_stack_depth: 1,
                open_cl_mode: false,
                entry: name.clone(),
                constants: Default::default(),
            };
            let Ok(e) = crate::air::specialize(d, &req, &mut meter()) else {
                continue;
            };
            for p in &profiles {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    analyze_memory(&e, p, &mut meter())
                }));
                match r {
                    Err(_) => panic!("{}:{name}: analysis panicked", path.display()),
                    Ok(Err(Error::Internal(i))) => {
                        panic!("{}:{name}: internal error {i}", path.display())
                    }
                    Ok(Err(Error::Unsupported(u))) => {
                        eprintln!(
                            "refused {}:{name} ({:?}): {:?}",
                            path.display(),
                            p.fields().buffer_access,
                            u.feature
                        );
                        refused += 1;
                    }
                    Ok(Err(other)) => panic!("{}:{name}: {other:?}", path.display()),
                    Ok(Ok(_)) => analyzed += 1,
                }
            }
        }
    }
    eprintln!("analyzed {analyzed}, refused {refused}");
    assert!(analyzed > 0);
}

/// The fixtures whose memory is known by hand: the 32 KiB threadgroup array is exactly 32768
/// bytes, the last byte of a 6-byte constant reference is a `[5, 6)` window, and a storage texture
/// parameter is a texture handle.
#[test]
fn hand_checked_fixtures() {
    use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
    let spec = |stem: &str| -> Option<Entry> {
        let bitcode = match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
            Ok(b) => b,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return None;
            }
            Err(e) => panic!("{stem}: {e:?}"),
        };
        let d = crate::air::decode(&bitcode, &mut meter()).expect("decode");
        let req = crate::air::SpecRequest {
            call_stack_depth: 1,
            open_cl_mode: false,
            entry: stem.to_owned(),
            constants: Default::default(),
        };
        Some(crate::air::specialize(d, &req, &mut meter()).expect("specialize"))
    };
    let Some(tg) = spec("threadgroup_32k") else {
        return;
    };
    for p in [addressed(), floor()] {
        let plan = plan(&tg, &p).expect("threadgroup_32k");
        assert_eq!(plan.tg_arena.static_size, 32768);
    }

    let e = spec("buffer_end_byte_load_constant").expect("fixture");
    let plan = ok(&e);
    let input = plan.slots.iter().find(|s| s.param == 0).expect("slot 0");
    assert_eq!(input.window, Some(Window::Bounded { start: 5, len: 1 }));
    assert_eq!(input.access, AccessSet::READ);
    assert_eq!(input.buffer_size, Some(6));

    let e = spec("storage_texture_write").expect("fixture");
    let plan = ok(&e);
    let texture =
        e.ep.params
            .iter()
            .find(|b| matches!(b.role, ParamRole::Texture(_)))
            .expect("texture parameter");
    let v = Value::Arg(crate::air::ir::ArgIdx::new(texture.param));
    assert!(matches!(
        plan.repr(v),
        Some(Repr::Handle {
            class: HandleClass::Texture,
            ..
        })
    ));
}

/// A texture-array parameter (`array<texture2d<float>, 2>`, location count 2) is a pointer to its
/// handles: a load through it is a texture handle whose candidate is the parameter and whose
/// offset names the element, not a dynamic handle.
#[test]
fn a_texture_array_parameter_is_a_pointer_to_its_handles() {
    let mut k = Kernel::new();
    k.param(
        AddrSpace::Thread,
        ParamRole::Texture(TextureArg {
            location: Slot {
                index: AbiInt::Const(0),
                count: AbiInt::Const(2),
            },
            access: Access::Sample,
            raster_order_group: None,
        }),
    );
    let mut t = Value::Arg(crate::air::ir::ArgIdx::new(0));
    let e = k.build(|fb| {
        let i32t = fb.module().int(32);
        let i64t = fb.module().int(64);
        let tex = fb.module().ptr(AddrSpace::Device);
        let st = fb.module().struct_ty(&[tex]);
        let (one, zero) = (
            fb.module().const_int(i64t, 1),
            fb.module().const_int(i32t, 0),
        );
        let at = gep(fb, st, fb.arg(0), &[one, zero]);
        t = fb.push(
            tex,
            InstKind::Load {
                ptr: at,
                align: align(8),
                volatile: false,
                atomic: None,
                range: None,
            },
        );
        let _ = fb.push(tex, InstKind::Freeze(t));
        fb.ret(None);
    });
    let plan = ok(&e);
    assert_eq!(
        plan.repr(Value::Arg(crate::air::ir::ArgIdx::new(0))),
        Some(Repr::HandleArray {
            class: HandleClass::Texture
        })
    );
    assert!(matches!(
        plan.repr(t),
        Some(Repr::Handle {
            class: HandleClass::Texture,
            ..
        })
    ));
    assert_eq!(repr_roots(&plan, t), vec![Root::Param(0)]);
    assert_eq!(plan.offset(t).and_then(|o| o.exact()), Some(8));
    assert_eq!(plan.needs.handle_table, None);
    assert!(plan.slots.is_empty());
}

/// A load through a handle-array parameter laid out before the GEP it reads (canon keeps layout
/// order, where a use may precede its definition): the worklist revisits the load once the GEP
/// is known, so it is the parameter's handle, never a dynamic one.
#[test]
fn a_handle_array_load_laid_out_before_its_gep_is_not_dynamic() {
    let mut k = Kernel::new();
    k.param(
        AddrSpace::Thread,
        ParamRole::Texture(TextureArg {
            location: Slot {
                index: AbiInt::Const(0),
                count: AbiInt::Const(2),
            },
            access: Access::Sample,
            raster_order_group: None,
        }),
    );
    let e = k.build(|fb| {
        let i32t = fb.module().int(32);
        let i64t = fb.module().int(64);
        let tex = fb.module().ptr(AddrSpace::Device);
        let st = fb.module().struct_ty(&[tex]);
        let (one, zero) = (
            fb.module().const_int(i64t, 1),
            fb.module().const_int(i32t, 0),
        );
        // Layout: entry, use, def; control: entry -> def -> use.
        let (use_block, def_block) = (fb.block(), fb.block());
        fb.br(def_block);
        fb.switch_to(def_block);
        let at = gep(fb, st, fb.arg(0), &[one, zero]);
        fb.br(use_block);
        fb.switch_to(use_block);
        let t = fb.push(
            tex,
            InstKind::Load {
                ptr: at,
                align: align(8),
                volatile: false,
                atomic: None,
                range: None,
            },
        );
        let _ = fb.push(tex, InstKind::Freeze(t));
        fb.ret(None);
    });
    // `finish` renumbers in layout order: find the load (laid out before the GEP) again.
    let body = e
        .module
        .func(e.func)
        .and_then(|f| f.body.as_ref())
        .expect("body");
    let first = |pick: fn(&InstKind) -> bool| {
        body.insts
            .iter()
            .position(|(_, i)| pick(&i.kind))
            .expect("instruction")
    };
    let load_at = first(|k| matches!(k, InstKind::Load { .. }));
    assert!(load_at < first(|k| matches!(k, InstKind::Gep { .. })));
    let t = Value::Inst(body.insts.iter().nth(load_at).expect("load").0);
    let plan = ok(&e);
    assert!(matches!(
        plan.repr(t),
        Some(Repr::Handle {
            class: HandleClass::Texture,
            ..
        })
    ));
    assert_eq!(repr_roots(&plan, t), vec![Root::Param(0)]);
    assert_eq!(plan.offset(t).and_then(|o| o.exact()), Some(8));
    assert_eq!(plan.needs.handle_table, None);
}

/// A chain of `blocks` blocks, each a pointer phi and `triples` offset/load/store triples on
/// buffer 0: `1 + blocks·(1 + triples)` pointer values (the spike's shape).
fn chain_kernel(blocks: u32, triples: u64) -> Entry {
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, buffer(0, AddrSpace::Device, None));
    k.build(|fb| {
        let i32t = fb.module().int(32);
        let i64t = fb.module().int(64);
        let ptr = fb.module().ptr(AddrSpace::Device);
        let offsets: Vec<Value> = (0..triples)
            .map(|i| fb.module().const_int(i64t, i))
            .collect();
        let ids: Vec<BlockId> = (0..blocks).map(|_| fb.block()).collect();
        let mut prev = (fb.arg(0), BlockId::new(0));
        fb.br(ids[0]);
        for (n, &b) in ids.iter().enumerate() {
            fb.switch_to(b);
            let p = fb.push(
                ptr,
                InstKind::Phi {
                    incoming: Box::new([prev]),
                    fmf: FastMath::NONE,
                },
            );
            let mut last = p;
            for &o in &offsets {
                let q = gep(fb, i32t, p, &[o]);
                let v = load(fb, i32t, q);
                store(fb, v, q);
                last = q;
            }
            match ids.get(n + 1) {
                Some(&next) => fb.br(next),
                None => fb.ret(None),
            }
            prev = (last, b);
        }
    })
}

/// Every input-scaled table the analysis builds is charged: on a mid-size entry the ledger peak
/// covers at least the sum of those tables' sizes, so dropping a charge fails here before the
/// allocator check would notice it.
#[test]
fn the_ledger_peak_covers_every_input_scaled_table() {
    use crate::air::ir::ValueId;
    use crate::memory::provenance::PtrFact;
    use std::mem::size_of;
    const BLOCKS: u32 = 20;
    const TRIPLES: u64 = 50;
    let e = chain_kernel(BLOCKS, TRIPLES);
    let func = e.module.func(e.func).expect("func");
    let params = func.params.len();
    let insts = func.body.as_ref().map_or(0, |b| b.insts.len());
    let consts = e.module.consts().len();
    let local = params + insts;
    // The pointer values: the buffer, and per block its phi and one GEP per triple.
    let n = 1 + BLOCKS as usize * (1 + TRIPLES as usize);
    // One flow edge per phi incoming and per GEP base.
    let edges = n - 1;
    // The tables the analysis holds to its end.
    let held: [(&str, usize); 13] = [
        ("param types", params * size_of::<TypeId>()),
        ("pointer index", local * size_of::<u32>()),
        ("pointer values", n * size_of::<ValueId>()),
        ("facts", n * size_of::<PtrFact>()),
        ("change counts", n),
        ("out degrees", (n + 1) * size_of::<u32>()),
        ("edge starts", (n + 1) * size_of::<u32>()),
        ("edge users", edges * size_of::<u32>()),
        ("edge fill", (n + 1) * size_of::<u32>()),
        ("queued", n * size_of::<bool>()),
        ("worklist", n * size_of::<u32>()),
        ("resolved classes", n * size_of::<Class>()),
        ("fact table", n * size_of::<(ValueId, PtrFact)>()),
    ];
    // The handle classes' tables, held until provenance resolves them, and the plan's tables after.
    let classes: [(&str, usize); 4] = [
        ("class parents", n * size_of::<u32>()),
        ("class sizes", n * size_of::<u32>()),
        ("class labels", n * size_of::<u32>()),
        ("classes", n * size_of::<Class>()),
    ];
    let planned: [(&str, usize); 2] = [
        (
            "representations",
            (local + consts) * size_of::<Option<Repr>>(),
        ),
        ("offsets", n * size_of::<(ValueId, OffsetRange)>()),
    ];
    let bytes = |t: &[(&str, usize)]| t.iter().map(|(_, b)| b).sum::<usize>();
    let sum = bytes(&held) + bytes(&classes).max(bytes(&planned));
    let mut m = meter();
    let plan = analyze_memory(&e, &addressed(), &mut m).expect("analyze");
    assert_eq!(plan.slots.len(), 1);
    assert!(
        m.ledger_peak() >= sum as u64,
        "ledger peak {} below the {sum} bytes of the input-scaled tables {held:?} with \
         {classes:?} or {planned:?}",
        m.ledger_peak()
    );
    assert_eq!(m.ledger(), plan.retained_bytes() as u64);
}

/// The budget spike, part 2 (G-rel): memory analysis of a synthetic entry the size of the corpus's
/// largest inlined row (about 1.35M instructions: 1,000 blocks, each a pointer phi and 450
/// offset/load/store triples). The corpus's own largest row stops in canon before analysis (it
/// needs function constants, and with zero constants it is a linked function), so this is the
/// analysis cost at that size. Reports wall time, work units, and the ledger peak.
#[test]
#[ignore = "G-rel: run in release with --ignored (an analysis cost spike)"]
fn spike_analysis_at_the_largest_row_size() {
    const BLOCKS: u32 = 1000;
    const TRIPLES: u64 = 450;
    let e = chain_kernel(BLOCKS, TRIPLES);
    let insts = e
        .module
        .func(e.func)
        .and_then(|f| f.body.as_ref())
        .map_or(0, |b| b.insts.len());
    let mut m = meter();
    let start = std::time::Instant::now();
    let plan = analyze_memory(&e, &addressed(), &mut m).expect("analyze");
    let wall = start.elapsed();
    eprintln!(
        "spike: {insts} instructions, analyze {} ms, work {} units, ledger peak {} bytes, retained {} bytes",
        wall.as_millis(),
        m.work(),
        m.ledger_peak(),
        plan.retained_bytes()
    );
    assert!(insts > 1_350_000);
    assert_eq!(plan.slots.len(), 1);
    // The chain is acyclic, so every offset stays exact: block n's phi is at n·449·4 bytes and
    // its last access ends 449·4 + 4 bytes later.
    let len = u64::from(BLOCKS) * (TRIPLES - 1) * 4 + 4;
    assert_eq!(
        plan.slots.first().and_then(|s| s.window),
        Some(Window::Bounded { start: 0, len })
    );
}

mod gaps;

/// A kernel whose argument buffer holds one sampled texture member at byte 8, loaded `n + 1`
/// times; the loads are merged by a chain of `n` selects, `h{i+1} = select c, h{i}, l{i}`. Every
/// load and select is a heap texture handle, and all of them form one group: `n + 1` loads of one
/// member, and `2n` merged values.
fn heap_chain_kernel(n: usize) -> (Entry, Value) {
    let mut k = Kernel::new();
    k.structs.push(StructInfo {
        members: vec![Member {
            offset: 8,
            size: 8,
            array_len: 0,
            layout: None,
            kind: MemberKind::Indirect(IndirectArg {
                ordinal: 0,
                role: texture(0),
                ty: ArgTypeInfo::default(),
                name: None,
            }),
            render_target: None,
            raster_order_group: None,
            type_name: DiagName::new("texture2d<float>"),
            name: DiagName::new("t"),
            key: crate::air::abi::UserKey(b"t".as_slice().into()),
        }],
    });
    k.param(
        AddrSpace::Constant,
        ParamRole::ArgumentBuffer(BufferArg {
            location: slot(0),
            access: Access::Read,
            space: AddrSpace::Constant,
            buffer_size: Some(16),
            layout: Some(StructRef(0)),
            raster_order_group: None,
        }),
    );
    let mut end = Value::Arg(crate::air::ir::ArgIdx::new(0));
    let e = k.build(|fb| {
        let i1 = fb.module().int(1);
        let i8t = fb.module().int(8);
        let i64t = fb.module().int(64);
        let ptr = fb.module().ptr(AddrSpace::Device);
        let c = fb.module().const_int(i1, 1);
        let eight = fb.module().const_int(i64t, 8);
        let at = gep(fb, i8t, fb.arg(0), &[eight]);
        let handle = |fb: &mut FunctionBuilder<'_>| {
            fb.push(
                ptr,
                InstKind::Load {
                    ptr: at,
                    align: align(8),
                    volatile: false,
                    atomic: None,
                    range: None,
                },
            )
        };
        end = handle(fb);
        for _ in 0..n {
            let l = handle(fb);
            end = fb.push(
                ptr,
                InstKind::Select {
                    c,
                    t: end,
                    f: l,
                    fmf: FastMath::NONE,
                },
            );
        }
        fb.ret(None);
    });
    (e, end)
}

/// `HeapGroups::analyze` over [`heap_chain_kernel`] of `n`: the groups, the work units, and the
/// ledger bytes it leaves.
fn heap_groups_of(n: usize) -> (crate::memory::handles::HeapGroups, Value, u64, u64) {
    let (e, end) = heap_chain_kernel(n);
    let plan = ok(&e);
    let body = e
        .module
        .func(e.func)
        .and_then(|f| f.body.as_ref())
        .expect("body");
    let mut m = meter();
    let groups =
        crate::memory::handles::HeapGroups::analyze(&e, body, &plan, &mut m).expect("groups");
    (groups, end, m.work(), m.ledger())
}

/// Grouping a function's heap handles is metered by what it reads, one work unit per instruction of
/// the body, per heap handle, and per value a handle merges (besides the union-find's find steps),
/// and keeps only the groups on the ledger. At 256 selects that is at least 7 units a select (514
/// instructions, 513 handles, 768 merged values and loads joined), and twice the chain takes under
/// 2.2 times the units. Previously `analyze` took no meter: it charged no work and no bytes.
#[test]
fn heap_handle_groups_are_metered_by_the_handles_they_join() {
    let (groups, end, work, ledger) = heap_groups_of(256);
    assert_eq!(groups.len(), 1);
    let Value::Inst(end) = end else {
        panic!("the chain's end is an instruction");
    };
    assert_eq!(groups.group(end), Some(0));
    assert_eq!(
        groups.access(0),
        Some(crate::pipeline::sym::ImageAccess::Sampled)
    );
    assert!(work >= 7 * 256, "{work} units for 256 selects");
    // The groups the plan keeps: one entry per handle, and the group's access.
    let entry = crate::memory::provenance::map_entry_bytes::<crate::air::ir::InstId, u32>() as u64;
    assert!(
        ledger >= 513 * entry,
        "{ledger} ledger bytes for 513 handles"
    );
    assert!(
        ledger < 2 * 513 * entry,
        "{ledger} ledger bytes: the working tables stay charged"
    );
    let (_, _, twice, _) = heap_groups_of(512);
    assert!(
        twice * 10 < work * 22,
        "{twice} units for 512 selects, {work} for 256"
    );
}

/// An arena's entries are placed in object order, so `ArenaLayout::entry` is a search: it finds
/// each of 4096 placed allocas, and placing an object at or before the last one is an invariant
/// violation. Previously `entry` scanned the entries and `place` took any order.
#[test]
fn arena_entries_are_placed_in_order_and_searched() {
    let mut m = meter();
    let mut arena = ArenaLayout::default();
    for i in 0..4096u32 {
        let at = arena
            .place(ArenaObject::Alloca(InstId::new(2 * i)), 4, 4, &mut m)
            .expect("place");
        assert_eq!(at, 4 * u64::from(i));
    }
    for i in 0..4096u32 {
        let e = arena
            .entry(ArenaObject::Alloca(InstId::new(2 * i)))
            .expect("entry");
        assert_eq!(e.offset, 4 * u64::from(i));
        assert!(arena
            .entry(ArenaObject::Alloca(InstId::new(2 * i + 1)))
            .is_none());
    }
    let late = arena.place(ArenaObject::Alloca(InstId::new(10)), 4, 4, &mut m);
    assert!(
        matches!(late, Err(Error::Internal(_))),
        "an out-of-order placement: {late:?}"
    );
    let again = arena.place(ArenaObject::Alloca(InstId::new(8190)), 4, 4, &mut m);
    assert!(
        matches!(again, Err(Error::Internal(_))),
        "a repeated object: {again:?}"
    );
}

/// An arena placed in any order (the threadgroup arena's globals, in Metal's order) gives the first
/// object placed offset 0, and in object order every entry is found at the offset it was placed at;
/// an object placed twice is an invariant violation.
#[test]
fn an_arena_placed_in_any_order_is_searched_in_object_order() {
    let mut m = meter();
    // 0, 37, 10, 47, ... : every object once, in no monotone order.
    let objects: Vec<(ArenaObject, u64, u64)> = (0..64u32)
        .map(|k| (ArenaObject::Global(GlobalId::new((k * 37) % 64)), 8, 8))
        .collect();
    let arena = ArenaLayout::in_placement_order(&objects, &mut m).expect("placed");
    for (k, (object, _, _)) in objects.iter().enumerate() {
        let e = arena.entry(*object).expect("entry");
        assert_eq!(e.offset, 8 * k as u64);
    }
    assert_eq!(arena.static_size, 512);
    let mut twice = objects.clone();
    twice.push(objects[5]);
    let again = ArenaLayout::in_placement_order(&twice, &mut m);
    assert!(
        matches!(again, Err(Error::Internal(_))),
        "a repeated object: {again:?}"
    );
}
