//! Barrier and fence scopes and semantics per memory model: the operand the family emits for every
//! Metal `mem_flags` combination, both memory models, `spirv-val` on the emitted barriers, and the
//! barriers ordering workgroup memory on the CPU interpreter.

use super::*;
use crate::device::DeviceProfile;
use crate::pipeline::sym::{ResourceKind, ResourceSlot};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::{
    natural_layout, BinOp, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass, Type, TypedModule,
};
use crate::spirv::testing::{assert_valid, committed_profiles};

fn flags(bits: u64) -> MemFlags {
    MemFlags::from_raw(bits).expect("four flags")
}

/// `GLSL450` names the memories only; the Vulkan model adds `MakeAvailable` on release and
/// `MakeVisible` on acquire; no memory flag, or a relaxed order, is the `None` operand.
#[test]
fn semantics_follow_the_flags_and_the_memory_model() {
    for bits in 0..16u64 {
        let f = flags(bits);
        for model in [BarrierModel::Glsl450, BarrierModel::Vulkan] {
            let s = semantics(f, MemoryOrder::AcquireRelease, model);
            assert_eq!(s.uniform_memory, f.device, "{bits:#x}");
            assert_eq!(
                s.workgroup_memory,
                f.threadgroup || f.threadgroup_imageblock,
                "{bits:#x}"
            );
            assert_eq!(s.image_memory, f.texture, "{bits:#x}");
            if bits == 0 {
                assert_eq!(s, Semantics::RELAXED);
                continue;
            }
            assert_eq!(s.order, MemoryOrder::AcquireRelease);
            let vulkan = model == BarrierModel::Vulkan;
            assert_eq!((s.make_available, s.make_visible), (vulkan, vulkan));
            assert_eq!(
                semantics(f, MemoryOrder::Relaxed, model),
                Semantics::RELAXED
            );
            let rel = semantics(f, MemoryOrder::Release, model);
            assert_eq!((rel.make_available, rel.make_visible), (vulkan, false));
            let acq = semantics(f, MemoryOrder::Acquire, model);
            assert_eq!((acq.make_available, acq.make_visible), (false, vulkan));
        }
    }
}

/// Metal orders map onto Vulkan's; `seq_cst` is acquire-release (Vulkan has no stronger fence).
#[test]
fn fence_orders_map_onto_vulkan() {
    assert_eq!(fence_order(AirOrder::Relaxed), MemoryOrder::Relaxed);
    assert_eq!(fence_order(AirOrder::Acquire), MemoryOrder::Acquire);
    assert_eq!(fence_order(AirOrder::Release), MemoryOrder::Release);
    assert_eq!(fence_order(AirOrder::AcqRel), MemoryOrder::AcquireRelease);
    assert_eq!(fence_order(AirOrder::SeqCst), MemoryOrder::AcquireRelease);
}

/// A simdgroup barrier makes the class cross-lane, and joins the classes it meets.
#[test]
fn a_simdgroup_barrier_is_cross_lane() {
    let basic = SimdClass::CrossLane(OpBits::BASIC);
    assert_eq!(with_simdgroup_barrier(SimdClass::None), basic);
    assert_eq!(with_simdgroup_barrier(SimdClass::LaneArithmetic), basic);
    assert_eq!(
        with_simdgroup_barrier(SimdClass::Quad),
        SimdClass::CrossLane(OpBits::BASIC.union(OpBits::QUAD))
    );
    assert_eq!(
        with_simdgroup_barrier(SimdClass::CrossLane(OpBits::BALLOT)),
        SimdClass::CrossLane(OpBits::BALLOT.union(OpBits::BASIC))
    );
    assert_eq!(with_simdgroup_barrier(SimdClass::Matrix), SimdClass::Matrix);
}

/// `wg[li] = li; barrier; out[li] = wg[(li + 1) % n]` (or the subgroup neighbour for a subgroup
/// barrier), with the barrier as the family emits it for `mem_threadgroup | mem_device` under
/// `model`, and optionally a fence.
fn exchange(n: u32, scope: Scope, model: BarrierModel, fence: Option<Scope>) -> TypedModule {
    let mut mb = ModuleBuilder::new();
    let u32_ = mb
        .ty(Type::Int {
            width: 32,
            signed: false,
        })
        .expect("u32");
    let rta = mb
        .ty(Type::RuntimeArray {
            elem: u32_,
            stride: Some(4),
        })
        .expect("rta");
    let layout = natural_layout(mb.types(), LayoutRule::Std430, true, &[rta]).expect("layout");
    let block = mb
        .ty(Type::Struct {
            members: [rta].into(),
            layout: Some(layout),
        })
        .expect("block");
    let out = mb
        .resource(
            ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            },
            StorageClass::StorageBuffer,
            block,
        )
        .expect("resource");
    let li_var = mb
        .builtin(
            crate::spv::BuiltIn::LocalInvocationIndex,
            IoDir::Input,
            u32_,
        )
        .expect("li");
    let void = mb.ty(Type::Void).expect("void");
    let mut f = mb.function(void).expect("function");
    let k = |f: &mut FunctionBuilder<'_>, n: u32| {
        let c = f.u32_const(n).expect("const");
        f.module().const_value(c).expect("value")
    };
    let len = f.u32_const(n).expect("len");
    let arr = f
        .module()
        .ty(Type::Array {
            elem: u32_,
            len,
            stride: None,
        })
        .expect("array");
    let g = f
        .module()
        .plain_global(StorageClass::Workgroup, arr, None)
        .expect("wg");
    let base = f.module().global_value(g).expect("base");
    let p = f.module().global_value(li_var).expect("li");
    let li = f.load(p, MemAccess::default()).expect("load");
    let at = f.access_chain(base, &[li]).expect("chain");
    f.store(at, li, MemAccess::default()).expect("store");
    let sem = semantics(flags(3), MemoryOrder::AcquireRelease, model);
    f.control_barrier(scope, scope, sem).expect("barrier");
    if let Some(s) = fence {
        let sem = semantics(flags(1), MemoryOrder::AcquireRelease, model);
        f.memory_barrier(s, sem).expect("fence");
    }
    // The neighbour within the barrier's scope: the workgroup, or the 32-lane subgroup.
    let one = k(&mut f, 1);
    let j = f.binary(BinOp::IAdd, li, one).expect("add");
    let j = match scope {
        Scope::Subgroup => {
            let m = k(&mut f, 31);
            let lane = f.binary(BinOp::BitwiseAnd, j, m).expect("and");
            let hi = k(&mut f, !31);
            let group = f.binary(BinOp::BitwiseAnd, li, hi).expect("and");
            f.binary(BinOp::BitwiseOr, group, lane).expect("or")
        }
        _ => {
            let nn = k(&mut f, n);
            f.binary(BinOp::UMod, j, nn).expect("mod")
        }
    };
    let q = f.access_chain(base, &[j]).expect("chain");
    let v = f.load(q, MemAccess::default()).expect("load");
    let obase = f.module().global_value(out).expect("out");
    let zero = k(&mut f, 0);
    let o = f.access_chain(obase, &[zero, li]).expect("chain");
    f.store(o, v, MemAccess::default()).expect("store");
    f.ret(None).expect("ret");
    let id = f.finish().expect("finish");
    mb.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: n, y: 1, z: 1 }],
    )
    .expect("entry");
    mb.finalize().expect("finalize")
}

fn vulkan_model_profile() -> (String, DeviceProfile) {
    committed_profiles()
        .expect("profiles")
        .into_iter()
        .find(|(_, p)| p.fields().vulkan_memory_model && p.fields().subgroup.kernel.supported)
        .expect("a committed profile with the Vulkan memory model")
}

/// The emitted barriers order workgroup memory on the interpreter (both scheduling orders, both
/// subgroup widths for the workgroup barrier), and pass L3 under each memory model on a
/// profile that has it.
#[test]
fn emitted_barriers_order_memory_and_validate() {
    let (stem, vk) = vulkan_model_profile();
    let glsl = crate::pipeline::testing::profile(
        "moltenvk-m3",
        crate::device::DevicePolicy::RESIDENT_TABLES,
    )
    .expect("m3");
    for model in [BarrierModel::Glsl450, BarrierModel::Vulkan] {
        for (scope, fence) in [
            (Scope::Workgroup, None),
            (Scope::Workgroup, Some(Scope::Device)),
            (Scope::Subgroup, Some(Scope::Subgroup)),
        ] {
            let m = exchange(64, scope, model, fence);
            let sgs: &[SubgroupSize] = match scope {
                Scope::Subgroup => &[SubgroupSize::W32],
                _ => &[SubgroupSize::W32, SubgroupSize::W64],
            };
            for sg in sgs {
                let mut outs = Vec::new();
                for order in [Order::Forward, Order::Reverse] {
                    let res = Resources {
                        buffers: vec![Buffer {
                            slot: Some(ResourceSlot {
                                stage: Stage::Kernel,
                                kind: ResourceKind::Buffer,
                                index: 0,
                            }),
                            address: None,
                            bytes: vec![0; 64 * 4],
                        }],
                        ..Resources::default()
                    };
                    let cfg = Config {
                        subgroup: *sg,
                        order,
                        ..Config::default()
                    };
                    let out = run(&m, &cfg, res).expect("run");
                    outs.push(out.buffers[0].clone());
                }
                assert_eq!(outs[0], outs[1]);
                let w: Vec<u32> = outs[0]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let want: Vec<u32> = (0..64u32)
                    .map(|i| match scope {
                        Scope::Subgroup => (i & !31) | ((i + 1) & 31),
                        _ => (i + 1) % 64,
                    })
                    .collect();
                assert_eq!(w, want, "{model:?} {scope:?}");
            }
            match model {
                BarrierModel::Glsl450 => {
                    assert_valid(m, &glsl);
                }
                BarrierModel::Vulkan => {
                    let words = assert_valid(m, &vk);
                    // The Vulkan-model operands select the Vulkan memory model (capability 5345).
                    assert!(
                        words.windows(2).any(|w| w == [(2 << 16) | 17, 5345]),
                        "{stem}: no VulkanMemoryModel capability"
                    );
                }
            };
        }
    }
}
