//! Per-construct requirement derivation (A4) and entry interfaces, each on a module that also
//! passes the `spirv-val` sweep.

use super::families::tys;
use super::families_image::{image_ty, ivec, load_global, sampler, texture, SAMPLED_2D};
use super::{assert_spirv_valid, compute, cval, elem_ptr, finish_compute, gid_x, slot, ssbo};
use crate::device::requirements::{Requirements, SpirvCapability, VkExtension, VkFeature};
use crate::pipeline::sym::ResourceKind;
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::ir::{
    natural_layout, AtomicOp, ConvOp, ExecModel, ImageOperands, IoDir, LayoutRule, MemAccess,
    MemoryOrder, Scope, Semantics, StorageClass, Type, TypedModule,
};
use crate::spirv::requirements::{uses_physical_storage_buffer, uses_vulkan_memory_model};
use crate::spirv::testing::{spirv_val_with, TargetEnv, ValOptions};
use crate::spv::{BuiltIn, Capability};

fn has(r: &Requirements, caps: &[Capability], features: &[VkFeature]) {
    for c in caps {
        assert!(
            r.spirv_capabilities.contains(&SpirvCapability::of(*c)),
            "{c:?} missing: {r:?}"
        );
    }
    for f in features {
        assert!(r.features.contains(f), "{f:?} missing: {r:?}");
    }
}

fn checked(m: TypedModule) -> Requirements {
    let r = m.requirements().clone();
    assert_spirv_valid(m);
    r
}

#[test]
fn int64_atomics_pick_the_storage_class_feature() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let buf = ssbo(&mut b, 0, t.u64_, 8);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let p = elem_ptr(f, buf, i);
        let one = cval(f, t.u64_, 1);
        f.atomic(AtomicOp::IAdd, p, Scope::Device, Semantics::RELAXED, one)
            .expect("atomic");
    });
    let r = checked(finish_compute(b, main));
    has(
        &r,
        &[Capability::Int64Atomics],
        &[VkFeature::ShaderBufferInt64Atomics],
    );
    assert!(!r.features.contains(&VkFeature::ShaderSharedInt64Atomics));
}

#[test]
fn availability_semantics_select_the_vulkan_memory_model() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let buf = ssbo(&mut b, 0, t.u32_, 4);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let p = elem_ptr(f, buf, i);
        let one = cval(f, t.u32_, 1);
        let release = Semantics {
            order: MemoryOrder::Release,
            uniform_memory: true,
            make_available: true,
            ..Semantics::RELAXED
        };
        f.atomic_store(p, Scope::Device, release, one)
            .expect("release store");
    });
    let r = checked(finish_compute(b, main));
    assert!(uses_vulkan_memory_model(&r));
    has(
        &r,
        &[
            Capability::VulkanMemoryModel,
            Capability::VulkanMemoryModelDeviceScope,
        ],
        &[
            VkFeature::VulkanMemoryModel,
            VkFeature::VulkanMemoryModelDeviceScope,
        ],
    );
    assert!(!uses_physical_storage_buffer(&r));
}

#[test]
fn block_layouts_select_their_relaxation_features() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let rta = b
        .ty(Type::RuntimeArray {
            elem: t.u32_,
            stride: Some(4),
        })
        .expect("rta");
    let members = [t.vec3, t.f32_, rta];
    let layout = natural_layout(b.types(), LayoutRule::Scalar, true, &members).expect("scalar");
    let scalar = b
        .ty(Type::Struct {
            members: members.into(),
            layout: Some(layout),
        })
        .expect("scalar block");
    let g = b
        .resource(
            slot(ResourceKind::Buffer, 0),
            StorageClass::StorageBuffer,
            scalar,
        )
        .expect("scalar ssbo");
    let main = compute(&mut b, |f| {
        let zero = cval(f, t.u32_, 0);
        let base = f.module().global_value(g).expect("base");
        let p = f.access_chain(base, &[zero]).expect("member");
        let v = f.load(p, MemAccess::default()).expect("load");
        f.store(p, v, MemAccess::default()).expect("store");
    });
    let r = checked(finish_compute(b, main));
    has(&r, &[], &[VkFeature::ScalarBlockLayout]);
    assert!(!r.features.contains(&VkFeature::UniformBufferStandardLayout));
}

/// A compute module sampling (`gather: false`, at an explicit level) or gathering a 2D texture
/// with a dynamic `Offset`.
fn dynamic_offset_module(gather: bool) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let ty = image_ty(&mut b, t.f32_, SAMPLED_2D);
    let tex = texture(&mut b, 0, ty);
    let smp = sampler(&mut b, 0);
    let main = compute(&mut b, |f| {
        let img = load_global(f, tex);
        let s = load_global(f, smp);
        let si = f.sampled_image(img, s).expect("sampled image");
        let half = cval(f, t.f32_, 0x3f00_0000);
        let uv = f.composite_construct(t.vec2, &[half, half]).expect("uv");
        let offset = ivec(f, 2, 1);
        let ops = ImageOperands {
            offset: Some(offset),
            ..ImageOperands::default()
        };
        if gather {
            f.gather(si, uv, 0, &ops).expect("gather");
        } else {
            let lod = ImageOperands {
                lod: Some(half),
                ..ops
            };
            f.sample(si, uv, None, false, &lod).expect("sample");
        }
    });
    finish_compute(b, main)
}

/// `Offset` on a sample needs `maintenance8` beside `ImageGatherExtended`
/// (VUID-RuntimeSpirv-Offset-10213), and `spirv-val` accepts it only with its relaxation; on a
/// gather it needs `ImageGatherExtended` alone.
#[test]
fn a_dynamic_sample_offset_needs_maintenance8() {
    let words = {
        let m = dynamic_offset_module(false);
        let r = m.requirements().clone();
        has(
            &r,
            &[Capability::ImageGatherExtended],
            &[
                VkFeature::ShaderImageGatherExtended,
                VkFeature::Maintenance8,
            ],
        );
        // Extension-only: the derivation names its extension.
        assert!(r.extensions.contains(&VkExtension::KhrMaintenance8));
        assert!(ValOptions::for_requirements(&r).allow_offset_texture_operand);
        assert_spirv_valid(m)
    };
    assert!(spirv_val_with(&words, TargetEnv::Vulkan1_2, ValOptions::default()).is_err());
    let r = checked(dynamic_offset_module(true));
    has(
        &r,
        &[Capability::ImageGatherExtended],
        &[VkFeature::ShaderImageGatherExtended],
    );
    assert!(!r.features.contains(&VkFeature::Maintenance8));
    assert!(!r.extensions.contains(&VkExtension::KhrMaintenance8));
}

#[test]
fn a_fragment_viewport_index_input_needs_multi_viewport() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let vi = b
        .builtin(BuiltIn::ViewportIndex, IoDir::Input, t.u32_)
        .expect("viewport index");
    let mut f = b.function(t.void).expect("main");
    let p = f.module().global_value(vi).expect("ptr");
    f.load(p, MemAccess::default()).expect("load");
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish");
    b.entry(ExecModel::Fragment, main, vec![]).expect("entry");
    let r = checked(b.finalize().expect("finalize"));
    has(
        &r,
        &[Capability::MultiViewport],
        &[VkFeature::MultiViewport],
    );
    assert!(!r.features.contains(&VkFeature::ShaderOutputViewportIndex));
}

#[test]
fn the_entry_interface_is_the_referenced_globals_in_id_order() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let unused = ssbo(&mut b, 0, t.u32_, 4);
    let wg = b
        .plain_global(StorageClass::Workgroup, t.u32_, None)
        .expect("workgroup");
    let used = ssbo(&mut b, 1, t.u32_, 4);
    // Referenced only from a helper the entry calls.
    let helper = {
        let mut f = b.function(t.void).expect("helper");
        let p = f.module().global_value(wg).expect("wg");
        let zero = cval(&mut f, t.u32_, 0);
        f.store(p, zero, MemAccess::default()).expect("store");
        f.ret(None).expect("ret");
        f.finish().expect("finish")
    };
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let p = elem_ptr(f, used, i);
        f.store(p, i, MemAccess::default()).expect("store");
        f.call(helper, &[]).expect("call");
    });
    let m = finish_compute(b, main);
    let interface = m.interface().to_vec();
    assert!(!interface.contains(&unused));
    let wg_at = interface.iter().position(|g| *g == wg).expect("workgroup");
    let used_at = interface.iter().position(|g| *g == used).expect("buffer");
    assert!(wg_at < used_at, "{interface:?}");
    assert!(interface.windows(2).all(|w| w[0] < w[1]), "{interface:?}");
    assert_spirv_valid(m);
}
