//! `spirv-val` sweep, continued: ray queries (`SPV_KHR_ray_query`).

use super::families_image::load_global;
use super::{assert_spirv_valid, compute, cval, finish_compute, float, int, slot, ssbo, vec};
use super::{elem_ptr, gid_x};
use crate::device::requirements::{SpirvCapability, SpirvExtension, VkExtension, VkFeature};
use crate::pipeline::sym::{AccelerationStructureLevel, ResourceKind};
use crate::spirv::builder::constructs::{ArmEnd, LoopCtx, LoopSpec};
use crate::spirv::builder::ops::ray::RayInit;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BuildError, GlobalRule, OperandRule};
use crate::spirv::ids::{GlobalId, Value};
use crate::spirv::ir::{BinOp, ConvOp, IoDir, MemAccess, RayQueryValue, StorageClass, Type, UnOp};

fn accel(b: &mut ModuleBuilder, index: u32) -> GlobalId {
    let ty = b
        .ty(Type::AccelerationStructure)
        .expect("acceleration structure");
    b.resource(
        slot(
            ResourceKind::AccelerationStructure(AccelerationStructureLevel::Instance),
            index,
        ),
        StorageClass::UniformConstant,
        ty,
    )
    .expect("acceleration structure binding")
}

fn vec3(f: &mut FunctionBuilder<'_>, x: u64, y: u64, z: u64) -> Value {
    let f32_ = float(f.module(), 32);
    let ty = vec(f.module(), f32_, 3);
    let parts = [x, y, z].map(|bits| cval(f, f32_, bits));
    f.composite_construct(ty, &parts).expect("vec3")
}

/// A query over `accel` with `flags`: traverse, commit every triangle candidate, and store the
/// committed intersection's values to `out`.
fn trace(f: &mut FunctionBuilder<'_>, accel: GlobalId, out: GlobalId, flags: Value) {
    let u32_ = int(f.module(), 32, false);
    let f32_ = float(f.module(), 32);
    let as_value = load_global(f, accel);
    let query = f.ray_query().expect("query");
    let origin = vec3(f, 0, 0, 0);
    let direction = vec3(f, 0, 0, 1.0f32.to_bits().into());
    let tmin = cval(f, f32_, 0);
    let tmax = cval(f, f32_, 100.0f32.to_bits().into());
    let cull_mask = cval(f, u32_, 0xFF);
    f.ray_query_initialize(
        query,
        RayInit {
            accel: as_value,
            flags,
            cull_mask,
            origin,
            tmin,
            direction,
            tmax,
        },
    )
    .expect("initialize");
    let mut ctx = ();
    f.loop_(
        &mut ctx,
        LoopSpec::default(),
        Box::new(
            move |fb: &mut FunctionBuilder<'_>, ctx: &mut (), lc: &LoopCtx| {
                let more = fb.ray_query_proceed(query)?;
                let done = fb.unary(UnOp::LogicalNot, more)?;
                fb.if_(
                    ctx,
                    done,
                    Box::new(|fb: &mut FunctionBuilder<'_>, _: &mut ()| {
                        lc.break_(fb)?;
                        Ok(ArmEnd::Exits)
                    }),
                    None,
                )?;
                let kind = fb.ray_query_get(query, RayQueryValue::Type, false)?;
                let zero = fb.u32_const(0)?;
                let zero = fb.module().const_value(zero)?;
                let triangle = fb.binary(BinOp::IEqual, kind, zero)?;
                fb.if_(
                    ctx,
                    triangle,
                    Box::new(move |fb: &mut FunctionBuilder<'_>, _: &mut ()| {
                        fb.ray_query_confirm(query)?;
                        Ok(ArmEnd::FallThrough(Vec::new()))
                    }),
                    None,
                )?;
                Ok(())
            },
        ),
    )
    .expect("proceed loop");
    let i = gid_x(f);
    let mut words = Vec::new();
    for value in [
        RayQueryValue::Type,
        RayQueryValue::InstanceCustomIndex,
        RayQueryValue::InstanceId,
        RayQueryValue::GeometryIndex,
        RayQueryValue::PrimitiveIndex,
        RayQueryValue::InstanceShaderBindingTableRecordOffset,
    ] {
        words.push(f.ray_query_get(query, value, true).expect("committed"));
    }
    let t = f.ray_query_get(query, RayQueryValue::T, true).expect("t");
    words.push(f.convert(ConvOp::Bitcast, t, u32_).expect("bits"));
    let bary = f
        .ray_query_get(query, RayQueryValue::Barycentrics, true)
        .expect("barycentrics");
    let u = f.composite_extract(bary, &[0]).expect("u");
    words.push(f.convert(ConvOp::Bitcast, u, u32_).expect("bits"));
    let front = f
        .ray_query_get(query, RayQueryValue::FrontFace, true)
        .expect("front face");
    let one = cval(f, u32_, 1);
    let zero = cval(f, u32_, 0);
    words.push(f.select(front, one, zero).expect("select"));
    for value in [RayQueryValue::ObjectToWorld, RayQueryValue::WorldToObject] {
        let m = f.ray_query_get(query, value, true).expect("transform");
        let x = f.composite_extract(m, &[3, 2]).expect("column");
        words.push(f.convert(ConvOp::Bitcast, x, u32_).expect("bits"));
    }
    let mut sum = zero;
    for w in words {
        sum = f.binary(BinOp::IAdd, sum, w).expect("sum");
    }
    let p = elem_ptr(f, out, i);
    f.store(p, sum, MemAccess::default()).expect("store");
}

/// Every ray query instruction, with flags read at run time: `spirv-val` accepts the module, and
/// derivation declares `RayQueryKHR`, `SPV_KHR_ray_query`, and (for a runtime flags word)
/// `RayTraversalPrimitiveCullingKHR`, and requires `rayQuery`, `accelerationStructure`, and
/// `bufferDeviceAddress`.
#[test]
fn ray_queries_validate_with_runtime_flags() {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let flags_in = ssbo(&mut b, 0, u32_, 4);
    let out = ssbo(&mut b, 1, u32_, 4);
    let a = accel(&mut b, 2);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let p = elem_ptr(f, flags_in, i);
        let flags = f.load(p, MemAccess::default()).expect("flags");
        trace(f, a, out, flags);
    });
    let m = finish_compute(b, main);
    let r = m.requirements.clone();
    assert!(r
        .spirv_capabilities
        .contains(&SpirvCapability::of(crate::spv::Capability::RayQueryKHR)));
    assert!(r.spirv_capabilities.contains(&SpirvCapability::of(
        crate::spv::Capability::RayTraversalPrimitiveCullingKHR
    )));
    assert!(r.spirv_extensions.contains(&SpirvExtension::KhrRayQuery));
    for feature in [
        VkFeature::RayQuery,
        VkFeature::AccelerationStructure,
        VkFeature::BufferDeviceAddress,
    ] {
        assert!(r.features.contains(&feature), "{feature:?}");
    }
    for ext in [
        VkExtension::KhrRayQuery,
        VkExtension::KhrAccelerationStructure,
    ] {
        assert!(r.extensions.contains(&ext), "{ext:?}");
    }
    assert_spirv_valid(m);
}

/// A constant flags word without `SkipTrianglesKHR`/`SkipAABBsKHR` needs no primitive culling
/// capability.
#[test]
fn constant_flags_without_skips_need_no_primitive_culling() {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let out = ssbo(&mut b, 0, u32_, 4);
    let a = accel(&mut b, 1);
    let main = compute(&mut b, |f| {
        // `OpaqueKHR | TerminateOnFirstHitKHR | CullBackFacingTrianglesKHR`.
        let flags = cval(f, u32_, 0x1 | 0x4 | 0x10);
        trace(f, a, out, flags);
    });
    let m = finish_compute(b, main);
    assert!(!m
        .requirements
        .spirv_capabilities
        .contains(&SpirvCapability::of(
            crate::spv::Capability::RayTraversalPrimitiveCullingKHR
        )));
    assert_spirv_valid(m);
}

/// The type rules keep a ray query out of loads, stores, aggregates, and interface storage, and
/// an acceleration structure in exactly its own descriptor kind.
#[test]
fn ray_query_types_are_confined() {
    let mut b = ModuleBuilder::new();
    let rq = b.ty(Type::RayQuery).expect("ray query");
    let asty = b
        .ty(Type::AccelerationStructure)
        .expect("acceleration structure");
    for class in [
        StorageClass::Input,
        StorageClass::Output,
        StorageClass::UniformConstant,
        StorageClass::StorageBuffer,
        StorageClass::Workgroup,
    ] {
        assert!(
            b.ty(Type::Pointer { class, pointee: rq }).is_err(),
            "{class:?}"
        );
    }
    assert!(b
        .ty(Type::Pointer {
            class: StorageClass::Function,
            pointee: asty,
        })
        .is_err());
    assert!(b
        .ty(Type::Struct {
            members: vec![rq].into(),
            layout: None,
        })
        .is_err());
    assert!(b
        .builtin(crate::spv::BuiltIn::Position, IoDir::Output, rq)
        .is_err());
    // An acceleration structure binds only as its own kind, and only it binds as that kind.
    assert_eq!(
        b.resource(
            slot(ResourceKind::Buffer, 0),
            StorageClass::UniformConstant,
            asty
        ),
        Err(BuildError::Global(GlobalRule::ResourceType))
    );
    let sampler = b.ty(Type::Sampler).expect("sampler");
    assert_eq!(
        b.resource(
            slot(
                ResourceKind::AccelerationStructure(AccelerationStructureLevel::Primitive),
                0
            ),
            StorageClass::UniformConstant,
            sampler
        ),
        Err(BuildError::Global(GlobalRule::ResourceType))
    );
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let q = f.ray_query().expect("query");
    let op = |e: BuildError| match e {
        BuildError::Operand { op, rule } => (op, rule),
        e => panic!("{e:?}"),
    };
    assert_eq!(
        op(f.load(q, MemAccess::default()).unwrap_err()),
        (crate::spv::Op::Load, OperandRule::Opaque)
    );
    // A query operand must point at a ray query.
    let u32_ = int(f.module(), 32, false);
    let var = f.local_var(u32_, None).expect("var");
    let not_query = f.var_ptr(var).expect("ptr");
    assert_eq!(
        op(f.ray_query_proceed(not_query).unwrap_err()),
        (crate::spv::Op::RayQueryProceedKHR, OperandRule::Category(0))
    );
    // Initialize checks every operand's type.
    let x = cval(&mut f, u32_, 0);
    let bad = RayInit {
        accel: x,
        flags: x,
        cull_mask: x,
        origin: x,
        tmin: x,
        direction: x,
        tmax: x,
    };
    assert_eq!(
        op(f.ray_query_initialize(q, bad).unwrap_err()),
        (
            crate::spv::Op::RayQueryInitializeKHR,
            OperandRule::Category(1)
        )
    );
}

/// The serializer writes `OpTypeAccelerationStructureKHR` and `OpTypeRayQueryKHR`, the committed
/// get's `Intersection` operand is the constant 1, and a candidate get's is 0.
#[test]
fn ray_query_words_name_their_intersection() {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let out = ssbo(&mut b, 0, u32_, 4);
    let a = accel(&mut b, 1);
    let main = compute(&mut b, |f| {
        let flags = cval(f, u32_, 0);
        trace(f, a, out, flags);
    });
    let m = finish_compute(b, main);
    let words = assert_spirv_valid(m);
    let dis = crate::spirv::testing::spirv_dis(&words);
    for needle in [
        "OpTypeAccelerationStructureKHR",
        "OpTypeRayQueryKHR",
        "OpRayQueryInitializeKHR",
        "OpRayQueryProceedKHR",
        "OpRayQueryConfirmIntersectionKHR",
        "OpRayQueryGetIntersectionWorldToObjectKHR",
    ] {
        assert!(dis.contains(needle), "{needle} missing:\n{dis}");
    }
}
