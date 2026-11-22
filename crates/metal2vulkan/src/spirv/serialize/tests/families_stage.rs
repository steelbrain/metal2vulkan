//! `spirv-val` sweep, continued: constructs, stage interfaces and builtins, execution modes,
//! push constants, uniform buffers and layouts, specialization constants, fragment operations.

use super::families::{tys, with_params, Tys};
use super::families_image::{image_ty, load_global, sampler, texture, Img, SAMPLED_2D};
use super::{
    assert_spirv_valid, compute, cval, elem_ptr, finish_compute, gid_x, key, scalar, slot, ssbo,
};
use crate::device::requirements::SpirvCapability;
use crate::pipeline::sym::{PushField, ResourceKind, SpecKey};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, KillForm, LoopCtx, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BuildError, GlobalRule, TypeRule};
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, DerivOp, ExecMode, ExecModel, GlslOp,
    ImageOperands, ImageUsage, Interpolation, IoDecor, IoDir, LayoutRule, MemAccess, Sampling,
    StorageClass, TexelMemory, Type, TypedModule,
};
use crate::spv::BuiltIn;

fn arm<'a>(v: Option<Value>) -> ArmFn<'a, ()> {
    Box::new(move |_: &mut FunctionBuilder<'_>, _: &mut ()| {
        Ok(ArmEnd::FallThrough(v.into_iter().collect()))
    })
}

// ---- Constructs (constructs.rs) ----

#[test]
fn nested_constructs() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let out = ssbo(&mut b, 0, t.u32_, 4);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let zero = cval(f, t.u32_, 0);
        let one = cval(f, t.u32_, 1);
        let ten = cval(f, t.u32_, 10);
        let acc = f.local_var(t.u32_, None).expect("acc");
        let accp = f.var_ptr(acc).expect("acc ptr");
        f.store(accp, zero, MemAccess::default()).expect("init");
        let sel = f.local_var(t.u32_, None).expect("exit selector");
        let c = f.binary(BinOp::ULessThan, i, ten).expect("cmp");
        let mut ctx = ();
        // if/else merging two values.
        let merged = f
            .if_(&mut ctx, c, arm(Some(one)), Some(arm(Some(ten))))
            .expect("if");
        // if without else, one arm returning.
        f.if_(
            &mut ctx,
            c,
            Box::new(|fb: &mut FunctionBuilder<'_>, _: &mut ()| {
                fb.ret(None)?;
                Ok(ArmEnd::Exits)
            }),
            None,
        )
        .expect("if-return");
        // A loop with a nested loop, break, continue, and a switch.
        f.loop_(
            &mut ctx,
            LoopSpec {
                exit_sel: Some(sel),
                ..LoopSpec::default()
            },
            Box::new(
                move |fb: &mut FunctionBuilder<'_>, ctx: &mut (), lc: &LoopCtx| {
                    let x = fb.load(accp, MemAccess::default())?;
                    let done = fb.binary(BinOp::UGreaterThanEqual, x, merged.0[0])?;
                    fb.if_(
                        ctx,
                        done,
                        Box::new(|fb: &mut FunctionBuilder<'_>, _: &mut ()| {
                            lc.break_(fb)?;
                            Ok(ArmEnd::Exits)
                        }),
                        None,
                    )?;
                    fb.loop_(
                        ctx,
                        LoopSpec::default(),
                        Box::new(
                            move |fb: &mut FunctionBuilder<'_>, ctx: &mut (), inner: &LoopCtx| {
                                let y = fb.load(accp, MemAccess::default())?;
                                let y1 = fb.binary(BinOp::IAdd, y, one)?;
                                fb.store(accp, y1, MemAccess::default())?;
                                let m = fb.switch_(
                                    ctx,
                                    y1,
                                    vec![(vec![1, 2], arm(Some(one))), (vec![7], arm(Some(ten)))],
                                    arm(Some(zero)),
                                )?;
                                let odd = fb.binary(BinOp::IEqual, m.0[0], one)?;
                                fb.if_(
                                    ctx,
                                    odd,
                                    Box::new(|fb: &mut FunctionBuilder<'_>, _: &mut ()| {
                                        inner.continue_(fb)?;
                                        Ok(ArmEnd::Exits)
                                    }),
                                    None,
                                )?;
                                inner.break_(fb)
                            },
                        ),
                    )?;
                    Ok(())
                },
            ),
        )
        .expect("loop");
        let x = f.load(accp, MemAccess::default()).expect("load");
        let p = elem_ptr(f, out, i);
        f.store(p, x, MemAccess::default()).expect("store");
    });
    assert_spirv_valid(finish_compute(b, main));
}

#[test]
fn loop_without_break_and_barrier_loops() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let main = compute(&mut b, |f| {
        let mut ctx = ();
        let i = gid_x(f);
        let four = cval(f, t.u32_, 4);
        let c = f.binary(BinOp::ULessThan, i, four).expect("cmp");
        f.loop_(
            &mut ctx,
            LoopSpec::default(),
            Box::new(
                move |fb: &mut FunctionBuilder<'_>, ctx: &mut (), lc: &LoopCtx| {
                    fb.control_barrier(
                        crate::spirv::ir::Scope::Workgroup,
                        crate::spirv::ir::Scope::Workgroup,
                        crate::spirv::ir::Semantics::RELAXED,
                    )?;
                    fb.if_(
                        ctx,
                        c,
                        Box::new(|fb: &mut FunctionBuilder<'_>, _: &mut ()| {
                            lc.break_(fb)?;
                            Ok(ArmEnd::Exits)
                        }),
                        None,
                    )?;
                    Ok(())
                },
            ),
        )
        .expect("barrier loop");
        // A loop that only returns: its merge is unreachable.
        f.loop_(
            &mut ctx,
            LoopSpec::default(),
            Box::new(|fb: &mut FunctionBuilder<'_>, _: &mut (), _: &LoopCtx| fb.ret(None)),
        )
        .expect("returning loop");
    });
    assert_spirv_valid(finish_compute(b, main));
}

// ---- Fragment operations (constructs.rs, ops/misc.rs, ops/glsl.rs) ----

fn fragment(
    modes: Vec<ExecMode>,
    body: impl FnOnce(&mut FunctionBuilder<'_>, &Tys, GlobalId),
) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let input = b
        .io(key("uv"), IoDir::Input, t.vec4, IoDecor::default())
        .expect("input");
    let mut f = b.function(t.void).expect("main");
    body(&mut f, &t, input);
    if f.current_block().is_some() {
        f.ret(None).expect("ret");
    }
    let main = f.finish().expect("finish");
    b.entry(ExecModel::Fragment, main, modes).expect("entry");
    b.finalize().expect("finalize")
}

#[test]
fn kill_terminate_and_demote() {
    for form in [KillForm::Kill, KillForm::TerminateInvocation] {
        let m = fragment(vec![], |f, t, _| {
            let mut ctx = ();
            let h = f.undef(t.bool_).expect("condition");
            f.if_(
                &mut ctx,
                h,
                Box::new(move |fb: &mut FunctionBuilder<'_>, _: &mut ()| {
                    fb.kill(form)?;
                    Ok(ArmEnd::Exits)
                }),
                None,
            )
            .expect("if");
            f.demote().expect("demote");
        });
        assert_spirv_valid(m);
    }
}

#[test]
fn interpolation_functions() {
    let m = fragment(vec![], |f, t, input| {
        let p = f.module().global_value(input).expect("input ptr");
        f.glsl(GlslOp::InterpolateAtCentroid, &[p])
            .expect("centroid");
        let s = cval(f, t.i32_, 0);
        f.glsl(GlslOp::InterpolateAtSample, &[p, s])
            .expect("sample");
        let x = cval(f, t.f32_, 0);
        let o = f.composite_construct(t.vec2, &[x, x]).expect("offset");
        f.glsl(GlslOp::InterpolateAtOffset, &[p, o])
            .expect("offset");
    });
    assert_spirv_valid(m);
}

#[test]
fn fragment_interlock() {
    for mode in [ExecMode::PixelInterlockOrdered] {
        let m = fragment(vec![mode], |f, _, _| {
            f.begin_interlock().expect("begin");
            f.end_interlock().expect("end");
        });
        assert_spirv_valid(m);
    }
}

// ---- Stage interfaces and builtins ----

#[test]
fn vertex_interface() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let two = scalar(&mut b, t.u32_, 2);
    let clip = b
        .ty(Type::Array {
            elem: t.f32_,
            len: two,
            stride: None,
        })
        .expect("clip array");
    let pos = b
        .builtin(BuiltIn::Position, IoDir::Output, t.vec4)
        .expect("position");
    let size = b
        .builtin(BuiltIn::PointSize, IoDir::Output, t.f32_)
        .expect("point size");
    let clip_g = b
        .builtin(BuiltIn::ClipDistance, IoDir::Output, clip)
        .expect("clip");
    let cull_g = b
        .builtin(BuiltIn::CullDistance, IoDir::Output, clip)
        .expect("cull");
    let layer = b
        .builtin(BuiltIn::Layer, IoDir::Output, t.u32_)
        .expect("layer");
    let viewport = b
        .builtin(BuiltIn::ViewportIndex, IoDir::Output, t.u32_)
        .expect("viewport");
    let ins: Vec<GlobalId> = [
        BuiltIn::VertexIndex,
        BuiltIn::InstanceIndex,
        BuiltIn::BaseVertex,
        BuiltIn::BaseInstance,
        BuiltIn::DrawIndex,
        BuiltIn::ViewIndex,
    ]
    .into_iter()
    .map(|bi| b.builtin(bi, IoDir::Input, t.i32_).expect("input builtin"))
    .collect();
    let attr = b
        .io(key("attr"), IoDir::Input, t.vec4, IoDecor::default())
        .expect("attribute");
    let flat = b
        .io(
            key("flat"),
            IoDir::Output,
            t.u32_,
            IoDecor {
                interpolation: Interpolation::Flat,
                ..IoDecor::default()
            },
        )
        .expect("flat output");
    let centroid = b
        .io(
            key("centroid"),
            IoDir::Output,
            t.vec4,
            IoDecor {
                interpolation: Interpolation::NoPerspective,
                sampling: Sampling::Centroid,
                ..IoDecor::default()
            },
        )
        .expect("centroid output");
    let mut f = b.function(t.void).expect("main");
    let a = load_global(&mut f, attr);
    let st = |f: &mut FunctionBuilder<'_>, g: GlobalId, v: Value| {
        let p = f.module().global_value(g).expect("ptr");
        f.store(p, v, MemAccess::default()).expect("store");
    };
    st(&mut f, pos, a);
    st(&mut f, centroid, a);
    let x = cval(&mut f, t.f32_, 0x3f80_0000);
    st(&mut f, size, x);
    let c = f.composite_construct(clip, &[x, x]).expect("clip value");
    st(&mut f, clip_g, c);
    st(&mut f, cull_g, c);
    let mut sum = cval(&mut f, t.i32_, 0);
    for g in ins {
        let v = load_global(&mut f, g);
        sum = f.binary(BinOp::IAdd, sum, v).expect("add");
    }
    let u = f
        .convert(crate::spirv::ir::ConvOp::Bitcast, sum, t.u32_)
        .expect("bitcast");
    st(&mut f, flat, u);
    st(&mut f, layer, u);
    st(&mut f, viewport, u);
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish");
    b.entry(ExecModel::Vertex, main, vec![]).expect("entry");
    assert_spirv_valid(b.finalize().expect("finalize"));
}

#[test]
fn fragment_interface() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let one = scalar(&mut b, t.u32_, 1);
    let mask_ty = b
        .ty(Type::Array {
            elem: t.u32_,
            len: one,
            stride: None,
        })
        .expect("mask array");
    let frag_coord = b
        .builtin(BuiltIn::FragCoord, IoDir::Input, t.vec4)
        .expect("frag coord");
    let front = b
        .builtin(BuiltIn::FrontFacing, IoDir::Input, t.bool_)
        .expect("front facing");
    let helper = b
        .builtin(BuiltIn::HelperInvocation, IoDir::Input, t.bool_)
        .expect("helper");
    let point = b
        .builtin(BuiltIn::PointCoord, IoDir::Input, t.vec2)
        .expect("point coord");
    let ints: Vec<GlobalId> = [
        BuiltIn::SampleId,
        BuiltIn::PrimitiveId,
        BuiltIn::Layer,
        BuiltIn::ViewportIndex,
        BuiltIn::ViewIndex,
    ]
    .into_iter()
    .map(|bi| b.builtin(bi, IoDir::Input, t.i32_).expect("int builtin"))
    .collect();
    let sample_pos = b
        .builtin(BuiltIn::SamplePosition, IoDir::Input, t.vec2)
        .expect("sample position");
    let mask_in = b
        .builtin(BuiltIn::SampleMask, IoDir::Input, mask_ty)
        .expect("mask in");
    let mask_out = b
        .builtin(BuiltIn::SampleMask, IoDir::Output, mask_ty)
        .expect("mask out");
    let depth = b
        .builtin(BuiltIn::FragDepth, IoDir::Output, t.f32_)
        .expect("depth");
    let stencil = b
        .builtin(BuiltIn::FragStencilRefEXT, IoDir::Output, t.i32_)
        .expect("stencil");
    let flat = b
        .io(
            key("flat"),
            IoDir::Input,
            t.u32_,
            IoDecor {
                interpolation: Interpolation::Flat,
                ..IoDecor::default()
            },
        )
        .expect("flat input");
    let sample_in = b
        .io(
            key("per-sample"),
            IoDir::Input,
            t.vec4,
            IoDecor {
                sampling: Sampling::Sample,
                ..IoDecor::default()
            },
        )
        .expect("sample input");
    let color0 = b
        .io(
            key("c0"),
            IoDir::Output,
            t.vec4,
            IoDecor {
                index: Some(0),
                ..IoDecor::default()
            },
        )
        .expect("color 0");
    let color1 = b
        .io(
            key("c1"),
            IoDir::Output,
            t.vec4,
            IoDecor {
                index: Some(1),
                ..IoDecor::default()
            },
        )
        .expect("color 1");
    let mut f = b.function(t.void).expect("main");
    let fc = load_global(&mut f, frag_coord);
    let fr = load_global(&mut f, front);
    let _ = load_global(&mut f, helper);
    let _ = load_global(&mut f, point);
    let _ = load_global(&mut f, sample_pos);
    let _ = load_global(&mut f, sample_in);
    let _ = load_global(&mut f, flat);
    let mut sum = cval(&mut f, t.i32_, 0);
    for g in ints {
        let v = load_global(&mut f, g);
        sum = f.binary(BinOp::IAdd, sum, v).expect("add");
    }
    let st = |f: &mut FunctionBuilder<'_>, g: GlobalId, v: Value| {
        let p = f.module().global_value(g).expect("ptr");
        f.store(p, v, MemAccess::default()).expect("store");
    };
    let m = load_global(&mut f, mask_in);
    st(&mut f, mask_out, m);
    let z = f.composite_extract(fc, &[2]).expect("z");
    let z = f.select(fr, z, z).expect("select");
    st(&mut f, depth, z);
    st(&mut f, stencil, sum);
    st(&mut f, color0, fc);
    st(&mut f, color1, fc);
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish");
    b.entry(
        ExecModel::Fragment,
        main,
        vec![
            ExecMode::DepthReplacing,
            ExecMode::DepthGreater,
            ExecMode::StencilRefReplacing,
            ExecMode::EarlyFragmentTests,
        ],
    )
    .expect("entry");
    assert_spirv_valid(b.finalize().expect("finalize"));
}

/// A `Sample`-decorated input without `SampleId` beside it declares the decoration's capability,
/// `SampleRateShading`, and requires `sampleRateShading`.
#[test]
fn a_sample_input_alone_declares_sample_rate_shading() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let input = b
        .io(
            key("per-sample"),
            IoDir::Input,
            t.vec4,
            IoDecor {
                sampling: Sampling::Sample,
                ..IoDecor::default()
            },
        )
        .expect("sample input");
    let mut f = b.function(t.void).expect("main");
    let _ = load_global(&mut f, input);
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish");
    b.entry(ExecModel::Fragment, main, vec![]).expect("entry");
    let m = b.finalize().expect("finalize");
    assert!(m
        .requirements
        .spirv_capabilities
        .contains(&SpirvCapability::of(
            crate::spv::Capability::SampleRateShading
        )));
    assert!(m
        .requirements
        .features
        .contains(&crate::device::requirements::VkFeature::SampleRateShading));
    assert_spirv_valid(m);
}

/// A fragment `PrimitiveId`, `Layer`, and `ViewportIndex` input declared through each of their
/// alternative enabling capabilities, and the barycentric builtins, validate and derive exactly the
/// chosen capability; a choice outside the alternatives, on a variable that is not a builtin, or a
/// second, different choice is refused.
#[test]
fn fragment_builtin_enablers_and_barycentrics() {
    use crate::spv::Capability as Cap;
    for (primitive, layer, viewport) in [
        (Cap::Geometry, Cap::Geometry, Cap::MultiViewport),
        (
            Cap::Tessellation,
            Cap::ShaderLayer,
            Cap::ShaderViewportIndex,
        ),
    ] {
        let mut b = ModuleBuilder::new();
        let t = tys(&mut b);
        let mut ints = Vec::new();
        for (bi, cap) in [
            (BuiltIn::PrimitiveId, primitive),
            (BuiltIn::Layer, layer),
            (BuiltIn::ViewportIndex, viewport),
        ] {
            let g = b.builtin(bi, IoDir::Input, t.u32_).expect("int builtin");
            b.enable_builtin(g, cap).expect("enabler");
            ints.push(g);
        }
        let bary = b
            .builtin(BuiltIn::BaryCoordKHR, IoDir::Input, t.vec3)
            .expect("barycentric");
        let linear = b
            .builtin(BuiltIn::BaryCoordNoPerspKHR, IoDir::Input, t.vec3)
            .expect("linear barycentric");
        let color = b
            .io(key("c0"), IoDir::Output, t.vec4, IoDecor::default())
            .expect("color");
        let mut f = b.function(t.void).expect("main");
        let mut sum = cval(&mut f, t.u32_, 0);
        for g in ints {
            let v = load_global(&mut f, g);
            sum = f.binary(BinOp::IAdd, sum, v).expect("add");
        }
        let x = load_global(&mut f, bary);
        let y = load_global(&mut f, linear);
        let xy = f.vector_shuffle(x, y, &[0, 1, 3, 4]).expect("shuffle");
        let p = f.module().global_value(color).expect("color ptr");
        f.store(p, xy, MemAccess::default()).expect("store");
        let _ = sum;
        f.ret(None).expect("ret");
        let main = f.finish().expect("finish");
        b.entry(ExecModel::Fragment, main, vec![]).expect("entry");
        let m = b.finalize().expect("finalize");
        let caps = &m.requirements().spirv_capabilities;
        for cap in [primitive, layer, viewport, Cap::FragmentBarycentricKHR] {
            assert!(
                caps.contains(&SpirvCapability::of(cap)),
                "{cap:?} in {caps:?}"
            );
        }
        let chosen = [primitive, layer, viewport];
        for other in [
            Cap::Geometry,
            Cap::Tessellation,
            Cap::ShaderLayer,
            Cap::MultiViewport,
            Cap::ShaderViewportIndex,
        ] {
            assert_eq!(
                caps.contains(&SpirvCapability::of(other)),
                chosen.contains(&other),
                "{other:?}"
            );
        }
        assert_spirv_valid(m);
    }

    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let prim = b
        .builtin(BuiltIn::PrimitiveId, IoDir::Input, t.u32_)
        .expect("primitive id");
    let rule = |b: BuiltIn| Err(BuildError::Global(GlobalRule::BuiltInEnabler(Some(b))));
    assert_eq!(
        b.enable_builtin(prim, Cap::ShaderLayer),
        rule(BuiltIn::PrimitiveId)
    );
    b.enable_builtin(prim, Cap::Tessellation)
        .expect("tessellation");
    b.enable_builtin(prim, Cap::Tessellation)
        .expect("the same choice again");
    assert_eq!(
        b.enable_builtin(prim, Cap::Geometry),
        rule(BuiltIn::PrimitiveId)
    );
    let coord = b
        .builtin(BuiltIn::FragCoord, IoDir::Input, t.vec4)
        .expect("frag coord");
    assert_eq!(
        b.enable_builtin(coord, Cap::Geometry),
        rule(BuiltIn::FragCoord)
    );
    let flat = b
        .io(key("flat"), IoDir::Input, t.u32_, IoDecor::default())
        .expect("input");
    assert_eq!(
        b.enable_builtin(flat, Cap::Geometry),
        Err(BuildError::Global(GlobalRule::BuiltInEnabler(None)))
    );
}

#[test]
fn compute_builtins_and_subgroup_builtins() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let uvec3 = super::vec(&mut b, t.u32_, 3);
    let v3: Vec<GlobalId> = [
        BuiltIn::GlobalInvocationId,
        BuiltIn::LocalInvocationId,
        BuiltIn::WorkgroupId,
        BuiltIn::NumWorkgroups,
    ]
    .into_iter()
    .map(|bi| b.builtin(bi, IoDir::Input, uvec3).expect("uvec3 builtin"))
    .collect();
    let s: Vec<GlobalId> = [
        BuiltIn::LocalInvocationIndex,
        BuiltIn::SubgroupSize,
        BuiltIn::SubgroupLocalInvocationId,
        BuiltIn::NumSubgroups,
        BuiltIn::SubgroupId,
    ]
    .into_iter()
    .map(|bi| b.builtin(bi, IoDir::Input, t.u32_).expect("u32 builtin"))
    .collect();
    let out = ssbo(&mut b, 0, t.u32_, 4);
    let main = compute(&mut b, |f| {
        let mut sum = cval(f, t.u32_, 0);
        for g in &v3 {
            let v = load_global(f, *g);
            let x = f.composite_extract(v, &[0]).expect("x");
            sum = f.binary(BinOp::IAdd, sum, x).expect("add");
        }
        for g in &s {
            let v = load_global(f, *g);
            sum = f.binary(BinOp::IAdd, sum, v).expect("add");
        }
        let p = elem_ptr(f, out, sum);
        f.store(p, sum, MemAccess::default()).expect("store");
    });
    assert_spirv_valid(finish_compute(b, main));
}

// ---- Execution modes ----

#[test]
fn compute_modes() {
    let modes = vec![
        ExecMode::SignedZeroInfNanPreserve { width: 16 },
        ExecMode::SignedZeroInfNanPreserve { width: 32 },
        ExecMode::SignedZeroInfNanPreserve { width: 64 },
        ExecMode::SubgroupUniformControlFlow,
        ExecMode::MaximallyReconverges,
    ];
    let m = with_params(
        ExecModel::GLCompute,
        modes,
        |t| vec![t.f16, t.f32_, t.f64_],
        |f, p, _| {
            for v in p {
                f.binary(BinOp::FAdd, *v, *v).expect("fadd");
            }
        },
    );
    assert_spirv_valid(m);
}

#[test]
fn derivative_group_compute() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let smp = sampler(&mut b, 0);
    let tex_ty = image_ty(&mut b, t.f32_, SAMPLED_2D);
    let tex = texture(&mut b, 0, tex_ty);
    let main = compute(&mut b, |f| {
        let x = cval(f, t.f32_, 0);
        let c = f.composite_construct(t.vec2, &[x, x]).expect("coord");
        f.derivative(DerivOp::DPdx, c).expect("dpdx");
        let img = load_global(f, tex);
        let s = load_global(f, smp);
        let si = f.sampled_image(img, s).expect("si");
        f.sample(si, c, None, false, &ImageOperands::default())
            .expect("implicit sample");
    });
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![
            ExecMode::DerivativeGroupLinear,
            ExecMode::LocalSize { x: 8, y: 8, z: 1 },
        ],
    )
    .expect("entry");
    assert_spirv_valid(b.finalize().expect("finalize"));
}

#[test]
fn spec_constants() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let flag = b
        .spec(SpecKey::MultiRegion, t.bool_, ConstBits(1))
        .expect("bool spec");
    let fspec = b
        .spec(SpecKey::TgLength(0), t.f32_, ConstBits(0x3f80_0000))
        .expect("float spec");
    let out = ssbo(&mut b, 0, t.f32_, 4);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let fl = f.module().const_value(flag).expect("flag");
        let fv = f.module().const_value(fspec).expect("float");
        let zero = cval(f, t.f32_, 0);
        let v = f.select(fl, fv, zero).expect("select");
        let p = elem_ptr(f, out, i);
        f.store(p, v, MemAccess::default()).expect("store");
    });
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 8, y: 1, z: 1 }],
    )
    .expect("entry");
    assert_spirv_valid(b.finalize().expect("finalize"));
}

/// `OpSpecConstantOp` constants (the threadgroup arena's length from a specialized argument length)
/// size a `Workgroup` array and are read in the body; operand types are checked.
#[test]
fn spec_constant_ops_size_a_workgroup_array() {
    use crate::spirv::error::ConstRule;
    use crate::spirv::ir::SpecOp;
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let len = b
        .spec(SpecKey::TgLength(0), t.u32_, ConstBits(16))
        .expect("length");
    let k = |b: &mut ModuleBuilder, v| scalar(b, t.u32_, v);
    let (fifteen, mask, two, zero, one) = (
        k(&mut b, 15),
        k(&mut b, u64::from(!15u32)),
        k(&mut b, 2),
        k(&mut b, 0),
        k(&mut b, 1),
    );
    let up = b
        .spec_op(SpecOp::IAdd, t.u32_, &[len, fifteen])
        .expect("add");
    let up = b
        .spec_op(SpecOp::BitwiseAnd, t.u32_, &[up, mask])
        .expect("and");
    let words = b
        .spec_op(SpecOp::ShiftRightLogical, t.u32_, &[up, two])
        .expect("shift");
    let eight = k(&mut b, 8);
    let small = b
        .spec_op(SpecOp::ULessThan, t.bool_, &[words, eight])
        .expect("less");
    let words = b
        .spec_op(SpecOp::Select, t.u32_, &[small, eight, words])
        .expect("select");
    assert_eq!(
        b.spec_op(SpecOp::IAdd, t.u32_, &[len, fifteen]),
        Ok(b.spec_op(SpecOp::IAdd, t.u32_, &[len, fifteen])
            .expect("same")),
        "interned"
    );
    let izero = scalar(&mut b, t.i32_, 0);
    for (op, ty, operands) in [
        (SpecOp::IAdd, t.u32_, vec![fifteen, two]),
        (SpecOp::IAdd, t.i32_, vec![len, fifteen]),
        (SpecOp::ULessThan, t.u32_, vec![len, zero]),
        (SpecOp::Select, t.u32_, vec![len, one, zero]),
        (SpecOp::ULessThan, t.bool_, vec![len, izero]),
        (SpecOp::IAdd, t.u32_, vec![len]),
    ] {
        assert_eq!(
            b.spec_op(op, ty, &operands),
            Err(BuildError::Constant(ConstRule::SpecOpOperands)),
            "{op:?}"
        );
    }
    let arr = b
        .ty(Type::Array {
            elem: t.u32_,
            len: words,
            stride: None,
        })
        .expect("spec-length array");
    let g = b
        .plain_global(StorageClass::Workgroup, arr, None)
        .expect("workgroup");
    let out = ssbo(&mut b, 0, t.u32_, 4);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let base = f.module().global_value(g).expect("arena");
        let p = f.access_chain(base, &[i]).expect("chain");
        f.store(p, i, MemAccess::default()).expect("store");
        let n = f.module().const_value(words).expect("words");
        let q = elem_ptr(f, out, i);
        f.store(q, n, MemAccess::default()).expect("store");
    });
    assert_spirv_valid(finish_compute(b, main));
}

// ---- Push constants, uniform buffers, layouts, descriptor arrays ----

fn laid(b: &mut ModuleBuilder, rule: LayoutRule, block: bool, members: &[TypeId]) -> TypeId {
    let layout = natural_layout(b.types(), rule, block, members).expect("layout");
    b.ty(Type::Struct {
        members: members.into(),
        layout: Some(layout),
    })
    .expect("laid struct")
}

#[test]
fn push_constants_uniform_buffers_and_layouts() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    b.push_member(PushField::RootTableAddress, t.u64_, 0)
        .expect("push u64");
    b.push_member(PushField::DispatchRegion, t.uvec4, 16)
        .expect("push uvec4");
    let push = b.push_global().expect("push block");
    let four = scalar(&mut b, t.u32_, 4);
    // std140 UBO with a matrix and a vec4 array.
    let arr16 = b
        .ty(Type::Array {
            elem: t.vec4,
            len: four,
            stride: Some(16),
        })
        .expect("vec4 array");
    let ubo140 = laid(&mut b, LayoutRule::Std140, true, &[t.mat4, arr16, t.f32_]);
    let u140 = b
        .resource(slot(ResourceKind::Buffer, 0), StorageClass::Uniform, ubo140)
        .expect("std140 ubo");
    // std430 `u32[]`-style UBO view (needs uniformBufferStandardLayout).
    let arr4 = b
        .ty(Type::Array {
            elem: t.u32_,
            len: four,
            stride: Some(4),
        })
        .expect("u32 array");
    let ubo430 = laid(&mut b, LayoutRule::Std430, true, &[arr4]);
    let u430 = b
        .resource(slot(ResourceKind::Buffer, 1), StorageClass::Uniform, ubo430)
        .expect("std430 ubo");
    // Scalar-layout SSBO with a vec3 followed by a float (needs scalarBlockLayout).
    let rta = b
        .ty(Type::RuntimeArray {
            elem: t.u32_,
            stride: Some(4),
        })
        .expect("rta");
    let sc = laid(&mut b, LayoutRule::Scalar, true, &[t.vec3, t.f32_, rta]);
    let ssbo_scalar = b
        .resource(
            slot(ResourceKind::Buffer, 2),
            StorageClass::StorageBuffer,
            sc,
        )
        .expect("scalar ssbo");
    // 8- and 16-bit storage.
    let narrow = laid(&mut b, LayoutRule::Std430, true, &[t.u8_, t.u16_, t.f16]);
    let ssbo_narrow = b
        .resource(
            slot(ResourceKind::Buffer, 3),
            StorageClass::StorageBuffer,
            narrow,
        )
        .expect("narrow ssbo");
    let main = compute(&mut b, |f| {
        let zero = cval(f, t.u32_, 0);
        let one = cval(f, t.u32_, 1);
        let two = cval(f, t.u32_, 2);
        let m = |f: &mut FunctionBuilder<'_>, g: GlobalId, idx: &[Value]| {
            let base = f.module().global_value(g).expect("base");
            let p = f.access_chain(base, idx).expect("chain");
            f.load(p, MemAccess::default()).expect("load")
        };
        m(f, push, &[zero]);
        m(f, push, &[one]);
        m(f, u140, &[zero]);
        m(f, u140, &[one, two]);
        m(f, u430, &[zero, one]);
        let v3 = m(f, ssbo_scalar, &[zero]);
        let base = f.module().global_value(ssbo_scalar).expect("base");
        let p = f.access_chain(base, &[zero]).expect("chain");
        f.store(p, v3, MemAccess::default()).expect("store vec3");
        for k in [zero, one, two] {
            let base = f.module().global_value(ssbo_narrow).expect("base");
            let p = f.access_chain(base, &[k]).expect("chain");
            let v = f.load(p, MemAccess::default()).expect("load narrow");
            f.store(p, v, MemAccess::default()).expect("store narrow");
        }
    });
    assert_spirv_valid(finish_compute(b, main));
}

/// A compute module with two implicitly laid-out workgroup variables, the second
/// zero-initialized.
fn workgroup_module() -> Result<TypedModule, BuildError> {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let first = b
        .ty(Type::Struct {
            members: [t.u32_, t.u16_, t.u8_].into(),
            layout: None,
        })
        .expect("implicit struct");
    let wg = b
        .plain_global(StorageClass::Workgroup, first, None)
        .expect("workgroup struct");
    let null = b.constant(Constant::Null { ty: t.vec4 }).expect("null");
    let zeroed = b
        .plain_global(StorageClass::Workgroup, t.vec4, Some(null))
        .expect("zero-initialized workgroup");
    let main = compute(&mut b, |f| {
        let zero = cval(f, t.u32_, 0);
        let base = f.module().global_value(wg).expect("wg");
        let p = f.access_chain(base, &[zero]).expect("member");
        let v = f.load(p, MemAccess::default()).expect("load");
        f.store(p, v, MemAccess::default()).expect("store");
        let _ = load_global(f, zeroed);
    });
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 8, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// `Workgroup` memory is implicitly laid out: an explicitly laid-out variable or pointee there,
/// `Block` or not, is refused, so no module needs `SPV_KHR_workgroup_memory_explicit_layout`.
#[test]
fn workgroup_memory_is_implicitly_laid_out_and_zero_initialized() {
    assert_spirv_valid(workgroup_module().expect("implicit layout"));
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let refused = Some(BuildError::Type(TypeRule::PointeeForClass(
        StorageClass::Workgroup,
    )));
    for block in [true, false] {
        let explicit = laid(&mut b, LayoutRule::Std430, block, &[t.u32_, t.u16_, t.u8_]);
        assert_eq!(
            b.plain_global(StorageClass::Workgroup, explicit, None)
                .err(),
            refused,
            "block {block}"
        );
        let pointer = Type::Pointer {
            class: StorageClass::Workgroup,
            pointee: explicit,
        };
        assert_eq!(b.ty(pointer).err(), refused, "block {block}");
    }
}

#[test]
fn descriptor_arrays_with_dynamic_indices() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let tex_ty = image_ty(&mut b, t.f32_, SAMPLED_2D);
    let storage = image_ty(
        &mut b,
        t.f32_,
        Img {
            usage: ImageUsage::Storage,
            format: crate::spv::ImageFormat::Rgba32f,
            ..SAMPLED_2D
        },
    );
    let rt_tex = b
        .ty(Type::RuntimeArray {
            elem: tex_ty,
            stride: None,
        })
        .expect("texture array");
    let eight = scalar(&mut b, t.u32_, 8);
    let arr_storage = b
        .ty(Type::Array {
            elem: storage,
            len: eight,
            stride: None,
        })
        .expect("storage image array");
    let tg = texture(&mut b, 0, rt_tex);
    let sg = texture(&mut b, 1, arr_storage);
    let rta = b
        .ty(Type::RuntimeArray {
            elem: t.u32_,
            stride: Some(4),
        })
        .expect("rta");
    let blk = laid(&mut b, LayoutRule::Std430, true, &[rta]);
    let bufs = b
        .ty(Type::Array {
            elem: blk,
            len: eight,
            stride: None,
        })
        .expect("buffer array");
    let bg = b
        .resource(
            slot(ResourceKind::Buffer, 0),
            StorageClass::StorageBuffer,
            bufs,
        )
        .expect("buffer array resource");
    let smp = sampler(&mut b, 0);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let zero = cval(f, t.u32_, 0);
        let base = f.module().global_value(tg).expect("textures");
        let p = f.access_chain(base, &[i]).expect("texture element");
        let img = f.load(p, MemAccess::default()).expect("image");
        let s = load_global(f, smp);
        let si = f.sampled_image(img, s).expect("si");
        let x = cval(f, t.f32_, 0);
        let c = f.composite_construct(t.vec2, &[x, x]).expect("coord");
        f.sample(
            si,
            c,
            None,
            false,
            &ImageOperands {
                lod: Some(x),
                ..ImageOperands::default()
            },
        )
        .expect("sample");
        let base = f.module().global_value(sg).expect("storage images");
        let p = f.access_chain(base, &[i]).expect("storage element");
        let img = f.load(p, MemAccess::default()).expect("storage image");
        let ic = super::families_image::ivec(f, 2, 0);
        f.read(img, ic, &ImageOperands::default(), TexelMemory::PRIVATE)
            .expect("read");
        let base = f.module().global_value(bg).expect("buffers");
        let p = f
            .access_chain(base, &[i, zero, zero])
            .expect("buffer element");
        let v = f.load(p, MemAccess::default()).expect("load");
        f.store(p, v, MemAccess::default()).expect("store");
    });
    assert_spirv_valid(finish_compute(b, main));
}
