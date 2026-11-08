//! [`snapshot_textures`] on hand-built fragment modules: a fetched value that reaches only
//! arithmetic and an output qualifies, directly, through a variable or through a call; one that
//! reaches a derivative, a branch, an implicit sample, an integer divisor or a dynamic index, or a
//! store elsewhere, directly, through a variable, or into and out of a callee, does not; nor does a
//! module with an interlock. A branch around a demote alone qualifies.

use super::snapshot_textures;
use crate::budget::{Budget, Meter};
use crate::pipeline::sym::{CombinedSampler, ResourceKind, ResourceSlot, SamplerRef, Stage};
use crate::spirv::builder::constructs::ArmEnd;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, DerivOp, Dim, ExecMode, ExecModel, ImageOperands,
    ImageType, ImageUsage, IoDir, MemAccess, StorageClass, Type, TypedModule,
};

/// Where slot 0's fetched value goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sink {
    /// Added to itself and stored to a `Private` variable.
    Output,
    /// Through a `Function` variable to the output.
    ThroughVariable,
    /// Its derivative stored to the output.
    Derivative,
    /// Through a `Function` variable to a derivative.
    DerivativeThroughVariable,
    /// Compared, and the comparison branched on.
    Branch,
    /// The coordinate of an implicit-level sample of slot 1.
    SampleCoordinate,
    /// An output, in a module whose entry also holds an interlock.
    Interlocked,
    /// Through a call whose callee returns its square.
    ThroughCall,
    /// Its derivative, of a call's result that returns it.
    DerivativeOfCall,
    /// Through a call whose callee branches on it.
    BranchInCall,
    /// Stored in a variable whose pointer a callee takes, loads through, and branches on.
    BranchInCallThroughPointer,
    /// Handed to a callee that stores it through a pointer the caller loads and branches on.
    StoredByCallee,
    /// Its bits the divisor of an unsigned division.
    Divisor,
    /// Its bits the dividend of an unsigned division by 3.
    UnsignedDividend,
    /// Its bits the dividend of a signed division by 3.
    SignedDividend,
    /// Its bits the dividend of a signed division by -1.
    SignedDividendByMinusOne,
    /// Its bits, masked, the index of a dynamic extract.
    DynamicIndex,
    /// The vector of a dynamic extract at a constant index.
    DynamicVector,
    /// Compared, and the invocation demoted where the comparison holds.
    DemoteGuard,
    /// Compared, and the invocation demoted in the else arm of the comparison.
    DemoteInElse,
    /// Compared, and the invocation demoted and the output written where the comparison holds.
    DemoteAndStore,
    /// Compared, the invocation demoted where the comparison holds, and the arms' different values
    /// merged.
    DemoteMergingValues,
}

/// Whether lane 0 of `v` is above one half.
fn above_half(f: &mut FunctionBuilder<'_>, v: Value, f32_: TypeId) -> Value {
    let x = f.composite_extract(v, &[0]).expect("x");
    let half = scalar(f, f32_, u64::from(0.5f32.to_bits()));
    f.binary(BinOp::FOrdGreaterThan, x, half).expect("compare")
}

/// Branch on whether lane 0 of `v` is above one half.
fn branch_on(f: &mut FunctionBuilder<'_>, v: Value, f32_: TypeId) {
    let c = above_half(f, v, f32_);
    let arm: crate::spirv::builder::constructs::ArmFn<'_, ()> =
        Box::new(|_, _| Ok(ArmEnd::FallThrough(Vec::new())));
    f.if_(&mut (), c, arm, None).expect("if");
}

fn scalar(f: &mut FunctionBuilder<'_>, ty: crate::spirv::ids::TypeId, bits: u64) -> Value {
    let c = f
        .module()
        .constant(Constant::Scalar {
            ty,
            bits: ConstBits(bits),
        })
        .expect("constant");
    f.module().const_value(c).expect("value")
}

fn module(sink: Sink) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let u32_ = b
        .ty(Type::Int {
            width: 32,
            signed: false,
        })
        .expect("uint");
    let f32_ = b.ty(Type::Float { width: 32 }).expect("float");
    let vec2 = b.ty(Type::Vector { elem: f32_, len: 2 }).expect("vec2");
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 }).expect("vec4");
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 }).expect("uvec2");
    let image = b
        .ty(Type::Image(ImageType {
            sampled_type: f32_,
            dim: Dim::D2,
            depth: false,
            arrayed: false,
            multisampled: false,
            usage: ImageUsage::Sampled,
            format: crate::spv::ImageFormat::Unknown,
        }))
        .expect("image");
    let sampled = b.ty(Type::SampledImage { image }).expect("sampled image");
    let t = b
        .resource(
            ResourceSlot {
                stage: Stage::Fragment,
                kind: ResourceKind::Texture,
                index: 0,
            },
            StorageClass::UniformConstant,
            image,
        )
        .expect("slot 0");
    let u = b
        .resource(
            ResourceSlot {
                stage: Stage::Fragment,
                kind: ResourceKind::CombinedTextureSampler(CombinedSampler::new(
                    SamplerRef::Slot(1),
                    false,
                    false,
                )),
                index: 1,
            },
            StorageClass::UniformConstant,
            sampled,
        )
        .expect("slot 1");
    let out = b
        .builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, vec4)
        .expect("FragCoord");
    let color = b
        .plain_global(StorageClass::Private, vec4, None)
        .expect("color");
    let i32_ = b
        .ty(Type::Int {
            width: 32,
            signed: true,
        })
        .expect("int");
    let void = b.ty(Type::Void).expect("void");
    let ptr = b
        .ty(Type::Pointer {
            class: StorageClass::Function,
            pointee: vec4,
        })
        .expect("pointer");
    let callee = match sink {
        Sink::ThroughCall | Sink::DerivativeOfCall | Sink::BranchInCall => {
            let mut g = b.routine(vec4).expect("routine");
            let x = g.param(vec4).expect("x");
            let y = match sink {
                Sink::BranchInCall => {
                    branch_on(&mut g, x, f32_);
                    x
                }
                _ => g.binary(BinOp::FMul, x, x).expect("square"),
            };
            g.ret(Some(y)).expect("ret");
            Some(g.finish().expect("callee"))
        }
        Sink::BranchInCallThroughPointer => {
            let mut g = b.routine(void).expect("routine");
            let p = g.param(ptr).expect("p");
            let v = g.load(p, MemAccess::default()).expect("load");
            branch_on(&mut g, v, f32_);
            g.ret(None).expect("ret");
            Some(g.finish().expect("callee"))
        }
        Sink::StoredByCallee => {
            let mut g = b.routine(void).expect("routine");
            let p = g.param(ptr).expect("p");
            let x = g.param(f32_).expect("x");
            let zero = scalar(&mut g, u32_, 0);
            let e = g.access_chain(p, &[zero]).expect("lane");
            g.store(e, x, MemAccess::default()).expect("store");
            g.ret(None).expect("ret");
            Some(g.finish().expect("callee"))
        }
        _ => None,
    };
    let mut f = b.function(void).expect("main");
    let tv = f.module().global_value(t).expect("t");
    let uv = f.module().global_value(u).expect("u");
    let _ = out;
    let cv = f.module().global_value(color).expect("color");
    let img = f.load(tv, MemAccess::default()).expect("t image");
    let zero = scalar(&mut f, u32_, 0);
    let coord = f.composite_construct(uvec2, &[zero, zero]).expect("coord");
    let ops = ImageOperands {
        lod: Some(zero),
        ..ImageOperands::default()
    };
    if sink == Sink::Interlocked {
        f.begin_interlock().expect("begin");
    }
    let texel = f.fetch(img, coord, &ops).expect("fetch");
    if sink == Sink::Interlocked {
        f.end_interlock().expect("end");
    }
    let value = match sink {
        Sink::Output | Sink::Interlocked => f.binary(BinOp::FAdd, texel, texel).expect("sum"),
        Sink::ThroughVariable | Sink::DerivativeThroughVariable => {
            let var = f.local_var(vec4, None).expect("var");
            let p = f.var_ptr(var).expect("pointer");
            f.store(p, texel, MemAccess::default()).expect("store");
            let back = f.load(p, MemAccess::default()).expect("load");
            match sink {
                Sink::DerivativeThroughVariable => {
                    f.derivative(DerivOp::DPdx, back).expect("derivative")
                }
                _ => back,
            }
        }
        Sink::Derivative => f.derivative(DerivOp::DPdx, texel).expect("derivative"),
        Sink::Branch => {
            branch_on(&mut f, texel, f32_);
            texel
        }
        Sink::ThroughCall | Sink::BranchInCall | Sink::DerivativeOfCall => {
            let callee = callee.expect("callee");
            let r = f.call(callee, &[texel]).expect("call").expect("result");
            match sink {
                Sink::DerivativeOfCall => f.derivative(DerivOp::DPdx, r).expect("derivative"),
                _ => r,
            }
        }
        Sink::BranchInCallThroughPointer | Sink::StoredByCallee => {
            let callee = callee.expect("callee");
            let var = f.local_var(vec4, None).expect("var");
            let p = f.var_ptr(var).expect("pointer");
            if sink == Sink::StoredByCallee {
                let x = f.composite_extract(texel, &[0]).expect("x");
                f.call(callee, &[p, x]).expect("call");
                let back = f.load(p, MemAccess::default()).expect("load");
                branch_on(&mut f, back, f32_);
            } else {
                f.store(p, texel, MemAccess::default()).expect("store");
                f.call(callee, &[p]).expect("call");
            }
            texel
        }
        Sink::Divisor | Sink::UnsignedDividend => {
            let x = f.composite_extract(texel, &[0]).expect("x");
            let bits = f.convert(ConvOp::Bitcast, x, u32_).expect("bits");
            let other = scalar(&mut f, u32_, 3);
            let (a, d) = match sink {
                Sink::Divisor => (other, bits),
                _ => (bits, other),
            };
            f.binary(BinOp::UDiv, a, d).expect("quotient");
            texel
        }
        Sink::SignedDividend | Sink::SignedDividendByMinusOne => {
            let x = f.composite_extract(texel, &[0]).expect("x");
            let bits = f.convert(ConvOp::Bitcast, x, i32_).expect("bits");
            let d = match sink {
                Sink::SignedDividend => 3,
                _ => u64::from(u32::MAX),
            };
            let d = scalar(&mut f, i32_, d);
            f.binary(BinOp::SDiv, bits, d).expect("quotient");
            texel
        }
        Sink::DynamicIndex | Sink::DynamicVector => {
            let index = match sink {
                Sink::DynamicIndex => {
                    let x = f.composite_extract(texel, &[0]).expect("x");
                    let bits = f.convert(ConvOp::Bitcast, x, u32_).expect("bits");
                    let three = scalar(&mut f, u32_, 3);
                    f.binary(BinOp::BitwiseAnd, bits, three).expect("index")
                }
                _ => zero,
            };
            f.vector_extract_dynamic(texel, index).expect("lane");
            texel
        }
        Sink::DemoteGuard | Sink::DemoteAndStore => {
            let c = above_half(&mut f, texel, f32_);
            let store = sink == Sink::DemoteAndStore;
            let arm: crate::spirv::builder::constructs::ArmFn<'_, ()> = Box::new(move |f, _| {
                f.demote()?;
                if store {
                    f.store(cv, texel, MemAccess::default())?;
                }
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            f.if_(&mut (), c, arm, None).expect("if");
            texel
        }
        Sink::DemoteInElse | Sink::DemoteMergingValues => {
            let c = above_half(&mut f, texel, f32_);
            let merging = sink == Sink::DemoteMergingValues;
            let other = f.binary(BinOp::FAdd, texel, texel).expect("sum");
            let kept: crate::spirv::builder::constructs::ArmFn<'_, ()> = Box::new(move |_, _| {
                Ok(ArmEnd::FallThrough(match merging {
                    true => vec![other],
                    false => Vec::new(),
                }))
            });
            let demoted: crate::spirv::builder::constructs::ArmFn<'_, ()> =
                Box::new(move |f, _| {
                    f.demote()?;
                    Ok(ArmEnd::FallThrough(match merging {
                        true => vec![texel],
                        false => Vec::new(),
                    }))
                });
            let merged = f.if_(&mut (), c, kept, Some(demoted)).expect("if");
            merged.0.first().copied().unwrap_or(texel)
        }
        Sink::SampleCoordinate => {
            let si = f.load(uv, MemAccess::default()).expect("u");
            let xy = f.vector_shuffle(texel, texel, &[0, 1]).expect("xy");
            let _ = vec2;
            f.sample(si, xy, None, false, &ImageOperands::default())
                .expect("sample")
        }
    };
    f.store(cv, value, MemAccess::default()).expect("output");
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish");
    let modes = match sink {
        Sink::Interlocked => vec![ExecMode::PixelInterlockOrdered],
        _ => Vec::new(),
    };
    b.entry(ExecModel::Fragment, main, modes).expect("entry");
    b.finalize().expect("finalize")
}

/// A fetched value that reaches arithmetic and a color output alone, directly, through a
/// `Function` variable or through a call, qualifies, as does an unsigned dividend, a signed one
/// beside a constant divisor without `-1`, and a dynamically indexed vector; a derivative of it,
/// directly, through a variable or of a call's result, a branch on it, in the entry or in a callee
/// it reaches as an argument, through a pointer argument's memory or back out of it, an implicit
/// sample at it, a divisor, a signed dividend beside `-1`, or a dynamic index leaves the slot out,
/// as does an interlock anywhere in the module. A branch on it that only decides whether the
/// invocation demotes itself, in either arm, qualifies; one whose arm also writes, or whose arms'
/// values merge apart, does not. The analysis releases every charge.
#[test]
fn only_values_no_helper_can_pass_on_qualify() {
    for (sink, qualifies) in [
        (Sink::Output, true),
        (Sink::ThroughVariable, true),
        (Sink::Derivative, false),
        (Sink::DerivativeThroughVariable, false),
        (Sink::Branch, false),
        (Sink::SampleCoordinate, false),
        (Sink::Interlocked, false),
        (Sink::ThroughCall, true),
        (Sink::DerivativeOfCall, false),
        (Sink::BranchInCall, false),
        (Sink::BranchInCallThroughPointer, false),
        (Sink::StoredByCallee, false),
        (Sink::Divisor, false),
        (Sink::UnsignedDividend, true),
        (Sink::SignedDividend, true),
        (Sink::SignedDividendByMinusOne, false),
        (Sink::DynamicIndex, false),
        (Sink::DynamicVector, true),
        (Sink::DemoteGuard, true),
        (Sink::DemoteInElse, true),
        (Sink::DemoteAndStore, false),
        (Sink::DemoteMergingValues, false),
    ] {
        let mut meter = Meter::new(&Budget::default());
        let slots = snapshot_textures(&module(sink), &[0], &mut meter).expect("analysis");
        assert_eq!(meter.ledger(), 0, "{sink:?}: every charge released");
        let want: &[u32] = if qualifies { &[0] } else { &[] };
        assert_eq!(slots, want, "{sink:?}");
    }
}
