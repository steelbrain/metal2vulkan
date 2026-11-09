//! [`snapshot_textures`] on hand-built fragment modules: a fetched value that reaches only
//! arithmetic and an output qualifies, directly, through a variable or through a call; one that
//! reaches a derivative, a branch, an implicit sample, a quad operation, an integer divisor or a
//! dynamic index, a store elsewhere, an atomic or an image write, directly, through a variable,
//! or into and out of a callee, does not; nor does a module with an interlock or a coherent
//! resource, of a function that writes memory, or a slot past 127. A branch around a demote alone
//! qualifies. Each slot left out is stated with the clauses that leave it out.

use super::snapshot_textures;
use crate::budget::{Budget, Meter};
use crate::pipeline::interface::SnapshotRefusal;
use crate::pipeline::sym::{CombinedSampler, ResourceKind, ResourceSlot, SamplerRef, Stage};
use crate::spirv::builder::constructs::ArmEnd;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ConstBits, Constant, ConvOp, DerivOp, Dim, ExecMode,
    ExecModel, ImageOperands, ImageType, ImageUsage, IoDir, LayoutRule, MemAccess, QuadDirection,
    Scope, Semantics, StorageClass, TexelMemory, Type, TypedModule,
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
    /// Swapped across the quad.
    QuadSwap,
    /// Its bits stored to a storage buffer's word.
    StorageStore,
    /// Its bits added to a storage buffer's word by an atomic.
    AtomicAdd,
    /// Written to a storage image of slot 3.
    ImageWrite,
    /// An output, in a module whose storage image of slot 3 is `Coherent`.
    CoherentImage,
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
    // A storage buffer of words at buffer slot 2, and a float storage image at texture slot 3,
    // for the sinks that write them.
    let words = match sink {
        Sink::StorageStore | Sink::AtomicAdd => {
            let rta = b
                .ty(Type::RuntimeArray {
                    elem: u32_,
                    stride: Some(4),
                })
                .expect("array");
            let layout =
                natural_layout(b.types(), LayoutRule::Std430, true, &[rta]).expect("layout");
            let block = b
                .ty(Type::Struct {
                    members: [rta].into(),
                    layout: Some(layout),
                })
                .expect("block");
            let slot = ResourceSlot {
                stage: Stage::Fragment,
                kind: ResourceKind::Buffer,
                index: 2,
            };
            Some(
                b.resource(slot, StorageClass::StorageBuffer, block)
                    .expect("slot 2"),
            )
        }
        _ => None,
    };
    let storage = match sink {
        Sink::ImageWrite | Sink::CoherentImage => {
            let ty = b
                .ty(Type::Image(ImageType {
                    sampled_type: f32_,
                    dim: Dim::D2,
                    depth: false,
                    arrayed: false,
                    multisampled: false,
                    usage: ImageUsage::Storage,
                    format: crate::spv::ImageFormat::Rgba32f,
                }))
                .expect("storage image");
            let slot = ResourceSlot {
                stage: Stage::Fragment,
                kind: ResourceKind::Texture,
                index: 3,
            };
            let g = b
                .resource(slot, StorageClass::UniformConstant, ty)
                .expect("slot 3");
            if sink == Sink::CoherentImage {
                b.coherent(g).expect("coherent");
            }
            Some(g)
        }
        _ => None,
    };
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
        Sink::Output | Sink::Interlocked | Sink::CoherentImage => {
            f.binary(BinOp::FAdd, texel, texel).expect("sum")
        }
        Sink::QuadSwap => f
            .group_quad_swap(texel, QuadDirection::Horizontal)
            .expect("quad swap"),
        Sink::StorageStore | Sink::AtomicAdd => {
            let words = f
                .module()
                .global_value(words.expect("words"))
                .expect("words");
            let word = f.access_chain(words, &[zero, zero]).expect("word");
            let x = f.composite_extract(texel, &[0]).expect("x");
            let bits = f.convert(ConvOp::Bitcast, x, u32_).expect("bits");
            match sink {
                Sink::StorageStore => f.store(word, bits, MemAccess::default()).expect("store"),
                _ => {
                    f.atomic(
                        AtomicOp::IAdd,
                        word,
                        Scope::Device,
                        Semantics::RELAXED,
                        bits,
                    )
                    .expect("atomic");
                }
            }
            texel
        }
        Sink::ImageWrite => {
            let g = f
                .module()
                .global_value(storage.expect("storage"))
                .expect("image");
            let image = f.load(g, MemAccess::default()).expect("storage image");
            f.write(
                image,
                coord,
                texel,
                &ImageOperands::default(),
                TexelMemory::Private { volatile: false },
            )
            .expect("write");
            texel
        }
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
/// sample at it, a quad swap of it, a divisor, a signed dividend beside `-1`, a dynamic index, a
/// store of it to a storage buffer, an atomic of it, or an image write of it leaves the slot out,
/// as does an interlock or a coherent resource anywhere in the module. A branch on it that only
/// decides whether the invocation demotes itself, in either arm, qualifies; one whose arm also
/// writes, or whose arms' values merge apart, does not. Each slot left out is stated with the one
/// clause that leaves it out. The analysis releases every charge.
#[test]
fn only_values_no_helper_can_pass_on_qualify() {
    use SnapshotRefusal as R;
    for (sink, refused) in [
        (Sink::Output, None),
        (Sink::ThroughVariable, None),
        (Sink::Derivative, Some(R::Derivative)),
        (Sink::DerivativeThroughVariable, Some(R::Derivative)),
        (Sink::Branch, Some(R::Branch)),
        (Sink::SampleCoordinate, Some(R::Sample)),
        (Sink::Interlocked, Some(R::Interlock)),
        (Sink::ThroughCall, None),
        (Sink::DerivativeOfCall, Some(R::Derivative)),
        (Sink::BranchInCall, Some(R::Branch)),
        (Sink::BranchInCallThroughPointer, Some(R::Branch)),
        (Sink::StoredByCallee, Some(R::Branch)),
        (Sink::Divisor, Some(R::UndefinedOperand)),
        (Sink::UnsignedDividend, None),
        (Sink::SignedDividend, None),
        (Sink::SignedDividendByMinusOne, Some(R::UndefinedOperand)),
        (Sink::DynamicIndex, Some(R::UndefinedOperand)),
        (Sink::DynamicVector, None),
        (Sink::DemoteGuard, None),
        (Sink::DemoteInElse, None),
        (Sink::DemoteAndStore, Some(R::Branch)),
        (Sink::DemoteMergingValues, Some(R::Branch)),
        (Sink::QuadSwap, Some(R::GroupOperation)),
        (Sink::StorageStore, Some(R::Store)),
        (Sink::AtomicAdd, Some(R::Atomic)),
        (Sink::ImageWrite, Some(R::ImageWrite)),
        (Sink::CoherentImage, Some(R::Coherent)),
    ] {
        let mut meter = Meter::new(&Budget::default());
        let judged = snapshot_textures(&module(sink), &[0], false, &mut meter).expect("analysis");
        assert_eq!(meter.ledger(), 0, "{sink:?}: every charge released");
        let (slots, refusals): (&[u32], Vec<(u32, SnapshotRefusal)>) = match refused {
            None => (&[0], Vec::new()),
            Some(why) => (&[], vec![(0, why)]),
        };
        assert_eq!(judged.slots, slots, "{sink:?}");
        assert_eq!(judged.refusals, refusals, "{sink:?}");
    }
}

/// A function that writes memory leaves every slot out for it, beside what the walk finds of its
/// values: the qualifying output alone, and the derivative too. A slot past 127 is left out
/// unwalked, beside a slot below it that qualifies.
#[test]
fn a_slot_left_out_by_the_function_or_its_index_says_so() {
    use SnapshotRefusal as R;
    let mut meter = Meter::new(&Budget::default());
    let mut judged = |sink: Sink, own: &[u32], writes: bool| {
        snapshot_textures(&module(sink), own, writes, &mut meter).expect("analysis")
    };
    let written = judged(Sink::Output, &[0], true);
    assert!(written.slots.is_empty(), "writes memory");
    assert_eq!(written.refusals, [(0, R::WritesMemory)], "writes memory");
    let both = judged(Sink::Derivative, &[0], true);
    assert_eq!(
        both.refusals,
        [(0, R::Derivative), (0, R::WritesMemory)],
        "writes memory, and a derivative"
    );
    let past = judged(Sink::Output, &[0, 200], false);
    assert_eq!(past.slots, [0], "slot 0 qualifies");
    assert_eq!(past.refusals, [(200, R::Other)], "slot 200 is not walked");
    assert_eq!(meter.ledger(), 0, "every charge released");
}
