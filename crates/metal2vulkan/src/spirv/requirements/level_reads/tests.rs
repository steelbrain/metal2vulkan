//! [`level_reads`] on hand-built fragment modules, one read of texture slot 0 each, and
//! [`LevelReads::highest_level`] under samplers of each kind.

use super::level_reads;
use crate::budget::{Budget, Meter};
use crate::pipeline::interface::{LevelReads, SampledLevels, SamplerLevels};
use crate::pipeline::sym::{CombinedSampler, ResourceKind, ResourceSlot, SamplerRef, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, Dim, ExecModel, GlslOp, ImageOperands, ImageQuery,
    ImageType, ImageUsage, MemAccess, StorageClass, TexelMemory, Type, TypedModule,
};

/// How a module reads texture slot 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Read {
    /// An implicit-LOD sample through the combined binding of runtime sampler 0.
    Implicit,
    /// A sample at the constant level of detail 1.5.
    ExplicitConstant,
    /// A sample at a level of detail the module computes.
    ExplicitComputed,
    /// A sample from gradients.
    Gradients,
    /// A sample with a minimum level of detail.
    MinLod,
    /// A sample through the pairing of a separate image with constexpr sampler words.
    SeparatePair,
    /// A gather.
    Gather,
    /// A fetch at the constant level 2.
    FetchConstant,
    /// A fetch at the lowering's containment of level 1: `UMin(1, levels - 1)`, beside `1 <
    /// levels`.
    FetchContained,
    /// A fetch at a level the module computes.
    FetchComputed,
    /// A fetch without a level.
    FetchBase,
    /// A level count added to itself.
    LevelsElsewhere,
    /// A level-of-detail query.
    QueryLod,
    /// A size query at the constant level 3.
    SizeConstant,
    /// A size query without a level.
    Size,
    /// An image write of a storage image of the slot.
    Write,
}

const CONSTEXPR: [u64; 2] = [0x12, 0x34];

fn scalar(f: &mut FunctionBuilder<'_>, ty: TypeId, bits: u64) -> Value {
    let c = f
        .module()
        .constant(Constant::Scalar {
            ty,
            bits: ConstBits(bits),
        })
        .expect("constant");
    f.module().const_value(c).expect("value")
}

fn slot(kind: ResourceKind, index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Fragment,
        kind,
        index,
    }
}

fn module(read: Read) -> TypedModule {
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
    let image_of = |b: &mut ModuleBuilder, usage| {
        b.ty(Type::Image(ImageType {
            sampled_type: f32_,
            dim: Dim::D2,
            depth: false,
            arrayed: false,
            multisampled: false,
            usage,
            format: match usage {
                ImageUsage::Storage => crate::spv::ImageFormat::Rgba32f,
                _ => crate::spv::ImageFormat::Unknown,
            },
        }))
        .expect("image")
    };
    let image = image_of(&mut b, ImageUsage::Sampled);
    let storage = image_of(&mut b, ImageUsage::Storage);
    let sampled = b.ty(Type::SampledImage { image }).expect("sampled image");
    let sampler_ty = b.ty(Type::Sampler).expect("sampler");
    let combined = CombinedSampler::new(SamplerRef::Slot(0), false, false);
    let pair = CombinedSampler::new(SamplerRef::Constexpr(CONSTEXPR), false, false);
    let t = b
        .resource(
            slot(ResourceKind::Texture, 0),
            StorageClass::UniformConstant,
            match read {
                Read::Write => storage,
                _ => image,
            },
        )
        .expect("texture");
    let ts = b
        .resource(
            slot(ResourceKind::CombinedTextureSampler(combined), 0),
            StorageClass::UniformConstant,
            sampled,
        )
        .expect("combined");
    let separate = ResourceSlot::separate_texture(Stage::Fragment, 0, pair).expect("separate");
    let st = b
        .resource(separate, StorageClass::UniformConstant, image)
        .expect("separate texture");
    let ss = b
        .resource(
            ResourceSlot::separate_sampler(Stage::Fragment, pair),
            StorageClass::UniformConstant,
            sampler_ty,
        )
        .expect("separate sampler");
    let color = b
        .plain_global(StorageClass::Private, vec4, None)
        .expect("color");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("main");
    let tv = f.module().global_value(t).expect("t");
    let tsv = f.module().global_value(ts).expect("ts");
    let stv = f.module().global_value(st).expect("st");
    let ssv = f.module().global_value(ss).expect("ss");
    let cv = f.module().global_value(color).expect("color");
    let zero = scalar(&mut f, u32_, 0);
    let one = scalar(&mut f, u32_, 1);
    let half = scalar(&mut f, f32_, 0.5f32.to_bits().into());
    let uv = f.composite_construct(vec2, &[half, half]).expect("uv");
    let texel = f.composite_construct(uvec2, &[zero, zero]).expect("texel");
    let at = |f: &mut FunctionBuilder<'_>, lod: Option<Value>| ImageOperands {
        lod,
        ..ImageOperands::default()
    };
    // A level the module computes: the color's first lane, converted.
    let computed_float = |f: &mut FunctionBuilder<'_>| {
        let c = f.load(cv, MemAccess::default()).expect("color");
        f.composite_extract(c, &[0]).expect("lane")
    };
    let value = match read {
        Read::Implicit
        | Read::ExplicitConstant
        | Read::ExplicitComputed
        | Read::Gradients
        | Read::MinLod
        | Read::Gather
        | Read::QueryLod => {
            let si = f.load(tsv, MemAccess::default()).expect("ts");
            let ops = match read {
                Read::ExplicitConstant => {
                    let lod = scalar(&mut f, f32_, 1.5f32.to_bits().into());
                    at(&mut f, Some(lod))
                }
                Read::ExplicitComputed => {
                    let lod = computed_float(&mut f);
                    at(&mut f, Some(lod))
                }
                Read::Gradients => ImageOperands {
                    grad: Some((uv, uv)),
                    ..ImageOperands::default()
                },
                Read::MinLod => ImageOperands {
                    min_lod: Some(half),
                    ..ImageOperands::default()
                },
                _ => ImageOperands::default(),
            };
            match read {
                Read::Gather => f.gather(si, uv, 0, &ops).expect("gather"),
                Read::QueryLod => {
                    let lod = f.query_lod(si, uv).expect("query lod");
                    let x = f.composite_extract(lod, &[0]).expect("x");
                    f.composite_construct(vec4, &[x, x, x, x]).expect("splat")
                }
                _ => f.sample(si, uv, None, false, &ops).expect("sample"),
            }
        }
        Read::SeparatePair => {
            let img = f.load(stv, MemAccess::default()).expect("st");
            let s = f.load(ssv, MemAccess::default()).expect("ss");
            let si = f.sampled_image(img, s).expect("pair");
            f.sample(si, uv, None, false, &ImageOperands::default())
                .expect("sample")
        }
        Read::FetchConstant | Read::FetchContained | Read::FetchComputed | Read::FetchBase => {
            let img = f.load(tv, MemAccess::default()).expect("t");
            let lod = match read {
                Read::FetchConstant => Some(scalar(&mut f, u32_, 2)),
                Read::FetchContained => {
                    let levels = f.query(ImageQuery::Levels, img).expect("levels");
                    let top = f.binary(BinOp::ISub, levels, one).expect("top");
                    let _ = f.binary(BinOp::ULessThan, one, levels).expect("in range");
                    Some(f.glsl(GlslOp::UMin, &[one, top]).expect("clamped"))
                }
                Read::FetchComputed => {
                    let x = computed_float(&mut f);
                    Some(f.convert(ConvOp::ConvertFToU, x, u32_).expect("level"))
                }
                _ => None,
            };
            f.fetch(img, texel, &at(&mut f, lod)).expect("fetch")
        }
        Read::LevelsElsewhere | Read::SizeConstant | Read::Size => {
            let img = f.load(tv, MemAccess::default()).expect("t");
            let n = match read {
                Read::LevelsElsewhere => {
                    let levels = f.query(ImageQuery::Levels, img).expect("levels");
                    f.binary(BinOp::IAdd, levels, levels).expect("twice")
                }
                Read::SizeConstant => {
                    let three = scalar(&mut f, u32_, 3);
                    let size = f.query(ImageQuery::SizeLod(three), img).expect("size");
                    f.composite_extract(size, &[0]).expect("width")
                }
                _ => {
                    let size = f.query(ImageQuery::Size, img).expect("size");
                    f.composite_extract(size, &[0]).expect("width")
                }
            };
            let x = f.convert(ConvOp::ConvertUToF, n, f32_).expect("x");
            f.composite_construct(vec4, &[x, x, x, x]).expect("splat")
        }
        Read::Write => {
            let img = f.load(tv, MemAccess::default()).expect("t");
            let c = f.load(cv, MemAccess::default()).expect("color");
            let mem = TexelMemory::Private { volatile: false };
            f.write(img, texel, c, &ImageOperands::default(), mem)
                .expect("write");
            c
        }
    };
    f.store(cv, value, MemAccess::default()).expect("output");
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish");
    b.entry(ExecModel::Fragment, main, Vec::new())
        .expect("entry");
    b.finalize().expect("finalize")
}

fn sampled(sampler: SamplerRef, lod: Option<f32>, computed: bool) -> SampledLevels {
    SampledLevels {
        sampler,
        highest_lod_bits: lod.map(f32::to_bits),
        computed,
    }
}

/// Each read states the levels it reaches: a sample by its sampler, a constant level of detail
/// or a computed one; a gather, a fetch and a size query the level they name, the lowering's
/// containment of a constant included; and a fetch or level at a computed level, a level count
/// used otherwise, a level-of-detail query, a minimum level of detail and a storage write any
/// level.
#[test]
fn each_read_states_the_levels_it_reaches() {
    let runtime = SamplerRef::Slot(0);
    let constexpr = SamplerRef::Constexpr(CONSTEXPR);
    let reads = |any: bool, named: Option<u32>, sampled: Vec<SampledLevels>| LevelReads {
        slot: 0,
        any,
        named,
        sampled,
    };
    for (read, want) in [
        (
            Read::Implicit,
            reads(false, None, vec![sampled(runtime, None, true)]),
        ),
        (
            Read::ExplicitConstant,
            reads(false, None, vec![sampled(runtime, Some(1.5), false)]),
        ),
        (
            Read::ExplicitComputed,
            reads(false, None, vec![sampled(runtime, None, true)]),
        ),
        (
            Read::Gradients,
            reads(false, None, vec![sampled(runtime, None, true)]),
        ),
        (Read::MinLod, reads(true, None, Vec::new())),
        (
            Read::SeparatePair,
            reads(false, None, vec![sampled(constexpr, None, true)]),
        ),
        (Read::Gather, reads(false, Some(0), Vec::new())),
        (Read::FetchConstant, reads(false, Some(2), Vec::new())),
        (Read::FetchContained, reads(false, Some(1), Vec::new())),
        (Read::FetchComputed, reads(true, None, Vec::new())),
        (Read::FetchBase, reads(false, Some(0), Vec::new())),
        (Read::LevelsElsewhere, reads(true, None, Vec::new())),
        (Read::QueryLod, reads(true, None, Vec::new())),
        (Read::SizeConstant, reads(false, Some(3), Vec::new())),
        (Read::Size, reads(false, None, Vec::new())),
        (Read::Write, reads(true, None, Vec::new())),
    ] {
        let mut meter = Meter::new(&Budget::default());
        let found = level_reads(&module(read), &mut meter).expect("analysis");
        assert_eq!(meter.ledger(), 0, "{read:?}: every charge released");
        assert_eq!(found, vec![want], "{read:?}");
    }
}

fn levels(min_lod: f32, max_lod: f32, linear: bool, finite_texels: bool) -> SamplerLevels {
    SamplerLevels {
        min_lod,
        max_lod,
        bias: 0.0,
        linear,
        finite_texels,
    }
}

/// The highest level a slot's reads reach under a sampler: the nearest mipmap mode's rounding of
/// the clamped level of detail, the linear one's next level unless a whole level of finite texels
/// gives it no weight, a constant clamped into the sampler's bounds, a NaN at `maxLod`, and none
/// where a read reaches any level, the sampler is unknown, or its bounds are crossed.
#[test]
fn the_highest_level_follows_the_bound_sampler() {
    let s = SamplerRef::Slot(0);
    let computed = LevelReads {
        slot: 0,
        any: false,
        named: None,
        sampled: vec![sampled(s, None, true)],
    };
    let constant = |c: f32| LevelReads {
        sampled: vec![sampled(s, Some(c), false)],
        ..computed.clone()
    };
    let at = |reads: &LevelReads, l: SamplerLevels| {
        reads.highest_level(|r| (r.sampler == s).then(|| r.level(l)).flatten())
    };
    // Metal's `NotMipmapped`, as `vk_sampler` maps it: the first level alone.
    assert_eq!(at(&computed, levels(0.0, 0.25, false, true)), Some(0));
    assert_eq!(at(&computed, levels(0.0, 1.0, false, true)), Some(1));
    assert_eq!(at(&computed, levels(0.0, 1.4, false, true)), Some(1));
    assert_eq!(at(&computed, levels(0.0, 1.5, false, true)), Some(2));
    assert_eq!(at(&computed, levels(0.0, 1.0, true, true)), Some(1));
    assert_eq!(at(&computed, levels(0.0, 1.0, true, false)), Some(2));
    assert_eq!(at(&computed, levels(0.0, 1.25, true, true)), Some(2));
    assert_eq!(at(&constant(1.0), levels(0.0, 1000.0, true, true)), Some(1));
    assert_eq!(
        at(&constant(1.0), levels(0.0, 1000.0, false, false)),
        Some(1)
    );
    assert_eq!(
        at(&constant(1.0), levels(0.0, 1000.0, true, false)),
        Some(2)
    );
    assert_eq!(at(&constant(3.0), levels(0.0, 1.0, false, true)), Some(1));
    assert_eq!(at(&constant(-2.0), levels(0.0, 1.0, false, true)), Some(0));
    assert_eq!(
        at(&constant(f32::NAN), levels(0.0, 2.0, false, true)),
        Some(2)
    );
    // A constexpr sampler's bias raises a constant level of detail, and a negative one is not
    // counted on to lower it.
    let biased = |bias: f32| SamplerLevels {
        bias,
        ..levels(0.0, 1000.0, false, true)
    };
    assert_eq!(at(&constant(1.0), biased(1.0)), Some(2));
    assert_eq!(at(&constant(2.0), biased(-1.0)), Some(2));
    assert_eq!(at(&computed, biased(1.0)), None);
    assert!(SamplerLevels::finite_texels(
        crate::vk::Format::B8G8R8A8_UNORM
    ));
    assert!(!SamplerLevels::finite_texels(
        crate::vk::Format::R16G16_SFLOAT
    ));
    // Metal's `lodMaxClamp` of `FLT_MAX`: every level.
    assert_eq!(at(&computed, levels(0.0, f32::MAX, false, true)), None);
    assert_eq!(at(&computed, levels(2.0, 1.0, false, true)), None);
    assert_eq!(computed.highest_level(|_| None), None);
    let fetched = LevelReads {
        named: Some(2),
        ..computed.clone()
    };
    assert_eq!(at(&fetched, levels(0.0, 1.0, false, true)), Some(2));
    let any = LevelReads {
        any: true,
        ..fetched
    };
    assert_eq!(at(&any, levels(0.0, 0.25, false, true)), None);
}
