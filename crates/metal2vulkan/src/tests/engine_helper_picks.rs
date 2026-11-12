//! `spirv::helper_branches` on fragment modules built here: an `optnone` call outside every branch
//! whose value is read once, in a place SPIRV-Cross then writes it.
//!
//! Each module samples a texture at LOD 0 at the fragment's coordinate, adds 1 to the sample's
//! red with `NoContraction` (SPIRV-Cross's `optnone` `spvFAdd`) in the entry block, and stores
//! the sum to a storage buffer, its one reader: in the arm of a selection on the coordinate (where
//! only helper invocations may take it), in a loop, or in the arm of a selection on a
//! specialization constant (one every invocation of a draw takes alike). SPIRV-Cross moves a value
//! read once into the arm that reads it and keeps one a loop reads where it is defined (checked
//! with Homebrew's `spirv-cross`), so only the first is fenced on `moltenvk-m3`, and none is on a
//! device without the defect (`intel-arl-anv`).

use crate::budget::{Budget, Meter};
use crate::device::DevicePolicy;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::profile;
use crate::spirv::builder::constructs::{ArmEnd, LoopSpec};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::helper_branches::guard_for;
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, Contraction, Dim, ExecMode, ExecModel,
    ImageOperands, ImageType, ImageUsage, IoDir, LayoutRule, MemAccess, StorageClass, Type,
    TypedModule,
};
use crate::spv::ImageFormat;

/// Where the sum's one reader, a store, is.
#[derive(Clone, Copy, Debug)]
enum Reader {
    /// The arm of a selection on the fragment's coordinate.
    Arm,
    /// A loop's body.
    Loop,
    /// The arm of a selection on a specialization constant.
    UniformArm,
}

fn slot(kind: ResourceKind, index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Fragment,
        kind,
        index,
    }
}

fn module(reader: Reader) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let void = b.ty(Type::Void)?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let vec2 = b.ty(Type::Vector { elem: f32_, len: 2 })?;
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
    let coord = b.builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, vec4)?;
    let image = b.ty(Type::Image(ImageType {
        sampled_type: f32_,
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: false,
        usage: ImageUsage::Sampled,
        format: ImageFormat::Unknown,
    }))?;
    let texture = b.resource(
        slot(ResourceKind::Texture, 0),
        StorageClass::UniformConstant,
        image,
    )?;
    let sampler_ty = b.ty(Type::Sampler)?;
    let sampler = b.resource(
        slot(ResourceKind::Sampler, 0),
        StorageClass::UniformConstant,
        sampler_ty,
    )?;
    let rta = b.ty(Type::RuntimeArray {
        elem: f32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    let out = b.resource(
        slot(ResourceKind::Buffer, 0),
        StorageClass::StorageBuffer,
        block,
    )?;
    let spec = b.spec(SpecKey::SampleCount, u32_, ConstBits(1))?;
    let spec = b.const_value(spec)?;
    fn scalar(b: &mut ModuleBuilder, ty: TypeId, bits: u64) -> BResult<Value> {
        let c = b.constant(Constant::Scalar {
            ty,
            bits: ConstBits(bits),
        })?;
        b.const_value(c)
    }
    let zero = scalar(&mut b, u32_, 0)?;
    let fzero = scalar(&mut b, f32_, 0)?;
    let one = scalar(&mut b, f32_, u64::from(1.0f32.to_bits()))?;
    let half = scalar(&mut b, f32_, u64::from(0.5f32.to_bits()))?;
    let mut fb = b.function(void)?;
    let c = fb.module().global_value(coord)?;
    let c = fb.load(c, MemAccess::default())?;
    let x = fb.composite_extract(c, &[0])?;
    let img = fb.module().global_value(texture)?;
    let img = fb.load(img, MemAccess::default())?;
    let s = fb.module().global_value(sampler)?;
    let s = fb.load(s, MemAccess::default())?;
    let si = fb.sampled_image(img, s)?;
    let uv = fb.composite_construct(vec2, &[x, x])?;
    let level = ImageOperands {
        lod: Some(fzero),
        ..ImageOperands::default()
    };
    let texel = fb.sample(si, uv, None, false, &level)?;
    let red = fb.composite_extract(texel, &[0])?;
    let sum = fb.binary_with(BinOp::FAdd, red, one, Contraction::Forbidden)?;
    let base = fb.module().global_value(out)?;
    let p = fb.access_chain(base, &[zero, zero])?;
    let store = move |f: &mut crate::spirv::builder::FunctionBuilder<'_>| {
        f.store(p, sum, MemAccess::default())
    };
    match reader {
        Reader::Arm | Reader::UniformArm => {
            let cond = match reader {
                Reader::Arm => fb.binary(BinOp::FOrdLessThan, x, half)?,
                _ => fb.binary(BinOp::IEqual, spec, zero)?,
            };
            fb.if_(
                &mut (),
                cond,
                Box::new(move |f, _| {
                    store(f)?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                }),
                None,
            )?;
        }
        Reader::Loop => fb.loop_(
            &mut (),
            LoopSpec::default(),
            Box::new(move |f, _, l| {
                store(f)?;
                l.break_(f)
            }),
        )?,
    }
    fb.ret(None)?;
    let id = fb.finish()?;
    b.entry(ExecModel::Fragment, id, vec![ExecMode::OriginUpperLeft])?;
    b.finalize()
}

/// The sum read once in a selection's arm on the coordinate is fenced on `moltenvk-m3`; read in a
/// loop, or in an arm every invocation of the draw takes alike, it is not; nothing is fenced on
/// `intel-arl-anv`. No module records a call the fence could not follow (`exposed`).
#[test]
fn a_call_read_once_in_a_helper_arm_is_fenced_where_spirv_cross_writes_it() {
    let mvk = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let anv = profile("intel-arl-anv", DevicePolicy::default()).expect("profile");
    for (reader, fenced) in [
        (Reader::Arm, true),
        (Reader::Loop, false),
        (Reader::UniformArm, false),
    ] {
        for (stem, p, fenced) in [
            ("moltenvk-m3", &mvk, fenced),
            ("intel-arl-anv", &anv, false),
        ] {
            let mut m = module(reader).expect("module");
            let mut meter = Meter::new(&Budget::standard());
            let (guarded, _) = guard_for(&mut m, p, &mut meter).expect("guard");
            eprintln!("observed: {reader:?} on {stem}: {guarded:?}");
            assert_eq!(guarded.selections, 0, "{reader:?} on {stem}: guards");
            assert_eq!(
                guarded.fences,
                u32::from(fenced),
                "{reader:?} on {stem}: fences"
            );
            assert!(!guarded.exposed, "{reader:?} on {stem}: exposed");
        }
    }
}
