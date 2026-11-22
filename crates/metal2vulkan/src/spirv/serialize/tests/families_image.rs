//! `spirv-val` sweep, continued: images, samplers, derivatives, image atomics.

use super::families::{tys, Tys};
use super::{assert_spirv_valid, compute, cval, finish_compute, int, key, slot, vec};
use crate::device::requirements::SpirvCapability;
use crate::pipeline::sym::ResourceKind;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    AtomicOp, DerivOp, Dim, ExecModel, ImageOperands, ImageQuery, ImageType, ImageUsage, IoDecor,
    IoDir, MemAccess, Scope, Semantics, StorageClass, TexelMemory, Type,
};
use crate::spv::ImageFormat;

/// An image type (all `OpTypeImage` operands).
#[derive(Clone, Copy)]
pub(super) struct Img {
    pub dim: Dim,
    pub depth: bool,
    pub arrayed: bool,
    pub ms: bool,
    pub usage: ImageUsage,
    pub format: ImageFormat,
}

pub(super) const SAMPLED_2D: Img = Img {
    dim: Dim::D2,
    depth: false,
    arrayed: false,
    ms: false,
    usage: ImageUsage::Sampled,
    format: ImageFormat::Unknown,
};

pub(super) fn image_ty(b: &mut ModuleBuilder, sampled_type: TypeId, i: Img) -> TypeId {
    b.ty(Type::Image(ImageType {
        sampled_type,
        dim: i.dim,
        depth: i.depth,
        arrayed: i.arrayed,
        multisampled: i.ms,
        usage: i.usage,
        format: i.format,
    }))
    .expect("image type")
}

pub(super) fn texture(b: &mut ModuleBuilder, index: u32, ty: TypeId) -> GlobalId {
    b.resource(
        slot(ResourceKind::Texture, index),
        StorageClass::UniformConstant,
        ty,
    )
    .expect("texture")
}

pub(super) fn sampler(b: &mut ModuleBuilder, index: u32) -> GlobalId {
    let ty = b.ty(Type::Sampler).expect("sampler");
    b.resource(
        slot(ResourceKind::Sampler, index),
        StorageClass::UniformConstant,
        ty,
    )
    .expect("sampler")
}

pub(super) fn load_global(f: &mut FunctionBuilder<'_>, g: GlobalId) -> Value {
    let p = f.module().global_value(g).expect("global");
    f.load(p, MemAccess::default()).expect("load")
}

pub(super) fn ivec(f: &mut FunctionBuilder<'_>, n: u8, v: u64) -> Value {
    let i = int(f.module(), 32, true);
    let c = cval(f, i, v);
    let ty = vec(f.module(), i, n);
    f.composite_construct(ty, &vec![c; usize::from(n)])
        .expect("ivec")
}

/// A constant `ivec<n>` (for `ConstOffset`).
pub(super) fn ivec_const(f: &mut FunctionBuilder<'_>, n: u8, v: u64) -> Value {
    let i = int(f.module(), 32, true);
    let c = super::scalar(f.module(), i, v);
    let ty = vec(f.module(), i, n);
    let k = f
        .module()
        .constant(crate::spirv::ir::Constant::Composite {
            ty,
            parts: vec![c; usize::from(n)].into(),
        })
        .expect("constant vector");
    f.module().const_value(k).expect("value")
}

fn fvec(f: &mut FunctionBuilder<'_>, t: &Tys, n: u8) -> Value {
    let x = cval(f, t.f32_, 0x3f00_0000);
    if n == 1 {
        return x;
    }
    let ty = vec(f.module(), t.f32_, n);
    f.composite_construct(ty, &vec![x; usize::from(n)])
        .expect("vec")
}

#[test]
fn sampling_in_fragment() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let smp = sampler(&mut b, 0);
    let kinds = [
        SAMPLED_2D,
        Img {
            arrayed: true,
            ..SAMPLED_2D
        },
        Img {
            depth: true,
            ..SAMPLED_2D
        },
        Img {
            dim: Dim::D3,
            ..SAMPLED_2D
        },
        Img {
            dim: Dim::Cube,
            ..SAMPLED_2D
        },
        Img {
            dim: Dim::Cube,
            arrayed: true,
            ..SAMPLED_2D
        },
        Img {
            dim: Dim::D1,
            ..SAMPLED_2D
        },
        Img {
            dim: Dim::D1,
            arrayed: true,
            ..SAMPLED_2D
        },
    ];
    let g: Vec<GlobalId> = kinds
        .iter()
        .enumerate()
        .map(|(i, k)| {
            let ty = image_ty(&mut b, t.f32_, *k);
            texture(&mut b, i as u32, ty)
        })
        .collect();
    let out = b
        .io(key("color"), IoDir::Output, t.vec4, IoDecor::default())
        .expect("output");
    let mut f = b.function(t.void).expect("main");
    let x = fvec(&mut f, &t, 1);
    let c2 = fvec(&mut f, &t, 2);
    let c3 = fvec(&mut f, &t, 3);
    let c4 = fvec(&mut f, &t, 4);
    let off = ivec_const(&mut f, 2, 1);
    let dyn_off = ivec(&mut f, 2, 1);
    let none = ImageOperands::default();
    let s = load_global(&mut f, smp);
    // 2D: implicit (bias, const offset, min lod), explicit (lod, grad + min lod), proj, gather.
    let img = load_global(&mut f, g[0]);
    let si = f.sampled_image(img, s).expect("sampled image");
    let last = f.sample(si, c2, None, false, &none).expect("implicit");
    let with = |o: ImageOperands| o;
    f.sample(
        si,
        c2,
        None,
        false,
        &with(ImageOperands {
            bias: Some(x),
            const_offset: Some(off),
            min_lod: Some(x),
            ..none
        }),
    )
    .expect("implicit with operands");
    f.sample(
        si,
        c2,
        None,
        false,
        &ImageOperands {
            lod: Some(x),
            ..none
        },
    )
    .expect("explicit lod");
    f.sample(
        si,
        c2,
        None,
        false,
        &ImageOperands {
            grad: Some((c2, c2)),
            min_lod: Some(x),
            ..none
        },
    )
    .expect("explicit grad");
    f.sample(si, c3, None, true, &none).expect("proj");
    f.sample(
        si,
        c3,
        None,
        true,
        &ImageOperands {
            lod: Some(x),
            ..none
        },
    )
    .expect("proj explicit");
    f.gather(si, c2, 1, &none).expect("gather");
    f.gather(
        si,
        c2,
        0,
        &ImageOperands {
            const_offset: Some(off),
            ..none
        },
    )
    .expect("gather const offset");
    f.gather(
        si,
        c2,
        2,
        &ImageOperands {
            offset: Some(dyn_off),
            ..none
        },
    )
    .expect("gather offset");
    f.query_lod(si, c2).expect("query lod");
    f.image(si).expect("image of sampled image");
    let level = cval(&mut f, t.i32_, 0);
    f.query(ImageQuery::SizeLod(level), img).expect("size lod");
    f.query(ImageQuery::Levels, img).expect("levels");
    // Arrays, 3D, cubes, 1D.
    for (gi, coord) in [(1usize, c3), (3, c3), (4, c3), (5, c4), (6, x), (7, c2)] {
        let img = load_global(&mut f, g[gi]);
        let si = f.sampled_image(img, s).expect("si");
        f.sample(si, coord, None, false, &none).expect("sample");
        f.query(ImageQuery::SizeLod(level), img).expect("size lod");
    }
    let img = load_global(&mut f, g[4]);
    let si = f.sampled_image(img, s).expect("si");
    f.gather(si, c3, 3, &none).expect("cube gather");
    // Depth: dref implicit/explicit/proj and dref gather.
    let img = load_global(&mut f, g[2]);
    let si = f.sampled_image(img, s).expect("si");
    f.sample(si, c2, Some(x), false, &none).expect("dref");
    f.sample(
        si,
        c2,
        Some(x),
        false,
        &ImageOperands {
            lod: Some(x),
            ..none
        },
    )
    .expect("dref explicit");
    f.sample(si, c3, Some(x), true, &none).expect("dref proj");
    f.dref_gather(si, c2, x, &none).expect("dref gather");
    // Derivatives.
    for op in [DerivOp::DPdx, DerivOp::DPdy, DerivOp::Fwidth] {
        f.derivative(op, c4)
            .unwrap_or_else(|e| panic!("{op:?}: {e}"));
        f.derivative(op, x)
            .unwrap_or_else(|e| panic!("{op:?}: {e}"));
    }
    let po = f.module().global_value(out).expect("out");
    f.store(po, last, MemAccess::default()).expect("store");
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish");
    b.entry(ExecModel::Fragment, main, vec![]).expect("entry");
    assert_spirv_valid(b.finalize().expect("finalize"));
}

/// The sparse form of every access (sample with and without `Dref`, implicit and explicit LOD,
/// gather, dref gather, fetch, storage read) and `OpImageSparseTexelsResident` of its code validate
/// and declare `SparseResidency`, which the registry gates on `shaderResourceResidency`; a sparse
/// fetch of a texel buffer and a residency test of a value that is no `u32` code are refused.
#[test]
fn sparse_accesses_and_residency() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let smp = sampler(&mut b, 0);
    let kinds = [
        SAMPLED_2D,
        Img {
            depth: true,
            ..SAMPLED_2D
        },
        Img {
            usage: ImageUsage::Storage,
            format: ImageFormat::Rgba32f,
            ..SAMPLED_2D
        },
        Img {
            dim: Dim::Buffer,
            ..SAMPLED_2D
        },
    ];
    let g: Vec<GlobalId> = kinds
        .iter()
        .enumerate()
        .map(|(i, k)| {
            let ty = image_ty(&mut b, t.f32_, *k);
            texture(&mut b, i as u32, ty)
        })
        .collect();
    let out = b
        .io(key("color"), IoDir::Output, t.vec4, IoDecor::default())
        .expect("output");
    let mut f = b.function(t.void).expect("main");
    let x = fvec(&mut f, &t, 1);
    let c2 = fvec(&mut f, &t, 2);
    let i2 = ivec(&mut f, 2, 0);
    let zero = cval(&mut f, t.i32_, 0);
    let none = ImageOperands::default();
    let lod = ImageOperands {
        lod: Some(x),
        ..none
    };
    let s = load_global(&mut f, smp);
    let mut results = Vec::new();
    let img = load_global(&mut f, g[0]);
    let si = f.sampled_image(img, s).expect("sampled image");
    results.push(
        f.sparse_sample(si, c2, None, &none)
            .expect("sparse implicit"),
    );
    results.push(
        f.sparse_sample(si, c2, None, &lod)
            .expect("sparse explicit"),
    );
    results.push(f.sparse_gather(si, c2, 1, &none).expect("sparse gather"));
    let level = ImageOperands {
        lod: Some(zero),
        ..none
    };
    results.push(f.sparse_fetch(img, i2, &level).expect("sparse fetch"));
    let img = load_global(&mut f, g[1]);
    let si = f.sampled_image(img, s).expect("sampled image");
    results.push(
        f.sparse_sample(si, c2, Some(x), &none)
            .expect("sparse dref"),
    );
    results.push(
        f.sparse_sample(si, c2, Some(x), &lod)
            .expect("sparse dref explicit"),
    );
    results.push(
        f.sparse_dref_gather(si, c2, x, &none)
            .expect("sparse dref gather"),
    );
    let img = load_global(&mut f, g[2]);
    results.push(
        f.sparse_read(img, i2, &none, TexelMemory::PRIVATE)
            .expect("sparse read"),
    );
    let img = load_global(&mut f, g[3]);
    assert!(f.sparse_fetch(img, zero, &none).is_err(), "texel buffer");
    assert!(f.texels_resident(x).is_err(), "a float code");
    let mut all = None;
    for r in &results {
        let code = f.composite_extract(*r, &[0]).expect("code");
        let resident = f.texels_resident(code).expect("resident");
        all = Some(match all {
            None => resident,
            Some(a) => f
                .binary(crate::spirv::ir::BinOp::LogicalAnd, a, resident)
                .expect("and"),
        });
    }
    let texel = f.composite_extract(results[0], &[1]).expect("texel");
    let null = f
        .module()
        .constant(crate::spirv::ir::Constant::Null { ty: t.vec4 })
        .expect("null");
    let null = f.module().const_value(null).expect("null value");
    let v = f.select(all.expect("codes"), texel, null).expect("select");
    let po = f.module().global_value(out).expect("out");
    f.store(po, v, MemAccess::default()).expect("store");
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish");
    b.entry(ExecModel::Fragment, main, vec![]).expect("entry");
    let m = b.finalize().expect("finalize");
    let req = &m.requirements;
    assert!(req.spirv_capabilities.contains(&SpirvCapability::of(
        crate::spv::Capability::SparseResidency
    )));
    assert!(req
        .features
        .contains(&crate::device::requirements::VkFeature::ShaderResourceResidency));
    assert_spirv_valid(m);
}

#[test]
fn fetch_read_write_and_queries() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let storage = |format| Img {
        usage: ImageUsage::Storage,
        format,
        ..SAMPLED_2D
    };
    let kinds = [
        (t.f32_, SAMPLED_2D),
        (
            t.f32_,
            Img {
                ms: true,
                ..SAMPLED_2D
            },
        ),
        (
            t.u32_,
            Img {
                dim: Dim::Buffer,
                ..SAMPLED_2D
            },
        ),
        (t.f32_, storage(ImageFormat::Rgba32f)),
        (
            t.f32_,
            Img {
                arrayed: true,
                ..storage(ImageFormat::Unknown)
            },
        ),
        (t.f32_, storage(ImageFormat::Rg16f)),
        (
            t.u32_,
            Img {
                dim: Dim::Buffer,
                ..storage(ImageFormat::R32ui)
            },
        ),
        (
            t.i32_,
            Img {
                dim: Dim::D1,
                ..storage(ImageFormat::R32i)
            },
        ),
        (
            t.f32_,
            Img {
                ms: true,
                ..storage(ImageFormat::Rgba8)
            },
        ),
    ];
    let g: Vec<GlobalId> = kinds
        .iter()
        .enumerate()
        .map(|(i, (st, k))| {
            let ty = image_ty(&mut b, *st, *k);
            texture(&mut b, i as u32, ty)
        })
        .collect();
    let main = compute(&mut b, |f| {
        let zero = cval(f, t.i32_, 0);
        let c2 = ivec(f, 2, 0);
        let c3 = ivec(f, 3, 0);
        let k2 = ivec_const(f, 2, 1);
        let none = ImageOperands::default();
        let img = load_global(f, g[0]);
        f.fetch(
            img,
            c2,
            &ImageOperands {
                lod: Some(zero),
                ..none
            },
        )
        .expect("fetch lod");
        f.fetch(
            img,
            c2,
            &ImageOperands {
                lod: Some(zero),
                const_offset: Some(k2),
                ..none
            },
        )
        .expect("fetch offset");
        let img = load_global(f, g[1]);
        f.fetch(
            img,
            c2,
            &ImageOperands {
                sample: Some(zero),
                ..none
            },
        )
        .expect("fetch ms");
        f.query(ImageQuery::Size, img).expect("ms size");
        f.query(ImageQuery::Samples, img).expect("samples");
        let img = load_global(f, g[2]);
        f.fetch(img, zero, &none).expect("texel buffer fetch");
        f.query(ImageQuery::Size, img).expect("buffer size");
        let x = cval(f, t.f32_, 0);
        let texel = f.composite_construct(t.vec4, &[x, x, x, x]).expect("texel");
        for (gi, coord) in [(3usize, c2), (4, c3), (5, c2)] {
            let img = load_global(f, g[gi]);
            let v = f
                .read(img, coord, &none, TexelMemory::PRIVATE)
                .expect("read");
            f.write(img, coord, v, &none, TexelMemory::PRIVATE)
                .expect("write");
            f.write(img, coord, texel, &none, TexelMemory::PRIVATE)
                .expect("write texel");
            f.query(ImageQuery::Size, img).expect("storage size");
        }
        let img = load_global(f, g[6]);
        let u = f
            .read(img, zero, &none, TexelMemory::PRIVATE)
            .expect("buffer read");
        f.write(img, zero, u, &none, TexelMemory::PRIVATE)
            .expect("buffer write");
        let img = load_global(f, g[7]);
        let i = f
            .read(img, zero, &none, TexelMemory::PRIVATE)
            .expect("1d read");
        f.write(img, zero, i, &none, TexelMemory::PRIVATE)
            .expect("1d write");
        let img = load_global(f, g[8]);
        let ms = ImageOperands {
            sample: Some(zero),
            ..none
        };
        let v = f.read(img, c2, &ms, TexelMemory::PRIVATE).expect("ms read");
        f.write(img, c2, v, &ms, TexelMemory::PRIVATE)
            .expect("ms write");
    });
    assert_spirv_valid(finish_compute(b, main));
}

#[test]
fn image_atomics() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let st = |format| Img {
        usage: ImageUsage::Storage,
        format,
        ..SAMPLED_2D
    };
    let tu = image_ty(&mut b, t.u32_, st(ImageFormat::R32ui));
    let tf = image_ty(&mut b, t.f32_, st(ImageFormat::R32f));
    let gu = texture(&mut b, 0, tu);
    let gf = texture(&mut b, 1, tf);
    let main = compute(&mut b, |f| {
        let zero = cval(f, t.i32_, 0);
        let c2 = ivec(f, 2, 0);
        let one = cval(f, t.u32_, 1);
        let x = cval(f, t.f32_, 0);
        let r = Semantics::RELAXED;
        let pu = f.module().global_value(gu).expect("image ptr");
        let tp = f.image_texel_pointer(pu, c2, zero).expect("texel pointer");
        f.atomic(AtomicOp::IAdd, tp, Scope::Device, r, one)
            .expect("image atomic");
        f.atomic_compare_exchange(tp, Scope::Device, r, r, one, one)
            .expect("image cas");
        let pf = f.module().global_value(gf).expect("image ptr");
        let tp = f.image_texel_pointer(pf, c2, zero).expect("texel pointer");
        f.atomic(AtomicOp::Exchange, tp, Scope::Device, r, x)
            .expect("float exchange");
    });
    assert_spirv_valid(finish_compute(b, main));
}

/// Non-uniform elements of runtime descriptor arrays (the heap): each access chain and each load of
/// one is decorated `NonUniform`, the module requires `ShaderNonUniform` and the array kind's
/// non-uniform indexing capability and feature (not its dynamic one), and `spirv-val` accepts it. A
/// non-uniform index into anything but a descriptor array of images, samplers, or blocks is refused
/// at construction.
#[test]
fn non_uniform_descriptor_array_elements() {
    use crate::device::requirements::VkFeature;
    use crate::pipeline::sym::{HandleClass, ImageAccess, ImageDim, ImageHandleClass};
    use crate::pipeline::sym::{SampledType, StorageFormatClass};
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let class = |access, format| {
        HandleClass::Image(ImageHandleClass {
            dim: ImageDim::D2,
            arrayed: false,
            multisampled: false,
            sampled_type: SampledType::Float,
            depth_compare: false,
            access,
            format,
        })
    };
    let heap = |b: &mut ModuleBuilder, img: Img, c: HandleClass| {
        let ty = image_ty(b, t.f32_, img);
        let rta = b
            .ty(Type::RuntimeArray {
                elem: ty,
                stride: None,
            })
            .expect("runtime array");
        b.resource(
            slot(ResourceKind::Heap(c), 0),
            StorageClass::UniformConstant,
            rta,
        )
        .expect("heap array")
    };
    let sampled = heap(
        &mut b,
        SAMPLED_2D,
        class(ImageAccess::Sampled, StorageFormatClass::NotStorage),
    );
    let storage = heap(
        &mut b,
        Img {
            usage: ImageUsage::Storage,
            format: ImageFormat::Rgba32f,
            ..SAMPLED_2D
        },
        class(
            ImageAccess::Storage,
            StorageFormatClass::Format(crate::vk::Format::R32G32B32A32_SFLOAT),
        ),
    );
    let single_ty = image_ty(&mut b, t.f32_, SAMPLED_2D);
    let single = texture(&mut b, 0, single_ty);
    let main = compute(&mut b, |f| {
        let h = super::gid_x(f);
        let coord = ivec(f, 2, 0);
        let lod = cval(f, t.i32_, 0);
        let base = f.module().global_value(sampled).expect("global");
        let p = f
            .access_chain_non_uniform(base, h)
            .expect("sampled element");
        let img = f.load(p, MemAccess::default()).expect("image");
        let ops = ImageOperands {
            lod: Some(lod),
            ..ImageOperands::default()
        };
        let x = f.fetch(img, coord, &ops).expect("fetch");
        f.query(ImageQuery::SizeLod(lod), img).expect("query");
        let base = f.module().global_value(storage).expect("global");
        let p = f
            .access_chain_non_uniform(base, h)
            .expect("storage element");
        let img = f.load(p, MemAccess::default()).expect("image");
        f.write(
            img,
            coord,
            x,
            &ImageOperands::default(),
            TexelMemory::PRIVATE,
        )
        .expect("write");
        // Not a descriptor array.
        let one = f.module().global_value(single).expect("global");
        assert!(f.access_chain_non_uniform(one, h).is_err());
    });
    let m = finish_compute(b, main);
    let req = &m.requirements;
    for c in [
        crate::spv::Capability::ShaderNonUniform,
        crate::spv::Capability::RuntimeDescriptorArray,
        crate::spv::Capability::SampledImageArrayNonUniformIndexing,
        crate::spv::Capability::StorageImageArrayNonUniformIndexing,
    ] {
        assert!(
            req.spirv_capabilities.contains(&SpirvCapability::of(c)),
            "{c:?}"
        );
    }
    assert!(!req.spirv_capabilities.contains(&SpirvCapability::of(
        crate::spv::Capability::SampledImageArrayDynamicIndexing
    )));
    for f in [
        VkFeature::RuntimeDescriptorArray,
        VkFeature::ShaderSampledImageArrayNonUniformIndexing,
        VkFeature::ShaderStorageImageArrayNonUniformIndexing,
    ] {
        assert!(req.features.contains(&f), "{f:?}");
    }
    let words = assert_spirv_valid(m);
    // `OpDecorate %x NonUniform` (decoration 5300): two access chains and two loads.
    let mut i = 5;
    let mut decorated = 0;
    while i < words.len() {
        let n = (words[i] >> 16) as usize;
        if words[i] & 0xFFFF == crate::spv::Op::Decorate as u32
            && words.get(i + 2) == Some(&(crate::spv::Decoration::NonUniform as u32))
        {
            decorated += 1;
        }
        i += n.max(1);
    }
    assert_eq!(decorated, 4);
}

/// A non-uniform element reaches the instruction consuming it decorated through every value that
/// carries it (VUID-RuntimeSpirv-None-10148 and VUID-RuntimeSpirv-subgroupSize-10149): the chain,
/// its loaded image, a sampled image of that image and the sampled image's `OpImage`, and an image
/// texel pointer an atomic uses. The builder refuses the uses that would carry one undecorated (a
/// call argument, a composite part).
#[test]
fn non_uniform_elements_are_decorated_through_what_carries_them() {
    use crate::pipeline::sym::{HandleClass, ImageAccess, ImageDim, ImageHandleClass};
    use crate::pipeline::sym::{SampledType, StorageFormatClass};
    use crate::spirv::error::{BuildError, OperandRule};
    use crate::spirv::ir::{ConstBits, Constant};
    let class = |access, sampled_type, format| {
        HandleClass::Image(ImageHandleClass {
            dim: ImageDim::D2,
            arrayed: false,
            multisampled: false,
            sampled_type,
            depth_compare: false,
            access,
            format,
        })
    };
    let heap = |b: &mut ModuleBuilder, ty: TypeId, c: HandleClass| {
        let rta = b
            .ty(Type::RuntimeArray {
                elem: ty,
                stride: None,
            })
            .expect("runtime array");
        b.resource(
            slot(ResourceKind::Heap(c), 0),
            StorageClass::UniformConstant,
            rta,
        )
        .expect("heap array")
    };
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let sampled_ty = image_ty(&mut b, t.f32_, SAMPLED_2D);
    let sampled = heap(
        &mut b,
        sampled_ty,
        class(
            ImageAccess::Sampled,
            SampledType::Float,
            StorageFormatClass::NotStorage,
        ),
    );
    let atomic_ty = image_ty(
        &mut b,
        t.u32_,
        Img {
            usage: ImageUsage::Storage,
            format: ImageFormat::R32ui,
            ..SAMPLED_2D
        },
    );
    let atomics = heap(
        &mut b,
        atomic_ty,
        class(
            ImageAccess::Storage,
            SampledType::Uint,
            StorageFormatClass::Format(crate::vk::Format::R32_UINT),
        ),
    );
    let s = sampler(&mut b, 0);
    let mut carried = Vec::new();
    let main = compute(&mut b, |f| {
        let h = super::gid_x(f);
        let base = f.module().global_value(sampled).expect("global");
        let p = f.access_chain_non_uniform(base, h).expect("element");
        let img = f.load(p, MemAccess::default()).expect("image");
        let smp = load_global(f, s);
        let si = f.sampled_image(img, smp).expect("sampled image");
        let coord = fvec(f, &t, 2);
        let lod = cval(f, t.f32_, 0);
        let ops = ImageOperands {
            lod: Some(lod),
            ..ImageOperands::default()
        };
        f.sample(si, coord, None, false, &ops).expect("sample");
        let im = f.image(si).expect("image of the sampled image");
        let ilod = cval(f, t.i32_, 0);
        f.query(ImageQuery::SizeLod(ilod), im).expect("query");
        let base = f.module().global_value(atomics).expect("global");
        let q = f.access_chain_non_uniform(base, h).expect("element");
        let c2 = ivec(f, 2, 0);
        let zero = cval(f, t.i32_, 0);
        let tp = f.image_texel_pointer(q, c2, zero).expect("texel pointer");
        let one = cval(f, t.u32_, 1);
        f.atomic(AtomicOp::IAdd, tp, Scope::Device, Semantics::RELAXED, one)
            .expect("image atomic");
        carried = vec![p, img, si, im, q, tp];
    });
    let m = finish_compute(b, main);
    let want: std::collections::BTreeSet<Value> = carried.into_iter().collect();
    assert_eq!(m.non_uniform, want);
    let words = assert_spirv_valid(m);
    let mut i = 5;
    let mut decorated = 0;
    while i < words.len() {
        let n = (words[i] >> 16) as usize;
        if words[i] & 0xFFFF == crate::spv::Op::Decorate as u32
            && words.get(i + 2) == Some(&(crate::spv::Decoration::NonUniform as u32))
        {
            decorated += 1;
        }
        i += n.max(1);
    }
    assert_eq!(decorated, 6);

    // The refusals (not validated: nothing here is emitted).
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let sampled_ty = image_ty(&mut b, t.f32_, SAMPLED_2D);
    let sampled = heap(
        &mut b,
        sampled_ty,
        class(
            ImageAccess::Sampled,
            SampledType::Float,
            StorageFormatClass::NotStorage,
        ),
    );
    let two = b
        .constant(Constant::Scalar {
            ty: t.u32_,
            bits: ConstBits(2),
        })
        .expect("length");
    let pair = b
        .ty(Type::Array {
            elem: sampled_ty,
            len: two,
            stride: None,
        })
        .expect("image pair");
    let void = b.ty(Type::Void).expect("void");
    let callee = {
        let mut f = b.function(void).expect("callee");
        f.param(sampled_ty).expect("image parameter");
        f.ret(None).expect("return");
        f.finish().expect("finish")
    };
    compute(&mut b, |f| {
        let h = super::gid_x(f);
        let base = f.module().global_value(sampled).expect("global");
        let p = f.access_chain_non_uniform(base, h).expect("element");
        let img = f.load(p, MemAccess::default()).expect("image");
        let refused = |r: crate::spirv::error::BResult<()>| matches!(r, Err(BuildError::Operand { rule, .. }) if rule == OperandRule::NonUniform(0));
        assert!(refused(f.call(callee, &[img]).map(|_| ())));
        let other = f.load(p, MemAccess::default()).expect("image");
        assert!(refused(
            f.composite_construct(pair, &[img, other]).map(|_| ())
        ));
    });
}

/// A non-uniform element of a storage-buffer array of blocks (the function-table heap): the
/// element's chain and the chain into it are decorated `NonUniform`, and the load, the store, and
/// `OpArrayLength` read through them; a word loaded from it is data, undecorated, which a `select`
/// may take. The module requires `ShaderNonUniform` and storage-buffer non-uniform indexing (not
/// its dynamic one), and `spirv-val` accepts it.
#[test]
fn non_uniform_storage_buffer_elements() {
    use crate::device::requirements::VkFeature;
    use crate::pipeline::sym::HandleClass;
    use crate::spirv::ir::{BinOp, LayoutRule};
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let words = b
        .ty(Type::RuntimeArray {
            elem: t.u32_,
            stride: Some(4),
        })
        .expect("words");
    let layout = crate::spirv::ir::natural_layout(b.types(), LayoutRule::Std430, true, &[words])
        .expect("layout");
    let block = b
        .ty(Type::Struct {
            members: [words].into(),
            layout: Some(layout),
        })
        .expect("block");
    let blocks = b
        .ty(Type::RuntimeArray {
            elem: block,
            stride: None,
        })
        .expect("blocks");
    let buffers = b
        .resource(
            slot(ResourceKind::Heap(HandleClass::FunctionTable), 0),
            StorageClass::StorageBuffer,
            blocks,
        )
        .expect("buffer array");
    let mut carried = Vec::new();
    let main = compute(&mut b, |f| {
        let h = super::gid_x(f);
        let zero = cval(f, t.u32_, 0);
        let base = f.module().global_value(buffers).expect("global");
        let e = f.access_chain_non_uniform(base, h).expect("element");
        let len = f.array_length(e, 0).expect("length");
        let p = f.access_chain(e, &[zero, h]).expect("word");
        let w = f.load(p, MemAccess::default()).expect("load");
        let inside = f.binary(BinOp::ULessThan, h, len).expect("compare");
        let v = f.select(inside, w, zero).expect("data selects");
        f.store(p, v, MemAccess::default()).expect("store");
        carried = vec![e, p];
    });
    let m = finish_compute(b, main);
    let want: std::collections::BTreeSet<Value> = carried.into_iter().collect();
    assert_eq!(m.non_uniform, want);
    let req = &m.requirements;
    for c in [
        crate::spv::Capability::ShaderNonUniform,
        crate::spv::Capability::RuntimeDescriptorArray,
        crate::spv::Capability::StorageBufferArrayNonUniformIndexing,
    ] {
        assert!(
            req.spirv_capabilities.contains(&SpirvCapability::of(c)),
            "{c:?}"
        );
    }
    assert!(!req.spirv_capabilities.contains(&SpirvCapability::of(
        crate::spv::Capability::StorageBufferArrayDynamicIndexing
    )));
    assert!(req
        .features
        .contains(&VkFeature::ShaderStorageBufferArrayNonUniformIndexing));
    let words = assert_spirv_valid(m);
    let mut i = 5;
    let mut decorated = 0;
    while i < words.len() {
        let n = (words[i] >> 16) as usize;
        if words[i] & 0xFFFF == crate::spv::Op::Decorate as u32
            && words.get(i + 2) == Some(&(crate::spv::Decoration::NonUniform as u32))
        {
            decorated += 1;
        }
        i += n.max(1);
    }
    assert_eq!(decorated, 2);
}
