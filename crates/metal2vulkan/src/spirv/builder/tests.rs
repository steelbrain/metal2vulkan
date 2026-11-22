//! L0 tests of the P8a data model and builder: construction-time acceptance and rejection.

use crate::error::{Error, InternalKind};
use crate::pipeline::sym::{
    AirUserKey, Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage, SyntheticBinding,
    VaryingKey,
};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::{
    BuildError, ConstRule, ControlRule, EntryRule, GlobalRule, LayoutViolation, OperandRule,
    TypeRule,
};
use crate::spirv::ids::{ConstId, GlobalId, TypeId};
use crate::spirv::ir::{
    natural_layout, ConstBits, Constant, Decl, Dim, ExecMode, ExecModel, ImageType, ImageUsage,
    Inst, IoDecor, IoDir, LayoutRule, MemAccess, MemberLayout, Scope, SpecOp, StorageClass,
    StructLayout, Terminator, Type, TypedModule, ValueOrigin,
};
use std::collections::BTreeSet;

use crate::spv::BuiltIn;

struct T {
    b: ModuleBuilder,
}

impl T {
    fn new() -> T {
        T {
            b: ModuleBuilder::new(),
        }
    }
    fn ty(&mut self, t: Type) -> TypeId {
        self.b.ty(t).expect("valid type")
    }
    fn u32(&mut self) -> TypeId {
        self.ty(Type::Int {
            width: 32,
            signed: false,
        })
    }
    fn f32(&mut self) -> TypeId {
        self.ty(Type::Float { width: 32 })
    }
    fn vec(&mut self, elem: TypeId, len: u8) -> TypeId {
        self.ty(Type::Vector { elem, len })
    }
    fn void(&mut self) -> TypeId {
        self.ty(Type::Void)
    }
    fn uconst(&mut self, v: u64) -> ConstId {
        let ty = self.u32();
        self.b
            .constant(Constant::Scalar {
                ty,
                bits: ConstBits(v),
            })
            .expect("valid constant")
    }
    fn array(&mut self, elem: TypeId, len: u64, stride: Option<u32>) -> TypeId {
        let len = self.uconst(len);
        self.ty(Type::Array { elem, len, stride })
    }
    fn laid(&mut self, rule: LayoutRule, block: bool, members: &[TypeId]) -> TypeId {
        let layout = natural_layout(self.b.types(), rule, block, members).expect("layout");
        self.ty(Type::Struct {
            members: members.into(),
            layout: Some(layout),
        })
    }
    fn offsets(&self, rule: LayoutRule, members: &[TypeId]) -> Vec<u32> {
        natural_layout(self.b.types(), rule, false, members)
            .expect("layout")
            .members
            .iter()
            .map(|m| m.offset)
            .collect()
    }
    fn explicit(
        &mut self,
        rule: LayoutRule,
        members: &[TypeId],
        offsets: &[u32],
    ) -> Result<TypeId, BuildError> {
        self.b.ty(Type::Struct {
            members: members.into(),
            layout: Some(StructLayout {
                rule,
                block: false,
                non_writable: false,
                members: offsets
                    .iter()
                    .map(|o| MemberLayout {
                        offset: *o,
                        matrix_stride: None,
                    })
                    .collect(),
            }),
        })
    }
}

fn slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

fn key(name: &str) -> VaryingKey {
    VaryingKey(AirUserKey(name.as_bytes().into()))
}

/// A finished compute module around an empty `main`.
fn finish_compute(mut b: ModuleBuilder) -> Result<TypedModule, BuildError> {
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

// ---- Types ----

#[test]
fn scalar_widths_are_checked() {
    let mut t = T::new();
    for w in [8, 16, 32, 64] {
        t.b.ty(Type::Int {
            width: w,
            signed: true,
        })
        .expect("int width");
    }
    for w in [16, 32, 64] {
        t.b.ty(Type::Float { width: w }).expect("float width");
    }
    assert_eq!(
        t.b.ty(Type::Int {
            width: 7,
            signed: false
        }),
        Err(BuildError::Type(TypeRule::ScalarWidth))
    );
    assert_eq!(
        t.b.ty(Type::Float { width: 8 }),
        Err(BuildError::Type(TypeRule::ScalarWidth))
    );
}

#[test]
fn interning_dedups_and_layout_is_in_the_key() {
    let mut t = T::new();
    let f = t.f32();
    assert_eq!(t.f32(), f);
    let v4 = t.vec(f, 4);
    assert_eq!(t.vec(f, 4), v4);
    let plain = t.ty(Type::Struct {
        members: [f, v4].into(),
        layout: None,
    });
    let laid = t.laid(LayoutRule::Std430, false, &[f, v4]);
    let block = t.laid(LayoutRule::Std430, true, &[f, v4]);
    assert_ne!(plain, laid);
    assert_ne!(laid, block);
    assert_eq!(t.laid(LayoutRule::Std430, false, &[f, v4]), laid);
    assert_eq!(t.b.types().len(), 5);
}

#[test]
fn vector_and_matrix_shapes_are_checked() {
    let mut t = T::new();
    let f = t.f32();
    let u = t.u32();
    let v4 = t.vec(f, 4);
    assert_eq!(
        t.b.ty(Type::Vector { elem: f, len: 5 }),
        Err(BuildError::Type(TypeRule::VectorShape))
    );
    assert_eq!(
        t.b.ty(Type::Vector { elem: v4, len: 2 }),
        Err(BuildError::Type(TypeRule::VectorShape))
    );
    let uv = t.vec(u, 4);
    assert_eq!(
        t.b.ty(Type::Matrix { col: uv, cols: 4 }),
        Err(BuildError::Type(TypeRule::MatrixShape))
    );
    t.b.ty(Type::Matrix { col: v4, cols: 4 }).expect("mat4");
}

#[test]
fn natural_layouts_follow_each_rule() {
    let mut t = T::new();
    let f = t.f32();
    let v3 = t.vec(f, 3);
    let v2 = t.vec(f, 2);
    assert_eq!(t.offsets(LayoutRule::Scalar, &[f, v3, f]), [0, 4, 16]);
    assert_eq!(t.offsets(LayoutRule::Std430, &[f, v3, f]), [0, 16, 28]);
    assert_eq!(t.offsets(LayoutRule::Std430, &[f, v2]), [0, 8]);
    // std140: arrays and structs round their alignment up to 16.
    let arr = t.array(f, 4, Some(16));
    assert_eq!(t.offsets(LayoutRule::Std140, &[f, arr, f]), [0, 16, 80]);
    // std430 pads after an aggregate to its alignment; scalar does not.
    let inner430 = t.laid(LayoutRule::Std430, false, &[v3]);
    assert_eq!(t.offsets(LayoutRule::Std430, &[inner430, f]), [0, 16]);
    let inner_scalar = t.laid(LayoutRule::Scalar, false, &[v3]);
    assert_eq!(t.offsets(LayoutRule::Scalar, &[inner_scalar, f]), [0, 12]);
}

#[test]
fn explicit_offsets_are_checked_against_the_rule() {
    let mut t = T::new();
    let f = t.f32();
    let v3 = t.vec(f, 3);
    t.explicit(LayoutRule::Scalar, &[f, v3], &[0, 4])
        .expect("scalar");
    assert_eq!(
        t.explicit(LayoutRule::Std430, &[f, v3], &[0, 4]),
        Err(BuildError::Layout {
            member: Some(1),
            violation: LayoutViolation::Misaligned {
                offset: 4,
                align: 16
            }
        })
    );
    assert_eq!(
        t.explicit(LayoutRule::Scalar, &[v3, f], &[0, 8]),
        Err(BuildError::Layout {
            member: Some(1),
            violation: LayoutViolation::Overlap { offset: 8, min: 12 }
        })
    );
    assert_eq!(
        t.explicit(LayoutRule::Scalar, &[f], &[0, 4]),
        Err(BuildError::Layout {
            member: None,
            violation: LayoutViolation::MemberCount
        })
    );
    // A std140 struct cannot nest a std430-only array (stride 4 is not a multiple of 16).
    let arr = t.array(f, 4, Some(4));
    assert_eq!(
        t.explicit(LayoutRule::Std140, &[arr], &[0]),
        Err(BuildError::Layout {
            member: Some(0),
            violation: LayoutViolation::NestedRule
        })
    );
    t.explicit(LayoutRule::Std430, &[arr], &[0])
        .expect("std430 array");
}

#[test]
fn array_strides_and_lengths_are_checked() {
    let mut t = T::new();
    let f = t.f32();
    let v4 = t.vec(f, 4);
    let four = t.uconst(4);
    assert_eq!(
        t.b.ty(Type::Array {
            elem: v4,
            len: four,
            stride: Some(8)
        }),
        Err(BuildError::Layout {
            member: None,
            violation: LayoutViolation::ArrayStride { stride: 8 }
        })
    );
    let zero = t.uconst(0);
    assert_eq!(
        t.b.ty(Type::Array {
            elem: f,
            len: zero,
            stride: None
        }),
        Err(BuildError::Type(TypeRule::ArrayLength))
    );
    let fc =
        t.b.constant(Constant::Scalar {
            ty: f,
            bits: ConstBits(4.0f32.to_bits().into()),
        })
        .expect("float const");
    assert_eq!(
        t.b.ty(Type::Array {
            elem: f,
            len: fc,
            stride: None
        }),
        Err(BuildError::Type(TypeRule::ArrayLength))
    );
    // A spec-constant length is fine for implicit layout (threadgroup memory) but has no size for
    // an explicit one.
    let u = t.u32();
    let tg =
        t.b.spec(SpecKey::TgLength(0), u, ConstBits(16))
            .expect("spec");
    let wg = t.ty(Type::Array {
        elem: f,
        len: tg,
        stride: None,
    });
    t.b.ty(Type::Pointer {
        class: StorageClass::Workgroup,
        pointee: wg,
    })
    .expect("workgroup array");
    assert_eq!(
        t.b.ty(Type::Array {
            elem: f,
            len: tg,
            stride: Some(4)
        }),
        Err(BuildError::Layout {
            member: None,
            violation: LayoutViolation::UnsizedElement
        })
    );
}

#[test]
fn runtime_arrays_only_end_blocks_or_index_descriptors() {
    let mut t = T::new();
    let f = t.f32();
    let u = t.u32();
    let rta = t.ty(Type::RuntimeArray {
        elem: f,
        stride: Some(4),
    });
    t.laid(LayoutRule::Std430, true, &[u, rta]);
    let not_last = natural_layout(t.b.types(), LayoutRule::Std430, true, &[u, f]).expect("layout");
    assert_eq!(
        t.b.ty(Type::Struct {
            members: [rta, u].into(),
            layout: Some(StructLayout {
                members: [
                    MemberLayout {
                        offset: 0,
                        matrix_stride: None
                    },
                    MemberLayout {
                        offset: 4,
                        matrix_stride: None
                    },
                ]
                .into(),
                ..not_last.clone()
            }),
        }),
        Err(BuildError::Type(TypeRule::RuntimeArrayPlacement))
    );
    assert_eq!(
        t.b.ty(Type::Struct {
            members: [u, rta].into(),
            layout: Some(StructLayout {
                block: false,
                non_writable: false,
                ..not_last
            }),
        }),
        Err(BuildError::Type(TypeRule::RuntimeArrayPlacement))
    );
    assert_eq!(
        t.b.ty(Type::RuntimeArray {
            elem: f,
            stride: None
        }),
        Err(BuildError::Type(TypeRule::RuntimeArrayPlacement))
    );
}

#[test]
fn layout_contexts_do_not_mix() {
    let mut t = T::new();
    let f = t.f32();
    let b = t.ty(Type::Bool);
    let implicit = t.ty(Type::Struct {
        members: [f].into(),
        layout: None,
    });
    let laid = t.laid(LayoutRule::Std430, false, &[f]);
    let block = t.laid(LayoutRule::Std430, true, &[f]);
    // bool cannot be laid out.
    assert_eq!(
        natural_layout(t.b.types(), LayoutRule::Std430, false, &[b]).and_then(|l| t.b.ty(
            Type::Struct {
                members: [b].into(),
                layout: Some(l)
            }
        )),
        Err(BuildError::Type(TypeRule::NotLayoutable))
    );
    // Implicit inside explicit, and explicit inside implicit.
    assert_eq!(
        t.explicit(LayoutRule::Std430, &[implicit], &[0]),
        Err(BuildError::Type(TypeRule::MixedLayout))
    );
    assert_eq!(
        t.b.ty(Type::Struct {
            members: [laid].into(),
            layout: None
        }),
        Err(BuildError::Type(TypeRule::MixedLayout))
    );
    // A Block is never nested.
    assert_eq!(
        t.explicit(LayoutRule::Std430, &[block], &[0]),
        Err(BuildError::Type(TypeRule::NestedBlock))
    );
}

#[test]
fn pointer_classes_admit_only_their_layouts() {
    let mut t = T::new();
    let f = t.f32();
    let implicit = t.ty(Type::Struct {
        members: [f].into(),
        layout: None,
    });
    let laid = t.laid(LayoutRule::Std430, false, &[f]);
    let block = t.laid(LayoutRule::Std430, true, &[f]);
    let rta = t.ty(Type::RuntimeArray {
        elem: f,
        stride: Some(4),
    });
    let rblock = t.laid(LayoutRule::Std430, true, &[rta]);
    let ok = |t: &mut T, class, pointee| t.b.ty(Type::Pointer { class, pointee }).is_ok();
    assert!(ok(&mut t, StorageClass::Function, implicit));
    assert!(!ok(&mut t, StorageClass::Function, block));
    assert!(!ok(&mut t, StorageClass::Private, laid));
    assert!(ok(&mut t, StorageClass::StorageBuffer, block));
    assert!(ok(&mut t, StorageClass::StorageBuffer, rblock));
    assert!(!ok(&mut t, StorageClass::StorageBuffer, implicit));
    assert!(!ok(&mut t, StorageClass::Uniform, rblock));
    assert!(!ok(&mut t, StorageClass::PushConstant, rblock));
    assert!(ok(&mut t, StorageClass::PhysicalStorageBuffer, laid));
    // `Workgroup` memory is implicitly laid out: no explicitly laid-out pointee or variable there,
    // `Block` or not.
    assert!(ok(&mut t, StorageClass::Workgroup, implicit));
    assert!(!ok(&mut t, StorageClass::Workgroup, laid));
    assert!(!ok(&mut t, StorageClass::Workgroup, block));
    assert!(!ok(&mut t, StorageClass::Workgroup, rblock));
    for laid_out in [laid, block] {
        assert_eq!(
            t.b.plain_global(StorageClass::Workgroup, laid_out, None),
            Err(BuildError::Type(TypeRule::PointeeForClass(
                StorageClass::Workgroup
            )))
        );
    }
    assert!(!ok(&mut t, StorageClass::UniformConstant, f));
    assert!(ok(&mut t, StorageClass::Image, f));
    assert!(!ok(&mut t, StorageClass::Image, implicit));
}

#[test]
fn matrix_members_need_a_valid_matrix_stride() {
    let mut t = T::new();
    let f = t.f32();
    let v3 = t.vec(f, 3);
    let m3 = t.ty(Type::Matrix { col: v3, cols: 3 });
    let l = natural_layout(t.b.types(), LayoutRule::Std430, false, &[m3, f]).expect("layout");
    assert_eq!(l.members[0].matrix_stride, Some(16));
    assert_eq!(l.members[1].offset, 48);
    t.ty(Type::Struct {
        members: [m3, f].into(),
        layout: Some(l),
    });
    let with = |rule, ms| Type::Struct {
        members: [m3].into(),
        layout: Some(StructLayout {
            rule,
            block: false,
            non_writable: false,
            members: [MemberLayout {
                offset: 0,
                matrix_stride: ms,
            }]
            .into(),
        }),
    };
    assert_eq!(
        t.b.ty(with(LayoutRule::Std430, None)),
        Err(BuildError::Layout {
            member: Some(0),
            violation: LayoutViolation::MatrixStridePresence
        })
    );
    assert_eq!(
        t.b.ty(with(LayoutRule::Std430, Some(12))),
        Err(BuildError::Layout {
            member: Some(0),
            violation: LayoutViolation::MatrixStride { stride: 12 }
        })
    );
    t.b.ty(with(LayoutRule::Scalar, Some(12)))
        .expect("scalar mat3 stride 12");
    // An array of matrices: its stride must hold the whole matrix.
    let arr = t.array(m3, 2, Some(32));
    let too_small = Type::Struct {
        members: [arr].into(),
        layout: Some(StructLayout {
            rule: LayoutRule::Std430,
            block: false,
            non_writable: false,
            members: [MemberLayout {
                offset: 0,
                matrix_stride: Some(16),
            }]
            .into(),
        }),
    };
    assert_eq!(
        t.b.ty(too_small),
        Err(BuildError::Layout {
            member: Some(0),
            violation: LayoutViolation::ArrayStride { stride: 32 }
        })
    );
    let arr = t.array(m3, 2, Some(48));
    t.laid(LayoutRule::Std430, false, &[arr, f]);
}

#[test]
fn image_types_are_checked() {
    let mut t = T::new();
    let f = t.f32();
    let img = |dim, usage, ms, format| {
        Type::Image(ImageType {
            sampled_type: f,
            dim,
            depth: false,
            arrayed: false,
            multisampled: ms,
            usage,
            format,
        })
    };
    let sampled = t.ty(img(
        Dim::D2,
        ImageUsage::Sampled,
        false,
        crate::spv::ImageFormat::Unknown,
    ));
    t.ty(Type::SampledImage { image: sampled });
    let storage = t.ty(img(
        Dim::D2,
        ImageUsage::Storage,
        false,
        crate::spv::ImageFormat::Rgba8,
    ));
    assert_eq!(
        t.b.ty(Type::SampledImage { image: storage }),
        Err(BuildError::Type(TypeRule::SampledImageShape))
    );
    for bad in [
        img(
            Dim::D2,
            ImageUsage::Sampled,
            false,
            crate::spv::ImageFormat::Rgba8,
        ),
        img(
            Dim::D3,
            ImageUsage::Sampled,
            true,
            crate::spv::ImageFormat::Unknown,
        ),
        img(
            Dim::Buffer,
            ImageUsage::Sampled,
            true,
            crate::spv::ImageFormat::Unknown,
        ),
    ] {
        assert_eq!(t.b.ty(bad), Err(BuildError::Type(TypeRule::ImageShape)));
    }
    // Descriptor arrays of opaque values have no stride.
    let arr = t.array(sampled, 8, None);
    t.b.ty(Type::Pointer {
        class: StorageClass::UniformConstant,
        pointee: arr,
    })
    .expect("descriptor array");
}

#[test]
fn a_rejected_type_leaves_the_builder_unchanged() {
    let mut t = T::new();
    let f = t.f32();
    let v3 = t.vec(f, 3);
    let before = (t.b.types().len(), t.b.consts().len());
    assert!(t.explicit(LayoutRule::Std430, &[f, v3], &[0, 4]).is_err());
    assert_eq!((t.b.types().len(), t.b.consts().len()), before);
}

// ---- Constants ----

#[test]
fn constants_are_checked_and_deduped() {
    let mut t = T::new();
    let u = t.u32();
    let b = t.ty(Type::Bool);
    let f = t.f32();
    let v2 = t.vec(u, 2);
    let one = t.uconst(1);
    assert_eq!(t.uconst(1), one);
    assert_eq!(
        t.b.constant(Constant::Scalar {
            ty: u,
            bits: ConstBits(1 << 32)
        }),
        Err(BuildError::Constant(ConstRule::BitsExceedWidth))
    );
    assert_eq!(
        t.b.constant(Constant::Scalar {
            ty: b,
            bits: ConstBits(2)
        }),
        Err(BuildError::Constant(ConstRule::BitsExceedWidth))
    );
    assert_eq!(
        t.b.constant(Constant::Scalar {
            ty: v2,
            bits: ConstBits(0)
        }),
        Err(BuildError::Constant(ConstRule::ScalarType))
    );
    let c =
        t.b.constant(Constant::Composite {
            ty: v2,
            parts: [one, one].into(),
        })
        .expect("uvec2");
    assert_eq!(
        t.b.constant(Constant::Composite {
            ty: v2,
            parts: [one].into()
        }),
        Err(BuildError::Constant(ConstRule::CompositeParts))
    );
    let fone =
        t.b.constant(Constant::Scalar {
            ty: f,
            bits: ConstBits(1.0f32.to_bits().into()),
        })
        .expect("f32");
    assert_eq!(
        t.b.constant(Constant::Composite {
            ty: v2,
            parts: [one, fone].into()
        }),
        Err(BuildError::Constant(ConstRule::CompositeParts))
    );
    let v = t.b.const_value(c).expect("value");
    assert_eq!(t.b.value_ty(v), Ok(v2));
    let sampler = t.ty(Type::Sampler);
    assert_eq!(
        t.b.constant(Constant::Null { ty: sampler }),
        Err(BuildError::Constant(ConstRule::NullType))
    );
    t.b.constant(Constant::Null { ty: v2 }).expect("null uvec2");
}

/// The null of an array whose length an `OpSpecConstantOp` gives, as a threadgroup arena whose
/// regions specialization sizes, is a constant like any other array's null, and initializes a
/// `Workgroup` variable of that type; a runtime array has none.
#[test]
fn a_specialization_sized_array_has_a_null() {
    let mut t = T::new();
    let u = t.u32();
    let len =
        t.b.spec(SpecKey::TgLength(0), u, ConstBits(16))
            .expect("spec");
    let four = t.uconst(4);
    let sized = t.b.spec_op(SpecOp::IAdd, u, &[len, four]).expect("spec op");
    let wg = t.ty(Type::Array {
        elem: u,
        len: sized,
        stride: None,
    });
    let null =
        t.b.constant(Constant::Null { ty: wg })
            .expect("null of a spec-op array");
    t.b.plain_global(StorageClass::Workgroup, wg, Some(null))
        .expect("an initialized workgroup array");
    let runtime = t.ty(Type::RuntimeArray {
        elem: u,
        stride: Some(4),
    });
    assert_eq!(
        t.b.constant(Constant::Null { ty: runtime }),
        Err(BuildError::Constant(ConstRule::NullType))
    );
}

#[test]
fn spec_constants_are_keyed_not_value_deduped() {
    let mut t = T::new();
    let u = t.u32();
    let one = t.uconst(1);
    let s =
        t.b.spec(SpecKey::LocalSize(Axis::X), u, ConstBits(1))
            .expect("spec");
    assert_ne!(s, one);
    assert_eq!(
        t.b.spec(SpecKey::LocalSize(Axis::X), u, ConstBits(1)),
        Ok(s)
    );
    assert_eq!(
        t.b.spec(SpecKey::LocalSize(Axis::X), u, ConstBits(2)),
        Err(BuildError::Constant(ConstRule::SpecKeyConflict(
            SpecKey::LocalSize(Axis::X)
        )))
    );
    let v2 = t.vec(u, 2);
    assert_eq!(
        t.b.spec(SpecKey::SampleCount, v2, ConstBits(1)),
        Err(BuildError::Constant(ConstRule::SpecType))
    );
    assert_eq!(
        t.b.constant(Constant::Composite {
            ty: v2,
            parts: [s, one].into()
        }),
        Err(BuildError::Constant(ConstRule::SpecInComposite))
    );
}

// ---- Globals ----

#[test]
fn resources_need_their_class_shape_and_a_unique_slot() {
    let mut t = T::new();
    let f = t.f32();
    let block = t.laid(LayoutRule::Std430, true, &[f]);
    let laid = t.laid(LayoutRule::Std430, false, &[f]);
    let g =
        t.b.resource(slot(0), StorageClass::StorageBuffer, block)
            .expect("ssbo");
    assert_eq!(
        t.b.resource(slot(0), StorageClass::StorageBuffer, block),
        Err(BuildError::Global(GlobalRule::DuplicateResource(slot(0))))
    );
    assert_eq!(
        t.b.resource(slot(1), StorageClass::StorageBuffer, laid),
        Err(BuildError::Global(GlobalRule::ResourceType))
    );
    assert_eq!(
        t.b.resource(slot(1), StorageClass::Private, block),
        Err(BuildError::Global(GlobalRule::ResourceClass(
            StorageClass::Private
        )))
    );
    let blocks = t.array(block, 4, None);
    t.b.resource(slot(2), StorageClass::StorageBuffer, blocks)
        .expect("descriptor array of blocks");
    let sampler = t.ty(Type::Sampler);
    let samplers = t.ty(Type::RuntimeArray {
        elem: sampler,
        stride: None,
    });
    let heap = ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Synthetic(SyntheticBinding::HandleTable),
        index: 0,
    };
    t.b.resource(heap, StorageClass::UniformConstant, samplers)
        .expect("sampler heap");
    let gv = t.b.global(g).expect("global").clone();
    assert_eq!(t.b.value_ty(gv.value), Ok(gv.ptr_ty));
}

/// Only a storage-image resource, or a descriptor array of them, can be decorated `Volatile`, and a
/// module under the Vulkan memory model, which bans the decoration, cannot carry one.
#[test]
fn only_storage_images_are_volatile_and_never_under_the_vulkan_model() {
    let texture = |index| ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Texture,
        index,
    };
    let image = |t: &mut T, usage, format| {
        let u = t.u32();
        t.ty(Type::Image(ImageType {
            sampled_type: u,
            dim: Dim::D2,
            depth: false,
            arrayed: false,
            multisampled: false,
            usage,
            format,
        }))
    };
    let mut t = T::new();
    let storage = image(&mut t, ImageUsage::Storage, crate::spv::ImageFormat::R32ui);
    let sampled = image(
        &mut t,
        ImageUsage::Sampled,
        crate::spv::ImageFormat::Unknown,
    );
    let views = t.array(storage, 4, None);
    let f = t.f32();
    let block = t.laid(LayoutRule::Std430, true, &[f]);
    let one =
        t.b.resource(texture(0), StorageClass::UniformConstant, storage);
    let array =
        t.b.resource(texture(1), StorageClass::UniformConstant, views);
    let read =
        t.b.resource(texture(2), StorageClass::UniformConstant, sampled);
    let ssbo = t.b.resource(slot(0), StorageClass::StorageBuffer, block);
    let (one, array, read, ssbo) = (
        one.expect("image"),
        array.expect("views"),
        read.expect("sampled"),
        ssbo.expect("ssbo"),
    );
    assert_eq!(t.b.volatile(one), Ok(()));
    assert_eq!(t.b.volatile(array), Ok(()));
    for g in [read, ssbo] {
        assert_eq!(
            t.b.volatile(g),
            Err(BuildError::Global(GlobalRule::VolatileTarget))
        );
    }
    finish_compute(t.b).expect("a GLSL450 module with volatile images");

    let mut t = T::new();
    let storage = image(&mut t, ImageUsage::Storage, crate::spv::ImageFormat::R32ui);
    let g =
        t.b.resource(texture(0), StorageClass::UniformConstant, storage)
            .expect("image");
    t.b.volatile(g).expect("volatile");
    let void = t.void();
    let mut fb = t.b.function(void).expect("function");
    fb.memory_barrier(
        Scope::Device,
        crate::spirv::ir::Semantics {
            order: crate::spirv::ir::MemoryOrder::AcquireRelease,
            image_memory: true,
            make_available: true,
            make_visible: true,
            ..crate::spirv::ir::Semantics::RELAXED
        },
    )
    .expect("a Vulkan-model barrier");
    fb.ret(None).expect("ret");
    let id = fb.finish().expect("finish");
    t.b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .expect("entry");
    assert_eq!(
        t.b.finalize().err(),
        Some(BuildError::Global(GlobalRule::VolatileTarget))
    );
}

/// `coherent_resources` decorates every storage buffer and storage image resource of the module
/// `Coherent` (a descriptor array of storage images included, those declared after it too), and no
/// uniform buffer or sampled image; a module under the Vulkan memory model, which bans the
/// decoration, cannot carry it.
#[test]
fn coherent_resources_are_the_shared_ones_and_never_under_the_vulkan_model() {
    let texture = |index| ResourceSlot {
        stage: Stage::Fragment,
        kind: ResourceKind::Texture,
        index,
    };
    let image = |t: &mut T, usage, format| {
        let u = t.u32();
        t.ty(Type::Image(ImageType {
            sampled_type: u,
            dim: Dim::D2,
            depth: false,
            arrayed: false,
            multisampled: false,
            usage,
            format,
        }))
    };
    let mut t = T::new();
    t.b.coherent_resources();
    let storage = image(&mut t, ImageUsage::Storage, crate::spv::ImageFormat::R32ui);
    let sampled = image(
        &mut t,
        ImageUsage::Sampled,
        crate::spv::ImageFormat::Unknown,
    );
    let views = t.array(storage, 4, None);
    let f = t.f32();
    let block = t.laid(LayoutRule::Std430, true, &[f]);
    let uniform_block = t.laid(LayoutRule::Std140, true, &[f]);
    let one =
        t.b.resource(texture(0), StorageClass::UniformConstant, storage)
            .expect("image");
    let array =
        t.b.resource(texture(1), StorageClass::UniformConstant, views)
            .expect("views");
    let read =
        t.b.resource(texture(2), StorageClass::UniformConstant, sampled)
            .expect("sampled");
    let ssbo =
        t.b.resource(slot(0), StorageClass::StorageBuffer, block)
            .expect("ssbo");
    let ubo =
        t.b.resource(slot(1), StorageClass::Uniform, uniform_block)
            .expect("ubo");
    let m = finish_compute(t.b).expect("a GLSL450 module with coherent resources");
    let want: BTreeSet<GlobalId> = [one, array, ssbo].into_iter().collect();
    assert_eq!(m.coherent, want);
    assert!(!m.coherent.contains(&read) && !m.coherent.contains(&ubo));

    let mut t = T::new();
    t.b.coherent_resources();
    let void = t.void();
    let mut fb = t.b.function(void).expect("function");
    fb.memory_barrier(
        Scope::Device,
        crate::spirv::ir::Semantics {
            order: crate::spirv::ir::MemoryOrder::AcquireRelease,
            uniform_memory: true,
            make_available: true,
            make_visible: true,
            ..crate::spirv::ir::Semantics::RELAXED
        },
    )
    .expect("a Vulkan-model barrier");
    fb.ret(None).expect("ret");
    let id = fb.finish().expect("finish");
    t.b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .expect("entry");
    assert_eq!(
        t.b.finalize().err(),
        Some(BuildError::Global(GlobalRule::CoherentTarget))
    );
}

#[test]
fn push_members_are_placed_in_order_and_sealed() {
    let mut t = T::new();
    let u = t.u32();
    let ulong = t.ty(Type::Int {
        width: 64,
        signed: false,
    });
    let m0 =
        t.b.push_member(PushField::RootTableAddress, ulong, 0)
            .expect("m0");
    let m1 =
        t.b.push_member(PushField::BaseVertexInstance, u, 8)
            .expect("m1");
    assert_eq!((m0.index, m1.index), (0, 1));
    assert_eq!(
        t.b.push_member(PushField::DispatchRegion, u, 8),
        Err(BuildError::Global(GlobalRule::PushOrder { offset: 8 }))
    );
    assert_eq!(
        t.b.push_member(PushField::DispatchRegion, u, 14),
        Err(BuildError::Global(GlobalRule::PushOrder { offset: 14 }))
    );
    let g = t.b.push_global().expect("seal");
    assert_eq!(t.b.push_global(), Ok(g));
    assert_eq!(
        t.b.push_member(PushField::DispatchRegion, u, 16),
        Err(BuildError::Global(GlobalRule::PushSealed))
    );
    let pointee = t.b.global(g).expect("push").pointee;
    match t.b.types().get(pointee) {
        Ok(Type::Struct {
            layout: Some(l), ..
        }) => assert_eq!((l.rule, l.block), (LayoutRule::Std430, true)),
        other => panic!("push block type {other:?}"),
    }
}

#[test]
fn a_push_block_that_is_not_std430_uses_scalar_layout() {
    let mut t = T::new();
    let u = t.u32();
    let f = t.f32();
    let v3 = t.vec(f, 3);
    t.b.push_member(PushField::DispatchRegion, u, 0)
        .expect("u32");
    t.b.push_member(PushField::BaseVertexInstance, v3, 4)
        .expect("vec3 at 4");
    let m = finish_compute(t.b).expect("module");
    let push = m
        .globals()
        .iter()
        .find(|g| g.class == StorageClass::PushConstant)
        .expect("push block sealed by finalize");
    match m.types().get(push.pointee) {
        Ok(Type::Struct {
            layout: Some(l), ..
        }) => assert_eq!(l.rule, LayoutRule::Scalar),
        other => panic!("push block type {other:?}"),
    }
    assert_eq!(m.push_members().len(), 2);
}

/// Only the `Position` output (Metal `[[invariant]]`) and the `ClipDistance` output (a
/// rate-mapped pass's copies) can be decorated `Invariant`.
#[test]
fn only_the_position_and_clip_distances_are_invariant() {
    let mut t = T::new();
    let f = t.f32();
    let v4 = t.vec(f, 4);
    let pos =
        t.b.builtin(BuiltIn::Position, IoDir::Output, v4)
            .expect("position");
    let clips = t.array(f, 4, None);
    let clips =
        t.b.builtin(BuiltIn::ClipDistance, IoDir::Output, clips)
            .expect("clip distances");
    let depth =
        t.b.builtin(BuiltIn::FragDepth, IoDir::Output, f)
            .expect("depth");
    let size =
        t.b.builtin(BuiltIn::PointSize, IoDir::Output, f)
            .expect("point size");
    let color =
        t.b.io(key("color"), IoDir::Output, v4, IoDecor::default())
            .expect("varying");
    assert_eq!(t.b.invariant(pos), Ok(()));
    for g in [clips, depth, size, color] {
        assert_eq!(
            t.b.invariant(g),
            Err(BuildError::Global(GlobalRule::InvariantTarget))
        );
    }
}

#[test]
fn varyings_and_builtins_are_typed() {
    let mut t = T::new();
    let f = t.f32();
    let u = t.u32();
    let b = t.ty(Type::Bool);
    let v4 = t.vec(f, 4);
    let uv3 = t.vec(u, 3);
    t.b.io(key("color"), IoDir::Output, v4, IoDecor::default())
        .expect("varying");
    assert_eq!(
        t.b.io(key("color"), IoDir::Output, v4, IoDecor::default()),
        Err(BuildError::Global(GlobalRule::DuplicateVarying(
            key("color"),
            IoDir::Output
        )))
    );
    t.b.io(key("color"), IoDir::Input, v4, IoDecor::default())
        .expect("same key, other direction");
    assert_eq!(
        t.b.io(key("flag"), IoDir::Output, b, IoDecor::default()),
        Err(BuildError::Global(GlobalRule::VaryingType))
    );
    let dual = IoDecor {
        index: Some(1),
        ..IoDecor::default()
    };
    assert_eq!(
        t.b.io(key("in"), IoDir::Input, v4, dual),
        Err(BuildError::Global(GlobalRule::VaryingType))
    );
    let gid =
        t.b.builtin(BuiltIn::GlobalInvocationId, IoDir::Input, uv3)
            .expect("gid");
    assert_eq!(
        t.b.builtin(BuiltIn::GlobalInvocationId, IoDir::Input, uv3),
        Ok(gid)
    );
    assert_eq!(
        t.b.builtin(BuiltIn::GlobalInvocationId, IoDir::Output, uv3),
        Err(BuildError::Global(GlobalRule::BuiltInDirection(
            BuiltIn::GlobalInvocationId,
            IoDir::Output
        )))
    );
    assert_eq!(
        t.b.builtin(BuiltIn::LocalInvocationId, IoDir::Input, v4),
        Err(BuildError::Global(GlobalRule::BuiltInType(
            BuiltIn::LocalInvocationId
        )))
    );
    assert_eq!(
        t.b.builtin(BuiltIn::BaryCoordKHR, IoDir::Input, v4),
        Err(BuildError::Global(GlobalRule::BuiltInType(
            BuiltIn::BaryCoordKHR
        )))
    );
    assert_eq!(
        t.b.builtin(BuiltIn::WorkDim, IoDir::Input, v4),
        Err(BuildError::Global(GlobalRule::UnsupportedBuiltIn(
            BuiltIn::WorkDim
        )))
    );
    // Tessellation builtins: `TessCoord` is a `vec3` input, the levels `float` arrays.
    assert_eq!(
        t.b.builtin(BuiltIn::TessCoord, IoDir::Input, v4),
        Err(BuildError::Global(GlobalRule::BuiltInType(
            BuiltIn::TessCoord
        )))
    );
    let v3 = t.vec(f, 3);
    t.b.builtin(BuiltIn::TessCoord, IoDir::Input, v3)
        .expect("tess coord");
    let outer = t.array(f, 4, None);
    t.b.builtin(BuiltIn::TessLevelOuter, IoDir::Output, outer)
        .expect("outer levels");
    t.b.builtin(BuiltIn::FrontFacing, IoDir::Input, b)
        .expect("bool builtin");
    let clip = t.array(f, 2, None);
    t.b.builtin(BuiltIn::ClipDistance, IoDir::Output, clip)
        .expect("clip distances");
}

#[test]
fn plain_globals_are_private_or_workgroup() {
    let mut t = T::new();
    let u = t.u32();
    let f = t.f32();
    let one = t.uconst(1);
    t.b.plain_global(StorageClass::Private, u, Some(one))
        .expect("private");
    assert_eq!(
        t.b.plain_global(StorageClass::Private, f, Some(one)),
        Err(BuildError::Global(GlobalRule::InitializerType))
    );
    assert_eq!(
        t.b.plain_global(StorageClass::Function, u, None),
        Err(BuildError::Global(GlobalRule::PlainClass(
            StorageClass::Function
        )))
    );
    let arr = t.array(f, 64, None);
    t.b.plain_global(StorageClass::Workgroup, arr, None)
        .expect("workgroup array");
}

/// Deriving a module's requirements reads each type once. A `Private` variable of a struct DAG 40
/// levels deep (`2^40` leaves through 41 types) finalizes, and its 16-bit leaf still requires
/// `Int16`. Previously the walk for the 8- and 16-bit scalars inside a pointer's pointee followed
/// every path, so this did not finish.
#[test]
fn a_shared_struct_dag_finalizes_reading_each_type_once() {
    let mut t = T::new();
    let mut s = t.ty(Type::Int {
        width: 16,
        signed: false,
    });
    for _ in 0..40 {
        s = t.ty(Type::Struct {
            members: [s, s].into(),
            layout: None,
        });
    }
    t.b.plain_global(StorageClass::Private, s, None)
        .expect("private");
    let m = finish_compute(t.b).expect("finalize");
    assert!(m.requirements.spirv_capabilities.contains(
        &crate::device::requirements::SpirvCapability::of(crate::spv::Capability::Int16)
    ));
}

// ---- Functions ----

#[test]
fn a_function_with_locals_loads_and_stores_finalizes() {
    let mut t = T::new();
    let void = t.void();
    let u = t.u32();
    let seven = t.uconst(7);
    let mut f = t.b.function(void).expect("function");
    let var = f.local_var(u, None).expect("var");
    let ptr = f.var_ptr(var).expect("ptr");
    let c = f.module().const_value(seven).expect("value");
    f.store(ptr, c, MemAccess::default()).expect("store");
    let x = f.load(ptr, MemAccess::default()).expect("load");
    assert_eq!(f.value_ty(x), Ok(u));
    f.ret(None).expect("ret");
    let id = f.finish().expect("finish");
    t.b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 64, y: 1, z: 1 }],
    )
    .expect("entry");
    let m = t.b.finalize().expect("finalize");
    let func = &m.functions()[id.index()];
    assert_eq!(func.vars, vec![var]);
    assert_eq!(func.blocks.len(), 1);
    let block = &m.blocks()[func.blocks[0].index()];
    assert_eq!(block.term, Terminator::Return);
    assert!(matches!(block.insts[0].inst, Inst::Store { .. }));
    assert_eq!(block.insts[1].result, Some(x));
    assert_eq!(
        m.value(x).map(|d| d.origin),
        Ok(ValueOrigin::Inst(func.blocks[0]))
    );
    assert_eq!(m.entry().func, id);
}

#[test]
fn operand_rules_are_enforced() {
    let mut t = T::new();
    let void = t.void();
    let u = t.u32();
    let f32 = t.f32();
    let uv3 = t.vec(u, 3);
    let gid =
        t.b.builtin(BuiltIn::GlobalInvocationId, IoDir::Input, uv3)
            .expect("gid");
    let gid = t.b.global_value(gid).expect("gid value");
    let psb_ptr = t.ty(Type::Pointer {
        class: StorageClass::PhysicalStorageBuffer,
        pointee: u,
    });
    let one = t.uconst(1);
    let mut f = t.b.function(void).expect("function");
    let p = f.param(psb_ptr).expect("param");
    let c = f.module().const_value(one).expect("value");
    assert_eq!(
        f.store(gid, c, MemAccess::default()),
        Err(BuildError::Operand {
            op: crate::spv::Op::Store,
            rule: OperandRule::ReadOnlyClass(StorageClass::Input)
        })
    );
    assert_eq!(
        f.load(c, MemAccess::default()),
        Err(BuildError::Operand {
            op: crate::spv::Op::Load,
            rule: OperandRule::NotPointer
        })
    );
    assert_eq!(
        f.load(p, MemAccess::default()),
        Err(BuildError::Operand {
            op: crate::spv::Op::Load,
            rule: OperandRule::Alignment
        })
    );
    let aligned = |a| MemAccess {
        aligned: Some(a),
        ..MemAccess::default()
    };
    assert_eq!(
        f.load(p, aligned(3)),
        Err(BuildError::Operand {
            op: crate::spv::Op::Load,
            rule: OperandRule::Alignment
        })
    );
    f.load(p, aligned(4)).expect("aligned psb load");
    // Vulkan-memory-model operands: `MakePointerVisible` is a load's and needs `NonPrivatePointer`;
    // `NonPrivatePointer` names only shared memory.
    let device = f.scope_id(Scope::Device).expect("scope");
    let visible = MemAccess {
        make_visible: Some(device),
        non_private: true,
        ..aligned(4)
    };
    f.load(p, visible).expect("coherent psb load");
    assert_eq!(
        f.load(
            p,
            MemAccess {
                non_private: false,
                ..visible
            }
        ),
        Err(BuildError::Operand {
            op: crate::spv::Op::Load,
            rule: OperandRule::Semantics
        })
    );
    assert_eq!(
        f.load(
            p,
            MemAccess {
                make_available: Some(device),
                ..aligned(4)
            }
        ),
        Err(BuildError::Operand {
            op: crate::spv::Op::Load,
            rule: OperandRule::Semantics
        })
    );
    let var = f.local_var(f32, None).expect("var");
    let ptr = f.var_ptr(var).expect("ptr");
    assert_eq!(
        f.load(
            ptr,
            MemAccess {
                non_private: true,
                ..MemAccess::default()
            }
        ),
        Err(BuildError::Operand {
            op: crate::spv::Op::Load,
            rule: OperandRule::Class(StorageClass::Function)
        })
    );
    assert_eq!(
        f.store(ptr, c, MemAccess::default()),
        Err(BuildError::Operand {
            op: crate::spv::Op::Store,
            rule: OperandRule::StoredType
        })
    );
    assert_eq!(
        f.local_var(f32, Some(one)),
        Err(BuildError::Control(ControlRule::LocalType))
    );
    let block = f
        .module()
        .ty(Type::Struct {
            members: [u].into(),
            layout: Some(StructLayout {
                rule: LayoutRule::Std430,
                block: true,
                non_writable: false,
                members: [MemberLayout {
                    offset: 0,
                    matrix_stride: None,
                }]
                .into(),
            }),
        })
        .expect("block");
    assert_eq!(
        f.local_var(block, None),
        Err(BuildError::Control(ControlRule::LocalType))
    );
}

#[test]
fn terminators_and_function_structure_are_enforced() {
    let mut t = T::new();
    let void = t.void();
    let u = t.u32();
    let one = t.uconst(1);
    let mut f = t.b.function(u).expect("function");
    assert_eq!(
        f.ret(None),
        Err(BuildError::Control(ControlRule::ReturnType))
    );
    let c = f.module().const_value(one).expect("value");
    f.ret(Some(c)).expect("ret value");
    assert_eq!(
        f.unreachable(),
        Err(BuildError::Control(ControlRule::NoCurrentBlock))
    );
    let helper = f.finish().expect("finish");

    let g = t.b.function(void).expect("function");
    let open = g.current_block().expect("entry block open");
    assert_eq!(
        g.finish(),
        Err(BuildError::Control(ControlRule::UnterminatedBlock(open)))
    );
    // `g` was never finished; the entry cannot use it and finalize refuses the module.
    let unfinished = crate::spirv::ids::FuncId::new(1);
    assert_eq!(
        t.b.entry(ExecModel::GLCompute, unfinished, vec![]),
        Err(BuildError::Control(ControlRule::UnfinishedFunction(
            unfinished
        )))
    );
    assert_eq!(
        t.b.entry(
            ExecModel::GLCompute,
            helper,
            vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }]
        ),
        Err(BuildError::Entry(EntryRule::Signature))
    );
    let mut h = t.b.function(void).expect("function");
    h.ret(None).expect("ret");
    let main = h.finish().expect("finish");
    t.b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .expect("entry");
    assert_eq!(
        t.b.finalize().map(|_| ()),
        Err(BuildError::Control(ControlRule::UnfinishedFunction(
            unfinished
        )))
    );
}

#[test]
fn entry_modes_are_checked_per_model() {
    let entry = |model, modes: Vec<ExecMode>| -> Result<Vec<ExecMode>, BuildError> {
        let mut b = ModuleBuilder::new();
        let void = b.ty(Type::Void)?;
        let mut f = b.function(void)?;
        f.ret(None)?;
        let id = f.finish()?;
        b.entry(model, id, modes)?;
        Ok(b.finalize()?.entry().modes.clone())
    };
    let ls = ExecMode::LocalSize { x: 8, y: 8, z: 1 };
    assert_eq!(
        entry(ExecModel::Fragment, vec![ExecMode::EarlyFragmentTests]),
        Ok(vec![
            ExecMode::OriginUpperLeft,
            ExecMode::EarlyFragmentTests
        ])
    );
    assert_eq!(
        entry(ExecModel::GLCompute, vec![]),
        Err(BuildError::Entry(EntryRule::MissingLocalSize))
    );
    assert_eq!(
        entry(
            ExecModel::GLCompute,
            vec![ExecMode::LocalSize { x: 0, y: 1, z: 1 }]
        ),
        Err(BuildError::Entry(EntryRule::LocalSize))
    );
    assert_eq!(
        entry(ExecModel::Fragment, vec![ls]),
        Err(BuildError::Entry(EntryRule::ModeForModel))
    );
    assert_eq!(
        entry(
            ExecModel::Fragment,
            vec![ExecMode::DepthGreater, ExecMode::DepthLess]
        ),
        Err(BuildError::Entry(EntryRule::ConflictingModes))
    );
    assert_eq!(
        entry(
            ExecModel::GLCompute,
            vec![
                ls,
                ExecMode::SignedZeroInfNanPreserve { width: 32 },
                ExecMode::SignedZeroInfNanPreserve { width: 32 }
            ]
        ),
        Err(BuildError::Entry(EntryRule::ConflictingModes))
    );
    entry(
        ExecModel::GLCompute,
        vec![
            ls,
            ExecMode::SignedZeroInfNanPreserve { width: 16 },
            ExecMode::SignedZeroInfNanPreserve { width: 32 },
        ],
    )
    .expect("per-width float controls");
    assert_eq!(
        entry(
            ExecModel::Vertex,
            vec![ExecMode::SignedZeroInfNanPreserve { width: 8 }]
        ),
        Err(BuildError::Entry(EntryRule::FloatWidth))
    );
}

#[test]
fn finalize_requires_an_entry_point() {
    let mut b = ModuleBuilder::new();
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    f.ret(None).expect("ret");
    f.finish().expect("finish");
    assert_eq!(
        b.finalize().map(|_| ()),
        Err(BuildError::Entry(EntryRule::Missing))
    );
}

/// Every declaration refers only to earlier declarations, so `decls` is a valid SPIR-V
/// declaration order, and every type, constant, and global is declared exactly once.
#[test]
fn declaration_order_is_topological() {
    let mut t = T::new();
    let f = t.f32();
    let u = t.u32();
    let v3 = t.vec(f, 3);
    let m3 = t.ty(Type::Matrix { col: v3, cols: 3 });
    let arr = t.array(m3, 2, Some(48));
    let block = t.laid(LayoutRule::Std430, true, &[u, arr]);
    t.b.resource(slot(0), StorageClass::StorageBuffer, block)
        .expect("ssbo");
    let tg =
        t.b.spec(SpecKey::TgLength(0), u, ConstBits(4))
            .expect("spec");
    let wg = t.ty(Type::Array {
        elem: f,
        len: tg,
        stride: None,
    });
    t.b.plain_global(StorageClass::Workgroup, wg, None)
        .expect("workgroup");
    let m = finish_compute(t.b).expect("module");

    let mut seen_t = vec![false; m.types().len()];
    let mut seen_c = vec![false; m.consts().len()];
    let mut seen_g = vec![false; m.globals().len()];
    for d in m.decls() {
        match *d {
            Decl::Type(id) => {
                let (tys, cs): (Vec<TypeId>, Vec<ConstId>) = match m.types().get(id).expect("type")
                {
                    Type::Vector { elem, .. } | Type::RuntimeArray { elem, .. } => {
                        (vec![*elem], vec![])
                    }
                    Type::Matrix { col, .. } => (vec![*col], vec![]),
                    Type::Array { elem, len, .. } => (vec![*elem], vec![*len]),
                    Type::Struct { members, .. } => (members.to_vec(), vec![]),
                    Type::Pointer { pointee, .. } => (vec![*pointee], vec![]),
                    Type::Function { ret, params } => {
                        let mut v = vec![*ret];
                        v.extend(params.iter().copied());
                        (v, vec![])
                    }
                    _ => (vec![], vec![]),
                };
                assert!(
                    tys.iter().all(|c| seen_t[c.index()]),
                    "{id:?} before children"
                );
                assert!(
                    cs.iter().all(|c| seen_c[c.index()]),
                    "{id:?} before its length"
                );
                assert!(!seen_t[id.index()], "{id:?} declared twice");
                seen_t[id.index()] = true;
            }
            Decl::Const(id) => {
                assert!(seen_t[m.consts().get(id).expect("const").ty.index()]);
                assert!(!seen_c[id.index()], "{id:?} declared twice");
                seen_c[id.index()] = true;
            }
            Decl::Global(id) => {
                assert!(seen_t[m.globals()[id.index()].ptr_ty.index()]);
                assert!(!seen_g[id.index()], "{id:?} declared twice");
                seen_g[id.index()] = true;
            }
        }
    }
    assert!(seen_t.iter().chain(&seen_c).chain(&seen_g).all(|s| *s));
}

#[test]
fn build_errors_are_internal_errors() {
    let e: Error = BuildError::Entry(EntryRule::Missing).into();
    match e {
        Error::Internal(i) => assert_eq!(i.kind, InternalKind::Build),
        other => panic!("{other:?}"),
    }
}

#[test]
fn typed_modules_are_send_and_sync() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<TypedModule>();
    send_sync::<ModuleBuilder>();
    send_sync::<BuildError>();
}

/// The builder's footprint counts the heap behind its entries: an instruction's boxed operand list,
/// and a struct type's or composite constant's list in both the interner's list and its index key.
#[test]
fn the_builder_footprint_counts_boxed_operands_and_interned_keys() {
    const N: usize = 256;
    let mut mb = ModuleBuilder::new();
    let void = mb.ty(Type::Void).unwrap();
    let u = mb
        .ty(Type::Int {
            width: 32,
            signed: false,
        })
        .unwrap();
    let before = mb.footprint();
    let record = mb
        .ty(Type::Struct {
            members: vec![u; N].into(),
            layout: None,
        })
        .unwrap();
    let lists = N * std::mem::size_of::<TypeId>();
    assert!(
        mb.footprint() - before >= 2 * lists,
        "a struct type's members are counted in the list and the index"
    );
    let one = mb
        .constant(Constant::Scalar {
            ty: u,
            bits: ConstBits(1),
        })
        .unwrap();
    let before = mb.footprint();
    let parts = vec![one; N];
    mb.constant(Constant::Composite {
        ty: record,
        parts: parts.into(),
    })
    .unwrap();
    assert!(
        mb.footprint() - before >= 2 * N * std::mem::size_of::<ConstId>(),
        "a composite constant's parts are counted in the list and the index"
    );
    let mut fb = mb.function(void).unwrap();
    let x = fb.module().const_value(one).unwrap();
    let before = fb.module().footprint();
    fb.composite_construct(record, &vec![x; N]).unwrap();
    let grown = fb.module().footprint() - before;
    assert!(
        grown >= N * std::mem::size_of::<crate::spirv::ids::Value>(),
        "an instruction's operands are counted ({grown} bytes)"
    );
}
