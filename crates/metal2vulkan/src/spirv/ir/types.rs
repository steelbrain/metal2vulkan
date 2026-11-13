//! SPIR-V types and the type interner.
//!
//! [`Type`] is a closed enum. Interning makes equal types the same [`TypeId`], and explicit layout
//! (`Offset`, `ArrayStride`, `MatrixStride`, `Block`) is part of the type, never a free decoration:
//! a laid-out struct and the same members without layout are different types, so a `Block` struct
//! can never be reused in `Function` storage. [`Types::intern`] checks every typing and layout rule
//! before a type exists, so an ill-formed type is unrepresentable in a built module.
//!
//! A type may refer only to types (and constants) that already exist, so child ids are smaller
//! than their parents'. The per-type [`TypeFacts`] are therefore computed once at interning from
//! the children's facts, with no recursion.

use crate::spirv::error::{BResult, BuildError, Dangling, LayoutViolation, TypeRule};
use crate::spirv::ids::{ConstId, TypeId};
use crate::spirv::ir::constant::Consts;
use crate::spirv::ir::intern::FxHashMap;

/// The storage classes the new core emits.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StorageClass {
    UniformConstant,
    Input,
    Uniform,
    Output,
    Workgroup,
    Private,
    Function,
    PushConstant,
    Image,
    StorageBuffer,
    PhysicalStorageBuffer,
    /// An object function's payload, which the mesh functions it launches read
    /// (`SPV_EXT_mesh_shader`): at most one per entry point.
    TaskPayloadWorkgroupEXT,
}

impl StorageClass {
    /// The SPIR-V enumerant.
    pub fn to_spirv(self) -> crate::spv::StorageClass {
        match self {
            StorageClass::UniformConstant => crate::spv::StorageClass::UniformConstant,
            StorageClass::Input => crate::spv::StorageClass::Input,
            StorageClass::Uniform => crate::spv::StorageClass::Uniform,
            StorageClass::Output => crate::spv::StorageClass::Output,
            StorageClass::Workgroup => crate::spv::StorageClass::Workgroup,
            StorageClass::Private => crate::spv::StorageClass::Private,
            StorageClass::Function => crate::spv::StorageClass::Function,
            StorageClass::PushConstant => crate::spv::StorageClass::PushConstant,
            StorageClass::Image => crate::spv::StorageClass::Image,
            StorageClass::StorageBuffer => crate::spv::StorageClass::StorageBuffer,
            StorageClass::PhysicalStorageBuffer => crate::spv::StorageClass::PhysicalStorageBuffer,
            StorageClass::TaskPayloadWorkgroupEXT => {
                crate::spv::StorageClass::TaskPayloadWorkgroupEXT
            }
        }
    }
}

/// The direction of a stage-interface variable.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IoDir {
    Input,
    Output,
}

impl IoDir {
    /// The storage class of a variable in this direction.
    pub fn class(self) -> StorageClass {
        match self {
            IoDir::Input => StorageClass::Input,
            IoDir::Output => StorageClass::Output,
        }
    }
}

/// An explicit-layout rule (Vulkan "Offset and Stride Assignment"), ordered by strictness: every
/// layout valid under a rule is valid under each less strict one.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LayoutRule {
    /// Scalar alignment (`scalarBlockLayout`).
    Scalar,
    /// Base alignment (std430).
    Std430,
    /// Extended alignment (std140).
    Std140,
}

impl LayoutRule {
    const ALL: [LayoutRule; 3] = [LayoutRule::Scalar, LayoutRule::Std430, LayoutRule::Std140];

    fn slot(self) -> usize {
        match self {
            LayoutRule::Scalar => 0,
            LayoutRule::Std430 => 1,
            LayoutRule::Std140 => 2,
        }
    }
}

/// The layout of one struct member. Matrices are always column-major (Metal's order).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemberLayout {
    pub offset: u32,
    /// `MatrixStride`: present exactly when the member is a matrix or a (nested) strided array of
    /// matrices.
    pub matrix_stride: Option<u32>,
}

/// The explicit layout of a struct: its rule, member offsets, and whether it is a `Block`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StructLayout {
    pub rule: LayoutRule,
    pub block: bool,
    pub members: Box<[MemberLayout]>,
}

/// Image dimensionality.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dim {
    D1,
    D2,
    D3,
    Cube,
    Buffer,
    SubpassData,
}

/// Whether an image is sampled (`Sampled = 1`) or a storage image (`Sampled = 2`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ImageUsage {
    Sampled,
    Storage,
}

/// The operands of `OpTypeImage`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageType {
    /// A 32-bit integer or float scalar, or a 64-bit integer scalar.
    pub sampled_type: TypeId,
    pub dim: Dim,
    pub depth: bool,
    pub arrayed: bool,
    pub multisampled: bool,
    pub usage: ImageUsage,
    pub format: crate::spv::ImageFormat,
}

/// A SPIR-V type. Interned: equal types have equal [`TypeId`]s.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Void,
    Bool,
    Int {
        width: u8,
        signed: bool,
    },
    Float {
        width: u8,
    },
    /// An 8-bit float of `encoding` (`OpTypeFloat 8` with its `FPEncoding`, `SPV_EXT_float8`):
    /// `Float8E4M3EXT` (OCP's E4M3, finite: `S.1111.111` is NaN) or `Float8E5M2EXT` (OCP's E5M2,
    /// IEEE-like). Only conversions and bitcasts read or make one.
    Float8 {
        encoding: crate::spv::FPEncoding,
    },
    Vector {
        elem: TypeId,
        len: u8,
    },
    Matrix {
        col: TypeId,
        cols: u8,
    },
    /// A sized array. `stride` is its `ArrayStride`: present for explicit-layout arrays, absent for
    /// implicit-layout arrays and descriptor arrays.
    Array {
        elem: TypeId,
        len: ConstId,
        stride: Option<u32>,
    },
    /// A runtime array: the last member of a `Block` struct (strided) or a descriptor array.
    RuntimeArray {
        elem: TypeId,
        stride: Option<u32>,
    },
    Struct {
        members: Box<[TypeId]>,
        layout: Option<StructLayout>,
    },
    Pointer {
        class: StorageClass,
        pointee: TypeId,
    },
    Function {
        ret: TypeId,
        params: Box<[TypeId]>,
    },
    Image(ImageType),
    Sampler,
    SampledImage {
        image: TypeId,
    },
    /// `OpTypeAccelerationStructureKHR`: an opaque descriptor (`SPV_KHR_ray_query`).
    AccelerationStructure,
    /// `OpTypeRayQueryKHR`: the state of one ray query, held only in a `Function` or `Private`
    /// variable and used only through the ray query instructions.
    RayQuery,
}

/// The structural class of a type, which decides where it may appear.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Shape {
    /// A scalar, vector, matrix, or `PhysicalStorageBuffer` pointer: valid in every data context.
    Plain,
    /// A struct with layout or a strided array.
    Explicit,
    /// A struct without layout or an unstrided array of plain/implicit data.
    Implicit,
    /// An image, sampler, sampled image, or acceleration structure.
    Opaque,
    /// A ray query: the pointee of a `Function` or `Private` variable, never loaded, stored, or
    /// held in an aggregate.
    RayQuery,
    /// An unstrided (runtime) array of opaque values or of `Block` structs.
    DescriptorArray { opaque: bool },
    /// `void`, a function type, or a logical pointer.
    NotData,
}

/// Facts derived once per interned type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypeFacts {
    pub shape: Shape,
    /// Can appear in an explicitly laid-out aggregate (no `bool`, opaque, or logical pointer).
    pub layoutable: bool,
    /// A struct decorated `Block`.
    pub block: bool,
    /// A runtime array, or a struct ending in one.
    pub runtime_sized: bool,
    /// A matrix, or a strided array (of arrays) of matrices: needs a member `MatrixStride`.
    pub matrix: bool,
    /// Alignment under each [`LayoutRule`] (meaningful when `layoutable`).
    align: [u32; 3],
    /// Byte size, when fixed and independent of a matrix stride.
    size: Option<u32>,
    /// The strictest rule this type's own layout satisfies (plain types satisfy all).
    max_rule: LayoutRule,
}

impl TypeFacts {
    /// Alignment of the type under `rule`.
    pub fn align(&self, rule: LayoutRule) -> u32 {
        self.align.get(rule.slot()).copied().unwrap_or(1)
    }

    /// Byte size in an explicit layout, when fixed.
    pub fn size(&self) -> Option<u32> {
        self.size
    }

    /// The strictest layout rule this type's own explicit layout satisfies (its declared rule for
    /// a laid-out struct; every rule for types without an explicit layout).
    pub fn max_rule(&self) -> LayoutRule {
        self.max_rule
    }

    fn other(shape: Shape) -> TypeFacts {
        TypeFacts {
            shape,
            layoutable: false,
            block: false,
            runtime_sized: false,
            matrix: false,
            align: [1; 3],
            size: None,
            max_rule: LayoutRule::Std140,
        }
    }

    fn plain(align: [u32; 3], size: Option<u32>) -> TypeFacts {
        TypeFacts {
            shape: Shape::Plain,
            layoutable: true,
            align,
            size,
            ..TypeFacts::other(Shape::Plain)
        }
    }
}

/// The type interner: insertion-ordered, with a hash index used only for lookup (never iterated).
#[derive(Clone, Debug, Default)]
pub struct Types {
    list: Vec<(Type, TypeFacts)>,
    index: FxHashMap<Type, TypeId>,
    /// The heap bytes of the member and parameter lists inside `list` and the index keys,
    /// counted as they are interned.
    boxed: usize,
}

impl Type {
    /// The heap bytes of the type's boxed member, layout, or parameter lists.
    pub fn heap_bytes(&self) -> usize {
        match self {
            Type::Struct { members, layout } => {
                std::mem::size_of_val::<[TypeId]>(members)
                    + layout
                        .as_ref()
                        .map_or(0, |l| std::mem::size_of_val::<[MemberLayout]>(&l.members))
            }
            Type::Function { params, .. } => std::mem::size_of_val::<[TypeId]>(params),
            _ => 0,
        }
    }
}

impl Types {
    /// The heap bytes of the interner's list and index, with the boxed lists of the types and of
    /// the index keys, for the byte ledger.
    pub fn footprint(&self) -> usize {
        crate::budget::vec_bytes(&self.list) + crate::budget::map_bytes(&self.index) + self.boxed
    }

    /// The type with this id.
    pub fn get(&self, id: TypeId) -> BResult<&Type> {
        self.entry(id).map(|(t, _)| t)
    }

    /// The facts of the type with this id.
    pub fn facts(&self, id: TypeId) -> BResult<&TypeFacts> {
        self.entry(id).map(|(_, f)| f)
    }

    /// Every type, in interning order (a valid declaration order).
    pub fn iter(&self) -> impl Iterator<Item = (TypeId, &Type)> {
        self.list
            .iter()
            .enumerate()
            .map(|(i, (t, _))| (TypeId::new(i as u32), t))
    }

    /// Stop interning type `id`: a later [`Types::intern`] of an equal type makes a new one. Its
    /// entry stays, so its id stays valid; nothing declares it once its module dropped its
    /// declaration.
    pub fn forget(&mut self, id: TypeId) {
        if let Some((t, _)) = self.list.get(id.index()) {
            if self.index.get(t) == Some(&id) {
                if let Some((key, _)) = self.index.remove_entry(t) {
                    self.boxed = self.boxed.saturating_sub(key.heap_bytes());
                }
            }
        }
    }

    /// The number of interned types.
    pub(crate) fn len(&self) -> usize {
        self.list.len()
    }

    /// The id of an already-interned type.
    pub fn lookup(&self, t: &Type) -> Option<TypeId> {
        self.index.get(t).copied()
    }

    fn entry(&self, id: TypeId) -> BResult<&(Type, TypeFacts)> {
        // A match, not `ok_or`, as `TypedModule::value`: no error built at every lookup.
        match self.list.get(id.index()) {
            Some(entry) => Ok(entry),
            None => Err(BuildError::Dangling(Dangling::Type(id))),
        }
    }

    /// Intern `t` after checking every typing and layout rule. Returns `(id, fresh)`, where
    /// `fresh` says the type was not interned before.
    pub fn intern(&mut self, t: Type, consts: &Consts) -> BResult<(TypeId, bool)> {
        if let Some(id) = self.lookup(&t) {
            return Ok((id, false));
        }
        let facts = self.check(&t, consts)?;
        let id =
            TypeId::new(u32::try_from(self.list.len()).map_err(|_| BuildError::IdSpaceExhausted)?);
        // The list and the index each hold the type's lists.
        self.boxed = self.boxed.saturating_add(t.heap_bytes().saturating_mul(2));
        self.index.insert(t.clone(), id);
        self.list.push((t, facts));
        Ok((id, true))
    }

    fn check(&self, t: &Type, consts: &Consts) -> BResult<TypeFacts> {
        let ty_err = |r| Err(BuildError::Type(r));
        match t {
            Type::Void | Type::Function { .. } => {
                if let Type::Function { ret, params } = t {
                    if matches!(self.get(*ret)?, Type::Function { .. }) {
                        return ty_err(TypeRule::FunctionSignature);
                    }
                    for p in params.iter() {
                        if matches!(self.get(*p)?, Type::Void | Type::Function { .. }) {
                            return ty_err(TypeRule::FunctionSignature);
                        }
                    }
                }
                Ok(TypeFacts::other(Shape::NotData))
            }
            Type::Bool => Ok(TypeFacts {
                layoutable: false,
                ..TypeFacts::plain([1; 3], None)
            }),
            Type::Int { width, .. } => match width {
                8 | 16 | 32 | 64 => {
                    let b = u32::from(*width) / 8;
                    Ok(TypeFacts::plain([b; 3], Some(b)))
                }
                _ => ty_err(TypeRule::ScalarWidth),
            },
            Type::Float { width } => match width {
                16 | 32 | 64 => {
                    let b = u32::from(*width) / 8;
                    Ok(TypeFacts::plain([b; 3], Some(b)))
                }
                _ => ty_err(TypeRule::ScalarWidth),
            },
            Type::Float8 { .. } => Ok(TypeFacts::plain([1; 3], Some(1))),
            Type::Vector { elem, len } => {
                let e = self.get(*elem)?;
                let scalar = matches!(
                    e,
                    Type::Bool | Type::Int { .. } | Type::Float { .. } | Type::Float8 { .. }
                );
                if !scalar || !(2..=4).contains(len) {
                    return ty_err(TypeRule::VectorShape);
                }
                let ef = *self.facts(*elem)?;
                let n = u32::from(*len);
                match ef.size {
                    None => Ok(TypeFacts {
                        layoutable: false,
                        ..TypeFacts::plain([1; 3], None)
                    }),
                    Some(c) => {
                        let base = if n == 2 { 2 * c } else { 4 * c };
                        Ok(TypeFacts::plain([c, base, base], Some(n * c)))
                    }
                }
            }
            Type::Matrix { col, cols } => {
                let is_float_vec = match self.get(*col)? {
                    Type::Vector { elem, .. } => matches!(self.get(*elem)?, Type::Float { .. }),
                    _ => false,
                };
                if !is_float_vec || !(2..=4).contains(cols) {
                    return ty_err(TypeRule::MatrixShape);
                }
                let cf = *self.facts(*col)?;
                Ok(TypeFacts {
                    matrix: true,
                    ..TypeFacts::plain(
                        [
                            cf.align(LayoutRule::Scalar),
                            cf.align(LayoutRule::Std430),
                            round16(cf.align(LayoutRule::Std140))?,
                        ],
                        None,
                    )
                })
            }
            Type::Array { elem, len, stride } => {
                let n = consts.array_len(*len, self)?;
                self.check_array(*elem, Some(n), *stride, false)
            }
            Type::RuntimeArray { elem, stride } => self.check_array(*elem, None, *stride, true),
            Type::Struct { members, layout } => self.check_struct(members, layout.as_ref()),
            Type::Pointer { class, pointee } => {
                self.check_pointee(*class, *pointee)?;
                if *class == StorageClass::PhysicalStorageBuffer {
                    Ok(TypeFacts::plain([8; 3], Some(8)))
                } else {
                    Ok(TypeFacts::other(Shape::NotData))
                }
            }
            Type::Image(img) => {
                self.check_image(img)?;
                Ok(TypeFacts::other(Shape::Opaque))
            }
            Type::Sampler | Type::AccelerationStructure => Ok(TypeFacts::other(Shape::Opaque)),
            Type::RayQuery => Ok(TypeFacts::other(Shape::RayQuery)),
            Type::SampledImage { image } => match self.get(*image)? {
                Type::Image(ImageType {
                    usage: ImageUsage::Sampled,
                    dim,
                    ..
                }) if !matches!(dim, Dim::Buffer | Dim::SubpassData) => {
                    Ok(TypeFacts::other(Shape::Opaque))
                }
                _ => ty_err(TypeRule::SampledImageShape),
            },
        }
    }

    /// `len` is `Some(Some(n))` for a sized array with a known length, `Some(None)` for a
    /// spec-constant length, and `None` for a runtime array.
    fn check_array(
        &self,
        elem: TypeId,
        len: Option<Option<u32>>,
        stride: Option<u32>,
        runtime: bool,
    ) -> BResult<TypeFacts> {
        let ef = *self.facts(elem)?;
        // A descriptor array of blocks may hold blocks ending in a runtime array: each element is
        // its own interface block, so only a data array rejects a runtime-sized element.
        let block_descriptors = stride.is_none() && ef.block && ef.shape == Shape::Explicit;
        if (ef.runtime_sized && !block_descriptors)
            || matches!(ef.shape, Shape::NotData | Shape::DescriptorArray { .. })
        {
            return Err(BuildError::Type(TypeRule::Element));
        }
        let Some(stride) = stride else {
            // Implicit layout, or a descriptor array.
            return match ef.shape {
                Shape::Opaque => Ok(desc_array(true)),
                Shape::Explicit if ef.block => Ok(desc_array(false)),
                Shape::Plain | Shape::Implicit if !runtime => Ok(TypeFacts {
                    layoutable: false,
                    ..TypeFacts::other(Shape::Implicit)
                }),
                Shape::Plain | Shape::Implicit => {
                    Err(BuildError::Type(TypeRule::RuntimeArrayPlacement))
                }
                Shape::Explicit => Err(BuildError::Type(TypeRule::MixedLayout)),
                _ => Err(BuildError::Type(TypeRule::Element)),
            };
        };
        // Explicit layout.
        if ef.block {
            return Err(BuildError::Type(TypeRule::NestedBlock));
        }
        if !ef.layoutable {
            return Err(BuildError::Type(match ef.shape {
                Shape::Implicit => TypeRule::MixedLayout,
                _ => TypeRule::NotLayoutable,
            }));
        }
        let violation = |v| BuildError::Layout {
            member: None,
            violation: v,
        };
        if let Some(size) = ef.size {
            if stride < size {
                return Err(violation(LayoutViolation::ArrayStride { stride }));
            }
        }
        let mut align = [1; 3];
        let mut max_rule = None;
        for rule in LayoutRule::ALL {
            let a = if rule == LayoutRule::Std140 {
                round16(ef.align(rule))?
            } else {
                ef.align(rule)
            };
            if let Some(slot) = align.get_mut(rule.slot()) {
                *slot = a;
            }
            if stride.is_multiple_of(a) && rule <= ef.max_rule {
                max_rule = Some(rule);
            }
        }
        let Some(max_rule) = max_rule else {
            return Err(violation(LayoutViolation::ArrayStride { stride }));
        };
        let size = match len {
            None => None,
            Some(None) => return Err(violation(LayoutViolation::UnsizedElement)),
            Some(Some(n)) => Some(
                n.checked_mul(stride)
                    .ok_or(violation(LayoutViolation::Overflow))?,
            ),
        };
        Ok(TypeFacts {
            shape: Shape::Explicit,
            layoutable: true,
            block: false,
            runtime_sized: runtime,
            matrix: ef.matrix,
            align,
            size,
            max_rule,
        })
    }

    fn check_struct(
        &self,
        members: &[TypeId],
        layout: Option<&StructLayout>,
    ) -> BResult<TypeFacts> {
        if members.is_empty() {
            return Err(BuildError::Type(TypeRule::EmptyStruct));
        }
        let Some(layout) = layout else {
            for m in members {
                let f = self.facts(*m)?;
                match f.shape {
                    Shape::Plain | Shape::Implicit => {}
                    Shape::Explicit => return Err(BuildError::Type(TypeRule::MixedLayout)),
                    _ => return Err(BuildError::Type(TypeRule::Element)),
                }
            }
            return Ok(TypeFacts {
                layoutable: false,
                ..TypeFacts::other(Shape::Implicit)
            });
        };
        if layout.members.len() != members.len() {
            return Err(BuildError::Layout {
                member: None,
                violation: LayoutViolation::MemberCount,
            });
        }
        let rule = layout.rule;
        let last = members.len() - 1;
        let mut end = 0u32;
        let mut runtime_sized = false;
        let mut align = [1u32; 3];
        for (i, (m, ml)) in members.iter().zip(layout.members.iter()).enumerate() {
            let at = |v| BuildError::Layout {
                member: Some(i as u32),
                violation: v,
            };
            let f = *self.facts(*m)?;
            if f.block {
                return Err(BuildError::Type(TypeRule::NestedBlock));
            }
            if !f.layoutable {
                return Err(BuildError::Type(match f.shape {
                    Shape::Implicit => TypeRule::MixedLayout,
                    _ => TypeRule::NotLayoutable,
                }));
            }
            if f.runtime_sized {
                let is_rta = matches!(self.get(*m)?, Type::RuntimeArray { .. });
                if i != last || !is_rta || !layout.block {
                    return Err(BuildError::Type(TypeRule::RuntimeArrayPlacement));
                }
                runtime_sized = true;
            }
            if f.max_rule < rule {
                return Err(at(LayoutViolation::NestedRule));
            }
            let a = f.align(rule);
            if !ml.offset.is_multiple_of(a) {
                return Err(at(LayoutViolation::Misaligned {
                    offset: ml.offset,
                    align: a,
                }));
            }
            if ml.offset < end {
                return Err(at(LayoutViolation::Overlap {
                    offset: ml.offset,
                    min: end,
                }));
            }
            let size = if f.matrix {
                let Some(ms) = ml.matrix_stride else {
                    return Err(at(LayoutViolation::MatrixStridePresence));
                };
                self.matrix_member_size(*m, ms, rule).map_err(at)?
            } else if ml.matrix_stride.is_some() {
                return Err(at(LayoutViolation::MatrixStridePresence));
            } else {
                f.size
            };
            for r in LayoutRule::ALL {
                if let Some(slot) = align.get_mut(r.slot()) {
                    *slot = (*slot).max(f.align(r));
                }
            }
            end = match size {
                // Only the last (runtime array) member has no size.
                None => u32::MAX,
                Some(size) => {
                    let e = ml
                        .offset
                        .checked_add(size)
                        .ok_or(at(LayoutViolation::Overflow))?;
                    let aggregate = f.matrix || f.shape == Shape::Explicit;
                    if aggregate {
                        round_up(e, a).ok_or(at(LayoutViolation::Overflow))?
                    } else {
                        e
                    }
                }
            };
        }
        if let Some(slot) = align.get_mut(LayoutRule::Std140.slot()) {
            *slot = round16(*slot)?;
        }
        Ok(TypeFacts {
            shape: Shape::Explicit,
            layoutable: true,
            block: layout.block,
            runtime_sized,
            matrix: false,
            align,
            size: if runtime_sized { None } else { Some(end) },
            max_rule: rule,
        })
    }

    /// Size of a matrix member (or strided array of matrices) with matrix stride `ms`, after
    /// checking the matrix stride and every enclosing array stride under `rule`.
    fn matrix_member_size(
        &self,
        ty: TypeId,
        ms: u32,
        rule: LayoutRule,
    ) -> Result<Option<u32>, LayoutViolation> {
        // Walk the array chain down to the matrix, remembering (len, stride) outermost first.
        let mut chain = Vec::new();
        let mut cur = ty;
        let (col, cols) = loop {
            match self.get(cur).map_err(|_| LayoutViolation::Overflow)? {
                Type::Matrix { col, cols } => break (*col, *cols),
                Type::Array { elem, stride, .. } | Type::RuntimeArray { elem, stride } => {
                    let len = self
                        .facts(cur)
                        .ok()
                        .and_then(|f| f.size)
                        .zip(*stride)
                        .and_then(|(size, s)| size.checked_div(s));
                    chain.push((len, stride.unwrap_or(0)));
                    cur = *elem;
                }
                _ => return Err(LayoutViolation::MatrixStridePresence),
            }
        };
        let cf = *self.facts(col).map_err(|_| LayoutViolation::Overflow)?;
        let col_size = cf.size.ok_or(LayoutViolation::UnsizedElement)?;
        let col_align = if rule == LayoutRule::Std140 {
            round16(cf.align(rule)).map_err(|_| LayoutViolation::Overflow)?
        } else {
            cf.align(rule)
        };
        if ms < col_size || !ms.is_multiple_of(col_align) {
            return Err(LayoutViolation::MatrixStride { stride: ms });
        }
        let mut size = u32::from(cols)
            .checked_mul(ms)
            .ok_or(LayoutViolation::Overflow)?;
        // Innermost array first: each stride must hold the element below it.
        for (len, stride) in chain.iter().rev() {
            if *stride < size {
                return Err(LayoutViolation::ArrayStride { stride: *stride });
            }
            size = match len {
                Some(n) => n.checked_mul(*stride).ok_or(LayoutViolation::Overflow)?,
                None => return Ok(None),
            };
        }
        Ok(Some(size))
    }

    fn check_pointee(&self, class: StorageClass, pointee: TypeId) -> BResult<()> {
        let f = *self.facts(pointee)?;
        let t = self.get(pointee)?;
        let ok = match class {
            StorageClass::StorageBuffer | StorageClass::Uniform => match f.shape {
                Shape::Plain | Shape::Explicit => {
                    f.layoutable && !(class == StorageClass::Uniform && f.runtime_sized)
                }
                Shape::DescriptorArray { opaque } => {
                    // A uniform block never ends in a runtime array, in an array or not.
                    let elem_runtime = match t {
                        Type::Array { elem, .. } | Type::RuntimeArray { elem, .. } => {
                            self.facts(*elem)?.runtime_sized
                        }
                        _ => false,
                    };
                    !opaque && !(class == StorageClass::Uniform && elem_runtime)
                }
                _ => false,
            },
            StorageClass::PushConstant => {
                matches!(f.shape, Shape::Plain | Shape::Explicit)
                    && f.layoutable
                    && !f.runtime_sized
            }
            StorageClass::PhysicalStorageBuffer => {
                matches!(f.shape, Shape::Plain | Shape::Explicit) && f.layoutable
            }
            StorageClass::UniformConstant => matches!(
                f.shape,
                Shape::Opaque | Shape::DescriptorArray { opaque: true }
            ),
            StorageClass::Function | StorageClass::Private => {
                matches!(f.shape, Shape::Plain | Shape::Implicit | Shape::RayQuery)
            }
            StorageClass::Input | StorageClass::Output => {
                matches!(f.shape, Shape::Plain | Shape::Implicit)
            }
            StorageClass::Workgroup => {
                // Explicit layout in `Workgroup` is `SPV_KHR_workgroup_memory_explicit_layout`, which
                // lays out `Block` variables only (`ModuleBuilder::plain_global` requires that);
                // pointers to their laid-out members come from access chains.
                match f.shape {
                    Shape::Plain | Shape::Implicit => true,
                    Shape::Explicit => !f.runtime_sized,
                    _ => false,
                }
            }
            StorageClass::Image => matches!(t, Type::Int { .. } | Type::Float { .. }),
            // An object function's payload is implicitly laid out data.
            StorageClass::TaskPayloadWorkgroupEXT => {
                matches!(f.shape, Shape::Plain | Shape::Implicit) && !f.runtime_sized
            }
        };
        if ok {
            Ok(())
        } else {
            Err(BuildError::Type(TypeRule::PointeeForClass(class)))
        }
    }

    fn check_image(&self, img: &ImageType) -> BResult<()> {
        let sampled_ok = matches!(
            self.get(img.sampled_type)?,
            Type::Int { width: 32 | 64, .. } | Type::Float { width: 32 }
        );
        let dim_ok = match img.dim {
            Dim::Buffer => !img.arrayed && !img.multisampled,
            Dim::SubpassData => {
                img.usage == ImageUsage::Storage
                    && img.format == crate::spv::ImageFormat::Unknown
                    && !img.arrayed
                    && !img.depth
            }
            Dim::D3 => !img.arrayed && !img.multisampled,
            Dim::Cube | Dim::D1 => !img.multisampled,
            Dim::D2 => true,
        };
        let usage_ok = match img.usage {
            ImageUsage::Sampled => img.format == crate::spv::ImageFormat::Unknown,
            ImageUsage::Storage => !img.depth,
        };
        if sampled_ok && dim_ok && usage_ok {
            Ok(())
        } else {
            Err(BuildError::Type(TypeRule::ImageShape))
        }
    }
}

/// A descriptor (runtime) array: the outermost dimension of a descriptor binding, never the tail
/// of a block, so it is not `runtime_sized` in the layout sense.
fn desc_array(opaque: bool) -> TypeFacts {
    TypeFacts::other(Shape::DescriptorArray { opaque })
}

fn round_up(v: u32, a: u32) -> Option<u32> {
    let a = a.max(1);
    v.checked_add(a - 1).map(|x| x / a * a)
}

fn round16(v: u32) -> BResult<u32> {
    round_up(v, 16).ok_or(BuildError::Layout {
        member: None,
        violation: LayoutViolation::Overflow,
    })
}

/// The natural layout of `members` under `rule`: each member at the next offset its alignment
/// allows, matrices with the smallest valid stride. Used by lowering and tests to build
/// [`StructLayout`]s; interning still checks the result.
pub fn natural_layout(
    types: &Types,
    rule: LayoutRule,
    block: bool,
    members: &[TypeId],
) -> BResult<StructLayout> {
    let overflow = |i: usize| BuildError::Layout {
        member: Some(i as u32),
        violation: LayoutViolation::Overflow,
    };
    let mut out = Vec::with_capacity(members.len());
    let mut end = 0u32;
    for (i, m) in members.iter().enumerate() {
        let f = *types.facts(*m)?;
        let a = f.align(rule);
        let offset = round_up(end, a).ok_or(overflow(i))?;
        let (matrix_stride, size) = if f.matrix {
            let ms = matrix_col_stride(types, *m, rule)?;
            let size = types
                .matrix_member_size(*m, ms, rule)
                .map_err(|v| BuildError::Layout {
                    member: Some(i as u32),
                    violation: v,
                })?;
            (Some(ms), size)
        } else {
            (None, f.size)
        };
        out.push(MemberLayout {
            offset,
            matrix_stride,
        });
        end = match size {
            None => u32::MAX,
            Some(s) => {
                let e = offset.checked_add(s).ok_or(overflow(i))?;
                if f.matrix || f.shape == Shape::Explicit {
                    round_up(e, a).ok_or(overflow(i))?
                } else {
                    e
                }
            }
        };
    }
    Ok(StructLayout {
        rule,
        block,
        members: out.into_boxed_slice(),
    })
}

/// The smallest valid matrix stride for the matrix at the bottom of `ty`'s array chain.
fn matrix_col_stride(types: &Types, ty: TypeId, rule: LayoutRule) -> BResult<u32> {
    let mut cur = ty;
    loop {
        match types.get(cur)? {
            Type::Matrix { col, .. } => {
                let cf = *types.facts(*col)?;
                let size = cf.size.unwrap_or(0);
                let a = if rule == LayoutRule::Std140 {
                    round16(cf.align(rule))?
                } else {
                    cf.align(rule)
                };
                return round_up(size, a).ok_or(BuildError::Layout {
                    member: None,
                    violation: LayoutViolation::Overflow,
                });
            }
            Type::Array { elem, .. } | Type::RuntimeArray { elem, .. } => cur = *elem,
            _ => {
                return Err(BuildError::Layout {
                    member: None,
                    violation: LayoutViolation::MatrixStridePresence,
                })
            }
        }
    }
}
