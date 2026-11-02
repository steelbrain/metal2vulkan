//! Typed intrinsic signatures: what each operand position is, checked against the declaration.
//!
//! Every decoded intrinsic derives one [`Signature`] from its variant. Decode checks the
//! declaration's function type against it, reads the [`Operand::Imm`] positions of each call into
//! the variant's fields, and exposes the [`Operand::Handle`] positions as the typed seeds of the
//! handle-provenance lattice. Handles are recognized only by position, never by the name of a
//! pointee struct.

use super::intersect::ResultShape;
use super::ty::NumTy;
use crate::error::IntrinsicError;
use crate::ir::{AddrSpace, Const, ConstTable, Type, TypeId, TypeTable, Value};

/// The most operands any modeled intrinsic takes: an `air.intersect` call with every
/// operand-adding tag (multi-level instancing, motion, a function buffer, and user data).
pub const MAX_OPERANDS: usize = 29;

/// The dimension of a texture handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ImageDim {
    /// 1D: `1d`, `1d_array`.
    D1,
    /// 2D: `2d`, `2d_array`, `2d_ms`, `2d_ms_array`.
    D2,
    /// 3D: `3d`.
    D3,
    /// A cube: `cube`, `cube_array`.
    Cube,
    /// A texture buffer: `buffer_1d`.
    Buffer,
}

/// The shape of a texture handle an intrinsic operates on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageShape {
    /// The dimension.
    pub dim: ImageDim,
    /// An array texture (`_array`).
    pub array: bool,
    /// A multisampled texture (`_ms`).
    pub ms: bool,
    /// A depth texture (`_depth_`) rather than a color one (`_texture_`).
    pub depth: bool,
}

/// The class of a resource handle at a typed operand or result position (a seed of the handle
/// typing).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HandleKind {
    /// A texture: `ptr addrspace(1)` in AIR.
    Texture(ImageShape),
    /// A sampler: `ptr addrspace(2)` in AIR.
    Sampler,
    /// A visible function table: `ptr addrspace(1)`.
    VisibleFunctionTable,
    /// An intersection function table: `ptr addrspace(1)`.
    IntersectionFunctionTable,
    /// An instance acceleration structure: `ptr addrspace(1)`.
    AccelerationStructure,
    /// An indirect command buffer (an encoder's command): `ptr addrspace(1)`.
    Command,
    /// A compute pipeline state (MSL `compute_pipeline_state`, the object an indirect compute
    /// command is set to run): `ptr addrspace(1)`.
    ComputePipelineState,
    /// A render pipeline state (MSL `render_pipeline_state`, the object an indirect render
    /// command is set to draw with): `ptr addrspace(1)`.
    RenderPipelineState,
    /// A depth/stencil state (MSL `depth_stencil_state`, the state a render command's
    /// `set_depth_stencil_state` sets): `ptr addrspace(1)`.
    DepthStencilState,
    /// A simdgroup async-copy event: a thread pointer.
    Event,
    /// An object function's mesh grid (`mesh_grid_properties`): a threadgroup pointer to an
    /// opaque struct.
    MeshGrid,
    /// A mesh function's output (`metal::mesh<..>`): a pointer in its own address space.
    Mesh,
    /// A fragment function's `interpolant` input: `ptr addrspace(1)`.
    Interpolant,
}

impl HandleKind {
    /// The address space AIR gives a handle of this kind.
    pub const fn space(self) -> AddrSpace {
        match self {
            HandleKind::Texture(_)
            | HandleKind::VisibleFunctionTable
            | HandleKind::IntersectionFunctionTable
            | HandleKind::AccelerationStructure
            | HandleKind::Command
            | HandleKind::ComputePipelineState
            | HandleKind::RenderPipelineState
            | HandleKind::DepthStencilState
            | HandleKind::Interpolant => AddrSpace::Device,
            HandleKind::Sampler => AddrSpace::Constant,
            HandleKind::Event => AddrSpace::Thread,
            HandleKind::MeshGrid => AddrSpace::Threadgroup,
            HandleKind::Mesh => AddrSpace::MeshOutput,
        }
    }
}

/// An operand that must be a constant, read into the variant at decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ImmKind {
    /// An `i1` flag.
    Bool,
    /// An `i32` enumerator.
    I32,
}

impl ImmKind {
    fn matches(self, ty: &Type) -> bool {
        match self {
            ImmKind::Bool => matches!(ty, Type::Int(1)),
            ImmKind::I32 => matches!(ty, Type::Int(32)),
        }
    }
}

/// What one operand position (or the result) of an intrinsic is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Operand {
    /// No value (a void result).
    Void,
    /// A scalar or vector value.
    Num(NumTy),
    /// A literal struct `{ T, i8 }`: a texel and the status byte AIR's texture reads return.
    WithStatus(NumTy),
    /// A byte pointer into one address space.
    Ptr(AddrSpace),
    /// A resource handle of a known class.
    Handle(HandleKind),
    /// A constant operand read into the variant.
    Imm(ImmKind),
    /// The literal struct an `air.intersect` call returns (its fields follow from the shape).
    IntersectionResult(ResultShape),
}

impl Operand {
    /// Whether IR type `id` has this operand's shape.
    fn matches(self, types: &TypeTable, id: TypeId) -> bool {
        let Some(ty) = types.get(id) else {
            return false;
        };
        match self {
            Operand::Void => matches!(ty, Type::Void),
            Operand::Num(n) => n.matches(types, id),
            Operand::WithStatus(n) => match ty {
                Type::Struct {
                    fields,
                    packed: false,
                    ..
                } => match fields[..] {
                    [t, status] => {
                        n.matches(types, t) && matches!(types.get(status), Some(Type::Int(8)))
                    }
                    _ => false,
                },
                _ => false,
            },
            Operand::Ptr(space) => matches!(ty, Type::Ptr { space: s } if *s == space),
            Operand::Handle(h) => matches!(ty, Type::Ptr { space: s } if *s == h.space()),
            Operand::Imm(k) => k.matches(ty),
            Operand::IntersectionResult(shape) => match ty {
                Type::Struct {
                    fields,
                    packed: false,
                    ..
                } => {
                    fields.len() == shape.fields().count()
                        && fields
                            .iter()
                            .zip(shape.fields())
                            .all(|(&t, f)| f.operand().matches(types, t))
                }
                _ => false,
            },
        }
    }
}

/// An intrinsic's typed signature: its result and operand positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Signature {
    /// The result.
    pub ret: Operand,
    params: [Operand; MAX_OPERANDS],
    len: u8,
}

impl Signature {
    /// A signature returning `ret` and taking `params` (at most `MAX_OPERANDS`; more is a
    /// table bug, caught by the totality tests).
    pub fn new(ret: Operand, params: &[Operand]) -> Signature {
        let mut out = [Operand::Void; MAX_OPERANDS];
        let len = params.len().min(MAX_OPERANDS);
        for (slot, p) in out.iter_mut().zip(params) {
            *slot = *p;
        }
        Signature {
            ret,
            params: out,
            len: len as u8,
        }
    }

    /// The operand positions.
    pub fn params(&self) -> &[Operand] {
        self.params.get(..usize::from(self.len)).unwrap_or(&[])
    }

    /// Operand `i`.
    pub fn param(&self, i: usize) -> Option<Operand> {
        self.params().get(i).copied()
    }

    /// The handle operand positions (the handle typing's seeds).
    #[cfg(test)]
    pub fn handles(&self) -> impl Iterator<Item = (usize, HandleKind)> + '_ {
        self.params()
            .iter()
            .enumerate()
            .filter_map(|(i, p)| match p {
                Operand::Handle(h) => Some((i, *h)),
                _ => None,
            })
    }

    /// The handle a call returns, when the result is a handle (a seed of the handle typing:
    /// `get_null_*`, `get_read_sampler`).
    pub fn ret_handle(&self) -> Option<HandleKind> {
        match self.ret {
            Operand::Handle(h) => Some(h),
            _ => None,
        }
    }

    /// The immediate operand positions.
    #[cfg(test)]
    pub fn immediates(&self) -> impl Iterator<Item = (usize, ImmKind)> + '_ {
        self.params()
            .iter()
            .enumerate()
            .filter_map(|(i, p)| match p {
                Operand::Imm(k) => Some((i, *k)),
                _ => None,
            })
    }

    /// Whether the non-vararg function type `fn_ty` has exactly this signature.
    pub fn matches(&self, types: &TypeTable, fn_ty: TypeId) -> bool {
        let Some(Type::Function {
            ret,
            params,
            vararg: false,
        }) = types.get(fn_ty)
        else {
            return false;
        };
        self.ret.matches(types, *ret)
            && params.len() == self.params().len()
            && params
                .iter()
                .zip(self.params())
                .all(|(&t, p)| p.matches(types, t))
    }
}

/// Reads the immediate operands of one call.
pub struct Immediates<'a> {
    sig: Signature,
    args: &'a [Value],
    consts: &'a ConstTable,
}

impl<'a> Immediates<'a> {
    /// The immediates of a call with `args` to an intrinsic of signature `sig`.
    pub fn new(sig: Signature, args: &'a [Value], consts: &'a ConstTable) -> Immediates<'a> {
        Immediates { sig, args, consts }
    }

    /// The value of immediate operand `i`: an integer constant (`zeroinitializer` is 0).
    /// Anything else, including `undef` and a position the signature does not mark immediate,
    /// is [`IntrinsicError::ImmediateNotConstant`].
    pub fn get(&self, i: usize) -> Result<u64, IntrinsicError> {
        let not_constant = IntrinsicError::ImmediateNotConstant;
        if !matches!(self.sig.param(i), Some(Operand::Imm(_))) {
            return Err(not_constant);
        }
        let Some(Value::Const(c)) = self.args.get(i) else {
            return Err(not_constant);
        };
        match self.consts.get(*c).map(|c| &c.value) {
            Some(Const::Int(v)) => Ok(*v),
            Some(Const::Null) => Ok(0),
            _ => Err(not_constant),
        }
    }

    /// Immediate operand `i` as an `i1` flag.
    pub fn flag(&self, i: usize) -> Result<bool, IntrinsicError> {
        Ok(self.get(i)? != 0)
    }
}
