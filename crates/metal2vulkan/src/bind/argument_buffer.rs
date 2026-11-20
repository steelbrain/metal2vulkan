//! Argument-buffer layouts: what Metal's argument encoder (`MTLArgumentEncoder`) writes where.
//!
//! An argument buffer holds, at fixed byte offsets, the values a translated function reads to
//! reach resources: a buffer's Metal address (`gpuAddress` plus the bound offset), and a
//! texture's, sampler's, pipeline state's, indirect command buffer's or function table's resource
//! ID (`gpuResourceID`), each 8 bytes, little endian; and plain constants. A program may write
//! those bytes itself, or through an argument encoder. Both write the same bytes, which the
//! translated function resolves through the resident and handle tables. An [`ArgumentBufferLayout`]
//! says where each argument goes, by its Metal argument index (`[[id(n)]]`), so an encoder over it
//! writes exactly the bytes the function reads.
//!
//! A layout comes from one of two sources, each Metal's own rule:
//!
//! - **A function's argument buffer** (`newArgumentEncoderWithBufferIndex:`,
//!   [`Function::argument_buffer_layout`](crate::library::Function::argument_buffer_layout)): the
//!   AIR `air.struct_type_info` of the argument buffer the function takes at the buffer index,
//!   the same layout the function's own loads address. `encodedLength` and `alignment` are the
//!   argument's `air.arg_type_size` and `air.arg_type_align_size`, and each member sits at its
//!   stated offset. A member's argument index is its `air.location_index` plus the first ids of
//!   the structs that hold it (`MemberKind::IndirectStruct`). An array takes one index per
//!   element; element `k` of an array of structs takes the struct's index span times `k` more,
//!   the span being one past the struct's highest index (an M3 Max and an M4 agree, probes). A
//!   pointer to a struct that holds arguments is a nested argument buffer
//!   (`newArgumentEncoderForBufferAtIndex:`), with the pointee's layout; a pointer to any other
//!   struct is a plain buffer, for which Metal answers no nested encoder.
//! - **Argument descriptors** (`newArgumentEncoderWithArguments:`,
//!   [`ArgumentBufferLayout::from_descriptors`]): the descriptors in the order given (not by
//!   index), each at the next offset its type's alignment allows, as a C struct of MSL types. A
//!   run of consecutive constants is a constant block: its first constant may set
//!   `constantBlockAlignment`, which then aligns it as `(offset + a - 1) & !(a - 1)` for `a` the
//!   larger of the alignment set and its type's. The encoder's `alignment` is the largest type
//!   alignment (never the block alignment), and `encodedLength` the end rounded up to it. Every
//!   rule here is measured on an M3 Max and an M4 (probes).
//!
//! An index that two descriptors claim (the same index, or one inside another's array) is
//! resolved to the later descriptor, as Metal's encoder resolves it.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::air::abi::{AbiInt, ArgTypeInfo, MemberKind, ParamBinding, ParamRole, StructInfo};
use crate::air::abi::{Slot, StructRef};
use crate::air::error::{AbiError, InputError, InputErrorKind, Location};
use crate::budget::Meter;
use crate::error::{Error, Limit};
use crate::metal::{BindingAccess, DataType, TextureType};

/// Bytes of every argument that is no constant: an address or a resource ID.
pub const ARGUMENT_HANDLE_BYTES: u64 = 8;

/// What an argument of an argument buffer holds, and which encoder call writes it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArgumentType {
    /// A buffer's Metal address plus its offset (`setBuffer:offset:atIndex:`).
    Buffer,
    /// A texture's resource ID (`setTexture:atIndex:`).
    Texture,
    /// A sampler state's resource ID (`setSamplerState:atIndex:`).
    Sampler,
    /// Plain bytes the program writes itself (`constantDataAtIndex:`).
    Constant,
    /// An indirect command buffer's resource ID (`setIndirectCommandBuffer:atIndex:`).
    IndirectCommandBuffer,
    /// A compute pipeline state's resource ID (`setComputePipelineState:atIndex:`).
    ComputePipelineState,
    /// A render pipeline state's resource ID (`setRenderPipelineState:atIndex:`).
    RenderPipelineState,
    /// A visible-function table's resource ID (`setVisibleFunctionTable:atIndex:`).
    VisibleFunctionTable,
    /// An intersection function table's resource ID (`setIntersectionFunctionTable:atIndex:`).
    IntersectionFunctionTable,
    /// An acceleration structure's resource ID (`setAccelerationStructure:atIndex:`), instance
    /// or primitive.
    AccelerationStructure,
    /// A depth-stencil state's resource ID: a descriptor's `MTLDataTypeDepthStencilState`, which
    /// Metal lays out as 8 bytes and no public encoder call writes.
    DepthStencilState,
    /// A struct held by value whose members are arguments ([`ArgumentMember::layout`] lists one
    /// element's): it takes no index of its own.
    Struct,
}

/// One member of an [`ArgumentBufferLayout`]: an argument, or an array of them.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArgumentMember {
    /// The argument index (`[[id(n)]]`) of element 0.
    pub id: u32,
    /// Elements: 1 for a single argument.
    pub count: u32,
    /// Argument indices each element takes: 1, or a struct's span ([`ArgumentType::Struct`]).
    pub ids_per_element: u32,
    /// Byte offset of element 0 in the argument buffer.
    pub offset: u64,
    /// The distance from one element to the next: the element's size, except in a layout made
    /// from a pipeline's reflected binding, whose arrays of textures, samplers and pointers, and
    /// of structs of them only, Metal reflects (and encodes) at stride 0.
    pub stride: u64,
    /// Bytes of one element.
    pub size: u64,
    pub ty: ArgumentType,
    /// For a [`ArgumentType::Buffer`] that points to an argument buffer, that buffer's layout
    /// (`newArgumentEncoderForBufferAtIndex:`); for a [`ArgumentType::Struct`], one element's
    /// members, at indices and offsets from the element's (its `encoded_length` the element's
    /// size, its `alignment` 1: Metal makes no encoder for it). `None` otherwise.
    pub layout: Option<Arc<ArgumentBufferLayout>>,
}

/// Where one argument index lands ([`ArgumentBufferLayout::argument`]).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArgumentPlace {
    pub ty: ArgumentType,
    /// Byte offset in the argument buffer.
    pub offset: u64,
    /// Bytes the argument holds.
    pub size: u64,
    /// The layout of the argument buffer a [`ArgumentType::Buffer`] points to, when its pointee
    /// holds arguments.
    pub nested: Option<Arc<ArgumentBufferLayout>>,
}

/// The layout of one argument buffer: its length, alignment and members (the module docs).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArgumentBufferLayout {
    /// `encodedLength`: the bytes one argument buffer of this layout takes, and the distance
    /// between elements of an array of them (`setArgumentBuffer:startOffset:arrayElement:`).
    pub encoded_length: u64,
    /// `alignment`.
    pub alignment: u64,
    /// The members, in layout order.
    pub members: Vec<ArgumentMember>,
}

/// One argument of `newArgumentEncoderWithArguments:` (`MTLArgumentDescriptor`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArgumentDescriptor {
    /// `dataType`.
    pub data_type: DataType,
    /// `index`: the argument index of the argument, or of an array's element 0.
    pub index: u32,
    /// `arrayLength`: 0 (or 1) for a single argument.
    pub array_length: u32,
    /// `access`. It does not change the layout.
    pub access: BindingAccess,
    /// `textureType`. It does not change the layout.
    pub texture_type: TextureType,
    /// `constantBlockAlignment`: 0 for the type's own alignment. Only a constant block's first
    /// constant may set it.
    pub constant_block_alignment: u32,
}

impl ArgumentDescriptor {
    /// An argument of `data_type` at `index`, single, read-only, of a 2D texture type, with no
    /// constant block alignment (Metal's defaults).
    pub fn new(data_type: DataType, index: u32) -> ArgumentDescriptor {
        ArgumentDescriptor {
            data_type,
            index,
            array_length: 0,
            access: BindingAccess::ReadOnly,
            texture_type: TextureType::Type2D,
            constant_block_alignment: 0,
        }
    }
}

/// Why no argument-buffer layout is made: what Metal stops, raises or answers nil for.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArgumentLayoutError {
    /// The function takes no argument buffer at the buffer index: a plain buffer, or nothing.
    /// Metal's API validation stops it ("Buffer argument out (buffer index: 2) of function ... is
    /// not an indirect argument buffer", an M4 under macOS 27.0.1); without
    /// validation Metal answers an encoder of no bytes for a plain buffer.
    NoArgumentBuffer { index: u32 },
    /// The argument buffer holds an array of arrays, the argument at `id`: Metal's encoder
    /// creation raises (`elementIsArgumentBuffer` unrecognized, a probe).
    ArrayOfArrays { id: u32 },
    /// `newArgumentEncoderWithArguments:` of no descriptors ("count must be greater than zero").
    NoDescriptors,
    /// Descriptor `position` has a data type no argument has: `None`, `Struct`, `Array`, a pixel
    /// format type, or `Tensor` (Metal raises "Invalid argument type" or "not a valid
    /// MTLDataType").
    DataType {
        position: usize,
        data_type: DataType,
    },
    /// Descriptor `position` sets `constantBlockAlignment` on a constant that is not its block's
    /// first ("Constant block alignment can only be set the first constant in a constant
    /// block").
    ConstantBlockAlignment { position: usize },
    /// The layout's length does not fit 64 bits.
    TooLarge,
    /// The function's AIR did not decode or specialize (the compile layer's error).
    Compile(Error),
}

impl From<Error> for ArgumentLayoutError {
    fn from(error: Error) -> ArgumentLayoutError {
        ArgumentLayoutError::Compile(error)
    }
}

/// `(offset + a - 1) & !(a - 1)`: Metal's alignment step, which equals rounding up to a multiple
/// of `a` when `a` is a power of two (a probe measured other values too).
fn metal_align(offset: u64, a: u64) -> Option<u64> {
    let a = a.max(1);
    Some(offset.checked_add(a - 1)? & !(a - 1))
}

/// The bytes and alignment of one element of `ty` in an argument buffer: 8 for every handle,
/// and for a constant its MSL size and alignment (a 3-vector as a 4-vector; a matrix its columns'
/// vectors). `None` for a type no argument has.
fn type_size_align(ty: DataType) -> Option<(u64, u64, ArgumentType)> {
    use DataType as D;
    let handle = |t| Some((ARGUMENT_HANDLE_BYTES, ARGUMENT_HANDLE_BYTES, t));
    let (scalar, columns, rows) = match ty {
        D::Pointer => return handle(ArgumentType::Buffer),
        D::Texture => return handle(ArgumentType::Texture),
        D::Sampler => return handle(ArgumentType::Sampler),
        D::IndirectCommandBuffer => return handle(ArgumentType::IndirectCommandBuffer),
        D::ComputePipeline => return handle(ArgumentType::ComputePipelineState),
        D::RenderPipeline => return handle(ArgumentType::RenderPipelineState),
        D::VisibleFunctionTable => return handle(ArgumentType::VisibleFunctionTable),
        D::IntersectionFunctionTable => return handle(ArgumentType::IntersectionFunctionTable),
        D::PrimitiveAccelerationStructure | D::InstanceAccelerationStructure => {
            return handle(ArgumentType::AccelerationStructure)
        }
        D::DepthStencilState => return handle(ArgumentType::DepthStencilState),
        D::Float => (4, 1, 1),
        D::Float2 => (4, 1, 2),
        D::Float3 => (4, 1, 3),
        D::Float4 => (4, 1, 4),
        D::Float2x2 => (4, 2, 2),
        D::Float2x3 => (4, 2, 3),
        D::Float2x4 => (4, 2, 4),
        D::Float3x2 => (4, 3, 2),
        D::Float3x3 => (4, 3, 3),
        D::Float3x4 => (4, 3, 4),
        D::Float4x2 => (4, 4, 2),
        D::Float4x3 => (4, 4, 3),
        D::Float4x4 => (4, 4, 4),
        D::Half => (2, 1, 1),
        D::Half2 => (2, 1, 2),
        D::Half3 => (2, 1, 3),
        D::Half4 => (2, 1, 4),
        D::Half2x2 => (2, 2, 2),
        D::Half2x3 => (2, 2, 3),
        D::Half2x4 => (2, 2, 4),
        D::Half3x2 => (2, 3, 2),
        D::Half3x3 => (2, 3, 3),
        D::Half3x4 => (2, 3, 4),
        D::Half4x2 => (2, 4, 2),
        D::Half4x3 => (2, 4, 3),
        D::Half4x4 => (2, 4, 4),
        D::BFloat => (2, 1, 1),
        D::BFloat2 => (2, 1, 2),
        D::BFloat3 => (2, 1, 3),
        D::BFloat4 => (2, 1, 4),
        D::Int | D::UInt => (4, 1, 1),
        D::Int2 | D::UInt2 => (4, 1, 2),
        D::Int3 | D::UInt3 => (4, 1, 3),
        D::Int4 | D::UInt4 => (4, 1, 4),
        D::Short | D::UShort => (2, 1, 1),
        D::Short2 | D::UShort2 => (2, 1, 2),
        D::Short3 | D::UShort3 => (2, 1, 3),
        D::Short4 | D::UShort4 => (2, 1, 4),
        D::Char | D::UChar | D::Bool => (1, 1, 1),
        D::Char2 | D::UChar2 | D::Bool2 => (1, 1, 2),
        D::Char3 | D::UChar3 | D::Bool3 => (1, 1, 3),
        D::Char4 | D::UChar4 | D::Bool4 => (1, 1, 4),
        D::Long | D::ULong => (8, 1, 1),
        D::Long2 | D::ULong2 => (8, 1, 2),
        D::Long3 | D::ULong3 => (8, 1, 3),
        D::Long4 | D::ULong4 => (8, 1, 4),
        _ => return None,
    };
    // A vector of 3 takes the room of 4; a vector is aligned to its size.
    let vector = scalar * if rows == 3 { 4 } else { rows };
    Some((vector * columns, vector, ArgumentType::Constant))
}

impl ArgumentBufferLayout {
    /// The layout `newArgumentEncoderWithArguments:` makes of `descriptors` (the module docs).
    /// Refused as Metal refuses: no descriptors, a data type no argument has, and a constant block
    /// alignment on a constant that does not start its block.
    pub fn from_descriptors(
        descriptors: &[ArgumentDescriptor],
    ) -> Result<ArgumentBufferLayout, ArgumentLayoutError> {
        if descriptors.is_empty() {
            return Err(ArgumentLayoutError::NoDescriptors);
        }
        let mut cursor = 0u64;
        let mut alignment = 1u64;
        let mut in_block = false;
        let mut members = Vec::with_capacity(descriptors.len());
        for (position, d) in descriptors.iter().enumerate() {
            let (size, natural, ty) =
                type_size_align(d.data_type).ok_or(ArgumentLayoutError::DataType {
                    position,
                    data_type: d.data_type,
                })?;
            let mut align = natural;
            if ty == ArgumentType::Constant {
                if d.constant_block_alignment != 0 {
                    if in_block {
                        return Err(ArgumentLayoutError::ConstantBlockAlignment { position });
                    }
                    align = align.max(u64::from(d.constant_block_alignment));
                }
                in_block = true;
            } else {
                in_block = false;
            }
            let count = d.array_length.max(1);
            let offset = metal_align(cursor, align).ok_or(ArgumentLayoutError::TooLarge)?;
            cursor = size
                .checked_mul(u64::from(count))
                .and_then(|bytes| offset.checked_add(bytes))
                .ok_or(ArgumentLayoutError::TooLarge)?;
            alignment = alignment.max(natural);
            members.push(ArgumentMember {
                id: d.index,
                count,
                ids_per_element: 1,
                offset,
                stride: size,
                size,
                ty,
                layout: None,
            });
        }
        // Type alignments are powers of two, so this rounds up.
        let encoded_length = metal_align(cursor, alignment).ok_or(ArgumentLayoutError::TooLarge)?;
        Ok(ArgumentBufferLayout {
            encoded_length,
            alignment,
            members,
        })
    }

    /// Where argument `index` lands: in the last member whose indices hold it, inside a struct
    /// member's element as that element's layout says. `None` when no member holds it.
    pub fn argument(&self, index: u32) -> Option<ArgumentPlace> {
        self.place(index, 0)
    }

    /// [`argument`](Self::argument) for a layout whose byte 0 is at `base`. A struct element
    /// that does not hold the index leaves it to the members before; the recursion is as deep as
    /// the layout's struct nesting, which the build capped.
    fn place(&self, index: u32, base: u64) -> Option<ArgumentPlace> {
        self.members.iter().rev().find_map(|m| {
            let rel = index.checked_sub(m.id)?;
            let per = m.ids_per_element.max(1);
            let element = rel / per;
            if element >= m.count {
                return None;
            }
            let offset = base
                .checked_add(m.offset)?
                .checked_add(m.stride.checked_mul(u64::from(element))?)?;
            match m.ty {
                ArgumentType::Struct => m.layout.as_ref()?.place(rel % per, offset),
                ty => Some(ArgumentPlace {
                    ty,
                    offset,
                    size: m.size,
                    nested: m.layout.clone(),
                }),
            }
        })
    }

    /// One past the highest argument index the layout's members take (0 for none): a struct of
    /// this layout's index span.
    pub fn id_span(&self) -> u32 {
        self.members
            .iter()
            .map(|m| {
                m.id.saturating_add(m.count.saturating_mul(m.ids_per_element.max(1)))
            })
            .max()
            .unwrap_or(0)
    }
}

/// Why a decoded entry's argument buffer needs the specialized entry: an index or count the AIR
/// states through a function constant, or a parameter at the index that a function constant
/// gates (or two such alternatives).
pub(crate) struct NeedsSpecialization;

/// What reading an entry's argument buffer gave: a layout (or the error Metal's creation gives),
/// or the need to specialize first.
pub(crate) enum EntryLayout {
    Done(Result<ArgumentBufferLayout, ArgumentLayoutError>),
    Specialize,
}

/// The `air.struct_type_info` metadata disagrees with itself (a member past the buffer, an
/// array whose count does not divide its bytes, an argument without its index).
fn mismatch() -> ArgumentLayoutError {
    ArgumentLayoutError::Compile(
        InputError::new(
            InputErrorKind::Abi(AbiError::StructInfoMismatch),
            Location::metadata(None, None, Some("air.struct_type_info")),
        )
        .into(),
    )
}

/// The constant value of `v`, or [`NeedsSpecialization`].
fn constant(v: AbiInt) -> Result<u32, NeedsSpecialization> {
    match v {
        AbiInt::Const(v) => Ok(v),
        AbiInt::Global(_) => Err(NeedsSpecialization),
    }
}

/// The argument-table slot an argument-buffer member's role binds at.
fn member_slot(role: &ParamRole) -> Option<(Slot, ArgumentType)> {
    Some(match role {
        ParamRole::Buffer(b) | ParamRole::ArgumentBuffer(b) => (b.location, ArgumentType::Buffer),
        ParamRole::Texture(t) => (t.location, ArgumentType::Texture),
        ParamRole::Sampler { location } => (*location, ArgumentType::Sampler),
        ParamRole::IndirectConstant { location } => (*location, ArgumentType::Constant),
        ParamRole::CommandBuffer(a) => (a.location, ArgumentType::IndirectCommandBuffer),
        ParamRole::ComputePipelineState(a) => (a.location, ArgumentType::ComputePipelineState),
        ParamRole::RenderPipelineState(a) => (a.location, ArgumentType::RenderPipelineState),
        ParamRole::DepthStencilState(a) => (a.location, ArgumentType::DepthStencilState),
        ParamRole::FunctionTable(a) => (a.location, ArgumentType::VisibleFunctionTable),
        ParamRole::IntersectionTable(a) => (a.location, ArgumentType::IntersectionFunctionTable),
        ParamRole::AccelerationStructure { arg, .. } => {
            (arg.location, ArgumentType::AccelerationStructure)
        }
        ParamRole::Threadgroup(_)
        | ParamRole::Builtin(_)
        | ParamRole::StageIn { .. }
        | ParamRole::VertexInput { .. }
        | ParamRole::FragmentInput(_)
        | ParamRole::ColorInput { .. }
        | ParamRole::PatchInput(_)
        | ParamRole::Imageblock(_)
        | ParamRole::Payload { .. }
        | ParamRole::MeshGridProperties
        | ParamRole::Mesh(_) => return None,
    })
}

/// Builds layouts from an entry's decoded (or specialized) `air.struct_type_info` layouts,
/// sharing each nested layout among the members that name it.
struct Builder<'a, 'm> {
    structs: &'a [StructInfo],
    meter: &'m mut Meter,
    /// The layouts built so far, by struct and the encoder's length and alignment (a nested
    /// argument buffer) or element size (a struct member).
    built: BTreeMap<(u32, u64, u64), Arc<ArgumentBufferLayout>>,
}

/// One layout the builder walks: a struct, at its nesting depth.
type Built = Result<Arc<ArgumentBufferLayout>, Result<ArgumentLayoutError, NeedsSpecialization>>;

impl Builder<'_, '_> {
    /// The layout of struct `r` as an argument buffer of `length` bytes aligned to `alignment`
    /// (or, with `alignment` 1, a struct member's element of `length` bytes).
    fn layout(&mut self, r: StructRef, length: u64, alignment: u64, depth: u64) -> Built {
        if let Some(built) = self.built.get(&(r.0, length, alignment)) {
            return Ok(built.clone());
        }
        self.meter
            .cap(Limit::LayoutNesting, depth)
            .map_err(|e| Ok(ArgumentLayoutError::Compile(e.into())))?;
        let info = self
            .structs
            .get(r.0 as usize)
            .ok_or_else(|| Ok(mismatch()))?;
        let mut members = Vec::new();
        for member in &info.members {
            self.meter
                .tick(1)
                .map_err(|e| Ok(ArgumentLayoutError::Compile(e.into())))?;
            let offset = u64::from(member.offset);
            let size = u64::from(member.size);
            let outer = member.array_len.max(1);
            let pushed = match &member.kind {
                MemberKind::Indirect(arg) => {
                    let (slot, ty) = member_slot(&arg.role).ok_or_else(|| Ok(mismatch()))?;
                    let id = constant(slot.index).map_err(Err)?;
                    let per_slot = constant(slot.count).map_err(Err)?.max(1);
                    if member.array_len > 0 && per_slot > 1 {
                        return Err(Ok(ArgumentLayoutError::ArrayOfArrays { id }));
                    }
                    if !size.is_multiple_of(u64::from(per_slot)) {
                        return Err(Ok(mismatch()));
                    }
                    let nested = match &arg.role {
                        ParamRole::ArgumentBuffer(b) => match b.layout {
                            Some(inner) => {
                                let ArgTypeInfo {
                                    size: Some(length),
                                    align: Some(alignment),
                                    ..
                                } = arg.ty
                                else {
                                    return Err(Ok(mismatch()));
                                };
                                Some(self.layout(
                                    inner,
                                    u64::from(length),
                                    u64::from(alignment),
                                    depth + 1,
                                )?)
                            }
                            None => None,
                        },
                        _ => None,
                    };
                    ArgumentMember {
                        id,
                        count: outer.saturating_mul(per_slot),
                        ids_per_element: 1,
                        offset,
                        stride: size / u64::from(per_slot),
                        size: size / u64::from(per_slot),
                        ty,
                        layout: nested,
                    }
                }
                MemberKind::IndirectStruct { first_id } => {
                    let inner = member.layout.ok_or_else(|| Ok(mismatch()))?;
                    let element = self.layout(inner, size, 1, depth + 1)?;
                    ArgumentMember {
                        id: *first_id,
                        count: outer,
                        ids_per_element: element.id_span(),
                        offset,
                        stride: size,
                        size,
                        ty: ArgumentType::Struct,
                        layout: Some(element),
                    }
                }
                // A struct without argument ids of its own: its members' indices are the
                // enclosing struct's, so it holds them in place (a single element; Metal gives
                // an array of structs ids per element).
                MemberKind::Plain => match member.layout {
                    Some(inner) => {
                        let element = self.layout(inner, size, 1, depth + 1)?;
                        if element.members.is_empty() {
                            continue;
                        }
                        if member.array_len > 1 {
                            return Err(Ok(mismatch()));
                        }
                        ArgumentMember {
                            id: 0,
                            count: 1,
                            ids_per_element: element.id_span(),
                            offset,
                            stride: size,
                            size,
                            ty: ArgumentType::Struct,
                            layout: Some(element),
                        }
                    }
                    // Bytes no argument index names.
                    None => continue,
                },
            };
            let end = pushed
                .stride
                .checked_mul(u64::from(pushed.count))
                .and_then(|bytes| bytes.checked_add(pushed.offset));
            if end.is_none_or(|end| end > length) {
                return Err(Ok(mismatch()));
            }
            self.meter
                .reserve(&mut members, 1)
                .map_err(|e| Ok(ArgumentLayoutError::Compile(e.into())))?;
            members.push(pushed);
        }
        let built = Arc::new(ArgumentBufferLayout {
            encoded_length: length,
            alignment,
            members,
        });
        self.built.insert((r.0, length, alignment), built.clone());
        Ok(built)
    }
}

/// The layout of the argument buffer that entry parameters `params` take at buffer `index`,
/// over `structs` (the module's layouts, or the specialized entry's). Decoded parameters may
/// leave the answer to specialization ([`EntryLayout::Specialize`]): a parameter gated by a
/// function constant at the index, two at it, or an index or count a function constant states.
pub(crate) fn entry_layout(
    params: &[ParamBinding],
    structs: &[StructInfo],
    index: u32,
    meter: &mut Meter,
) -> EntryLayout {
    let mut found = None;
    for p in params {
        let location = match &p.role {
            ParamRole::ArgumentBuffer(b) | ParamRole::Buffer(b) => b.location,
            _ => continue,
        };
        let Ok(at) = constant(location.index) else {
            return EntryLayout::Specialize;
        };
        if at != index {
            continue;
        }
        if p.predicate.is_some() || found.is_some() {
            return EntryLayout::Specialize;
        }
        found = Some(p);
    }
    let Some(p) = found else {
        return EntryLayout::Done(Err(ArgumentLayoutError::NoArgumentBuffer { index }));
    };
    let ParamRole::ArgumentBuffer(b) = &p.role else {
        return EntryLayout::Done(Err(ArgumentLayoutError::NoArgumentBuffer { index }));
    };
    let (Some(r), Some(length), Some(alignment)) = (b.layout, p.ty.size, p.ty.align) else {
        return EntryLayout::Done(Err(mismatch()));
    };
    let mut builder = Builder {
        structs,
        meter,
        built: BTreeMap::new(),
    };
    match builder.layout(r, u64::from(length), u64::from(alignment), 1) {
        Ok(layout) => EntryLayout::Done(Ok(Arc::unwrap_or_clone(layout))),
        Err(Ok(error)) => EntryLayout::Done(Err(error)),
        Err(Err(NeedsSpecialization)) => EntryLayout::Specialize,
    }
}
