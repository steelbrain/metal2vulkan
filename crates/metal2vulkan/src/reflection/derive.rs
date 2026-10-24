//! Reflection from an entry's AIR ABI: each parameter's role, `air.arg_*` type information and
//! `air.struct_type_info` layout, as Metal's answers state them (the module docs list the rules
//! and their measurements).

use super::types::{array_element, data_type, texture};
use super::{
    ArrayType, Binding, BindingKind, BufferBinding, ObjectPayloadBinding, PointerType,
    StructMember, StructType, TextureBinding, TextureReferenceType, ThreadgroupBinding, TypeDetail,
};
use std::collections::BTreeSet;

use crate::air::abi::{
    AbiInt, AccelerationStructureKind, Access, AirAbi, ArgTypeInfo, BufferArg, EntryPoint,
    ImageblockKind, Member, MemberKind, ParamBinding, ParamRole, PatchInput, Slot, Stage,
    StructInfo, StructRef,
};
use crate::air::error::{DiagName, Location};
use crate::air::ir::{AddrSpace, FuncId, Module, Value};
use crate::error::{Error, Feature, Unsupported};
use crate::metal::{BindingAccess, BindingType, DataType};

/// The least alignment Metal reports for a `constant` buffer binding, whatever its pointee's
/// own alignment is (the measurement is stated where it is applied, in `buffer`).
const CONSTANT_ALIGNMENT: u64 = 4;

/// Whose answer a reflection gives. They differ in one measured respect: an argument buffer's
/// array of textures, samplers, pointers or structs has its element size as its stride in a
/// library's answer and 0 in a pipeline's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Channel {
    /// `-[MTLLibrary reflectionForFunctionWithName:]`.
    Library,
    /// The `reflection:` out-parameter of a pipeline state's creation.
    Pipeline,
}

/// The bindings of entry `ep` (its parameters in declaration order; builtins and stage inputs
/// bind nothing) over the ABI's struct layouts `structs`, as `channel` answers them.
///
/// `tables` is the specialized entry's `function_tables`: the Metal buffer indices of its
/// visible-function-table parameters, which devirtualization states as read-only device buffers of
/// words. A binding of one is the table it was, as Metal answers it; an unspecialized entry states
/// the role itself and passes an empty set.
pub(crate) fn bindings(
    ep: &EntryPoint,
    structs: &[StructInfo],
    channel: Channel,
    tables: &BTreeSet<u32>,
) -> Result<Vec<Binding>, Error> {
    let cx = Cx {
        ep,
        structs,
        channel,
        tables,
    };
    let mut out = Vec::new();
    for p in &ep.params {
        if let Some(b) = cx.binding(p)? {
            out.push(b);
        }
    }
    Ok(out)
}

/// `MTLBindingAccess` of a declared access.
fn binding_access(access: Access) -> BindingAccess {
    match access {
        Access::Read | Access::Sample => BindingAccess::ReadOnly,
        Access::Write => BindingAccess::WriteOnly,
        Access::ReadWrite => BindingAccess::ReadWrite,
    }
}

/// Whether entry `ep` states a binding only its function constants decide: a predicate, or an
/// index or count a function constant computes, of an argument or an argument-buffer member. A
/// pipeline's answer then comes from the entry its constants specialize.
pub(crate) fn needs_resolution(ep: &EntryPoint, structs: &[StructInfo]) -> bool {
    let computed = |role: &ParamRole| {
        role_slot(role).is_some_and(|s| {
            matches!(s.index, AbiInt::Global(_)) || matches!(s.count, AbiInt::Global(_))
        })
    };
    ep.params
        .iter()
        .any(|p| p.predicate.is_some() || computed(&p.role))
        || structs.iter().any(|s| {
            s.members.iter().any(|m| match &m.kind {
                MemberKind::Indirect(arg) => computed(&arg.role),
                MemberKind::Plain | MemberKind::IndirectStruct { .. } => false,
            })
        })
}

/// Whether a function made with constant values answers which of its arguments it reads from its
/// specialized code: its module declares function constants (`abi`), which Metal's compile of the
/// function folds. Measured on the M3 Max and the M4 alike (fixture `reflection/specialized_reads`;
/// fixture `reflection/specialized_inputs`): an argument the constants leave unread is inactive or
/// unused though the AIR does not mark it `air.arg_unused` (a vertex input, a kernel's stage input,
/// a post-tessellation function's per-patch input or its control points, and a binding of every
/// kind measured: a buffer, a constant reference, a texture, a sampler, threadgroup memory and an
/// argument buffer), and a pipeline's vertex or stage input descriptor buffer that only such an
/// input reads is unused.
pub(crate) fn specialization_decides_reads(abi: &AirAbi) -> bool {
    !abi.function_constants.is_empty()
}

/// What an entry reads before its constants specialize it, which [`specialized_reads`] compares
/// the specialized code with: the parameters its body's instructions name, and its control-point
/// inputs, which specialization drops where the constants leave them unread. Empty where its module
/// declares no function constant.
#[derive(Clone, Debug, Default)]
pub(crate) struct UnspecializedReads {
    read: BTreeSet<u32>,
    control_points: Vec<ParamBinding>,
}

impl UnspecializedReads {
    /// What entry `ep` of `module` (whose ABI is `abi`) reads before specialization.
    pub(crate) fn of(module: &Module, abi: &AirAbi, ep: &EntryPoint) -> UnspecializedReads {
        if !specialization_decides_reads(abi) {
            return UnspecializedReads::default();
        }
        let control_points = ep
            .params
            .iter()
            .filter(|p| {
                matches!(
                    p.role,
                    ParamRole::PatchInput(PatchInput::ControlPoints { .. })
                )
            })
            .cloned()
            .collect();
        UnspecializedReads {
            read: args_read(module, ep.func),
            control_points,
        }
    }
}

/// The parameters (by index) that the instructions of function `func`'s body in `module` name.
fn args_read(module: &Module, func: FuncId) -> BTreeSet<u32> {
    let mut read = BTreeSet::new();
    if let Some(body) = module.func(func).and_then(|f| f.body.as_ref()) {
        for inst in body.insts.values() {
            inst.kind.for_each_value(|v| {
                if let Value::Arg(a) = v {
                    read.insert(a.raw());
                }
            });
        }
    }
    read
}

/// The specialized entry `spec`'s entry point, each argument its code read before specialization
/// (`before`) and the specialized code never reads marked unused, as Metal answers them for a
/// function made with constant values (see [`specialization_decides_reads`]); `spec.ep` itself
/// where its module declares no constant. An argument the code never names either way keeps the
/// AIR's answer: an explicit imageblock, which the code reaches through its intrinsics, is used
/// (measured, fixture `reflection/leftovers`). Control points the constants leave unread, which
/// specialization drops, are put back unused, as Metal lists their fields inactive.
///
/// Refused (`Unsupported(Reflection)`): such control points with a field a constant gates, whose
/// presence the dropped input no longer resolves.
pub(crate) fn specialized_reads(
    spec: &crate::air::Entry,
    before: &UnspecializedReads,
) -> Result<EntryPoint, Error> {
    let mut ep = spec.ep.clone();
    if !spec.function_constants {
        return Ok(ep);
    }
    let read = args_read(&spec.module, spec.func);
    for p in ep
        .params
        .iter_mut()
        .filter(|p| before.read.contains(&p.param) && !read.contains(&p.param))
    {
        p.unused = true;
    }
    for dropped in &before.control_points {
        if ep.params.iter().any(|p| p.param == dropped.param) {
            continue;
        }
        if let ParamRole::PatchInput(PatchInput::ControlPoints { fields, .. }) = &dropped.role {
            if fields.iter().any(|f| f.predicate.is_some()) {
                return Err(Error::Unsupported(
                    Unsupported::not_implemented(Feature::Reflection).at(Location::metadata(
                        Some(ep.name.clone()),
                        Some(dropped.param),
                        Some("air.patch_control_point_input"),
                    )),
                ));
            }
        }
        let mut unread = dropped.clone();
        unread.unused = true;
        let at = ep.params.partition_point(|p| p.param < unread.param);
        ep.params.insert(at, unread);
    }
    Ok(ep)
}

/// The attribute indices (`[[attribute(n)]]`) entry `ep` reads: its vertex or stage inputs that
/// are not `air.arg_unused` (a specialized entry has no gated-off one). A post-tessellation
/// vertex function's control-point and patch inputs are not among them: Metal answers the
/// buffers they read unused (measured). An index a function constant computes is refused: a
/// pipeline's answer reads the specialized entry, whose indices are constants.
pub(crate) fn read_attributes(ep: &EntryPoint) -> Result<BTreeSet<u32>, Error> {
    let mut out = BTreeSet::new();
    for p in ep.params.iter().filter(|p| !p.unused) {
        let (ParamRole::StageIn { location } | ParamRole::VertexInput { location }) = &p.role
        else {
            continue;
        };
        match location.index {
            AbiInt::Const(i) => {
                out.insert(i);
            }
            AbiInt::Global(_) => {
                return Err(Error::Unsupported(
                    Unsupported::not_implemented(Feature::Reflection).at(Location::metadata(
                        Some(ep.name.clone()),
                        Some(p.param),
                        Some("air.location_index"),
                    )),
                ))
            }
        }
    }
    Ok(out)
}

struct Cx<'a> {
    ep: &'a EntryPoint,
    structs: &'a [StructInfo],
    channel: Channel,
    /// The Metal buffer indices devirtualization rewrote from visible-function tables.
    tables: &'a BTreeSet<u32>,
}

impl Cx<'_> {
    /// The refusal of a binding of parameter `param` whose type `ty` Metal's answer was not
    /// measured for.
    fn gap(&self, param: Option<u32>, ty: Option<&DiagName>) -> Error {
        self.gap_at(param, ty, "air.arg_type_name")
    }

    /// [`Cx::gap`], located at metadata key `key`.
    fn gap_at(&self, param: Option<u32>, ty: Option<&DiagName>, key: &'static str) -> Error {
        let mut u = Unsupported::not_implemented(Feature::Reflection).at(Location::metadata(
            Some(self.ep.name.clone()),
            param,
            Some(key),
        ));
        if let Some(ty) = ty {
            u = u.naming(ty.clone());
        }
        Error::Unsupported(u)
    }

    /// An index or count the metadata states. One a function constant computes (a library's
    /// answer, before any value is given) is refused: Metal's library states values no value of
    /// the constant gives (28, 4294967236 and 18 for `base`, `base + 2` and `base * 2` on both
    /// chips). A pipeline's answer reads the specialized entry, whose integers are constants.
    fn int(&self, param: u32, v: AbiInt, ty: Option<&DiagName>) -> Result<u32, Error> {
        match v {
            AbiInt::Const(x) => Ok(x),
            AbiInt::Global(_) => Err(self.gap_at(Some(param), ty, "air.location_index")),
        }
    }

    fn type_name(&self, param: u32, ty: &ArgTypeInfo) -> Result<String, Error> {
        ty.name
            .as_ref()
            .map(ToString::to_string)
            .ok_or_else(|| self.gap(Some(param), None))
    }

    /// The size and alignment of a pointee.
    fn size_align(&self, param: u32, ty: &ArgTypeInfo) -> Result<(u64, u64), Error> {
        match (ty.size, ty.align) {
            (Some(size), Some(align)) => Ok((u64::from(size), u64::from(align))),
            _ => Err(self.gap(Some(param), ty.name.as_ref())),
        }
    }

    fn binding(&self, p: &ParamBinding) -> Result<Option<Binding>, Error> {
        let arg = p.param;
        let ty = p.ty.name.as_ref();
        let (binding_type, access, index, kind) = match &p.role {
            // A table devirtualization states as the buffer of words it reads: the binding is the
            // table, which is what Metal answers for the same argument.
            ParamRole::Buffer(b) if matches!(b.location.index, AbiInt::Const(i) if self.tables.contains(&i)) => {
                (
                    BindingType::VisibleFunctionTable,
                    BindingAccess::ReadOnly,
                    self.int(arg, b.location.index, ty)?,
                    BindingKind::Plain,
                )
            }
            ParamRole::Buffer(b) | ParamRole::ArgumentBuffer(b) => {
                let argument_buffer = matches!(p.role, ParamRole::ArgumentBuffer(_));
                (
                    BindingType::Buffer,
                    binding_access(b.access),
                    self.int(arg, b.location.index, ty)?,
                    BindingKind::Buffer(self.buffer(arg, &p.ty, b, argument_buffer)?),
                )
            }
            ParamRole::Threadgroup(b) => {
                let (size, align) = self.size_align(arg, &p.ty)?;
                (
                    BindingType::ThreadgroupMemory,
                    binding_access(b.access),
                    self.int(arg, b.location.index, ty)?,
                    BindingKind::Threadgroup(ThreadgroupBinding::new(align, size)),
                )
            }
            ParamRole::Texture(t) => {
                let name = self.type_name(arg, &p.ty)?;
                let tex = texture(array_element(&name)).ok_or_else(|| self.gap(Some(arg), ty))?;
                (
                    BindingType::Texture,
                    binding_access(t.access),
                    self.int(arg, t.location.index, ty)?,
                    BindingKind::Texture(TextureBinding::new(
                        tex.texture_type,
                        tex.data_type,
                        tex.depth,
                        self.int(arg, t.location.count, ty)?,
                    )),
                )
            }
            ParamRole::Sampler { location } => (
                BindingType::Sampler,
                BindingAccess::ReadOnly,
                self.int(arg, location.index, ty)?,
                BindingKind::Plain,
            ),
            ParamRole::FunctionTable(o) => (
                BindingType::VisibleFunctionTable,
                BindingAccess::ReadOnly,
                self.int(arg, o.location.index, ty)?,
                BindingKind::Plain,
            ),
            ParamRole::IntersectionTable(o) => (
                BindingType::IntersectionFunctionTable,
                BindingAccess::ReadOnly,
                self.int(arg, o.location.index, ty)?,
                BindingKind::Plain,
            ),
            ParamRole::AccelerationStructure { kind, arg: o } => (
                match kind {
                    AccelerationStructureKind::Instance => {
                        BindingType::InstanceAccelerationStructure
                    }
                    AccelerationStructureKind::Primitive => {
                        BindingType::PrimitiveAccelerationStructure
                    }
                },
                BindingAccess::ReadOnly,
                self.int(arg, o.location.index, ty)?,
                BindingKind::Plain,
            ),
            // An imageblock: a tile function's implicit one, and an explicit one in a tile
            // function or a compute kernel, measured index 0, read-only, whichever the function
            // does with it (for the explicit ones); a fragment function's imageblock data likewise,
            // as `ImageblockData`.
            ParamRole::Imageblock(ib) => (
                match ib.kind {
                    ImageblockKind::Implicit | ImageblockKind::Explicit => BindingType::Imageblock,
                    ImageblockKind::Data => BindingType::ImageblockData,
                },
                BindingAccess::ReadOnly,
                0,
                BindingKind::Plain,
            ),
            // Builtins and stage inputs bind no argument table entry.
            ParamRole::Builtin(_)
            | ParamRole::StageIn { .. }
            | ParamRole::VertexInput { .. }
            | ParamRole::FragmentInput(_)
            | ParamRole::ColorInput { .. }
            | ParamRole::PatchInput(_) => return Ok(None),
            // The mesh grid and the mesh output bind no argument table entry.
            ParamRole::MeshGridProperties | ParamRole::Mesh(_) => return Ok(None),
            // An object or mesh function's payload (`MTLBindingTypeObjectPayload`): at index
            // `0xFFFFFFFF`, read-write in the object function that writes it and read-only in a
            // mesh function, with the payload type's size and alignment (fixture `mesh_functions`).
            ParamRole::Payload { .. } => {
                let access = match self.ep.stage {
                    Stage::Object { .. } => BindingAccess::ReadWrite,
                    Stage::Mesh { .. } => BindingAccess::ReadOnly,
                    _ => return Err(self.gap(Some(arg), ty)),
                };
                let (size, align) = self.size_align(arg, &p.ty)?;
                (
                    BindingType::ObjectPayload,
                    access,
                    u32::MAX,
                    BindingKind::ObjectPayload(ObjectPayloadBinding::new(align, size)),
                )
            }
            // Roles of argument-buffer members only.
            ParamRole::IndirectConstant { .. }
            | ParamRole::CommandBuffer(_)
            | ParamRole::ComputePipelineState(_)
            | ParamRole::RenderPipelineState(_)
            | ParamRole::DepthStencilState(_) => return Err(self.gap(Some(arg), ty)),
        };
        let name = p.name.as_ref().map(ToString::to_string).unwrap_or_default();
        Ok(Some(Binding::new(
            name,
            binding_type,
            access,
            index,
            !p.unused,
            true,
            kind,
        )))
    }

    /// The `MTLBufferBinding` of a buffer or argument buffer parameter: its pointee's type,
    /// size and alignment, its struct, and itself as a pointer.
    fn buffer(
        &self,
        arg: u32,
        ty: &ArgTypeInfo,
        b: &BufferArg,
        argument_buffer: bool,
    ) -> Result<BufferBinding, Error> {
        let (size, natural) = self.size_align(arg, ty)?;
        // A `constant` buffer's reported alignment has a floor of 4; a `device` buffer's is the
        // pointee's own. Measured on an Apple M3 Max under Metal 27.0.1, over the four shapes that
        // separate the two variables: `constant half&` answers 4 and `device half*` answers 2;
        // `constant char&` answers 4 and `device char*` answers 1; and wrapping either scalar in a
        // one-member struct changes neither answer, so it is the address space that decides and
        // not the pointee's shape. Above the floor the natural value stands on both sides
        // (`constant half3&` answers 8). The data **size** is never clamped: `constant char&`
        // answers size 1 beside alignment 4.
        let align = match b.space {
            AddrSpace::Constant => natural.max(CONSTANT_ALIGNMENT),
            _ => natural,
        };
        let (data_type, detail) = self.pointee(arg, ty, b.layout, argument_buffer)?;
        let struct_type = detail.struct_type().cloned();
        let pointer = PointerType::new(
            data_type,
            binding_access(b.access),
            align,
            size,
            argument_buffer,
            detail,
        );
        Ok(BufferBinding::new(
            align,
            size,
            data_type,
            struct_type,
            Some(pointer),
        ))
    }

    /// A pointee's data type and, for a struct, its members (an argument buffer's when
    /// `argument_buffer`).
    fn pointee(
        &self,
        arg: u32,
        ty: &ArgTypeInfo,
        layout: Option<StructRef>,
        argument_buffer: bool,
    ) -> Result<(DataType, TypeDetail), Error> {
        match layout {
            Some(r) => Ok((
                DataType::Struct,
                TypeDetail::Struct(self.struct_type(arg, r, argument_buffer)?),
            )),
            None => {
                let name = self.type_name(arg, ty)?;
                let dt = data_type(&name).ok_or_else(|| self.gap(Some(arg), ty.name.as_ref()))?;
                Ok((dt, TypeDetail::Plain))
            }
        }
    }

    fn layout(&self, arg: u32, r: StructRef) -> Result<&StructInfo, Error> {
        self.structs
            .get(r.0 as usize)
            .ok_or_else(|| self.gap(Some(arg), None))
    }

    /// The `MTLStructType` of layout `r`, reached through parameter `arg`, inside an argument
    /// buffer when `argument_buffer`.
    fn struct_type(
        &self,
        arg: u32,
        r: StructRef,
        argument_buffer: bool,
    ) -> Result<StructType, Error> {
        let info = self.layout(arg, r)?;
        let mut members = Vec::with_capacity(info.members.len());
        for m in &info.members {
            members.push(self.member(arg, m, argument_buffer)?);
        }
        Ok(StructType::new(members))
    }

    /// The argument ids one element of layout `r` takes: one past its last member's.
    fn id_span(&self, arg: u32, r: StructRef) -> Result<u32, Error> {
        let info = self.layout(arg, r)?;
        let mut span = 0u32;
        for m in &info.members {
            let (first, count) = match &m.kind {
                MemberKind::Plain => continue,
                MemberKind::Indirect(ia) => {
                    let (first, ids) = self.member_ids(arg, m, &ia.role)?;
                    (first, ids)
                }
                MemberKind::IndirectStruct { first_id } => {
                    let inner = m.layout.ok_or_else(|| self.gap(Some(arg), None))?;
                    let per = self.id_span(arg, inner)?;
                    (*first_id, per.saturating_mul(m.array_len.max(1)))
                }
            };
            span = span.max(first.saturating_add(count));
        }
        Ok(span)
    }

    /// An argument-buffer member's first id and the ids it takes.
    fn member_ids(&self, arg: u32, m: &Member, role: &ParamRole) -> Result<(u32, u32), Error> {
        let slot = role_slot(role).ok_or_else(|| self.gap(Some(arg), Some(&m.type_name)))?;
        let first = self.int(arg, slot.index, Some(&m.type_name))?;
        let count = self.int(arg, slot.count, Some(&m.type_name))?;
        Ok((first, count.max(m.array_len)))
    }

    /// One `MTLStructMember` of a layout reached through parameter `arg`, an argument buffer's
    /// when `argument_buffer`.
    fn member(&self, arg: u32, m: &Member, argument_buffer: bool) -> Result<StructMember, Error> {
        let name = m.name.to_string();
        let offset = u64::from(m.offset);
        let type_name = m.type_name.to_string();
        let gap = || self.gap(Some(arg), Some(&m.type_name));
        match &m.kind {
            // An argument buffer's member the AIR gives no argument index: Apple's compiler gave
            // every member one in each shape measured (plain-data structs held
            // by value and arrays of them, explicit ids with a gap, padding), so what Metal
            // answers for one is unmeasured.
            MemberKind::Plain if argument_buffer => Err(gap()),
            MemberKind::Plain => {
                let (dt, detail) = match m.layout {
                    Some(r) => (
                        DataType::Struct,
                        TypeDetail::Struct(self.struct_type(arg, r, false)?),
                    ),
                    None => (data_type(&type_name).ok_or_else(gap)?, TypeDetail::Plain),
                };
                let (dt, detail) = array_of(m, dt, detail, 0, u64::from(m.size));
                Ok(StructMember::new(name, offset, dt, 0, detail))
            }
            MemberKind::IndirectStruct { first_id } => {
                let r = m.layout.ok_or_else(gap)?;
                let detail = TypeDetail::Struct(self.struct_type(arg, r, argument_buffer)?);
                let per = self.id_span(arg, r)?;
                let stride = self.struct_array_stride(arg, r, u64::from(m.size), &m.type_name)?;
                let (dt, detail) = array_of(m, DataType::Struct, detail, per, stride);
                Ok(StructMember::new(name, offset, dt, *first_id, detail))
            }
            MemberKind::Indirect(ia) => {
                let (first, ids) = self.member_ids(arg, m, &ia.role)?;
                // Handles and pointers are resources; an inline constant is data.
                let resource = !matches!(ia.role, ParamRole::IndirectConstant { .. });
                let (dt, detail) = match &ia.role {
                    ParamRole::Texture(t) => {
                        let tex = texture(array_element(&type_name)).ok_or_else(gap)?;
                        (
                            DataType::Texture,
                            TypeDetail::TextureReference(TextureReferenceType::new(
                                tex.data_type,
                                tex.texture_type,
                                binding_access(t.access),
                                tex.depth,
                            )),
                        )
                    }
                    ParamRole::Sampler { .. } => (DataType::Sampler, TypeDetail::Plain),
                    ParamRole::Buffer(b) | ParamRole::ArgumentBuffer(b) => {
                        let nested = matches!(ia.role, ParamRole::ArgumentBuffer(_));
                        let (size, align) = self.size_align(arg, &ia.ty)?;
                        let (pointee, detail) = self.pointee(arg, &ia.ty, b.layout, nested)?;
                        let pointer = PointerType::new(
                            pointee,
                            binding_access(b.access),
                            align,
                            size,
                            nested,
                            detail,
                        );
                        (DataType::Pointer, TypeDetail::Pointer(Box::new(pointer)))
                    }
                    ParamRole::IndirectConstant { .. } => {
                        (data_type(&type_name).ok_or_else(gap)?, TypeDetail::Plain)
                    }
                    ParamRole::CommandBuffer(_) => {
                        (DataType::IndirectCommandBuffer, TypeDetail::Plain)
                    }
                    ParamRole::ComputePipelineState(_) => {
                        (DataType::ComputePipeline, TypeDetail::Plain)
                    }
                    // Measured as a compute pipeline state is: one, and an array of two with the
                    // stride of an array of resources.
                    ParamRole::RenderPipelineState(_) => {
                        (DataType::RenderPipeline, TypeDetail::Plain)
                    }
                    // A visible function table, likewise (the fixture `tables_in_arguments`).
                    ParamRole::FunctionTable(_) => {
                        (DataType::VisibleFunctionTable, TypeDetail::Plain)
                    }
                    _ => return Err(gap()),
                };
                // An `array<T, N>` of handles states its total size, and its length as its slot's
                // count; a C array (`T x [N]`) states its element size and its length.
                let element_size = if m.array_len > 0 {
                    u64::from(m.size)
                } else {
                    u64::from(m.size) / u64::from(ids.max(1))
                };
                let stride = if resource {
                    self.resource_stride(element_size)
                } else {
                    element_size
                };
                let (dt, detail) = if m.array_len > 0 {
                    array_of(m, dt, detail, 1, stride)
                } else if type_name.starts_with("array<") {
                    let array = ArrayType::new(dt, ids, stride, 1, detail);
                    (DataType::Array, TypeDetail::Array(Box::new(array)))
                } else {
                    (dt, detail)
                };
                Ok(StructMember::new(name, offset, dt, first, detail))
            }
        }
    }

    /// The stride of an argument buffer's array of resources (textures, samplers, pointers) of
    /// elements `element_size` bytes: the element size in a library's answer, 0 in a pipeline's.
    fn resource_stride(&self, element_size: u64) -> u64 {
        match self.channel {
            Channel::Library => element_size,
            Channel::Pipeline => 0,
        }
    }

    /// The stride of an argument buffer's array of structs of layout `r` (type `ty`, reached
    /// through parameter `arg`) of elements `element_size` bytes. A library's answer is the
    /// element size. A pipeline's is 0 when every member of the struct is a resource (a texture,
    /// sampler, pointer, ...), as for an array of resources, and the element size when a member
    /// is a constant (argument_buffers' `pairs`, encoder_arrays' `a0` to `a4`, `a7` and `a8`,
    /// encoders' `a8`, on an M3 Max and an M4). A pipeline's stride of a struct whose members are
    /// no constant but hold a struct is unmeasured: refused.
    fn struct_array_stride(
        &self,
        arg: u32,
        r: StructRef,
        element_size: u64,
        ty: &DiagName,
    ) -> Result<u64, Error> {
        if matches!(self.channel, Channel::Library) {
            return Ok(element_size);
        }
        let mut nested = false;
        for m in &self.layout(arg, r)?.members {
            match &m.kind {
                MemberKind::Indirect(ia) => {
                    if matches!(ia.role, ParamRole::IndirectConstant { .. }) {
                        return Ok(element_size);
                    }
                }
                MemberKind::IndirectStruct { .. } | MemberKind::Plain => nested = true,
            }
        }
        if nested {
            Err(self.gap(Some(arg), Some(ty)))
        } else {
            Ok(0)
        }
    }
}

/// Member `m` as an `MTLArrayType` of `dt` with `stride` when it is a C array (`array_len` above
/// 0), each element taking `ids_per_element` argument ids.
fn array_of(
    m: &Member,
    dt: DataType,
    detail: TypeDetail,
    ids_per_element: u32,
    stride: u64,
) -> (DataType, TypeDetail) {
    if m.array_len == 0 {
        return (dt, detail);
    }
    let array = ArrayType::new(dt, m.array_len, stride, ids_per_element, detail);
    (DataType::Array, TypeDetail::Array(Box::new(array)))
}

/// The argument-table slot of an argument-buffer member's role.
fn role_slot(role: &ParamRole) -> Option<Slot> {
    match role {
        ParamRole::Buffer(b) | ParamRole::ArgumentBuffer(b) | ParamRole::Threadgroup(b) => {
            Some(b.location)
        }
        ParamRole::Texture(t) => Some(t.location),
        ParamRole::Sampler { location } | ParamRole::IndirectConstant { location } => {
            Some(*location)
        }
        ParamRole::AccelerationStructure { arg, .. }
        | ParamRole::FunctionTable(arg)
        | ParamRole::IntersectionTable(arg)
        | ParamRole::CommandBuffer(arg)
        | ParamRole::ComputePipelineState(arg)
        | ParamRole::RenderPipelineState(arg)
        | ParamRole::DepthStencilState(arg) => Some(arg.location),
        _ => None,
    }
}
