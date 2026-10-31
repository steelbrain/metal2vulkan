//! Cross-checks of the decoded ABI against the IR and against a metallib function's container
//! metadata.

use std::collections::BTreeSet;

use super::md::{abi_error, Site};
use super::{
    AirAbi, CoreImageKernel, CoreImageParam, CoreImageRole, EntryPoint, ParamBinding, ParamRole,
    PatchInput, Predicate, Stage, StructRef, VisibleFunction,
};
use crate::bitcode::PointeeTable;
use crate::budget::Meter;
use crate::container::{FunctionStage, LibraryFunction};
use crate::error::{AbiError, InputError, InputErrorKind, Location};
use crate::error::{Error, Feature};
use crate::ir::{AddrSpace, Const, FuncId, Module, Section, Type, TypeId, TypeLayouts};
use crate::metal::DataType;

/// Check `abi` against `m` (and, for typed-pointer bitcode, the pointees). `layouts` are the
/// struct layouts [`super::decode`] built for that bitcode, `None` when no pointee is recorded.
pub(super) fn module(
    m: &Module,
    pointees: &PointeeTable,
    layouts: Option<&TypeLayouts>,
    abi: &AirAbi,
    meter: &mut Meter,
) -> Result<(), Error> {
    let mut funcs = BTreeSet::new();
    for e in &abi.entries {
        meter.tick(1)?;
        let site = Site {
            entry: Some(e.name.clone()),
            arg: None,
        };
        meter.charge(FUNC_ENTRY)?;
        if !funcs.insert(e.func) {
            return Err(site.err(AbiError::BadValue, None));
        }
        entry(m, pointees, layouts, abi, e, &site, meter)?;
    }
    let mut visible = BTreeSet::new();
    for v in &abi.visible_functions {
        meter.tick(1)?;
        let site = Site {
            entry: Some(v.name.clone()),
            arg: None,
        };
        meter.charge(FUNC_ENTRY)?;
        if !visible.insert(v.func) {
            return Err(site.err(AbiError::BadValue, Some("air.visible")));
        }
        visible_function(m, v, &site, meter)?;
    }
    let mut core_image = BTreeSet::new();
    for k in &abi.core_image_kernels {
        meter.tick(1)?;
        let site = Site {
            entry: Some(k.name.clone()),
            arg: None,
        };
        meter.charge(FUNC_ENTRY)?;
        if !core_image.insert(k.func) {
            return Err(site.err(AbiError::BadValue, Some(super::core_image::LIST)));
        }
        core_image_kernel(m, pointees, layouts, k, &site, meter)?;
    }
    let mut indices = BTreeSet::new();
    for fc in &abi.function_constants {
        meter.tick(1)?;
        let site = Site::module();
        let key = Some("air.function_constants");
        let g = m
            .global(fc.global)
            .ok_or_else(|| site.err(AbiError::BadValue, key))?;
        // The host supplies the value, so the initializer is `undef` and the object is written
        // from outside the module before execution starts. Apple's emitter also places the
        // object in `air.fc_initializer` from AIR 2.5 on, and in no section from AIR 2.0 to 2.4,
        // so the section is not part of the test: the `!air.function_constants` record is what
        // names the global, and these are the properties the binding needs of it. The record
        // itself is byte-identical from AIR 2.0 to 2.9; only this attribute of the global moves.
        let undef = g
            .init
            .and_then(|c| m.consts().get(c))
            .is_some_and(|c| c.value == Const::Undef);
        let placed = matches!(g.section, Some(Section::FcInitializer) | None);
        let is_initializer = placed
            && g.constant
            && g.space == AddrSpace::Constant
            && g.externally_initialized
            && undef;
        if !is_initializer {
            return Err(site.err(AbiError::BadValue, key));
        }
        meter.charge(INDEX_ENTRY)?;
        if !indices.insert(fc.index) {
            return Err(site.err(AbiError::BadValue, key));
        }
    }
    meter.release(FUNC_ENTRY.saturating_mul(funcs.len()));
    meter.release(FUNC_ENTRY.saturating_mul(visible.len()));
    meter.release(FUNC_ENTRY.saturating_mul(core_image.len()));
    meter.release(INDEX_ENTRY.saturating_mul(indices.len()));
    Ok(())
}

// Ledger charges for one entry of a transient tree set: twice the entry's size (node overhead),
// charged before the insertion and returned when the set is dropped.
const FUNC_ENTRY: usize = 2 * std::mem::size_of::<FuncId>();
const INDEX_ENTRY: usize = 2 * std::mem::size_of::<u32>();
const DONE_ENTRY: usize = 2 * std::mem::size_of::<(StructRef, u64)>();

/// Ledger charge for one entry of the [`fits`] walk's stack (released when it is popped).
const STACK_ENTRY: usize = std::mem::size_of::<(StructRef, u64, usize)>();

fn entry(
    m: &Module,
    pointees: &PointeeTable,
    layouts: Option<&TypeLayouts>,
    abi: &AirAbi,
    e: &EntryPoint,
    site: &Site,
    meter: &mut Meter,
) -> Result<(), Error> {
    let mismatch = |arg: Option<u32>| {
        abi_error(
            AbiError::SignatureMismatch,
            Location::metadata(site.entry.clone(), arg, None),
        )
    };
    let f = m.func(e.func).ok_or_else(|| mismatch(None))?;
    if f.body.is_none() {
        return Err(mismatch(None));
    }
    let Some(Type::Function { ret, params, .. }) = m.types().get(f.ty) else {
        return Err(mismatch(None));
    };
    // One binding per parameter.
    if e.params.len() != params.len() {
        return Err(mismatch(None));
    }
    meter.charge(params.len())?;
    let mut seen = vec![false; params.len()];
    for p in &e.params {
        meter.tick(1)?;
        let slot = seen
            .get_mut(p.param as usize)
            .ok_or_else(|| mismatch(Some(p.param)))?;
        if *slot {
            return Err(mismatch(Some(p.param)));
        }
        *slot = true;
        let ty = params
            .get(p.param as usize)
            .copied()
            .ok_or_else(|| mismatch(Some(p.param)))?;
        let arg_site = Site {
            entry: site.entry.clone(),
            arg: Some(p.param),
        };
        role_type(m, p, ty, &arg_site)?;
        predicate(m, p.predicate.as_ref(), &arg_site)?;
        if let ParamRole::PatchInput(PatchInput::ControlPoints { fields, .. }) = &p.role {
            for field in fields {
                meter.tick(1)?;
                predicate(m, field.predicate.as_ref(), &arg_site)?;
            }
        }
        if let ParamRole::Mesh(mesh) = &p.role {
            for field in mesh.vertex.iter().chain(&mesh.primitive) {
                meter.tick(1)?;
                predicate(m, field.predicate.as_ref(), &arg_site)?;
            }
        }
        if let (Some(layouts), Some(pointee)) = (layouts, pointees.param(e.func, p.param as usize))
        {
            pointee_check(abi, layouts, p, pointee, &arg_site, meter)?;
        }
    }
    outputs(m, e, *ret, site)?;
    for o in &e.outputs {
        meter.tick(1)?;
        predicate(m, o.predicate.as_ref(), site)?;
    }
    if let Stage::Vertex { patch: Some(_) } = e.stage {
        // A post-tessellation vertex function reads its control points through a function.
        for p in &e.params {
            meter.tick(1)?;
            if let ParamRole::PatchInput(PatchInput::ControlPoints { function, fields }) = &p.role {
                control_point_function(m, *function, fields.len(), site)?;
            }
        }
    }
    meter.release(params.len());
    Ok(())
}

/// A visible function against its IR signature: it is defined here, every LLVM parameter has
/// exactly one binding, and it has one output exactly when it returns a value.
fn visible_function(
    m: &Module,
    v: &VisibleFunction,
    site: &Site,
    meter: &mut Meter,
) -> Result<(), Error> {
    let mismatch = |arg: Option<u32>| {
        abi_error(
            AbiError::SignatureMismatch,
            Location::metadata(site.entry.clone(), arg, Some("air.visible")),
        )
    };
    let f = m.func(v.func).ok_or_else(|| mismatch(None))?;
    if f.body.is_none() {
        return Err(mismatch(None));
    }
    let Some(Type::Function { ret, params, .. }) = m.types().get(f.ty) else {
        return Err(mismatch(None));
    };
    if v.params.len() != params.len() {
        return Err(mismatch(None));
    }
    meter.charge(params.len())?;
    let mut seen = vec![false; params.len()];
    for p in &v.params {
        meter.tick(1)?;
        let slot = seen
            .get_mut(p.param as usize)
            .ok_or_else(|| mismatch(Some(p.param)))?;
        if *slot {
            return Err(mismatch(Some(p.param)));
        }
        *slot = true;
    }
    meter.release(params.len());
    let returns = m.types().get(*ret) != Some(&Type::Void);
    if v.outputs.len() != usize::from(returns) {
        return Err(mismatch(None));
    }
    Ok(())
}

/// A Core Image kernel against its IR signature: it is defined here, every LLVM parameter has
/// exactly one binding whose role agrees with the parameter's type, and it has an output exactly
/// when it returns a value, which is then a scalar or vector number. For typed-pointer bitcode, a
/// pointer's `air.arg_type_size` and `air.arg_type_align_size` are checked against its pointee like
/// a buffer's.
fn core_image_kernel(
    m: &Module,
    pointees: &PointeeTable,
    layouts: Option<&TypeLayouts>,
    k: &CoreImageKernel,
    site: &Site,
    meter: &mut Meter,
) -> Result<(), Error> {
    let key = Some(super::core_image::LIST);
    let at = |kind: AbiError, arg: Option<u32>| {
        abi_error(kind, Location::metadata(site.entry.clone(), arg, key))
    };
    let mismatch = |arg: Option<u32>| at(AbiError::SignatureMismatch, arg);
    let f = m.func(k.func).ok_or_else(|| mismatch(None))?;
    if f.body.is_none() {
        return Err(mismatch(None));
    }
    let Some(Type::Function { ret, params, .. }) = m.types().get(f.ty) else {
        return Err(mismatch(None));
    };
    if k.params.len() != params.len() {
        return Err(mismatch(None));
    }
    meter.charge(params.len())?;
    let mut seen = vec![false; params.len()];
    for p in &k.params {
        meter.tick(1)?;
        let slot = seen
            .get_mut(p.param as usize)
            .ok_or_else(|| mismatch(Some(p.param)))?;
        if *slot {
            return Err(mismatch(Some(p.param)));
        }
        *slot = true;
        let ty = params
            .get(p.param as usize)
            .copied()
            .ok_or_else(|| mismatch(Some(p.param)))?;
        if !core_image_role_type(m, p.role, ty) {
            return Err(at(AbiError::RoleTypeMismatch, Some(p.param)));
        }
        if let (Some(layouts), Some(pointee)) = (layouts, pointees.param(k.func, p.param as usize))
        {
            let arg_site = Site {
                entry: site.entry.clone(),
                arg: Some(p.param),
            };
            core_image_pointee_check(layouts, p, pointee, &arg_site)?;
        }
    }
    meter.release(params.len());
    match (&k.output, m.types().get(*ret)) {
        (None, Some(Type::Void)) => Ok(()),
        (None, _) | (Some(_), Some(Type::Void)) => Err(mismatch(None)),
        (Some(_), _) if number_or_vector(m, *ret) => Ok(()),
        (Some(_), _) => Err(at(AbiError::RoleTypeMismatch, None)),
    }
}

/// The ABI cross-check of a Core Image kernel parameter with a typed-pointer pointee: a pointer's
/// `air.arg_type_size` and `air.arg_type_align_size` against the DataLayout of its pointee, as
/// for a buffer. No other role states a pointee size.
pub(super) fn core_image_pointee_check(
    layouts: &TypeLayouts,
    p: &CoreImageParam,
    pointee: TypeId,
    site: &Site,
) -> Result<(), Error> {
    let CoreImageRole::Pointer { size, align, .. } = p.role else {
        return Ok(());
    };
    let Some(layout) = layouts.get(pointee) else {
        return Ok(());
    };
    size_and_align(Some(size), Some(align), layout.alloc_size, site)
}

/// A Core Image kernel parameter's role against its IR type. A texture or sampler handle is a
/// pointer in the space the entry-point roles give it (device, constant), and a matrix is an
/// array of vectors; each is passed bare or as the only field of a by-value struct. A pointer is
/// a pointer, and a builtin is a scalar or vector number.
fn core_image_role_type(m: &Module, role: CoreImageRole, ty: TypeId) -> bool {
    let types = m.types();
    let ptr_in = |ty: TypeId, want: Option<AddrSpace>| match types.get(ty) {
        Some(Type::Ptr { space }) => want.is_none_or(|w| *space == w),
        _ => false,
    };
    let matrix = |ty: TypeId| match types.get(ty) {
        Some(Type::Array { elem, .. }) => matches!(types.get(*elem), Some(Type::Vector { .. })),
        _ => false,
    };
    // `ty` itself, or the only field of a struct holding it.
    let bare_or_wrapped = |is: &dyn Fn(TypeId) -> bool| {
        is(ty)
            || matches!(types.get(ty), Some(Type::Struct { fields, .. })
                if matches!(**fields, [field] if is(field)))
    };
    match role {
        CoreImageRole::Builtin => number_or_vector(m, ty),
        CoreImageRole::Matrix => bare_or_wrapped(&matrix),
        CoreImageRole::Texture { .. } => bare_or_wrapped(&|t| ptr_in(t, Some(AddrSpace::Device))),
        CoreImageRole::Sampler => bare_or_wrapped(&|t| ptr_in(t, Some(AddrSpace::Constant))),
        CoreImageRole::Pointer { .. } => ptr_in(ty, None),
    }
}

/// An integer or floating-point scalar, or a vector of them.
fn number_or_vector(m: &Module, ty: TypeId) -> bool {
    let number = |ty: TypeId| {
        matches!(
            m.types().get(ty),
            Some(Type::Int(_) | Type::Half | Type::BFloat | Type::Float | Type::Double)
        )
    };
    match m.types().get(ty) {
        Some(Type::Vector { elem, .. }) => number(*elem),
        _ => number(ty),
    }
}

/// A function-constant predicate names the global the static initializer stores the predicate
/// into: a defined, writable, non-externally-initialized constant-space global of integer type
/// (the frontend stores `bool` predicates as `i8`, and a few as `i16` or `i32`).
fn predicate(m: &Module, p: Option<&Predicate>, site: &Site) -> Result<(), Error> {
    let Some(p) = p else {
        return Ok(());
    };
    let ok = m.global(p.global).is_some_and(|g| {
        g.space == AddrSpace::Constant
            && !g.constant
            && !g.externally_initialized
            && g.section.is_none()
            && g.init.is_some()
            && matches!(m.types().get(g.value_ty), Some(Type::Int(_)))
    });
    if ok {
        Ok(())
    } else {
        Err(site.err(AbiError::BadValue, Some("air.function_constant")))
    }
}

/// A parameter's role against its IR type: resource and buffer roles are pointers in the space
/// the role implies, value roles are not pointers. A buffer, texture, or sampler table passed
/// as an `array_ref` is a thread-space pointer to its elements.
fn role_type(m: &Module, p: &ParamBinding, ty: TypeId, site: &Site) -> Result<(), Error> {
    let space = m.types().ptr_space(ty);
    let ok = match &p.role {
        ParamRole::Buffer(b) | ParamRole::ArgumentBuffer(b) | ParamRole::Threadgroup(b) => {
            space == Some(b.space) || space == Some(AddrSpace::Thread)
        }
        ParamRole::Texture(_) => {
            space == Some(AddrSpace::Device) || space == Some(AddrSpace::Thread)
        }
        ParamRole::Sampler { .. } => {
            space == Some(AddrSpace::Constant) || space == Some(AddrSpace::Thread)
        }
        ParamRole::AccelerationStructure { .. }
        | ParamRole::FunctionTable(_)
        | ParamRole::IntersectionTable(_)
        | ParamRole::CommandBuffer(_)
        | ParamRole::ComputePipelineState(_)
        | ParamRole::RenderPipelineState(_)
        | ParamRole::DepthStencilState(_) => space.is_some(),
        ParamRole::Imageblock(_) | ParamRole::PatchInput(PatchInput::ControlPoints { .. }) => true,
        // The payload is an `object_data` pointer, the grid an opaque threadgroup pointer, the mesh
        // output an opaque pointer in its own space.
        ParamRole::Payload { .. } => space == Some(AddrSpace::ObjectData),
        ParamRole::MeshGridProperties => space == Some(AddrSpace::Threadgroup),
        ParamRole::Mesh(_) => space == Some(AddrSpace::MeshOutput),
        ParamRole::Builtin(_)
        | ParamRole::StageIn { .. }
        | ParamRole::VertexInput { .. }
        | ParamRole::FragmentInput(_)
        | ParamRole::ColorInput { .. }
        | ParamRole::PatchInput(PatchInput::Patch { .. })
        | ParamRole::IndirectConstant { .. } => space.is_none(),
    };
    if ok {
        Ok(())
    } else {
        Err(site.err(AbiError::RoleTypeMismatch, None))
    }
}

/// `air.arg_type_size`, `air.arg_type_align_size`, `air.buffer_size`, and `air.struct_type_info` of
/// a pointer parameter against the DataLayout of its pointee.
pub(super) fn pointee_check(
    abi: &AirAbi,
    layouts: &TypeLayouts,
    p: &ParamBinding,
    pointee: TypeId,
    site: &Site,
    meter: &mut Meter,
) -> Result<(), Error> {
    let buffer = match &p.role {
        ParamRole::Buffer(b) | ParamRole::ArgumentBuffer(b) | ParamRole::Threadgroup(b) => b,
        _ => return Ok(()),
    };
    let Some(layout) = layouts.get(pointee) else {
        return Ok(());
    };
    size_and_align(p.ty.size, p.ty.align, layout.alloc_size, site)?;
    // `air.buffer_size` is the size of the referenced object, which is an array of the pointee
    // for a reference to an array (`constant T (&)[n]`).
    if buffer
        .buffer_size
        .is_some_and(|s| layout.alloc_size == 0 || u64::from(s) % layout.alloc_size != 0)
    {
        return Err(site.err(AbiError::TypeSizeMismatch, Some("air.buffer_size")));
    }
    if let Some(r) = buffer.layout {
        let mut done = BTreeSet::new();
        fits(abi, r, layout.alloc_size, site, &mut done, meter)?;
        meter.release(DONE_ENTRY.saturating_mul(done.len()));
    }
    Ok(())
}

/// `air.arg_type_size` and `air.arg_type_align_size` of a pointer against its pointee's
/// allocation size. The LLVM type's alignment is not Metal's: clang lowers a struct whose natural
/// LLVM layout differs from the C++ one as a packed struct (alignment 1), and `alignas` raises
/// Metal's. The size must agree, and Metal's alignment must be a power of two (it need not
/// divide the size: an over-aligned struct keeps its unpadded size here).
fn size_and_align(
    size: Option<u32>,
    align: Option<u32>,
    alloc_size: u64,
    site: &Site,
) -> Result<(), Error> {
    if size.is_some_and(|s| u64::from(s) != alloc_size) {
        return Err(site.err(AbiError::TypeSizeMismatch, Some("air.arg_type_size")));
    }
    if align.is_some_and(|a| !a.is_power_of_two()) {
        return Err(site.err(AbiError::TypeSizeMismatch, Some("air.arg_type_align_size")));
    }
    Ok(())
}

/// Every member of layout `r` (and, recursively, of its nested layouts) lies within `size` bytes.
/// A scalar member only has to start inside: a bitfield's metadata names the storage unit's
/// type and the byte its bits start in, so its type can extend past the struct's end.
///
/// The walk visits members in depth-first order with an explicit stack, so its host stack use
/// does not grow with nesting, and visits each (layout, size) pair once.
fn fits(
    abi: &AirAbi,
    r: StructRef,
    size: u64,
    site: &Site,
    done: &mut BTreeSet<(StructRef, u64)>,
    meter: &mut Meter,
) -> Result<(), Error> {
    let bad = || site.err(AbiError::StructInfoMismatch, Some("air.struct_type_info"));
    // One entry per layout being checked: the layout, its size, and its next member.
    let mut stack: Vec<(StructRef, u64, usize)> = Vec::new();
    let mut next = Some((r, size));
    loop {
        if let Some((r, size)) = next.take() {
            if !done.contains(&(r, size)) {
                meter.charge(DONE_ENTRY)?;
                done.insert((r, size));
                meter.reserve(&mut stack, 1)?;
                stack.push((r, size, 0));
            }
        }
        let Some((r, size, at)) = stack.last_mut() else {
            return Ok(());
        };
        let info = abi.struct_info(*r).ok_or_else(bad)?;
        let Some(member) = info.members.get(*at) else {
            stack.pop();
            meter.release(STACK_ENTRY);
            continue;
        };
        *at += 1;
        let size = *size;
        meter.tick(1)?;
        let count = u64::from(member.array_len.max(1));
        let end = u64::from(member.size)
            .checked_mul(count)
            .and_then(|b| b.checked_add(u64::from(member.offset)))
            .ok_or_else(bad)?;
        let whole = member.array_len > 0 || member.layout.is_some();
        if u64::from(member.offset) >= size.max(1) || (whole && end > size) {
            return Err(bad());
        }
        next = member.layout.map(|nested| (nested, u64::from(member.size)));
    }
}

/// The outputs against the return type: none for `void`, one for a non-struct value, and one
/// per field of a returned struct.
fn outputs(m: &Module, e: &EntryPoint, ret: TypeId, site: &Site) -> Result<(), Error> {
    let fields = match m.types().get(ret) {
        Some(Type::Void) => 0,
        Some(Type::Struct { fields, .. }) => fields.len(),
        Some(_) => 1,
        None => return Err(site.err(AbiError::SignatureMismatch, None)),
    };
    if fields == e.outputs.len() {
        Ok(())
    } else {
        Err(site.err(AbiError::SignatureMismatch, None))
    }
}

/// A control-point function is a declaration of the module (Metal synthesizes its body from the
/// pipeline's vertex descriptor) taking the control point's index and the control-point input,
/// `(i32, ptr)`, and returning a struct with one member per field.
fn control_point_function(m: &Module, f: FuncId, fields: usize, site: &Site) -> Result<(), Error> {
    let key = Some("air.patch_control_point_function");
    let func = m.func(f).ok_or_else(|| site.err(AbiError::BadValue, key))?;
    let types = m.types();
    let ok = func.body.is_none()
        && match types.get(func.ty) {
            Some(Type::Function {
                ret,
                params,
                vararg: false,
            }) => {
                matches!(params.as_ref(), [i, p]
                    if matches!(types.get(*i), Some(Type::Int(32)))
                        && matches!(types.get(*p), Some(Type::Ptr { .. })))
                    && matches!(types.get(*ret), Some(Type::Struct { fields: members, .. })
                        if members.len() == fields)
            }
            _ => false,
        };
    if ok {
        Ok(())
    } else {
        Err(site.err(AbiError::SignatureMismatch, key))
    }
}

/// `Unsupported(Stage)` for a container `TYPE` tag outside the modeled stages.
fn unsupported_stage(tag: u8) -> Error {
    Error::InvalidInput(InputError::new(
        InputErrorKind::Unsupported(Feature::Stage(tag)),
        Location::module(),
    ))
}

/// Cross-check against a metallib function's container metadata (see
/// [`AirAbi::check_container`]). Every `CNST` entry is validated before any is recorded, so a
/// refusal leaves `abi` unchanged.
pub(super) fn container(
    abi: &mut AirAbi,
    m: &Module,
    f: &LibraryFunction<'_>,
    meter: &mut Meter,
) -> Result<(), Error> {
    let site = Site::module();
    let mismatch = || site.err(AbiError::ContainerMismatch, None);
    let stage = match f.stage {
        FunctionStage::Kernel => |s: &Stage| matches!(s, Stage::Kernel { .. }),
        FunctionStage::Vertex => |s: &Stage| matches!(s, Stage::Vertex { .. }),
        FunctionStage::Fragment => |s: &Stage| matches!(s, Stage::Fragment { .. }),
        FunctionStage::Mesh => |s: &Stage| matches!(s, Stage::Mesh { .. }),
        FunctionStage::Object => |s: &Stage| matches!(s, Stage::Object { .. }),
        // Intersection (`TYPE` 6) and every other stage are unmodeled, not malformed.
        FunctionStage::Intersection => return Err(unsupported_stage(6)),
        FunctionStage::Other(tag) => return Err(unsupported_stage(tag)),
        // A visible function, a function a dynamic library exports, or a Core Image kernel
        // library's extern function, is no entry point: its module declares none to match. A
        // library never compiles one as a stage (`library::Function` refuses it as
        // `EntryNotFound`).
        FunctionStage::Visible | FunctionStage::Exported | FunctionStage::Extern => {
            return Err(mismatch())
        }
    };
    // A metallib function's bitcode is its own module holding only that entry, its callees, and
    // the globals they need, so it declares exactly one entry point.
    meter.tick(1)?;
    match abi.entries.as_slice() {
        [only] if stage(&only.stage) => {}
        _ => return Err(mismatch()),
    }
    // The declarations sorted by index (decode refused duplicate indices), so each `CNST`
    // entry is one binary search: (index, position, value type, `MTLDataType` so far).
    let mut by_index: Vec<(u32, usize, TypeId, Option<DataType>)> = Vec::new();
    meter.reserve(&mut by_index, abi.function_constants.len())?;
    for (i, d) in abi.function_constants.iter().enumerate() {
        meter.tick(1)?;
        by_index.push((d.index, i, d.ty, d.data_type));
    }
    by_index.sort_unstable_by_key(|&(index, ..)| index);
    for c in &f.public.constants {
        meter.tick(1)?;
        let (_, _, ir, recorded) = by_index
            .binary_search_by_key(&u32::from(c.index), |&(index, ..)| index)
            .ok()
            .and_then(|at| by_index.get_mut(at))
            .ok_or_else(mismatch)?;
        let ty = DataType::try_from(u32::from(c.data_type)).map_err(|_| mismatch())?;
        if recorded.is_some_and(|old| old != ty) || !data_type_is(m, ty, *ir) {
            return Err(mismatch());
        }
        *recorded = Some(ty);
    }
    // Record only after every entry passed, and meter the recording loop up front, so neither a
    // mismatch nor an exhausted budget leaves `abi` partly updated.
    meter.tick(u32::try_from(by_index.len()).unwrap_or(u32::MAX))?;
    for (_, i, _, ty) in by_index {
        if let Some(decl) = abi.function_constants.get_mut(i) {
            decl.data_type = ty;
        }
    }
    Ok(())
}

/// The IR scalar a function constant of an `MTLDataType` family is stored as.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scalar {
    Int(u8),
    Half,
    BFloat,
    Float,
}

/// The stored form of a function constant of type `ty`: its scalar and vector length (1 for a
/// scalar). `None` for a type that is not a scalar or vector (a function constant never is).
///
/// Each `MTLDataType` names an MSL type; the corpus pairs the MSL type (the `!"type"` operand of
/// `!air.function_constants`) with the initializer global's value type. `bool` is stored as `i8`,
/// never `i1`; `char`, `short`, `int`, and `long` and their unsigned forms are `i8` to `i64`, and a
/// vector is an LLVM vector of the scalar.
fn stored_form(ty: DataType) -> Option<(Scalar, u32)> {
    use DataType as D;
    use Scalar::{BFloat, Float, Half, Int};
    Some(match ty {
        D::Bool | D::Char | D::UChar => (Int(8), 1),
        D::Bool2 | D::Char2 | D::UChar2 => (Int(8), 2),
        D::Bool3 | D::Char3 | D::UChar3 => (Int(8), 3),
        D::Bool4 | D::Char4 | D::UChar4 => (Int(8), 4),
        D::Short | D::UShort => (Int(16), 1),
        D::Short2 | D::UShort2 => (Int(16), 2),
        D::Short3 | D::UShort3 => (Int(16), 3),
        D::Short4 | D::UShort4 => (Int(16), 4),
        D::Int | D::UInt => (Int(32), 1),
        D::Int2 | D::UInt2 => (Int(32), 2),
        D::Int3 | D::UInt3 => (Int(32), 3),
        D::Int4 | D::UInt4 => (Int(32), 4),
        D::Long | D::ULong => (Int(64), 1),
        D::Long2 | D::ULong2 => (Int(64), 2),
        D::Long3 | D::ULong3 => (Int(64), 3),
        D::Long4 | D::ULong4 => (Int(64), 4),
        D::Half => (Half, 1),
        D::Half2 => (Half, 2),
        D::Half3 => (Half, 3),
        D::Half4 => (Half, 4),
        D::BFloat => (BFloat, 1),
        D::BFloat2 => (BFloat, 2),
        D::BFloat3 => (BFloat, 3),
        D::BFloat4 => (BFloat, 4),
        D::Float => (Float, 1),
        D::Float2 => (Float, 2),
        D::Float3 => (Float, 3),
        D::Float4 => (Float, 4),
        _ => return None,
    })
}

/// Whether IR type `ir` is the stored form of a function constant of `MTLDataType` `ty`.
pub fn data_type_is(m: &Module, ty: DataType, ir: TypeId) -> bool {
    let Some((scalar, len)) = stored_form(ty) else {
        return false;
    };
    let types = m.types();
    let elem = match types.get(ir) {
        Some(Type::Vector { elem, len: n }) if len > 1 && *n == len => *elem,
        _ if len == 1 => ir,
        _ => return false,
    };
    let found = match types.get(elem) {
        Some(Type::Int(w)) => Scalar::Int(*w),
        Some(Type::Half) => Scalar::Half,
        Some(Type::BFloat) => Scalar::BFloat,
        Some(Type::Float) => Scalar::Float,
        _ => return false,
    };
    found == scalar
}
