//! Module-level named metadata and entry-point nodes.

use super::args::Cx;
use super::md::{abi_error, diag, kept_bytes, kept_diag, node_ops, tok, Ops, Site, Tok};
use super::sampler::decode_global;
use super::structs::Structs;
use super::{
    AirAbi, Builtin, CompileOptions, EntryPoint, FunctionConstantDecl, LanguageVersion,
    LinkageSignature, LinkageSignatureKind, ModuleLimits, ParamRole, PatchInfo, PatchKind, Stage,
    StaticSampler, Version, VisibleFunctionRef, VisibleSymbol, VisibleTarget,
};
use crate::bitcode::PointeeTable;
use crate::budget::Meter;
use crate::error::Error;
use crate::error::{AbiError, DiagName, Feature, InputError, InputErrorKind, Location};
use crate::ir::{Md, MdId, Module, Type, TypeId, TypeLayouts};

/// The closed set of named metadata lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Named {
    Version,
    LanguageVersion,
    CompileOptions,
    Kernel,
    Vertex,
    Fragment,
    Object,
    Mesh,
    FunctionConstants,
    SamplerStates,
    VisibleFunctionReferences,
    Visible,
    CoreImage,
    Signature,
    SourceFileName,
    ModuleFlags,
    Ident,
    CommandLine,
    WorkingDirectory,
    RecordedSources,
    DebugCompileUnits,
}

const NAMED: &[(&str, Named)] = &[
    ("air.version", Named::Version),
    ("air.language_version", Named::LanguageVersion),
    ("air.compile_options", Named::CompileOptions),
    ("air.kernel", Named::Kernel),
    ("air.vertex", Named::Vertex),
    ("air.fragment", Named::Fragment),
    ("air.object", Named::Object),
    ("air.mesh", Named::Mesh),
    ("air.function_constants", Named::FunctionConstants),
    ("air.sampler_states", Named::SamplerStates),
    (
        "air.visible_function_references",
        Named::VisibleFunctionReferences,
    ),
    ("air.visible", Named::Visible),
    (super::core_image::LIST, Named::CoreImage),
    ("air.signature", Named::Signature),
    ("air.source_file_name", Named::SourceFileName),
    ("llvm.module.flags", Named::ModuleFlags),
    ("llvm.ident", Named::Ident),
    ("llvm.commandline", Named::CommandLine),
    ("llvm_utils.workingdir", Named::WorkingDirectory),
    ("llvm_utils.sources", Named::RecordedSources),
    ("llvm.dbg.cu", Named::DebugCompileUnits),
];

/// Apple's named list of a module's program-scope buffer bindings (MSL 3.2): each entry
/// `!{!"air.global_binding", <global>, <binding>}` binds a buffer to a program-scope pointer.
const GLOBAL_BINDINGS: &str = "air.global_bindings";

/// Decode every named list of `m`. Entries are decoded kernel, vertex, fragment, object, mesh,
/// each in list order.
pub(super) fn decode(
    m: &Module,
    pointees: &PointeeTable,
    layouts: Option<&TypeLayouts>,
    meter: &mut Meter,
) -> Result<AirAbi, Error> {
    let mut lists: Vec<(Named, &[MdId])> = Vec::new();
    for named in m.named_md() {
        meter.tick(1)?;
        let name = m.symbols().get(named.name).unwrap_or_default();
        // MSL 3.2's program-scope buffer bindings: valid AIR, not implemented yet, so refused as
        // unsupported rather than as a key outside the input's contract.
        if name == GLOBAL_BINDINGS.as_bytes() {
            return Err(Error::from(InputError::new(
                InputErrorKind::Unsupported(Feature::ProgramScopeBinding),
                Location::metadata(Some(diag(name)), None, None),
            )));
        }
        let kind = NAMED
            .iter()
            .find(|(s, _)| s.as_bytes() == name)
            .map(|&(_, k)| k)
            .ok_or_else(|| {
                abi_error(
                    AbiError::UnknownKey,
                    Location::metadata(Some(diag(name)), None, None),
                )
            })?;
        meter.reserve(&mut lists, 1)?;
        lists.push((kind, &named.ops));
    }
    let mut abi = AirAbi {
        air_version: None,
        language: None,
        compile: CompileOptions::default(),
        limits: ModuleLimits::default(),
        entries: Vec::new(),
        function_constants: Vec::new(),
        static_samplers: Vec::new(),
        visible_function_refs: Vec::new(),
        visible_functions: Vec::new(),
        core_image_kernels: Vec::new(),
        linkage_signature: None,
        structs: Vec::new(),
        producer: None,
    };
    let mut cx = Cx {
        m,
        meter,
        structs: Structs::default(),
        pointees,
        layouts,
    };
    for &(kind, ops) in &lists {
        let site = Site::module();
        match kind {
            Named::Version => abi.air_version = Some(version(m, single(ops, &site)?, &site)?),
            Named::LanguageVersion => {
                let mut n = Ops::node(m, single(ops, &site)?, site.clone())?;
                if n.string(Some("air.language_version"))? != b"Metal" {
                    return Err(site.err(AbiError::BadValue, Some("air.language_version")));
                }
                abi.language = Some(LanguageVersion(Version {
                    major: n.u32(None)?,
                    minor: n.u32(None)?,
                    patch: n.u32(None)?,
                }));
                n.finish()?;
            }
            Named::CompileOptions => abi.compile = compile_options(m, ops, &site, cx.meter)?,
            Named::ModuleFlags => abi.limits = module_flags(m, ops, &site, cx.meter)?,
            Named::Ident => {
                // Only the first producer is kept, so only it is allocated and charged.
                for &id in ops {
                    cx.meter.tick(1)?;
                    let mut n = Ops::node(m, id, site.clone())?;
                    let producer = n.string(Some("llvm.ident"))?;
                    n.finish()?;
                    if abi.producer.is_none() {
                        abi.producer = Some(kept_diag(producer, cx.meter)?);
                    }
                }
            }
            Named::SourceFileName => strings(m, ops, &site, "air.source_file_name", cx.meter)?,
            // `-frecord-sources` (Xcode's `MTL_ENABLE_DEBUG_INFO=INCLUDE_SOURCE`) records the
            // compiler's command line, its working directory, and each source it read as
            // `!{!"<path>", !"<text>"}`. The linker moves them out of the module, so only an
            // unlinked `.air` carries them.
            Named::CommandLine => strings(m, ops, &site, "llvm.commandline", cx.meter)?,
            Named::WorkingDirectory => {
                strings(m, ops, &site, "llvm_utils.workingdir", cx.meter)?;
            }
            Named::RecordedSources => {
                let key = Some("llvm_utils.sources");
                for &id in ops {
                    cx.meter.tick(1)?;
                    let mut files = Ops::node(m, id, site.clone())?;
                    while !files.is_done() {
                        cx.meter.tick(1)?;
                        let mut file = Ops::node(m, files.child(key)?, site.clone())?;
                        file.string(key)?;
                        file.string(key)?;
                        file.finish()?;
                    }
                }
            }
            // Debug info's compile units (`-gline-tables-only`, which Xcode's
            // `MTL_ENABLE_DEBUG_INFO` passes): debug-info records, which carry nothing the
            // translation reads.
            Named::DebugCompileUnits => {
                for &id in ops {
                    cx.meter.tick(1)?;
                    if !matches!(m.md(id), Some(Md::Debug)) {
                        return Err(site.err(AbiError::BadValue, Some("llvm.dbg.cu")));
                    }
                }
            }
            Named::Kernel | Named::Vertex | Named::Fragment | Named::Object | Named::Mesh => {}
            Named::FunctionConstants => {
                cx.meter.reserve(&mut abi.function_constants, ops.len())?;
                for &id in ops {
                    cx.meter.tick(1)?;
                    abi.function_constants
                        .push(function_constant(m, id, &site, cx.meter)?);
                }
            }
            Named::SamplerStates => {
                cx.meter.reserve(&mut abi.static_samplers, ops.len())?;
                for &id in ops {
                    cx.meter.tick(1)?;
                    let key = Some("air.sampler_state");
                    let mut n = Ops::node(m, id, site.clone())?;
                    if n.string(key)? != b"air.sampler_state" {
                        return Err(site.err(AbiError::UnknownKey, key));
                    }
                    let global = n.global(key)?;
                    n.finish()?;
                    let state = decode_global(m, global, &site)?;
                    abi.static_samplers.push(StaticSampler { global, state });
                }
            }
            Named::VisibleFunctionReferences => {
                cx.meter
                    .reserve(&mut abi.visible_function_refs, ops.len())?;
                for &id in ops {
                    cx.meter.tick(1)?;
                    let key = Some("air.visible_function_reference");
                    let mut n = Ops::node(m, id, site.clone())?;
                    if n.string(key)? != b"air.visible_function_reference" {
                        return Err(site.err(AbiError::UnknownKey, key));
                    }
                    let target = match n.next(key)? {
                        Tok::Global(g) => VisibleTarget::Global(g),
                        Tok::Func(f) => VisibleTarget::Function(f),
                        _ => return Err(site.err(AbiError::BadValue, key)),
                    };
                    let symbol = VisibleSymbol(kept_bytes(n.string(key)?, cx.meter)?);
                    n.finish()?;
                    abi.visible_function_refs
                        .push(VisibleFunctionRef { target, symbol });
                }
            }
            Named::Visible => {
                cx.meter.reserve(&mut abi.visible_functions, ops.len())?;
                for &id in ops {
                    abi.visible_functions
                        .push(super::visible::function(m, id, cx.meter)?);
                }
            }
            Named::CoreImage => {
                cx.meter.reserve(&mut abi.core_image_kernels, ops.len())?;
                for &id in ops {
                    abi.core_image_kernels
                        .push(super::core_image::kernel(m, id, cx.meter)?);
                }
            }
            Named::Signature => {
                abi.linkage_signature = Some(signature(m, single(ops, &site)?, &site)?);
            }
        }
    }
    for stage in [
        Named::Kernel,
        Named::Vertex,
        Named::Fragment,
        Named::Object,
        Named::Mesh,
    ] {
        for &(kind, ops) in &lists {
            if kind != stage {
                continue;
            }
            cx.meter.reserve(&mut abi.entries, ops.len())?;
            for &id in ops {
                abi.entries.push(entry(&mut cx, stage, id)?);
            }
        }
    }
    abi.structs = cx.structs.list;
    Ok(abi)
}

/// A list of single-string nodes, which carry nothing the translation reads.
fn strings(
    m: &Module,
    ops: &[MdId],
    site: &Site,
    key: &'static str,
    meter: &mut Meter,
) -> Result<(), Error> {
    for &id in ops {
        meter.tick(1)?;
        let mut n = Ops::node(m, id, site.clone())?;
        n.string(Some(key))?;
        n.finish()?;
    }
    Ok(())
}

/// The one node of a single-node list.
fn single(ops: &[MdId], site: &Site) -> Result<MdId, Error> {
    match ops {
        [one] => Ok(*one),
        _ => Err(site.err(AbiError::BadValue, None)),
    }
}

fn version(m: &Module, id: MdId, site: &Site) -> Result<Version, Error> {
    let key = Some("air.version");
    let mut n = Ops::node(m, id, site.clone())?;
    let v = Version {
        major: n.u32(key)?,
        minor: n.u32(key)?,
        patch: n.u32(key)?,
    };
    n.finish()?;
    Ok(v)
}

/// `!air.compile_options`: a closed set of single-string nodes.
fn compile_options(
    m: &Module,
    ops: &[MdId],
    site: &Site,
    meter: &mut Meter,
) -> Result<CompileOptions, Error> {
    let key = Some("air.compile_options");
    let mut c = CompileOptions::default();
    for &id in ops {
        meter.tick(1)?;
        let mut n = Ops::node(m, id, site.clone())?;
        let option = n.string(key)?;
        n.finish()?;
        let (slot, value) = match option {
            b"air.compile.fast_math_enable" => (&mut c.fast_math, true),
            b"air.compile.fast_math_disable" => (&mut c.fast_math, false),
            b"air.compile.framebuffer_fetch_enable" => (&mut c.framebuffer_fetch, true),
            b"air.compile.framebuffer_fetch_disable" => (&mut c.framebuffer_fetch, false),
            b"air.compile.denorms_disable" if !c.denorms_disable => {
                c.denorms_disable = true;
                continue;
            }
            b"air.compile.denorms_disable" => return Err(site.err(AbiError::BadValue, key)),
            _ => return Err(site.err(AbiError::UnknownKey, key)),
        };
        if slot.replace(value).is_some() {
            return Err(site.err(AbiError::BadValue, key));
        }
    }
    Ok(c)
}

/// An `!air.signature` node: `!{!"<kind>", <operands>...}`. A linked-constants or stitching
/// signature is read by its kind and operand count. `air.rename_qualified_function_signature`
/// (Apple's compiler at macOS 14) is `!{!"<kind>", !<signature>, !"<name>"}`: the module was
/// produced as the linked-constants or stitching signature it wraps describes, and a function
/// renamed to the qualified name. The module's entry lists name its functions themselves, so the
/// wrapper is read as the signature it wraps, and the name is not kept.
fn signature(m: &Module, id: MdId, site: &Site) -> Result<LinkageSignature, Error> {
    let key = Some("air.signature");
    let mut n = Ops::node(m, id, site.clone())?;
    let rename: &[u8] = b"air.rename_qualified_function_signature";
    if matches!(n.peek(), Some(Tok::Str(s)) if s == rename) {
        n.string(key)?;
        let inner = n.child(key)?;
        n.string(key)?;
        n.finish()?;
        return plain_signature(&mut Ops::node(m, inner, site.clone())?, site);
    }
    plain_signature(&mut n, site)
}

/// A linked-constants or stitching signature node's kind and operand count; any other kind
/// (a rename wrapped in another) is unknown.
fn plain_signature(n: &mut Ops<'_>, site: &Site) -> Result<LinkageSignature, Error> {
    let key = Some("air.signature");
    let kind = match n.string(key)? {
        b"air.linked_constants_signature" => LinkageSignatureKind::LinkedConstants,
        b"air.stitching_modules_signature" => LinkageSignatureKind::StitchingModules,
        _ => return Err(site.err(AbiError::UnknownKey, key)),
    };
    let operands =
        u32::try_from(n.len().saturating_sub(1)).map_err(|_| site.err(AbiError::BadValue, key))?;
    Ok(LinkageSignature { kind, operands })
}

/// `!llvm.module.flags`: `!{i32 <behavior>, !"<key>", <value>}`. The `air.max_*` limits are
/// decoded; `wchar_size`, `frame-pointer`, `SDK Version`, and debug info's `Dwarf Version` and
/// `Debug Info Version` are LLVM's and carry nothing the translation reads; `air.workaround.sampler_state_variable_length_encoding` (an `i1`) names the
/// form of the module's constexpr sampler words, which each `air.sampler_state` global's own type
/// states; any other flag is unknown.
fn module_flags(
    m: &Module,
    ops: &[MdId],
    site: &Site,
    meter: &mut Meter,
) -> Result<ModuleLimits, Error> {
    let key = Some("llvm.module.flags");
    let mut l = ModuleLimits::default();
    for &id in ops {
        meter.tick(1)?;
        let mut n = Ops::node(m, id, site.clone())?;
        n.u32(key)?;
        let flag = n.string(key)?;
        let slot = match flag {
            b"air.max_device_buffers" => &mut l.max_device_buffers,
            b"air.max_constant_buffers" => &mut l.max_constant_buffers,
            b"air.max_threadgroup_buffers" => &mut l.max_threadgroup_buffers,
            b"air.max_textures" => &mut l.max_textures,
            b"air.max_read_write_textures" => &mut l.max_read_write_textures,
            b"air.max_samplers" => &mut l.max_samplers,
            b"wchar_size"
            | b"frame-pointer"
            | b"SDK Version"
            | b"Dwarf Version"
            | b"Debug Info Version" => {
                n.next(key)?;
                n.finish()?;
                continue;
            }
            // Apple's metalfe-32023.886 sets it (`i1 true`) on an AIR 2.7 module whose constexpr
            // samplers it writes in AIR 2.6's one-word form (bit 63 set) rather than AIR 2.7's two
            // words. `sampler::decode_global` reads either form from the global's type.
            b"air.workaround.sampler_state_variable_length_encoding" => {
                n.bool(key)?;
                n.finish()?;
                continue;
            }
            _ => return Err(site.err(AbiError::UnknownKey, key)),
        };
        if slot.replace(n.u32(key)?).is_some() {
            return Err(site.err(AbiError::BadValue, key));
        }
        n.finish()?;
    }
    Ok(l)
}

/// `!{ptr @init, !"<type>", !"<name>", i32 <index>[, i1 <flag>]}`.
fn function_constant(
    m: &Module,
    id: MdId,
    site: &Site,
    meter: &mut Meter,
) -> Result<FunctionConstantDecl, Error> {
    let key = Some("air.function_constants");
    let mut n = Ops::node(m, id, site.clone())?;
    let global = n.global(key)?;
    // The `CNST` check reads the initializer's value type; reflection reads the MSL type name.
    let type_name = n.diag(key, meter)?;
    let name = n.diag(key, meter)?;
    let index = n.u32(key)?;
    let flag = if n.is_done() {
        None
    } else {
        Some(n.bool(key)?)
    };
    n.finish()?;
    let ty = m
        .global(global)
        .map(|g| g.value_ty)
        .ok_or_else(|| site.err(AbiError::BadValue, key))?;
    Ok(FunctionConstantDecl {
        index,
        global,
        ty,
        flag,
        data_type: None,
        name,
        type_name,
    })
}

/// An entry node: `!{ptr @fn, !outputs, !inputs, <stage attribute>...}`.
fn entry(cx: &mut Cx<'_, '_>, stage: Named, id: MdId) -> Result<EntryPoint, Error> {
    cx.meter.tick(1)?;
    let m = cx.m;
    let module_site = Site::module();
    let mut n = Ops::node(m, id, module_site.clone())?;
    let func = match n.next(None)? {
        Tok::Func(f) => f,
        _ => return Err(module_site.err(AbiError::BadValue, None)),
    };
    let name = m
        .func(func)
        .and_then(|f| m.symbols().get(f.name))
        .ok_or_else(|| module_site.err(AbiError::BadValue, None))?;
    let name = kept_diag(name, cx.meter)?;
    let site = Site {
        entry: Some(name.clone()),
        arg: None,
    };
    n.site = site.clone();
    let outputs_node = n.child(None)?;
    let inputs_node = n.child(None)?;
    let mut stage = match stage {
        Named::Kernel => Stage::Kernel {
            max_total_threads: None,
        },
        Named::Vertex => Stage::Vertex { patch: None },
        Named::Fragment => Stage::Fragment {
            early_fragment_tests: false,
        },
        Named::Object => Stage::Object {
            max_total_threads: None,
            max_mesh_threadgroups: None,
        },
        Named::Mesh => Stage::Mesh {
            max_total_threads: None,
        },
        _ => return Err(module_site.err(AbiError::BadValue, None)),
    };
    let mut user_annotation = None;
    while !n.is_done() {
        stage_attribute(m, &mut n, &mut stage, &mut user_annotation, &site, cx.meter)?;
    }
    let output_ids = node_ops(m, outputs_node).ok_or_else(|| site.err(AbiError::BadValue, None))?;
    let input_ids = node_ops(m, inputs_node).ok_or_else(|| site.err(AbiError::BadValue, None))?;
    // Kernels, object and mesh functions return nothing: a mesh function's outputs are its
    // `air.mesh` argument.
    if matches!(
        stage,
        Stage::Kernel { .. } | Stage::Object { .. } | Stage::Mesh { .. }
    ) && !output_ids.is_empty()
    {
        return Err(site.err(AbiError::SignatureMismatch, None));
    }
    let mut outputs = Vec::new();
    cx.meter.reserve(&mut outputs, output_ids.len())?;
    for (field, &op) in output_ids.iter().enumerate() {
        let op = op.ok_or_else(|| site.err(AbiError::BadValue, None))?;
        let field = u32::try_from(field).map_err(|_| site.err(AbiError::BadValue, None))?;
        outputs.push(cx.output(op, field, &name, &stage)?);
    }
    // The function's parameter types: a buffer node that states no address space takes its
    // pointer's (`Cx::param`).
    let types: &[TypeId] = match m.func(func).and_then(|f| m.types().get(f.ty)) {
        Some(Type::Function { params, .. }) => params,
        _ => &[],
    };
    let mut params = Vec::new();
    cx.meter.reserve(&mut params, input_ids.len())?;
    for &op in input_ids {
        let op = op.ok_or_else(|| site.err(AbiError::BadValue, None))?;
        params.push(cx.param(op, &name, &stage, types, func)?);
    }
    // Metal's compiler requires `[[early_fragment_tests]]` of a function that reads the coverage
    // after the depth test ("'early_fragment_tests' is required ... by attribute
    // 'post_depth_coverage'"): without it the coverage the input names is not defined.
    let post_depth = params
        .iter()
        .any(|p| p.role == ParamRole::Builtin(Builtin::PostDepthCoverage));
    if post_depth
        && stage
            != (Stage::Fragment {
                early_fragment_tests: true,
            })
    {
        return Err(site.err(AbiError::MissingKey, Some("early_fragment_tests")));
    }
    Ok(EntryPoint {
        func,
        stage,
        params,
        outputs,
        name,
        user_annotation,
    })
}

/// One trailing entry attribute: `!{"air.max_work_group_size", i32 N}` (kernel, object, mesh),
/// `!{"air.max_mesh_work_groups", i32 N}` (object),
/// `!{"air.patch", !"quad"|"triangle"[, "air.patch_control_point", i32 N]}` (vertex),
/// `!"early_fragment_tests"` (fragment), or `!{"air.user_annotation", !"text"}` (any stage, kept
/// in `annotation`, its text charged to `meter`). Any other attribute, or one of these on another
/// stage, is `UnknownStageAttribute` naming its key; an operand that states no key is
/// `UnknownKey`; an attribute stated twice is `BadValue`.
fn stage_attribute(
    m: &Module,
    n: &mut Ops<'_>,
    stage: &mut Stage,
    annotation: &mut Option<DiagName>,
    site: &Site,
    meter: &mut Meter,
) -> Result<(), Error> {
    let attribute = n.next(None)?;
    let key = match attribute {
        Tok::Str(s) => Some(s),
        Tok::Node(id) => node_ops(m, id)
            .and_then(|ops| ops.first().copied())
            .and_then(|op| match tok(m, op) {
                Some(Tok::Str(s)) => Some(s),
                _ => None,
            }),
        _ => None,
    };
    match (attribute, key, &mut *stage) {
        (
            Tok::Str(_),
            Some(b"early_fragment_tests"),
            Stage::Fragment {
                early_fragment_tests,
            },
        ) => {
            if *early_fragment_tests {
                return Err(site.err(AbiError::BadValue, Some("early_fragment_tests")));
            }
            *early_fragment_tests = true;
        }
        (
            Tok::Node(id),
            Some(b"air.max_work_group_size"),
            Stage::Kernel { max_total_threads }
            | Stage::Object {
                max_total_threads, ..
            }
            | Stage::Mesh { max_total_threads },
        ) => {
            let key = Some("air.max_work_group_size");
            let mut a = Ops::node(m, id, site.clone())?;
            a.string(key)?;
            if max_total_threads.replace(a.u32(key)?).is_some() {
                return Err(site.err(AbiError::BadValue, key));
            }
            a.finish()?;
        }
        (
            Tok::Node(id),
            Some(b"air.max_mesh_work_groups"),
            Stage::Object {
                max_mesh_threadgroups,
                ..
            },
        ) => {
            let key = Some("air.max_mesh_work_groups");
            let mut a = Ops::node(m, id, site.clone())?;
            a.string(key)?;
            if max_mesh_threadgroups.replace(a.u32(key)?).is_some() {
                return Err(site.err(AbiError::BadValue, key));
            }
            a.finish()?;
        }
        (Tok::Node(id), Some(b"air.patch"), Stage::Vertex { patch }) => {
            let key = Some("air.patch");
            let mut a = Ops::node(m, id, site.clone())?;
            a.string(key)?;
            let kind = match a.string(key)? {
                b"triangle" => PatchKind::Triangle,
                b"quad" => PatchKind::Quad,
                _ => return Err(site.err(AbiError::BadValue, key)),
            };
            // `[[patch(kind)]]` states no count: the node ends at the kind.
            let control_points = if a.is_done() {
                None
            } else {
                let cp = Some("air.patch_control_point");
                if a.string(cp)? != b"air.patch_control_point" {
                    return Err(site.err(AbiError::MissingKey, cp));
                }
                Some(a.u32(cp)?)
            };
            a.finish()?;
            if patch
                .replace(PatchInfo {
                    kind,
                    control_points,
                })
                .is_some()
            {
                return Err(site.err(AbiError::BadValue, key));
            }
        }
        (Tok::Node(id), Some(b"air.user_annotation"), _) => {
            let key = Some("air.user_annotation");
            let mut a = Ops::node(m, id, site.clone())?;
            a.string(key)?;
            let text = a.diag(key, meter)?;
            a.finish()?;
            if annotation.replace(text).is_some() {
                return Err(site.err(AbiError::BadValue, key));
            }
        }
        (_, Some(key), _) => {
            return Err(abi_error(
                AbiError::UnknownStageAttribute(diag(key)),
                site.at(None),
            ))
        }
        (_, None, _) => return Err(site.err(AbiError::UnknownKey, None)),
    }
    Ok(())
}
