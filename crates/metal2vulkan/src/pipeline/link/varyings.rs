//! `Location` and `Component` assignment of a pipeline's stage interface (lowering spec B9), read
//! from each stage's [`StageIo`] contract only.
//!
//! - **Attributes** (vertex inputs) are numbered densely in Metal attribute order.
//! - **Colors** (fragment outputs) sit at their attachment index (Vulkan's fragment output
//!   location is the color attachment); the module carries the dual-source `Index`.
//! - **Varyings** match across the stages by user key, never by ordinal. The vertex function's
//!   outputs are numbered densely in output order, one location each at component 0, and each
//!   fragment input takes the number of its key; a fragment input without a vertex output of its
//!   key and type is a Metal link error (`UnmatchedVarying`). Unmatched vertex outputs stay (legal
//!   in both APIs). A fragment function alone numbers its inputs densely in declaration order.
//!   The raw viewport index (`IoRole::ViewportIndex`) is such a varying, of its own key, which the
//!   fragment stage reads flat: one more flat input component under the limits below. A fragment
//!   function reads it exactly when compiled for a vertex function that writes
//!   `[[viewport_array_index]]` (`LoweringInputs::viewport_index_varying`); one compiled for the
//!   other kind of vertex function, which would read the clamped `ViewportIndex` or a varying no
//!   stage writes, is `NeedsInput(ViewportIndex)`. The raw render target slice index
//!   (`IoRole::RenderTargetArrayIndex`) is the same for `[[render_target_array_index]]`
//!   (`LoweringInputs::render_target_array_index_varying`, `NeedsInput(RenderTargetArrayIndex)`),
//!   which a vertex function writes in every variant, the one without `Layer` included. The window
//!   depth (`IoRole::WindowDepth`) is such a varying too, read without perspective: a fragment
//!   function that reads it was compiled for a draw whose vertex function writes it
//!   (`LoweringInputs::window_depth_varying`), and one linked with another vertex function is
//!   `NeedsInput(FramebufferFetch)`.
//!   The primitive ID (`IoRole::PrimitiveId`) is such a varying, read flat: a fragment function
//!   compiled for a device that cannot declare a fragment `PrimitiveId`, or for an emulated
//!   post-tessellation vertex function, reads its `[[primitive_id]]` from it, and its pipeline's
//!   vertex function writes it, compiled under `LoweringInputs::primitive_id_varying`
//!   (`NeedsInput(PrimitiveId)` otherwise, and for a fragment function that reads Vulkan's
//!   `PrimitiveId` behind a vertex function that writes the varying). A mesh function writes none,
//!   so a mesh render pipeline whose fragment function reads it is refused as needing
//!   `geometryShader` or `tessellationShader`.
//! - **Barycentric weights** (`IoRole::Barycentric`), which a fragment function compiled for a
//!   device without `fragmentShaderBarycentric` reads and the pipeline's synthesized geometry
//!   stage writes (`link::geometry`), no vertex function: each takes one whole location after
//!   every vertex output's, below `maxFragmentInputComponents / 4`, and is never packed. A
//!   fragment function alone numbers them with its other inputs.
//! - **Mesh outputs**, per vertex and per primitive alike, are numbered densely in output order,
//!   one location each at component 0, within `maxMeshOutputComponents / 4` locations
//!   (`maxVertexOutputComponents / 4` for an emulated mesh function's, which its pipeline's
//!   synthesized vertex stage writes). In a mesh render pipeline each fragment input takes the
//!   number of the mesh output of its key and type (`UnmatchedVarying` without one), below
//!   `maxFragmentInputComponents / 4`; the raw indices follow the vertex rule above. The fragment
//!   function reads per primitive exactly the inputs whose keys the mesh function writes per
//!   primitive (`StageIo::primitive_inputs`), and a render pipeline's fragment function none:
//!   otherwise it was compiled for another pipeline (`NeedsInput(PrimitiveVaryings)`).
//! - **Uniform varyings**. A fragment function reads flat each interpolated varying its vertex
//!   function writes the same at every vertex of a primitive (`StageIo::uniform_inputs`,
//!   `LoweringInputs::uniform_varyings`), and the varying packs with the flat ones. One compiled
//!   for a vertex or mesh function that does not list the key among its `StageIo::uniform_outputs`
//!   was compiled for another pipeline (`NeedsInput(UniformVaryings)`).
//! - **Limits.** The vertex outputs must fit `maxVertexOutputComponents / 4` locations and every
//!   fragment input location must be below `maxFragmentInputComponents / 4`. When the dense
//!   numbering does not fit, link packs as little as the limits need:
//!   1. **unread outputs only**: the varyings the fragment stage reads keep one whole location
//!      each, densely in output order, and the unmatched vertex outputs follow, first-fit by
//!      decreasing width into four-component locations shared only by outputs of one numeric
//!      class. Only the vertex module's unread outputs move to a `Component` other than 0, so
//!      this needs a relocatable `Component` on those alone, and no interpolated input shares a
//!      location;
//!   2. **everything**: when that does not fit either and every varying carries a relocatable
//!      `Component`, the read varyings are packed too, first-fit by decreasing width into
//!      locations shared only by varyings of one numeric class and one interpolation (Vulkan
//!      requires both to agree within a location), then the unread outputs as in 1.
//!
//!   A pipeline that does not fit packed either is refused naming the limit.

use std::collections::{BTreeMap, BTreeSet};

use crate::device::requirements::VkLimit;
use crate::device::DeviceProfile;
use crate::error::{
    Capability, Error, Feature, InternalError, InternalKind, RuntimeInput, Unsupported,
};
use crate::library::FunctionArtifact;
use crate::pipeline::stage_io::{Interpolation, IoNumeric, IoRole, IoType, StageBuiltin, StageIo};
use crate::pipeline::sym::{Stage, Symbol, VaryingKey};

/// Where each key of a pipeline's stage interfaces lives.
pub type Locations = BTreeMap<VaryingKey, (u32, u32)>;

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(
        InternalKind::Invariant,
        format!("stage interface: {detail}"),
    ))
}

fn limit_error(limit: VkLimit, locations: u64, max_components: u32) -> Error {
    Unsupported::needs(
        Feature::InterfaceComponents,
        Capability::Limit {
            limit,
            required: locations.saturating_mul(4),
            available: u64::from(max_components),
        },
    )
    .into()
}

/// The roles each stage may list: vertex attributes in and varyings out, fragment varyings in
/// (with their interpolation) and colors out, nothing for a kernel but its `[[stage_in]]`
/// attributes, which only a kernel has.
fn check_roles(stage: Stage, io: &StageIo) -> Result<(), Error> {
    let ok = match stage {
        // An object function has no stage interface.
        Stage::Kernel | Stage::Object => {
            io.inputs.is_empty()
                && io.outputs.is_empty()
                && io.builtins.is_empty()
                && io.pulled.is_empty()
        }
        // A mesh function's outputs are per vertex or per primitive, a primitive's raw render
        // target slice and viewport indices among them.
        Stage::Mesh => {
            io.inputs.is_empty()
                && io.pulled.is_empty()
                && io.stage_in.is_empty()
                && io.outputs.iter().all(|v| {
                    matches!(
                        v.role,
                        IoRole::Varying {
                            interpolation: None,
                            ..
                        } | IoRole::PrimitiveVarying { .. }
                            | IoRole::ViewportIndex
                            | IoRole::RenderTargetArrayIndex
                    )
                })
        }
        _ if !io.stage_in.is_empty() => false,
        Stage::Vertex => {
            io.inputs
                .iter()
                .all(|v| matches!(v.role, IoRole::Attribute { .. }))
                && io.outputs.iter().all(|v| {
                    matches!(
                        v.role,
                        IoRole::Varying {
                            interpolation: None,
                            ..
                        } | IoRole::ViewportIndex
                            | IoRole::RenderTargetArrayIndex
                            | IoRole::WindowDepth
                            | IoRole::SelectedLayer
                            | IoRole::SelectedViewport
                            | IoRole::PrimitiveId
                    )
                })
        }
        Stage::Fragment => {
            io.pulled.is_empty()
                && io.inputs.iter().all(|v| {
                    matches!(
                        v.role,
                        IoRole::Varying {
                            interpolation: Some(_),
                            ..
                        } | IoRole::ViewportIndex
                            | IoRole::RenderTargetArrayIndex
                            | IoRole::WindowDepth
                            | IoRole::Barycentric { .. }
                            | IoRole::PrimitiveId
                    )
                })
                && io
                    .outputs
                    .iter()
                    .all(|v| matches!(v.role, IoRole::Color { .. }))
        }
    };
    if ok {
        Ok(())
    } else {
        Err(invariant("a role outside its stage"))
    }
}

/// The fragment function `f` reads an index from its raw varying exactly when it was compiled
/// for a vertex or mesh function that writes the index, as `v` does when it declares the varying
/// among its outputs (see the module docs): `[[viewport_array_index]]` and
/// `[[render_target_array_index]]`, each read otherwise from its builtin, or as 0 on a device that
/// cannot declare it (`StageIo::zero_indices`).
fn check_raw_indices(v: &StageIo, f: &StageIo) -> Result<(), Error> {
    for (role, builtin, input) in [
        (
            IoRole::ViewportIndex,
            StageBuiltin::ViewportIndex,
            RuntimeInput::ViewportIndex,
        ),
        (
            IoRole::RenderTargetArrayIndex,
            StageBuiltin::Layer,
            RuntimeInput::RenderTargetArrayIndex,
        ),
    ] {
        let writes = v.outputs.iter().any(|x| x.role == role);
        let reads_varying = f.inputs.iter().any(|x| x.role == role);
        let reads_builtin = f.builtins.contains(&builtin) || f.zero_indices.contains(&builtin);
        if (writes && reads_builtin) || (!writes && reads_varying) {
            return Err(Error::NeedsInput(input));
        }
    }
    // The window depth: compiled for a draw whose vertex function writes it.
    let writes = v.outputs.iter().any(|x| x.role == IoRole::WindowDepth);
    let reads = f.inputs.iter().any(|x| x.role == IoRole::WindowDepth);
    if reads && !writes {
        return Err(Error::NeedsInput(RuntimeInput::FramebufferFetch));
    }
    Ok(())
}

/// The fragment function reads `[[primitive_id]]` from its varying (`IoRole::PrimitiveId`)
/// exactly where the vertex function `vertex` writes it (see the module docs): a vertex function
/// compiled without `LoweringInputs::primitive_id_varying` behind one that reads it, and one that
/// writes it (an emulated post-tessellation vertex function, behind which Vulkan's `PrimitiveId`
/// counts the tessellator's triangles) behind one that reads Vulkan's `PrimitiveId`, are
/// `NeedsInput(PrimitiveId)`; a fragment function linked alone reads it as any other input. A mesh
/// function writes none, and a mesh render pipeline whose fragment function reads the varying is
/// refused as needing a device that declares a fragment `PrimitiveId`.
fn check_primitive_id(
    mesh: Option<&StageIo>,
    vertex: Option<&StageIo>,
    f: &StageIo,
) -> Result<(), Error> {
    let role = |io: &StageIo, outputs: bool| {
        let list = match outputs {
            true => &io.outputs,
            false => &io.inputs,
        };
        list.iter().any(|x| x.role == IoRole::PrimitiveId)
    };
    if vertex.is_some_and(|v| role(v, true)) && f.builtins.contains(&StageBuiltin::PrimitiveId) {
        return Err(Error::NeedsInput(RuntimeInput::PrimitiveId));
    }
    if !role(f, false) {
        return Ok(());
    }
    if mesh.is_some() {
        return Err(Unsupported::needs_any_of(
            Feature::PrimitiveId,
            crate::spirv::requirements::builtin_enablers(
                crate::spv::BuiltIn::PrimitiveId,
                crate::spirv::ir::IoDir::Input,
            )
            .iter()
            .map(|&(_, feature)| vec![Capability::Feature(feature)])
            .collect(),
        )
        .into());
    }
    // A fragment function linked alone numbers the input with its others.
    match vertex {
        Some(v) if !role(v, true) => Err(Error::NeedsInput(RuntimeInput::PrimitiveId)),
        _ => Ok(()),
    }
}

/// The fragment function reads flat only interpolated varyings that its vertex function `g`
/// writes the same at every vertex of a primitive (see the module docs); a mesh function lists
/// none.
fn check_uniform_inputs(g: &StageIo, f: &StageIo) -> Result<(), Error> {
    if f.uniform_inputs
        .iter()
        .all(|k| g.uniform_outputs.binary_search(k).is_ok())
    {
        Ok(())
    } else {
        Err(Error::NeedsInput(RuntimeInput::UniformVaryings))
    }
}

/// The fragment function reads per primitive exactly the inputs of the keys its mesh function
/// writes per primitive, and a fragment function without a mesh function none (see the module
/// docs).
fn check_primitive_inputs(mesh: Option<&StageIo>, f: &StageIo) -> Result<(), Error> {
    let per_primitive: BTreeSet<&VaryingKey> = mesh
        .into_iter()
        .flat_map(|m| &m.outputs)
        .filter(|x| {
            matches!(
                x.role,
                IoRole::PrimitiveVarying { .. }
                    | IoRole::ViewportIndex
                    | IoRole::RenderTargetArrayIndex
            )
        })
        .map(|x| &x.key)
        .collect();
    let expected: BTreeSet<&VaryingKey> = f
        .inputs
        .iter()
        .map(|x| &x.key)
        .filter(|k| per_primitive.contains(k))
        .collect();
    if f.primitive_inputs.iter().collect::<BTreeSet<_>>() != expected {
        return Err(Error::NeedsInput(RuntimeInput::PrimitiveVaryings));
    }
    Ok(())
}

/// A mesh render pipeline's fragment inputs: each at the location of the mesh output of its key
/// and type, below `maxFragmentInputComponents / 4`; its colors at their attachments.
fn mesh_fragment(
    m: &StageIo,
    f: &StageIo,
    table: &mut Locations,
    p: &DeviceProfile,
) -> Result<(), Error> {
    let l = &p.fields().limits;
    let mut end = 0u64;
    for x in &f.inputs {
        let at = match m.outputs.iter().find(|o| o.key == x.key) {
            Some(o) if o.ty == x.ty => table
                .get(&x.key)
                .copied()
                .ok_or_else(|| invariant("a mesh output without its location"))?,
            _ => return Err(Unsupported::not_representable(Feature::UnmatchedVarying).into()),
        };
        end = end.max(u64::from(at.0) + 1);
    }
    if end.saturating_mul(4) > u64::from(l.max_fragment_input_components) {
        return Err(limit_error(
            VkLimit::MaxFragmentInputComponents,
            end,
            l.max_fragment_input_components,
        ));
    }
    for x in &f.outputs {
        if let IoRole::Color { attachment, .. } = x.role {
            table.insert(x.key.clone(), (attachment, 0));
        }
    }
    Ok(())
}

/// One user varying of the pipeline.
struct Varying<'a> {
    key: &'a VaryingKey,
    components: u8,
    numeric: IoNumeric,
    /// The fragment stage reads it (with this interpolation).
    read: Option<Interpolation>,
}

/// The location and component of every key of `stages`' interfaces (see the module docs).
pub fn assign(stages: &[&FunctionArtifact], p: &DeviceProfile) -> Result<Locations, Error> {
    let mut vertex: Option<&StageIo> = None;
    let mut fragment: Option<&StageIo> = None;
    let mut mesh: Option<&StageIo> = None;
    let mut emulated_mesh = false;
    for a in stages {
        let io = &a.interface.io;
        check_roles(a.interface.stage, io)?;
        match a.interface.stage {
            Stage::Vertex => vertex = Some(io),
            Stage::Fragment => fragment = Some(io),
            Stage::Mesh => {
                mesh = Some(io);
                emulated_mesh = a.interface.mesh_records.is_some();
            }
            Stage::Kernel | Stage::Object => {}
        }
    }
    if let Some(f) = fragment {
        if let Some(g) = mesh.or(vertex) {
            check_raw_indices(g, f)?;
            check_uniform_inputs(g, f)?;
        }
        check_primitive_inputs(mesh, f)?;
        check_primitive_id(mesh, vertex, f)?;
    }
    let l = &p.fields().limits;
    let mut table = Locations::new();
    if let Some(m) = mesh {
        let need = m.outputs.len() as u64;
        let (limit, available) = match emulated_mesh {
            true => (
                VkLimit::MaxVertexOutputComponents,
                l.max_vertex_output_components,
            ),
            false => (
                VkLimit::MaxMeshOutputComponents,
                l.mesh.max_mesh_output_components,
            ),
        };
        if need.saturating_mul(4) > u64::from(available) {
            return Err(limit_error(limit, need, available));
        }
        for (n, x) in m.outputs.iter().enumerate() {
            table.insert(x.key.clone(), (n as u32, 0));
        }
        if let Some(f) = fragment {
            mesh_fragment(m, f, &mut table, p)?;
        }
        return Ok(table);
    }
    if let Some(v) = vertex {
        let mut attrs: Vec<(u32, &VaryingKey)> = v
            .inputs
            .iter()
            .filter_map(|x| match x.role {
                IoRole::Attribute { index, .. } => Some((index, &x.key)),
                _ => None,
            })
            .collect();
        attrs.sort();
        if attrs.len() as u64 > u64::from(l.max_vertex_input_attributes) {
            return Err(Unsupported::needs(
                Feature::InterfaceComponents,
                Capability::Limit {
                    limit: VkLimit::MaxVertexInputAttributes,
                    required: attrs.len() as u64,
                    available: u64::from(l.max_vertex_input_attributes),
                },
            )
            .into());
        }
        for (n, (_, key)) in attrs.into_iter().enumerate() {
            table.insert(key.clone(), (n as u32, 0));
        }
    }
    if let Some(f) = fragment {
        for x in &f.outputs {
            if let IoRole::Color { attachment, .. } = x.role {
                table.insert(x.key.clone(), (attachment, 0));
            }
        }
    }
    let varyings = varyings(vertex, fragment)?;
    let dense: Vec<(u32, u32)> = (0..varyings.len()).map(|i| (i as u32, 0)).collect();
    let placed = match check_limits(&varyings, &dense, vertex.is_some(), p) {
        Ok(()) => dense,
        Err(e) => {
            let unread = varyings.iter().filter(|v| v.read.is_none());
            let outputs_only = pack(&varyings, Pack::UnreadOnly);
            if packable(stages, unread.map(|v| v.key))
                && check_limits(&varyings, &outputs_only, vertex.is_some(), p).is_ok()
            {
                outputs_only
            } else if packable(stages, varyings.iter().map(|v| v.key)) {
                let packed = pack(&varyings, Pack::Everything);
                check_limits(&varyings, &packed, vertex.is_some(), p)?;
                packed
            } else {
                return Err(e);
            }
        }
    };
    let mut next = placed.iter().map(|at| at.0 + 1).max().unwrap_or(0);
    for (v, at) in varyings.iter().zip(placed) {
        table.insert(v.key.clone(), at);
    }
    // The barycentric weights a synthesized geometry stage writes: whole locations after the
    // vertex outputs', which a fragment function alone numbered with its other inputs.
    if vertex.is_some() {
        let mut weights = false;
        for x in fragment.into_iter().flat_map(|f| &f.inputs) {
            if matches!(x.role, IoRole::Barycentric { .. }) {
                table.insert(x.key.clone(), (next, 0));
                next += 1;
                weights = true;
            }
        }
        if weights && u64::from(next).saturating_mul(4) > u64::from(l.max_fragment_input_components)
        {
            return Err(limit_error(
                VkLimit::MaxFragmentInputComponents,
                u64::from(next),
                l.max_fragment_input_components,
            ));
        }
    }
    Ok(table)
}

/// The pipeline's varyings in numbering order: the vertex outputs in output order (each marked
/// with the interpolation of the fragment input that reads it), or a fragment function's inputs
/// alone in declaration order.
fn varyings<'a>(
    vertex: Option<&'a StageIo>,
    fragment: Option<&'a StageIo>,
) -> Result<Vec<Varying<'a>>, Error> {
    let interpolation = |x: &crate::pipeline::stage_io::IoVariable| match &x.role {
        IoRole::Varying { interpolation, .. } => *interpolation,
        // A fragment input: a raw index is read flat. A selected index, which only the
        // synthesized geometry stage reads, is a flat integer too.
        IoRole::ViewportIndex
        | IoRole::RenderTargetArrayIndex
        | IoRole::SelectedLayer
        | IoRole::SelectedViewport
        | IoRole::PrimitiveId => Some(Interpolation::Flat),
        // The window depth is read without perspective, at the centre and at a sample.
        IoRole::WindowDepth => Some(Interpolation::Smooth {
            sampling: crate::pipeline::stage_io::Sampling::Center,
            perspective: false,
        }),
        IoRole::Barycentric { perspective } => Some(Interpolation::Smooth {
            sampling: crate::pipeline::stage_io::Sampling::Center,
            perspective: *perspective,
        }),
        IoRole::Attribute { .. } | IoRole::Color { .. } | IoRole::PrimitiveVarying { .. } => None,
    };
    // The barycentric weights are the synthesized geometry stage's, no vertex output's.
    let reads: BTreeMap<&VaryingKey, (IoType, Option<Interpolation>)> = fragment
        .map(|f| {
            f.inputs
                .iter()
                .filter(|x| !matches!(x.role, IoRole::Barycentric { .. }))
                .map(|x| (&x.key, (x.ty, interpolation(x))))
                .collect()
        })
        .unwrap_or_default();
    let Some(v) = vertex else {
        return Ok(fragment
            .map(|f| {
                f.inputs
                    .iter()
                    .map(|x| Varying {
                        key: &x.key,
                        components: x.ty.components,
                        numeric: x.ty.numeric,
                        read: interpolation(x),
                    })
                    .collect()
            })
            .unwrap_or_default());
    };
    let mut out = Vec::with_capacity(v.outputs.len());
    let mut matched = BTreeSet::new();
    for x in &v.outputs {
        let read = match reads.get(&x.key) {
            Some((ty, interp)) if *ty == x.ty => {
                matched.insert(&x.key);
                *interp
            }
            Some(_) => return Err(Unsupported::not_representable(Feature::UnmatchedVarying).into()),
            None => None,
        };
        out.push(Varying {
            key: &x.key,
            components: x.ty.components,
            numeric: x.ty.numeric,
            read,
        });
    }
    if matched.len() != reads.len() {
        return Err(Unsupported::not_representable(Feature::UnmatchedVarying).into());
    }
    Ok(out)
}

/// Every varying of `keys` carries a relocatable `Component` in every stage that declares it, so
/// link may place it at a component other than 0.
fn packable<'k>(stages: &[&FunctionArtifact], keys: impl Iterator<Item = &'k VaryingKey>) -> bool {
    let keys: BTreeSet<&VaryingKey> = keys.collect();
    stages.iter().all(|a| {
        let io = &a.interface.io;
        let declared: BTreeSet<&VaryingKey> = io
            .variables()
            .filter(|x| x.role.is_varying())
            .map(|x| &x.key)
            .collect();
        let with_component: BTreeSet<&VaryingKey> = a
            .module
            .relocs
            .iter()
            .filter_map(|(s, _)| match s {
                Symbol::Component(k) => Some(k),
                _ => None,
            })
            .collect();
        declared
            .iter()
            .all(|k| !keys.contains(k) || with_component.contains(k))
    })
}

/// The vertex output and fragment input location limits for `placed` (one `(location,
/// component)` per varying).
fn check_limits(
    varyings: &[Varying<'_>],
    placed: &[(u32, u32)],
    has_vertex: bool,
    p: &DeviceProfile,
) -> Result<(), Error> {
    let l = &p.fields().limits;
    let end = |read_only: bool| -> u64 {
        varyings
            .iter()
            .zip(placed)
            .filter(|(v, _)| !read_only || v.read.is_some())
            .map(|(_, at)| u64::from(at.0) + 1)
            .max()
            .unwrap_or(0)
    };
    if has_vertex {
        let need = end(false);
        if need.saturating_mul(4) > u64::from(l.max_vertex_output_components) {
            return Err(limit_error(
                VkLimit::MaxVertexOutputComponents,
                need,
                l.max_vertex_output_components,
            ));
        }
    }
    let need = end(true);
    if need.saturating_mul(4) > u64::from(l.max_fragment_input_components) {
        return Err(limit_error(
            VkLimit::MaxFragmentInputComponents,
            need,
            l.max_fragment_input_components,
        ));
    }
    Ok(())
}

/// How much [`pack`] packs (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pack {
    /// The read varyings keep one whole location each; only the unread outputs share.
    UnreadOnly,
    /// Every varying shares locations with its numeric class and interpolation.
    Everything,
}

/// The packed placement of `varyings` (see the module docs), one `(location, component)` each.
fn pack(varyings: &[Varying<'_>], how: Pack) -> Vec<(u32, u32)> {
    // (numeric, interpolation, next free component) of each location, by location.
    let mut bins: Vec<(IoNumeric, Option<Interpolation>, u32)> = Vec::new();
    let mut placed = vec![(0, 0); varyings.len()];
    if how == Pack::UnreadOnly {
        // Whole locations: nothing else fits in a bin whose next free component is 4.
        for (i, v) in varyings.iter().enumerate() {
            if v.read.is_some() {
                placed[i] = (bins.len() as u32, 0);
                bins.push((v.numeric, v.read, 4));
            }
        }
    }
    for read in [true, false] {
        if read && how == Pack::UnreadOnly {
            continue;
        }
        let mut group: Vec<usize> = (0..varyings.len())
            .filter(|&i| varyings[i].read.is_some() == read)
            .collect();
        // Stable: equal widths keep their numbering order.
        group.sort_by_key(|&i| std::cmp::Reverse(varyings[i].components));
        for i in group {
            let v = &varyings[i];
            let width = u32::from(v.components);
            let fit = bins.iter().position(|(numeric, interp, used)| {
                *numeric == v.numeric && *interp == v.read && used + width <= 4
            });
            let at = match fit {
                Some(b) => b,
                None => {
                    bins.push((v.numeric, v.read, 0));
                    bins.len() - 1
                }
            };
            placed[i] = (at as u32, bins[at].2);
            bins[at].2 += width;
        }
    }
    placed
}
