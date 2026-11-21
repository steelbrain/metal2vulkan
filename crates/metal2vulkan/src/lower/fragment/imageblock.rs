//! Fragment imageblocks: `[[imageblock_data]]` inputs and outputs of a fragment function.
//!
//! **What Metal gives.** A render pass keeps an imageblock in tile memory: one cell per pixel for
//! the whole pass, of the explicit layout's `air.imageblock_data_size` bytes, undefined when the
//! pass begins, and shared by every pipeline of the pass. A fragment function reads a *view* of
//! its pixel's cell as a struct parameter (`air.imageblock_data` input) and writes a view as a
//! member of its returned struct (`air.imageblock_data` output). A view's members name members of
//! the *master* layout (`air.imageblock_master`; the view's own layout when it has none) by their
//! user key, the fifth string of each member tuple (`Member::key`; MSL §4.7.1: `[[user(name)]]`,
//! else the declared name, "must match a data member declared in the master"), and each member
//! lives at its master offset. The master's fingerprint (SHA-256 of its data size and its
//! layout's Merkle digest over each member's offset, size, array length, key, and nested
//! layout's digest, each shared layout hashed once) is the fact's and
//! `FragmentImageblock::master`: a consumer draws only pipelines of one master in a pass.
//! Explicit imageblock layouts cannot alias color attachments (no `[[color(n)]]` member, MSL
//! §4.7), and in a fragment function every master member names a raster order group, which
//! orders the accesses of overlapping fragments in primitive order.
//!
//! **The representation** (`ImageblockMechanism::StorageImageInterlock`). The pass's imageblock
//! is one coherent `R32Uint` 2D-array storage image, `SyntheticBinding::Imageblock`: layer `w`
//! holds word `w` of every pixel's cell, the cell of pixel `(x, y)` at texel `(x, y)`, which the
//! function takes from `FragCoord`. The input view is read once, member by member, in the
//! function's first block ([`emit_entry`]); the output view is written member by member at the
//! return ([`emit_return`]), each member a leaf of the cell's words (`memory::leaf`, a sub-word
//! member replaces only its bytes). The whole body runs inside the ordered pixel-interlock critical
//! section (`fragment::rog`), so the read, the body, and the write of each fragment are one step in
//! primitive order against every other fragment of the pixel, across draws, and its accesses'
//! device-scope coherence (the `Coherent` image under `GLSL450`, `MakeTexel*` under the Vulkan
//! model: `sync::memory_model`) makes each critical section's writes visible to the next one. That
//! is Metal's raster order group ordering, and more (groups only relax it).
//!
//! A discarded fragment writes nothing: a demoted invocation's image writes have no effect, a
//! terminate-form discard returns before the write, and a predicated one skips the write
//! (`discarded`). Nor does a fragment whose returned `[[sample_mask]]` does not cover its one
//! sample (bit 0 clear; the mask's other bits do not keep it): Metal drops the fragment and its
//! imageblock write with it (measured on the M3 Max and an M4), so the write is guarded by the
//! returned mask's bit 0 ([`ImageblockPlan::mask`]).
//!
//! **Two masters.** The input and output views may view different masters: each view's members
//! live at its own master's offsets of the one cell, which spans the largest master (measured on
//! the M3 Max and an M4, a `half` read at byte 0 of one master and written through a smaller master
//! that keys it at byte 2, and back, each reached those bytes and kept the rest). The fact's master
//! is the largest's.
//!
//! **Refused.** A profile without `fragmentShaderPixelInterlock` is
//! `Unsupported{Imageblocks, Feature(FragmentShaderPixelInterlock)}`; an output without
//! `fragmentStoresAndAtomics` names that feature. Not implemented: a second input or output view,
//! an output aliasing the implicit imageblock (`air.alias_implicit_imageblock`: it is the color
//! attachments' storage; an aliased input never reaches this family, `select::param_family`
//! refuses it as `TileShading`), a member that is an array or a struct, names a color
//! attachment, or whose value is wider than its slot, a view member without exactly one master
//! member of its size, a view struct field no member covers, and a view without a data size.
//! Link refuses what the one plane does not
//! model (multisampling, and a write that alpha to coverage, or a late depth or stencil test, can
//! drop: `pipeline::link::imageblock`), and the binder a layered or amplified pass. A packed
//! pixel-format member (`_pixel_type`) is refused before this family plans it, by its conversion
//! (`Lowering(Arith)` or `UnknownIntrinsic`).

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use crate::air::abi::{ImageblockArg, ImageblockKind, Member, OutputRole, StructInfo, StructRef};
use crate::air::container::Hash256;
use crate::air::ir::{ArgIdx, Type, TypeId as AirTypeId, Value};
use crate::device::requirements::VkFeature;
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::stage_io::{self, Returned};
use crate::lower::sync::memory_model;
use crate::memory::leaf::{self, k32, Coherence, Leaf, WordMem};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, SyntheticBinding};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, GlobalId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConvOp, Dim, ImageOperands, ImageType, ImageUsage, MemAccess, StorageClass,
    Type as SType, UnOp,
};

fn not_implemented() -> Error {
    Unsupported::not_implemented(Feature::Imageblocks).into()
}

/// The device capabilities a function that reads (and, when `writes`, writes) the imageblock
/// needs: the ordered pixel interlock, and stores in the fragment stage for a write. A missing
/// one refuses the function naming every missing one.
fn require(cx: &SelectCx<'_>, writes: bool) -> Result<(), Error> {
    let f = cx.profile.fields();
    let mut missing = Vec::new();
    if !super::fetch::interlock_forms(cx.profile) {
        missing.push(Capability::Feature(VkFeature::FragmentShaderPixelInterlock));
    }
    if writes && !f.leaf.fragment_stores_and_atomics {
        missing.push(Capability::Feature(VkFeature::FragmentStoresAndAtomics));
    }
    if missing.is_empty() {
        return Ok(());
    }
    Err(Unsupported::needs_any_of(Feature::Imageblocks, vec![missing]).into())
}

/// One member of a view: the view struct's field that holds it, its leaf, and its byte offset
/// in the cell (the master layout's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewMember {
    pub field: u32,
    pub leaf: Leaf,
    pub offset: u32,
}

/// A view of the imageblock: the members of its struct value, and the struct's lowered type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct View {
    pub members: Vec<ViewMember>,
    pub ty: DemoteTy,
}

/// The fragment imageblock part of the fragment family's plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImageblockPlan {
    /// The 32-bit words of a cell the views reach: the largest data size, rounded up to a word.
    pub words: u32,
    /// The bytes of a cell (`air.imageblock_data_size`), the largest over the views.
    pub data_size: u32,
    /// The alignment Metal states for the views (`air.arg_type_align_size`, their master
    /// layout's), the largest: what Metal's pipeline aligns the color attachments' bytes beside
    /// the cell to (`FragmentImageblock::alignment`).
    pub alignment: u32,
    /// The input view and the LLVM parameter that holds it.
    pub input: Option<(u32, View)>,
    /// The output view and the returned struct's field that holds it.
    pub output: Option<(u32, View)>,
    /// The stage interface's `FragCoord` object, which gives the pixel.
    pub coord: usize,
    /// The fingerprint of the master layout of the most bytes the function's views view
    /// (`fingerprint`; the first planned of equal size).
    pub master: Option<Hash256>,
    /// The returned struct's field holding the `[[sample_mask]]` the function returns beside an
    /// output view: the write is kept only where the mask covers the invocation's sample.
    pub mask: Option<u32>,
    /// The stage interface's `SampleId` object of a function compiled for a multisampled
    /// pipeline (`LoweringInputs::multisampled`), which runs once per covered sample and reaches
    /// its own sample's cell: words `sample * words` on.
    pub sample: Option<usize>,
    /// The module declares early fragment tests, which the function does not, so that a depth or
    /// stencil test the fragment fails drops its write as Metal's does ([`plan_tests`]).
    pub early_tests: bool,
}

/// The alignment a constant byte offset guarantees (at most 16).
fn offset_align(offset: u32) -> u64 {
    1u64 << offset.trailing_zeros().min(4)
}

/// The layout of struct reference `r`.
fn layout<'a>(cx: &'a SelectCx<'_>, r: StructRef) -> Result<&'a StructInfo, Error> {
    cx.entry
        .structs
        .get(r.0 as usize)
        .ok_or_else(|| cx.invariant("an imageblock layout out of range"))
}

/// The master member view member `m` names: the one whose user key equals `m`'s (MSL §4.7.1,
/// the Metal view-matching ABI), which must be unique and of `m`'s size.
fn master_member<'a>(master: &'a StructInfo, m: &Member) -> Result<&'a Member, Error> {
    let mut named = master.members.iter().filter(|x| x.key == m.key);
    match (named.next(), named.next()) {
        (Some(x), None) if x.size == m.size && x.array_len == 0 && x.layout.is_none() => Ok(x),
        _ => Err(not_implemented()),
    }
}

/// The fingerprint of master layout `master` of a cell of `data_size` bytes, over the module's
/// layouts `structs`, and how many layouts it hashed; `None` when a layout reference is out of
/// range or the layouts nest in a cycle.
///
/// It is SHA-256 over the data size and the master's Merkle digest: a layout's digest is SHA-256
/// over its member count and, in declaration order, each member's offset, size, array length,
/// and user key, and a nested member's layout digest. Decoded layouts are shared by reference (a
/// DAG), so each is hashed once (`memo`): the work is linear in the distinct layouts the master
/// reaches, however deep they nest. Equal fingerprints are structurally equal masters, whatever
/// metadata nodes state them.
fn fingerprint(
    structs: &[StructInfo],
    master: StructRef,
    data_size: u32,
) -> Option<(Hash256, usize)> {
    let mut memo: BTreeMap<StructRef, Hash256> = BTreeMap::new();
    // Each frame is a layout and its first member whose nested layout is not hashed yet.
    let mut stack = vec![(master, 0usize)];
    while let Some(&(r, next)) = stack.last() {
        let s = structs.get(usize::try_from(r.0).ok()?)?;
        let pending = s
            .members
            .iter()
            .enumerate()
            .skip(next)
            .find_map(|(i, m)| m.layout.filter(|l| !memo.contains_key(l)).map(|l| (i, l)));
        if let Some((at, child)) = pending {
            // A layout on the stack twice nests in itself.
            if stack.len() > structs.len() {
                return None;
            }
            if let Some(top) = stack.last_mut() {
                top.1 = at;
            }
            stack.push((child, 0));
            continue;
        }
        stack.pop();
        if memo.contains_key(&r) {
            continue;
        }
        let mut h = Sha256::new();
        h.update((s.members.len() as u64).to_le_bytes());
        for m in &s.members {
            for n in [m.offset, m.size, m.array_len] {
                h.update(n.to_le_bytes());
            }
            h.update((m.key.0.len() as u64).to_le_bytes());
            h.update(&m.key.0);
            match m.layout.and_then(|l| memo.get(&l)) {
                Some(d) => {
                    h.update([1]);
                    h.update(d);
                }
                None => h.update([0]),
            }
        }
        memo.insert(r, h.finalize().into());
    }
    let mut h = Sha256::new();
    h.update(data_size.to_le_bytes());
    h.update(memo.get(&master)?);
    Some((h.finalize().into(), memo.len()))
}

/// The cell a view reaches: its master's data size, the alignment Metal states for the view
/// (`air.arg_type_align_size`: its master layout's, a view of part of a master included), and
/// the master's `fingerprint`.
#[derive(Clone, Copy)]
struct Reached {
    data_size: u32,
    alignment: u32,
    master: Hash256,
}

/// The view of AIR struct type `ty` that imageblock argument `ib` of stated alignment `align`
/// (`air.arg_type_align_size`) describes, and the cell it reaches. Metal's compiler states the
/// alignment of every view, its master layout's; a view that states none (hand-written AIR)
/// takes its own layout's ABI alignment, which is the master's where the view is its master.
/// A stated alignment that is not a power of two is not implemented.
fn view(
    cx: &SelectCx<'_>,
    ty: AirTypeId,
    ib: &ImageblockArg,
    align: Option<u32>,
) -> Result<(View, Reached), Error> {
    if ib.kind != ImageblockKind::Data {
        return Err(cx.invariant("a fragment imageblock of another kind"));
    }
    // An imageblock aliasing the implicit one is the color attachments' own storage, which the
    // side image is not.
    if ib.alias_implicit {
        return Err(not_implemented());
    }
    let data_size = ib
        .data_size
        .filter(|n| *n > 0)
        .ok_or_else(not_implemented)?;
    let own = ib.layout.ok_or_else(not_implemented)?;
    let view_layout = layout(cx, own)?;
    let master_ref = ib.master.unwrap_or(own);
    let master = layout(cx, master_ref)?;
    // An explicit layout holds no color attachment (MSL §4.7); a member that names one is the
    // implicit imageblock's, which the side image is not.
    let color = |s: &StructInfo| s.members.iter().any(|m| m.render_target.is_some());
    if color(view_layout) || color(master) {
        return Err(not_implemented());
    }
    let fields = match cx.entry.module.types().get(ty) {
        Some(Type::Struct {
            fields,
            packed: false,
            ..
        }) => fields.clone(),
        _ => return Err(not_implemented()),
    };
    let offsets = cx
        .memory
        .layouts
        .field_offsets(ty)
        .ok_or_else(not_implemented)?
        .to_vec();
    if view_layout.members.len() != fields.len() {
        return Err(not_implemented());
    }
    let mut members = Vec::with_capacity(fields.len());
    for m in &view_layout.members {
        if m.array_len != 0 || m.layout.is_some() {
            return Err(not_implemented());
        }
        // The view member is the struct field at its offset in the view.
        let field = offsets
            .iter()
            .position(|o| *o == u64::from(m.offset))
            .ok_or_else(not_implemented)?;
        if members
            .iter()
            .any(|x: &ViewMember| x.field as usize == field)
        {
            return Err(not_implemented());
        }
        let field_ty = *fields.get(field).ok_or_else(not_implemented)?;
        let at = master_member(master, m)?;
        let leaf = leaf::leaf_of(
            cx.entry,
            cx.profile,
            &cx.memory.layouts,
            field_ty,
            offset_align(at.offset),
            offset_align(at.offset),
        )?;
        // A value wider than its slot is a packed pixel-format member, which Metal converts.
        if leaf.bytes > at.size
            || at
                .offset
                .checked_add(leaf.bytes)
                .is_none_or(|e| e > data_size)
        {
            return Err(not_implemented());
        }
        members.push(ViewMember {
            field: field as u32,
            leaf,
            offset: at.offset,
        });
    }
    let alignment = match align {
        Some(stated) => stated,
        None => cx
            .memory
            .layouts
            .abi_align(ty)
            .and_then(|a| u32::try_from(a).ok())
            .ok_or_else(not_implemented)?,
    };
    if !alignment.is_power_of_two() {
        return Err(not_implemented());
    }
    let lowered = crate::lower::arith::lowered_type(cx.entry, cx.profile, ty)?;
    let (master, _) = fingerprint(&cx.entry.structs, master_ref, data_size)
        .ok_or_else(|| cx.invariant("an imageblock master whose layouts do not resolve"))?;
    let cell = Reached {
        data_size,
        alignment,
        master,
    };
    Ok((
        View {
            members,
            ty: lowered,
        },
        cell,
    ))
}

/// Plan the imageblock storage the function reaches with a view of `cell`. Views of structurally
/// different masters each reach their own master's offsets of the one cell, which spans the
/// largest of them (measured on the M3 Max and an M4).
fn reach(cx: &mut SelectCx<'_>, cell: Reached) -> Result<(), Error> {
    if cx.stage != Stage::Fragment {
        return Err(cx.invariant("a fragment imageblock outside a fragment function"));
    }
    let known = cx.families.fragment.imageblock.is_some();
    let coord = stage_io::frag_coord(cx);
    // Metal keeps a multisampled pass's imageblock per sample, and runs a function that reaches
    // it once per covered sample, each run its own sample's cell (measured on the M3 Max).
    let sample = if cx.inputs.multisampled {
        if !cx.profile.fields().fragment.sample_rate_shading {
            return Err(Unsupported::needs(
                Feature::Imageblocks,
                Capability::Feature(VkFeature::SampleRateShading),
            )
            .into());
        }
        Some(stage_io::sample_id(cx))
    } else {
        None
    };
    let ib = cx
        .families
        .fragment
        .imageblock
        .get_or_insert_with(ImageblockPlan::default);
    if ib.master.is_none() || cell.data_size > ib.data_size {
        ib.master = Some(cell.master);
    }
    ib.coord = coord;
    ib.sample = sample;
    ib.words = ib.words.max(cell.data_size.div_ceil(4));
    ib.data_size = ib.data_size.max(cell.data_size);
    ib.alignment = ib.alignment.max(cell.alignment);
    if !known {
        cx.descriptors.storage_images = cx.descriptors.storage_images.saturating_add(1);
    }
    Ok(())
}

/// Plan the imageblock input parameter `param` of role `ib` and stated alignment `align`.
pub(super) fn select_input(
    cx: &mut SelectCx<'_>,
    param: u32,
    ib: &ImageblockArg,
    align: Option<u32>,
) -> Result<(), Error> {
    let v = Value::Arg(ArgIdx::new(param));
    let ty = cx
        .value_ty(v)
        .ok_or_else(|| cx.invariant("an imageblock parameter without a type"))?;
    let (view, cell) = view(cx, ty, ib, align)?;
    require(cx, false)?;
    reach(cx, cell)?;
    let plan = cx
        .families
        .fragment
        .imageblock
        .as_mut()
        .ok_or_else(|| invariant_error("an imageblock without its plan"))?;
    if plan.input.is_some() {
        return Err(not_implemented());
    }
    plan.input = Some((param, view));
    Ok(())
}

/// Plan the imageblock output of role `ib` and stated alignment `align` held in field `field` of
/// the returned value, of AIR type `ty` (`stage_io` plans the entry's other outputs).
pub fn select_output(
    cx: &mut SelectCx<'_>,
    field: u32,
    ib: &ImageblockArg,
    align: Option<u32>,
    ty: AirTypeId,
) -> Result<(), Error> {
    let (view, cell) = view(cx, ty, ib, align)?;
    require(cx, true)?;
    reach(cx, cell)?;
    let plan = cx
        .families
        .fragment
        .imageblock
        .as_mut()
        .ok_or_else(|| invariant_error("an imageblock without its plan"))?;
    if plan.output.is_some() {
        return Err(not_implemented());
    }
    plan.output = Some((field, view));
    let early = plan_tests(cx)?;
    if let Some(plan) = cx.families.fragment.imageblock.as_mut() {
        plan.early_tests = early;
    }
    Ok(())
}

/// The entry declares `[[early_fragment_tests]]`.
fn declared_early(cx: &SelectCx<'_>) -> bool {
    matches!(
        cx.entry.ep.stage,
        crate::air::abi::Stage::Fragment {
            early_fragment_tests: true
        }
    )
}

/// Plan how the depth and stencil tests bear on the imageblock write of a function that writes
/// one, before the stage interface is complete; whether the module declares early fragment tests
/// that the function does not.
///
/// A fragment its depth or stencil test fails drops its imageblock write, as does one alpha to
/// coverage drops (measured on the M3 Max, at one sample and at four). Tested early, a fragment
/// that fails never writes; tested late, the storage already holds its write. So:
///
/// - a function whose only effects are its outputs and its imageblock write, which neither
///   discards nor returns depth, stencil or a sample mask, leaves the attachments under early
///   tests as Metal's late tests leave them: the module declares them;
/// - any other writes before Vulkan's tests and states `RuntimeFact::TestedStorageWrites`, as the
///   framebuffer fetch's storage form does: a draw whose tests drop no fragment runs it as
///   compiled, one whose tests drop some but write and count nothing the variant that declares
///   early tests (where only a discard kept them away, `early_variant`); the binder refuses a draw
///   whose depth or stencil test acts (`bind::graphics`).
fn plan_tests(cx: &mut SelectCx<'_>) -> Result<bool, Error> {
    if declared_early(cx) {
        return Ok(false);
    }
    let exported = cx.entry.ep.outputs.iter().any(|o| {
        matches!(
            o.role,
            OutputRole::Depth(_) | OutputRole::Stencil | OutputRole::SampleMask
        )
    });
    let body = super::fetch::body_facts(cx);
    let only_outputs = !exported && !body.side_effects;
    let fp = &cx.families.fragment;
    // A framebuffer fetch's storage form that declares early tests holds the function to them,
    // which keeps the write Metal's: it declares them only where the function would here.
    if fp.fetch_early_tests {
        return Ok(false);
    }
    if only_outputs && !body.discards {
        return Ok(true);
    }
    let stated = cx.facts.iter().find_map(|f| match *f {
        RuntimeFact::TestedStorageWrites { early_variant, .. } => Some(early_variant),
        _ => None,
    });
    if stated.is_none() {
        cx.facts.insert(RuntimeFact::TestedStorageWrites {
            early_tests: only_outputs && cx.inputs.read_only_tests,
            early_variant: only_outputs,
        });
    }
    Ok(stated.is_none() && only_outputs && cx.inputs.read_only_tests)
}

fn invariant_error(detail: &str) -> Error {
    Error::Internal(crate::error::InternalError::new(
        crate::error::InternalKind::Invariant,
        detail,
    ))
}

/// Complete the imageblock plan once the function is selected: its fact.
pub(super) fn finish(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    // A sample mask the function returns that does not cover the invocation's sample drops its
    // imageblock write with the fragment (measured on the M3 Max and an M4 at one sample, and on
    // the M3 Max at four, where each sample's run writes where the mask has its bit).
    let mask = cx
        .entry
        .ep
        .outputs
        .iter()
        .find(|o| matches!(o.role, OutputRole::SampleMask))
        .map(|o| o.field);
    let Some(plan) = cx.families.fragment.imageblock.as_mut() else {
        return Ok(());
    };
    let writes = plan.output.is_some();
    plan.mask = mask.filter(|_| writes);
    let Some(master) = plan.master else {
        return Err(invariant_error("an imageblock plan without its master"));
    };
    let fact = RuntimeFact::FragmentImageblock {
        words: plan.words,
        writes,
        master,
        data_size: plan.data_size,
        alignment: plan.alignment,
    };
    cx.facts.insert(fact);
    Ok(())
}

/// Declare the imageblock storage image when the plan reaches it.
pub(super) fn declare(
    mb: &mut ModuleBuilder,
    plan: Option<&ImageblockPlan>,
) -> BResult<Option<GlobalId>> {
    if plan.is_none() {
        return Ok(None);
    }
    let sampled_type = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let ty = mb.ty(SType::Image(ImageType {
        sampled_type,
        dim: Dim::D2,
        depth: false,
        arrayed: true,
        multisampled: false,
        usage: ImageUsage::Storage,
        format: crate::spv::ImageFormat::R32ui,
    }))?;
    let slot = ResourceSlot {
        stage: Stage::Fragment,
        kind: ResourceKind::Synthetic(SyntheticBinding::Imageblock),
        index: 0,
    };
    mb.resource(slot, StorageClass::UniformConstant, ty)
        .map(Some)
}

/// The cell of one pixel as a word memory: word `w` is texel `(x, y, w)` of the image, and
/// in a function that runs once per sample texel `(x, y, sample * words + w)`, accessed with the
/// function's coherence (`sync::memory_model`).
struct Cell {
    image: SValue,
    x: SValue,
    y: SValue,
    /// The first layer of the invocation's sample's cell, in a function that runs per sample.
    base: Option<SValue>,
    coherence: Coherence,
}

impl Cell {
    /// The cell of the invocation's pixel.
    fn of_pixel(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<Cell> {
        let Some(plan) = &cx.plan.families.fragment.imageblock else {
            return Err(cx.fail("an imageblock access without its plan"));
        };
        let (coord, words) = (plan.coord, plan.words);
        let Some(g) = cx.decl.families.fragment.imageblock else {
            return Err(cx.fail("the imageblock storage is not declared"));
        };
        let p = fb.module().global_value(g)?;
        let image = fb.load(p, MemAccess::default())?;
        let frag = stage_io::object_global(cx, coord)?;
        let frag = fb.module().global_value(frag)?;
        let frag = fb.load(frag, MemAccess::default())?;
        let u32_ = fb.module().ty(SType::Int {
            width: 32,
            signed: false,
        })?;
        // `FragCoord` is the pixel's center (or a sample's position inside the pixel): its
        // integer part is the pixel.
        let fx = fb.composite_extract(frag, &[0])?;
        let fy = fb.composite_extract(frag, &[1])?;
        let x = fb.convert(ConvOp::ConvertFToU, fx, u32_)?;
        let y = fb.convert(ConvOp::ConvertFToU, fy, u32_)?;
        let base = match sample_of(cx, fb)? {
            Some(sample) => {
                let words = k32(fb, words)?;
                Some(fb.binary(BinOp::IMul, sample, words)?)
            }
            None => None,
        };
        Ok(Cell {
            image,
            x,
            y,
            base,
            coherence: memory_model::coherence(cx.plan),
        })
    }

    fn texel(&self, fb: &mut FunctionBuilder<'_>, word: SValue) -> BResult<SValue> {
        let u32_ = fb.module().ty(SType::Int {
            width: 32,
            signed: false,
        })?;
        let uvec3 = fb.module().ty(SType::Vector { elem: u32_, len: 3 })?;
        let layer = match self.base {
            Some(base) => fb.binary(BinOp::IAdd, base, word)?,
            None => word,
        };
        fb.composite_construct(uvec3, &[self.x, self.y, layer])
    }
}

/// The invocation's sample (`SampleId`) in a function that runs once per sample.
fn sample_of(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<Option<SValue>> {
    let Some(object) = cx
        .plan
        .families
        .fragment
        .imageblock
        .as_ref()
        .and_then(|p| p.sample)
    else {
        return Ok(None);
    };
    let g = stage_io::object_global(cx, object)?;
    let p = fb.module().global_value(g)?;
    fb.load(p, MemAccess::default()).map(Some)
}

impl WordMem for Cell {
    fn read(&mut self, fb: &mut FunctionBuilder<'_>, word: SValue) -> BResult<SValue> {
        let at = self.texel(fb, word)?;
        let mem = self.coherence.texel(fb)?;
        let t = fb.read(self.image, at, &ImageOperands::default(), mem)?;
        fb.composite_extract(t, &[0])
    }

    fn write(&mut self, fb: &mut FunctionBuilder<'_>, word: SValue, value: SValue) -> BResult<()> {
        let at = self.texel(fb, word)?;
        let u32_ = fb.module().ty(SType::Int {
            width: 32,
            signed: false,
        })?;
        let uvec4 = fb.module().ty(SType::Vector { elem: u32_, len: 4 })?;
        let zero = k32(fb, 0)?;
        let t = fb.composite_construct(uvec4, &[value, zero, zero, zero])?;
        let mem = self.coherence.texel(fb)?;
        fb.write(self.image, at, t, &ImageOperands::default(), mem)
    }

    /// Inside the critical section no other invocation touches the pixel's cell, so a plain
    /// read-modify-write replaces exactly the bits `mask`.
    fn write_bits(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        word: SValue,
        mask: SValue,
        bits: SValue,
    ) -> BResult<()> {
        let old = self.read(fb, word)?;
        let keep = match leaf::known(fb, mask)? {
            Some(m) => k32(fb, !m)?,
            None => fb.unary(UnOp::Not, mask)?,
        };
        let kept = fb.binary(BinOp::BitwiseAnd, old, keep)?;
        let new = fb.binary(BinOp::BitwiseOr, kept, bits)?;
        self.write(fb, word, new)
    }
}

/// Read the input view, in the function's first block, and hold it for every use: Metal's input
/// is the cell's value when the fragment's critical section begins.
pub(super) fn emit_entry(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
    let Some((param, view)) = cx
        .plan
        .families
        .fragment
        .imageblock
        .as_ref()
        .and_then(|p| p.input.clone())
    else {
        return Ok(());
    };
    let Some(v) = cx.plan.values.id(Value::Arg(ArgIdx::new(param))) else {
        return Err(cx.fail(format!("imageblock parameter {param} is out of range")));
    };
    let mut cell = Cell::of_pixel(cx, fb)?;
    let mut parts = vec![None; view.members.len()];
    for m in &view.members {
        let off = k32(fb, m.offset)?;
        let x = leaf::load(fb, &mut cell, &m.leaf, off)?;
        match parts.get_mut(m.field as usize) {
            Some(slot) => *slot = Some(x),
            None => return Err(cx.fail("an imageblock member past its struct")),
        }
    }
    let parts: Vec<SValue> = parts
        .into_iter()
        .collect::<Option<_>>()
        .ok_or_else(|| cx.fail("an imageblock field without a member"))?;
    let Some(ty) = crate::lower::arith::spirv_type(fb.module(), &view.ty)? else {
        return Err(cx.fail("an imageblock view without a lowered type"));
    };
    let x = fb.composite_construct(ty, &parts)?;
    cx.hold(v, x);
    Ok(())
}

/// Write the output view of returned value `v` (in the entry's returned shape) at the return,
/// unless a predicated discard set `discarded` or the returned sample mask does not cover the
/// pixel's one sample (bit 0: the fragment is dropped, and its write with it).
pub(super) fn emit_return(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    v: Option<crate::air::ir::ValueId>,
) -> BResult<()> {
    let Some((field, view, mask)) = cx
        .plan
        .families
        .fragment
        .imageblock
        .as_ref()
        .and_then(|p| p.output.clone().map(|(f, v)| (f, v, p.mask)))
    else {
        return Ok(());
    };
    let Some(v) = v else {
        return Err(cx.fail("an imageblock output without a returned value"));
    };
    let whole = cx.operand(fb, v)?;
    let (x, covered) = match (cx.plan.families.stage_io.returned, mask) {
        (Returned::Struct, mask) => {
            let x = fb.composite_extract(whole, &[field])?;
            let covered = match mask {
                Some(m) => {
                    let m = fb.composite_extract(whole, &[m])?;
                    // The invocation's own sample's bit; bit 0 at one sample.
                    let m = match sample_of(cx, fb)? {
                        Some(sample) => fb.binary(BinOp::ShiftRightLogical, m, sample)?,
                        None => m,
                    };
                    let one = k32(fb, 1)?;
                    let zero = k32(fb, 0)?;
                    let bit = fb.binary(BinOp::BitwiseAnd, m, one)?;
                    Some(fb.binary(BinOp::INotEqual, bit, zero)?)
                }
                None => None,
            };
            (x, covered)
        }
        (Returned::Value, None) => (whole, None),
        (Returned::Value, Some(_)) => {
            return Err(cx.fail("a sample mask beside an imageblock output that is the value"))
        }
        (Returned::Void, _) => return Err(cx.fail("an imageblock output of a void entry")),
    };
    let mut values = Vec::with_capacity(view.members.len());
    for m in &view.members {
        values.push(fb.composite_extract(x, &[m.field])?);
    }
    let cell = Cell::of_pixel(cx, fb)?;
    let write = move |fb: &mut FunctionBuilder<'_>, cell: &mut Cell| -> BResult<()> {
        for (m, x) in view.members.iter().zip(values) {
            let off = k32(fb, m.offset)?;
            leaf::store(fb, cell, &m.leaf, off, x)?;
        }
        Ok(())
    };
    let mut cell = cell;
    let kept = if cx.plan.families.fragment.records_discards() {
        let Some(g) = cx.decl.families.fragment.discarded else {
            return Err(cx.fail("a recorded discard without its predicate"));
        };
        let p = fb.module().global_value(g)?;
        let discarded = fb.load(p, MemAccess::default())?;
        Some(fb.unary(UnOp::LogicalNot, discarded)?)
    } else {
        None
    };
    let kept = match (kept, covered) {
        (None, None) => return write(fb, &mut cell),
        (Some(k), None) | (None, Some(k)) => k,
        (Some(k), Some(c)) => fb.binary(BinOp::LogicalAnd, k, c)?,
    };
    let arm: ArmFn<'_, Cell> = Box::new(move |fb, cell| {
        write(fb, cell)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut cell, kept, arm, None)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::abi::{MemberKind, UserKey};
    use crate::air::error::DiagName;

    fn member(offset: u32, key: &str, layout: Option<u32>) -> Member {
        Member {
            offset,
            size: 4,
            array_len: 0,
            layout: layout.map(StructRef),
            kind: MemberKind::Plain,
            render_target: None,
            raster_order_group: Some(0),
            type_name: DiagName::new("t"),
            name: DiagName::new(key),
            key: UserKey(key.as_bytes().into()),
        }
    }

    /// Layouts `0..=depth` where each of layout `l`'s `width` members nests layout `l + 1` (the
    /// last is plain), numbered from `base`: a DAG whose in-place expansion has `width^depth`
    /// leaves.
    fn dag(base: u32, depth: u32, width: u32, deep_key: &str) -> Vec<StructInfo> {
        (0..=depth)
            .map(|l| StructInfo {
                members: (0..width)
                    .map(|i| {
                        let nested = (l < depth).then_some(base + l + 1);
                        let key = if l == depth && i == 0 { deep_key } else { "m" };
                        member(i * 4, key, nested)
                    })
                    .collect(),
            })
            .collect()
    }

    /// The fingerprint hashes each distinct layout once: a 32-wide DAG 20 levels deep (32^20
    /// leaves in place) is 21 layouts. Structurally equal masters at other references hash
    /// equal; a different key 20 levels down, or a different data size, does not.
    #[test]
    fn the_fingerprint_hashes_each_shared_layout_once() {
        let a = dag(0, 20, 32, "deep");
        let (fa, hashed) = fingerprint(&a, StructRef(0), 16).expect("fingerprint");
        assert_eq!(hashed, 21);
        // The same structure behind a layout it does not reach, at other references.
        let mut b = vec![StructInfo {
            members: vec![member(0, "unreached", None)],
        }];
        b.extend(dag(1, 20, 32, "deep"));
        let (fb, _) = fingerprint(&b, StructRef(1), 16).expect("fingerprint");
        assert_eq!(fa, fb);
        let c = dag(0, 20, 32, "other");
        assert_ne!(
            fingerprint(&c, StructRef(0), 16).expect("fingerprint").0,
            fa
        );
        assert_ne!(
            fingerprint(&a, StructRef(0), 32).expect("fingerprint").0,
            fa
        );
        // A reference out of range, and a layout nesting itself, have none.
        assert!(fingerprint(&a, StructRef(99), 16).is_none());
        let cycle = vec![StructInfo {
            members: vec![member(0, "self", Some(0))],
        }];
        assert!(fingerprint(&cycle, StructRef(0), 16).is_none());
    }
}
