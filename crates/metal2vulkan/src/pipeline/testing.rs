//! Shared pipeline test helpers: owned fixtures to bitcode, committed profiles under a chosen
//! policy, compile, link, relocate, and L3 validation (`G-val`). Every engine test
//! (`src/tests/engine_*.rs`) uses these.
//!
//! The crate's tests take the fixture assemblers and fixture directories from here too: they live
//! in the dev-tools crate (`devtools/`), named here beside the helpers that use them. The helpers
//! of this module run external tools of their own (`spirv-val`, `spirv-cross`, `xcrun metal`),
//! resolved like every other tool (`METAL2VULKAN_<TOOL>`, then the usual install directories,
//! then `PATH`). They are the crate's test helpers, never on a product path.

use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::budget::{Budget, Meter, SPIRV_WORDS};
use crate::device::profile::DeviceProfile;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::Error;
use crate::library::{FunctionArtifact, FunctionConstantValues};
use crate::lower::select::LowerInputs;
use crate::lower::select::Plan;
use crate::pipeline::compile::{
    compile_bitcode, compile_bitcode_kept, specialized_bitcode_entry, EntryRef, LinkedSource,
};
use crate::pipeline::link::{link, LinkInputs, LinkTable};
use crate::pipeline::relocate::relocate;
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::check_words_valid;
pub use metal2vulkan_devtools::{
    engine_dir, fixtures_dir, llvm_assemble, llvm_assemble_text, metal_assemble, AssembleError,
};

/// A scratch directory of the caller's own, `<temp dir>/<stem>-<process id>-<n>` with `n` from a
/// process-wide counter: tests on threads of one process (`cargo test`) and tests in processes of
/// their own (nextest) never share one. Removed, with what it holds, on drop.
pub(crate) struct Scratch(PathBuf);

impl Scratch {
    /// A fresh scratch directory named after `stem`; panics when it cannot be made.
    pub(crate) fn new(stem: &str) -> Scratch {
        Scratch::try_new(stem).unwrap_or_else(|e| panic!("scratch directory {stem}: {e}"))
    }

    /// [`Scratch::new`], with the error returned.
    pub(crate) fn try_new(stem: &str) -> std::io::Result<Scratch> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "{stem}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir)?;
        Ok(Scratch(dir))
    }

    /// The directory.
    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The description of committed profile `stem` (`validation/profiles/<stem>.json`).
pub fn description(stem: &str) -> Result<DeviceDescription, String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    DeviceDescription::from_khronos_json(&text, None).map_err(|e| format!("{stem}: {e}"))
}

/// The description of committed profile `stem` as a context at Vulkan 1.2, the runtime's instance
/// version, reads the device (`DeviceDescription::query`'s rule), with
/// `VK_KHR_format_feature_flags2` left out of what it enabled unless `flags2`: the version is the
/// lower of the two, a feature that version does not expose with the remaining extensions is not
/// read, and only the formats the product asks about are. The format tables keep the per-format
/// bits the device reported (`DeviceDescription::khr_format_feature_flags2` stays).
pub fn description_at_1_2(stem: &str, flags2: bool) -> Result<DeviceDescription, String> {
    use crate::device::requirements::{ApiVersion, VkExtension};
    let mut desc = description(stem)?;
    let api = desc.api_version.min(ApiVersion::V1_2);
    desc.api_version = api;
    if !flags2 {
        desc.extensions.remove(&VkExtension::KhrFormatFeatureFlags2);
    }
    let extensions = desc.extensions.clone();
    desc.features
        .retain(|&feature| feature.exposure().exposed(api, &extensions));
    let queried: std::collections::BTreeSet<_> =
        crate::device::FormatSupport::queries_for(api, &extensions)
            .into_iter()
            .collect();
    desc.formats
        .formats
        .retain(|format, _| queried.contains(format));
    Ok(desc)
}

/// Committed profile `stem` (`validation/profiles/<stem>.json`) derived under `policy`.
pub fn profile(stem: &str, policy: DevicePolicy) -> Result<DeviceProfile, String> {
    let dev = Device::new(description(stem)?, policy).map_err(|e| format!("{stem}: {e}"))?;
    Ok(dev.profile().clone())
}

/// Committed profile `stem` derived under `policy`, with `storage` storage buffers a stage
/// (`maxPerStageDescriptorStorageBuffers`).
pub fn profile_with_storage_buffers(
    stem: &str,
    policy: DevicePolicy,
    storage: u32,
) -> Result<DeviceProfile, String> {
    let mut desc = description(stem)?;
    desc.limits.max_per_stage_descriptor_storage_buffers = storage;
    let dev = Device::new(desc, policy).map_err(|e| format!("{stem}: {e}"))?;
    Ok(dev.profile().clone())
}

/// The name [`formatless_floor`]'s profile goes by in a test's messages.
pub const FORMATLESS_FLOOR: &str = "vk12-floor+formatless";

/// `vk12-floor` derived under `policy` with the two optional features that let a shader access a
/// storage image without its format (`shaderStorageImageReadWithoutFormat` and
/// `shaderStorageImageWriteWithoutFormat`): the floor, which states neither, beside a device that
/// leaves a storage texture's format to the bound view.
pub fn formatless_floor(policy: DevicePolicy) -> Result<DeviceProfile, String> {
    use crate::device::requirements::VkFeature;
    let mut desc = description("vk12-floor")?;
    desc.features.extend([
        VkFeature::ShaderStorageImageReadWithoutFormat,
        VkFeature::ShaderStorageImageWriteWithoutFormat,
    ]);
    let dev = Device::new(desc, policy).map_err(|e| format!("{FORMATLESS_FLOOR}: {e}"))?;
    Ok(dev.profile().clone())
}

/// The storage-buffer descriptors the stage of `a` declares.
pub fn storage_buffers(a: &FunctionArtifact) -> u32 {
    a.interface
        .descriptors
        .iter()
        .filter(|d| d.ty == crate::vk::DescriptorType::STORAGE_BUFFER)
        .map(|d| d.count)
        .sum()
}

/// Whether artifact `a` leaves a NaN's bits to the device, which the interpreter does not model:
/// it records `PrecisionNote::NativeNanEncoding`, or it requires `VkProperty::MetalNanMoves`, whose
/// measured device answers Metal's bits where the interpreter's conversions answer their own.
pub fn device_nan_bits(a: &FunctionArtifact) -> bool {
    a.precision
        .contains(&crate::pipeline::artifact::PrecisionNote::NativeNanEncoding)
        || a.requirements
            .properties
            .contains(&crate::device::requirements::VkProperty::MetalNanMoves)
}

/// The default policy with Metal's answers wherever a choice still offers them
/// ([`DevicePolicy::exact_answers`]).
pub fn exact_maths() -> DevicePolicy {
    DevicePolicy::default().exact_answers()
}

/// A fresh per-attempt meter with the standard caps and no deadline.
pub fn meter() -> Meter {
    Meter::new(&Budget::default())
}

/// Compile entry `entry` of `bitcode` for `p` with no function constants and default lowering
/// inputs.
pub fn compile(
    bitcode: &[u8],
    entry: &str,
    p: &DeviceProfile,
) -> Result<Arc<FunctionArtifact>, Error> {
    compile_with(bitcode, entry, p, &LowerInputs::new())
}

/// [`compile`], given each storage texture's format where `p` asks for it: a profile that reads or
/// writes storage images only with a declared format (the floor, which states neither
/// `shaderStorageImageReadWithoutFormat` nor `shaderStorageImageWriteWithoutFormat`) needs every
/// storage slot's format (`NeedsInput(StorageTextureFormat)`), and the runtime compiles the variant
/// of the format bound there. Each slot named takes `pf`'s read-and-write choice
/// ([`storage_image_choice`](crate::metal::format::storage_image_choice)); a profile that asks for
/// none compiles as [`compile`] does.
pub fn compile_given_storage_format(
    bitcode: &[u8],
    entry: &str,
    p: &DeviceProfile,
    pf: crate::metal::PixelFormat,
) -> Result<Arc<FunctionArtifact>, Error> {
    let mut inputs = LowerInputs::new();
    loop {
        match compile_with(bitcode, entry, p, &inputs) {
            Err(Error::NeedsInput(crate::error::RuntimeInput::StorageTextureFormat { slot }))
                if !inputs.storage_images.contains_key(&slot) =>
            {
                let choice = crate::metal::format::storage_image_choice(pf, p)?;
                inputs.storage_images.insert(slot, choice);
            }
            r => return r,
        }
    }
}

/// The lowering inputs a fragment function compiles under for the color attachments of `r` on
/// `p`: how they take its float outputs (`LoweringInputs::color_writes`) and which hold
/// extended-range texels (`LoweringInputs::extended_range_colors`), as
/// `Device::compile_render_pipeline` derives them.
pub fn fragment_inputs(
    r: &crate::pipeline::link::RenderLinkInputs,
    p: &DeviceProfile,
) -> LowerInputs {
    let mut inputs = LowerInputs::new();
    inputs.color_writes = crate::pipeline::color_writes::color_writes(
        0..r.color_attachments.len() as u32,
        &r.color_attachments,
        (r.alpha_to_coverage, r.texture_write_rounding_mode),
        p,
    );
    inputs.extended_range_colors = crate::pipeline::color_writes::extended_range_colors(
        &r.color_attachments,
        r.texture_write_rounding_mode,
    );
    inputs.word_colors =
        crate::pipeline::color_writes::word_colors(&r.color_attachments, r.alpha_to_coverage, p);
    inputs.rotated_colors = crate::pipeline::color_writes::rotated_colors(&r.color_attachments, p);
    inputs.alpha_in_red_colors =
        crate::pipeline::color_writes::alpha_in_red_colors(&r.color_attachments, p);
    inputs.rotated_factors =
        crate::pipeline::color_writes::rotated_factors(&r.color_attachments, p);
    inputs.rotated_blends = crate::pipeline::color_writes::rotated_blends(&r.color_attachments, p);
    inputs.texture_write_rounding = r.texture_write_rounding_mode;
    inputs
}

/// [`compile`] under lowering inputs `inputs` (a pipeline's state, such as its framebuffer
/// fetch's).
pub fn compile_with(
    bitcode: &[u8],
    entry: &str,
    p: &DeviceProfile,
    inputs: &LowerInputs,
) -> Result<Arc<FunctionArtifact>, Error> {
    compile_bitcode(
        bitcode,
        EntryRef::Name(entry),
        &FunctionConstantValues::defaults(),
        p,
        inputs,
        &mut meter(),
        SPIRV_WORDS,
    )
}

/// [`compile`], also returning the plan and the finished typed module whose serialization the
/// artifact holds (`compile_bitcode_kept`): one compile, so the CPU interpreter runs the module
/// whose words `link_and_validate` links and validates, not a second compile of the same source.
pub fn compile_kept(
    bitcode: &[u8],
    entry: &str,
    p: &DeviceProfile,
) -> Result<(Arc<FunctionArtifact>, Plan, TypedModule), Error> {
    compile_bitcode_kept(
        bitcode,
        EntryRef::Name(entry),
        p,
        &LowerInputs::new(),
        &mut meter(),
        SPIRV_WORDS,
    )
    .map(|(compiled, plan, module)| (Arc::new(compiled.artifact().clone()), plan, module))
}

/// The plan and the finished typed module of entry `entry` of `bitcode` for `p`, with no function
/// constants and default lowering inputs: the module `compile` serializes (P9 applied,
/// verified), which the CPU interpreter runs. It compiles again, apart from [`compile`];
/// [`compile_kept`] returns both from one compile.
pub fn typed_module(
    bitcode: &[u8],
    entry: &str,
    p: &DeviceProfile,
) -> Result<(Plan, TypedModule), Error> {
    typed_module_with(bitcode, entry, p, &LowerInputs::new())
}

/// [`typed_module`] under lowering inputs `inputs` (a vertex fetch projection, a
/// `BufferRangeLimit` variant).
pub fn typed_module_with(
    bitcode: &[u8],
    entry: &str,
    p: &DeviceProfile,
    inputs: &LowerInputs,
) -> Result<(Plan, TypedModule), Error> {
    typed_module_of(
        bitcode,
        entry,
        p,
        (inputs, &[]),
        &FunctionConstantValues::defaults(),
    )
}

/// [`typed_module`] with the function constants `consts` supplied: the module a pipeline built with
/// those values runs.
pub fn typed_module_specialized(
    bitcode: &[u8],
    entry: &str,
    p: &DeviceProfile,
    consts: &FunctionConstantValues,
) -> Result<(Plan, TypedModule), Error> {
    typed_module_of(bitcode, entry, p, (&LowerInputs::new(), &[]), consts)
}

/// [`typed_module`] with the module's visible-function references resolved against `linked`.
pub fn typed_module_linked(
    bitcode: &[u8],
    entry: &str,
    p: &DeviceProfile,
    linked: &[LinkedSource<'_>],
) -> Result<(Plan, TypedModule), Error> {
    typed_module_linked_with(bitcode, entry, p, linked, &LowerInputs::new())
}

/// [`typed_module_linked`] under lowering inputs `inputs` (a kernel's `maxCallStackDepth`).
pub fn typed_module_linked_with(
    bitcode: &[u8],
    entry: &str,
    p: &DeviceProfile,
    linked: &[LinkedSource<'_>],
    inputs: &LowerInputs,
) -> Result<(Plan, TypedModule), Error> {
    typed_module_of(
        bitcode,
        entry,
        p,
        (inputs, linked),
        &FunctionConstantValues::defaults(),
    )
}

/// Entry `entry` of `bitcode` as a compile specializes it before selection: its function
/// constants `consts`, its visible-function references resolved against the `linked` functions, and
/// the order Apple's simdgroup lockstep gives threadgroup accesses made explicit for `p`. It is the
/// compile's own entry, built by the same code, so a refusal's location (function, block,
/// instruction) names an instruction of it.
pub fn specialized_entry(
    bitcode: &[u8],
    entry: &str,
    p: &DeviceProfile,
    consts: &FunctionConstantValues,
    linked: &[LinkedSource<'_>],
    m: &mut Meter,
) -> Result<crate::air::Entry, Error> {
    let inputs = LowerInputs::new();
    specialized_bitcode_entry(
        bitcode,
        EntryRef::Name(entry),
        (consts, linked, &inputs),
        p,
        m,
    )
}

fn typed_module_of(
    bitcode: &[u8],
    entry: &str,
    p: &DeviceProfile,
    (inputs, linked): (&LowerInputs, &[LinkedSource<'_>]),
    consts: &FunctionConstantValues,
) -> Result<(Plan, TypedModule), Error> {
    let m = &mut meter();
    let spec = specialized_bitcode_entry(
        bitcode,
        EntryRef::Name(entry),
        (consts, linked, inputs),
        p,
        m,
    )?;
    let memory = crate::memory::static_root::analyze_memory(&spec, p, m)?;
    let plan = crate::lower::select::select(&spec, &memory, p, inputs, m)?;
    let crate::lower::emit::Emitted { mut module, .. } =
        crate::lower::emit::emit_module(&spec, &plan, m)?;
    crate::spirv::roll::roll_entry(&mut module, m)?;
    let (_, fences) = crate::spirv::helper_branches::guard_for(&mut module, p, m)?;
    fences.apply(&mut module)?;
    crate::spirv::verify::verify(&module, p, m)?;
    Ok((plan, module))
}

/// Link `a` alone and relocate it: the table and the final words, without L3.
pub fn link_standalone(
    a: &FunctionArtifact,
    p: &DeviceProfile,
) -> Result<(LinkTable, Vec<u32>), Error> {
    let table = link(&[a], &LinkInputs::standalone(), p)?.table;
    let words = relocate(a, &table)?;
    Ok((table, words))
}

/// Link `a` alone, relocate it, and run L3 (requirements ⊆ profile, `spirv-val` under the
/// profile's environments and relaxations). Returns the table and the final words. A key the
/// function reads as data left a specialization constant fails it: a read of the key's constant the
/// entry hook did not reach, which would make a pipeline per bound value.
pub fn link_and_validate(
    a: &FunctionArtifact,
    p: &DeviceProfile,
) -> Result<(LinkTable, Vec<u32>), String> {
    if let Some(c) = a.interface.spec.iter().find(|c| c.key.read_as_data()) {
        return Err(format!("{:?} is still a specialization constant", c.key));
    }
    let (table, words) = link_standalone(a, p).map_err(|e| format!("link: {e}"))?;
    check_words_valid(&words, &a.requirements, p)?;
    Ok((table, words))
}

/// `words` with every relocation slot of `a` set to 0.
pub fn zero_relocations(a: &FunctionArtifact, words: &[u32]) -> Vec<u32> {
    let mut out = words.to_vec();
    for (_, at) in &a.module.relocs {
        if let Some(w) = out.get_mut(*at as usize) {
            *w = 0;
        }
    }
    out
}

/// The instructions of the module `words` after its header: each opcode with its whole
/// instruction (the first word included).
fn instructions(words: &[u32]) -> Result<Vec<(crate::spv::Op, &[u32])>, String> {
    let mut insts = Vec::new();
    let mut at = words.get(5..).ok_or("no SPIR-V header")?;
    while let Some(first) = at.first() {
        let count = (first >> 16) as usize;
        let inst = at
            .get(..count)
            .filter(|_| count > 0)
            .ok_or("truncated instruction")?;
        let op = crate::spv::Op::from_u32(first & 0xffff)
            .ok_or_else(|| format!("opcode {}", first & 0xffff))?;
        insts.push((op, inst));
        at = at.get(count..).unwrap_or_default();
    }
    Ok(insts)
}

/// Check that every atomic of the module `words` operates on a place: its pointer operand is the
/// result of `OpVariable`, `OpAccessChain`, or `OpImageTexelPointer`, the pointers SPIRV-Cross
/// writes as MSL places. Returns the number of atomics.
pub fn atomics_on_places(words: &[u32]) -> Result<usize, String> {
    use crate::spv::Op;
    let insts = instructions(words)?;
    let mut defs = std::collections::HashMap::new();
    for (op, inst) in &insts {
        let (typed, result) = crate::spirv_binary::instruction_result_shape(*op)
            .ok_or_else(|| format!("{op:?}: no grammar"))?;
        if result {
            let id = inst.get(if typed { 2 } else { 1 }).ok_or("no result id")?;
            defs.insert(*id, *op);
        }
    }
    let mut atomics = 0;
    for (op, inst) in &insts {
        let Some(ptr) = atomic_pointer_operand(*op) else {
            continue;
        };
        let ptr = inst.get(ptr).ok_or("no pointer operand")?;
        match defs.get(ptr) {
            Some(Op::Variable | Op::AccessChain | Op::ImageTexelPointer) => atomics += 1,
            other => return Err(format!("{op:?} on %{ptr}, defined by {other:?}")),
        }
    }
    Ok(atomics)
}

/// The accesses [`interlocked_coherence`] checked, by the memory each reaches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CoherentAccesses {
    /// The module's memory model is Vulkan (else `GLSL450`).
    pub vulkan: bool,
    /// Loads and stores through a `StorageBuffer` pointer.
    pub buffer: usize,
    /// Loads and stores of an application word through a `PhysicalStorageBuffer` pointer.
    pub device_address: usize,
    /// `OpImageRead`, `OpImageSparseRead`, and `OpImageWrite` of a storage image.
    pub storage_image: usize,
    /// Those of them that carry `VolatileTexel` (a volatile image under the Vulkan model).
    pub volatile_texel: usize,
    /// 64-bit `PhysicalStorageBuffer` loads: the root table's entries, which the host writes before
    /// the work and no invocation writes, so they need no coherence.
    pub root_table: usize,
}

/// Check that every access of memory another invocation may write in the module `words` is
/// coherent at device scope, as a fragment-interlock critical section needs: under `GLSL450` its
/// buffer or storage image variable is decorated `Coherent` (a device address has no variable, so
/// none is allowed), and under the Vulkan model a load carries `NonPrivatePointer` and
/// `MakePointerVisible`, a store `NonPrivatePointer` and `MakePointerAvailable`, a storage-image
/// read `NonPrivateTexel` and `MakeTexelVisible`, and a write `NonPrivateTexel` and
/// `MakeTexelAvailable`, each at `Device` scope. Atomics are coherent by themselves and not
/// counted. Returns the accesses checked, or the first incoherent one.
pub fn interlocked_coherence(words: &[u32]) -> Result<CoherentAccesses, String> {
    use crate::spv::Op;
    use std::collections::{HashMap, HashSet};
    const COHERENT: u32 = crate::spv::Decoration::Coherent as u32;
    const STORAGE_BUFFER: u32 = crate::spv::StorageClass::StorageBuffer as u32;
    const DEVICE_ADDRESS: u32 = crate::spv::StorageClass::PhysicalStorageBuffer as u32;
    const DEVICE: u32 = crate::spv::Scope::Device as u32;
    let insts = instructions(words)?;
    let mut out = CoherentAccesses::default();
    let mut coherent = HashSet::new();
    // Pointer types (class, pointee), 64-bit integer types, storage image types, arrays (their
    // element), integer constants, each result's type, and each pointer's root variable.
    let mut pointers: HashMap<u32, (u32, u32)> = HashMap::new();
    let mut wide = HashSet::new();
    let mut storage_images = HashSet::new();
    let mut constants: HashMap<u32, u32> = HashMap::new();
    let mut types: HashMap<u32, u32> = HashMap::new();
    let mut roots: HashMap<u32, u32> = HashMap::new();
    let at = |inst: &[u32], i: usize| -> Result<u32, String> {
        inst.get(i)
            .copied()
            .ok_or_else(|| "a short instruction".to_string())
    };
    for (op, inst) in &insts {
        match op {
            Op::MemoryModel => out.vulkan = at(inst, 2)? == crate::spv::MemoryModel::Vulkan as u32,
            Op::Decorate if at(inst, 2)? == COHERENT => {
                coherent.insert(at(inst, 1)?);
            }
            Op::TypePointer => {
                pointers.insert(at(inst, 1)?, (at(inst, 2)?, at(inst, 3)?));
            }
            Op::TypeInt if at(inst, 2)? == 64 => {
                wide.insert(at(inst, 1)?);
            }
            // Storage (sampled = 2), not subpass data.
            Op::TypeImage
                if at(inst, 7)? == 2 && at(inst, 3)? != crate::spv::Dim::SubpassData as u32 =>
            {
                storage_images.insert(at(inst, 1)?);
            }
            Op::Constant => {
                constants.insert(at(inst, 2)?, at(inst, 3)?);
            }
            _ => {}
        }
        if let Some((true, true)) = crate::spirv_binary::instruction_result_shape(*op) {
            types.insert(at(inst, 2)?, at(inst, 1)?);
        }
        match op {
            Op::Variable => {
                roots.insert(at(inst, 2)?, at(inst, 2)?);
            }
            Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain | Op::Load => {
                if let Some(&r) = roots.get(&at(inst, 3)?) {
                    roots.insert(at(inst, 2)?, r);
                }
            }
            _ => {}
        }
    }
    let device_scope = |id: u32| constants.get(&id) == Some(&DEVICE);
    let decorated = |id: u32| roots.get(&id).is_some_and(|r| coherent.contains(r));
    for (op, inst) in &insts {
        let (ptr, mask_at, load) = match op {
            Op::Load => (at(inst, 3)?, 4, true),
            Op::Store => (at(inst, 1)?, 3, false),
            Op::ImageRead | Op::ImageSparseRead | Op::ImageWrite => {
                let write = *op == Op::ImageWrite;
                let image = at(inst, if write { 1 } else { 3 })?;
                let ty = types.get(&image).copied().unwrap_or_default();
                if !storage_images.contains(&ty) {
                    continue;
                }
                out.storage_image += 1;
                let what = format!("{op:?} of %{image}");
                let mask_at = if write { 4 } else { 5 };
                let mask = inst.get(mask_at).copied().unwrap_or(0);
                if !out.vulkan {
                    if mask & 0xf00 != 0 || !decorated(image) {
                        return Err(format!("{what}: not Coherent under GLSL450"));
                    }
                    continue;
                }
                // The operands' ids follow the mask in bit order; count those before the scope.
                let ids_before = |bit: u32| -> u32 {
                    let mut n = 0;
                    for (b, k) in [
                        (0x1, 1),
                        (0x2, 1),
                        (0x4, 2),
                        (0x8, 1),
                        (0x10, 1),
                        (0x20, 1),
                        (0x40, 1),
                        (0x80, 1),
                        (0x100, 1),
                    ] {
                        if b < bit && mask & b != 0 {
                            n += k;
                        }
                    }
                    n
                };
                if mask & 0x800 != 0 {
                    out.volatile_texel += 1;
                }
                let make = if write { 0x100 } else { 0x200 };
                let scope = inst.get(mask_at + 1 + ids_before(make) as usize).copied();
                if mask & 0x400 == 0 || mask & make == 0 || !scope.is_some_and(device_scope) {
                    return Err(format!("{what}: operands {mask:#x} under the Vulkan model"));
                }
                continue;
            }
            _ => continue,
        };
        let Some(&(class, pointee)) = types.get(&ptr).and_then(|t| pointers.get(t)) else {
            continue;
        };
        let what = format!("{op:?} through %{ptr}");
        match class {
            STORAGE_BUFFER => out.buffer += 1,
            DEVICE_ADDRESS if load && wide.contains(&pointee) => {
                out.root_table += 1;
                continue;
            }
            DEVICE_ADDRESS => out.device_address += 1,
            _ => continue,
        }
        let mask = inst.get(mask_at).copied().unwrap_or(0);
        if !out.vulkan {
            if class == DEVICE_ADDRESS || !decorated(ptr) {
                return Err(format!("{what}: not Coherent under GLSL450"));
            }
            continue;
        }
        // `Aligned`'s literal precedes the scopes; `MakePointerAvailable`'s precedes
        // `MakePointerVisible`'s.
        let (make, before) = if load {
            (0x10, mask & 0x8 != 0)
        } else {
            (0x8, false)
        };
        let scope_at = mask_at + 1 + usize::from(mask & 0x2 != 0) + usize::from(before);
        let scope = inst.get(scope_at).copied();
        if mask & 0x20 == 0 || mask & make == 0 || !scope.is_some_and(device_scope) {
            return Err(format!("{what}: operands {mask:#x} under the Vulkan model"));
        }
    }
    Ok(out)
}

/// The word index of an atomic instruction's pointer operand, or `None` for another opcode.
fn atomic_pointer_operand(op: crate::spv::Op) -> Option<usize> {
    use crate::spv::Op;
    match op {
        Op::AtomicStore | Op::AtomicFlagClear => Some(1),
        Op::AtomicLoad
        | Op::AtomicExchange
        | Op::AtomicCompareExchange
        | Op::AtomicCompareExchangeWeak
        | Op::AtomicIIncrement
        | Op::AtomicIDecrement
        | Op::AtomicIAdd
        | Op::AtomicISub
        | Op::AtomicSMin
        | Op::AtomicUMin
        | Op::AtomicSMax
        | Op::AtomicUMax
        | Op::AtomicAnd
        | Op::AtomicOr
        | Op::AtomicXor
        | Op::AtomicFlagTestAndSet
        | Op::AtomicFAddEXT
        | Op::AtomicFMinEXT
        | Op::AtomicFMaxEXT => Some(3),
        _ => None,
    }
}

/// The longest walk the builder's place check (`FunctionBuilder::is_place`) takes for an atomic of
/// the module `words`: it looks the pointer's definition up from the end of the definition's block
/// as the block stood when the atomic was built, so the walk is the instructions from the atomic
/// back to the definition in the same block, or from the end of an earlier block back to it; a
/// variable takes none. Read from the final words, which keep the builder's order within a block; a
/// block's merge instruction and terminator are not among the instructions the builder walks.
pub fn atomic_place_scan_steps(words: &[u32]) -> Result<usize, String> {
    use crate::spv::Op;
    let insts = instructions(words)?;
    // (block ordinal, position after the label) of every result, and each block's length.
    let (mut block, mut pos) = (0usize, 0usize);
    let mut defs = std::collections::HashMap::new();
    let mut lengths = vec![0usize];
    let mut atomics = Vec::new();
    for (op, inst) in &insts {
        if *op == Op::Label {
            block += 1;
            pos = 0;
            lengths.push(0);
            continue;
        }
        let (typed, result) = crate::spirv_binary::instruction_result_shape(*op)
            .ok_or_else(|| format!("{op:?}: no grammar"))?;
        if result {
            let id = inst.get(if typed { 2 } else { 1 }).ok_or("no result id")?;
            defs.insert(*id, (*op, block, pos));
        }
        if let Some(ptr) = atomic_pointer_operand(*op) {
            atomics.push((*inst.get(ptr).ok_or("no pointer operand")?, block, pos));
        }
        if matches!(
            op,
            Op::SelectionMerge
                | Op::LoopMerge
                | Op::Branch
                | Op::BranchConditional
                | Op::Switch
                | Op::Return
                | Op::ReturnValue
                | Op::Kill
                | Op::TerminateInvocation
                | Op::Unreachable
        ) {
            continue;
        }
        pos += 1;
        if let Some(n) = lengths.last_mut() {
            *n = pos;
        }
    }
    let mut steps = 0;
    for (ptr, b, p) in atomics {
        let walk = match defs.get(&ptr) {
            Some((Op::Variable, _, _)) => 0,
            Some((_, db, dp)) if *db == b => p - dp,
            Some((_, db, dp)) => lengths.get(*db).copied().unwrap_or_default() - dp,
            None => return Err(format!("an atomic on the undefined %{ptr}")),
        };
        steps = steps.max(walk);
    }
    Ok(steps)
}

/// The `spirv-cross` and Metal compiler arguments of one module ([`moltenvk_msl_args`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoltenVkMslArgs {
    /// `spirv-cross` arguments after the input path; the caller appends `--output <msl>`.
    pub spirv_cross: Vec<String>,
    /// `xcrun` arguments; the caller appends `<msl> -o <air>`.
    pub metal: Vec<String>,
}

/// The arguments with which `spirv-cross` and `xcrun metal` convert and compile the module `words`
/// as MoltenVK 1.4.1 does when it creates a pipeline from it on the `moltenvk-m3` device, as far
/// as the two command lines express MoltenVK's options.
///
/// SPIRV-Cross (MoltenVK's `SPIRVToMSLConversionOptions` and each pipeline's `getMTLFunction`):
/// MSL 4.0 (macOS 26 and an Xcode 26 build), Metal argument buffers at tier 2
/// (`MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS` defaults on), native texel buffers, 1D images as 2D
/// (`MVK_CONFIG_TEXTURE_1D_AS_2D` defaults on), recursive-input replacement (macOS 14 and later),
/// padded fragment outputs, the linear `R32Uint` texture alignment 16, and the entry point by name
/// and stage. Compute and fragment stages fix the subgroup size at the device's 32 (MoltenVK does
/// unless the stage allows a varying size), graphics stages fix up cube gradients (Apple GPUs), a
/// vertex stage flips Y (`MVK_CONFIG_SHADER_CONVERSION_FLIP_VERTEX_Y` defaults on), and a fragment
/// stage checks discarded stores.
///
/// Metal (`MVKDevice::getMTLCompileOptions` under the default on-demand fast math): language 4.0,
/// and the math mode MoltenVK derives from the fast-math flags SPIRV-Cross reports for the entry
/// point (every flag, less what `SignedZeroInfNanPreserve`, `ContractionOff`, `FPFastMathDefault`,
/// and `FPFastMathMode` decorations leave out): fast with fast fp32 functions when NotNaN, NotInf,
/// NSZ, AllowRecip, AllowReassoc, and AllowContract all remain, relaxed when the last four do, safe
/// with precise functions otherwise; invariance is preserved when the position output is
/// invariant.
pub fn moltenvk_msl_args(words: &[u32]) -> Result<MoltenVkMslArgs, String> {
    use crate::spv::{BuiltIn, Decoration, ExecutionMode, ExecutionModel, FPFastMathMode, Op};
    let insts = instructions(words)?;
    let entries = insts
        .iter()
        .filter(|(op, _)| *op == Op::EntryPoint)
        .collect::<Vec<_>>();
    let [(_, entry)] = entries.as_slice() else {
        return Err(format!(
            "{} entry points (a MoltenVK pipeline stage names one)",
            entries.len()
        ));
    };
    let (model, id, name) = match entry {
        [_, model, id, name @ ..] if !name.is_empty() => (*model, *id, literal_string(name)?),
        _ => return Err("truncated OpEntryPoint".to_owned()),
    };
    let (stage, stage_args): (&str, &[&str]) = match ExecutionModel::from_u32(model) {
        Some(ExecutionModel::GLCompute) => ("comp", &["--msl-fixed-subgroup-size", "32"]),
        Some(ExecutionModel::Fragment) => (
            "frag",
            &[
                "--msl-fixed-subgroup-size",
                "32",
                "--msl-check-discarded-frag-stores",
                "--msl-agx-manual-cube-grad-fixup",
            ],
        ),
        Some(ExecutionModel::Vertex) => (
            "vert",
            &["--flip-vert-y", "--msl-agx-manual-cube-grad-fixup"],
        ),
        other => return Err(format!("execution model {other:?} ({model})")),
    };
    let spirv_cross = [
        "--msl",
        "--msl-version",
        "40000",
        "--msl-argument-buffers",
        // `MTLArgumentBuffersTier2`.
        "--msl-argument-buffer-tier",
        "1",
        "--msl-texture-buffer-native",
        "--msl-texture-1d-as-2d",
        "--msl-replace-recursive-inputs",
        "--msl-r32ui-linear-texture-align",
        "16",
        "--msl-pad-fragment-output",
        "--entry",
        name.as_str(),
        "--stage",
        stage,
    ]
    .iter()
    .chain(stage_args)
    .map(|arg| (*arg).to_owned())
    .collect::<Vec<_>>();

    let constants = insts
        .iter()
        .filter(|(op, _)| *op == Op::Constant)
        .filter_map(|(_, inst)| Some((*inst.get(2)?, *inst.get(3)?)))
        .collect::<std::collections::HashMap<u32, u32>>();
    // SPIRV-Cross starts from every flag (`CompilerMSL::get_fp_fast_math_flags`).
    let mut fast = u32::MAX;
    let mut invariant = std::collections::HashSet::new();
    let mut position = std::collections::HashSet::new();
    for (op, inst) in &insts {
        match (op, inst.get(1..).unwrap_or_default()) {
            (Op::ExecutionMode, [target, mode, ..]) if *target == id => {
                if *mode == ExecutionMode::SignedZeroInfNanPreserve as u32 {
                    fast &=
                        !(FPFastMathMode::NSZ | FPFastMathMode::NOT_INF | FPFastMathMode::NOT_NAN)
                            .bits();
                } else if *mode == ExecutionMode::ContractionOff as u32 {
                    fast &= !FPFastMathMode::ALLOW_CONTRACT.bits();
                }
            }
            (Op::ExecutionModeId, [target, mode, _ty, flags, ..])
                if *target == id && *mode == ExecutionMode::FPFastMathDefault as u32 =>
            {
                fast &= constants
                    .get(flags)
                    .ok_or_else(|| format!("FPFastMathDefault flags %{flags} is no constant"))?;
            }
            (Op::Decorate, [_, decoration, mask])
                if *decoration == Decoration::FPFastMathMode as u32 =>
            {
                fast &= mask;
            }
            (Op::Decorate, [target, decoration, rest @ ..])
            | (Op::MemberDecorate, [target, _, decoration, rest @ ..]) => {
                let member = inst.get(2).copied().filter(|_| *op == Op::MemberDecorate);
                let place = (*target, member);
                if *decoration == Decoration::Invariant as u32 {
                    invariant.insert(place);
                } else if *decoration == Decoration::BuiltIn as u32
                    && rest.first() == Some(&(BuiltIn::Position as u32))
                {
                    position.insert(place);
                }
            }
            _ => {}
        }
    }
    let relaxed = (FPFastMathMode::NSZ
        | FPFastMathMode::ALLOW_RECIP
        | FPFastMathMode::ALLOW_REASSOC
        | FPFastMathMode::ALLOW_CONTRACT)
        .bits();
    let (mode, functions) = if fast & relaxed != relaxed {
        ("safe", "precise")
    } else if fast & (FPFastMathMode::NOT_NAN | FPFastMathMode::NOT_INF).bits()
        == (FPFastMathMode::NOT_NAN | FPFastMathMode::NOT_INF).bits()
    {
        ("fast", "fast")
    } else {
        ("relaxed", "precise")
    };
    let mut metal = [
        "-sdk",
        "macosx",
        "metal",
        "-std=metal4.0",
        &format!("-fmetal-math-mode={mode}"),
        &format!("-fmetal-math-fp32-functions={functions}"),
    ]
    .map(str::to_owned)
    .to_vec();
    if position.intersection(&invariant).next().is_some() {
        metal.push("-fpreserve-invariance".to_owned());
    }
    metal.push("-c".to_owned());
    Ok(MoltenVkMslArgs { spirv_cross, metal })
}

/// A SPIR-V literal string: UTF-8 bytes up to the first NUL.
fn literal_string(words: &[u32]) -> Result<String, String> {
    let bytes = words
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .take_while(|b| *b != 0)
        .collect::<Vec<_>>();
    String::from_utf8(bytes).map_err(|e| format!("literal string: {e}"))
}

/// Under `METAL2VULKAN_TEST_REQUIRE_SPIRV_CROSS=1` a run requires `spirv-cross`: a MoltenVK MSL
/// check that cannot start it fails instead of skipping, so a host without it cannot pass those
/// checks unseen (a host that runs them sets it, as CI's Linux job sets
/// `METAL2VULKAN_TEST_REQUIRE_PINNED_LLVM=1` for `llvm-as`).
const REQUIRE_SPIRV_CROSS: &str = "METAL2VULKAN_TEST_REQUIRE_SPIRV_CROSS";

/// Run `cmd` for a MoltenVK MSL check: a tool that cannot start is `Missing` (a failure where the
/// run requires it, [`REQUIRE_SPIRV_CROSS`]), one that refuses its input is `Rejected` with its
/// output.
fn msl_tool(cmd: &str, args: &[&std::ffi::OsStr]) -> Result<(), AssembleError> {
    match metal2vulkan_devtools::tools::run(cmd, args) {
        Ok(_) => Ok(()),
        Err(metal2vulkan_devtools::tools::ToolError::Launch { executable, .. }) => {
            let why = format!("{cmd} not found ({})", executable.display());
            assert!(
                cmd != "spirv-cross"
                    || std::env::var_os(REQUIRE_SPIRV_CROSS).is_none_or(|v| v != "1"),
                "{REQUIRE_SPIRV_CROSS}=1: {why}"
            );
            Err(AssembleError::Missing(why))
        }
        Err(e) => Err(AssembleError::Rejected(e.to_string())),
    }
}

/// Convert the module `words` to MSL in `scratch` with `spirv-cross` and `args` (MoltenVK's
/// options, [`moltenvk_msl_args`]); returns the MSL path.
fn convert_to_msl(
    words: &[u32],
    args: &MoltenVkMslArgs,
    scratch: &Scratch,
) -> Result<PathBuf, AssembleError> {
    let spv = scratch.path().join("module.spv");
    let msl = scratch.path().join("module.metal");
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&spv, bytes).map_err(|e| AssembleError::Missing(format!("scratch: {e}")))?;
    let mut convert = vec![spv.as_os_str()];
    convert.extend(args.spirv_cross.iter().map(std::ffi::OsStr::new));
    convert.extend(["--output".as_ref(), msl.as_os_str()]);
    msl_tool("spirv-cross", &convert)?;
    Ok(msl)
}

/// The MSL `spirv-cross` writes for the module `words` with MoltenVK 1.4.1's options
/// ([`moltenvk_msl_args`]; the source MoltenVK compiles). `Missing` when `spirv-cross` is not
/// installed; `Rejected` when it refuses the module.
pub fn moltenvk_msl(words: &[u32]) -> Result<String, AssembleError> {
    let args = moltenvk_msl_args(words).map_err(AssembleError::Rejected)?;
    let scratch = Scratch::try_new("m2v-pipeline-testing")
        .map_err(|e| AssembleError::Missing(format!("scratch: {e}")))?;
    let msl = convert_to_msl(words, &args, &scratch)?;
    std::fs::read_to_string(&msl).map_err(|e| AssembleError::Rejected(format!("msl: {e}")))
}

/// Convert the module `words` to MSL and compile it as MoltenVK 1.4.1 does when it creates a
/// pipeline from it ([`moltenvk_msl_args`]: `spirv-cross`, then `xcrun -sdk macosx metal -c`;
/// compile only, nothing runs). `Missing` when `spirv-cross` or the Metal compiler is not installed
/// (a test skips visibly); `Rejected` with the tool's output when either refuses the module, as
/// MoltenVK's pipeline creation would.
pub fn moltenvk_msl_compile(words: &[u32]) -> Result<(), AssembleError> {
    if !cfg!(target_os = "macos") {
        return Err(AssembleError::Missing(
            "the Metal compiler needs macOS".to_owned(),
        ));
    }
    let args = moltenvk_msl_args(words).map_err(AssembleError::Rejected)?;
    let scratch = Scratch::try_new("m2v-pipeline-testing")
        .map_err(|e| AssembleError::Missing(format!("scratch: {e}")))?;
    let air = scratch.path().join("module.air");
    // The Metal toolchain is a separate download: probe for it before blaming the module.
    msl_tool(
        "xcrun",
        &[
            "-sdk".as_ref(),
            "macosx".as_ref(),
            "metal".as_ref(),
            "--version".as_ref(),
        ],
    )
    .map_err(|e| match e {
        AssembleError::Rejected(why) => AssembleError::Missing(why),
        e => e,
    })?;
    let msl = convert_to_msl(words, &args, &scratch)?;
    let mut compile = args
        .metal
        .iter()
        .map(std::ffi::OsStr::new)
        .collect::<Vec<_>>();
    compile.extend([msl.as_os_str(), "-o".as_ref(), air.as_os_str()]);
    msl_tool("xcrun", &compile)
}

/// How a compiled stage reaches the command buffer's `ExecutionStatus` buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusReach {
    /// It does not.
    None,
    /// By the device address in its push field (`PushField::ExecutionStatusAddress`).
    Address,
    /// Through its storage-buffer binding (`SyntheticBinding::ExecutionStatus`).
    Binding,
}

/// How the stage of `a` reaches the `ExecutionStatus` buffer: by address or through its binding,
/// never both (a stage with both panics).
pub fn status_reach(a: &FunctionArtifact) -> StatusReach {
    use crate::pipeline::sym::{PushField, ResourceKind, SyntheticBinding};
    let address = a
        .interface
        .push
        .fields
        .iter()
        .any(|f| f.field == PushField::ExecutionStatusAddress);
    let binding = a.interface.descriptors.iter().any(|d| {
        d.slot.kind == ResourceKind::Synthetic(SyntheticBinding::ExecutionStatus)
            && d.ty == crate::vk::DescriptorType::STORAGE_BUFFER
    });
    match (address, binding) {
        (false, false) => StatusReach::None,
        (true, false) => StatusReach::Address,
        (false, true) => StatusReach::Binding,
        (true, true) => panic!("a stage reaches its status by address and through a binding"),
    }
}

/// The reach of the `ExecutionStatus` buffer a stage that writes it has on `p`: by address where
/// `p` addresses buffers, else through its binding.
pub fn status_reach_on(p: &DeviceProfile) -> StatusReach {
    match p.fields().buffer_access {
        crate::device::profile::BufferAccess::Addressed => StatusReach::Address,
        crate::device::profile::BufferAccess::Floor => StatusReach::Binding,
    }
}

mod tests;
