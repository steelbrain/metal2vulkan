//! Profile F end to end.
//!
//! Profile F (`BufferAccess::Floor`: no `bufferDeviceAddress` or no `shaderInt64`) reaches every
//! buffer through a descriptor. No committed profile is Profile F: the runtime's floor requires
//! device addresses and 64-bit integers. Over every public fixture on the devices below that floor
//! this file builds (`runtime-floor` without `bufferDeviceAddress`, and without `shaderInt64`, which
//! keeps device addresses), and on the first with the formatless storage-image reads
//! (`Profile F+formatless`), every compiled function:
//!
//! - reaches every buffer slot through a descriptor view, never by device address;
//! - reads no root table and no execution status by address, and needs no pointer translation;
//! - declares no `DeviceAddresses` fact, and every `BufferRangeLimit` it declares is honored by a
//!   plan-time check (the binder's `DescriptorRangeExceeded`), never a device-address variant;
//! - requires neither `bufferDeviceAddress` nor `PhysicalStorageBufferAddresses`.
//!
//! The device without device addresses has the inventory of the formatless one except where a
//! storage texture needs its format as a lowering input, or where a storage heap class, refused on
//! both, names the formatless access among the features it lacks, or where a slot's format is the
//! module's own extended format (a `ulong` texture's `RG32Uint`), which the formatless device
//! declares formatless and the other refuses. An integer imageblock slice takes its texel class
//! from the slot's format on both.
//!
//! The per-construct refusals (a dynamic-root dereference, the per-stage descriptor limit, a
//! `BufferRangeLimit` variant) are asserted on the Profile F devices in
//! `engine_dynamic_roots.rs` and `engine_access.rs`; the device-address-capable
//! Profile F device is checked here.
//!
//! Fixtures are assembled with `llvm-as` (and `xcrun metal-as` for typed-pointer `.air.ll`);
//! without them the tests skip visibly.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::mechanism;
use crate::device::profile::BufferAccess;
use crate::device::requirements::{VkFeature, VkLimit};
use crate::device::SpirvCapability;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{Capability, Error, Feature, Needs};
use crate::library::FunctionArtifact;
use crate::pipeline::facts::{Mechanism, RuntimeFact};
use crate::pipeline::interface::{BufferForm, SlotInterface};
use crate::pipeline::sym::PushField;
use crate::pipeline::testing::{
    compile, engine_dir, fixtures_dir, formatless_floor, link_and_validate, llvm_assemble,
    metal_assemble, meter, AssembleError, FORMATLESS_FLOOR, PROFILE_F,
};

/// The Profile F device with device addresses: `runtime-floor` without `shaderInt64`.
const NO_INT64: &str = "runtime-floor without shaderInt64";

/// [`NO_INT64`]'s description.
fn no_int64() -> DeviceDescription {
    let mut desc = crate::pipeline::testing::description("runtime-floor").expect("profile");
    desc.features.remove(&VkFeature::ShaderInt64);
    desc
}

/// The formatless floor, then the Profile F devices this file checks: `runtime-floor` without
/// device addresses, and [`NO_INT64`].
fn profile_f() -> Vec<(&'static str, DeviceProfile)> {
    let policy = || DevicePolicy::RESIDENT_TABLES;
    let formatless = formatless_floor(policy()).expect("profile");
    let no_addresses = crate::pipeline::testing::device_f(policy()).expect("profile");
    let no_int64 = Device::from_queries(no_int64(), policy()).expect("profile");
    let out = vec![
        (FORMATLESS_FLOOR, formatless),
        (PROFILE_F, no_addresses.profile().clone()),
        (NO_INT64, no_int64.profile().clone()),
    ];
    for (stem, p) in &out {
        assert_eq!(p.fields().buffer_access, BufferAccess::Floor, "{stem}");
    }
    out
}

/// `.ll` files directly in `dir`, sorted by name.
fn ll_files(dir: &Path) -> Vec<PathBuf> {
    let mut paths = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|entry| entry.expect("read fixture directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "ll"))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

/// The bitcode of a public fixture; `None` when its assembler is missing or rejects it (the
/// inventory test records those).
fn assemble_fixture(path: &Path) -> Option<Vec<u8>> {
    let r = if path.to_str().is_some_and(|s| s.ends_with(".air.ll")) {
        metal_assemble(path)
    } else {
        llvm_assemble(path)
    };
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED {}: {why}", path.display());
            None
        }
        Err(AssembleError::Rejected(_)) => None,
    }
}

/// The Profile F contract of one compiled function.
fn check_descriptor_only(a: &FunctionArtifact, p: &DeviceProfile, what: &str) {
    for (slot, s) in &a.interface.slots {
        if let SlotInterface::Buffer(b) = s {
            assert!(
                matches!(b.form, BufferForm::Descriptor),
                "{what}: {slot:?} is {:?}",
                b.form
            );
        }
    }
    assert!(!a.interface.needs_pointer_translation, "{what}");
    for f in &a.interface.push.fields {
        assert!(
            !matches!(
                f.field,
                PushField::RootTableAddress | PushField::ExecutionStatusAddress
            ),
            "{what}: {:?}",
            f.field
        );
    }
    for fact in &a.facts {
        assert!(
            !matches!(fact, RuntimeFact::DeviceAddresses { .. }),
            "{what}: {fact:?}"
        );
        if let RuntimeFact::BufferRangeLimit { .. } = fact {
            assert_eq!(
                mechanism(fact, p),
                Mechanism::PlanTimeCheck,
                "{what}: {fact:?}"
            );
        }
    }
    assert!(
        !a.requirements
            .features
            .contains(&VkFeature::BufferDeviceAddress),
        "{what}"
    );
    assert!(
        !a.requirements
            .spirv_capabilities
            .contains(&SpirvCapability::from_raw(
                crate::spv::Capability::PhysicalStorageBufferAddresses as u32
            )),
        "{what}"
    );
}

/// The inventory cell of an outcome, as the inventory test spells it (without the location).
fn cell(r: &Result<std::sync::Arc<FunctionArtifact>, Error>) -> String {
    match r {
        Ok(_) => "ok".to_owned(),
        Err(Error::Unsupported(u)) => format!("Unsupported{{{:?}; {:?}}}", u.feature, u.needs),
        Err(Error::NeedsInput(i)) => format!("NeedsInput{{{i:?}}}"),
        Err(e) => format!("{e:?}"),
    }
}

/// Whether `floor` is `formatless`'s refusal naming, beside what `formatless` names, only the
/// formatless storage-image features `vk12-floor` lacks: a storage heap class names every feature
/// the profile lacks in one `AnyOf`, the formatless access among them.
fn refused_also_for_formatless(
    formatless: &Result<std::sync::Arc<FunctionArtifact>, Error>,
    floor: &Result<std::sync::Arc<FunctionArtifact>, Error>,
) -> bool {
    let (Err(Error::Unsupported(f)), Err(Error::Unsupported(b))) = (formatless, floor) else {
        return false;
    };
    let (Needs::AnyOf(fa), Needs::AnyOf(ba)) = (&f.needs, &b.needs) else {
        return false;
    };
    let formatless = |c: &Capability| {
        matches!(
            c,
            Capability::Feature(
                VkFeature::ShaderStorageImageReadWithoutFormat
                    | VkFeature::ShaderStorageImageWriteWithoutFormat
            )
        )
    };
    f.feature == b.feature
        && fa.len() == ba.len()
        && fa.iter().zip(ba).all(|(x, y)| {
            x.iter().all(|c| y.contains(c)) && y.iter().all(|c| x.contains(c) || formatless(c))
        })
}

/// Whether `floor` refuses naming formats' formatless storage-image access alone (every way a set
/// of formats read or written without a format): on a description without the formatless
/// features (`runtime-floor` without them) no format states that access, so what the formatless
/// floor accepts through its formatless bits is refused so.
fn refused_for_formatless_access(floor: &Result<std::sync::Arc<FunctionArtifact>, Error>) -> bool {
    use crate::device::requirements::FormatUsage;
    let Err(Error::Unsupported(b)) = floor else {
        return false;
    };
    let Needs::AnyOf(ways) = &b.needs else {
        return false;
    };
    !ways.is_empty()
        && ways.iter().all(|way| {
            !way.is_empty()
                && way.iter().all(|c| {
                    matches!(
                        c,
                        Capability::Format {
                            usage: FormatUsage::StorageImageReadWithoutFormat
                                | FormatUsage::StorageImageWriteWithoutFormat,
                            ..
                        }
                    )
                })
        })
}

/// Whether `floor` refuses the storage format of a slot whose format the module fixes as an
/// extended one (a `ulong` texture is `RG32Uint` whatever is bound): its ways are
/// `shaderStorageImageExtendedFormats`, which neither floor has, or the formatless access the slot
/// uses, which the formatless floor has and `vk12-floor` lacks. The formatless floor declares such
/// a slot formatless, so `vk12-floor` refuses it whatever the formatless floor makes of the
/// function, and lacks whatever else the formatless floor refuses.
fn refused_for_a_fixed_extended_format(
    floor: &Result<std::sync::Arc<FunctionArtifact>, Error>,
) -> bool {
    let Err(Error::Unsupported(b)) = floor else {
        return false;
    };
    let Needs::AnyOf(ways) = &b.needs else {
        return false;
    };
    let extended = Capability::Feature(VkFeature::ShaderStorageImageExtendedFormats);
    let formatless = |c: &Capability| {
        matches!(
            c,
            Capability::Feature(
                VkFeature::ShaderStorageImageReadWithoutFormat
                    | VkFeature::ShaderStorageImageWriteWithoutFormat
            )
        )
    };
    b.feature == Feature::Format
        && ways.len() == 2
        && ways.iter().any(|way| way.len() == 1 && way[0] == extended)
        && ways
            .iter()
            .any(|way| !way.is_empty() && way.iter().all(formatless))
}

/// Every public fixture compiled for a Profile F device uses the static-root descriptor machinery
/// only and passes L3; and the device without device addresses refuses exactly what the formatless
/// floor accepts only through its formatless storage-image bits (asking for a storage format, or
/// naming formats' formatless access), or refuses as the formatless floor does naming those bits
/// too, or refuses a module-fixed extended format that the formatless floor declares formatless,
/// or asks for a storage format where the formatless floor refuses the function.
#[test]
fn profile_f_functions_reach_every_buffer_through_a_descriptor() {
    let profiles = profile_f();
    let mut compiled = 0usize;
    let mut format_needed = 0usize;
    for dir in [fixtures_dir(), engine_dir()] {
        for path in ll_files(&dir) {
            let Some(bc) = assemble_fixture(&path) else {
                continue;
            };
            let Ok(d) = crate::air::decode(&bc, &mut meter()) else {
                continue;
            };
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            for e in d.abi.entries.iter() {
                let entry = e.name.to_string();
                let mut cells = Vec::new();
                let mut results = Vec::new();
                for (stem, p) in &profiles {
                    let what = format!("{name} {entry} on {stem}");
                    let r = compile(&bc, &entry, p);
                    if let Ok(a) = &r {
                        check_descriptor_only(a, p, &what);
                        link_and_validate(a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                        compiled += 1;
                    }
                    cells.push(cell(&r));
                    results.push(r);
                }
                let (formatless, floor) = (&cells[0], &cells[1]);
                if formatless != floor
                    && !refused_also_for_formatless(&results[0], &results[1])
                    && !refused_for_a_fixed_extended_format(&results[1])
                {
                    // The storage format is asked for before a refusal is reached: where the
                    // formatless floor refuses the function, the device without the formatless
                    // reads may ask for the format first.
                    let format_asked = floor.starts_with("NeedsInput{StorageTextureFormat");
                    assert!(
                        (formatless == "ok"
                            && (format_asked || refused_for_formatless_access(&results[1])))
                            || (format_asked && formatless.starts_with("Unsupported{")),
                        "{name} {entry}: {FORMATLESS_FLOOR} {formatless}, {PROFILE_F} {floor}"
                    );
                    format_needed += 1;
                }
            }
        }
    }
    eprintln!(
        "{compiled} Profile F functions compiled; {format_needed} need a format on {PROFILE_F}"
    );
    assert!(compiled > 0, "no fixture compiled");
}

// ---- The device-address-capable Profile F device ----

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-profile-f-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}\n{text}"),
    }
}

/// A kernel over `n` read-write `device uint *` buffers running `body` before `ret void`.
fn kernel(name: &str, n: u32, body: &str) -> String {
    let params: Vec<String> = (0..n)
        .map(|i| format!("ptr addrspace(1) noundef \"air-buffer-no-alias\" %{i}"))
        .collect();
    let refs: Vec<String> = (0..n).map(|i| format!("!{}", 100 + i)).collect();
    let mut meta = String::new();
    for i in 0..n {
        meta.push_str(&format!(
            "!{} = !{{i32 {i}, !\"air.buffer\", !\"air.location_index\", i32 {i}, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"b{i}\"}}\n",
            100 + i
        ));
    }
    format!(
        "{HEADER}define void @{name}({}) {{\nentry:\n{body}  ret void\n}}\n\n\
         !air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @{name}, !10, !11}}\n!10 = !{{}}\n!11 = !{{{}}}\n{meta}\
         !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n",
        params.join(", "),
        refs.join(", ")
    )
}

/// [`NO_INT64`] with at most `storage` storage buffers per stage.
fn no_int64_with(storage: u32) -> DeviceProfile {
    let mut desc = no_int64();
    desc.limits.max_per_stage_descriptor_storage_buffers = storage;
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
        .expect("device")
        .profile()
        .clone()
}

/// [`NO_INT64`] has `bufferDeviceAddress` but no `shaderInt64`, so it is Profile F: a function
/// over more storage buffers than the stage allows is refused naming the limit (it never falls back
/// to device addresses), and a loaded-pointer dereference is refused naming only `shaderInt64`, the
/// one Profile A requirement the device lacks.
#[test]
fn a_profile_f_device_with_device_addresses_still_uses_descriptors_only() {
    let name = "three_stores";
    let body: String = (0..3)
        .map(|i| format!("  store i32 {i}, ptr addrspace(1) %{i}, align 4\n"))
        .collect();
    let Some(bc) = assemble_text(name, &kernel(name, 3, &body)) else {
        return;
    };
    let p = no_int64_with(2);
    assert!(p.fields().buffer_access == BufferAccess::Floor);
    match compile(&bc, name, &p) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::DescriptorCount);
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![vec![Capability::Limit {
                    limit: VkLimit::MaxPerStageDescriptorStorageBuffers,
                    required: 3,
                    available: 2,
                }]])
            );
        }
        other => panic!("expected a descriptor-count refusal, got {other:?}"),
    }
    let a = compile(&bc, name, &no_int64_with(3)).unwrap_or_else(|e| panic!("{e}"));
    check_descriptor_only(&a, &no_int64_with(3), name);

    let name = "chase";
    let body = "  %q = load ptr addrspace(1), ptr addrspace(1) %0, align 8\n  store i32 7, ptr addrspace(1) %q, align 4\n";
    let Some(bc) = assemble_text(name, &kernel(name, 1, body)) else {
        return;
    };
    let p = Device::from_queries(no_int64(), DevicePolicy::RESIDENT_TABLES)
        .expect("profile")
        .profile()
        .clone();
    match compile(&bc, name, &p) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::DereferenceLoadedPointer);
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![vec![Capability::Feature(VkFeature::ShaderInt64)]])
            );
        }
        other => panic!("expected a loaded-pointer refusal, got {other:?}"),
    }
}
