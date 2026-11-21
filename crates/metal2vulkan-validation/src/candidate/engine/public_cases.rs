//! The committed public kernel cases with textures or samplers through the engine's
//! device-free half: each is mapped, compiled for committed profiles with the inputs its bind state
//! decides (`variants`), and planned by the binder, or refused typed. No device is opened.

use super::*;
use crate::check::loop_free::DecodedModule;
use crate::source::public_sources;
use metal2vulkan::bind::{ImageSource, WriteSource};
use metal2vulkan::device::FormatUsage;
use metal2vulkan::device::{Device, DeviceDescription, DevicePolicy};
use metal2vulkan::error::{BindError, Capability, Feature, Needs, Unsupported};
use metal2vulkan::library::Library;
use metal2vulkan::metal::format::vk_format;
use metal2vulkan::pipeline::FunctionDescriptor;
use metal2vulkan::Budget;
use metal2vulkan_devtools::{llvm_assemble, AssembleError};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn device(stem: &str) -> Device {
    let path = crate::engine::profiles_dir().join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).unwrap();
    let description = DeviceDescription::from_khronos_json(&text, None).unwrap();
    Device::from_queries(description, DevicePolicy::RESIDENT_TABLES).unwrap()
}

/// The committed cases whose AIR is a public fixture, with that fixture's source row.
fn public_cases() -> Vec<(AuthoredCase, crate::source::SourceRow)> {
    let root = metal2vulkan_devtools::validation_dir().join("corpus");
    let store = CorpusStore::new(&root);
    let sources = public_sources().unwrap();
    let mut out = Vec::new();
    for shard in 0..64 {
        for case in store.read_case_shard(shard).unwrap() {
            // The row of the case's module and entry.
            if let Some(row) = sources
                .iter()
                .find(|row| row.air_sha256 == case.air_sha256 && row.entry == case.entry)
            {
                out.push((case, row.clone()));
            }
        }
    }
    out
}

/// How one case ended in the device-free half.
enum Planned {
    /// Refused typed before a device was involved (the mapping or the product's compile).
    Refused,
    /// No `llvm-as` is installed (the only reason the test skips).
    NoAssembler(String),
    /// The installed `llvm-as` rejected the fixture's text (an older LLVM than the one it was
    /// written for): that one case is skipped, with a printed line.
    Unassembled(String),
    /// Planned: the families of its descriptor writes, and "pipeline variant" when its plan
    /// needed one.
    Writes(BTreeSet<&'static str>),
}

/// Whether every alternative of `sets` needs some format's support, and `device`'s profile
/// cannot use one of the formats `case` binds a texture in as `sets` names (or sample it).
fn lacks_a_bound_format(device: &Device, case: &AuthoredCase, sets: &[Vec<Capability>]) -> bool {
    // The usages the refusal names (linear filtering, say), and sampling itself.
    let usages: BTreeSet<FormatUsage> = sets
        .iter()
        .flatten()
        .filter_map(|capability| match capability {
            Capability::Format { usage, .. } => Some(*usage),
            _ => None,
        })
        .chain([FormatUsage::SampledImage])
        .collect();
    sets.iter().all(|set| {
        set.iter()
            .any(|capability| matches!(capability, Capability::Format { .. }))
    }) && case.textures.iter().any(|texture| {
        usages.iter().any(|&usage| {
            vk_format(
                resources::pixel_format(texture.format),
                &BTreeSet::from([usage]),
                device.profile(),
            )
            .is_err()
        })
    })
}

/// The AIR-layer interface of `case`'s entry in `bitcode`, as the checker derives it.
fn interface(case: &AuthoredCase, bitcode: &[u8]) -> AirInterface {
    let decoded = DecodedModule::from_bitcode(&case.entry, bitcode).unwrap();
    let constants = crate::check::case_constants(case).unwrap();
    AirInterface::new(
        decoded.module(),
        decoded.abi().unwrap(),
        &case.entry,
        &constants,
    )
    .unwrap()
}

fn plan_case(device: &Device, case: &AuthoredCase, row: &crate::source::SourceRow) -> Planned {
    let scratch = crate::ScratchDir::new("public-texture-case").unwrap();
    let ll = scratch.path().join("case.ll");
    std::fs::write(&ll, &row.air_ll).unwrap();
    let bitcode = match llvm_assemble(&ll) {
        Ok(bitcode) => bitcode,
        Err(AssembleError::Missing(why)) => return Planned::NoAssembler(why),
        Err(AssembleError::Rejected(why)) => return Planned::Unassembled(why),
    };
    let Ok(compute) = compute_case(case, &interface(case, &bitcode)) else {
        return Planned::Refused;
    };
    let library = Library::from_air(&bitcode).unwrap();
    let mut function = FunctionDescriptor::new(library.function(&compute.entry).unwrap());
    function.constants = compute.constants.clone();
    let shape = compute_shape(&compute, true, BufferRanges::default()).unwrap();
    let (pipeline, inputs) = match variants::compile_compute_for_shape(
        device,
        &function,
        &shape,
        Some(&compute.dispatch),
        compute.stage_input.as_ref(),
        &Budget::standard(),
    ) {
        Ok(compiled) => compiled,
        Err(error) if !Outcome::from(&error).is_defect() => return Planned::Refused,
        Err(error) => panic!("{}: compile: {error:?}", case.name),
    };
    let addresses = compute
        .buffers
        .keys()
        .map(|&index| {
            let address = 0x1000_0000 * u64::from(index + 1);
            (
                index,
                BufferAddress {
                    guest: address,
                    host: Some(address),
                },
            )
        })
        .collect();
    let embedded = compute
        .argument_buffer_buffers
        .keys()
        .enumerate()
        .map(|(token, &key)| {
            let address = 0x8000_0000_0000 + 0x1000_0000 * token as u64;
            (
                key,
                BufferAddress {
                    guest: address,
                    host: Some(address),
                },
            )
        })
        .collect();
    let state = bind_state(&compute, &addresses, &embedded, BufferRanges::default()).unwrap();
    // The executor's shape is the one it compiles for.
    assert_eq!(state.shape(), shape, "{}", case.name);
    let plan = match pipeline.plan(device, &state.shape()) {
        Ok(plan) => plan,
        // A sparse texture whose residency the function observes plans only where the device
        // states shader residency: elsewhere the binder refuses it typed, as the executor reports
        // it.
        Err(BindError::Unsupported(Unsupported {
            feature: Feature::SparseTexture { slot },
            ..
        })) if case
            .textures
            .iter()
            .any(|texture| texture.sparse.is_some() && texture.binding == slot.index) =>
        {
            return Planned::Refused;
        }
        // A texture of a format the profile cannot sample, or filter as the case's sampler
        // does, is refused typed as the device's gap (`Feature::Format`, needing that format's
        // support): the case's literal binds it, and no
        // other mapping of the case changes it (the sRGB and packed formats).
        Err(BindError::Unsupported(Unsupported {
            feature: Feature::Format,
            needs: Needs::AnyOf(sets),
            ..
        })) if lacks_a_bound_format(device, case, &sets) => {
            return Planned::Refused;
        }
        Err(error) => panic!("{}: plan: {error}", case.name),
    };
    // The dispatch plans the variant its pipeline was compiled for; the executor refuses
    // diagnostics, and no public case has one.
    assert!(
        variants::compiled_for(&inputs, &plan.variant),
        "{}: planned variant {:?}, compiled {inputs:?}",
        case.name,
        plan.variant
    );
    assert!(
        plan.diagnostics.is_empty(),
        "{}: {:?}",
        case.name,
        plan.diagnostics
    );
    // A plan that needed a variant counts as its own family, so the lowering-input path is
    // known to run.
    let variant = (!plan.variant.is_empty()).then_some("pipeline variant");
    Planned::Writes(
        plan.descriptors
            .iter()
            .map(|write| match &write.source {
                WriteSource::Texture { .. }
                    if write.ty == metal2vulkan::vk::DescriptorType::STORAGE_IMAGE =>
                {
                    "storage texture"
                }
                WriteSource::Texture { .. } => "texture",
                WriteSource::CombinedImageSampler { .. } => "combined image sampler",
                WriteSource::Sampler { .. } => "sampler",
                WriteSource::Neutral(_) => "neutral",
                _ => "buffer-like",
            })
            .chain(variant)
            .collect(),
    )
}

/// Every public kernel case with textures or samplers that maps and compiles for its bind state
/// plans the variant it was compiled for and no plan diagnostic (which the executor refuses), and
/// together they write a sampled texture, a storage texture and a sampler. On a profile that needs
/// storage formats as inputs, some of them plan a variant. At least one case plans on every
/// profile; the test skips only when no
/// `llvm-as` is installed, and a case whose fixture the installed `llvm-as` rejects is skipped
/// alone, with a printed line.
#[test]
fn public_texture_cases_plan_their_compiled_variant_without_diagnostics() {
    let cases = public_cases();
    for stem in ["lavapipe", "moltenvk-m3", "runtime-floor"] {
        let device = device(stem);
        let mut families = BTreeSet::new();
        let mut planned = 0;
        for (case, row) in &cases {
            if case.stage != Stage::Kernel || (case.textures.is_empty() && case.samplers.is_empty())
            {
                continue;
            }
            match plan_case(&device, case, row) {
                Planned::Refused => {}
                Planned::Unassembled(why) => {
                    eprintln!("SKIPPED {}: llvm-as rejected its fixture: {why}", case.name);
                }
                Planned::NoAssembler(why) => {
                    eprintln!("SKIPPED: {why}");
                    return;
                }
                Planned::Writes(writes) => {
                    planned += 1;
                    families.extend(writes);
                }
            }
        }
        // Refusing every case would pass the checks below vacuously.
        assert!(planned > 0, "{stem}: no public texture case planned");
        for family in ["texture", "storage texture"] {
            assert!(families.contains(family), "{stem}: no {family} write");
        }
        assert!(
            families.contains("sampler") || families.contains("combined image sampler"),
            "{stem}: no sampler write"
        );
        // The runtime floor reads storage images only with a format: its storage cases compile
        // with an input and plan that variant.
        if stem == "runtime-floor" {
            assert!(families.contains("pipeline variant"), "{stem}: no variant");
        }
    }
}

/// Every public kernel case over texture arrays maps each element to the texture at its slot
/// and, compiled for its bind state, plans a texture write of exactly the element slots its
/// function reads, with no neutral and no diagnostic: the product reads an element through its
/// slot's own binding, which the executor binds as the Metal oracle does. The elements each case's
/// function reads, as positions in the element slots of its arrays in binding order, are stated per
/// entry, so a new such case must say which:
/// - `texture_array_width` (`array<texture2d<float>, 2> [[texture(0)]]`) reads `textures[1]`;
/// - `a4_texture_array_dynamic` (`array<texture2d<float>, 4> [[texture(0)]]`) reads
///   `ts[gid & 3]`, every element;
/// - `ob3_texture_array` (two `array<texture2d<uint>, 2>` at `[[texture(0)]]` and
///   `[[texture(2)]]`) reads `tex[idx[t]]`, clamped onto both of its elements, and `spare[t & 1]`,
///   every element of both;
/// - `smp_runtime_array` (`array<texture2d<float>, 32> [[texture(0)]]`) samples `t[gid & 31]`,
///   every element, each through a combined image sampler or, on `moltenvk-m3`, whose 16
///   samplers per stage the 32 pairs exceed, a separate texture. A sampled element's texture is the
///   image of its combined write, or its separate texture's write.
#[test]
fn public_texture_array_cases_plan_their_elements_at_their_slots() {
    const EVERY_32: [u32; 32] = {
        let mut every = [0; 32];
        let mut i = 0;
        while i < 32 {
            every[i] = i as u32;
            i += 1;
        }
        every
    };
    let read_elements = |entry: &str| -> Option<&'static [u32]> {
        match entry {
            "texture_array_width" => Some(&[1]),
            "a4_texture_array_dynamic" | "ob3_texture_array" => Some(&[0, 1, 2, 3]),
            "smp_runtime_array" => Some(&EVERY_32),
            _ => None,
        }
    };
    let cases = public_cases();
    let arrays = cases
        .iter()
        .filter(|(case, _)| case.stage == Stage::Kernel && !case.texture_arrays.is_empty())
        .collect::<Vec<_>>();
    assert!(
        arrays.len() >= 2,
        "the public kernel cases with a texture array"
    );
    for (case, row) in arrays {
        let read = read_elements(&case.entry).unwrap_or_else(|| {
            panic!(
                "{}: state which texture-array elements {} reads",
                case.name, case.entry
            )
        });
        let scratch = crate::ScratchDir::new("public-texture-array-case").unwrap();
        let ll = scratch.path().join("case.ll");
        std::fs::write(&ll, &row.air_ll).unwrap();
        let bitcode = match llvm_assemble(&ll) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => {
                panic!("{}: llvm-as rejected it: {why}", case.name)
            }
        };
        let compute = compute_case(case, &interface(case, &bitcode)).unwrap();
        let slots = case
            .texture_arrays
            .iter()
            .flat_map(|array| {
                (0..array.elements.len() as u32).map(move |element| array.binding + element)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            compute.textures.keys().copied().collect::<Vec<_>>(),
            slots,
            "{}",
            case.name
        );
        let library = Library::from_air(&bitcode).unwrap();
        let mut function = FunctionDescriptor::new(library.function(&compute.entry).unwrap());
        function.constants = compute.constants.clone();
        for stem in ["lavapipe", "moltenvk-m3", "runtime-floor"] {
            let device = device(stem);
            let shape = compute_shape(&compute, true, BufferRanges::default()).unwrap();
            let compiled = variants::compile_compute_for_shape(
                &device,
                &function,
                &shape,
                None,
                None,
                &Budget::standard(),
            );
            let sampled = device
                .profile()
                .limits()
                .max_per_stage_descriptor_sampled_images;
            let (pipeline, inputs) = match compiled {
                // More elements than the stage's sampled images (the runtime floor's 16): refused
                // typed, naming the limit.
                Err(error) if slots.len() > sampled as usize => {
                    let text = format!("{error:?}");
                    assert!(
                        text.contains("MaxPerStageDescriptorSampledImages"),
                        "{} {stem}: {text}",
                        case.name
                    );
                    continue;
                }
                compiled => compiled
                    .unwrap_or_else(|error| panic!("{} {stem}: compile: {error:?}", case.name)),
            };
            let plan = pipeline.plan(&device, &shape).unwrap();
            assert!(
                variants::compiled_for(&inputs, &plan.variant),
                "{} {stem}",
                case.name
            );
            assert!(
                plan.diagnostics.is_empty(),
                "{} {stem}: {:?}",
                case.name,
                plan.diagnostics
            );
            let textures = plan
                .descriptors
                .iter()
                .filter_map(|write| match &write.source {
                    WriteSource::Texture { slot, .. }
                    | WriteSource::CombinedImageSampler {
                        image: ImageSource::Texture { slot, .. },
                        ..
                    } => Some(slot.index),
                    WriteSource::Neutral(neutral)
                    | WriteSource::CombinedImageSampler {
                        image: ImageSource::Neutral(neutral),
                        ..
                    } => {
                        panic!("{} {stem}: a neutral {neutral:?}", case.name)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            let expected = read
                .iter()
                .map(|&element| slots[element as usize])
                .collect::<Vec<_>>();
            assert_eq!(textures, expected, "{} {stem}", case.name);
        }
    }
}

/// Each public kernel case whose root argument points to further nodes (argument-buffer buffers)
/// maps, compiles for its bind state, and plans on devices with device addresses; its dispatch's
/// root table holds each pointee's resident entry (host and guest address), which is what a loaded
/// pointer is translated through. The case is the one-member case: the rate-map case whose maps are
/// members is the runtime's limit (`ToolingRequirement::RasterizationRateMap`).
#[test]
fn the_public_argument_buffer_cases_plan_their_pointees_resident() {
    let cases = public_cases();
    let selected = cases
        .iter()
        .filter(|(case, _)| {
            case.stage == Stage::Kernel
                && !case.argument_buffer_buffers.is_empty()
                && crate::executor_contract::case_requirements(case).is_empty()
        })
        .collect::<Vec<_>>();
    assert!(
        !selected.is_empty(),
        "the public kernel cases with argument-buffer buffers"
    );
    for (case, row) in selected {
        let scratch = crate::ScratchDir::new("public-argument-buffer-case").unwrap();
        let ll = scratch.path().join("case.ll");
        std::fs::write(&ll, &row.air_ll).unwrap();
        let bitcode = match llvm_assemble(&ll) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => {
                panic!("{}: llvm-as rejected it: {why}", case.name)
            }
        };
        let compute = compute_case(case, &interface(case, &bitcode)).unwrap();
        assert_eq!(
            compute.argument_buffer_buffers.len(),
            case.argument_buffer_buffers.len(),
            "{}",
            case.name
        );
        let library = Library::from_air(&bitcode).unwrap();
        let mut function = FunctionDescriptor::new(library.function(&compute.entry).unwrap());
        function.constants = compute.constants.clone();
        let address = |guest: u64| BufferAddress {
            guest,
            host: Some(guest + 0x10_0000_0000),
        };
        let addresses = compute
            .buffers
            .keys()
            .map(|&index| (index, address(0x1000_0000 * u64::from(index + 1))))
            .collect();
        // Each member's pointee at its own address.
        let embedded = compute
            .argument_buffer_buffers
            .keys()
            .zip(0u64..)
            .map(|(&key, ordinal)| (key, address(0x7000_0000 + 0x100_0000 * ordinal)))
            .collect::<BTreeMap<_, _>>();
        let state = bind_state(&compute, &addresses, &embedded, BufferRanges::default()).unwrap();
        for stem in ["lavapipe", "moltenvk-m3"] {
            let device = device(stem);
            let shape = compute_shape(&compute, true, BufferRanges::default()).unwrap();
            assert_eq!(state.shape(), shape);
            let (pipeline, inputs) = variants::compile_compute_for_shape(
                &device,
                &function,
                &shape,
                Some(&compute.dispatch),
                None,
                &Budget::standard(),
            )
            .unwrap_or_else(|error| panic!("{} {stem}: compile: {error:?}", case.name));
            let plan = pipeline.plan(&device, &shape).unwrap();
            assert!(variants::compiled_for(&inputs, &plan.variant), "{stem}");
            assert!(
                plan.diagnostics.is_empty(),
                "{} {stem}: {:?}",
                case.name,
                plan.diagnostics
            );
            let commands = plan
                .instantiate_dispatch(&state, &compute.dispatch)
                .unwrap();
            let table = commands
                .root_table
                .expect("a loaded pointer reads the root table");
            // A root-table entry is 32 bytes: host, guest, size, and bind offset.
            let word =
                |entry: &[u8], at: usize| u64::from_le_bytes(entry[at..at + 8].try_into().unwrap());
            let entries: Vec<[u64; 2]> = table
                .bytes
                .as_chunks::<32>()
                .0
                .iter()
                .map(|entry| [word(entry, 0), word(entry, 8)])
                .collect();
            for (key, pointee) in &embedded {
                assert!(
                    entries.contains(&[pointee.host.unwrap(), pointee.guest]),
                    "{} {stem}: the pointee of {key:?} is not resident",
                    case.name
                );
            }
        }
    }
}
