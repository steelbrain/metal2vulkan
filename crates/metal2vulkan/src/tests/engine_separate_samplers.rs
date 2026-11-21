//! Combined image samplers taken apart past the stage's sampler limit.
//!
//! The owned fixtures `smp_constexpr_array` (a fragment function sampling one element of a
//! 32-texture array through one constexpr sampler, the corpus rows' shape), `smp_runtime_array`
//! (a kernel doing so through a runtime sampler), and `smp_shared_sampler` (a fragment function
//! sampling two 16-texture arrays through one constexpr sampler) each need 32 combined image
//! samplers, so 32 samplers and 32 sampled images per stage:
//!
//! - On a profile with 32 samplers or more per stage they keep their 32 combined bindings.
//! - Past it, with 32 sampled images or more, each pair is a separate texture binding (a sampled
//!   image) and the stage has one separate sampler binding per use (a constexpr one its immutable
//!   sampler), sampled through `OpSampledImage`: one sampler, 32 sampled images. The census
//!   counts each sampling arm under `sampler.separate_binding`.
//! - Past both, the stage is refused naming the separate form's limit, the sampled images.
//! - The trigger is the sampler limit alone: under it the output does not depend on the limit.
//! - The binder writes a runtime separate sampler with the sampler bound at its slot, and each
//!   separate texture with the bound texture's view.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use crate::bind::{
    BindState, BoundBuffer, BoundSampler, BoundTexture, NeutralResource, SamplerDescriptor,
    StageTable, TextureFacts, WriteSource,
};
use crate::budget::Budget;
use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkLimit;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Capability, Error, Feature, Needs};
use crate::library::{FunctionArtifact, Library};
use crate::metal::{PixelFormat, SamplerMinMagFilter, TextureType};
use crate::pipeline::layout::BindingSource;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::sym::{ResourceKind, SamplerRef, SlotClass, SlotRef, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, AssembleError,
};
use crate::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};
use crate::spirv::testing::committed_profiles;
use crate::vk;

/// The fixtures, each with its entry, stage, and whether its sampler is a runtime one.
const FIXTURES: [(&str, Stage, bool); 3] = [
    ("smp_constexpr_array", Stage::Fragment, false),
    ("smp_runtime_array", Stage::Kernel, true),
    ("smp_shared_sampler", Stage::Fragment, false),
];

/// The combined image samplers each fixture needs.
const PAIRS: u64 = 32;

const OP_SAMPLED_IMAGE: u16 = 86;

fn fixture(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

/// `moltenvk-m3`, its limits changed by `edit`.
fn m3_with(edit: impl FnOnce(&mut crate::device::profile::Limits)) -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    edit(&mut desc.limits);
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device")
}

fn opcodes(words: &[u32]) -> Vec<u16> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = (words[i] >> 16) as usize;
        out.push((words[i] & 0xFFFF) as u16);
        i += n.max(1);
    }
    out
}

fn count(a: &FunctionArtifact, op: u16) -> usize {
    opcodes(&a.module.words)
        .iter()
        .filter(|o| **o == op)
        .count()
}

fn uses(a: &FunctionArtifact, name: &str) -> u32 {
    a.strategies.entries.get(name).map_or(0, |u| u.uses)
}

/// The module's descriptors of each kind: combined, separate textures, separate samplers.
fn kinds(a: &FunctionArtifact) -> (usize, usize, Vec<SamplerRef>) {
    let mut combined = 0;
    let mut textures = 0;
    let mut samplers = Vec::new();
    for d in &a.interface.descriptors {
        match d.slot.kind {
            ResourceKind::CombinedTextureSampler(_) => combined += 1,
            ResourceKind::SeparateTexture(_) => {
                assert_eq!(d.ty, vk::DescriptorType::SAMPLED_IMAGE);
                textures += 1;
            }
            ResourceKind::SeparateSampler(s) => {
                assert_eq!(d.ty, vk::DescriptorType::SAMPLER);
                samplers.push(s);
            }
            _ => {}
        }
    }
    (combined, textures, samplers)
}

/// On every committed profile each fixture keeps its 32 combined bindings within the stage's
/// sampler limit, takes them apart past it (32 separate textures, one separate sampler, a constexpr
/// one the linked layout's immutable sampler, each sampling arm through `OpSampledImage`), and is
/// refused naming the sampled-image limit where the separate form exceeds it too. Every compiled
/// module links and passes `spirv-val` under its profile.
#[test]
fn the_stage_takes_its_combined_bindings_apart_past_the_sampler_limit() {
    let mut seen = (0, 0, 0);
    for (name, _stage, runtime) in FIXTURES {
        let Some(bc) = fixture(name) else {
            return;
        };
        for (stem, p) in committed_profiles().expect("profiles") {
            let l = &p.fields().limits;
            let samplers = u64::from(l.max_per_stage_descriptor_samplers);
            let images = u64::from(l.max_per_stage_descriptor_sampled_images);
            let what = format!("{name} on {stem}");
            let a = match compile(&bc, name, &p) {
                Ok(a) => a,
                Err(Error::Unsupported(u)) if PAIRS > samplers && PAIRS > images => {
                    assert_eq!(u.feature, Feature::DescriptorCount, "{what}");
                    assert_eq!(
                        u.needs,
                        Needs::AnyOf(vec![vec![Capability::Limit {
                            limit: VkLimit::MaxPerStageDescriptorSampledImages,
                            required: PAIRS,
                            available: images,
                        }]]),
                        "{what}"
                    );
                    seen.2 += 1;
                    continue;
                }
                Err(e) => panic!("{what}: {e}"),
            };
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let (combined, textures, separate) = kinds(&a);
            if PAIRS <= samplers {
                assert_eq!((combined, textures), (32, 0), "{what}");
                assert!(separate.is_empty(), "{what}");
                assert_eq!(count(&a, OP_SAMPLED_IMAGE), 0, "{what}");
                assert_eq!(uses(&a, "sampler.separate_binding"), 0, "{what}");
                seen.0 += 1;
                continue;
            }
            assert!(PAIRS <= images, "{what} compiled past both limits");
            assert_eq!((combined, textures), (0, 32), "{what}");
            assert_eq!(separate.len(), 1, "{what}");
            assert_eq!(
                matches!(separate[0], SamplerRef::Slot(0)),
                runtime,
                "{what}: {separate:?}"
            );
            // One sampled image per pair: its sample's.
            let made = count(&a, OP_SAMPLED_IMAGE);
            assert_eq!(made, 32, "{what}");
            assert_eq!(uses(&a, "sampler.separate_binding"), made as u32, "{what}");
            let linked = link(&[&a], &LinkInputs::Standalone, &p).expect("link");
            let bindings = &linked.layout.sets[0].bindings;
            let sampler: Vec<_> = bindings
                .iter()
                .filter(|b| matches!(b.source, BindingSource::SeparateSampler { .. }))
                .collect();
            assert_eq!(sampler.len(), 1, "{what}");
            assert_eq!(sampler[0].ty, vk::DescriptorType::SAMPLER, "{what}");
            assert_eq!(sampler[0].immutable_sampler.is_none(), runtime, "{what}");
            let images: Vec<_> = bindings
                .iter()
                .filter(|b| matches!(b.source, BindingSource::SeparateTexture { .. }))
                .collect();
            assert_eq!(images.len(), 32, "{what}");
            assert!(images
                .iter()
                .all(|b| b.ty == vk::DescriptorType::SAMPLED_IMAGE
                    && b.immutable_sampler.is_none()
                    && b.count == 1));
            seen.1 += 1;
        }
    }
    assert!(
        seen.0 > 0 && seen.1 > 0 && seen.2 > 0,
        "combined, separate, refused: {seen:?}"
    );
}

/// The trigger is the sampler limit alone. With `moltenvk-m3`'s limit at 32 the 32 pairs stay
/// combined, and the module is the one a limit of 1000 gives, word for word (under the limit the
/// output does not depend on it); at 31 they are taken apart.
#[test]
fn the_trigger_is_the_sampler_limit() {
    for (name, _, _) in FIXTURES {
        let Some(bc) = fixture(name) else {
            return;
        };
        let at = |n: u32| {
            let dev = m3_with(|l| l.max_per_stage_descriptor_samplers = n);
            let p: DeviceProfile = dev.profile().clone();
            compile(&bc, name, &p).unwrap_or_else(|e| panic!("{name} at {n}: {e}"))
        };
        let (a32, a1000, a31) = (at(32), at(1000), at(31));
        assert_eq!(kinds(&a32).0, 32, "{name}");
        assert_eq!(a32.module.words, a1000.module.words, "{name}");
        assert_eq!(kinds(&a31).1, 32, "{name}");
        assert_ne!(a31.module.words, a32.module.words, "{name}");
    }
}

/// `smp_runtime_array` on `moltenvk-m3` samples 32 separate textures through one runtime separate
/// sampler. The plan writes the sampler with the one bound at the kernel's slot 0, made for the use
/// (the neutral sampler where none is bound), and each separate texture with its bound texture's
/// view; a texture slot left unbound reads the neutral image.
#[test]
fn the_binder_writes_a_runtime_separate_sampler_and_its_textures() {
    let Some(bc) = fixture("smp_runtime_array") else {
        return;
    };
    let dev = m3_with(|_| {});
    let lib = Library::from_air(&bc).expect("library");
    let f = lib.function("smp_runtime_array").expect("entry");
    let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    let pipe = dev
        .compile_compute_pipeline(&d, &Budget::standard())
        .expect("pipeline");
    let sampler_binding = pipe.layout.sets[0]
        .bindings
        .iter()
        .find(|b| matches!(b.source, BindingSource::SeparateSampler { .. }))
        .expect("separate sampler")
        .binding;
    let state = |bound: u32, sampler: Option<BoundSampler>| {
        let mut t = StageTable::default();
        for i in 0..bound {
            t.textures.insert(
                i,
                BoundTexture::Texture {
                    facts: TextureFacts::new(TextureType::Type2D, PixelFormat::RGBA16Float),
                    resource_id: None,
                },
            );
        }
        if let Some(s) = sampler {
            t.samplers.insert(0, s);
        }
        t.buffers.insert(
            0,
            BoundBuffer::Buffer {
                length: 32 * 16,
                offset: 0,
                gpu_address: 0x1000_0000,
                device_address: None,
                attribute_stride: None,
            },
        );
        let mut state = BindState::default();
        state.stages.insert(Stage::Kernel, t);
        state
    };
    let mut linear = SamplerDescriptor::new();
    linear.min_filter = SamplerMinMagFilter::Linear;
    linear.mag_filter = SamplerMinMagFilter::Linear;
    let s0 = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Sampler,
        index: 0,
    };
    for (bound, sampler, want) in [
        (
            32,
            Some(BoundSampler::new(linear)),
            WriteSource::Sampler {
                slot: s0,
                compare: false,
                integer: false,
                opaque_black: crate::metal::sampler::OpaqueBlackBorder::Stated,
            },
        ),
        (
            31,
            None,
            WriteSource::Neutral(NeutralResource::Sampler { compare: false }),
        ),
    ] {
        let plan = pipe
            .plan(&dev, &state(bound, sampler).shape())
            .expect("plan");
        let written: Vec<_> = plan
            .descriptors
            .iter()
            .filter(|w| w.set == 0 && w.binding == sampler_binding)
            .map(|w| w.source.clone())
            .collect();
        assert_eq!(written, vec![want]);
        let mut textures = 0;
        let mut neutral = 0;
        for w in &plan.descriptors {
            match &w.source {
                WriteSource::Texture { slot, .. } if slot.class == SlotClass::Texture => {
                    assert_eq!(w.ty, vk::DescriptorType::SAMPLED_IMAGE);
                    textures += 1;
                }
                WriteSource::Neutral(NeutralResource::Image { .. }) => neutral += 1,
                WriteSource::CombinedImageSampler { .. } => panic!("a combined write"),
                _ => {}
            }
        }
        assert_eq!((textures, neutral), (bound, 32 - bound));
    }
}
