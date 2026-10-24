//! Pulled vertex fetch end to end.
//!
//! The owned fixture `engine/stage_io`'s `stage_io_vertex` reads four attributes (`float3`,
//! `half2`, `ushort2`, `uchar4`). Compiled under projections that pull them from source buffers
//! with each step (per vertex, per instance at rates 1 and 3, constant), the module passes L3
//! (`spirv-val` and requirements within the profile) on every committed profile it compiles on,
//! and the artifact states what the binder must supply: a storage descriptor per source buffer,
//! the bind residual and stride push fields, the base instance by builtin or push field by the
//! profile's `shaderDrawParameters`, and the runtime facts. A source buffer that is also the
//! function's `scale` argument shares the argument's slot, widened to `Whole`; source buffers take
//! device addresses in a `BufferRangeLimit` variant and under the per-stage descriptor-limit
//! fallback like any storage slot. The fetch itself is checked on the CPU interpreter in
//! `lower::stage_io::fetch`'s unit tests; whole vertex functions run on it here to show the fetch
//! precedes the body's writes, which may then reach a shared slot. No GPU is used.
//!
//! `llvm-as` assembles the fixture; a missing tool skips visibly.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::budget::SPIRV_WORDS;
use crate::device::profile::BufferAccess;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{Error, Feature, LowerFamily, Needs};
use crate::library::{FunctionArtifact, FunctionConstantValues};
use crate::lower::select::Plan;
use crate::metal::{VertexFormat, VertexStepFunction};
use crate::pipeline::compile::{compile_bitcode, plan_bitcode, EntryRef};
use crate::pipeline::descriptor::{
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::{
    AccessSet, BufferForm, BufferSlot, DescriptorView, PushLoc, SlotInterface, Window,
};
use crate::pipeline::link::vertex::vertex_fetch_projection;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::lower_inputs::{LoweringInputs, VertexFetch, VertexFetchProjection};
use crate::pipeline::relocate::relocate;
use crate::pipeline::stage_io::{IoNumeric, PulledInput, StageBuiltin};
use crate::pipeline::sym::{
    PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, Stage, Symbol,
};
use crate::pipeline::testing::{
    engine_dir, llvm_assemble, meter, profile, typed_module_with, AssembleError, Scratch,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::testing::check_words_valid;
use crate::vk;

const PROFILES: [&str; 3] = ["moltenvk-m3", "lunarg-desktop-baseline-2023", "vk12-floor"];
const ENTRY: &str = "stage_io_vertex";

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("stage_io.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected stage_io.ll: {why}"),
    }
}

fn prof(stem: &str) -> DeviceProfile {
    profile(stem, DevicePolicy::default()).expect("committed profile")
}

fn inputs(proj: VertexFetchProjection) -> LoweringInputs {
    let mut i = LoweringInputs::new();
    i.vertex_fetch = Some(proj);
    i
}

fn compile_with(
    bc: &[u8],
    p: &DeviceProfile,
    proj: VertexFetchProjection,
) -> Result<Arc<FunctionArtifact>, Error> {
    compile_inputs(bc, p, &inputs(proj))
}

fn compile_inputs(
    bc: &[u8],
    p: &DeviceProfile,
    inputs: &LoweringInputs,
) -> Result<Arc<FunctionArtifact>, Error> {
    compile_bitcode(
        bc,
        EntryRef::Name(ENTRY),
        &FunctionConstantValues::defaults(),
        p,
        inputs,
        &mut meter(),
        SPIRV_WORDS,
    )
}

fn layout(step: VertexStepFunction, rate: u32) -> VertexBufferLayoutDescriptor {
    let mut l = VertexBufferLayoutDescriptor::new(Some(32));
    l.step_function = step;
    l.step_rate = rate;
    l
}

fn pulled(
    format: VertexFormat,
    offset: u32,
    buffer: u32,
    step: VertexStepFunction,
    rate: u32,
) -> VertexFetch {
    VertexFetch::pulled(
        &VertexAttributeDescriptor::new(format, offset, buffer),
        &layout(step, rate),
    )
}

/// Every attribute of the fixture pulled: the position and uv per vertex from buffer 0, the ids
/// per instance at `rate` from buffer 2, the tint constant from buffer 3 (buffer 1 is the
/// function's `scale` argument).
fn all_pulled(rate: u32) -> VertexFetchProjection {
    use VertexStepFunction as S;
    let mut proj = VertexFetchProjection::new();
    proj.attributes
        .insert(0, pulled(VertexFormat::Float3, 0, 0, S::PerVertex, 1));
    proj.attributes
        .insert(1, pulled(VertexFormat::Half2, 12, 0, S::PerVertex, 1));
    proj.attributes
        .insert(4, pulled(VertexFormat::UShort2, 2, 2, S::PerInstance, rate));
    proj.attributes
        .insert(7, pulled(VertexFormat::UChar4, 0, 3, S::Constant, 0));
    proj
}

/// [`all_pulled`] with the position and uv read from buffer 1, the function's `scale` argument
/// (the ids from buffer 2, the tint from buffer 3).
fn shared_pulled(rate: u32) -> VertexFetchProjection {
    use VertexStepFunction as S;
    let mut proj = all_pulled(rate);
    proj.attributes
        .insert(0, pulled(VertexFormat::Float3, 0, 1, S::PerVertex, 1));
    proj.attributes
        .insert(1, pulled(VertexFormat::Half2, 12, 1, S::PerVertex, 1));
    proj
}

/// [`all_pulled`] with the position and uv in normalized formats (signed 10:10:10:2 and SNORM16),
/// whose correctly rounded quotients are the dearest per-lane decode.
fn normalized_pulled(rate: u32) -> VertexFetchProjection {
    use VertexStepFunction as S;
    let mut proj = all_pulled(rate);
    proj.attributes.insert(
        0,
        pulled(VertexFormat::Int1010102Normalized, 0, 0, S::PerVertex, 1),
    );
    proj.attributes.insert(
        1,
        pulled(VertexFormat::Short2Normalized, 12, 0, S::PerVertex, 1),
    );
    proj
}

/// Buffers 0 (its own) and 1 (shared with `scale`) pulled, 2 and 3 as in [`all_pulled`].
fn both_pulled(rate: u32) -> VertexFetchProjection {
    use VertexStepFunction as S;
    let mut proj = all_pulled(rate);
    proj.attributes
        .insert(0, pulled(VertexFormat::Float3, 0, 1, S::PerVertex, 1));
    proj
}

fn buffer(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Vertex,
        kind: ResourceKind::Buffer,
        index,
    }
}

fn slot(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Vertex,
        class: SlotClass::Buffer,
        index,
    }
}

/// The class the fixture declares each attribute it reads as (`float3`, `half2`, `ushort2`,
/// `uchar4`; integers are held as `uint`).
fn classes() -> BTreeMap<u32, IoNumeric> {
    BTreeMap::from([
        (0, IoNumeric::Float),
        (1, IoNumeric::Float),
        (4, IoNumeric::Uint),
        (7, IoNumeric::Uint),
    ])
}

/// The attribute indices of pulled inputs, in order.
fn indices(pulled: &[PulledInput]) -> Vec<u32> {
    pulled.iter().map(|x| x.index).collect()
}

/// Whether the module decorates the variable bound at `sym` `NonWritable`.
fn non_writable(a: &FunctionArtifact, sym: ResourceSlot) -> bool {
    let words = &a.module.words;
    let decorate = (3 << 16) | crate::spv::Op::Decorate as u32;
    let nw = crate::spv::Decoration::NonWritable as u32;
    a.module.relocs.iter().any(|(s, at)| {
        *s == Symbol::Binding(sym)
            && (*at as usize)
                .checked_sub(2)
                .and_then(|i| words.get(i))
                .is_some_and(|id| {
                    words
                        .windows(3)
                        .any(|w| w[0] == decorate && w[1] == *id && w[2] == nw)
                })
    })
}

fn placed(a: &FunctionArtifact, f: PushField) -> bool {
    a.interface.push.fields.iter().any(|x| x.field == f)
}

/// The buffer slot the artifact states for Metal index `b`.
fn buffer_slot(a: &FunctionArtifact, b: u32) -> Option<BufferSlot> {
    match a.interface.slots.get(&slot(b)) {
        Some(SlotInterface::Buffer(s)) => Some(*s),
        _ => None,
    }
}

/// The descriptor bindings the module declares for Metal buffer `b`.
fn bindings_of(a: &FunctionArtifact, b: u32) -> Vec<vk::DescriptorType> {
    a.interface
        .descriptors
        .iter()
        .filter(|d| d.slot == buffer(b))
        .map(|d| d.ty)
        .collect()
}

/// Committed profile `stem` with its description changed by `change`.
fn prof_with(stem: &str, change: impl FnOnce(&mut DeviceDescription)) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    change(&mut desc);
    Device::new(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// The committed profiles, and each without buffer robustness (the in-shader containment check).
fn profiles_both_encodings() -> Vec<(String, DeviceProfile)> {
    let mut out = Vec::new();
    for stem in PROFILES {
        out.push((stem.to_owned(), prof(stem)));
        let bare = prof_with(stem, |d| {
            d.features.remove(&VkFeature::RobustBufferAccess);
            d.features.remove(&VkFeature::RobustBufferAccess2);
        });
        out.push((format!("{stem} without robustness"), bare));
    }
    out
}

/// Relocate `a` under the numbering link gives it alone and run L3.
fn validate(a: &FunctionArtifact, p: &DeviceProfile) {
    let t = link(&[a], &LinkInputs::standalone(), p)
        .unwrap_or_else(|e| panic!("link: {e}"))
        .table;
    let words = relocate(a, &t).expect("relocate");
    check_words_valid(&words, &a.requirements, p).unwrap_or_else(|e| panic!("L3: {e}"));
}

/// Every step compiles and validates on every profile, and the artifact carries the descriptors,
/// push fields, facts, and builtins the fetch reads.
#[test]
fn pulled_attributes_compile_and_state_their_inputs() {
    let Some(bc) = fixture() else { return };
    for stem in PROFILES {
        let p = prof(stem);
        let draw = p.fields().draw_parameters;
        for rate in [1u32, 3] {
            let what = format!("{stem} rate {rate}");
            let a =
                compile_with(&bc, &p, all_pulled(rate)).unwrap_or_else(|e| panic!("{what}: {e}"));
            validate(&a, &p);
            let io = &a.interface.io;
            assert_eq!(indices(&io.pulled), [0, 1, 4, 7], "{what}");
            assert!(
                io.inputs.is_empty(),
                "{what}: a pulled attribute has no Location"
            );
            for b in [0, 2, 3] {
                let d = a
                    .interface
                    .descriptors
                    .iter()
                    .find(|d| d.slot == buffer(b))
                    .unwrap_or_else(|| panic!("{what}: no descriptor for buffer {b}"));
                assert_eq!(d.ty, vk::DescriptorType::STORAGE_BUFFER, "{what}");
                // The binder binds it from the slot the artifact states, and the module declares it
                // read-only.
                assert!(
                    matches!(
                        a.interface.slots.get(&slot(b)),
                        Some(SlotInterface::Buffer(s)) if s.access == AccessSet::READ
                            && s.window == Window::Whole
                            && s.form == BufferForm::Descriptor(DescriptorView::Storage)
                    ),
                    "{what} {b}: {:?}",
                    a.interface.slots.get(&slot(b))
                );
                assert!(non_writable(&a, buffer(b)), "{what} {b}");
                assert!(placed(&a, PushField::SlotBindOffset(slot(b))), "{what} {b}");
                assert!(a
                    .facts
                    .contains(&RuntimeFact::BufferRangeLimit { slot: slot(b) }));
            }
            // The constant step (buffer 3) reads no stride.
            assert!(placed(&a, PushField::VertexBufferStride(0)), "{what}");
            assert!(placed(&a, PushField::VertexBufferStride(2)), "{what}");
            assert!(!placed(&a, PushField::VertexBufferStride(3)), "{what}");
            for attr in [0, 1, 4, 7] {
                assert!(
                    a.facts.contains(&RuntimeFact::VertexFetch { attr }),
                    "{what}"
                );
            }
            let divided = rate > 1;
            assert_eq!(
                a.facts.contains(&RuntimeFact::BaseVertexInstance),
                divided,
                "{what}"
            );
            assert_eq!(
                placed(&a, PushField::BaseVertexInstance),
                divided && !draw,
                "{what}"
            );
            assert_eq!(
                io.builtins.contains(&StageBuiltin::BaseInstance),
                divided && draw,
                "{what}"
            );
            assert!(io.builtins.contains(&StageBuiltin::VertexIndex), "{what}");
            assert!(io.builtins.contains(&StageBuiltin::InstanceIndex), "{what}");
        }
    }
}

/// A projection built from a vertex descriptor: fixed function and pulled attributes mix in one
/// module (here on `vk12-floor`, whose formats and divisor support decide per attribute).
#[test]
fn a_descriptor_projection_mixes_fixed_function_and_pulled() {
    let Some(bc) = fixture() else { return };
    for stem in PROFILES {
        let p = prof(stem);
        let mut desc = VertexDescriptor::new();
        desc.layouts
            .insert(0, layout(VertexStepFunction::PerVertex, 1));
        desc.layouts
            .insert(2, layout(VertexStepFunction::PerInstance, 5));
        desc.layouts
            .insert(3, layout(VertexStepFunction::Constant, 0));
        for (attr, format, offset, b) in [
            (0, VertexFormat::Float3, 0, 0),
            (1, VertexFormat::Half2, 12, 0),
            (4, VertexFormat::UShort2, 0, 2),
            (7, VertexFormat::UChar4, 0, 3),
        ] {
            desc.attributes
                .insert(attr, VertexAttributeDescriptor::new(format, offset, b));
        }
        let proj = vertex_fetch_projection(Some(&desc), &classes(), &p).expect("projection");
        assert!(matches!(proj.attributes[&7], VertexFetch::Pulled { .. }));
        let pulled: Vec<u32> = proj
            .attributes
            .iter()
            .filter(|(_, f)| matches!(f, VertexFetch::Pulled { .. }))
            .map(|(a, _)| *a)
            .collect();
        let a = compile_with(&bc, &p, proj).unwrap_or_else(|e| panic!("{stem}: {e}"));
        validate(&a, &p);
        assert_eq!(indices(&a.interface.io.pulled), pulled, "{stem}");
        assert_eq!(
            a.interface.io.inputs.len() + pulled.len(),
            4,
            "{stem}: every attribute is an input or pulled"
        );
    }
}

/// A float format read into an integer attribute is not a conversion whose Metal meaning is
/// established: refused, never guessed, by the one projection and again by the pulled fetch for a
/// hand-built projection. A 32-bit integer format read into a float attribute is an invalid
/// descriptor naming the attribute, as Metal refuses its pipeline, by both; a tessellation step is
/// refused too.
#[test]
fn unestablished_conversions_are_refused() {
    let p = prof("moltenvk-m3");
    use VertexStepFunction as S;
    let refused = Error::from(crate::error::Unsupported::not_implemented(
        Feature::Lowering(LowerFamily::StageIo),
    ));
    let invalid = |attribute| {
        Error::InvalidDescriptor(crate::error::DescriptorError::VertexAttribute { attribute })
    };
    for (attr, format, want) in [
        (4, VertexFormat::Half2, refused.clone()),
        (0, VertexFormat::UInt3, invalid(0)),
        (1, VertexFormat::Int2, invalid(1)),
    ] {
        let mut desc = VertexDescriptor::new();
        desc.layouts.insert(0, layout(S::PerVertex, 1));
        desc.attributes
            .insert(attr, VertexAttributeDescriptor::new(format, 0, 0));
        let classes = BTreeMap::from([(attr, classes()[&attr])]);
        assert_eq!(
            vertex_fetch_projection(Some(&desc), &classes, &p),
            Err(want),
            "{format:?}"
        );
    }
    let Some(bc) = fixture() else { return };
    for (attr, fetch, want) in [
        (
            4,
            pulled(VertexFormat::Half2, 0, 0, S::PerVertex, 1),
            refused.clone(),
        ),
        (
            0,
            pulled(VertexFormat::UInt3, 0, 0, S::PerVertex, 1),
            invalid(0),
        ),
        (
            0,
            pulled(VertexFormat::Float3, 0, 0, S::PerPatch, 1),
            crate::error::Unsupported::not_implemented(Feature::Tessellation).into(),
        ),
    ] {
        let mut proj = VertexFetchProjection::new();
        proj.attributes.insert(attr, fetch);
        // The compile locates a refusal at its parameter; the error is the projection's.
        let got = match compile_with(&bc, &p, proj) {
            Err(Error::Unsupported(u)) => Err(Error::Unsupported(crate::error::Unsupported {
                location: None,
                ..u
            })),
            other => other.map(|_| ()),
        };
        assert_eq!(got, Err(want), "{fetch:?}");
    }
}

/// An unnormalized 8- or 16-bit integer format read into the fixture's `float3` and `half2`
/// attributes is fixed function in its scaled form where the profile fetches that, and pulled
/// (converted in the shader) where the profile has no buffer robustness; both compile and pass
/// L3 on every profile.
#[test]
fn integer_formats_feed_float_attributes() {
    let Some(bc) = fixture() else { return };
    use VertexStepFunction as S;
    let mut desc = VertexDescriptor::new();
    desc.layouts.insert(0, layout(S::PerVertex, 1));
    for (attr, format, offset) in [
        (0, VertexFormat::UShort3, 0),
        (1, VertexFormat::Char2, 8),
        (4, VertexFormat::UShort2, 12),
        (7, VertexFormat::UChar4, 16),
    ] {
        desc.attributes
            .insert(attr, VertexAttributeDescriptor::new(format, offset, 0));
    }
    for (stem, p) in profiles_both_encodings() {
        let proj = vertex_fetch_projection(Some(&desc), &classes(), &p)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let robust = !stem.ends_with("without robustness");
        for (attr, scaled) in [
            (0, vk::Format::R16G16B16_USCALED),
            (1, vk::Format::R8G8_SSCALED),
        ] {
            let fixed = robust
                && p.supports_format(
                    scaled,
                    crate::device::requirements::FormatUsage::VertexBuffer,
                );
            let entry = VertexFetch::FixedFunction {
                narrow_float: false,
            };
            assert_eq!(
                proj.attributes[&attr] == entry,
                fixed,
                "{stem} {attr}: {:?}",
                proj.attributes[&attr]
            );
        }
        let a = compile_with(&bc, &p, proj).unwrap_or_else(|e| panic!("{stem}: {e}"));
        validate(&a, &p);
        // Every attribute pulled converts in the shader.
        let mut pulled_all = VertexFetchProjection::new();
        for (attr, d) in &desc.attributes {
            pulled_all.attributes.insert(
                *attr,
                VertexFetch::pulled(d, &desc.layouts[&d.buffer_index]),
            );
        }
        let a = compile_with(&bc, &p, pulled_all).unwrap_or_else(|e| panic!("{stem}: {e}"));
        validate(&a, &p);
        assert_eq!(indices(&a.interface.io.pulled), [0, 1, 4, 7], "{stem}");
    }
}

/// The plan's size estimate bounds the emitted words with every attribute pulled, through views, a
/// shared slot, and (where the profile has them) device addresses, and with normalized formats'
/// exact quotients.
#[test]
fn the_estimate_bounds_pulled_fetches() {
    let Some(bc) = fixture() else { return };
    for stem in PROFILES {
        let p = prof(stem);
        let mut variant = inputs(both_pulled(3));
        variant.psb_slots = [0, 1, 2, 3].into_iter().map(slot).collect();
        let mut cases = vec![
            inputs(all_pulled(3)),
            inputs(shared_pulled(3)),
            inputs(normalized_pulled(3)),
        ];
        if p.fields().buffer_access == BufferAccess::Addressed {
            cases.push(variant);
        }
        for i in cases {
            let plan = plan_bitcode(
                &bc,
                EntryRef::Name(ENTRY),
                &FunctionConstantValues::defaults(),
                &p,
                &i,
                &mut meter(),
            )
            .expect("plan");
            let a = compile_inputs(&bc, &p, &i).expect("compile");
            let words = a.module.words.len() as u64;
            assert!(
                plan.estimate.words >= words,
                "{stem} {:?}: estimate {} < emitted {words}",
                i.psb_slots,
                plan.estimate.words
            );
        }
    }
}

/// A source buffer whose Metal index is also the function's `scale` argument reads through the
/// argument's slot. The slot is bound once, as a `Whole` storage buffer (alone, `scale` is a
/// `Bounded` window, a uniform view where the profile's containment allows one), read-only and
/// `NonWritable`, with the range fact and the bind residual the fetch reads.
#[test]
fn a_source_buffer_shares_its_arguments_slot() {
    let Some(bc) = fixture() else { return };
    for (stem, p) in profiles_both_encodings() {
        let alone = compile_with(&bc, &p, all_pulled(1)).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let scale = buffer_slot(&alone, 1).expect("scale's slot");
        assert!(
            matches!(scale.window, Window::Bounded { start: 0, .. }),
            "{stem}: {scale:?}"
        );
        for rate in [1, 3] {
            let what = format!("{stem} rate {rate}");
            let a = compile_with(&bc, &p, shared_pulled(rate))
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            validate(&a, &p);
            let s = buffer_slot(&a, 1);
            assert!(
                s.is_some_and(|s| s.access == AccessSet::READ
                    && s.window == Window::Whole
                    && s.form == BufferForm::Descriptor(DescriptorView::Storage)),
                "{what}: {s:?}"
            );
            assert_eq!(buffer_slot(&a, 0), None, "{what}: buffer 0 is not read");
            assert_eq!(
                bindings_of(&a, 1),
                [vk::DescriptorType::STORAGE_BUFFER],
                "{what}: one binding for the argument and the fetch"
            );
            assert!(non_writable(&a, buffer(1)), "{what}");
            assert!(placed(&a, PushField::SlotBindOffset(slot(1))), "{what}");
            assert!(placed(&a, PushField::VertexBufferStride(1)), "{what}");
            assert!(a
                .facts
                .contains(&RuntimeFact::BufferRangeLimit { slot: slot(1) }));
            assert_eq!(indices(&a.interface.io.pulled), [0, 1, 4, 7], "{what}");
        }
    }
}

/// Under a `BufferRangeLimit` variant a source buffer of its own and a shared one are reached by
/// device address like any storage slot: `Psb` slots (read-only for the source of its own),
/// `DeviceAddresses`, the root table instead of the bind residual, no descriptor, and the range
/// fact still declared (the variant is a fixed point). A profile without device addresses refuses
/// the variant.
#[test]
fn source_buffers_take_device_addresses_in_the_range_variant() {
    let Some(bc) = fixture() else { return };
    for stem in PROFILES {
        let p = prof(stem);
        let mut i = inputs(both_pulled(3));
        i.psb_slots = [slot(0), slot(1)].into_iter().collect();
        let r = compile_inputs(&bc, &p, &i);
        if p.fields().buffer_access != BufferAccess::Addressed {
            match r {
                Err(Error::Unsupported(u)) => assert_eq!(u.feature, Feature::BufferRange, "{stem}"),
                other => panic!("{stem}: {other:?}"),
            }
            continue;
        }
        let a = r.unwrap_or_else(|e| panic!("{stem}: {e}"));
        validate(&a, &p);
        for (b, access) in [(0, AccessSet::READ), (1, AccessSet::READ)] {
            let s = buffer_slot(&a, b);
            assert!(
                s.is_some_and(|s| s.access == access
                    && s.window == Window::Whole
                    && s.form == BufferForm::Psb),
                "{stem} {b}: {s:?}"
            );
            assert!(bindings_of(&a, b).is_empty(), "{stem} {b}");
            assert!(
                !placed(&a, PushField::SlotBindOffset(slot(b))),
                "{stem} {b}"
            );
            assert!(a
                .facts
                .contains(&RuntimeFact::DeviceAddresses { slot: slot(b) }));
            assert!(a
                .facts
                .contains(&RuntimeFact::BufferRangeLimit { slot: slot(b) }));
        }
        assert!(placed(&a, PushField::RootTableAddress), "{stem}");
        assert!(a.interface.needs_pointer_translation, "{stem}");
        for b in [2, 3] {
            assert_eq!(bindings_of(&a, b), [vk::DescriptorType::STORAGE_BUFFER]);
            assert!(placed(&a, PushField::SlotBindOffset(slot(b))), "{stem} {b}");
        }
    }
}

/// On Profile A the per-stage storage-buffer fallback moves the highest-index storage slots to
/// device addresses whether they are source buffers or the function's own; Profile F refuses naming
/// the descriptor count.
#[test]
fn the_descriptor_limit_fallback_moves_source_buffers() {
    let Some(bc) = fixture() else { return };
    for stem in PROFILES {
        let p = prof_with(stem, |d| {
            d.limits.max_per_stage_descriptor_storage_buffers = 2
        });
        let r = compile_with(&bc, &p, both_pulled(1));
        if p.fields().buffer_access != BufferAccess::Addressed {
            match r {
                Err(Error::Unsupported(u)) => {
                    assert_eq!(u.feature, Feature::DescriptorCount, "{stem}")
                }
                other => panic!("{stem}: {other:?}"),
            }
            continue;
        }
        let a = r.unwrap_or_else(|e| panic!("{stem}: {e}"));
        validate(&a, &p);
        let form = |b| buffer_slot(&a, b).map(|s| s.form);
        // Buffers 0 to 3 are storage slots (1 shared with `scale`); the two highest move.
        assert_eq!(form(3), Some(BufferForm::Psb), "{stem}");
        assert_eq!(form(2), Some(BufferForm::Psb), "{stem}");
        for b in [0, 1] {
            assert_eq!(
                form(b),
                Some(BufferForm::Descriptor(DescriptorView::Storage)),
                "{stem} {b}"
            );
        }
        let storage = a
            .interface
            .descriptors
            .iter()
            .filter(|d| d.ty == vk::DescriptorType::STORAGE_BUFFER)
            .count();
        assert!(storage <= 2, "{stem}: {storage} storage buffers");
    }
}

// ---- Shared slots the body writes, and far ones --------------------------------

/// A vertex function `probe(float4 position [[attribute(0)]], <arg> buf [[buffer(1)]])` whose
/// body is `body` (`%0` the attribute, `%1` the buffer, returning the position `%out`), with the
/// buffer's pointer attributes `ptr_attrs` and AIR access `access` (`air.read` or
/// `air.read_write`), and the module-level declarations `decls`.
fn probe_with(ptr_attrs: &str, access: &str, body: &str, decls: &str) -> String {
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <{{ <4 x float> }}> @probe(<4 x float> %0, ptr addrspace(1) noundef {ptr_attrs} "air-buffer-no-alias" %1) local_unnamed_addr #0 {{
{body}
  %r = insertvalue <{{ <4 x float> }}> undef, <4 x float> %out, 0
  ret <{{ <4 x float> }}> %r
}}

{decls}

attributes #0 = {{ nounwind }}

!air.vertex = !{{!9}}
!llvm.module.flags = !{{!3, !4, !5, !6, !7, !8}}
!air.version = !{{!73}}
!air.language_version = !{{!74}}
!3 = !{{i32 7, !"air.max_device_buffers", i32 31}}
!4 = !{{i32 7, !"air.max_constant_buffers", i32 31}}
!5 = !{{i32 7, !"air.max_threadgroup_buffers", i32 31}}
!6 = !{{i32 7, !"air.max_textures", i32 128}}
!7 = !{{i32 7, !"air.max_read_write_textures", i32 8}}
!8 = !{{i32 7, !"air.max_samplers", i32 16}}
!9 = !{{ptr @probe, !10, !16}}
!10 = !{{!11}}
!11 = !{{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}}
!16 = !{{!17, !21}}
!17 = !{{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}}
!21 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"{access}", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"buf"}}
!73 = !{{i32 2, i32 8, i32 0}}
!74 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// [`probe_with`] without declarations.
fn probe(ptr_attrs: &str, access: &str, body: &str) -> String {
    probe_with(ptr_attrs, access, body, "")
}

/// `text` assembled by `llvm-as` from a scratch file removed at once.
fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new(&format!("m2v-vertex-fetch-{name}"));
    let dir = scratch.path();
    let ll = dir.join("probe.ll");
    std::fs::write(&ll, text).expect("write");
    let r = llvm_assemble(&ll);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}\n{text}"),
    }
}

/// The projection pulling `probe`'s attribute 0 per vertex, a `float4` at offset 0, from buffer
/// `buffer`.
fn probe_projection(buffer: u32) -> VertexFetchProjection {
    let mut proj = VertexFetchProjection::new();
    proj.attributes.insert(
        0,
        pulled(
            VertexFormat::Float4,
            0,
            buffer,
            VertexStepFunction::PerVertex,
            1,
        ),
    );
    proj
}

/// `probe`'s lowering inputs on `p`: none, attribute 0 pulled per vertex from buffer 1, and that
/// pull in a `BufferRangeLimit` variant of buffer 1 (where `p` has device addresses).
fn probe_inputs(p: &DeviceProfile) -> Vec<(&'static str, LoweringInputs)> {
    let shared = inputs(probe_projection(1));
    let mut variant = shared.clone();
    variant.psb_slots = [slot(1)].into_iter().collect();
    let mut out = vec![("alone", LoweringInputs::new()), ("shared", shared)];
    if p.fields().buffer_access == BufferAccess::Addressed {
        out.push(("shared variant", variant));
    }
    out
}

/// `probe` compiled on `p` under each of [`probe_inputs`].
fn probe_compiles(
    bc: &[u8],
    p: &DeviceProfile,
) -> Vec<(&'static str, Result<Arc<FunctionArtifact>, Error>)> {
    probe_inputs(p)
        .into_iter()
        .map(|(how, i)| (how, compile_inputs_of(bc, "probe", p, &i)))
        .collect()
}

/// `buf[0] = 7.0` (an argument store to the attribute's first component when the source buffer
/// is `buf`'s memory), then `buf[8] = position.x`.
const OVERWRITE: &str = "  store float 7.0, ptr addrspace(1) %1, align 4
  %x = extractelement <4 x float> %0, i64 0
  %p = getelementptr inbounds float, ptr addrspace(1) %1, i64 8
  store float %x, ptr addrspace(1) %p, align 4
  %out = fadd <4 x float> %0, zeroinitializer";

/// An atomic `add` of 3 to `buf`'s word 0, then `buf[8] = position.x`.
const ATOMIC_ADD: &str = "  %o = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %1, i32 3, i32 0, i32 2, i1 true)
  %x = extractelement <4 x float> %0, i64 0
  %p = getelementptr inbounds float, ptr addrspace(1) %1, i64 8
  store float %x, ptr addrspace(1) %p, align 4
  %out = fadd <4 x float> %0, zeroinitializer";

/// `buf[0] = 7.0`, then a branch on `position.x < 5` whose taken arm stores `position.y` at float
/// 8 and whose other arm stores 9.0 there, and the position used again after the join: the
/// attribute is read in three blocks.
const BRANCHED: &str = "  store float 7.0, ptr addrspace(1) %1, align 4
  %x = extractelement <4 x float> %0, i64 0
  %c = fcmp olt float %x, 5.0
  %p = getelementptr inbounds float, ptr addrspace(1) %1, i64 8
  br i1 %c, label %small, label %large
small:
  %y = extractelement <4 x float> %0, i64 1
  store float %y, ptr addrspace(1) %p, align 4
  br label %done
large:
  store float 9.0, ptr addrspace(1) %p, align 4
  br label %done
done:
  %out = fadd <4 x float> %0, zeroinitializer";

const ATOMIC_DECL: &str =
    "declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, i32, i1)";

/// A slot the function writes, or reaches by an atomic, shares the fetch's source buffer: the fetch
/// runs once before the body, so the function's accesses never reach the attribute. The shared slot
/// is one `Whole`, read-write storage binding (not `NonWritable`), or a `Psb` slot in the
/// `BufferRangeLimit` variant, and passes L3, on every profile with
/// `vertexPipelineStoresAndAtomics` (the others refuse the store by name, as alone); the size
/// estimate bounds the emitted words.
#[test]
fn a_written_or_atomic_argument_shares_its_slot() {
    let (Some(write), Some(atomic)) = (
        assemble_text("write", &probe("", "air.read_write", OVERWRITE)),
        assemble_text(
            "atomic",
            &probe_with("", "air.read_write", ATOMIC_ADD, ATOMIC_DECL),
        ),
    ) else {
        return;
    };
    for stem in PROFILES {
        let p = prof(stem);
        for (name, bc) in [("write", &write), ("atomic", &atomic)] {
            for (how, i) in probe_inputs(&p) {
                let what = format!("{stem} {name} {how}");
                let r = compile_inputs_of(bc, "probe", &p, &i);
                if !p.fields().leaf.vertex_stores_and_atomics {
                    match r {
                        Err(Error::Unsupported(u)) => {
                            assert_eq!(u.feature, Feature::StageStoresAndAtomics, "{what}")
                        }
                        other => panic!("{what}: {:?}", other.map(|a| a.interface.slots.clone())),
                    }
                    continue;
                }
                let a = r.unwrap_or_else(|e| panic!("{what}: {e}"));
                validate(&a, &p);
                let plan = plan_bitcode(
                    bc,
                    EntryRef::Name("probe"),
                    &FunctionConstantValues::defaults(),
                    &p,
                    &i,
                    &mut meter(),
                )
                .unwrap_or_else(|e| panic!("{what}: {e}"));
                let words = a.module.words.len() as u64;
                assert!(
                    plan.estimate.words >= words,
                    "{what}: estimate {} < emitted {words}",
                    plan.estimate.words
                );
                let s = buffer_slot(&a, 1).expect("buf's slot");
                assert!(s.access.write, "{what}: {s:?}");
                if how == "alone" {
                    assert!(
                        matches!(s.window, Window::Bounded { start: 0, .. }),
                        "{what}: {s:?}"
                    );
                    continue;
                }
                // The function's access plus the fetch's read.
                assert_eq!(s.access, AccessSet::READ_WRITE, "{what}");
                assert_eq!(s.window, Window::Whole, "{what}");
                if how == "shared" {
                    assert_eq!(
                        s.form,
                        BufferForm::Descriptor(DescriptorView::Storage),
                        "{what}"
                    );
                    assert_eq!(
                        bindings_of(&a, 1),
                        [vk::DescriptorType::STORAGE_BUFFER],
                        "{what}: one binding for the argument and the fetch"
                    );
                    assert!(!non_writable(&a, buffer(1)), "{what}");
                } else {
                    assert_eq!(s.form, BufferForm::Psb, "{what}");
                }
                assert!(placed(&a, PushField::VertexBufferStride(1)), "{what}");
            }
        }
    }
}

/// A `Whole` window reaches only the first 2³² bytes above the bind offset, so an argument read at
/// a constant offset past them cannot share it: refused `Unsupported{Lowering(StageIo),
/// NotImplemented}`, never compiled to a changed value, while the function compiles alone; a read
/// ending exactly at 2³² bytes shares the slot, read-only.
#[test]
fn a_far_argument_is_not_shared() {
    let read_at = |offset: u64| {
        probe(
            "readonly",
            "air.read",
            &format!(
                "  %p = getelementptr inbounds i8, ptr addrspace(1) %1, i64 {offset}\n  %v = load float, ptr addrspace(1) %p, align 4\n  %out = insertelement <4 x float> %0, float %v, i64 3"
            ),
        )
    };
    let far = read_at((1 << 32) + 16);
    let near = read_at((1 << 32) - 4);
    let (Some(far), Some(near)) = (assemble_text("far", &far), assemble_text("near", &near)) else {
        return;
    };
    for stem in PROFILES {
        let p = prof(stem);
        for (how, r) in probe_compiles(&far, &p) {
            let what = format!("{stem} far {how}");
            match (how, r) {
                ("alone", Ok(a)) => validate(&a, &p),
                (_, Err(Error::Unsupported(u))) if how != "alone" => {
                    assert_eq!(u.feature, Feature::Lowering(LowerFamily::StageIo), "{what}");
                    assert_eq!(u.needs, Needs::NotImplemented, "{what}");
                }
                (_, other) => panic!("{what}: {:?}", other.map(|a| a.interface.slots.clone())),
            }
        }
        for (how, r) in probe_compiles(&near, &p) {
            let what = format!("{stem} near {how}");
            let a = r.unwrap_or_else(|e| panic!("{what}: {e}"));
            validate(&a, &p);
            if how != "alone" {
                let s = buffer_slot(&a, 1);
                assert!(
                    s.is_some_and(|s| s.access == AccessSet::READ && s.window == Window::Whole),
                    "{what}: {s:?}"
                );
            }
        }
    }
}

// ---- The fetch precedes the body, on the CPU interpreter -------------------------------

/// Device addresses of the interpreter's address-backed buffers: the root table, the sink, and
/// the allocation the source buffer and `buf` are bound to.
const TABLE: u64 = 0x0000_0010_0000_0000;
const SINK: u64 = 0x0000_0011_0000_0000;
const ALLOCATION: u64 = 0x0000_0020_0000_0000;
/// Bytes of the allocation (a multiple of 16).
const ALLOC: usize = 64;
/// The vertex stride the draw binds.
const STRIDE: u32 = 16;

/// The allocation's bytes: the floats `1, 2, 3, ...`.
fn allocation() -> Vec<u8> {
    (0..ALLOC / 4)
        .flat_map(|i| (i as f32 + 1.0).to_bits().to_le_bytes())
        .collect()
}

/// A root table whose entries `2 + n` for each `n` in `slots` bind the allocation at offset 0
/// (entry 1 the sink, with no resident entries).
fn root_table(slots: &[u32]) -> Vec<u8> {
    let n = 2 + slots.iter().copied().max().unwrap_or(0) as usize + 1;
    let mut t = vec![0u8; 32 * n];
    let mut put = |root: usize, field: usize, v: u64| {
        let at = 32 * root + 8 * field;
        t[at..at + 8].copy_from_slice(&v.to_le_bytes());
    };
    put(1, 0, SINK);
    put(1, 1, n as u64);
    for &s in slots {
        let root = 2 + s as usize;
        put(root, 0, ALLOCATION);
        put(root, 1, 0x1000 * u64::from(s + 1));
        put(root, 2, ALLOC as u64);
    }
    t
}

/// The push bytes of `plan`: zero bind residuals, the stride, and the root table's address.
fn push_bytes(plan: &Plan) -> Vec<u8> {
    let mut out = vec![0u8; 128];
    for f in &plan.push.fields {
        let PushLoc::Push { offset } = f.loc else {
            panic!("{:?} spilled to the ArgumentTable", f.field);
        };
        let at = offset as usize;
        match f.field {
            PushField::SlotBindOffset(_) => {}
            PushField::VertexBufferStride(_) => {
                out[at..at + 4].copy_from_slice(&STRIDE.to_le_bytes())
            }
            PushField::RootTableAddress => out[at..at + 8].copy_from_slice(&TABLE.to_le_bytes()),
            other => panic!("unexpected push field {other:?}"),
        }
    }
    out
}

/// The attribute is fetched before the body runs, so a write of the body to the source buffer's
/// memory before the attribute's uses never reaches it; the function reads the pre-write value, as
/// Metal's input does. `probe` stores 7.0 (or atomically adds 3 to the word) where vertex 0's `x`
/// lies, then stores `x` at float 8, or branches on `x` and stores `y` there: through the shared
/// descriptor slot, through the shared slot's device address, and through another Metal index (the
/// source buffer 2 of its own, `buf` at index 1) bound to the same allocation by device address.
/// Vertex 0 of one instance on the CPU interpreter, on every profile with
/// `vertexPipelineStoresAndAtomics` (device addresses where it has them).
#[test]
fn the_fetch_reads_the_attribute_before_the_body_writes() {
    let (Some(write), Some(atomic), Some(branched)) = (
        assemble_text("pre-write", &probe("", "air.read_write", OVERWRITE)),
        assemble_text(
            "pre-atomic",
            &probe_with("", "air.read_write", ATOMIC_ADD, ATOMIC_DECL),
        ),
        assemble_text("pre-branch", &probe("", "air.read_write", BRANCHED)),
    ) else {
        return;
    };
    let one = 1.0f32.to_bits();
    let mut ran = 0;
    for stem in PROFILES {
        let p = prof(stem);
        if !p.fields().leaf.vertex_stores_and_atomics {
            continue;
        }
        let addressed = p.fields().buffer_access == BufferAccess::Addressed;
        // (probe, its module, word 0 after the body, word 8: the pre-write attribute's x or y)
        for (name, bc, first, eighth) in [
            ("write", &write, 7.0f32.to_bits(), one),
            ("atomic", &atomic, one + 3, one),
            ("branched", &branched, 7.0f32.to_bits(), 2.0f32.to_bits()),
        ] {
            // (case, source buffer, device-address slots)
            let mut cases: Vec<(&str, u32, Vec<u32>)> = vec![("shared view", 1, vec![])];
            if addressed {
                cases.push(("shared address", 1, vec![1]));
                cases.push(("aliased index", 2, vec![1, 2]));
            }
            for (case, source, psb) in cases {
                let what = format!("{stem} {name} {case}");
                let mut i = inputs(probe_projection(source));
                i.psb_slots = psb.iter().map(|&b| slot(b)).collect();
                let a = compile_inputs_of(bc, "probe", &p, &i)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                validate(&a, &p);
                let (plan, m) = typed_module_with(bc, "probe", &p, &i)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let mut buffers = vec![
                    addressed_buffer(TABLE, root_table(&psb)),
                    addressed_buffer(SINK, vec![0; 64]),
                ];
                // The allocation: a descriptor binding of index 1, or reached by address.
                if psb.is_empty() {
                    buffers.push(Buffer {
                        slot: Some(buffer(1)),
                        address: None,
                        bytes: allocation(),
                    });
                } else {
                    buffers.push(addressed_buffer(ALLOCATION, allocation()));
                }
                let res = Resources {
                    buffers,
                    push: push_bytes(&plan),
                    spec: Default::default(),
                };
                let out =
                    run(&m, &Config::default(), res).unwrap_or_else(|e| panic!("{what}: {e}"));
                let words: Vec<u32> = out.buffers[2]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                assert_eq!(words[0], first, "{what}: the body's access");
                assert_eq!(
                    words[8], eighth,
                    "{what}: the attribute is the one Metal fetched before the body"
                );
                ran += 1;
            }
        }
    }
    assert!(ran > 0, "no profile has vertexPipelineStoresAndAtomics");
}

fn addressed_buffer(address: u64, bytes: Vec<u8>) -> Buffer {
    Buffer {
        slot: None,
        address: Some(address),
        bytes,
    }
}

/// `bc`'s entry `entry` compiled on `p` under `inputs`.
fn compile_inputs_of(
    bc: &[u8],
    entry: &str,
    p: &DeviceProfile,
    inputs: &LoweringInputs,
) -> Result<Arc<FunctionArtifact>, Error> {
    compile_bitcode(
        bc,
        EntryRef::Name(entry),
        &FunctionConstantValues::defaults(),
        p,
        inputs,
        &mut meter(),
        SPIRV_WORDS,
    )
}

/// The owned `dsx_int_attributes` label pipeline, a text label's shape (eight attributes, a
/// `UShort2` among them read as `float2`; nine varyings of mixed interpolation beside the
/// position; a `BGRA8Unorm_sRGB` target blended source-over beside `Depth32Float_Stencil8`),
/// makes on every committed profile with the depth and stencil format, and passes L3 (a profile
/// without the format refuses it, typed). Its `UShort2` attribute is fixed function in
/// `R16G16_USCALED` on the shipped profiles, and wherever else fixed-function input takes it;
/// elsewhere it is pulled.
#[test]
fn a_label_pipeline_reads_its_integer_attribute_as_float() {
    use crate::budget::Budget;
    use crate::library::Library;
    use crate::metal::{BlendFactor, PixelFormat};
    use crate::pipeline::{
        ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
    };
    let bc = match llvm_assemble(&engine_dir().join("dsx_int_attributes.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the fixture: {why}"),
    };
    let library = Library::from_air(&bc).expect("library");
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(function("label_vertex"));
    d.fragment = Some(function("label_fragment"));
    let mut vd = VertexDescriptor::new();
    vd.layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(48)));
    use VertexFormat as V;
    for (a, format, offset) in [
        (0, V::Float2, 0),
        (1, V::UShort2Normalized, 8),
        (2, V::UShort2, 12),
        (3, V::UChar4Normalized, 16),
        (4, V::UChar4Normalized, 20),
        (5, V::UShort2Normalized, 24),
        (6, V::Float2, 32),
        (7, V::Float2, 40),
    ] {
        vd.attributes
            .insert(a, VertexAttributeDescriptor::new(format, offset, 0));
    }
    d.vertex_descriptor = Some(vd);
    let mut c = ColorAttachmentDescriptor::new(PixelFormat::BGRA8UnormSRGB);
    c.blending_enabled = true;
    c.source_rgb_blend_factor = BlendFactor::SourceAlpha;
    c.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    c.source_alpha_blend_factor = BlendFactor::One;
    c.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    d.color_attachments[0] = Some(c);
    d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
    d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
    let shipped = [
        "moltenvk-m4",
        "nvidia-rtx50-linux",
        "nvidia-rtx5080-laptop",
        "amd-780m-radv",
        "intel-irisxe-rplp-anv",
        "lavapipe",
    ];
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles");
    let mut stems: Vec<String> = std::fs::read_dir(&dir)
        .expect("profiles")
        .filter_map(|e| {
            let path = e.ok()?.path();
            (path.extension()? == "json").then(|| path.file_stem()?.to_str().map(str::to_owned))?
        })
        .collect();
    stems.sort();
    let mut fixed = Vec::new();
    for stem in &stems {
        let text = std::fs::read_to_string(dir.join(format!("{stem}.json"))).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        let dev = Device::new(desc, DevicePolicy::default()).expect("device");
        let p = match dev.compile_render_pipeline(&d, &Budget::standard()) {
            Ok(p) => p,
            // A device without the depth and stencil format (roadmap-2022).
            Err(Error::Unsupported(u)) if u.feature == Feature::Format => {
                assert!(!shipped.contains(&stem.as_str()), "{stem}: {u}");
                eprintln!("{stem}: {u}");
                continue;
            }
            Err(e) => panic!("{stem}: {e}"),
        };
        for s in &p.stages {
            check_words_valid(&s.spirv, &p.requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{stem} L3: {e}"));
        }
        match p.vertex_input.attributes.iter().find(|a| a.offset == 12) {
            Some(a) => {
                assert_eq!(a.format, vk::Format::R16G16_USCALED, "{stem}");
                fixed.push(stem.as_str());
            }
            None => assert!(
                p.vertex_input.pulled.iter().any(|a| a.attribute == 2),
                "{stem}: {:?}",
                p.vertex_input
            ),
        }
    }
    for stem in shipped {
        assert!(fixed.contains(&stem), "{stem} pulls it: {fixed:?}");
    }
}

/// A fixed-function vertex attribute of a narrow-float format read as `float` answers Metal's
/// canonical NaN, and one of any other format keeps the bits the device's input delivered.
///
/// The owned `dsx_int_attributes` fixture's `int_attribute_to_float4` writes the one `float4`
/// attribute it reads to `out[vid]`, so the module's answer is read bit for bit. The interpreter
/// stands in for Vulkan's fixed-function vertex input: it holds the attribute's `Input` variable
/// at the words the device's own widening of a narrow-float field would deliver
/// (`VaryingKey::attribute(0)`), and the test reads what the shader stored.
///
/// Metal's answer, from `vertex-format-float4`'s recording of plain Metal on an M3 Max under
/// macOS 27.0.1: a `half` code and a `FloatRG11B10` field whose 5-bit exponent is all ones both
/// reach a `float` attribute as `0x7FC00000`, payload and sign dropped. A device's widening
/// carries the field's bits up into the payload instead (`0x7FC02000` for the `half` codes
/// `0x7E01` and `0xFE01`, `0x7FFE0000` for an all-ones `FloatRG11B10` field, measured on RADV).
/// Every finite, zero, subnormal and infinite word passes through, and a `Float4` attribute keeps
/// every word, NaNs included, as Metal's fetch of a 32-bit float does.
#[test]
fn a_narrow_float_attribute_fetched_in_fixed_function_answers_metals_canonical_nan() {
    use crate::pipeline::sym::VaryingKey;
    use crate::spirv::interp::run_with_inputs;
    const CANONICAL: u32 = 0x7FC0_0000;
    let bc = match llvm_assemble(&engine_dir().join("dsx_int_attributes.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the fixture: {why}"),
    };
    // The words a device's fixed-function input delivers, and Metal's answer for each: three NaN
    // payloads of either sign, an infinity, a subnormal, a zero and a finite value.
    let delivered = [
        0x7FC0_2000u32,
        0xFFC0_2000,
        0x7F80_2000,
        0x7FFE_0000,
        0x7F80_0000,
        0x0000_0001,
        0x8000_0000,
        0x3F80_0000,
    ];
    let canonical: Vec<u32> = delivered
        .iter()
        .map(|w| match w & 0x7FFF_FFFF > 0x7F80_0000 {
            true => CANONICAL,
            false => *w,
        })
        .collect();
    let mut ran = 0;
    for stem in [
        "moltenvk-m3",
        "amd-780m-radv",
        "nvidia-rtx5080-laptop",
        "lavapipe",
    ] {
        let p = prof(stem);
        if !p.fields().leaf.vertex_stores_and_atomics {
            continue;
        }
        for (format, narrow) in [
            (VertexFormat::Half4, true),
            (VertexFormat::FloatRG11B10, true),
            (VertexFormat::Float4, false),
        ] {
            let mut d = VertexDescriptor::new();
            d.layouts
                .insert(0, VertexBufferLayoutDescriptor::new(Some(16)));
            d.attributes
                .insert(0, VertexAttributeDescriptor::new(format, 0, 0));
            let what = format!("{stem} {format:?}");
            let proj =
                vertex_fetch_projection(Some(&d), &BTreeMap::from([(0, IoNumeric::Float)]), &p)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
            // The projection's own entry says whether the format is a narrow float; a format the
            // device's vertex input cannot fetch is pulled and is the pulled fetch's own test.
            match proj.attributes[&0] {
                VertexFetch::FixedFunction { narrow_float } => {
                    assert_eq!(narrow_float, narrow, "{what}: {:?}", proj.attributes[&0])
                }
                VertexFetch::Pulled { .. } => continue,
            }
            let i = inputs(proj);
            let a = compile_inputs_of(&bc, "int_attribute_to_float4", &p, &i)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            validate(&a, &p);
            let (plan, m) = typed_module_with(&bc, "int_attribute_to_float4", &p, &i)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            // One vertex per set of four words the attribute delivers.
            let vertices = (delivered.len() / 4) as u32;
            let mut got = Vec::new();
            for v in 0..vertices {
                let lanes = &delivered[4 * v as usize..4 * v as usize + 4];
                let res = Resources {
                    buffers: vec![Buffer {
                        slot: Some(buffer(0)),
                        address: None,
                        bytes: vec![0; 16],
                    }],
                    push: push_bytes(&plan),
                    spec: Default::default(),
                };
                let attribute = lanes.iter().flat_map(|w| w.to_le_bytes()).collect();
                let out = run_with_inputs(
                    &m,
                    &Config::default(),
                    res,
                    BTreeMap::from([(VaryingKey::attribute(0), attribute)]),
                )
                .unwrap_or_else(|e| panic!("{what}: {e}"));
                got.extend(
                    out.buffers[0]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                );
            }
            let want: Vec<u32> = match narrow {
                true => canonical.clone(),
                false => delivered.to_vec(),
            };
            assert_eq!(got, want, "{what}");
            ran += 1;
        }
    }
    assert!(
        ran > 0,
        "no profile fetched the attribute in fixed function"
    );
}
