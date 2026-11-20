//! A run of sibling pipeline creations over one compiled function **relocates its module once**,
//! not once each (`pipeline::relocate::relocated`).
//!
//! # What this guards
//!
//! Relocation writes only the literals the serializer reserved, so a stage's final words are a
//! pure function of its module and the value link assigned to each reserved slot. A sweep of
//! pipeline states over one function pair -- a blend sweep, where no state reaches a reserved
//! slot -- therefore wants the same words on every creation, and measured on an M3 Max every
//! creation built them again: a fresh `Arc<[u32]>` of identical words per creation, which
//! everything keyed on a module's content then hashed and encoded afresh. The clone is one pass
//! over the words and the hashes are more, so the cost of a creation rose with the size of a
//! module nothing about the creation changed.
//!
//! The words now come from the function's relocation slot, which the first creation filled.
//!
//! # Why this is a count
//!
//! How many times a run relocates is a fact of where the words come from, not of the host: it
//! holds under any load and needs no quiet machine. It is held in **both** directions. Over the
//! count is a creation that relocates again; under it is a run that relocates nothing at all,
//! which would mean the words came from somewhere other than the module -- so the first creation
//! is required to relocate exactly once a stage, which is what keeps the siblings' zero from
//! being a vacuous zero.
//!
//! [`a_differing_link_gets_its_own_words`] is the half a count catches on the low side, and the
//! one that matters for correctness rather than speed: a memo on a value that is supposed to vary
//! must still vary. Two links that assign a module's slots different values must get different
//! words, and the slot must never answer one link with another's. It also asserts what makes the
//! fast path sound at all -- that the words the slot gives are byte-identical to relocating
//! again -- so the invariant `relocate.rs` states is checked rather than trusted.

use std::sync::Arc;

use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::library::{CompiledFunction, Library};
use crate::metal::{BlendFactor, BlendOperation, PixelFormat};
use crate::pipeline::relocate::{relocate, relocated, relocations_done};
use crate::pipeline::sym::Symbol;
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageEntries,
    StageSource,
};

/// The fixture and the vertex-and-fragment pair the sweeps below make pipelines of.
const FIXTURE: &str = "ubo_bind_offsets";
const VERTEX: &str = "ubo_offsets_vertex";
const FRAGMENT: &str = "ubo_offsets_fragment";

/// The sibling creations the run makes after the first.
const RUN: u64 = 16;

/// Blend factors no two of which make one pipeline state, and none of which reaches a relocation
/// slot: each sibling is its own descriptor over the same two modules.
const FACTORS: [BlendFactor; 8] = [
    BlendFactor::Zero,
    BlendFactor::One,
    BlendFactor::SourceColor,
    BlendFactor::OneMinusSourceColor,
    BlendFactor::SourceAlpha,
    BlendFactor::OneMinusSourceAlpha,
    BlendFactor::DestinationColor,
    BlendFactor::DestinationAlpha,
];

/// A device of the committed `moltenvk-m3` profile.
fn device() -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(&path).expect("committed profile");
    let description = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::from_queries(description, DevicePolicy::default()).expect("device")
}

/// The fixture's library, or `None` where `llvm-as` is absent.
fn library() -> Option<Library> {
    match llvm_assemble(&engine_dir().join(format!("{FIXTURE}.ll"))) {
        Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the fixture parses")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {FIXTURE}.ll: {why}"),
    }
}

/// The pair compiled once, as a consumer's memo holds it across a sweep's creations.
fn pair(device: &Device, library: &Library) -> (CompiledFunction, CompiledFunction) {
    let vertex = FunctionDescriptor::new(library.function(VERTEX).expect("the vertex function"));
    let fragment =
        FunctionDescriptor::new(library.function(FRAGMENT).expect("the fragment function"));
    let budget = Budget::standard();
    let mut base = RenderPipelineDescriptor::new(StageSource::Function(vertex.clone()));
    base.fragment = Some(StageSource::Function(fragment.clone()));
    base.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let keys = device
        .render_variant_key(
            &base,
            &crate::cache::VariantKey::new(),
            &budget,
            &StageEntries::new(),
        )
        .expect("the pair's keys");
    let compile = |f: &FunctionDescriptor, inputs| {
        device
            .compile_function(f, inputs, &budget)
            .expect("the stage compiles")
    };
    (
        compile(&vertex, keys.vertex_inputs.as_ref().expect("vertex inputs")),
        compile(
            &fragment,
            keys.fragment_inputs.as_ref().expect("fragment inputs"),
        ),
    )
}

/// `RUN` sibling render pipeline creations over one compiled pair relocate **nothing**: the
/// first creation's two relocations are the ones every sibling's stages take.
#[test]
fn a_run_of_sibling_creations_relocates_once_a_stage() {
    let Some(library) = library() else { return };
    let device = device();
    let (vertex, fragment) = pair(&device, &library);
    let budget = Budget::standard();
    let sibling = |i: u64| {
        let mut d = RenderPipelineDescriptor::new(StageSource::Compiled(vertex.clone()));
        d.fragment = Some(StageSource::Compiled(fragment.clone()));
        let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
        color.blending_enabled = true;
        color.source_rgb_blend_factor = FACTORS[(i % 8) as usize];
        color.destination_rgb_blend_factor = FACTORS[(i / 8 % 8) as usize];
        color.rgb_blend_operation = BlendOperation::Add;
        d.color_attachments[0] = Some(color);
        d
    };
    // The first creation is the one that relocates: two stages, two relocations, which is what
    // keeps the siblings' zero from being a zero that measures nothing.
    let before = relocations_done();
    let first = device
        .compile_render_pipeline(&sibling(0), &budget)
        .expect("the first creation");
    let relocated_first = relocations_done() - before;
    assert_eq!(
        relocated_first, 2,
        "the first creation of a two-stage pipeline must relocate each stage exactly once, not \
         {relocated_first} times: a first creation that relocates nothing would make the \
         siblings' zero below vacuous"
    );
    let before = relocations_done();
    let mut kept = Vec::with_capacity(RUN as usize);
    for i in 0..RUN {
        kept.push(
            device
                .compile_render_pipeline(&sibling(1 + i), &budget)
                .expect("a sibling creation"),
        );
    }
    let again = relocations_done() - before;
    assert_eq!(
        again, 0,
        "{RUN} sibling creations over one compiled pair must relocate none and relocated \
         {again}: their blend states reach no relocation slot, so every stage's words are the \
         first creation's"
    );
    // The words are not merely equal: they are the first creation's own, so nothing keyed on a
    // module's content has anything new to hash.
    for (stage, first_stage) in kept[0].stages.iter().zip(&first.stages) {
        assert!(
            Arc::ptr_eq(&stage.spirv, &first_stage.spirv),
            "a sibling's stage must carry the first creation's words themselves"
        );
    }
}

/// Two links that assign a module's relocation slots different values get **different words**,
/// and neither is ever answered with the other's: the slot keys on what link assigned.
#[test]
fn a_differing_link_gets_its_own_words() {
    let Some(library) = library() else { return };
    let device = device();
    let (vertex, _fragment) = pair(&device, &library);
    let artifact = vertex.artifact();
    assert!(
        !artifact.module.relocs.is_empty(),
        "the fixture's vertex module must have a relocation slot for this test to key on anything"
    );
    let (table, _) = crate::pipeline::testing::link_standalone(artifact, device.profile())
        .expect("the function links alone");
    // A second link that assigns one slot another value: a binding one higher, as a pipeline of
    // other stages numbers it.
    let moved = artifact.module.relocs[0].0.clone();
    let shifted = |sym: &Symbol| {
        let assigned = crate::spirv::serialize::SymbolTable::resolve(&table, sym)?;
        Some(assigned + u32::from(*sym == moved))
    };
    let plain = relocated(&vertex, &table).expect("the plain link relocates");
    let other = relocated(&vertex, &shifted).expect("the shifted link relocates");
    assert_ne!(
        &plain[..],
        &other[..],
        "a link that assigns a relocation slot another value must get other words, else the slot \
         answered one link with another's"
    );
    // Back to the plain link, whose words the slot no longer holds: it must relocate rather than
    // hand back the shifted link's, and what it gives must be byte-identical to relocating the
    // module again, which is the invariant the fast path rests on.
    let plain_again = relocated(&vertex, &table).expect("the plain link relocates again");
    assert_eq!(
        &plain_again[..],
        &plain[..],
        "the plain link's words must not change because another link was relocated between"
    );
    let fresh = relocate(artifact, &table).expect("relocating the module again");
    assert_eq!(
        &plain_again[..],
        &fresh[..],
        "the words the slot gives must be the words relocating the module again gives: only the \
         reserved literals differ between links, so a memo keyed on them is exact or it is wrong"
    );
}
