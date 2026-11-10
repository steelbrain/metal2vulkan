//! A framebuffer read that the function's constants make dead.
//!
//! The owned fixture `engine/dfk_dead_fetch` reads color 0 in three shapes its function constants
//! decide: behind a branch on `dfk_reads` (`dfk_branch`, which Metal's compiler makes a `select`
//! on the constant), scaled by `dfk_weight` under fast math (`dfk_scaled`, whose product a zero
//! constant makes `+0`), and through an input `dfk_reads` gates (`dfk_gated`). Each returns one
//! `half4` to color 0, here `BGRA8Unorm`, blended with one over one minus the source alpha (a
//! composite's state) or unblended.
//!
//! Specialized with the read dead (`dfk_reads` false, `dfk_weight` zero), each function reads
//! nothing of the attachment, so on every committed profile its pipeline is the one of a function
//! that declares no color input: color 0 an ordinary color attachment that the device blends
//! where the pipeline blends, no storage attachment (no pixel interlock, no blend computed in the
//! shader) and no input attachment, its stages passing L3. Specialized with the read live, each
//! reads color 0 in the form the device orders a fetch by: a written storage attachment and no
//! color attachment where it orders fetch through the pixel interlock alone, an input attachment
//! where it orders it through rasterization-order input attachments, and refused where it has
//! neither. Before the fetch state was decided from the specialized function's uses, a dead read
//! took the storage form on every interlock profile, with the blend computed in the shader.
//! No GPU is used.

use std::path::Path;

use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::Error;
use crate::library::{FunctionConstantValue, FunctionConstantValues, Library};
use crate::lower::fragment::{inputs_order, interlock_orders};
use crate::metal::{BlendFactor, BlendOperation, DataType, PixelFormat};
use crate::pipeline::artifact::RenderPipeline;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::spirv::testing::{check_words_valid, committed_profiles};

/// The fixture's fragment functions.
const ENTRIES: [&str; 3] = ["dfk_branch", "dfk_scaled", "dfk_gated"];

/// The fixture's library, assembled with `llvm-as`; `None` where the tool is missing (the test
/// then skips visibly).
fn fixture_library() -> Option<Library> {
    match llvm_assemble(&engine_dir().join("dfk_dead_fetch.ll")) {
        Ok(bc) => Some(Library::from_air(&bc).expect("library")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected dfk_dead_fetch: {why}"),
    }
}

/// The device of committed profile `stem`.
fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// Function `name` of `library` specialized with `dfk_reads` = `live` and `dfk_weight` one where
/// `live`, else zero.
fn specialized(library: &Library, name: &str, live: bool) -> StageSource {
    let weight: u16 = if live { 0x3c00 } else { 0 };
    let mut d = FunctionDescriptor::new(library.function(name).expect("function"));
    d.constants = FunctionConstantValues::defaults()
        .with_index(
            0,
            FunctionConstantValue::new(DataType::Bool, [u8::from(live)]),
        )
        .with_index(
            1,
            FunctionConstantValue::new(DataType::Half, weight.to_le_bytes()),
        );
    d.specialized = true;
    StageSource::Function(d)
}

/// The pipeline of `entry` specialized live or not, behind `dfk_vertex`, color 0 `BGRA8Unorm`,
/// blended with one over one minus the source alpha where `blended`.
fn pipeline(
    dev: &Device,
    library: &Library,
    entry: &str,
    live: bool,
    blended: bool,
) -> Result<RenderPipeline, Error> {
    let vertex = StageSource::Function(FunctionDescriptor::new(
        library.function("dfk_vertex").expect("function"),
    ));
    let mut d = RenderPipelineDescriptor::new(vertex);
    d.fragment = Some(specialized(library, entry, live));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm);
    if blended {
        color.blending_enabled = true;
        color.source_rgb_blend_factor = BlendFactor::One;
        color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        color.rgb_blend_operation = BlendOperation::Add;
        color.source_alpha_blend_factor = BlendFactor::One;
        color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        color.alpha_blend_operation = BlendOperation::Add;
    }
    d.color_attachments[0] = Some(color);
    dev.compile_render_pipeline(&d, &Budget::standard())
}

/// Each storage attachment of `pipe`: its color and whether the function writes it.
fn storage(pipe: &RenderPipeline) -> Vec<(u32, bool)> {
    (pipe.graphics.storage_attachments.iter())
        .map(|u| (u.color_index, u.writes))
        .collect()
}

/// The dead specialization's misses in `pipe` (color 0 `blended` or not): each way it differs from
/// the pipeline of a function that declares no color input.
fn dead_misses(pipe: &RenderPipeline, blended: bool) -> Vec<&'static str> {
    let mut out = Vec::new();
    if !pipe.graphics.storage_attachments.is_empty() {
        out.push("it takes the storage form");
    }
    if !pipe.graphics.input_attachments.is_empty() {
        out.push("it takes an input attachment");
    }
    match pipe.graphics.color_attachments.first().copied().flatten() {
        None => out.push("color 0 is no color attachment"),
        Some(c) if c.blend.is_some() != blended => out.push("color 0 lost the pipeline's blend"),
        Some(_) => {}
    }
    out
}

#[test]
fn a_framebuffer_read_dead_after_specialization_takes_no_fetch_form() {
    let Some(library) = fixture_library() else {
        return;
    };
    let mut misses = std::collections::BTreeMap::<&str, Vec<String>>::new();
    // Pipelines checked dead, and live on an interlock profile.
    let (mut dead, mut interlocked) = (0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let dev = device(&stem);
        for entry in ENTRIES {
            for blended in [true, false] {
                let what = format!("{stem} {entry} blended {blended}");
                let pipe = pipeline(&dev, &library, entry, false, blended)
                    .unwrap_or_else(|e| panic!("{what}, dead: {e}"));
                for kind in dead_misses(&pipe, blended) {
                    misses.entry(kind).or_default().push(what.clone());
                }
                for s in &pipe.stages {
                    check_words_valid(&s.spirv, &pipe.requirements, dev.profile())
                        .unwrap_or_else(|e| panic!("{what}, dead: L3: {e}"));
                }
                dead += 1;
                let live = pipeline(&dev, &library, entry, true, blended);
                if interlock_orders(&p) {
                    let pipe = live.unwrap_or_else(|e| panic!("{what}, live: {e}"));
                    assert_eq!(storage(&pipe), [(0, true)], "{what}, live");
                    assert!(
                        pipe.graphics.color_attachments[0].is_none(),
                        "{what}, live: {:?}",
                        pipe.graphics.color_attachments[0]
                    );
                    interlocked += 1;
                } else if inputs_order(&p) {
                    let pipe = live.unwrap_or_else(|e| panic!("{what}, live: {e}"));
                    let read: Vec<u32> = (pipe.graphics.input_attachments.iter())
                        .map(|u| u.color_index)
                        .collect();
                    assert_eq!(read, [0], "{what}, live");
                    assert_eq!(storage(&pipe), [], "{what}, live");
                } else {
                    assert!(
                        matches!(live, Err(Error::Unsupported(_))),
                        "{what}, live: {:?}",
                        live.map(|_| ())
                    );
                }
            }
        }
    }
    eprintln!("{dead} dead pipelines, {interlocked} live interlocked ones");
    for (kind, whats) in &misses {
        eprintln!("{}: {kind} (first: {})", whats.len(), whats[0]);
    }
    assert!(misses.is_empty(), "{misses:#?}");
    assert!(interlocked > 0, "no interlock profile checked");
}
