//! The pixel interlock under the identity domains (`PointerDomain::Identity` with
//! `HandleDomain::Identity`): it orders a raster order group on memory the function reads and
//! writes, and chooses no other form (`lower::fragment::fetch::interlock_forms`).
//!
//! On lavapipe's profile under the identity domains without rasterization-order color attachment
//! access (which the runtime does not enable), with `fragmentShaderPixelInterlock` and without it,
//! each behind `colored_vertex` (`examples/runtime_examples`):
//!
//! - `rog_buffer` (engine `fragment_rog`), whose group holds a device buffer it reads and writes,
//!   runs its body in the ordered pixel interlock where the profile has the feature, its modules
//!   L3-valid, and is refused `Unsupported{RasterOrderGroups}` naming the feature where not.
//! - `fetch_ordered` (engine `fragment_fetch`), whose group holds only the color it reads, reads it
//!   through the pushed texture (`PushField::FetchedColor`) on both: no storage attachment, no
//!   interlock, and its fragment stage notes `UnorderedFetchWithinDraw`. Its modules are word for
//!   word the same with and without the feature, so the interlock chose nothing of its form.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly. No GPU is used.

use crate::budget::Budget;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DevicePolicy, HandleDomain, PointerDomain};
use crate::error::{Capability, Error, Feature, Needs};
use crate::library::Library;
use crate::metal::PixelFormat;
use crate::pipeline::artifact::{PrecisionNote, RenderPipeline, StageModule};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::sym::PushField;
use crate::pipeline::testing::{description, engine_dir, llvm_assemble, AssembleError};
use crate::spirv::testing::check_words_valid;
use crate::vk;

fn identity() -> DevicePolicy {
    DevicePolicy {
        pointer_domain: PointerDomain::Identity,
        handle_domain: HandleDomain::Identity,
        ..DevicePolicy::RESIDENT_TABLES
    }
}

/// The library of `ll`, under the engine fixtures' directory, or `None` (printed) where `llvm-as`
/// is missing.
fn library(ll: &str) -> Option<Library> {
    let ll = engine_dir().join(ll);
    match llvm_assemble(&ll) {
        Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(e) => panic!("{}: {e:?}", ll.display()),
    }
}

/// Lavapipe's profile under the identity domains without rasterization-order color attachment
/// access, with `fragmentShaderPixelInterlock` where `interlock`.
fn device(interlock: bool) -> Device {
    let mut desc = description("lavapipe").unwrap_or_else(|e| panic!("{e}"));
    desc.features
        .remove(&VkFeature::RasterizationOrderColorAttachmentAccess);
    if !interlock {
        desc.features
            .remove(&VkFeature::FragmentShaderPixelInterlock);
    }
    let device = Device::from_queries(desc, identity()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        device.profile().fragment().pixel_interlock,
        interlock,
        "lavapipe's profile and the interlock"
    );
    device
}

/// `colored_vertex` and `fragment` of `fragments`, into one `RGBA16Float` attachment.
fn pipeline(
    device: &Device,
    examples: &Library,
    fragments: &Library,
    fragment: &str,
) -> Result<RenderPipeline, Error> {
    let function = |library: &Library, name| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("the function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(function(examples, "colored_vertex"));
    d.fragment = Some(function(fragments, fragment));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
    device.compile_render_pipeline(&d, &Budget::standard())
}

fn fragment(p: &RenderPipeline) -> &StageModule {
    (p.stages.iter())
        .find(|m| m.stage == vk::ShaderStage::FRAGMENT)
        .expect("a fragment stage")
}

/// Whether `words` interlocks: an interlock execution mode (`PixelInterlockOrderedEXT` 5366 to
/// `ShadingRateInterlockUnorderedEXT` 5371) or an `OpBeginInvocationInterlockEXT` (5364).
fn interlocks(words: &[u32]) -> bool {
    const OP_EXECUTION_MODE: u32 = 16;
    const OP_BEGIN_INTERLOCK: u32 = 5364;
    let mut at = 5;
    while at < words.len() {
        let len = (words[at] >> 16) as usize;
        let op = words[at] & 0xffff;
        let mode = words.get(at + 2).copied().unwrap_or(0);
        if op == OP_BEGIN_INTERLOCK || (op == OP_EXECUTION_MODE && (5366..=5371).contains(&mode)) {
            return true;
        }
        at += len.max(1);
    }
    false
}

#[test]
fn the_pixel_interlock_orders_a_raster_order_group_on_memory_alone_under_the_identity_domains() {
    let (Some(examples), Some(groups), Some(fetches)) = (
        library("../examples/runtime_examples.ll"),
        library("fragment_rog.ll"),
        library("fragment_fetch.ll"),
    ) else {
        return;
    };
    let (with, without) = (device(true), device(false));

    // A group on a buffer the function reads and writes: the interlock orders it.
    let ordered = pipeline(&with, &examples, &groups, "rog_buffer")
        .unwrap_or_else(|e| panic!("rog_buffer with the interlock: {e}"));
    assert!(
        interlocks(&fragment(&ordered).spirv),
        "rog_buffer runs in the interlock"
    );
    for m in &ordered.stages {
        check_words_valid(&m.spirv, &ordered.requirements, with.profile())
            .unwrap_or_else(|e| panic!("rog_buffer: {:?}: {e}", m.stage));
    }
    match pipeline(&without, &examples, &groups, "rog_buffer") {
        Err(Error::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (
                Feature::RasterOrderGroups,
                Needs::AnyOf(vec![vec![Capability::Feature(
                    VkFeature::FragmentShaderPixelInterlock
                )]])
            ),
            "rog_buffer without the interlock"
        ),
        other => panic!(
            "rog_buffer without the interlock: {:?}",
            other.map(|_| "compiled")
        ),
    }

    // A group on the color the function reads alone: the fetch's own form, which the interlock
    // does not choose.
    let [read_with, read_without] = [&with, &without].map(|device| {
        pipeline(device, &examples, &fetches, "fetch_ordered")
            .unwrap_or_else(|e| panic!("fetch_ordered: {e}"))
    });
    for (p, what) in [(&read_with, "with"), (&read_without, "without")] {
        let m = fragment(p);
        assert!(!interlocks(&m.spirv), "fetch_ordered {what}: no interlock");
        assert!(
            p.graphics.storage_attachments.is_empty(),
            "fetch_ordered {what}: no storage attachment"
        );
        assert!(
            (m.precision_notes).contains(&PrecisionNote::UnorderedFetchWithinDraw),
            "fetch_ordered {what}: {:?}",
            m.precision_notes
        );
        let mut pushed = (p.interface.push.iter()).flat_map(|placement| &placement.fields);
        assert!(
            pushed.any(|f| matches!(f.field, PushField::FetchedColor { color: 0, .. })),
            "fetch_ordered {what}: the pushed texture of color 0"
        );
    }
    let words = |p: &RenderPipeline| -> Vec<Vec<u32>> {
        p.stages.iter().map(|m| m.spirv.to_vec()).collect()
    };
    assert_eq!(
        words(&read_with),
        words(&read_without),
        "fetch_ordered's modules with and without the interlock"
    );
}
