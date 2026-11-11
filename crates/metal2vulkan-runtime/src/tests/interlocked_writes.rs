//! What a stage's words state for the push of set 0 (`pipeline/interlocked_writes.rs`): read off
//! the translator's own modules for the committed MoltenVK profile, and off synthetic words for
//! the forms it does not write.

use crate::pipeline::interlocked_writes::{facts, StageFacts};
use metal2vulkan::device::{DeviceDescription, DevicePolicy};
use metal2vulkan::library::Library;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan::vk::ShaderStage;
use metal2vulkan::Budget;
use std::path::PathBuf;

/// The fragment and vertex stages' facts of each fragment function of
/// `ipo_primitive_order` (`validation/fixtures/public/engine`, `llvm-as`) drawn after `ipo_fan`
/// into an `R32Uint` attachment, compiled for the committed MoltenVK profile, which has the pixel
/// interlock and no rasterization-order color access: each fragment stage interlocks (the raster
/// order groups, and the fetch in its storage form); only `ipo_rog_buffer`'s declares a buffer of
/// set 0 it may write, its `words`, and none reads a buffer's length; `ipo_rog_texture` writes a
/// storage image, `ipo_mix` its attachment's. No vertex stage interlocks.
#[test]
fn the_translators_fragment_stages_state_their_interlocked_writes() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation");
    let ll = root.join("fixtures/public/engine/ipo_primitive_order.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).expect("llvm-as");
    let text = std::fs::read_to_string(root.join("profiles/moltenvk-m3.json")).expect("profile");
    let description = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let device =
        metal2vulkan::device::Device::new(description, DevicePolicy::default()).expect("device");
    let library = Library::from_air(&bitcode).expect("library");
    let function = |name: &str| {
        let function = library.function(name).expect("function");
        StageSource::Function(FunctionDescriptor::new(function))
    };
    for (fragment, writes) in [
        ("ipo_rog_buffer", true),
        ("ipo_rog_texture", false),
        ("ipo_mix", false),
    ] {
        let mut descriptor = RenderPipelineDescriptor::new(function("ipo_fan"));
        descriptor.fragment = Some(function(fragment));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
        let pipeline = device
            .compile_render_pipeline(&descriptor, &Budget::standard())
            .expect("compiles");
        for stage in &pipeline.stages {
            let read = facts(&stage.spirv);
            let want = match stage.stage == ShaderStage::FRAGMENT {
                true => StageFacts {
                    interlocks: true,
                    writable_set_zero_buffer: writes,
                    reads_lengths: false,
                },
                false => StageFacts::default(),
            };
            assert_eq!(read, want, "{fragment}: {:?}", stage.stage);
        }
    }
}

/// A module of `body` after a header: each instruction its opcode, then its operands.
fn module(body: &[&[u32]]) -> Vec<u32> {
    let mut words = vec![0x0723_0203, 0x0001_0500, 0, 100, 0];
    for instruction in body {
        words.push(((instruction.len() as u32) << 16) | instruction[0]);
        words.extend(&instruction[1..]);
    }
    words
}

/// `OpExecutionMode %1 PixelInterlockOrderedEXT`.
const INTERLOCK: &[u32] = &[16, 1, 5366];
/// `OpDecorate %7 DescriptorSet 0`.
const SET_ZERO: &[u32] = &[71, 7, 34, 0];
/// `OpVariable %6 %7 StorageBuffer`.
const BUFFER: &[u32] = &[59, 6, 7, 12];

/// Each fact from its own instruction: the interlock modes from `PixelInterlockOrderedEXT` to
/// `ShadingRateInterlockUnorderedEXT` and no other mode; a `StorageBuffer` variable of set 0
/// unless it is `NonWritable`, of another set, or of another storage class; either length
/// instruction; and a word count of 0 or past the words, which ends the walk as reading one.
#[test]
fn each_fact_comes_from_its_own_instruction() {
    let both = StageFacts {
        interlocks: true,
        writable_set_zero_buffer: true,
        reads_lengths: false,
    };
    assert_eq!(facts(&module(&[INTERLOCK, SET_ZERO, BUFFER])), both);
    for mode in 5366..=5371 {
        let read = facts(&module(&[&[16, 1, mode]]));
        assert!(read.interlocks, "{mode}");
    }
    for mode in [7, 5365, 5372] {
        assert_eq!(facts(&module(&[&[16, 1, mode]])), StageFacts::default());
    }
    let non_writable: &[u32] = &[71, 7, 24];
    let unwritten = facts(&module(&[SET_ZERO, non_writable, BUFFER]));
    assert!(!unwritten.writable_set_zero_buffer);
    let set_one: &[u32] = &[71, 7, 34, 1];
    assert!(!facts(&module(&[set_one, BUFFER])).writable_set_zero_buffer);
    let uniform: &[u32] = &[59, 6, 7, 2];
    assert!(!facts(&module(&[SET_ZERO, uniform])).writable_set_zero_buffer);
    for length in [68, 4425] {
        let read = facts(&module(&[&[length, 3, 4, 7, 0]]));
        assert!(read.reads_lengths, "{length}");
    }
    let mut zero = module(&[INTERLOCK]);
    zero.push(0);
    zero.extend(module(&[SET_ZERO, BUFFER]).drain(5..));
    let read = facts(&zero);
    assert!(read.interlocks && read.reads_lengths && !read.writable_set_zero_buffer);
    let mut past = module(&[INTERLOCK]);
    past.push((9 << 16) | 59);
    assert!(facts(&past).reads_lengths);
}
