//! What a kernel's `float` write to a storage texture records of the device's half-float and
//! small-float conversion, under the key the runtime compiles it under for the texture it binds.
//!
//! The fixture `hsw_storage_writes`' `hsw_write` and `hsw_read_write` write texture slot 0, and
//! `hsw_buffer` writes a texture buffer there. On a device that writes such a texture without a
//! format, the binder names no storage choice for a half-float texture (the default formatless
//! declaration serves it), so the dispatch runs the module compiled with none. That module records
//! what the write leaves to the device under the class of the format bound at the slot
//! (`SpecKey::BoundTexelFormat`), which the runtime reads at each dispatch
//! (`StageModule::bound_format_clauses`): bound in a half-float format, the NaN
//! (`PrecisionNote::DeviceNanEncoding`) and the rounding (`PrecisionNote::DeviceColorConversion`),
//! on every profile, requiring no `VkProperty::MetalHalfStorage*`; bound in `RG11B10Float`, the
//! NaN on every profile; bound in a format that holds no NaN, no NaN note.
//!
//! `llvm-as` assembles the fixture; without it the tests skip visibly.

use crate::bind::{BindState, BoundBuffer, BoundTexture, StageTable, TextureFacts};
use crate::budget::Budget;
use crate::device::requirements::VkProperty;
use crate::device::{Device, DevicePolicy};
use crate::error::Error;
use crate::library::Library;
use crate::metal::format::bound_texel_format;
use crate::metal::{PixelFormat, TextureType, TextureUsage};
use crate::pipeline::artifact::{PrecisionNote, StageModule};
use crate::pipeline::sym::{SlotClass, SlotRef, Stage};
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::{ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor};
use crate::pipeline::{LoweringInputs, StageSource};
use crate::spirv::testing::committed_descriptions;

/// The written texture's slot.
const SLOT: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Texture,
    index: 0,
};

/// The half-float formats.
const HALVES: [PixelFormat; 3] = [
    PixelFormat::R16Float,
    PixelFormat::RG16Float,
    PixelFormat::RGBA16Float,
];

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("hsw_storage_writes.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected hsw_storage_writes: {why}"),
    }
}

/// `entry` compiled into a compute pipeline for `dev` under `inputs`, or `None` where the device
/// needs the storage slot's format as an input.
fn pipeline(
    dev: &Device,
    bc: &[u8],
    entry: &str,
    inputs: &LoweringInputs,
) -> Option<ComputePipeline> {
    let lib = Library::from_air(bc).expect("library");
    let f = lib.function(entry).expect("function");
    let compiled =
        match dev.compile_function(&FunctionDescriptor::new(f), inputs, &Budget::standard()) {
            Ok(c) => c,
            Err(Error::NeedsInput(_)) => return None,
            Err(e) => panic!("{entry}: {e}"),
        };
    Some(
        dev.compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
            &Budget::standard(),
        )
        .expect("pipeline"),
    )
}

/// A kernel's state: texture slot 0 holding a writable texture of `pf` (a texture buffer where
/// `buffer`), buffers 0 and 1.
fn state(pf: PixelFormat, buffer: bool) -> BindState {
    let kind = match buffer {
        true => TextureType::TextureBuffer,
        false => TextureType::Type2D,
    };
    let mut facts = TextureFacts::new(kind, pf);
    facts.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let mut t = StageTable::default();
    t.textures.insert(
        0,
        BoundTexture::Texture {
            facts,
            resource_id: None,
        },
    );
    for (index, gpu_address) in [(0, 0x1_0000), (1, 0x2_0000)] {
        t.buffers.insert(
            index,
            BoundBuffer::Buffer {
                length: 65536,
                offset: 0,
                gpu_address,
                device_address: None,
                attribute_stride: None,
            },
        );
    }
    let mut s = BindState::default();
    s.stages.insert(Stage::Kernel, t);
    s
}
