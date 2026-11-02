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
//! (`PrecisionNote::DeviceNanEncoding`) where the profile does not take
//! `DriverDeviation::MetalHalfStorageNans` (whose property the module requires where it does), and
//! the rounding (`PrecisionNote::DeviceColorConversion`) where it does not take
//! `DriverDeviation::MetalHalfStorageWrites` (likewise); bound in `RG11B10Float`, the NaN on every
//! profile; bound in a format that holds no NaN, no NaN note.
//!
//! `llvm-as` assembles the fixture; without it the tests skip visibly.

use crate::bind::{BindState, BoundBuffer, BoundTexture, StageTable, TextureFacts};
use crate::budget::Budget;
use crate::device::requirements::VkProperty;
use crate::device::{Device, DevicePolicy};
use crate::error::{DriverDeviation, Error};
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

/// Whether the runtime announces `note` of `stage` at a dispatch that binds a texture of `pf` at
/// [`SLOT`]: a note no specialization removes, or a specialized one with a clause of the bound
/// formats that holds of `pf`'s class.
fn announced(stage: &StageModule, note: PrecisionNote, pf: PixelFormat) -> bool {
    if !stage.precision_notes.contains(&note) {
        return false;
    }
    let Some(s) = stage.specialized_notes.iter().find(|s| s.note == note) else {
        return true;
    };
    assert!(
        s.when.iter().all(|c| c.tests_bound_formats()),
        "{note:?}: {:?}",
        s.when
    );
    let class = |slot: SlotRef| {
        assert_eq!(slot, SLOT);
        Some(bound_texel_format(pf))
    };
    !stage.bound_format_clauses(note, &class).is_empty()
}

/// On every committed profile that compiles the kernels without a storage format, each form's
/// dispatch with a half-float, `RG11B10Float`, `RGBA32Float` or `RGBA8Unorm` texture bound runs the
/// variant the binder plans for it, whose module records the write's notes, or requires the
/// properties, as the module docs say, under the format bound there; the module of no storage
/// choice notes the NaN of exactly the half-float and small-float formats. Without a note under the
/// bound format, the dispatch of a half-float texture on lavapipe announced no
/// `device-nan-encoding` though the device stores `0xffc00000` as `0xfe00`, where Metal stores
/// `0x7e00`.
#[test]
fn a_float_write_notes_the_device_s_half_conversion_under_the_bound_format() {
    let Some(bc) = fixture() else {
        return;
    };
    let (mut noted, mut measured) = (0, 0);
    for (stem, desc) in committed_descriptions().expect("profiles") {
        let dev = Device::new(desc.clone(), DevicePolicy::default()).expect("device");
        let p = dev.profile();
        let rounds = p.takes_driver_deviation(DriverDeviation::MetalHalfStorageWrites);
        let nans = p.takes_driver_deviation(DriverDeviation::MetalHalfStorageNans);
        let small_nans = p.takes_driver_deviation(DriverDeviation::MetalSmallFloatNans);
        for (entry, buffer) in [
            ("hsw_write", false),
            ("hsw_read_write", false),
            ("hsw_buffer", true),
        ] {
            let Some(plain) = pipeline(&dev, &bc, entry, &LoweringInputs::new()) else {
                continue;
            };
            let formats = HALVES.into_iter().chain([
                PixelFormat::RG11B10Float,
                PixelFormat::RGBA32Float,
                PixelFormat::RGBA8Unorm,
            ]);
            for pf in formats {
                let what = format!("{entry} of {pf:?} on {stem}");
                // A format the device does not write so is refused by the plan.
                let Ok(plan) = plain.plan(&dev, &state(pf, buffer).shape()) else {
                    continue;
                };
                let key = plan.variant.lowering_inputs(Stage::Kernel);
                let variant = pipeline(&dev, &bc, entry, &key).expect("the variant's inputs");
                let stage = &variant.stage;
                let relied = &variant.requirements.properties;
                let nan = announced(stage, PrecisionNote::DeviceNanEncoding, pf);
                if HALVES.contains(&pf) {
                    assert_eq!(nan, !nans, "{what}: {:?}", stage.specialized_notes);
                    assert_eq!(
                        relied.contains(&VkProperty::MetalHalfStorageNans),
                        nans,
                        "{what}"
                    );
                    assert!(
                        rounds || announced(stage, PrecisionNote::DeviceColorConversion, pf),
                        "{what}: {:?}",
                        stage.precision_notes
                    );
                    assert_eq!(
                        relied.contains(&VkProperty::MetalHalfStorageWrites),
                        rounds,
                        "{what}"
                    );
                    noted += usize::from(nan);
                    measured += usize::from(!nan);
                    continue;
                }
                // Metal stores a small float's every NaN as its canonical one, and RADV and
                // NVIDIA keep another's payload bits, where a device of `MetalSmallFloatNans`
                // stores Metal's; a 32-bit float or normalized channel takes no half's NaN.
                let small = pf == PixelFormat::RG11B10Float;
                assert_eq!(
                    nan,
                    small && !small_nans,
                    "{what}: {:?}",
                    stage.specialized_notes
                );
                // Required wherever the module may store into a small float, on the deviation.
                let relies = relied.contains(&VkProperty::MetalSmallFloatNans);
                assert!(small_nans || !relies, "{what}");
                assert!(!small || relies == small_nans, "{what}");
                measured += usize::from(small && small_nans);
            }
            // The module of no storage choice serves every format the device writes without one:
            // its NaN note holds of the half-float classes (where the device's NaN is not Metal's)
            // and the small floats' alone.
            for &pf in PixelFormat::ALL {
                use PixelFormat as P;
                let half = matches!(
                    pf,
                    P::R16Float | P::RG16Float | P::RGBA16Float | P::RGBX16Float
                );
                let small = matches!(pf, P::RG11B10Float | P::RGB9E5Float);
                assert_eq!(
                    announced(&plain.stage, PrecisionNote::DeviceNanEncoding, pf),
                    (half && !nans) || (small && !small_nans),
                    "{entry} without a storage choice, bound as {pf:?}, on {stem}"
                );
            }
        }
    }
    assert!(
        noted > 0 && measured > 0,
        "{noted} noted, {measured} measured to store Metal's NaN"
    );
}
