//! The runtime's driver defects, keyed on the driver and its GPU (`driver_defects.rs`).

use super::super::driver_defects::{among, of, Affected, AFFECTED};
use metal2vulkan::device::DeviceDescription;
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::vk::DriverId;
use std::path::PathBuf;

const COPIES: RuntimeDriverDefect = RuntimeDriverDefect::MultisampledDepthStencilColourCopies;
const STENCIL_COPIES: RuntimeDriverDefect = RuntimeDriverDefect::MultisampledStencilColourCopies;
const DEPTHS: RuntimeDriverDefect = RuntimeDriverDefect::NormalizedDepthPastUnitRange;
const COMPRESSION: RuntimeDriverDefect = RuntimeDriverDefect::ImageSizeShortOfCompressionData;
const RESOLVES: RuntimeDriverDefect = RuntimeDriverDefect::DepthStencilResolveFirstSubresource;
const PLANE_VIEWS: RuntimeDriverDefect = RuntimeDriverDefect::LazilyAllocatedVolumePlaneViews;
const CACHE_LOAD: RuntimeDriverDefect = RuntimeDriverDefect::PipelineCacheDataCompiledAtLoad;
const COPY_REWRITES: RuntimeDriverDefect = RuntimeDriverDefect::CompressedImageCopiesRewriteTexels;
const NAN_COPIES: RuntimeDriverDefect = RuntimeDriverDefect::MultisampledCopiesConvertFloatNans;
const LATE_READS: RuntimeDriverDefect = RuntimeDriverDefect::ReadsCommandBuffersAfterTheirSignal;
const PREPROCESS: RuntimeDriverDefect = RuntimeDriverDefect::ImplicitPreprocessAfterExecutionSet;
const PAST_HEIGHT: RuntimeDriverDefect = RuntimeDriverDefect::ExplicitPlaneRowsPastHeight;
const COUNTS: RuntimeDriverDefect = RuntimeDriverDefect::OcclusionCountPerColorOutput;
const FREES: RuntimeDriverDefect = RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork;
const DESTROYS: RuntimeDriverDefect = RuntimeDriverDefect::DestroyedAddressBufferFailsSubmission;
const PUSHES: RuntimeDriverDefect = RuntimeDriverDefect::PushedBufferSizesUnset;
const STALE_PUSHES: RuntimeDriverDefect =
    RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds;
const BAKED_SAMPLERS: RuntimeDriverDefect = RuntimeDriverDefect::ImmutableSamplersLostAfterCopies;
const SET_SUPPORT: RuntimeDriverDefect = RuntimeDriverDefect::SetLayoutSupportUnderstated;
const PART_VIEWS: RuntimeDriverDefect = RuntimeDriverDefect::LazilyAllocatedSubresourceViews;
const BIND_COPIES: RuntimeDriverDefect = RuntimeDriverDefect::HostTransferImagesCopiedAtBind;
const LOST_SIGNALS: RuntimeDriverDefect = RuntimeDriverDefect::WaitForAnyObjectLosesASignal;
const CHROMA: RuntimeDriverDefect = RuntimeDriverDefect::TwoPlaneChromaMisplaced;
const TEN_BIT_WORDS: RuntimeDriverDefect = RuntimeDriverDefect::TenBitCodesReadAsWords;
const UNPINNABLE: RuntimeDriverDefect = RuntimeDriverDefect::UnpinnablePagesBreakHostMemory;

/// Every committed profile's driver has its defects at every release and GPU but those a probe
/// found clean: MoltenVK the multisampled depth/stencil colour copies, the inexact depths past
/// `[0, 1]`, the crashing 2D views of a 3D image in lazily allocated memory, the pipeline cache
/// compiled at load, the copies of compressed images that rewrite texels, the reads of command
/// buffers after their signal, the frees that fail pending work, the destroys that fail a
/// submission encoding meanwhile, the pushed sets' unset buffer sizes, the set layouts its support
/// query refuses though it runs them, the textureless views of some of a lazily allocated image's
/// levels or layers, the block-compressed image it copies out of host memory at bind time with a
/// one-block row pitch, the signal its wait for any of several objects loses and the misplaced
/// chroma of its two-plane Y′CbCr conversions; lavapipe the
/// colour copies, the depth/stencil resolve of the first layer and level, the explicit plane's
/// rows past its height, the baked samplers lost after a copy and the 10-bit codes its Y′CbCr
/// conversion reads as words;
/// RADV the multisampled copies that convert float NaNs, the 10-bit codes its Y′CbCr conversion
/// reads as words, and the implicit
/// preprocessing after an execution set on every GPU but the 780M (Phoenix's and Hawk Point's),
/// measured clean; ANV
/// the compressed image's short size, the occlusion count per color output and the pushed
/// descriptors a pipeline bind leaves stale on every GPU (Arrow Lake's, not measured, too);
/// NVIDIA's the
/// stencil colour copies on every GPU but the RTX 5080 Laptop GPU, measured clean (the GTX 1060 and
/// the RTX 5080 desktop GPU, not measured, take them), the 10-bit codes its Y′CbCr conversion reads
/// with their unused bits and the host memory a failed import of unpinnable pages breaks on every
/// GPU; SwiftShader none.
#[test]
fn the_committed_profiles_drivers_have_their_defects() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut drivers = 0;
    for path in std::fs::read_dir(&dir).expect("validation/profiles is readable") {
        let path = path.expect("directory entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("profile is readable");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
        let expected = match desc.driver_id {
            Some(DriverId::MOLTENVK) => {
                vec![
                    COPIES,
                    DEPTHS,
                    PLANE_VIEWS,
                    CACHE_LOAD,
                    COPY_REWRITES,
                    LATE_READS,
                    FREES,
                    DESTROYS,
                    PUSHES,
                    SET_SUPPORT,
                    PART_VIEWS,
                    BIND_COPIES,
                    LOST_SIGNALS,
                    CHROMA,
                ]
            }
            Some(DriverId::MESA_LLVMPIPE) => {
                vec![COPIES, RESOLVES, PAST_HEIGHT, BAKED_SAMPLERS, TEN_BIT_WORDS]
            }
            // The 780M, under Phoenix's `deviceID` and Hawk Point's.
            Some(DriverId::MESA_RADV) if matches!(desc.device_id, 0x15bf | 0x1900) => {
                vec![NAN_COPIES, TEN_BIT_WORDS]
            }
            Some(DriverId::MESA_RADV) => vec![NAN_COPIES, PREPROCESS, TEN_BIT_WORDS],
            Some(DriverId::INTEL_OPEN_SOURCE_MESA) => vec![COMPRESSION, COUNTS, STALE_PUSHES],
            Some(DriverId::NVIDIA_PROPRIETARY) if desc.device_id == 0x2c19 => {
                vec![TEN_BIT_WORDS, UNPINNABLE]
            }
            Some(DriverId::NVIDIA_PROPRIETARY) => vec![STENCIL_COPIES, TEN_BIT_WORDS, UNPINNABLE],
            _ => Vec::new(),
        };
        let defects: Vec<_> = of(&desc).into_iter().collect();
        assert_eq!(defects, expected, "{}", path.display());
        drivers += usize::from(!expected.is_empty());
    }
    assert!(
        drivers >= 14,
        "lavapipe's three, MoltenVK's two, ANV's four, RADV's three and NVIDIA's three"
    );
}

/// A driver has a defect on every GPU but one a probe found clean, at every release; another
/// driver, or none, has none of it.
#[test]
fn a_clean_gpu_alone_lacks_the_defect() {
    let table = [Affected {
        defect: COPIES,
        driver: DriverId::MESA_LLVMPIPE,
        clean: &[0],
    }];
    let lavapipe = Some(DriverId::MESA_LLVMPIPE);
    assert!(among(&table, lavapipe, 0).is_empty());
    assert!(among(&table, lavapipe, 1).contains(&COPIES));
    assert!(among(&table, Some(DriverId::MESA_RADV), 1).is_empty());
    assert!(among(&table, None, 1).is_empty());
}

/// NVIDIA's stencil colour-copy defect, measured on the RTX 3090 (`deviceID` 0x2204), is every
/// NVIDIA GPU's but the RTX 5080 Laptop GPU's (0x2c19), measured clean: the RTX 3070 (0x2484) and
/// an Ada GPU (0x2684) take it; a device that runs another driver has none. The unpinnable pages'
/// defect and the 10-bit codes read with their unused bits, measured on the 5080, are every NVIDIA
/// GPU's; no other driver has the unpinnable pages'.
#[test]
fn nvidia_s_colour_copies_are_clean_on_the_gpu_measured_alone() {
    let nvidia = Some(DriverId::NVIDIA_PROPRIETARY);
    let of = |device| among(AFFECTED, nvidia, device);
    for device in [0x2204, 0x2484, 0x2684] {
        assert_eq!(
            of(device).into_iter().collect::<Vec<_>>(),
            vec![STENCIL_COPIES, TEN_BIT_WORDS, UNPINNABLE],
            "{device:#x}"
        );
    }
    assert_eq!(
        of(0x2c19).into_iter().collect::<Vec<_>>(),
        vec![TEN_BIT_WORDS, UNPINNABLE]
    );
    let radv = among(AFFECTED, Some(DriverId::MESA_RADV), 0x2204);
    assert!(!radv.contains(&COPIES) && !radv.contains(&STENCIL_COPIES));
    for driver in [
        DriverId::MESA_RADV,
        DriverId::MESA_LLVMPIPE,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        DriverId::MOLTENVK,
    ] {
        assert!(!among(AFFECTED, Some(driver), 0x2c19).contains(&UNPINNABLE));
    }
}

/// RADV's implicit preprocessing after an execution set, measured broken on the RX 7900 XTX
/// (0x744c), is every RADV GPU's but the Radeon 780M's (`deviceID` 0x15bf), measured clean; a device
/// that runs another driver has none.
#[test]
fn radv_s_preprocessing_is_clean_on_the_gpu_measured_alone() {
    let radv = Some(DriverId::MESA_RADV);
    let of = |device| among(AFFECTED, radv, device);
    assert!(of(0x744c).contains(&PREPROCESS));
    assert!(!of(0x15bf).contains(&PREPROCESS));
    let nvidia = among(AFFECTED, Some(DriverId::NVIDIA_PROPRIETARY), 0x744c);
    assert!(!nvidia.contains(&PREPROCESS));
}

/// lavapipe's baked samplers lost after a copy, measured broken at 26.0.8, are lavapipe's at every
/// release. A device that runs another driver has none.
#[test]
fn lavapipe_s_baked_samplers_are_every_release_s() {
    let lavapipe = among(AFFECTED, Some(DriverId::MESA_LLVMPIPE), 0);
    assert!(lavapipe.contains(&BAKED_SAMPLERS));
    let radv = among(AFFECTED, Some(DriverId::MESA_RADV), 0);
    assert!(!radv.contains(&BAKED_SAMPLERS));
}

/// Hawk Point's Radeon 780M (`deviceID` 0x1900) is Phoenix's GPU (0x15bf): it takes Phoenix's
/// driver facts and defects, the implicit preprocessing measured clean among them, at every
/// release. A neighbouring `deviceID` (0x1901) is another GPU, which works the preprocessing
/// around.
#[test]
fn hawk_points_780m_is_phoenixs_gpu() {
    let (stem, phoenix) = super::capabilities::committed_descriptions()
        .into_iter()
        .find(|(stem, _)| stem == "amd-780m-radv")
        .expect("amd-780m-radv");
    let (_, measured) = super::capabilities::created(&stem, &phoenix);
    let mut hawk_point = phoenix.clone();
    hawk_point.device_id = 0x1900;
    let (_, properties) = super::capabilities::created(&stem, &hawk_point);
    assert_eq!(properties.driver_facts, measured.driver_facts);
    assert!(!properties.driver_facts.is_empty());
    assert_eq!(properties.driver_defects, measured.driver_defects);
    assert_eq!(
        of(&hawk_point).into_iter().collect::<Vec<_>>(),
        [NAN_COPIES, TEN_BIT_WORDS]
    );
    hawk_point.driver_version += 1;
    assert_eq!(
        of(&hawk_point).into_iter().collect::<Vec<_>>(),
        [NAN_COPIES, TEN_BIT_WORDS]
    );
    let mut neighbour = phoenix;
    neighbour.device_id = 0x1901;
    assert!(of(&neighbour).contains(&PREPROCESS));
}

/// The scheduler's thread waits for the GPU itself (one `vkWaitSemaphores` for any of the event
/// values its heads wait for, or the device's nudge) on every driver whose wait for any of several
/// objects loses no signal, and sleeps on its condition variable where one is lost
/// (`scheduler::waits_on_gpu`). On every committed MoltenVK profile it does not wait on the GPU,
/// and on every other committed profile it does; the decision reads the device's defects, so a
/// driver that gains or loses the row moves with it, and the driver's name decides nothing.
#[test]
fn the_scheduler_waits_on_the_gpu_where_no_signal_is_lost() {
    use crate::scheduler::waits_on_gpu;
    let mut moltenvk = 0;
    let mut others = 0;
    for (stem, desc) in super::capabilities::committed_descriptions() {
        let defects = of(&desc);
        let loses = desc.driver_id == Some(DriverId::MOLTENVK);
        assert_eq!(defects.contains(&LOST_SIGNALS), loses, "{stem}");
        assert_eq!(waits_on_gpu(&defects), !loses, "{stem}");
        *(if loses { &mut moltenvk } else { &mut others }) += 1;
    }
    assert!(
        moltenvk > 0 && others > 0,
        "{moltenvk} MoltenVK, {others} other"
    );
    // The defect alone decides: a set holding it waits on no GPU, and one without it does,
    // whatever else the set holds.
    assert!(!waits_on_gpu(&[LOST_SIGNALS].into_iter().collect()));
    assert!(!waits_on_gpu(
        &[LOST_SIGNALS, LATE_READS].into_iter().collect()
    ));
    assert!(waits_on_gpu(&Default::default()));
    assert!(waits_on_gpu(
        &[LATE_READS, BIND_COPIES].into_iter().collect()
    ));
}

/// Each clean GPU is a committed profile's GPU of the driver.
#[test]
fn every_clean_gpu_is_a_committed_profiles() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut committed = Vec::new();
    for path in std::fs::read_dir(&dir).expect("validation/profiles is readable") {
        let path = path.expect("directory entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("profile is readable");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
        committed.push((desc.driver_id, desc.canonical_device_id()));
    }
    for affected in AFFECTED {
        for &device in affected.clean {
            assert!(
                committed.contains(&(Some(affected.driver), device)),
                "{:?} clean on {device:#x}, which no committed profile describes",
                affected.defect
            );
        }
    }
}

/// A descriptor set layout of at most 1024 descriptors (the least `maxPerSetDescriptors` Vulkan
/// allows) is made without asking the driver; a larger one is made where the driver's
/// `vkGetDescriptorSetLayoutSupport` answers `VK_TRUE`, made and announced where it answers
/// `VK_FALSE` on a driver that runs what it refuses (`SetLayoutSupportUnderstated`), and refused
/// on any other: never made past the driver's answer unannounced.
#[test]
fn a_set_layout_past_the_least_per_set_limit_is_made_as_the_driver_answers() {
    use crate::pipeline::objects::{set_support, SetSupport, LEAST_PER_SET_DESCRIPTORS};
    let unasked = || -> bool { panic!("asked about a layout Vulkan promises") };
    for understated in [false, true] {
        for total in [0, 1, 1023, LEAST_PER_SET_DESCRIPTORS] {
            assert_eq!(
                set_support(total, understated, unasked),
                SetSupport::Supported
            );
        }
        let refused = match understated {
            true => SetSupport::Understated,
            false => SetSupport::Refused,
        };
        for total in [1025, 1212, 65536, 393_216] {
            assert_eq!(
                set_support(total, understated, || true),
                SetSupport::Supported
            );
            assert_eq!(set_support(total, understated, || false), refused);
        }
    }
}

/// A variant's kinds hold `set-past-layout-support` wherever a set of its layout is one its
/// driver may refuse though it runs it, whatever else the variant records: a bind announces only
/// the kinds its variant holds (`Kinds::of`), and the kinds were the stages' notes alone, so a
/// device announcing the notice at every occurrence (`DeviceOptions::announce_each`) announced it
/// only for a variant that also recorded a precision note. The kernel is `hsw_held`
/// (`validation/fixtures/public/engine/hsw_storage_writes.ll`, `llvm-as`), which writes a texture
/// it reaches through an argument buffer, compiled for the committed MoltenVK profile, whose heap
/// set holds far more than 1024 descriptors.
#[test]
fn a_set_past_the_drivers_support_is_among_its_variants_kinds() {
    use crate::degradation::{CompiledPipeline, Degradation, Kinds};
    use crate::pipeline::objects::{may_pass_support, LEAST_PER_SET_DESCRIPTORS};
    use metal2vulkan::device::{Device, DevicePolicy};
    use metal2vulkan::library::Library;
    use metal2vulkan::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};
    use metal2vulkan::Budget;

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation");
    let ll = root.join("fixtures/public/engine/hsw_storage_writes.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).expect("llvm-as");
    let text = std::fs::read_to_string(root.join("profiles/moltenvk-m3.json")).expect("profile");
    let description = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let device = Device::new(description, DevicePolicy::default()).expect("device");
    let function = Library::from_air(&bitcode)
        .expect("library")
        .function("hsw_held")
        .expect("function");
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(function)));
    let pipeline = device
        .compile_compute_pipeline(&descriptor, &Budget::standard())
        .expect("compiles");
    let totals: Vec<u64> = (pipeline.layout.sets.iter())
        .map(|set| set.bindings.iter().map(|b| u64::from(b.count)).sum())
        .collect();
    assert!(
        totals
            .iter()
            .any(|&total| total > LEAST_PER_SET_DESCRIPTORS),
        "no set past {LEAST_PER_SET_DESCRIPTORS} descriptors: {totals:?}"
    );
    let stages = [pipeline.stage.clone()];
    for understated in [true, false] {
        let kinds = Kinds::of(&stages, CompiledPipeline::Kernel(&pipeline), understated);
        assert_eq!(
            kinds.holds(Degradation::SetPastLayoutSupport),
            understated,
            "sets of {totals:?} descriptors, the driver understating its support: {understated}"
        );
    }
    // The kinds' test agrees with the making's: a layout whose making may be past the driver's
    // answer is exactly one `set_support` asks the driver about where it understates.
    use crate::pipeline::objects::{set_support, SetSupport};
    for understated in [false, true] {
        for total in [0, 1023, LEAST_PER_SET_DESCRIPTORS, 1025, 1212, 65536] {
            assert_eq!(
                may_pass_support(total, understated),
                set_support(total, understated, || false) == SetSupport::Understated,
                "{total} descriptors, understated: {understated}"
            );
        }
    }
}
