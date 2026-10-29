//! The table of measured driver facts (`device::driver_facts`), keyed on the driver and its GPU
//! at every release, and the runtime's facts each committed profile's driver has.

use crate::device::driver_facts::{among, Measured, MEASURED, NVIDIA_RTX_3090};
use crate::device::{DeviceDescription, DriverFact};
use crate::vk::DriverId;
use std::collections::BTreeMap;
use std::path::PathBuf;

const APPLE: u32 = 0x106B;

/// A driver has a fact at every release: a row of every GPU of the driver holds on any, a row of
/// one vendor's GPUs on that vendor's alone, a row of one GPU on that GPU alone, and no row holds
/// on another driver or a device that names none.
#[test]
fn a_fact_holds_on_its_driver_vendor_and_gpu_at_every_release() {
    let radv = Some(DriverId::MESA_RADV);
    let every = Measured {
        fact: 1u8,
        driver: DriverId::MESA_RADV,
        vendor: None,
        device: None,
    };
    let amd = Measured {
        vendor: Some(0x1002),
        ..every
    };
    let gpu = Measured {
        device: Some(0x15bf),
        ..amd
    };
    let at = |row: Measured<u8>, driver, identity| !among(&[row], driver, identity).is_empty();
    assert!(at(every, radv, (0x8086, 0x15bf)));
    assert!(at(amd, radv, (0x1002, 0x744c)));
    assert!(!at(amd, radv, (0x8086, 0x744c)));
    assert!(at(gpu, radv, (0x1002, 0x15bf)));
    assert!(!at(gpu, radv, (0x1002, 0x744c)));
    assert!(!at(every, Some(DriverId::MESA_LLVMPIPE), (0x1002, 0x15bf)));
    assert!(!at(every, None, (0x1002, 0x15bf)));
}

/// The deviations a profile takes, read from the routes it derives, each as
/// `DeviceProfile::takes_driver_deviation` answers it too.
fn taken(p: &crate::device::DeviceProfile) -> Vec<crate::error::DriverDeviation> {
    use crate::device::profile::{CubeGradientRoute, FloatFunctions, NanEncoding};
    use crate::device::requirements::VkProperty;
    use crate::error::DriverDeviation as D;
    let f = p.fields();
    let property = |v| f.enabled.properties.contains(&v);
    let routes = (f.cube_gradients.graphics, f.cube_gradients.compute);
    [
        (
            D::SimdgroupLockstep,
            property(VkProperty::SimdgroupLockstep),
        ),
        (
            D::MetalFloatFunctions,
            f.float_functions == FloatFunctions::Metal,
        ),
        (
            D::MetalCanonicalNans,
            f.nan_encoding == NanEncoding::MetalCanonical,
        ),
        (D::MetalSampler, property(VkProperty::MetalSampler)),
        (D::CubeGradientRoute, routes.1 == CubeGradientRoute::Metal),
        (
            D::GraphicsCubeGradientTangents,
            routes.0 == CubeGradientRoute::AppleProjected,
        ),
        (D::MetalColorWrites, property(VkProperty::MetalColorWrites)),
        (
            D::MetalFloatColorNans,
            property(VkProperty::MetalFloatColorNans),
        ),
        (
            D::MetalHalfColorNans,
            property(VkProperty::MetalHalfColorNans),
        ),
        (D::MetalHalfFma, property(VkProperty::MetalHalfFma)),
        (
            D::MetalFloatDenormals,
            property(VkProperty::MetalFloatDenormals),
        ),
        (
            D::MetalCenterVaryings,
            property(VkProperty::MetalCenterVaryings),
        ),
        (
            D::MetalSmallFloatColorWrites,
            property(VkProperty::MetalSmallFloatColorWrites),
        ),
        (
            D::MetalNorm16ColorWrites,
            property(VkProperty::MetalNorm16ColorWrites),
        ),
        (D::MetalMinMax, property(VkProperty::MetalMinMax)),
        (
            D::MetalHalfArithmetic,
            property(VkProperty::MetalHalfArithmetic),
        ),
        (
            D::MetalHalfStorageWrites,
            property(VkProperty::MetalHalfStorageWrites),
        ),
        (
            D::MetalHalfStorageNans,
            property(VkProperty::MetalHalfStorageNans),
        ),
        (
            D::MetalFragmentSimdgroups,
            property(VkProperty::MetalFragmentSimdgroups),
        ),
        (
            D::MetalVertexSimdgroups,
            property(VkProperty::MetalVertexSimdgroups),
        ),
        (
            D::CenterFragCoordDepth,
            property(VkProperty::CenterFragCoordDepth),
        ),
        (
            D::SampleFragCoordDepth,
            property(VkProperty::SampleFragCoordDepth),
        ),
        (
            D::MetalHalfTexelWrites,
            property(VkProperty::MetalHalfTexelWrites),
        ),
        (
            D::MetalNormStorageWrites,
            property(VkProperty::MetalNormStorageWrites),
        ),
        (
            D::MetalDepth16Stores,
            property(VkProperty::MetalDepth16Stores),
        ),
        (
            D::MetalDepth16BiasedStores,
            property(VkProperty::MetalDepth16BiasedStores),
        ),
        (
            D::MetalInterpolation,
            property(VkProperty::MetalInterpolation),
        ),
        (
            D::MetalFirstVertexIndices,
            property(VkProperty::MetalFirstVertexIndices),
        ),
        (D::MetalFullQuads, property(VkProperty::MetalFullQuads)),
        (
            D::MetalNormalizedVertexFetch,
            property(VkProperty::MetalNormalizedVertexFetch),
        ),
        (
            D::MetalAlphaCoverage,
            property(VkProperty::MetalAlphaCoverage),
        ),
    ]
    .into_iter()
    .inspect(|&(d, taken)| {
        // The public answer agrees with the routes (the gate lanes ask it).
        assert_eq!(p.takes_driver_deviation(d), taken, "{d:?}");
    })
    .filter_map(|(d, taken)| taken.then_some(d))
    .collect()
}

/// The deviations each committed profile takes are those its driver is measured with on its GPU,
/// at every release: every Apple GPU's (MoltenVK's lockstep, `float` functions, canonical NaNs,
/// sampler, compute cube route and the rest of the seventeen), with the graphics cube route and the
/// per-sample depth of macOS 27's Apple9 identity, that identity's normalized
/// storage writes, full quads, normalized vertex fetch and alpha to coverage, and every Apple GPU's 16-bit unorm depth stores, interpolation
/// and first-vertex indices; lavapipe's `float` color NaNs and half-float storage writes; RADV's on
/// the Radeon 780M, ANV's on the Iris Xe of RPL-P and the UHD 770 of RPL-S, and NVIDIA's on the
/// RTX 3090 and the RTX 5080 Laptop GPU, as measured there; the 8-bit linear filter's bound where
/// measured. Every other committed profile takes none. MoltenVK on another
/// vendor's GPU takes none, its cube routes unmeasured; another driver on Apple's vendor takes its
/// own driver's rows, with Vulkan's cube rule. `InterpolateAtOffsetCentered` follows the driver alone.
#[test]
fn the_profile_takes_the_deviations_its_driver_is_measured_with() {
    use crate::device::profile::{CubeGradientRoute, CubeGradients};
    use crate::device::requirements::VkProperty;
    use crate::device::{Device, DevicePolicy};
    use crate::error::DriverDeviation as D;
    use crate::spirv::testing::is_moltenvk_capture;
    // The seventeen of every Apple GPU measured, and the M3 Max's graphics cube route.
    let apple = vec![
        D::SimdgroupLockstep,
        D::MetalFloatFunctions,
        D::MetalCanonicalNans,
        D::MetalSampler,
        D::CubeGradientRoute,
        D::MetalColorWrites,
        D::MetalHalfFma,
        D::MetalFloatDenormals,
        D::MetalCenterVaryings,
        D::MetalMinMax,
        D::MetalHalfArithmetic,
        D::MetalHalfStorageWrites,
        D::MetalHalfStorageNans,
        D::MetalFragmentSimdgroups,
        D::MetalVertexSimdgroups,
    ];
    // NVIDIA's 615.71.09 on the RTX 3090 and the RTX 5080 Laptop GPU: the color-NaN stores, the
    // `float` denormals, the half-float storage writes, and the half texel writes' NaN bits.
    let nvidia_measured = vec![
        D::MetalFloatColorNans,
        D::MetalHalfColorNans,
        D::MetalFloatDenormals,
        D::MetalHalfStorageWrites,
        D::MetalHalfTexelWrites,
    ];
    // In `taken`'s order: the graphics route of the identity both chips share under macOS 27
    // before the color writes, and its per-sample depth last.
    let mut m4_measured = apple.clone();
    m4_measured.insert(5, D::GraphicsCubeGradientTangents);
    m4_measured.push(D::CenterFragCoordDepth);
    // The M4's identity's normalized storage writes, measured on it alone.
    m4_measured.push(D::MetalNormStorageWrites);
    // Every Apple GPU's 16-bit unorm depth, unbiased and biased, its interpolation and its
    // first-vertex indices, last in `taken`'s order.
    let mut apple_all = apple.clone();
    for list in [&mut m4_measured, &mut apple_all] {
        list.push(D::MetalDepth16Stores);
        list.push(D::MetalDepth16BiasedStores);
        list.push(D::MetalInterpolation);
        list.push(D::MetalFirstVertexIndices);
    }
    // macOS 27's full quads, measured on the M4's identity alone, and its normalized vertex
    // fetch and alpha to coverage, measured on the M3 Max under it.
    m4_measured.push(D::MetalFullQuads);
    m4_measured.push(D::MetalNormalizedVertexFetch);
    m4_measured.push(D::MetalAlphaCoverage);
    let all_of = |d: &DeviceDescription| match d.device_id {
        crate::device::driver_facts::APPLE9 => m4_measured.clone(),
        _ => apple_all.clone(),
    };
    let profile_of = |d: DeviceDescription| {
        Device::new(d, DevicePolicy::default())
            .expect("device")
            .profile()
            .clone()
    };
    let centered = |p: &crate::device::DeviceProfile| {
        p.fields()
            .enabled
            .properties
            .contains(&VkProperty::InterpolateAtOffsetCentered)
    };
    let committed = super::committed_profiles();
    assert!(committed.len() > 10);
    for (stem, _) in &committed {
        let d = super::profile(stem);
        let want_all = all_of(&d);
        let p = profile_of(d);
        let moltenvk = is_moltenvk_capture(stem);
        let want = match (moltenvk, stem.as_str()) {
            (true, _) => want_all,
            (false, "nvidia-rtx5080-laptop") => nvidia_measured.clone(),
            (false, "lavapipe" | "gate-lavapipe" | "gate-lavapipe-x86_64") => {
                vec![D::MetalFloatColorNans, D::MetalHalfStorageWrites]
            }
            // The per-sample `FragCoord.z` last, in `taken`'s order. The 780M under Phoenix's
            // `deviceID` and Hawk Point's alike.
            (false, "amd-780m-radv" | "amd-780m-hawkpoint-radv") => vec![
                D::MetalFloatDenormals,
                D::MetalSmallFloatColorWrites,
                D::MetalNorm16ColorWrites,
                D::MetalHalfStorageWrites,
                D::SampleFragCoordDepth,
                D::MetalHalfTexelWrites,
                D::MetalNormStorageWrites,
                D::MetalDepth16Stores,
            ],
            (false, "intel-irisxe-rplp-anv") => vec![
                D::MetalFloatDenormals,
                D::MetalSmallFloatColorWrites,
                D::MetalHalfStorageWrites,
                D::MetalDepth16Stores,
            ],
            (false, "intel-uhd770-rpls-anv") => vec![
                D::MetalFloatDenormals,
                D::MetalSmallFloatColorWrites,
                D::MetalHalfStorageWrites,
                D::MetalDepth16Stores,
            ],
            (false, _) => Vec::new(),
        };
        assert_eq!(taken(&p), want, "{stem}");
        assert_eq!(centered(&p), !moltenvk, "{stem}");
    }
    let m3 = super::profile("moltenvk-m3");
    assert_eq!(
        (m3.driver_id, m3.vendor_id, m3.device_id),
        (
            Some(DriverId::MOLTENVK),
            APPLE,
            crate::device::driver_facts::APPLE9
        )
    );
    // macOS 27's graphics cube route is the fixup's projected form, its compute route Metal's.
    assert_eq!(
        profile_of(m3.clone()).fields().cube_gradients,
        CubeGradients {
            graphics: CubeGradientRoute::AppleProjected,
            compute: CubeGradientRoute::Metal,
        }
    );
    // Every release of the driver takes the same.
    for version in [10400, 10401, 10402, 10403] {
        let mut other = m3.clone();
        other.driver_version = version;
        let p = profile_of(other);
        assert_eq!(taken(&p), m4_measured, "{version}");
        assert!(!centered(&p), "{version}");
    }
    // Another vendor's GPU: none, the cube routes unmeasured and so Vulkan's rule.
    let vulkan = CubeGradients {
        graphics: CubeGradientRoute::Vulkan,
        compute: CubeGradientRoute::Vulkan,
    };
    let mut intel = m3.clone();
    intel.vendor_id = 0x8086;
    let p = profile_of(intel);
    assert_eq!(taken(&p), Vec::new());
    assert_eq!(p.fields().cube_gradients, vulkan);
    assert!(!centered(&p));
    // Another driver on Apple's vendor: that driver's rows of every GPU (lavapipe's), with
    // Vulkan's rule and a centered offset.
    let mut lavapipe = m3;
    lavapipe.driver_id = Some(DriverId::MESA_LLVMPIPE);
    let p = profile_of(lavapipe);
    assert_eq!(
        taken(&p),
        [D::MetalFloatColorNans, D::MetalHalfStorageWrites]
    );
    assert_eq!(p.fields().cube_gradients, vulkan);
    assert!(centered(&p));
    // NVIDIA's color-NaN stores and `float` denormals: per GPU (the 3090 and the 5080 Laptop
    // GPU, not another RTX 50), at every release; the small float color writes the 3090's alone.
    use crate::device::driver_facts::NVIDIA_RTX_5080_LAPTOP;
    let laptop = super::profile("nvidia-rtx5080-laptop");
    assert_eq!(laptop.device_id, NVIDIA_RTX_5080_LAPTOP);
    let mut rtx_3090 = laptop.clone();
    rtx_3090.device_id = NVIDIA_RTX_3090;
    let mut desktop = laptop.clone();
    desktop.device_id = 0x2c02;
    let mut later = laptop.clone();
    later.driver_version += 1 << 6;
    let mut rtx_3090_measured = nvidia_measured.clone();
    rtx_3090_measured.insert(3, D::MetalSmallFloatColorWrites);
    assert_eq!(taken(&profile_of(rtx_3090)), rtx_3090_measured);
    assert_eq!(taken(&profile_of(desktop)), Vec::new());
    assert_eq!(taken(&profile_of(later)), nvidia_measured);
}

const ZERO: DriverFact = DriverFact::RobustIndexFetchReadsZero;
const DENORMALS: DriverFact = DriverFact::DepthImageKeepsDenormals;
const OUTSIDE: DriverFact = DriverFact::DepthImageKeepsValuesOutsideUnitRange;
const FRAGDEPTH: DriverFact = DriverFact::FragmentDepthStoresUnchanged;
const APART: DriverFact = DriverFact::DepthStencilImagesLieApart;
const AVERAGE: DriverFact = DriverFact::ColorResolveAverageIsMetals;
const CLEAR: DriverFact = DriverFact::ColorClearIsMetals;
const HIDE: DriverFact = DriverFact::CompressedImagesHideAliasedWrites;
const HOST: DriverFact = DriverFact::HostTransferImagesKeepTheirBytes;
const OPTIMIZED: DriverFact = DriverFact::DepthStencilImagesStayGpuOptimized;
const DEPTH16: DriverFact = DriverFact::FragmentDepthKeepsDepth16Codes;
const FETCH: DriverFact = DriverFact::SampledFetchRecoversCodes;
const SPLIT: DriverFact = DriverFact::InterlockedPassSplitNoSlower;
const RESTART: DriverFact = DriverFact::DynamicRestartWarns;
const FILL: DriverFact = DriverFact::FillCompiledPerCommandPool;
const DONT_CARE: DriverFact = DriverFact::DontCareStoreIsMetals;
const STAMPS: DriverFact = DriverFact::TimestampsWaitForEveryQueue;
const COLOR_BLOCKS: DriverFact = DriverFact::ColorBlockDecodeIsMetals;
const CHANNEL_BLOCKS: DriverFact = DriverFact::ChannelBlockDecodeIsMetals;
const PLANES: DriverFact = DriverFact::PositionedDepthReadAtStandardPositions;
const TESTS_DENORMALS: DriverFact = DriverFact::DepthTestKeepsDenormals;
const TESTS_MAGNITUDES: DriverFact = DriverFact::DepthTestComparesMagnitudes;
const SPARSE_WAIT: DriverFact = DriverFact::FirstSparseResourceWaitsForQueues;
const BC6H: DriverFact = DriverFact::Bc6hBlockDecodeIsMetals;
const BC7: DriverFact = DriverFact::Bc7BlockDecodeIsMetals;
const SCARCE_POOLS: DriverFact = DriverFact::TimestampPoolsAreCounterSampleBuffers;
const ONE_OF_METALS: DriverFact = DriverFact::DontCareStoreIsOneOfMetals;
const COPY_BACK: DriverFact = DriverFact::ShadowCopyBackDispatchFaster;
const COPY_IN: DriverFact = DriverFact::ShadowCopyInDispatchFaster;
const ROWS: DriverFact = DriverFact::RowCopiesDispatchFaster;
const MERGES: DriverFact = DriverFact::SubmissionsWithoutSignalsMerge;
const MESSENGER: DriverFact = DriverFact::QueueFailuresReportedByMessenger;
const STALL: DriverFact = DriverFact::SemaphoreOperationsStallSubmits;
const STALL_1024: DriverFact = DriverFact::SignallingSubmitsStallEvery1024;
const SAMPLED_HIZ: DriverFact = DriverFact::SampledDepthKeepsHiZ;
const MSAA_DEPTH: DriverFact = DriverFact::MultisampledDepthFasterOutOfGeneral;
const TILE_STORE: DriverFact = DriverFact::TileStoreIsMetals;
const FOLLOWS: DriverFact = DriverFact::PointerImportsFollowTheirAddress;
const INTERLOCK_ORDER: DriverFact = DriverFact::InterlockOrdersDraws;
const BLENDLESS: DriverFact = DriverFact::BlendlessColorAttachmentsWarn;
const HEAP_COLORS: DriverFact = DriverFact::ColorImagesLieAsMetalHeapTextures;
const HEAP_MULTISAMPLE_COLORS: DriverFact =
    DriverFact::MultisampleColorImagesLieAsMetalHeapTextures;

/// Each committed profile's driver has the runtime's driver facts its rows name, on its vendor and
/// GPU, at its release as at any other.
#[test]
fn the_committed_profiles_drivers_have_their_facts() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut claims = BTreeMap::new();
    for path in std::fs::read_dir(&dir).expect("validation/profiles is readable") {
        let path = path.expect("directory entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("profile is readable");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
        let facts: Vec<_> = desc.driver_facts().into_iter().collect();
        if !facts.is_empty() {
            let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
            claims.insert(stem, facts);
        }
    }
    let expected: BTreeMap<String, Vec<DriverFact>> = [
        (
            "amd-780m-hawkpoint-radv",
            vec![
                ZERO,
                FETCH,
                SPLIT,
                COLOR_BLOCKS,
                BC6H,
                BC7,
                ONE_OF_METALS,
                MERGES,
                FOLLOWS,
            ],
        ),
        (
            "amd-780m-radv",
            vec![
                ZERO,
                FETCH,
                SPLIT,
                COLOR_BLOCKS,
                BC6H,
                BC7,
                ONE_OF_METALS,
                MERGES,
                FOLLOWS,
            ],
        ),
        ("amd-rdna2-apu-radv", vec![ZERO]),
        ("amd-rdna3-radv", vec![ZERO]),
        (
            "gate-lavapipe",
            vec![DENORMALS, FRAGDEPTH, DEPTH16, FETCH, BC6H, MERGES, FOLLOWS],
        ),
        (
            "gate-lavapipe-x86_64",
            vec![DENORMALS, FRAGDEPTH, DEPTH16, FETCH, BC6H, MERGES, FOLLOWS],
        ),
        (
            "intel-arl-anv",
            vec![HIDE, HOST, SPARSE_WAIT, COPY_BACK, ROWS],
        ),
        (
            "intel-irisxe-anv",
            vec![HIDE, HOST, SPARSE_WAIT, COPY_BACK, ROWS],
        ),
        (
            "intel-irisxe-rplp-anv",
            vec![
                HIDE,
                HOST,
                FETCH,
                SPLIT,
                SPARSE_WAIT,
                BC6H,
                BC7,
                ONE_OF_METALS,
                COPY_BACK,
                ROWS,
                MERGES,
                SAMPLED_HIZ,
                MSAA_DEPTH,
                FOLLOWS,
                INTERLOCK_ORDER,
            ],
        ),
        (
            "intel-uhd770-rpls-anv",
            vec![
                ZERO,
                HIDE,
                HOST,
                FETCH,
                SPLIT,
                SPARSE_WAIT,
                ONE_OF_METALS,
                COPY_BACK,
                COPY_IN,
                ROWS,
                MERGES,
            ],
        ),
        (
            "lavapipe",
            vec![DENORMALS, FRAGDEPTH, DEPTH16, FETCH, BC6H, MERGES, FOLLOWS],
        ),
        (
            "moltenvk-m3",
            vec![
                DENORMALS,
                OUTSIDE,
                APART,
                AVERAGE,
                CLEAR,
                OPTIMIZED,
                DEPTH16,
                FETCH,
                SPLIT,
                RESTART,
                BLENDLESS,
                FILL,
                DONT_CARE,
                STAMPS,
                COLOR_BLOCKS,
                CHANNEL_BLOCKS,
                TESTS_DENORMALS,
                TESTS_MAGNITUDES,
                BC6H,
                BC7,
                SCARCE_POOLS,
                MESSENGER,
                TILE_STORE,
                HEAP_COLORS,
                HEAP_MULTISAMPLE_COLORS,
            ],
        ),
        ("nvidia-pascal-linux", vec![DENORMALS, PLANES]),
        ("nvidia-rtx50-linux", vec![DENORMALS, PLANES]),
        (
            "nvidia-rtx5080-laptop",
            vec![
                ZERO, DENORMALS, FETCH, SPLIT, PLANES, BC6H, BC7, ROWS, STALL, STALL_1024,
            ],
        ),
    ]
    .into_iter()
    .map(|(stem, facts)| (stem.to_owned(), facts))
    .collect();
    assert_eq!(claims, expected);
}

/// One GPU that reports more than one `deviceID` takes its measured rows under each: the Radeon
/// 780M of Hawk Point (0x1900) is Phoenix's (0x15bf), so a RADV description of it has Phoenix's
/// driver facts, deviations and defects, and derives Phoenix's profile, its Function storage among
/// it, and fingerprint, at every release. A neighbouring `deviceID` (0x1901), or 0x1900 of another
/// vendor, is another GPU: it takes only the rows of every GPU of the driver.
#[test]
fn a_gpu_under_another_of_its_device_ids_takes_its_measured_rows() {
    use crate::device::driver_facts::{defects, deviations};
    use crate::device::{Device, DevicePolicy, FunctionStorage};
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let text = std::fs::read_to_string(dir.join("amd-780m-radv.json")).expect("amd-780m-radv");
    let phoenix = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
    let profile = |desc: &DeviceDescription| {
        Device::new(desc.clone(), DevicePolicy::default())
            .expect("the device is made")
            .profile()
            .clone()
    };
    let mut hawk_point = phoenix.clone();
    hawk_point.device_id = 0x1900;
    assert_eq!(phoenix.canonical_device_id(), 0x15bf);
    assert_eq!(hawk_point.canonical_device_id(), 0x15bf);
    assert!(!hawk_point.driver_facts().is_empty());
    assert_eq!(hawk_point.driver_facts(), phoenix.driver_facts());
    assert!(!deviations(&hawk_point).is_empty());
    assert_eq!(deviations(&hawk_point), deviations(&phoenix));
    assert_eq!(defects(&hawk_point), defects(&phoenix));
    assert_eq!(
        profile(&hawk_point).fields().function_storage,
        FunctionStorage::Measured(96 * 1024)
    );
    assert_eq!(
        profile(&hawk_point).fingerprint(),
        profile(&phoenix).fingerprint()
    );

    let mut later = hawk_point.clone();
    later.driver_version += 1;
    assert_eq!(later.driver_facts(), phoenix.driver_facts());
    assert_eq!(deviations(&later), deviations(&phoenix));

    let mut neighbour = hawk_point.clone();
    neighbour.device_id = 0x1901;
    let mut other_vendor = hawk_point.clone();
    other_vendor.vendor_id = 0x8086;
    for (desc, device) in [(&neighbour, 0x1901), (&other_vendor, 0x1900)] {
        assert_eq!(desc.canonical_device_id(), device);
        assert!(
            desc.driver_facts().len() < phoenix.driver_facts().len(),
            "{device:#x}"
        );
    }
    assert!(deviations(&neighbour).len() < deviations(&phoenix).len());
    assert_eq!(
        profile(&neighbour).fields().function_storage,
        FunctionStorage::Assumed(16 * 1024)
    );
}

/// NVIDIA's driver stalls a submission about once in every thousand semaphore operations on the
/// two GPUs measured, the RTX 5080 Laptop GPU and the RTX 3090, which take the fact at every
/// release; another GPU of the driver (an RTX 5080 desktop GPU) does not, so it waits for
/// completed timelines as every other driver does. No other driver has it.
#[test]
fn nvidia_s_semaphore_stall_holds_on_the_gpus_measured() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let read = |stem: &str| {
        let text = std::fs::read_to_string(dir.join(format!("{stem}.json"))).expect(stem);
        DeviceDescription::from_khronos_json(&text, None).expect("profile parses")
    };
    let laptop = read("nvidia-rtx5080-laptop");
    assert!(laptop.driver_facts().contains(&STALL));
    let mut rtx_3090 = laptop.clone();
    rtx_3090.device_id = NVIDIA_RTX_3090;
    assert!(rtx_3090.driver_facts().contains(&STALL));
    let mut desktop = laptop.clone();
    desktop.device_id = 0x2c02;
    assert!(!desktop.driver_facts().contains(&STALL));
    let mut later = laptop.clone();
    later.driver_version += 1 << 6;
    assert!(later.driver_facts().contains(&STALL));
    let drivers: Vec<DriverId> = MEASURED
        .iter()
        .filter(|row| row.fact == STALL)
        .map(|row| row.driver)
        .collect();
    assert_eq!(drivers, [DriverId::NVIDIA_PROPRIETARY; 2]);
}

/// NVIDIA's driver stalls in one signalling submission of every 1024 to a queue on the GPU
/// measured, the RTX 5080 Laptop GPU, at every release, so the runtime takes that stall while
/// idle there; the RTX 3090 and an RTX 5080 desktop GPU, not measured in that form, do not take
/// the fact, nor does RADV, measured without it, or any other driver.
#[test]
fn nvidia_s_every_1024th_signalling_submit_stall_holds_on_the_gpu_measured() {
    use crate::device::driver_facts::NVIDIA_RTX_5080_LAPTOP;
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let read = |stem: &str| {
        let text = std::fs::read_to_string(dir.join(format!("{stem}.json"))).expect(stem);
        DeviceDescription::from_khronos_json(&text, None).expect("profile parses")
    };
    let laptop = read("nvidia-rtx5080-laptop");
    assert!(laptop.driver_facts().contains(&STALL_1024));
    let mut later = laptop.clone();
    later.driver_version += 1 << 6;
    assert!(later.driver_facts().contains(&STALL_1024));
    let mut rtx_3090 = laptop.clone();
    rtx_3090.device_id = NVIDIA_RTX_3090;
    assert!(!rtx_3090.driver_facts().contains(&STALL_1024));
    let mut desktop = laptop.clone();
    desktop.device_id = 0x2c02;
    assert!(!desktop.driver_facts().contains(&STALL_1024));
    for stem in [
        "amd-rdna3-radv",
        "moltenvk-m3",
        "intel-irisxe-rplp-anv",
        "lavapipe",
    ] {
        assert!(!read(stem).driver_facts().contains(&STALL_1024), "{stem}");
    }
    let rows: Vec<(DriverId, Option<u32>)> = MEASURED
        .iter()
        .filter(|row| row.fact == STALL_1024)
        .map(|row| (row.driver, row.device))
        .collect();
    assert_eq!(
        rows,
        [(DriverId::NVIDIA_PROPRIETARY, Some(NVIDIA_RTX_5080_LAPTOP))]
    );
}

/// RADV on the RX 7900 XTX takes the facts measured on it: its pointer imports follow their
/// address and its submissions without signals merge, as the 780M's do, and it reads zero past a
/// bound index range, RADV's on every GPU; no answer measured on the 780M alone (its block decodes,
/// `DONT_CARE` stores, interlocked split, sampled fetch) is taken. Another release takes the same.
#[test]
fn the_rx_7900_xtx_takes_the_facts_measured_on_it() {
    use crate::device::driver_facts::AMD_RX_7900_XTX;
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let text = std::fs::read_to_string(dir.join("amd-780m-radv.json")).expect("amd-780m-radv");
    let mut navi31 = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
    navi31.device_id = AMD_RX_7900_XTX;
    let measured =
        std::collections::BTreeSet::from([FOLLOWS, MERGES, DriverFact::RobustIndexFetchReadsZero]);
    assert_eq!(navi31.driver_facts(), measured);
    let mut other = navi31.clone();
    other.driver_version += 1;
    assert_eq!(other.driver_facts(), measured);
}

/// MoltenVK's warning of a pipeline whose restart may be disabled (`MVKPipeline.mm` lines 881-893
/// at 1.4.1, 883-895 at 1.4.2) is the diagnostic of a pipeline that makes the restart dynamic with
/// its topology dynamic or a strip or fan, exactly as the driver reports it; no other pipeline's,
/// and no other fact's. It is MoltenVK's on any GPU at every release, and no other driver's.
#[test]
fn moltenvk_warns_of_a_pipeline_whose_restart_is_dynamic() {
    use crate::device::DriverDiagnostic;
    use crate::vk::{DynamicState as D, PrimitiveTopology as T};
    let warning = DriverDiagnostic {
        id_name: "mvk-warn",
        id: 0,
        text: "VK_ERROR_FEATURE_NOT_PRESENT: Metal does not support disabling primitive restart."
            .into(),
    };
    let both = [D::PRIMITIVE_TOPOLOGY, D::PRIMITIVE_RESTART_ENABLE];
    let restart = [D::CULL_MODE, D::PRIMITIVE_RESTART_ENABLE];
    let topology = [D::PRIMITIVE_TOPOLOGY];
    let cases: [(&[D], T, bool); 10] = [
        (&both, T::TRIANGLE_LIST, true),
        (&both, T::LINE_LIST, true),
        (&both, T::POINT_LIST, true),
        (&restart, T::TRIANGLE_STRIP, true),
        (&restart, T::LINE_STRIP, true),
        (&restart, T::TRIANGLE_FAN, true),
        (&restart, T::TRIANGLE_LIST, false),
        (&topology, T::TRIANGLE_STRIP, false),
        (&[], T::TRIANGLE_STRIP, false),
        (&[D::CULL_MODE], T::LINE_STRIP, false),
    ];
    let blendless = [crate::vk::Format::R32_UINT];
    for (dynamic, top, warns) in cases {
        assert_eq!(
            RESTART.pipeline_creation_diagnostics(dynamic, top, &blendless),
            warns
                .then(|| warning.clone())
                .into_iter()
                .collect::<Vec<_>>(),
            "{dynamic:?} {top:?}"
        );
        for other in [ZERO, DENORMALS, FETCH, SPLIT] {
            assert!(other
                .pipeline_creation_diagnostics(dynamic, top, &blendless)
                .is_empty());
        }
    }
    let rows: Vec<_> = MEASURED
        .iter()
        .filter(|row| row.fact == RESTART)
        .map(|row| (row.driver, row.vendor, row.device))
        .collect();
    assert_eq!(rows, [(DriverId::MOLTENVK, None, None)]);
    let text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles/moltenvk-m3.json"),
    )
    .expect("moltenvk-m3");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
    for version in [10400, 10401, 10402, 10403] {
        desc.driver_version = version;
        assert!(desc.driver_facts().contains(&RESTART), "{version}");
    }
    desc.driver_id = Some(DriverId::MESA_LLVMPIPE);
    assert!(!desc.driver_facts().contains(&RESTART));
}

/// MoltenVK's warning of a color attachment whose format cannot blend (`MVKPipeline.mm`'s
/// `addFragmentOutputToPipeline` at 1.4.2), whether the pipeline blends it or not, is one
/// diagnostic for each such format, named as the registry names it, exactly as the driver reports
/// it; none for a pipeline without such an attachment, and none of another fact. It is measured on
/// the M4's identity (the Apple9 family under macOS 27), at every release.
#[test]
fn moltenvk_warns_of_each_color_attachment_it_cannot_blend() {
    use crate::device::DriverDiagnostic;
    use crate::vk::{Format as F, PrimitiveTopology as T};
    let warning = |name: &str| DriverDiagnostic {
        id_name: "mvk-warn",
        id: 0,
        text: format!(
            "VK_ERROR_FEATURE_NOT_PRESENT: Blending is enabled for attachment with format {name}, \
             which does not support it."
        )
        .into(),
    };
    let blendless = [F::R32_UINT, F::E5B9G9R9_UFLOAT_PACK32];
    assert_eq!(
        BLENDLESS.pipeline_creation_diagnostics(&[], T::TRIANGLE_LIST, &blendless),
        [
            warning("VK_FORMAT_R32_UINT"),
            warning("VK_FORMAT_E5B9G9R9_UFLOAT_PACK32")
        ]
    );
    // The text MoltenVK 1.4.2 reported for `R32_UINT` on the M3 Max (2026-10-21).
    assert_eq!(
        warning("VK_FORMAT_R32_UINT").text,
        "VK_ERROR_FEATURE_NOT_PRESENT: Blending is enabled for attachment with format \
         VK_FORMAT_R32_UINT, which does not support it."
    );
    assert!(BLENDLESS
        .pipeline_creation_diagnostics(&[], T::TRIANGLE_STRIP, &[])
        .is_empty());
    assert!(RESTART
        .pipeline_creation_diagnostics(&[], T::TRIANGLE_LIST, &blendless)
        .is_empty());
    let text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles/moltenvk-m3.json"),
    )
    .expect("moltenvk-m3");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
    for version in [10401, 10402, 10403] {
        desc.driver_version = version;
        assert!(desc.driver_facts().contains(&BLENDLESS), "{version}");
    }
}
