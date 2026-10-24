//! `families`: the GPU-family audit over every committed device profile
//! (`validation/profiles/`, read-only).
//!
//! A profile has no physical device to ask for an image's limits, so every image here is taken
//! at any size, layer count and sample count. The device test (validation's `runtime_lane`)
//! answers the same rows with the device's own limits.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use super::capabilities::{committed_descriptions, created};
use crate::families::formats::{answers, FormatAnswers, Missing};
use crate::families::{audit, Facts};
use crate::image_plan::{ImageFacts, ImagePlan};
use crate::{
    FamilyRow, FeatureRow, FormatCapabilities, FormatCapability, FormatRow, Framework, GpuFamily,
    LimitRow, Refusal, RowAudit, RowState, RuntimeGap,
};
use ash::vk;
use metal2vulkan::device::{Device, DevicePolicy, OpBits, VkExtension, VkFeature, VkLimit};
use metal2vulkan::error::{Capability, Feature, Needs, Unsupported};
use metal2vulkan::metal::format::WordAttachment;
use metal2vulkan::metal::PixelFormat;

/// Every family's rows on the device the profile `stem` describes, created as the runtime
/// creates one (`capabilities::created`).
fn audits(stem: &str) -> BTreeMap<GpuFamily, Vec<RowAudit>> {
    let (_, desc) = committed_descriptions()
        .into_iter()
        .find(|(s, _)| s == stem)
        .unwrap_or_else(|| panic!("no profile {stem}"));
    all_audits(stem, &desc)
}

/// Any size, layer count and sample count (the module docs).
fn any_limits(_: &ImagePlan) -> vk::ImageFormatProperties {
    vk::ImageFormatProperties {
        max_extent: vk::Extent3D {
            width: u32::MAX,
            height: u32::MAX,
            depth: u32::MAX,
        },
        max_mip_levels: 32,
        max_array_layers: u32::MAX,
        sample_counts: vk::SampleCountFlags::from_raw(0x7f),
        max_resource_size: u64::MAX,
    }
}

pub(super) fn all_audits(
    stem: &str,
    desc: &metal2vulkan::device::DeviceDescription,
) -> BTreeMap<GpuFamily, Vec<RowAudit>> {
    let (enabled, properties) = created(stem, desc);
    let compile = Device::new(enabled.clone(), DevicePolicy::UNTRUSTED).unwrap();
    let image = ImageFacts {
        profile: compile.profile(),
        formats: &enabled.formats,
        storage_multisample: enabled.has(VkFeature::ShaderStorageImageMultisample),
        depth_images: crate::image_plan::DepthImageFacts::of(&enabled.driver_facts()),
        linear_attachments: false,
        multisample_depth_twin: false,
    };
    let limits = any_limits;
    let memo: RefCell<BTreeMap<PixelFormat, FormatAnswers>> = RefCell::default();
    let format = |pf: PixelFormat| {
        if let Some(known) = memo.borrow().get(&pf) {
            return known.clone();
        }
        // A profile states no sparse support (the device's answer, the module docs).
        let sparse = Err(Missing::Gap(RuntimeGap::SparseHeaps));
        let found = answers(pf, &image, &limits, sparse);
        memo.borrow_mut().insert(pf, found.clone());
        found
    };
    let facts = Facts {
        profile: compile.profile(),
        metal: compile.metal_properties(),
        properties: &properties,
        format: &format,
    };
    GpuFamily::ALL
        .into_iter()
        .map(|family| (family, audit(family, &facts)))
        .collect()
}

/// The texture capabilities the runtime gives `pf` on the device created as `enabled`, whose
/// compile layer is `compile`, at `level`, as `Device::pixel_format_capabilities` answers them
/// (every image taken at any size, the module docs).
pub(super) fn format_capabilities(
    enabled: &metal2vulkan::device::DeviceDescription,
    compile: &Device,
    level: crate::MetalLevel,
    pf: PixelFormat,
) -> FormatCapabilities {
    if !level.describes(pf) {
        return FormatCapabilities::NONE;
    }
    let image = ImageFacts {
        profile: compile.profile(),
        formats: &enabled.formats,
        storage_multisample: enabled.has(VkFeature::ShaderStorageImageMultisample),
        depth_images: crate::image_plan::DepthImageFacts::of(&enabled.driver_facts()),
        linear_attachments: false,
        multisample_depth_twin: false,
    };
    let sparse = Err(Missing::Gap(RuntimeGap::SparseHeaps));
    answers(pf, &image, &any_limits, sparse).capabilities()
}

/// The state of `row` among `rows`, if listed.
fn state(rows: &[RowAudit], row: FamilyRow) -> Option<RowState> {
    rows.iter().find(|r| r.row == row).map(|r| r.state.clone())
}

fn feature(row: FeatureRow) -> FamilyRow {
    FamilyRow::Feature(row)
}

fn refused(feature: Feature, capability: Capability) -> RowState {
    RowState::Refused(Refusal::Unsupported(Unsupported::needs(
        feature, capability,
    )))
}

fn not_implemented(feature: Feature) -> RowState {
    RowState::Refused(Refusal::Unsupported(Unsupported::not_implemented(feature)))
}

/// On every committed profile: the families without a column have their one row, Apple1 and
/// Mac1 answer as Apple2 and Mac2, the frameworks never block, and every row has a state.
#[test]
fn every_committed_profile_audits_every_family() {
    let profiles = committed_descriptions();
    assert!(profiles.len() >= 18, "{} profiles", profiles.len());
    for (stem, desc) in &profiles {
        let all = all_audits(stem, desc);
        let one = |row: FamilyRow, state: RowState| vec![RowAudit { row, state }];
        assert_eq!(
            all[&GpuFamily::Metal4],
            one(FamilyRow::Metal4Api, RowState::Gap(RuntimeGap::Metal4Api)),
            "{stem}"
        );
        for catalyst in [GpuFamily::MacCatalyst1, GpuFamily::MacCatalyst2] {
            assert_eq!(
                all[&catalyst],
                one(
                    FamilyRow::MacCatalystProcess,
                    RowState::Refused(Refusal::NotACatalystProcess)
                ),
                "{stem}"
            );
        }
        assert_eq!(
            all[&GpuFamily::Apple11],
            one(
                FamilyRow::NoPublishedColumn,
                RowState::Refused(Refusal::NoPublishedColumn)
            ),
            "{stem}"
        );
        assert_eq!(all[&GpuFamily::Apple1], all[&GpuFamily::Apple2], "{stem}");
        assert_eq!(all[&GpuFamily::Mac1], all[&GpuFamily::Mac2], "{stem}");
        for (family, rows) in &all {
            assert!(!rows.is_empty(), "{stem} {family:?}");
            for r in rows {
                let framework = match r.row {
                    FamilyRow::Feature(FeatureRow::MetalKit) => Some(Framework::MetalKit),
                    FamilyRow::Feature(FeatureRow::MetalPerformanceShaders) => {
                        Some(Framework::MetalPerformanceShaders)
                    }
                    FamilyRow::Feature(
                        FeatureRow::MetalFxSpatialUpscaling
                        | FeatureRow::MetalFxTemporalUpscaling
                        | FeatureRow::MetalFxFrameInterpolation
                        | FeatureRow::MetalFxDenoisedUpscaling,
                    ) => Some(Framework::MetalFx),
                    _ => None,
                };
                match framework {
                    Some(framework) => {
                        assert_eq!(r.state, RowState::NotADeviceFeature(framework), "{stem}");
                        assert!(!r.state.blocks());
                    }
                    None => assert!(
                        !matches!(r.state, RowState::NotADeviceFeature(_)),
                        "{stem} {family:?} {:?}",
                        r.row
                    ),
                }
            }
            // Each row is listed once.
            let mut seen: Vec<FamilyRow> = rows.iter().map(|r| r.row).collect();
            seen.sort();
            seen.dedup();
            assert_eq!(seen.len(), rows.len(), "{stem} {family:?}");
        }
    }
}

/// The columns each family reads, by rows whose cells differ between the families: the 2026
/// edition's first Apple family, its Metal 4 rows, the 2023 edition's Common, Mac2 and Metal3
/// cells, and the format cells.
#[test]
fn each_family_lists_its_columns_rows() {
    let all = audits("nvidia-rtx5080-laptop");
    let listed = |family: GpuFamily, row: FamilyRow| all[&family].iter().any(|r| r.row == row);
    use FeatureRow as F;
    use GpuFamily as G;
    // BC: Apple9 on (some Apple7 and Apple8 devices only), Mac2, not Metal3 ("Varies").
    assert!(listed(G::Apple9, feature(F::BcPixelFormats)));
    assert!(!listed(G::Apple8, feature(F::BcPixelFormats)));
    assert!(listed(G::Mac2, feature(F::BcPixelFormats)));
    assert!(!listed(G::Metal3, feature(F::BcPixelFormats)));
    // The 2023 edition's one tier row, read onto the two.
    for family in [G::Metal3, G::Mac2, G::Apple6] {
        assert!(
            listed(family, feature(F::ArgumentBuffersTier2)),
            "{family:?}"
        );
    }
    assert!(listed(G::Apple5, feature(F::ArgumentBuffersTier1)));
    assert!(!listed(G::Apple5, feature(F::ArgumentBuffersTier2)));
    assert!(!listed(G::Common3, feature(F::ArgumentBuffersTier1)));
    // A row of Metal 4's programming model only is no family's.
    for family in GpuFamily::ALL {
        assert!(!listed(family, feature(F::ArgumentTables)), "{family:?}");
    }
    // A 2026 row of the Metal 3 model: Metal3 where its first family is at most Apple7.
    assert!(listed(G::Metal3, feature(F::RelaxedMath)));
    assert!(!listed(G::Mac2, feature(F::RelaxedMath)));
    assert!(!listed(G::Metal3, feature(F::SamplerLodBias)));
    assert!(listed(G::Apple10, feature(F::SamplerLodBias)));
    // A 2023 row only: Mac2's.
    assert!(listed(G::Mac2, feature(F::TextureBarriers)));
    assert!(!listed(G::Apple9, feature(F::TextureBarriers)));
    // Common1 to Common3.
    assert!(listed(G::Common1, feature(F::MetalKit)));
    assert!(!listed(G::Common1, feature(F::MetalPerformanceShaders)));
    assert!(listed(G::Common2, feature(F::MetalPerformanceShaders)));
    // Format cells: "Atomic, Write, Color, Sparse" on Apple6, MSAA from Apple7; none in a
    // Common column; "Varies" lists nothing.
    let r32 = |c| FamilyRow::Format(FormatRow::R32Uint, c);
    assert!(listed(G::Apple6, r32(FormatCapability::Atomic)));
    assert!(!listed(G::Apple5, r32(FormatCapability::Atomic)));
    assert!(!listed(G::Apple6, r32(FormatCapability::Msaa)));
    assert!(listed(G::Apple7, r32(FormatCapability::Msaa)));
    assert!(!all[&G::Common3]
        .iter()
        .any(|r| matches!(r.row, FamilyRow::Format(..))));
    let bc = FamilyRow::Format(FormatRow::BcFormats, FormatCapability::Filter);
    assert!(!listed(G::Apple8, bc));
    assert!(listed(G::Apple9, bc));
    let d24 = FamilyRow::Format(FormatRow::Depth24UnormStencil8, FormatCapability::Msaa);
    assert!(!listed(G::Mac2, d24));
    // A limit of a feature is listed where the feature is: Metal3 takes Apple7's limits, but
    // not its imageblocks'.
    let imageblock = FamilyRow::Limit(LimitRow::ExplicitImageblockMemory);
    assert!(listed(G::Apple7, imageblock));
    assert!(!listed(G::Metal3, imageblock));
    assert!(listed(
        G::Metal3,
        FamilyRow::Limit(LimitRow::ThreadgroupMemory)
    ));
}

/// States read from the device: framebuffer fetch's form, cube arrays, the SIMD and quad rows by
/// the stages' classes, and a limit's cell against the device's limit.
#[test]
fn states_follow_their_sources() {
    use FeatureRow as F;
    let mut storage_fetch = BTreeSet::new();
    for (stem, desc) in committed_descriptions() {
        let all = all_audits(&stem, &desc);
        let (enabled, _) = created(&stem, &desc);
        let apple9 = &all[&GpuFamily::Apple9];
        let blending = state(apple9, feature(F::ProgrammableBlending)).unwrap();
        // Through input attachments in rasterization order exact; through the pixel interlock's
        // storage form exact but where the device renders a color format only through its word
        // (the one- and two-channel sRGB formats on NVIDIA and ANV), whose fetch is the runtime's
        // gap; with neither, refused.
        if enabled.has(VkFeature::RasterizationOrderColorAttachmentAccess) {
            assert_eq!(blending, RowState::Exact, "{stem}");
        } else if enabled.has(VkFeature::FragmentShaderPixelInterlock) {
            let compile = Device::new(enabled.clone(), DevicePolicy::UNTRUSTED).unwrap();
            let words = (PixelFormat::ALL.iter())
                .any(|&pf| WordAttachment::of(pf, compile.profile()).is_some());
            let want = match words {
                true => not_implemented(Feature::FramebufferFetch),
                false => RowState::Exact,
            };
            assert_eq!(blending, want, "{stem}");
            // MoltenVK and RADV render every sRGB format the storage form fetches; NVIDIA and
            // ANV render `R8Unorm_sRGB` and `RG8Unorm_sRGB` through their words.
            let class = [
                ("moltenvk", true),
                ("amd-", true),
                ("nvidia", false),
                ("intel", false),
            ];
            if let Some(&(c, exact)) = class.iter().find(|(c, _)| stem.starts_with(c)) {
                assert_eq!(blending == RowState::Exact, exact, "{stem}");
                storage_fetch.insert(c);
            }
        } else {
            assert_eq!(
                blending,
                refused(
                    Feature::FramebufferFetch,
                    Capability::Feature(VkFeature::RasterizationOrderColorAttachmentAccess)
                ),
                "{stem}"
            );
        }
        let cube = state(apple9, feature(F::CubeMapTextureArrays)).unwrap();
        assert_eq!(
            cube == RowState::Exact,
            enabled.has(VkFeature::ImageCubeArray),
            "{stem}"
        );
        // Kernels emulate every simdgroup class, so each SIMD and quad row answers by the classes
        // its members lower to in the vertex and fragment stages: a quad's reductions and votes
        // read its lanes by the ballot and shuffles, a fill shuffles, and a quad permute's constant
        // lane takes Vulkan's quad class only where the stage has it, a shuffle elsewhere.
        let metal = Device::new(enabled.clone(), DevicePolicy::UNTRUSTED)
            .unwrap()
            .metal_properties()
            .clone();
        let both = |ops: OpBits| {
            [metal.simd.vertex, metal.simd.fragment]
                .iter()
                .all(|s| s.cross_lane.contains(ops))
        };
        let shuffles = OpBits::SHUFFLE.union(OpBits::BALLOT);
        for (row, ops) in [
            (F::SimdPermute, shuffles),
            (F::SimdReduction, OpBits::ARITHMETIC),
            (F::SimdShiftAndFill, OpBits::SHUFFLE),
            (F::QuadBallot, OpBits::BALLOT),
            (F::QuadPermute, shuffles),
            (F::QuadReduction, shuffles),
        ] {
            let got = state(apple9, feature(row)).unwrap();
            // MoltenVK on the M3 and the M4 runs a vertex function's ballot and shuffles as
            // Metal does (`DriverDeviation::MetalVertexSimdgroups`), and a constant lane's quad
            // permute shuffles there: every SIMD and quad row is exact (the consumer's key 33).
            if matches!(stem.as_str(), "moltenvk-m3" | "moltenvk-m4") {
                assert_eq!(got, RowState::Exact, "{stem}: {row:?}");
            }
            if both(ops) {
                assert_eq!(got, RowState::Exact, "{stem}: {row:?}");
            } else {
                assert!(
                    matches!(&got, RowState::Refused(Refusal::Unsupported(u))
                        if matches!(u.needs, Needs::AnyOf(_))),
                    "{stem}: {row:?} {got:?}"
                );
            }
        }
        // Apple3's texture buffers: 256 M texels (2^28) against `maxTexelBufferElements`.
        let width = state(
            &all[&GpuFamily::Apple3],
            FamilyRow::Limit(LimitRow::TextureBufferWidth),
        )
        .unwrap();
        let available = u64::from(enabled.limits.max_texel_buffer_elements);
        if available >= 1 << 28 {
            assert_eq!(width, RowState::Exact, "{stem}");
        } else {
            assert_eq!(
                width,
                refused(
                    Feature::TexelBuffer,
                    Capability::Limit {
                        limit: VkLimit::MaxTexelBufferElements,
                        required: 1 << 28,
                        available,
                    }
                ),
                "{stem}"
            );
        }
    }
    assert_eq!(storage_fetch.len(), 4, "{storage_fetch:?}");
}

/// Vertex amplification: Apple7's count of 8 is exact on a device with `multiview`, through the
/// layer output or the generated geometry stage where the device has fewer views; the feature row
/// needs `multiViewport` and the slice and viewport stores of the variant that draws one
/// amplification at a time, and then waits on tessellation pipelines drawn so.
#[test]
fn vertex_amplification_draws_past_the_views_one_at_a_time() {
    for (stem, desc) in committed_descriptions() {
        let all = all_audits(&stem, &desc);
        let (enabled, _) = created(&stem, &desc);
        let apple7 = &all[&GpuFamily::Apple7];
        let multiview = enabled.has(VkFeature::Multiview);
        let views = u64::from(enabled.limits.max_multiview_view_count);
        let count = state(apple7, FamilyRow::Limit(LimitRow::VertexAmplificationCount)).unwrap();
        let want = if !multiview {
            refused(
                Feature::VertexAmplification,
                Capability::Feature(VkFeature::Multiview),
            )
        } else if views < 8
            && !enabled.has(VkFeature::ShaderOutputLayer)
            && !enabled.has(VkFeature::GeometryShader)
        {
            RowState::Refused(Refusal::Unsupported(Unsupported::needs_any_of(
                Feature::VertexAmplification,
                vec![
                    vec![Capability::Feature(VkFeature::ShaderOutputLayer)],
                    vec![Capability::Feature(VkFeature::GeometryShader)],
                ],
            )))
        } else {
            RowState::Exact
        };
        assert_eq!(count, want, "{stem}");
        let row = state(apple7, feature(FeatureRow::VertexAmplification)).unwrap();
        let either = |output| enabled.has(output) || enabled.has(VkFeature::GeometryShader);
        if enabled.has(VkFeature::Multiview)
            && enabled.has(VkFeature::MultiViewport)
            && either(VkFeature::ShaderOutputLayer)
            && either(VkFeature::ShaderOutputViewportIndex)
        {
            assert_eq!(row, not_implemented(Feature::VertexAmplification), "{stem}");
        } else {
            assert!(
                matches!(&row, RowState::Refused(Refusal::Unsupported(u))
                    if matches!(u.needs, Needs::AnyOf(_))),
                "{stem}: {row:?}"
            );
        }
    }
}

/// Binary archives are exact wherever a family lists them (the vertex key's difference blocks no
/// family). The argument-buffer rows block on the device's lack alone: argument encoders and set 1
/// of draws and tile dispatches are built.
#[test]
fn archive_and_argument_buffer_rows() {
    use FeatureRow as F;
    for (stem, desc) in committed_descriptions() {
        let all = all_audits(&stem, &desc);
        for family in [GpuFamily::Apple3, GpuFamily::Mac2, GpuFamily::Metal3] {
            assert_eq!(
                state(&all[&family], feature(F::BinaryArchives)),
                Some(RowState::Exact),
                "{stem} {family:?}"
            );
        }
        let metal = Device::new(created(&stem, &desc).0, DevicePolicy::UNTRUSTED)
            .unwrap()
            .metal_properties()
            .clone();
        let tier2 = state(&all[&GpuFamily::Metal3], feature(F::ArgumentBuffersTier2)).unwrap();
        if metal.argument_buffers_tier == metal2vulkan::metal::ArgumentBuffersTier::Tier2 {
            assert_eq!(tier2, RowState::Exact, "{stem}");
        } else {
            assert!(
                matches!(&tier2, RowState::Refused(Refusal::Unsupported(u))
                    if u.feature == Feature::ArgumentBufferTextures),
                "{stem}: {tier2:?}"
            );
        }
    }
}

/// The point-size row answers from the device: exact where the upper bound of its `pointSizeRange`
/// reaches Metal's largest point (511 pixels, every family's cell), refused `PointSizeRange` where
/// it ends below (lavapipe's 256, ANV's 255.875); before, every device refused it. Each committed
/// profile is checked in every family that lists the row.
#[test]
fn the_point_size_row_answers_from_the_device() {
    let (mut exact, mut below) = (0, 0);
    for (stem, desc) in committed_descriptions() {
        let (enabled, _) = created(&stem, &desc);
        let reaches = enabled.limits.max_point_size >= 511;
        for (family, rows) in all_audits(&stem, &desc) {
            let Some(got) = state(&rows, FamilyRow::Limit(LimitRow::PointSize)) else {
                continue;
            };
            let want = match reaches {
                true => RowState::Exact,
                false => RowState::Refused(Refusal::PointSizeRange),
            };
            assert_eq!(got, want, "{stem} {family:?}");
        }
        *if reaches { &mut exact } else { &mut below } += 1;
    }
    assert!(
        exact >= 4 && below >= 4,
        "{exact} reach Metal's point, {below} do not"
    );
}

/// Families answer false where any row blocks; the burn-down names those rows.
#[test]
fn no_committed_profile_has_a_family_yet() {
    for (stem, desc) in committed_descriptions() {
        for (family, rows) in all_audits(&stem, &desc) {
            assert!(
                rows.iter().any(|r| r.state.blocks()),
                "{stem} {family:?} has no blocking row"
            );
        }
    }
}

/// Metal's values of the families, and the capability sets.
#[test]
fn family_values_and_capability_sets() {
    for family in GpuFamily::ALL {
        assert_eq!(GpuFamily::from_raw(family.raw()), Some(family));
    }
    assert_eq!(GpuFamily::Apple7.raw(), 1007);
    assert_eq!(GpuFamily::Apple11.raw(), 1011);
    assert_eq!(GpuFamily::Mac2.raw(), 2002);
    assert_eq!(GpuFamily::Common3.raw(), 3003);
    assert_eq!(GpuFamily::MacCatalyst2.raw(), 4002);
    assert_eq!(GpuFamily::Metal4.raw(), 5002);
    for raw in [0, 1000, 1012, 2003, 6001, -1009, i64::MIN, i64::MAX] {
        assert_eq!(GpuFamily::from_raw(raw), None, "{raw}");
    }
    let some = FormatCapabilities::of(&[FormatCapability::Write, FormatCapability::Atomic]);
    assert_eq!(
        some.iter().collect::<Vec<_>>(),
        [FormatCapability::Atomic, FormatCapability::Write]
    );
    assert!(some.contains(FormatCapability::Write));
    assert!(!some.contains(FormatCapability::Filter));
    assert_eq!(FormatCapabilities::NONE.iter().count(), 0);
    let all = FormatCapabilities::of(&FormatCapability::ALL);
    assert_eq!(all.iter().collect::<Vec<_>>(), FormatCapability::ALL);
}

/// The function-pointer rows: a table an argument buffer holds is read through the set-1 table
/// array, so a device without its four features refuses them as the compile layer does, naming
/// them; elsewhere they are exact. Recursion through a table runs on a call stack, through a
/// pointer passed as a value since item 24, and a pointer chosen at run time among functions'
/// addresses is a word since item 27, which the library's own tests
/// `air::canon::devirt_tests::a_pointer_chosen_at_run_time_calls_the_function_it_holds` and the CPU
/// interpreter's `tests::engine_chosen_pointers` pin.
#[test]
fn the_function_pointer_rows_are_the_table_array() {
    let mut refused_somewhere = false;
    let mut exact_somewhere = false;
    for (stem, desc) in committed_descriptions() {
        let all = all_audits(&stem, &desc);
        let (enabled, _) = created(&stem, &desc);
        let missing: Vec<Capability> = [
            VkFeature::RuntimeDescriptorArray,
            VkFeature::DescriptorBindingPartiallyBound,
            VkFeature::ShaderStorageBufferArrayNonUniformIndexing,
            VkFeature::DescriptorBindingStorageBufferUpdateAfterBind,
        ]
        .into_iter()
        .filter(|&x| !enabled.has(x))
        .map(Capability::Feature)
        .collect();
        let row = if missing.is_empty() {
            exact_somewhere = true;
            RowState::Exact
        } else {
            refused_somewhere = true;
            RowState::Refused(Refusal::Unsupported(Unsupported::needs_any_of(
                Feature::DynamicCall,
                vec![missing],
            )))
        };
        // The render row is Apple6's only.
        for (family, r) in [
            (GpuFamily::Apple6, FeatureRow::FunctionPointersCompute),
            (GpuFamily::Metal3, FeatureRow::FunctionPointersCompute),
            (GpuFamily::Apple6, FeatureRow::FunctionPointersRender),
        ] {
            assert_eq!(
                state(&all[&family], feature(r)),
                Some(row.clone()),
                "{stem} {family:?} {r:?}"
            );
        }
    }
    assert!(
        refused_somewhere,
        "no committed profile lacks the table array"
    );
    assert!(exact_somewhere, "no committed profile has the table array");
}

/// Tile dispatches run: the tile rows are what the compile layer and the binder refuse of Metal's
/// tile shading on every device, and a pass's tile memory is Metal's, but at 8 samples, which
/// Apple's measured GPUs lack.
#[test]
fn tile_rows_are_the_compile_layers_refusal_and_the_pass_metals() {
    let tile_shading = not_implemented(Feature::TileShading);
    for (stem, desc) in committed_descriptions() {
        let all = all_audits(&stem, &desc);
        for row in [
            FeatureRow::TileShaders,
            FeatureRow::ImageblockSampleCoverageControl,
            FeatureRow::NonsquareTileDispatch,
        ] {
            assert_eq!(
                state(&all[&GpuFamily::Apple9], feature(row)),
                Some(tile_shading.clone()),
                "{stem} {row:?}"
            );
        }
        for row in [
            LimitRow::TileSizeWithoutMsaa,
            LimitRow::TileSize2xMsaa,
            LimitRow::TileSize4xMsaa,
            LimitRow::ExplicitImageblockSizePerSample,
            LimitRow::ImplicitImageblockMemory,
        ] {
            assert_eq!(
                state(&all[&GpuFamily::Apple9], FamilyRow::Limit(row)),
                Some(RowState::Exact),
                "{stem} {row:?}"
            );
        }
        assert_eq!(
            state(
                &all[&GpuFamily::Apple10],
                FamilyRow::Limit(LimitRow::TileSize8xMsaa)
            ),
            Some(RowState::Gap(RuntimeGap::TileAtSampleCount)),
            "{stem}"
        );
        assert_eq!(
            state(
                &all[&GpuFamily::Apple9],
                FamilyRow::Limit(LimitRow::TileSize8xMsaa)
            ),
            None,
            "{stem}"
        );
    }
}

/// Kernel and fragment imageblocks run, so the imageblock row names what a fragment function's
/// imageblock needs: the device's pixel interlock and fragment stores first, then a pass whose
/// draws view imageblocks of different masters.
#[test]
fn the_imageblock_row_waits_on_fragment_imageblocks() {
    for (stem, desc) in committed_descriptions() {
        let all = all_audits(&stem, &desc);
        let (enabled, _) = created(&stem, &desc);
        let row = state(&all[&GpuFamily::Apple4], feature(FeatureRow::Imageblocks)).unwrap();
        let missing: Vec<Capability> = [
            VkFeature::FragmentShaderPixelInterlock,
            VkFeature::FragmentStoresAndAtomics,
        ]
        .into_iter()
        .filter(|&x| !enabled.has(x))
        .map(Capability::Feature)
        .collect();
        if missing.is_empty() {
            assert_eq!(row, RowState::Gap(RuntimeGap::ImageblockMasters), "{stem}");
        } else {
            assert_eq!(
                row,
                RowState::Refused(Refusal::Unsupported(Unsupported::needs_any_of(
                    Feature::Imageblocks,
                    vec![missing]
                ))),
                "{stem}"
            );
        }
    }
}

/// The runtime draws `[[stencil]]` output without `VK_EXT_shader_stencil_export` through the
/// shader-side stencil, but not a state whose stencil write the depth test decides, which needs the
/// extension: the stencil feedback row is the extension on every committed profile, never a runtime
/// gap.
#[test]
fn stencil_feedback_is_the_stencil_export_extension() {
    let needs = refused(
        Feature::StencilExport,
        Capability::Extension(VkExtension::ExtShaderStencilExport),
    );
    for (stem, desc) in committed_descriptions() {
        let all = all_audits(&stem, &desc);
        let (enabled, _) = created(&stem, &desc);
        let row = state(
            &all[&GpuFamily::Apple5],
            feature(FeatureRow::StencilFeedback),
        )
        .unwrap();
        if enabled
            .extensions
            .contains(&VkExtension::ExtShaderStencilExport)
        {
            assert_eq!(row, RowState::Exact, "{stem}");
        } else {
            assert_eq!(row, needs, "{stem}");
        }
    }
}

/// Indirect command buffers: the compute rows are exact on a device with device addresses, through
/// which a kernel writes commands, and refused naming them on one without; the render rows wait,
/// past that, for what render commands still lack (`IndirectRenderCommands`). The length row,
/// unbounded, is exact.
#[test]
fn indirect_command_buffer_rows_name_what_is_still_missing() {
    use FeatureRow as F;
    let mut refused_somewhere = false;
    for (stem, desc) in committed_descriptions() {
        let all = all_audits(&stem, &desc);
        let (enabled, _) = created(&stem, &desc);
        let compile = Device::new(enabled, DevicePolicy::UNTRUSTED).unwrap();
        let needs = compile.profile().addressed_needs();
        let (compute, render) = if needs.is_empty() {
            (
                None,
                Some(RowState::Gap(RuntimeGap::IndirectRenderCommands)),
            )
        } else {
            refused_somewhere = true;
            let refusal = RowState::Refused(Refusal::Unsupported(Unsupported::needs_any_of(
                Feature::GpuEncodedIndirectCommands,
                vec![needs],
            )));
            (Some(refusal.clone()), Some(refusal))
        };
        for family in [GpuFamily::Apple3, GpuFamily::Mac2] {
            let rows = &all[&family];
            let blocking = |row: FeatureRow| state(rows, feature(row)).filter(RowState::blocks);
            assert_eq!(blocking(F::IcbCompute), compute, "{stem} {family:?}");
            assert_eq!(
                blocking(F::IcbComputeMemoryBarriers),
                compute,
                "{stem} {family:?}"
            );
            assert_eq!(blocking(F::IcbRendering), render, "{stem} {family:?}");
        }
        assert_eq!(
            state(
                &all[&GpuFamily::Apple3],
                FamilyRow::Limit(LimitRow::IndirectCommandBufferLength)
            ),
            Some(RowState::Exact),
            "{stem}"
        );
    }
    assert!(
        refused_somewhere,
        "every committed profile has device addresses"
    );
}

/// Rows built since the audit began are exact or the compile layer's: the render encoder's
/// memory barriers and per-bind attribute strides are exact on every device, and tessellation
/// is the compile layer's refusal (a patch of no control-point count, and with the device's
/// tessellation stages a fragment function's `[[primitive_id]]` behind one) on every device,
/// with `tessellationShader` or emulated without it, as is the render function pointer row the
/// compute one's.
#[test]
fn rows_built_since_are_exact_or_the_compile_layer_s() {
    use FeatureRow as F;
    for (stem, desc) in committed_descriptions() {
        let all = all_audits(&stem, &desc);
        let apple6 = &all[&GpuFamily::Apple6];
        for row in [F::MemoryBarriers, F::DynamicAttributeStride] {
            assert_eq!(
                state(apple6, feature(row)),
                Some(RowState::Exact),
                "{stem} {row:?}"
            );
        }
        let tessellation = not_implemented(Feature::Tessellation);
        for row in [F::Tessellation, F::IndirectTessellationArguments] {
            assert_eq!(
                state(apple6, feature(row)),
                Some(tessellation.clone()),
                "{stem} {row:?}"
            );
        }
        assert_eq!(
            state(apple6, feature(F::FunctionPointersRender)),
            state(apple6, feature(F::FunctionPointersCompute)),
            "{stem}"
        );
    }
}

/// A format Metal makes no multisample texture of answers neither MSAA nor resolve on any
/// committed profile, as `new_texture` refuses the texture the probe stands for: the two-plane
/// extended-range formats (Metal's validation stops one) and the two-plane YCbCr ones. A one-plane
/// extended-range format answers MSAA as its held format does, and a resolve where it answers
/// MSAA, which the crate-owned resolve makes (`color_resolve.rs`). The stencil view `X32Stencil8` answers neither a filter nor a resolve, as
/// Apple's tables list neither. Nor do the padded 10-bit formats, whose multisample texture Metal's
/// validation stops.
#[test]
fn formats_without_a_made_multisample_texture_or_a_modelled_resolve_answer_neither() {
    let none = [
        PixelFormat::RGB10A82PXR10,
        PixelFormat::RGB10A82PXR10SRGB,
        PixelFormat::YCBCR84202P,
        PixelFormat::YCBCR104202P,
        PixelFormat::RGB84202P,
        PixelFormat::RGB104202P,
        PixelFormat::YCBCR84202PSRGB,
        PixelFormat::YCBCR104202PSRGB,
        PixelFormat::YCBCR84222P,
        PixelFormat::YCBCR84442P,
        PixelFormat::YCBCR104222P,
        PixelFormat::YCBCR104442P,
        PixelFormat::YCBCR84222PSRGB,
        PixelFormat::YCBCR84442PSRGB,
        PixelFormat::YCBCR104222PSRGB,
        PixelFormat::YCBCR104442PSRGB,
        PixelFormat::RGB84222P,
        PixelFormat::RGB84442P,
        PixelFormat::RGB104222P,
        PixelFormat::RGB104442P,
        PixelFormat::YCBCR104202PPACKED,
        PixelFormat::YCBCR104222PPACKED,
        PixelFormat::YCBCR104442PPACKED,
        PixelFormat::YCBCR104202PPACKEDSRGB,
        PixelFormat::YCBCR104222PPACKEDSRGB,
        PixelFormat::YCBCR104442PPACKEDSRGB,
        PixelFormat::R10UnormX6,
        PixelFormat::R10UnormX6SRGB,
        PixelFormat::RG10UnormX12,
        PixelFormat::RG10UnormX12SRGB,
    ];
    for (stem, desc) in committed_descriptions() {
        let (enabled, _) = created(&stem, &desc);
        let compile = Device::new(enabled.clone(), DevicePolicy::UNTRUSTED).unwrap();
        let image = ImageFacts {
            profile: compile.profile(),
            formats: &enabled.formats,
            storage_multisample: enabled.has(VkFeature::ShaderStorageImageMultisample),
            depth_images: crate::image_plan::DepthImageFacts::of(&enabled.driver_facts()),
            linear_attachments: false,
            multisample_depth_twin: false,
        };
        let found = |pf| {
            let sparse = Err(Missing::Gap(RuntimeGap::SparseHeaps));
            answers(pf, &image, &any_limits, sparse).capabilities()
        };
        for pf in none {
            let capabilities = found(pf);
            for capability in [FormatCapability::Msaa, FormatCapability::Resolve] {
                assert!(
                    !capabilities.contains(capability),
                    "{stem}: {pf:?} answers {capability:?}"
                );
            }
        }
        let one_plane = found(PixelFormat::BGRA10XR);
        let msaa = found(PixelFormat::RGBA16Uint).contains(FormatCapability::Msaa);
        assert_eq!(
            one_plane.contains(FormatCapability::Msaa),
            msaa,
            "{stem}: BGRA10_XR multisamples as its raw format does"
        );
        for pf in [
            PixelFormat::BGRA10XR,
            PixelFormat::BGRA10XRSRGB,
            PixelFormat::BGR10XR,
            PixelFormat::BGR10XRSRGB,
        ] {
            let capabilities = found(pf);
            assert_eq!(
                capabilities.contains(FormatCapability::Resolve),
                capabilities.contains(FormatCapability::Msaa),
                "{stem}: {pf:?} resolves where it multisamples"
            );
        }
        // The stencil view of the combined format: MSAA as the combined format, which it
        // views, but neither the filter its image's depth aspect has nor a resolve.
        let view = found(PixelFormat::X32Stencil8);
        let combined = found(PixelFormat::Depth32FloatStencil8);
        for capability in [FormatCapability::Filter, FormatCapability::Resolve] {
            assert!(
                !view.contains(capability),
                "{stem}: X32_Stencil8 answers {capability:?}"
            );
        }
        assert_eq!(
            view.contains(FormatCapability::Msaa),
            combined.contains(FormatCapability::Msaa),
            "{stem}: X32_Stencil8 multisamples as the format it views"
        );
        assert_eq!(
            combined.contains(FormatCapability::Resolve),
            combined.contains(FormatCapability::Msaa),
            "{stem}: Depth32Float_Stencil8 resolves where it multisamples"
        );
    }
}

/// The extended-range formats answer Filter wherever their `ShaderRead` texture is made (the
/// shader filters their raw texels' fetches on every device) and Blend exactly where the compile
/// layer's blend in the shader holds (`extended_range_blend`): every device that orders fetch
/// through the pixel interlock alone, which the families' rows of Apple's `BGRA10_XR` and
/// `BGR10_XR` then answer exact, and lavapipe, which blends over the attachment's input attachment.
#[test]
fn extended_range_formats_answer_the_shaders_filter_and_blend() {
    use metal2vulkan::metal::format::extended_range_blend;
    let formats = [
        PixelFormat::BGR10XR,
        PixelFormat::BGR10XRSRGB,
        PixelFormat::BGRA10XR,
        PixelFormat::BGRA10XRSRGB,
        PixelFormat::RGB10A82PXR10,
        PixelFormat::RGB10A82PXR10SRGB,
    ];
    let mut filtered = Vec::new();
    let mut blended = Vec::new();
    for (stem, desc) in committed_descriptions() {
        let (enabled, _) = created(&stem, &desc);
        let compile = Device::new(enabled.clone(), DevicePolicy::UNTRUSTED).unwrap();
        let image = ImageFacts {
            profile: compile.profile(),
            formats: &enabled.formats,
            storage_multisample: enabled.has(VkFeature::ShaderStorageImageMultisample),
            depth_images: crate::image_plan::DepthImageFacts::of(&enabled.driver_facts()),
            linear_attachments: false,
            multisample_depth_twin: false,
        };
        for pf in formats {
            let sparse = Err(Missing::Gap(RuntimeGap::SparseHeaps));
            let caps = answers(pf, &image, &any_limits, sparse).capabilities();
            let blends = extended_range_blend(pf, compile.profile()).is_ok();
            assert_eq!(
                caps.contains(FormatCapability::Blend),
                blends && caps.contains(FormatCapability::Color),
                "{stem}: {pf:?}"
            );
            if caps.contains(FormatCapability::Filter) {
                filtered.push((stem.clone(), pf));
            }
            if caps.contains(FormatCapability::Blend) {
                blended.push((stem.clone(), pf));
            }
        }
    }
    for stem in [
        "moltenvk-m3",
        "moltenvk-m4",
        "amd-rdna3-radv",
        "intel-arl-anv",
        "nvidia-rtx5080-laptop",
        "lavapipe",
    ] {
        for pf in formats {
            let at = (stem.to_owned(), pf);
            assert!(filtered.contains(&at), "{stem}: {pf:?} answers no Filter");
            assert!(blended.contains(&at), "{stem}: {pf:?}");
        }
    }
    let rows = audits("moltenvk-m4");
    for row in [FormatRow::BGRA10XR, FormatRow::BGR10XR] {
        for capability in [FormatCapability::Filter, FormatCapability::Blend] {
            let at = FamilyRow::Format(row, capability);
            assert_eq!(
                state(&rows[&GpuFamily::Apple9], at),
                Some(RowState::Exact),
                "{row:?} {capability:?}"
            );
        }
    }
}

/// An 8-bit sRGB format a device holds as its UNORM twin, unable to sample the sRGB format
/// (`R8Unorm_sRGB` on lavapipe, `RG8Unorm_sRGB` on lavapipe, ANV and NVIDIA), answers Filter where
/// the twin filters linearly: the device filters the twin's codes, which the function decodes
/// through Metal's table (`PrecisionNote::SrgbTexelDecode`), as every 8-bit sRGB format it samples
/// itself answers its own filter.
#[test]
fn srgb_formats_held_as_their_unorm_twin_answer_the_twins_filter() {
    use metal2vulkan::device::FormatUsage;
    use metal2vulkan::metal::format::{srgb_raw_twin, vk_format};
    let mut twins = Vec::new();
    for (stem, desc) in committed_descriptions() {
        let (enabled, _) = created(&stem, &desc);
        let compile = Device::new(enabled.clone(), DevicePolicy::UNTRUSTED).unwrap();
        let profile = compile.profile();
        let image = ImageFacts {
            profile,
            formats: &enabled.formats,
            storage_multisample: enabled.has(VkFeature::ShaderStorageImageMultisample),
            depth_images: crate::image_plan::DepthImageFacts::of(&enabled.driver_facts()),
            linear_attachments: false,
            multisample_depth_twin: false,
        };
        for pf in [
            PixelFormat::R8UnormSRGB,
            PixelFormat::RG8UnormSRGB,
            PixelFormat::RGBA8UnormSRGB,
            PixelFormat::BGRA8UnormSRGB,
        ] {
            let sampled = [FormatUsage::SampledImage].into();
            if vk_format(pf, &sampled, profile).is_ok() {
                continue;
            }
            let Some(twin) = srgb_raw_twin(pf).and_then(|t| vk_format(t, &sampled, profile).ok())
            else {
                continue;
            };
            let sparse = Err(Missing::Gap(RuntimeGap::SparseHeaps));
            let caps = answers(pf, &image, &any_limits, sparse).capabilities();
            let filters =
                profile.supports_format(twin.format, FormatUsage::SampledImageFilterLinear);
            assert_eq!(
                caps.contains(FormatCapability::Filter),
                filters,
                "{stem}: {pf:?}"
            );
            twins.push(format!("{stem} {pf:?}"));
        }
    }
    for want in [
        "lavapipe R8UnormSRGB",
        "lavapipe RG8UnormSRGB",
        "nvidia-rtx5080-laptop RG8UnormSRGB",
        "intel-arl-anv RG8UnormSRGB",
    ] {
        assert!(twins.iter().any(|t| t == want), "{want}: {twins:?}");
    }
}

/// No answer past the Apple GPU's the runtime presents (Apple9): a format Apple's tables list
/// answers at most its Apple9 cell on every committed profile, sparse textures taken as made
/// (which a profile cannot say), so the M3 Max's measured over-answers, Sparse for `GBGR422`,
/// `BGRG422` and `X32_Stencil8`, are gone. No family's column lists a format capability Apple9's
/// does not, so the cap changes no family's row.
#[test]
fn listed_formats_answer_at_most_their_apple9_cell() {
    let all = audits("moltenvk-m4");
    let format_rows = |family: GpuFamily| -> Vec<(FormatRow, FormatCapability)> {
        all[&family]
            .iter()
            .filter_map(|r| match r.row {
                FamilyRow::Format(row, capability) => Some((row, capability)),
                _ => None,
            })
            .collect()
    };
    let apple9 = format_rows(GpuFamily::Apple9);
    for family in GpuFamily::ALL {
        for at in format_rows(family) {
            assert!(apple9.contains(&at), "{family:?} lists {at:?}");
        }
    }
    for (stem, desc) in committed_descriptions() {
        let (enabled, _) = created(&stem, &desc);
        let compile = Device::new(enabled.clone(), DevicePolicy::UNTRUSTED).unwrap();
        let image = ImageFacts {
            profile: compile.profile(),
            formats: &enabled.formats,
            storage_multisample: enabled.has(VkFeature::ShaderStorageImageMultisample),
            depth_images: crate::image_plan::DepthImageFacts::of(&enabled.driver_facts()),
            linear_attachments: false,
            multisample_depth_twin: false,
        };
        for row in FormatRow::ALL {
            for &pf in row.formats() {
                let caps = answers(pf, &image, &any_limits, Ok(())).capabilities();
                for capability in caps.iter() {
                    assert!(
                        apple9.contains(&(row, capability)),
                        "{stem}: {pf:?} answers {capability:?}"
                    );
                }
            }
        }
        for pf in [
            PixelFormat::GBGR422,
            PixelFormat::BGRG422,
            PixelFormat::X32Stencil8,
        ] {
            let caps = answers(pf, &image, &any_limits, Ok(())).capabilities();
            assert!(!caps.contains(FormatCapability::Sparse), "{stem}: {pf:?}");
        }
    }
}
