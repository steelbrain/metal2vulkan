//! Declared image types, binding compatibility, and views.

use super::*;
use crate::bind::state::BoundSampler;
use crate::error::Needs;
use crate::metal::format::storage_choice;
use crate::pipeline::sym::SampledType;
use std::collections::BTreeSet;

fn slot_iface(texture_type: TextureType, sampled_type: SampledType) -> TextureSlot {
    TextureSlot {
        texture_type,
        sampled_type,
        depth_compare: false,
        access: TextureAccess::new(true, false, false),
    }
}

/// Every Metal texture type declares the dimension, arrayed-ness, and sample count of its SPIR-V
/// image, a cube with `ImageViewShape` declares the 2D array, and every declared type has a
/// neutral whose view type is the declaration's.
#[test]
fn declared_types_follow_the_texture_type_and_the_view_shape() {
    use TextureType as T;
    let sampled = vk::DescriptorType::SAMPLED_IMAGE;
    let cases = [
        (T::Type1D, false, vk::ImageViewType::TYPE_1D),
        (T::Type1DArray, false, vk::ImageViewType::TYPE_1D_ARRAY),
        (T::Type2D, false, vk::ImageViewType::TYPE_2D),
        (T::Type2DArray, false, vk::ImageViewType::TYPE_2D_ARRAY),
        (T::Type2DMultisample, false, vk::ImageViewType::TYPE_2D),
        (
            T::Type2DMultisampleArray,
            false,
            vk::ImageViewType::TYPE_2D_ARRAY,
        ),
        (T::Cube, false, vk::ImageViewType::CUBE),
        (T::CubeArray, false, vk::ImageViewType::CUBE_ARRAY),
        (T::Cube, true, vk::ImageViewType::TYPE_2D_ARRAY),
        (T::CubeArray, true, vk::ImageViewType::TYPE_2D_ARRAY),
        (T::Type3D, false, vk::ImageViewType::TYPE_3D),
    ];
    for (ty, as_array, view) in cases {
        let t = slot_iface(ty, SampledType::Float);
        let class =
            declared(&t, sampled, as_array, false, StorageFormatClass::Unknown).expect("declared");
        assert_eq!(view_type(&class), Ok(view), "{ty:?}");
        assert_eq!(class.format, StorageFormatClass::NotStorage);
        assert_eq!(
            class.multisampled,
            matches!(ty, T::Type2DMultisample | T::Type2DMultisampleArray)
        );
        match neutral(&class).expect("neutral") {
            NeutralResource::Image { view_type, .. } => assert_eq!(view_type, view, "{ty:?}"),
            other => panic!("{ty:?}: {other:?}"),
        }
    }
    // A texture buffer is exactly a texel-buffer binding.
    let tb = slot_iface(T::TextureBuffer, SampledType::Uint);
    let class = declared(
        &tb,
        vk::DescriptorType::STORAGE_TEXEL_BUFFER,
        false,
        false,
        StorageFormatClass::Unknown,
    )
    .expect("texel buffer");
    assert_eq!(
        neutral(&class),
        Ok(NeutralResource::TexelBuffer {
            format: vk::Format::R32_UINT
        })
    );
    assert!(declared(&tb, sampled, false, false, StorageFormatClass::Unknown).is_err());
    assert!(declared(
        &slot_iface(T::Type2D, SampledType::Float),
        vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
        false,
        false,
        StorageFormatClass::Unknown
    )
    .is_err());
}

/// A bound texture is compatible exactly when its Metal type, sample count, and numeric type are
/// the declared ones (and it has a depth aspect under comparison).
#[test]
fn compatibility_is_type_samples_numeric_type_and_depth() {
    let t = slot_iface(TextureType::Type2D, SampledType::Float);
    let sampled = vk::DescriptorType::SAMPLED_IMAGE;
    let class = declared(&t, sampled, false, false, StorageFormatClass::NotStorage).expect("2d");
    let tex = |texture_type, pixel_format, sample_count| TextureShape::Texture {
        address_modes: crate::metal::sampler::AddressModeMap::IDENTITY,
        offset_base_axes: 0,
        ycbcr_in_place: false,
        ycbcr_planes_sampler: None,
        texture_type,
        pixel_format,
        sample_count,
        usage: crate::metal::TextureUsage::SHADER_READ,
        mipmap_level_count: 1,
        sparse: false,
        swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
    };
    assert!(compatible(
        &tex(TextureType::Type2D, PixelFormat::RGBA8Unorm, 1),
        &t,
        &class
    ));
    assert!(compatible(
        &tex(TextureType::Type2D, PixelFormat::Depth32Float, 1),
        &t,
        &class
    ));
    assert!(!compatible(
        &tex(TextureType::Type2DArray, PixelFormat::RGBA8Unorm, 1),
        &t,
        &class
    ));
    assert!(!compatible(
        &tex(TextureType::Type2D, PixelFormat::RGBA8Uint, 1),
        &t,
        &class
    ));
    assert!(!compatible(
        &TextureShape::Buffer {
            pixel_format: PixelFormat::RGBA8Unorm,
            swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
        },
        &t,
        &class
    ));
    let depth = declared(&t, sampled, false, true, StorageFormatClass::NotStorage).expect("d");
    assert!(!compatible(
        &tex(TextureType::Type2D, PixelFormat::RGBA8Unorm, 1),
        &t,
        &depth
    ));
    assert!(compatible(
        &tex(TextureType::Type2D, PixelFormat::Depth32Float, 1),
        &t,
        &depth
    ));
    let ms = slot_iface(TextureType::Type2DMultisample, SampledType::Sint);
    let msc = declared(&ms, sampled, false, false, StorageFormatClass::NotStorage).expect("ms");
    assert!(compatible(
        &tex(TextureType::Type2DMultisample, PixelFormat::RGBA16Sint, 4),
        &ms,
        &msc
    ));
    assert!(!compatible(
        &tex(TextureType::Type2DMultisample, PixelFormat::RGBA16Sint, 1),
        &ms,
        &msc
    ));
}

/// A `ulong` slot binds only an `RG32Uint` texture, the one format Metal's API validation lets a
/// `ulong` texture parameter take (measured on Apple M3 Max), as a texture and as a texture buffer;
/// unbound, its neutral is an `RG32_UINT` image or texel buffer, the format its declaration names.
#[test]
fn a_ulong_slot_binds_only_rg32uint_textures() {
    let tex = |pixel_format| TextureShape::Texture {
        address_modes: crate::metal::sampler::AddressModeMap::IDENTITY,
        offset_base_axes: 0,
        ycbcr_in_place: false,
        ycbcr_planes_sampler: None,
        texture_type: TextureType::Type2D,
        pixel_format,
        sample_count: 1,
        usage: crate::metal::TextureUsage::SHADER_READ,
        mipmap_level_count: 1,
        sparse: false,
        swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
    };
    let t = slot_iface(TextureType::Type2D, SampledType::Uint64);
    for (ty, format) in [
        (
            vk::DescriptorType::SAMPLED_IMAGE,
            StorageFormatClass::NotStorage,
        ),
        (
            vk::DescriptorType::STORAGE_IMAGE,
            StorageFormatClass::Unknown,
        ),
    ] {
        let class = declared(&t, ty, false, false, format).expect("2d");
        assert!(compatible(&tex(PixelFormat::RG32Uint), &t, &class));
        for other in [
            PixelFormat::RG32Sint,
            PixelFormat::RG32Float,
            PixelFormat::RGBA16Uint,
            PixelFormat::RGBA16Sint,
            PixelFormat::R32Uint,
            PixelFormat::RGBA32Uint,
        ] {
            assert!(!compatible(&tex(other), &t, &class), "{other:?}");
        }
        match neutral(&class).expect("neutral") {
            n @ NeutralResource::Image { .. } => {
                assert_eq!(n.format(), Some(vk::Format::R32G32_UINT), "{ty:?}")
            }
            other => panic!("{other:?}"),
        }
    }
    let tb = slot_iface(TextureType::TextureBuffer, SampledType::Uint64);
    let class = declared(
        &tb,
        vk::DescriptorType::STORAGE_TEXEL_BUFFER,
        false,
        false,
        StorageFormatClass::Unknown,
    )
    .expect("texel buffer");
    let buffer = |pixel_format| TextureShape::Buffer {
        pixel_format,
        swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
    };
    assert!(compatible(&buffer(PixelFormat::RG32Uint), &tb, &class));
    assert!(!compatible(&buffer(PixelFormat::RGBA16Uint), &tb, &class));
    assert_eq!(
        neutral(&class),
        Ok(NeutralResource::TexelBuffer {
            format: vk::Format::R32G32_UINT
        })
    );
}

/// A slot whose integer texels take their signedness from the bound format
/// (`StorageTextureNumericType`) binds a texture of either integer type, and the bound format's
/// choice keys the variant: its interface's type is the format's its compile was given, a runtime's
/// stand-in (`R32Uint` here) where the dispatch binds `RGBA16Sint`. A float format still does not
/// bind it, and without the fact the interface's type is the function's own.
#[test]
fn a_numeric_slot_binds_either_integer_type() {
    let texture = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 1,
    };
    let p =
        crate::pipeline::testing::profile("moltenvk-m3", crate::device::DevicePolicy::default())
            .expect("profile");
    let slot = TextureSlot {
        access: TextureAccess::new(false, false, true),
        ..slot_iface(TextureType::Type2D, SampledType::Uint)
    };
    let slots = BTreeMap::from([(texture, SlotInterface::Texture(slot))]);
    let numeric = [RuntimeDependency {
        fact: RuntimeFact::StorageTextureNumericType { slot: texture },
        mechanism: crate::pipeline::facts::Mechanism::Variant,
    }];
    let plan = |pixel_format, dependencies: &[RuntimeDependency]| {
        let mut stage = StageShape::default();
        stage.textures.insert(
            texture.index,
            TextureShape::Texture {
                address_modes: crate::metal::sampler::AddressModeMap::IDENTITY,
                offset_base_axes: 0,
                ycbcr_in_place: false,
                ycbcr_planes_sampler: None,
                texture_type: TextureType::Type2D,
                pixel_format,
                sample_count: 1,
                usage: crate::metal::TextureUsage::SHADER_WRITE,
                mipmap_level_count: 1,
                sparse: false,
                swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
            },
        );
        let shape = BTreeMap::from([(Stage::Kernel, stage)]);
        let mut planner = ResourcePlanner {
            profile: &p,
            shape: &shape,
            slots: &slots,
            dependencies,
            stage_flags: |_| vk::PipelineStageFlags2::COMPUTE_SHADER,
            variant: VariantKey::new(),
            diagnostics: Vec::new(),
            uses: Vec::new(),
            template: Default::default(),
            alpha_red_texels: Default::default(),
        };
        let (write, _) = planner
            .plain_texture(0, 1, vk::DescriptorType::STORAGE_IMAGE, texture)
            .expect("plan");
        let bound = matches!(write, WriteSource::Texture { .. });
        (bound, planner.variant.storage_images.get(&texture).copied())
    };
    let choice = |pixel_format| {
        storage_choice(pixel_format, StorageUse::texture(false, true), &p).expect("choice")
    };
    for pf in [
        PixelFormat::RGBA16Sint,
        PixelFormat::R32Uint,
        PixelFormat::RG32Sint,
    ] {
        assert_eq!(plan(pf, &numeric), (true, Some(choice(pf))), "{pf:?}");
    }
    assert!(!plan(PixelFormat::RGBA32Float, &numeric).0);
    assert!(!plan(PixelFormat::RGBA16Sint, &[]).0);
    assert!(plan(PixelFormat::RGBA16Uint, &[]).0);
}

/// A texture of a format held for copies alone (`metal::format::copies_only`), which Metal reads as
/// floats, bound at a `float` slot that samples, reads, writes or reads and writes it, is the
/// format's own refusal, `NotImplemented(Format)`: its emulation is admitted for no usage, so no
/// view serves the slot. No neutral stands in for it, as one did while the binder took it for a
/// mismatch, so a kernel's read completed and read the neutral's zeros. At a `uint` slot it is a
/// mismatch, whose neutral the binder writes and reports.
#[test]
fn a_texture_held_for_copies_alone_is_refused_at_a_float_slot() {
    let texture = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 0,
    };
    let p =
        crate::pipeline::testing::profile("moltenvk-m3", crate::device::DevicePolicy::default())
            .expect("profile");
    let plan = |pixel_format, access: TextureAccess, sampled_type, ty| {
        let slot = TextureSlot {
            access,
            ..slot_iface(TextureType::Type2D, sampled_type)
        };
        let slots = BTreeMap::from([(texture, SlotInterface::Texture(slot))]);
        let mut stage = StageShape::default();
        stage.textures.insert(
            texture.index,
            TextureShape::Texture {
                address_modes: crate::metal::sampler::AddressModeMap::IDENTITY,
                offset_base_axes: 0,
                ycbcr_in_place: false,
                ycbcr_planes_sampler: None,
                texture_type: TextureType::Type2D,
                pixel_format,
                sample_count: 1,
                usage: crate::metal::TextureUsage::SHADER_READ
                    | crate::metal::TextureUsage::SHADER_WRITE,
                mipmap_level_count: 1,
                sparse: false,
                swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
            },
        );
        let shape = BTreeMap::from([(Stage::Kernel, stage)]);
        let mut planner = ResourcePlanner {
            profile: &p,
            shape: &shape,
            slots: &slots,
            dependencies: &[],
            stage_flags: |_| vk::PipelineStageFlags2::COMPUTE_SHADER,
            variant: VariantKey::new(),
            diagnostics: Vec::new(),
            uses: Vec::new(),
            template: Default::default(),
            alpha_red_texels: Default::default(),
        };
        let planned = planner
            .plain_texture(0, 0, ty, texture)
            .map(|(write, _)| write);
        (planned, planner.diagnostics)
    };
    let sampled = vk::DescriptorType::SAMPLED_IMAGE;
    let storage = vk::DescriptorType::STORAGE_IMAGE;
    let uses = [
        ("sampled", TextureAccess::new(true, false, false), sampled),
        ("read", TextureAccess::new(false, true, false), sampled),
        ("written", TextureAccess::new(false, false, true), storage),
        (
            "read and written",
            TextureAccess::new(false, true, true),
            storage,
        ),
    ];
    let held: Vec<PixelFormat> = (PixelFormat::ALL.iter().copied())
        .filter(|&pf| copies_only(pf))
        .collect();
    assert_eq!(held.len(), 20, "{held:?}");
    for pf in held {
        for (name, access, ty) in uses {
            match plan(pf, access, SampledType::Float, ty) {
                (Err(BindError::Unsupported(u)), diagnostics) => {
                    assert_eq!(
                        (u.feature, u.needs),
                        (Feature::Format, Needs::NotImplemented),
                        "{pf:?} {name}"
                    );
                    assert!(diagnostics.is_empty(), "{pf:?} {name}: {diagnostics:?}");
                }
                other => panic!("{pf:?} {name}: {other:?}"),
            }
        }
        let (mismatch, diagnostics) = plan(
            pf,
            TextureAccess::new(false, true, false),
            SampledType::Uint,
            sampled,
        );
        assert!(
            matches!(mismatch, Ok(WriteSource::Neutral(_))),
            "{pf:?} at a uint slot: {mismatch:?}"
        );
        assert!(
            matches!(
                diagnostics.as_slice(),
                [PlanDiagnostic::IncompatibleBinding { .. }
                    | PlanDiagnostic::CrossClassNeutral { .. }]
            ),
            "{pf:?} at a uint slot: {diagnostics:?}"
        );
    }
}

/// A sampled view selects the depth aspect of a depth (or combined) format, the stencil aspect of
/// a stencil-only one, and color otherwise; the formatless `Identity` choice is the default
/// declaration and stays out of the variant.
#[test]
fn view_aspects_and_the_default_storage_choice() {
    assert_eq!(
        view_aspect(PixelFormat::Depth32FloatStencil8),
        vk::ImageAspectFlags::DEPTH
    );
    assert_eq!(
        view_aspect(PixelFormat::Depth16Unorm),
        vk::ImageAspectFlags::DEPTH
    );
    assert_eq!(
        view_aspect(PixelFormat::Stencil8),
        vk::ImageAspectFlags::STENCIL
    );
    assert_eq!(
        view_aspect(PixelFormat::RGBA8Unorm),
        vk::ImageAspectFlags::COLOR
    );
    let rgba8 = StorageImageChoice::new(vk::Format::R8G8B8A8_UNORM, StorageImageClass::Identity);
    assert_eq!(
        storage_class(Some(&rgba8)),
        StorageFormatClass::Format(vk::Format::R8G8B8A8_UNORM)
    );
    assert_eq!(storage_class(None), StorageFormatClass::Unknown);
    // The default declaration serves an `Identity` choice exactly where the view's format serves
    // the uses without a format: the device's features, and the format's bits for an image or a
    // texel buffer.
    let write = StorageUse::texture(false, true);
    let read_write = StorageUse::texture(true, true);
    let rgba = vk::Format::R8G8B8A8_UNORM;
    let mut alias = rgba8;
    alias.class = StorageImageClass::SwizzleBgra;
    alias.alias = Some(crate::pipeline::lower_inputs::AliasView::new(
        vk::Format::B8G8R8A8_UNORM,
    ));
    let mut texel_variants = BTreeSet::new();
    for (stem, p) in crate::spirv::testing::committed_profiles().expect("profiles") {
        let i = &p.fields().images;
        // `R8G8B8A8_UNORM` is a storage image on every committed profile, and in the
        // specification's list of formats the device features cover.
        assert_eq!(
            needs_variant(&rgba8, rgba, write, &p),
            !i.storage_write_without_format,
            "{stem}"
        );
        assert_eq!(
            needs_variant(&rgba8, rgba, read_write, &p),
            !(i.storage_write_without_format && i.storage_read_without_format),
            "{stem}"
        );
        assert!(needs_variant(&alias, rgba, write, &p), "{stem}");
        // A texel buffer has no implied bit: its buffer features decide.
        let r32 = StorageImageChoice::new(vk::Format::R32_UINT, StorageImageClass::Identity);
        let buffer_write = StorageUse::texel_buffer(false, true);
        let has = i.storage_write_without_format
            && p.supports_format(
                vk::Format::R32_UINT,
                FormatUsage::StorageTexelBufferWriteWithoutFormat,
            );
        assert_eq!(
            needs_variant(&r32, vk::Format::R32_UINT, buffer_write, &p),
            !has,
            "{stem}"
        );
        if !has {
            texel_variants.insert(stem);
        }
    }
    // SwiftShader's `R32_UINT` buffer features lack `STORAGE_WRITE_WITHOUT_FORMAT` though it has
    // `shaderStorageImageWriteWithoutFormat`; lavapipe's state it.
    assert!(texel_variants.contains("swiftshader"), "{texel_variants:?}");
    assert!(!texel_variants.contains("lavapipe"), "{texel_variants:?}");
    // A view format without a SPIR-V image format is declared formatless.
    let bgra = StorageImageChoice::new(vk::Format::B8G8R8A8_UNORM, StorageImageClass::SrgbEncode);
    assert_eq!(storage_class(Some(&bgra)), StorageFormatClass::Unknown);
}

/// Metal reads a depth texel through a colour texture type as `(d, d, d, 1)` (Apple M4:
/// `Depth16Unorm`, `Depth32Float` and `Depth32Float_Stencil8` through `texture2d<float>` and
/// `texture2d<half>`, read, sampled, gathered and sampled with gradients, in a kernel and a
/// fragment function), and a stencil texel through `texture2d<uint>` as `(s, 0, 0, 1)`, which
/// Vulkan's identity reads too. So every depth format's sampled view that no
/// comparison reads replicates the depth with alpha one, on every committed profile where the
/// format is sampled; a comparison's view and every stencil view keep the identity, and the
/// border check reads the depth through the same mapping.
#[test]
fn a_depth_read_as_colour_replicates_its_depth_as_metal_does() {
    let t = TextureSlot {
        depth_compare: true,
        ..slot_iface(TextureType::Type2D, SampledType::Float)
    };
    let class = |compare| {
        declared(
            &t,
            vk::DescriptorType::SAMPLED_IMAGE,
            false,
            compare,
            StorageFormatClass::NotStorage,
        )
        .unwrap()
    };
    let u = slot_iface(TextureType::Type2D, SampledType::Uint);
    let uint = declared(
        &u,
        vk::DescriptorType::SAMPLED_IMAGE,
        false,
        false,
        StorageFormatClass::NotStorage,
    )
    .unwrap();
    let depths = [
        PixelFormat::Depth16Unorm,
        PixelFormat::Depth32Float,
        PixelFormat::Depth24UnormStencil8,
        PixelFormat::Depth32FloatStencil8,
    ];
    let stencils = [
        PixelFormat::Stencil8,
        PixelFormat::X32Stencil8,
        PixelFormat::X24Stencil8,
    ];
    let mut seen = 0;
    for (stem, p) in crate::spirv::testing::committed_profiles().expect("profiles") {
        for pf in depths {
            let Ok(read) = sampled_view(&p, pf, &class(false)) else {
                continue;
            };
            seen += 1;
            assert_eq!(
                (read.aspect, read.components),
                (vk::ImageAspectFlags::DEPTH, DEPTH_AS_COLOR),
                "{stem} {pf:?}"
            );
            let compare = sampled_view(&p, pf, &class(true)).unwrap();
            assert_eq!(
                compare.components,
                vk::ComponentMapping::IDENTITY,
                "{stem} {pf:?}"
            );
        }
        for pf in stencils {
            let Ok(read) = sampled_view(&p, pf, &uint) else {
                continue;
            };
            assert_eq!(
                (read.aspect, read.components),
                (
                    vk::ImageAspectFlags::STENCIL,
                    vk::ComponentMapping::IDENTITY
                ),
                "{stem} {pf:?}"
            );
        }
    }
    assert!(seen > 0, "no profile samples a depth format");
}

/// A bound runtime sampler's values are checked at plan time from its shape, with exactly the
/// verdict `vk_sampler` gives the descriptor for the binding's use: mirror-clamp needs
/// `samplerMirrorClampToEdge`, and NaN or inverted LOD clamps are invalid descriptors; anisotropy is
/// refused nowhere (without `samplerAnisotropy` the Vulkan sampler is isotropic and the module
/// emulates Metal's filter or announces the device's), a min reduction reads the weighted average,
/// refused nowhere, and an axis clamping to zero beside one clamping to a white border reads the one
/// color, refused nowhere. Every refusal kind occurs on some committed profile.
#[test]
fn runtime_sampler_values_are_checked_at_plan_time() {
    use crate::bind::state::SamplerDescriptor;
    use crate::metal::sampler::{vk_sampler, SamplerUse};
    use crate::metal::{SamplerAddressMode, SamplerBorderColor, SamplerReductionMode};
    let slot = SlotRef {
        stage: Stage::Fragment,
        class: SlotClass::Sampler,
        index: 2,
    };
    let base = SamplerDescriptor::new();
    let mut cases = vec![("default", base)];
    let mut d = base;
    d.max_anisotropy = 16;
    cases.push(("anisotropic", d));
    let mut d = base;
    d.t_address_mode = SamplerAddressMode::MirrorClampToEdge;
    cases.push(("mirror clamp", d));
    let mut d = base;
    d.s_address_mode = SamplerAddressMode::ClampToZero;
    d.r_address_mode = SamplerAddressMode::ClampToBorderColor;
    d.border_color = SamplerBorderColor::OpaqueWhite;
    cases.push(("zero beside a border", d));
    let mut d = base;
    d.reduction_mode = SamplerReductionMode::Minimum;
    cases.push(("min", d));
    let mut d = base;
    d.lod_max_clamp = f32::NAN;
    cases.push(("NaN LOD", d));
    let mut d = base;
    d.lod_min_clamp = 3.0;
    d.lod_max_clamp = 2.0;
    cases.push(("inverted LOD", d));
    let mut refused = BTreeMap::new();
    let mut accepted = BTreeMap::new();
    for (stem, p) in crate::spirv::testing::committed_profiles().expect("profiles") {
        for compare in [false, true] {
            let mut slots = BTreeMap::new();
            slots.insert(
                slot,
                SlotInterface::Sampler(SamplerSlot {
                    compare,
                    coordinate_sensitive: false,
                }),
            );
            for (name, desc) in &cases {
                let mut stage = StageShape::default();
                stage
                    .samplers
                    .insert(slot.index, BoundSampler::new(*desc).shape());
                let shape = BTreeMap::from([(Stage::Fragment, stage)]);
                let mut planner = ResourcePlanner {
                    profile: &p,
                    shape: &shape,
                    slots: &slots,
                    dependencies: &[],
                    stage_flags: |_| vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    variant: VariantKey::new(),
                    diagnostics: Vec::new(),
                    uses: Vec::new(),
                    template: Default::default(),
                    alpha_red_texels: Default::default(),
                };
                let planned = planner.sampler(slot);
                match vk_sampler(desc, SamplerUse::new(compare, false), &p) {
                    Ok(_) => {
                        assert_eq!(
                            planned,
                            Ok(WriteSource::Sampler {
                                slot,
                                compare,
                                integer: false,
                                opaque_black: crate::metal::sampler::OpaqueBlackBorder::Stated,
                            }),
                            "{stem} {name} compare={compare}"
                        );
                        *accepted.entry(*name).or_insert(0) += 1;
                    }
                    Err(e) => {
                        assert_eq!(planned, Err(bind_error(e)), "{stem} {name} {compare}");
                        *refused.entry(*name).or_insert(0) += 1;
                    }
                }
            }
            // An unbound runtime sampler is the neutral of the use's comparison.
            let shape = BTreeMap::new();
            let mut planner = ResourcePlanner {
                profile: &p,
                shape: &shape,
                slots: &slots,
                dependencies: &[],
                stage_flags: |_| vk::PipelineStageFlags2::FRAGMENT_SHADER,
                variant: VariantKey::new(),
                diagnostics: Vec::new(),
                uses: Vec::new(),
                template: Default::default(),
                alpha_red_texels: Default::default(),
            };
            assert_eq!(
                planner.sampler(slot),
                Ok(WriteSource::Neutral(NeutralResource::Sampler { compare }))
            );
        }
    }
    assert!(
        refused.contains_key("mirror clamp"),
        "mirror clamp is refused somewhere"
    );
    assert!(
        accepted.contains_key("mirror clamp"),
        "mirror clamp is accepted somewhere"
    );
    for name in ["NaN LOD", "inverted LOD"] {
        assert!(
            !accepted.contains_key(name),
            "{name} is never a Vulkan sampler"
        );
    }
    for name in ["default", "anisotropic", "min", "zero beside a border"] {
        assert!(!refused.contains_key(name), "{name} is refused");
    }
}

/// A layout's immutable (constexpr) sampler reads the bound view as a runtime sampler does. Through
/// a linear sampler a float texture needs its view format's `SAMPLED_IMAGE_FILTER_LINEAR` (refused
/// where a committed profile lacks it for `R32_SFLOAT`), while an integer texture's immutable
/// sampler is nearest (`metal::sampler::vk_sampler`) and needs nothing.
#[test]
fn an_immutable_linear_sampler_needs_a_filterable_view() {
    use crate::bind::state::SamplerDescriptor;
    use crate::metal::sampler::{vk_sampler, SamplerUse};
    use crate::metal::SamplerMinMagFilter;
    let texture = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 0,
    };
    let mut linear = SamplerDescriptor::new();
    linear.min_filter = SamplerMinMagFilter::Linear;
    linear.mag_filter = SamplerMinMagFilter::Linear;
    let (mut refused, mut accepted) = (0, 0);
    for (stem, p) in crate::spirv::testing::committed_profiles().expect("profiles") {
        for (pf, sampled_type) in [
            (PixelFormat::R32Float, SampledType::Float),
            (PixelFormat::R16Uint, SampledType::Uint),
        ] {
            let integer = sampled_type != SampledType::Float;
            let t = slot_iface(TextureType::Type2D, sampled_type);
            let slots = BTreeMap::from([(texture, SlotInterface::Texture(t))]);
            let mut stage = StageShape::default();
            stage.textures.insert(
                texture.index,
                TextureShape::Texture {
                    address_modes: crate::metal::sampler::AddressModeMap::IDENTITY,
                    offset_base_axes: 0,
                    ycbcr_in_place: false,
                    ycbcr_planes_sampler: None,
                    texture_type: TextureType::Type2D,
                    pixel_format: pf,
                    sample_count: 1,
                    usage: crate::metal::TextureUsage::SHADER_READ,
                    mipmap_level_count: 1,
                    sparse: false,
                    swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
                },
            );
            let shape = BTreeMap::from([(Stage::Kernel, stage)]);
            let mut planner = ResourcePlanner {
                profile: &p,
                shape: &shape,
                slots: &slots,
                dependencies: &[],
                stage_flags: |_| vk::PipelineStageFlags2::COMPUTE_SHADER,
                variant: VariantKey::new(),
                diagnostics: Vec::new(),
                uses: Vec::new(),
                template: Default::default(),
                alpha_red_texels: Default::default(),
            };
            let immutable = vk_sampler(&linear, SamplerUse::new(false, integer), &p).expect("maps");
            let sampler = CombinedSampler::new(SamplerRef::Constexpr([0, 0]), false, integer);
            let planned = planner.combined(texture, sampler, Some(&immutable));
            let class = declared(
                &t,
                vk::DescriptorType::SAMPLED_IMAGE,
                false,
                false,
                StorageFormatClass::NotStorage,
            )
            .expect("declared");
            let format = sampled_view(&p, pf, &class).expect("view").format;
            let filterable = p.supports_format(format, FormatUsage::SampledImageFilterLinear);
            if integer || filterable {
                assert!(
                    matches!(planned, Ok(WriteSource::CombinedImageSampler { .. })),
                    "{stem} {pf:?}: {planned:?}"
                );
                accepted += 1;
            } else {
                assert_eq!(
                    planned,
                    Err(BindError::Unsupported(Unsupported::needs(
                        Feature::Format,
                        Capability::Format {
                            format,
                            usage: FormatUsage::SampledImageFilterLinear
                        },
                    ))),
                    "{stem} {pf:?}"
                );
                refused += 1;
            }
        }
    }
    assert!(
        refused > 0 && accepted > 0,
        "{refused} refused, {accepted} accepted"
    );
}

/// A linear comparison sampler (a PCF shadow sampler: `filter::linear`, `compare_func::less`)
/// filters the comparison results, so its depth view needs no `SAMPLED_IMAGE_FILTER_LINEAR`
/// (VUID-vkCmdDispatch-magFilter-04553 and -mipmapMode-04770 exclude `compareEnable`): bound as a
/// layout's immutable sampler to `Depth32Float` or `Depth16Unorm` it is accepted on every
/// committed profile, including those whose depth formats are not linearly filterable
/// (`vk12-floor`). The same sampler without comparison needs the feature there.
#[test]
fn a_linear_compare_sampler_needs_no_filterable_view() {
    use crate::bind::state::SamplerDescriptor;
    use crate::metal::sampler::{vk_sampler, SamplerUse};
    use crate::metal::{CompareFunction, SamplerMinMagFilter, SamplerMipFilter};
    let texture = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 0,
    };
    let mut pcf = SamplerDescriptor::new();
    pcf.min_filter = SamplerMinMagFilter::Linear;
    pcf.mag_filter = SamplerMinMagFilter::Linear;
    pcf.mip_filter = SamplerMipFilter::Linear;
    pcf.compare_function = CompareFunction::Less;
    let (mut unfilterable, mut floor_seen) = (0, false);
    for (stem, p) in crate::spirv::testing::committed_profiles().expect("profiles") {
        for pf in [PixelFormat::Depth32Float, PixelFormat::Depth16Unorm] {
            for compare in [true, false] {
                let t = TextureSlot {
                    depth_compare: compare,
                    ..slot_iface(TextureType::Type2D, SampledType::Float)
                };
                let slots = BTreeMap::from([(texture, SlotInterface::Texture(t))]);
                let mut stage = StageShape::default();
                stage.textures.insert(
                    texture.index,
                    TextureShape::Texture {
                        address_modes: crate::metal::sampler::AddressModeMap::IDENTITY,
                        offset_base_axes: 0,
                        ycbcr_in_place: false,
                        ycbcr_planes_sampler: None,
                        texture_type: TextureType::Type2D,
                        pixel_format: pf,
                        sample_count: 1,
                        usage: crate::metal::TextureUsage::SHADER_READ,
                        mipmap_level_count: 1,
                        sparse: false,
                        swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
                    },
                );
                let shape = BTreeMap::from([(Stage::Kernel, stage)]);
                let mut planner = ResourcePlanner {
                    profile: &p,
                    shape: &shape,
                    slots: &slots,
                    dependencies: &[],
                    stage_flags: |_| vk::PipelineStageFlags2::COMPUTE_SHADER,
                    variant: VariantKey::new(),
                    diagnostics: Vec::new(),
                    uses: Vec::new(),
                    template: Default::default(),
                    alpha_red_texels: Default::default(),
                };
                let immutable =
                    vk_sampler(&pcf, SamplerUse::new(compare, false), &p).expect("maps");
                assert_eq!(immutable.compare.is_some(), compare, "{stem}");
                let sampler = CombinedSampler::new(SamplerRef::Constexpr([0, 0]), compare, false);
                let planned = planner.combined(texture, sampler, Some(&immutable));
                let class = declared(
                    &t,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    false,
                    compare,
                    StorageFormatClass::NotStorage,
                )
                .expect("declared");
                let format = sampled_view(&p, pf, &class).expect("view").format;
                let filterable = p.supports_format(format, FormatUsage::SampledImageFilterLinear);
                if compare || filterable {
                    assert!(
                        matches!(planned, Ok(WriteSource::CombinedImageSampler { .. })),
                        "{stem} {pf:?} compare={compare}: {planned:?}"
                    );
                } else {
                    assert_eq!(
                        planned,
                        Err(BindError::Unsupported(Unsupported::needs(
                            Feature::Format,
                            Capability::Format {
                                format,
                                usage: FormatUsage::SampledImageFilterLinear
                            },
                        ))),
                        "{stem} {pf:?}"
                    );
                }
                if compare && !filterable {
                    unfilterable += 1;
                    floor_seen |= stem == "vk12-floor";
                }
            }
        }
    }
    assert!(
        unfilterable > 0 && floor_seen,
        "no profile lacks linear filtering of a depth format ({unfilterable})"
    );
}

/// A texture slot sampled through Metal's bicubic filter specializes `BicubicTexels` from the bound
/// format's family; an unbound slot or an incompatible binding (an integer format on a float slot)
/// takes the neutral treatment, and a compatible format of no measured family (a compressed one) is
/// refused.
#[test]
fn bicubic_texels_follow_the_bound_format() {
    use crate::metal::bicubic::{texels, BicubicTexels};
    let texture = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 0,
    };
    let p =
        crate::pipeline::testing::profile("moltenvk-m3", crate::device::DevicePolicy::default())
            .expect("profile");
    let t = slot_iface(TextureType::Type2D, SampledType::Float);
    let slots = BTreeMap::from([(texture, SlotInterface::Texture(t))]);
    let plan = |pf: Option<PixelFormat>| {
        let mut stage = StageShape::default();
        if let Some(pf) = pf {
            stage.textures.insert(
                texture.index,
                TextureShape::Texture {
                    address_modes: crate::metal::sampler::AddressModeMap::IDENTITY,
                    offset_base_axes: 0,
                    ycbcr_in_place: false,
                    ycbcr_planes_sampler: None,
                    texture_type: TextureType::Type2D,
                    pixel_format: pf,
                    sample_count: 1,
                    usage: crate::metal::TextureUsage::SHADER_READ,
                    mipmap_level_count: 1,
                    sparse: false,
                    swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
                },
            );
        }
        let shape = BTreeMap::from([(Stage::Kernel, stage)]);
        let planner = ResourcePlanner {
            profile: &p,
            shape: &shape,
            slots: &slots,
            dependencies: &[],
            stage_flags: |_| vk::PipelineStageFlags2::COMPUTE_SHADER,
            variant: VariantKey::new(),
            diagnostics: Vec::new(),
            uses: Vec::new(),
            template: Default::default(),
            alpha_red_texels: Default::default(),
        };
        planner.bicubic_texels(texture)
    };
    for pf in [
        PixelFormat::RGBA8Unorm,
        PixelFormat::R8Snorm,
        PixelFormat::RGBA16Float,
        PixelFormat::RG32Float,
        PixelFormat::A8Unorm,
    ] {
        let want = texels(pf).expect("measured").0;
        assert_eq!(plan(Some(pf)), Ok(u64::from(want)), "{pf:?}");
    }
    let neutral = u64::from(BicubicTexels::NEUTRAL.0);
    assert_eq!(plan(None), Ok(neutral));
    assert_eq!(plan(Some(PixelFormat::RGBA8Uint)), Ok(neutral));
    assert_eq!(
        plan(Some(PixelFormat::BC1RGBA)),
        Err(BindError::Unsupported(Unsupported::not_implemented(
            Feature::SamplerBicubicFilter
        )))
    );
}

/// A runtime sampler slot specializes `SamplerState` to the bound sampler's word masked by the
/// fields the function reads, and an unbound slot to 0. So the pixel predicate holds as it did of
/// `SamplerCoords`: pixel coordinates wherever the sampler uses them, whatever its filters.
#[test]
fn sampler_state_follows_the_bound_sampler() {
    use crate::bind::state::SamplerDescriptor;
    use crate::metal::sampler::SamplerWord as W;
    use crate::metal::SamplerMinMagFilter as F;
    let slot = SlotRef {
        stage: Stage::Fragment,
        class: SlotClass::Sampler,
        index: 1,
    };
    let p =
        crate::pipeline::testing::profile("moltenvk-m3", crate::device::DevicePolicy::default())
            .expect("profile");
    let slots = BTreeMap::from([(
        slot,
        SlotInterface::Sampler(SamplerSlot {
            compare: false,
            coordinate_sensitive: true,
        }),
    )]);
    let plan = |desc: Option<SamplerDescriptor>| {
        let mut stage = StageShape::default();
        if let Some(d) = desc {
            stage
                .samplers
                .insert(slot.index, BoundSampler::new(d).shape());
        }
        let shape = BTreeMap::from([(Stage::Fragment, stage)]);
        let planner = ResourcePlanner {
            profile: &p,
            shape: &shape,
            slots: &slots,
            dependencies: &[],
            stage_flags: |_| vk::PipelineStageFlags2::FRAGMENT_SHADER,
            variant: VariantKey::new(),
            diagnostics: Vec::new(),
            uses: Vec::new(),
            template: Default::default(),
            alpha_red_texels: Default::default(),
        };
        [W::PIXEL_COORDINATES, W::LINEAR, W::FIELDS]
            .map(|fields| planner.sampler_state(slot, fields) as u32)
    };
    let with = |normalized: bool, filter: F| {
        let mut d = SamplerDescriptor::new();
        d.normalized_coordinates = normalized;
        d.min_filter = filter;
        d.mag_filter = filter;
        d.t_address_mode = crate::metal::SamplerAddressMode::Repeat;
        Some(d)
    };
    let linear = W::MIN_LINEAR | W::MAG_LINEAR;
    let repeat_t = crate::metal::SamplerAddressMode::Repeat.as_raw() << W::ADDRESS_SHIFT[1];
    let pixel = W::PIXEL_COORDINATES;
    assert_eq!(plan(None), [0, 0, 0]);
    assert_eq!(plan(with(true, F::Nearest)), [0, 0, repeat_t]);
    assert_eq!(plan(with(true, F::Linear)), [0, linear, linear | repeat_t]);
    assert_eq!(plan(with(false, F::Nearest)), [pixel, 0, pixel | repeat_t]);
    assert_eq!(
        plan(with(false, F::Linear)),
        [pixel, linear, pixel | linear | repeat_t]
    );
    // The pixel predicate: the word masked by it is exactly the pixel-coordinates bit.
    let states = |desc| plan(desc)[0] == pixel;
    assert!(!states(with(true, F::Nearest)));
    assert!(states(with(false, F::Linear)));
    assert!(states(with(false, F::Nearest)));
    assert!(!states(None));
}

/// A texture slot read or sampled with a `half` result specializes `TexelFormat` from the bound
/// format's class, a texture buffer's included; an unbound slot and a format whose conversion is
/// not measured take `TexelFormat::OTHER`.
#[test]
fn texel_format_follows_the_bound_format() {
    use crate::metal::texel_format::{texel_format, TexelFormat as C};
    let texture = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 0,
    };
    let p =
        crate::pipeline::testing::profile("moltenvk-m3", crate::device::DevicePolicy::default())
            .expect("profile");
    let t = slot_iface(TextureType::Type2D, SampledType::Float);
    let slots = BTreeMap::from([(texture, SlotInterface::Texture(t))]);
    let plan = |shape: Option<TextureShape>| {
        let mut stage = StageShape::default();
        if let Some(s) = shape {
            stage.textures.insert(texture.index, s);
        }
        let shape = BTreeMap::from([(Stage::Kernel, stage)]);
        let planner = ResourcePlanner {
            profile: &p,
            shape: &shape,
            slots: &slots,
            dependencies: &[],
            stage_flags: |_| vk::PipelineStageFlags2::COMPUTE_SHADER,
            variant: VariantKey::new(),
            diagnostics: Vec::new(),
            uses: Vec::new(),
            template: Default::default(),
            alpha_red_texels: Default::default(),
        };
        planner.texel_format(texture)
    };
    let texture2d = |pixel_format| {
        Some(TextureShape::Texture {
            address_modes: crate::metal::sampler::AddressModeMap::IDENTITY,
            offset_base_axes: 0,
            ycbcr_in_place: false,
            ycbcr_planes_sampler: None,
            texture_type: TextureType::Type2D,
            pixel_format,
            sample_count: 1,
            usage: crate::metal::TextureUsage::SHADER_READ,
            mipmap_level_count: 1,
            sparse: false,
            swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
        })
    };
    for pf in [
        PixelFormat::RGBA32Float,
        PixelFormat::R32Float,
        PixelFormat::RGBA16Float,
        PixelFormat::RGBA8Unorm,
        PixelFormat::BGRA8Unorm,
        PixelFormat::RG16Unorm,
        PixelFormat::R8Snorm,
        PixelFormat::RGBA16Snorm,
        PixelFormat::RGBA8UnormSRGB,
        PixelFormat::R8UnormSRGB,
        PixelFormat::BC1RGBASRGB,
    ] {
        let want = texel_format(pf);
        assert_ne!(want, C::OTHER, "{pf:?}");
        assert_eq!(plan(texture2d(pf)), u64::from(want.0), "{pf:?}");
    }
    assert_eq!(
        plan(texture2d(PixelFormat::BGRA10XRSRGB)),
        u64::from(C::OTHER.0)
    );
    // A cube gather's corner alone: its field, without border texels; and a packed format's family,
    // channels and border texels beside its corner and its field.
    let bc1 = texel_format(PixelFormat::BC1RGBA);
    assert_ne!(bc1, C::OTHER);
    assert_eq!(bc1.0 & !(C::CORNER_MASK | C::PACKED_MASK), 0);
    let codes = crate::metal::texel_format::PackedTexels::Argb8.class();
    assert_eq!(bc1.0 & C::PACKED_MASK, codes.0);
    assert_eq!(plan(texture2d(PixelFormat::BC1RGBA)), u64::from(bc1.0));
    for pf in [PixelFormat::RGB10A2Unorm, PixelFormat::RG11B10Float] {
        let want = texel_format(pf);
        assert_ne!(want.0 & C::CORNER_MASK, 0, "{pf:?}");
        assert_ne!(want.0 & C::PACKED_MASK, 0, "{pf:?}");
        assert!(want.has_channels(), "{pf:?}");
        assert_eq!(plan(texture2d(pf)), u64::from(want.0), "{pf:?}");
    }
    assert!(texel_format(PixelFormat::RGB10A2Unorm).packed_exact());
    // A swizzled packed normalized texture keeps its corner alone, dropping its packed field and
    // the family summed on the field's grids.
    let swizzled = crate::metal::TextureSwizzleChannels {
        red: crate::metal::TextureSwizzle::Blue,
        blue: crate::metal::TextureSwizzle::Red,
        ..crate::metal::TextureSwizzleChannels::DEFAULT
    };
    let mut shape = texture2d(PixelFormat::B5G6R5Unorm);
    if let Some(TextureShape::Texture { swizzle, .. }) = shape.as_mut() {
        *swizzle = swizzled;
    }
    assert_eq!(
        plan(shape),
        u64::from(texel_format(PixelFormat::B5G6R5Unorm).0 & C::CORNER_MASK)
    );
    assert_eq!(
        plan(texture2d(PixelFormat::B5G6R5Unorm)),
        u64::from(texel_format(PixelFormat::B5G6R5Unorm).0)
    );
    assert_eq!(plan(None), u64::from(C::OTHER.0));
    // A texture whose alpha reads 1 (the runtime's swizzle reading alpha as one) of a format
    // without alpha keeps its class: a two-plane YCbCr format's field and family, a packed
    // format's; a format with alpha takes the swizzled class.
    use crate::metal::TextureSwizzle as S;
    let alpha_one = crate::metal::TextureSwizzleChannels::new(S::Red, S::Green, S::Blue, S::One);
    for (pf, kept) in [
        (PixelFormat::YCBCR84202P, true),
        (PixelFormat::YCBCR104202P, true),
        (PixelFormat::RGB104202P, true),
        (PixelFormat::B5G6R5Unorm, true),
        (PixelFormat::RGB10A2Unorm, false),
    ] {
        let mut shape = texture2d(pf);
        if let Some(TextureShape::Texture { swizzle, .. }) = shape.as_mut() {
            *swizzle = alpha_one;
        }
        let class = texel_format(pf);
        let want = match kept {
            true => class.0,
            false => class.swizzled().0,
        };
        assert_ne!(class.0, class.swizzled().0, "{pf:?}");
        assert_eq!(plan(shape), u64::from(want), "{pf:?}");
    }
    assert!(texel_format(PixelFormat::YCBCR84202P).packed_exact());
    for pf in [
        PixelFormat::RGBA32Float,
        PixelFormat::RGBA16Float,
        PixelFormat::RGBA8Unorm,
        PixelFormat::RGBA8Snorm,
        PixelFormat::RGB10A2Unorm,
    ] {
        assert_eq!(
            plan(Some(TextureShape::Buffer {
                pixel_format: pf,
                swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
            })),
            u64::from(texel_format(pf).0),
            "{pf:?}"
        );
    }
}

/// A separate texture's write is the image its combined binding's would be, with the same checks. A
/// constexpr pair checks the stage's separate sampler's immutable sampler against the bound view,
/// as the combined write does. A runtime sampler's write meets no image, so each of its textures'
/// writes checks the bound sampler's filtering against its view at plan time: a linear sampler on
/// `R32Float` is refused where a committed profile cannot filter it (the format named), and passes
/// elsewhere, as on `RGBA8Unorm` everywhere; a nearest one needs nothing; an unbound sampler is the
/// neutral, which needs nothing. The separate sampler's own write is the bound sampler for the use,
/// or none for a constexpr one.
#[test]
fn a_separate_pair_takes_the_combined_pairs_checks() {
    use crate::bind::state::SamplerDescriptor;
    use crate::metal::sampler::{vk_sampler, SamplerUse};
    use crate::metal::SamplerMinMagFilter;
    let texture = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 3,
    };
    let sampler_slot = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Sampler,
        index: 1,
    };
    let mut linear = SamplerDescriptor::new();
    linear.min_filter = SamplerMinMagFilter::Linear;
    linear.mag_filter = SamplerMinMagFilter::Linear;
    let nearest = SamplerDescriptor::new();
    let (mut refused, mut passed) = (BTreeSet::new(), BTreeSet::new());
    for (stem, p) in crate::spirv::testing::committed_profiles().expect("profiles") {
        let t = slot_iface(TextureType::Type2D, SampledType::Float);
        for pf in [PixelFormat::R32Float, PixelFormat::RGBA8Unorm] {
            let class = declared(
                &t,
                vk::DescriptorType::SAMPLED_IMAGE,
                false,
                false,
                StorageFormatClass::NotStorage,
            )
            .expect("declared");
            let view = sampled_view(&p, pf, &class).expect("view");
            let filterable = p.supports_format(view.format, FormatUsage::SampledImageFilterLinear);
            let refusal = Err(BindError::Unsupported(Unsupported::needs(
                Feature::Format,
                Capability::Format {
                    format: view.format,
                    usage: FormatUsage::SampledImageFilterLinear,
                },
            )));
            let bound = WriteSource::Texture {
                slot: texture,
                view,
            };
            for (name, desc) in [
                ("linear", Some(linear)),
                ("nearest", Some(nearest)),
                ("unbound", None),
            ] {
                let slots = BTreeMap::from([
                    (texture, SlotInterface::Texture(t)),
                    (
                        sampler_slot,
                        SlotInterface::Sampler(SamplerSlot {
                            compare: false,
                            coordinate_sensitive: false,
                        }),
                    ),
                ]);
                let mut stage = StageShape::default();
                stage.textures.insert(
                    texture.index,
                    TextureShape::Texture {
                        address_modes: crate::metal::sampler::AddressModeMap::IDENTITY,
                        offset_base_axes: 0,
                        ycbcr_in_place: false,
                        ycbcr_planes_sampler: None,
                        texture_type: TextureType::Type2D,
                        pixel_format: pf,
                        sample_count: 1,
                        usage: crate::metal::TextureUsage::SHADER_READ,
                        mipmap_level_count: 1,
                        sparse: false,
                        swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
                    },
                );
                if let Some(d) = desc {
                    stage
                        .samplers
                        .insert(sampler_slot.index, BoundSampler::new(d).shape());
                }
                let shape = BTreeMap::from([(Stage::Kernel, stage)]);
                let mut planner = ResourcePlanner {
                    profile: &p,
                    shape: &shape,
                    slots: &slots,
                    dependencies: &[],
                    stage_flags: |_| vk::PipelineStageFlags2::COMPUTE_SHADER,
                    variant: VariantKey::new(),
                    diagnostics: Vec::new(),
                    uses: Vec::new(),
                    template: Default::default(),
                    alpha_red_texels: Default::default(),
                };
                // The runtime pair.
                let runtime =
                    CombinedSampler::new(SamplerRef::Slot(sampler_slot.index), false, false);
                let planned = planner.separate_texture(texture, runtime, None);
                let what = format!("{stem} {pf:?} runtime {name}");
                if name == "linear" && !filterable {
                    assert_eq!(planned, refusal, "{what}");
                    refused.insert(stem.clone());
                } else {
                    assert_eq!(planned, Ok(bound.clone()), "{what}");
                    if name == "linear" && pf == PixelFormat::R32Float {
                        passed.insert(stem.clone());
                    }
                }
                let written = planner.separate_sampler(Stage::Kernel, runtime, None);
                let expected = match desc {
                    Some(_) => WriteSource::Sampler {
                        slot: sampler_slot,
                        compare: false,
                        integer: false,
                        opaque_black: crate::metal::sampler::OpaqueBlackBorder::Stated,
                    },
                    None => WriteSource::Neutral(NeutralResource::Sampler { compare: false }),
                };
                assert_eq!(written, Ok(Some(expected)), "{what}");
                // The constexpr pair: the immutable sampler against the view, as combined.
                let Some(d) = desc else { continue };
                let immutable = vk_sampler(&d, SamplerUse::new(false, false), &p).expect("maps");
                let constexpr = CombinedSampler::new(SamplerRef::Constexpr([0, 0]), false, false);
                let combined = planner.combined(texture, constexpr, Some(&immutable));
                let planned = planner.separate_texture(texture, constexpr, Some(&immutable));
                let what = format!("{stem} {pf:?} constexpr {name}");
                match combined {
                    Ok(WriteSource::CombinedImageSampler {
                        image: ImageSource::Texture { slot, view },
                        sampler: SamplerSource::Immutable,
                    }) => assert_eq!(planned, Ok(WriteSource::Texture { slot, view }), "{what}"),
                    Err(e) => assert_eq!(planned, Err(e), "{what}"),
                    other => panic!("{what}: {other:?}"),
                }
                assert_eq!(
                    planner.separate_sampler(Stage::Kernel, constexpr, Some(&immutable)),
                    Ok(None),
                    "{what}"
                );
            }
        }
    }
    assert!(
        !refused.is_empty() && !passed.is_empty(),
        "refused on {refused:?}, passed on {passed:?}"
    );
}

/// On a device without `A8_UNORM` (lavapipe's profile), an `A8Unorm` texture buffer's bytes are
/// viewed as `R8_UNORM`, Metal's alpha in red: a storage slot through its choice's class
/// (`StorageImageClass::AlphaInRed`), whose swizzle stays the texture's; a uniform one with no
/// class, whose swizzle takes the alpha from red (`TextureSwizzleChannels::alpha_in_red`). A
/// uniform slot that reads without applying the swizzle is refused there. Where the device has
/// `A8_UNORM` (MoltenVK's), the uniform view is `A8_UNORM` and its swizzle the texture's.
#[test]
fn an_a8_texture_buffer_without_a8_unorm_views_its_bytes_in_red() {
    use crate::metal::{TextureSwizzle as S, TextureSwizzleChannels};
    use crate::pipeline::facts::{Mechanism, RuntimeFact};
    let slot = |index| SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index,
    };
    let (uniform, storage) = (slot(0), slot(1));
    let mut read = slot_iface(TextureType::TextureBuffer, SampledType::Float);
    read.access = TextureAccess::new(false, true, false);
    let mut read_write = read;
    read_write.access = TextureAccess::new(false, true, true);
    let slots = BTreeMap::from([
        (uniform, SlotInterface::Texture(read)),
        (storage, SlotInterface::Texture(read_write)),
    ]);
    let shape = |swizzle| {
        let mut stage = StageShape::default();
        for index in [0, 1] {
            let pixel_format = PixelFormat::A8Unorm;
            let buffer = TextureShape::Buffer {
                pixel_format,
                swizzle,
            };
            stage.textures.insert(index, buffer);
        }
        BTreeMap::from([(Stage::Kernel, stage)])
    };
    let swizzle = TextureSwizzleChannels::new(S::Alpha, S::Zero, S::Green, S::One);
    let swizzled = shape(swizzle);
    let stated = [uniform, storage].map(|slot| {
        let fact = RuntimeFact::TextureSwizzle { slot };
        RuntimeDependency::new(fact, Mechanism::Specialization)
    });
    fn make<'a>(
        profile: &'a DeviceProfile,
        shape: &'a BTreeMap<Stage, StageShape>,
        slots: &'a BTreeMap<SlotRef, SlotInterface>,
        dependencies: &'a [RuntimeDependency],
    ) -> ResourcePlanner<'a> {
        ResourcePlanner {
            profile,
            shape,
            slots,
            dependencies,
            stage_flags: |_| vk::PipelineStageFlags2::COMPUTE_SHADER,
            variant: VariantKey::new(),
            diagnostics: Vec::new(),
            uses: Vec::new(),
            template: Default::default(),
            alpha_red_texels: Default::default(),
        }
    }
    let uniform_ty = vk::DescriptorType::UNIFORM_TEXEL_BUFFER;
    let storage_ty = vk::DescriptorType::STORAGE_TEXEL_BUFFER;
    for (profile, held) in [("lavapipe", true), ("moltenvk-m3", false)] {
        let p = crate::pipeline::testing::profile(profile, crate::device::DevicePolicy::default())
            .expect("profile");
        let a8 = PixelFormat::A8Unorm;
        assert_eq!(
            crate::metal::format::alpha_in_red(a8, &p),
            held,
            "{profile}"
        );
        let mut planner = make(&p, &swizzled, &slots, &stated);
        let own = match held {
            true => vk::Format::R8_UNORM,
            false => vk::Format::A8_UNORM,
        };
        let planned = planner.plain_texture(0, 0, uniform_ty, uniform);
        let Ok((WriteSource::TexelBuffer { format, .. }, _)) = planned else {
            panic!("{profile}: {planned:?}");
        };
        assert_eq!(format, own, "{profile}: the uniform view");
        let planned = planner.plain_texture(0, 1, storage_ty, storage);
        let Ok((WriteSource::TexelBuffer { format, .. }, _)) = planned else {
            panic!("{profile}: {planned:?}");
        };
        // A storage view takes `R8_UNORM` wherever `A8_UNORM` does not serve the use, the device's
        // images aside.
        let class = planner
            .variant
            .storage_images
            .get(&storage)
            .map(|c| c.class);
        let in_red = class == Some(StorageImageClass::AlphaInRed);
        assert!(in_red || !held, "{profile}: {class:?}");
        let view = match in_red {
            true => vk::Format::R8_UNORM,
            false => vk::Format::A8_UNORM,
        };
        assert_eq!(format, view, "{profile}: the storage view");
        let read_swizzle = match held {
            true => TextureSwizzleChannels::new(S::Red, S::Zero, S::Zero, S::One),
            false => swizzle,
        };
        assert_eq!(
            planner.texture_swizzle(uniform),
            u64::from(read_swizzle.word()),
            "{profile}: the uniform slot's swizzle"
        );
        assert_eq!(
            planner.texture_swizzle(storage),
            u64::from(swizzle.word()),
            "{profile}: the storage slot's swizzle"
        );
        // A read that does not apply the swizzle.
        let plain = shape(TextureSwizzleChannels::DEFAULT);
        let mut planner = make(&p, &plain, &slots, &[]);
        let planned = planner.plain_texture(0, 0, uniform_ty, uniform);
        match held {
            true => assert_eq!(
                planned.map(|(w, _)| w),
                Err(BindError::Unsupported(Unsupported::not_implemented(
                    Feature::Format
                ))),
                "{profile}"
            ),
            false => assert!(planned.is_ok(), "{profile}: {planned:?}"),
        }
    }
}
