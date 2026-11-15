//! The Vulkan image views of bindings: the swizzle composition and the views' ranges, usage and
//! refusals.

use super::super::address_map::AddressModeMap;
use super::super::binding_view::{binding_view, texture_facts};
use super::super::chroma_pairs::holds_pairs;
use super::super::image_plan::plan_image;
use super::super::texture_descriptor::{TextureDescriptor, TextureSwizzleChannels};
use super::super::view_plan::{check_view, ViewProblem, ViewRecord, ViewRefusal, ViewRequest};
use super::image_plan::{committed, facts};
use super::texture_descriptor::APPLE;
use ash::vk;
use metal2vulkan::bind::ImageViewDesc;
use metal2vulkan::device::{Device, DeviceDescription, FormatUsage};
use metal2vulkan::error::{Capability, Feature, Needs, Unsupported};
use metal2vulkan::metal::{
    PixelFormat, SamplerAddressMode, TextureSwizzle, TextureType, TextureUsage,
};
use metal2vulkan::vk::{
    ComponentMapping, ComponentSwizzle, Format, ImageAspectFlags, ImageLayout, ImageUsageFlags,
    ImageViewType,
};

fn lavapipe() -> (DeviceDescription, Device) {
    let (_, desc, device) = committed()
        .into_iter()
        .find(|(n, ..)| n == "lavapipe")
        .unwrap();
    (desc, device)
}

fn swizzle(s: &str) -> TextureSwizzleChannels {
    let c = |ch: char| match ch {
        'R' => TextureSwizzle::Red,
        'G' => TextureSwizzle::Green,
        'B' => TextureSwizzle::Blue,
        'A' => TextureSwizzle::Alpha,
        '0' => TextureSwizzle::Zero,
        _ => TextureSwizzle::One,
    };
    let v: Vec<TextureSwizzle> = s.chars().map(c).collect();
    TextureSwizzleChannels::new(v[0], v[1], v[2], v[3])
}

fn desc(format: Format, view_type: ImageViewType, components: ComponentMapping) -> ImageViewDesc {
    let layout = ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    ImageViewDesc::new(
        view_type,
        format,
        ImageAspectFlags::COLOR,
        layout,
        components,
    )
}

#[test]
fn composition_keeps_format_mappings_and_is_canonical() {
    use ComponentSwizzle as S;
    let identity = ComponentMapping::IDENTITY;
    assert_eq!(TextureSwizzleChannels::DEFAULT.compose(identity), identity);
    let explicit = ComponentMapping {
        r: S::R,
        g: S::G,
        b: S::B,
        a: S::A,
    };
    assert_eq!(TextureSwizzleChannels::DEFAULT.compose(explicit), identity);
    // A red/blue twin's mapping, and `A8Unorm` read through `R8_UNORM`.
    let twin = ComponentMapping {
        r: S::B,
        g: S::IDENTITY,
        b: S::R,
        a: S::IDENTITY,
    };
    assert_eq!(TextureSwizzleChannels::DEFAULT.compose(twin), twin);
    assert_eq!(
        swizzle("GBAR").compose(twin),
        ComponentMapping {
            r: S::G,
            g: S::R,
            b: S::A,
            a: S::B,
        }
    );
    let alpha = ComponentMapping {
        r: S::ZERO,
        g: S::ZERO,
        b: S::ZERO,
        a: S::R,
    };
    assert_eq!(
        swizzle("A01R").compose(alpha),
        ComponentMapping {
            r: S::IDENTITY,
            g: S::ZERO,
            b: S::ONE,
            a: S::ZERO,
        }
    );
}

/// A sampled binding of a 4:2:2 texture whose texels a carrier holds for copies alone (one with no
/// decoded twin) is the format's gap, naming the texture's format, not a need of the twin's view
/// format, which every device samples.
#[test]
fn a_carrier_s_binding_is_the_format_s_gap() {
    let (d, device) = lavapipe();
    let facts = facts(&d, &device);
    let usage = TextureUsage::SHADER_READ;
    let view = desc(
        Format::R8G8B8A8_UNORM,
        ImageViewType::TYPE_2D,
        ComponentMapping::IDENTITY,
    );
    for pf in [PixelFormat::GBGR422, PixelFormat::BGRG422] {
        let t = TextureDescriptor {
            usage,
            ..TextureDescriptor::texture_2d(pf, 32, 4, false)
        };
        let layout = t.check(APPLE).unwrap();
        let image = plan_image(&layout, usage, false, &facts).unwrap();
        assert!(holds_pairs(&image), "{pf:?}");
        let record = ViewRecord::of_texture(&layout, usage, t.swizzle);
        assert_eq!(
            binding_view(&image, &record, &view, false),
            Err(ViewRefusal::Unsupported(
                Unsupported::not_implemented(Feature::Format).naming_format(pf)
            )),
            "{pf:?}"
        );
    }
}

#[test]
fn binding_views_take_the_plans_view_and_the_records_range_and_swizzle() {
    let (d, device) = lavapipe();
    let facts = facts(&d, &device);
    let usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
    let t = TextureDescriptor {
        texture_type: TextureType::Type2DArray,
        width: 8,
        height: 8,
        array_length: 4,
        mipmap_level_count: 3,
        usage,
        ..TextureDescriptor::default()
    };
    let layout = t.check(APPLE).unwrap();
    let image = plan_image(&layout, usage, false, &facts).unwrap();
    let whole = ViewRecord::of_texture(&layout, usage, t.swizzle);
    let part = check_view(
        &whole,
        &ViewRequest {
            texture_type: TextureType::Type2D,
            levels: 1..3,
            slices: 2..3,
            swizzle: Some(swizzle("B0R1")),
            ..ViewRequest::of_format(&whole, PixelFormat::RGBA8Unorm)
        },
        device.profile(),
    )
    .unwrap();
    let plan = desc(
        Format::R8G8B8A8_UNORM,
        ImageViewType::TYPE_2D,
        ComponentMapping::IDENTITY,
    );
    let v = binding_view(&image, &part, &plan, false).unwrap();
    assert_eq!(
        (v.base_level, v.level_count, v.base_layer, v.layer_count),
        (1, 2, 2, 1)
    );
    assert_eq!(v.usage, ImageUsageFlags::SAMPLED);
    assert_eq!(
        v.components,
        swizzle("B0R1").compose(ComponentMapping::IDENTITY)
    );
    assert_eq!(
        (v.format, v.view_type),
        (Format::R8G8B8A8_UNORM, ImageViewType::TYPE_2D)
    );
    // A storage slot's level view holds one level, counted from the record's first.
    let mut one = plan;
    one.level = Some(1);
    let v = binding_view(&image, &whole, &one, false).unwrap();
    assert_eq!((v.base_level, v.level_count, v.layer_count), (1, 1, 4));
    let mut past = plan;
    past.level = Some(3);
    assert_eq!(
        binding_view(&image, &whole, &past, false),
        Err(ViewRefusal::Invalid(ViewProblem::LevelRange))
    );
    // A swizzled view has no shader writes, and the image, made without them, no storage
    // usage: the runtime's gap (plain Metal writes it).
    match binding_view(&image, &part, &plan, true) {
        Err(ViewRefusal::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::UndeclaredTextureUsage)
        }
        other => panic!("{other:?}"),
    }
    // The image, made without shader writes, has no storage usage for a record that has them.
    let writes = ViewRecord {
        usage: usage | TextureUsage::SHADER_WRITE,
        ..whole
    };
    match binding_view(&image, &writes, &plan, true) {
        Err(ViewRefusal::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::Format);
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![vec![Capability::Format {
                    format: Format::R8G8B8A8_UNORM,
                    usage: FormatUsage::StorageImage
                }]])
            );
        }
        other => panic!("{other:?}"),
    }
    // A format outside the image's compatibility class.
    let wide = desc(
        Format::R16G16B16A16_SFLOAT,
        ImageViewType::TYPE_2D,
        ComponentMapping::IDENTITY,
    );
    match binding_view(&image, &whole, &wide, false) {
        Err(ViewRefusal::Unsupported(u)) => assert_eq!(u.feature, Feature::TextureView),
        other => panic!("{other:?}"),
    }
}

/// A swizzled view of a texture with shader writes (Metal probe cases `rw.view.read` and
/// `rw.view.write`, the M3 Max and the M4): Metal's validation stops a write binding of it, and
/// plain Metal reads it swizzled and writes through it with the swizzle undone, which a Vulkan
/// storage view, taking no swizzle, cannot. Where the binding plan's function applies the swizzle
/// (`ImageViewDesc::shader_swizzle`, a storage slot's) the view takes none; elsewhere (a texture a
/// function reaches through its handle) it stays the runtime's gap, not Metal's refusal.
#[test]
fn a_swizzled_view_bound_for_writing_takes_no_swizzle_where_the_function_applies_it() {
    let (d, device) = lavapipe();
    let facts = facts(&d, &device);
    let usage =
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::PIXEL_FORMAT_VIEW;
    let t = TextureDescriptor {
        usage,
        ..TextureDescriptor::default()
    };
    let layout = t.check(APPLE).unwrap();
    let image = plan_image(&layout, usage, false, &facts).unwrap();
    let whole = ViewRecord::of_texture(&layout, usage, t.swizzle);
    let pf = t.pixel_format;
    let view = check_view(
        &whole,
        &ViewRequest {
            swizzle: Some(swizzle("GBAR")),
            ..ViewRequest::of_format(&whole, pf)
        },
        device.profile(),
    )
    .unwrap();
    assert_eq!(
        view.usage,
        TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW
    );
    // A view of the view without a swizzle of its own keeps the swizzle, and the gap.
    let again = check_view(&view, &ViewRequest::of_format(&view, pf), device.profile()).unwrap();
    let plan = desc(
        Format::R8G8B8A8_UNORM,
        ImageViewType::TYPE_2D,
        ComponentMapping::IDENTITY,
    );
    let mut shader = plan;
    shader.shader_swizzle = true;
    for record in [&view, &again] {
        match binding_view(&image, record, &plan, true) {
            Err(ViewRefusal::Unsupported(u)) => assert_eq!(u.feature, Feature::TextureSwizzle),
            other => panic!("{other:?}"),
        }
        let v = binding_view(&image, record, &plan, false).unwrap();
        assert_eq!(
            v.components,
            swizzle("GBAR").compose(ComponentMapping::IDENTITY)
        );
        let v = binding_view(&image, record, &shader, true).unwrap();
        assert_eq!(v.components, ComponentMapping::IDENTITY);
        assert_eq!(v.usage, ImageUsageFlags::STORAGE);
    }
    // The texture itself, unswizzled, is written as any other.
    binding_view(&image, &whole, &plan, true).unwrap();
}

#[test]
fn a_swizzled_texture_of_unknown_usage_is_read_only() {
    let (d, device) = lavapipe();
    let facts = facts(&d, &device);
    let t = TextureDescriptor {
        usage: TextureUsage::empty(),
        swizzle: swizzle("GBAR"),
        ..TextureDescriptor::default()
    };
    let layout = t.check(APPLE).unwrap();
    let image = plan_image(&layout, t.usage, false, &facts).unwrap();
    let r = ViewRecord::of_texture(&layout, t.usage, t.swizzle);
    let plan = desc(
        Format::R8G8B8A8_UNORM,
        ImageViewType::TYPE_2D,
        ComponentMapping::IDENTITY,
    );
    // Metal reports such a texture's usage as `ShaderRead | PixelFormatView` (probe 4), and
    // plain Metal writes it; an image with storage usage takes the binding, whose swizzle is the
    // runtime's gap but where the function applies it.
    match binding_view(&image, &r, &plan, true) {
        Err(ViewRefusal::Unsupported(u)) => assert_eq!(u.feature, Feature::TextureSwizzle),
        other => panic!("{other:?}"),
    }
    let mut shader = plan;
    shader.shader_swizzle = true;
    let v = binding_view(&image, &r, &shader, true).unwrap();
    assert_eq!(v.components, ComponentMapping::IDENTITY);
    let v = binding_view(&image, &r, &plan, false).unwrap();
    assert_eq!(
        v.components,
        swizzle("GBAR").compose(ComponentMapping::IDENTITY)
    );
}

#[test]
fn the_view_create_info_carries_the_view() {
    let (d, device) = lavapipe();
    let facts = facts(&d, &device);
    let t = TextureDescriptor::default();
    let layout = t.check(APPLE).unwrap();
    let image = plan_image(&layout, t.usage, false, &facts).unwrap();
    let r = ViewRecord::of_texture(&layout, t.usage, swizzle("GBAR"));
    let plan = desc(
        Format::R8G8B8A8_SRGB,
        ImageViewType::TYPE_2D,
        ComponentMapping::IDENTITY,
    );
    let v = binding_view(&image, &r, &plan, false).unwrap();
    let mut usage = vk::ImageViewUsageCreateInfo::default();
    let info = v.create_info(vk::Image::null(), &mut usage);
    assert_eq!(Format::from(info.format), Format::R8G8B8A8_SRGB);
    assert_eq!(ImageViewType::from(info.view_type), ImageViewType::TYPE_2D);
    assert_eq!(ComponentMapping::from(info.components), v.components);
    assert_eq!(info.subresource_range.level_count, 1);
    assert!(!info.p_next.is_null());
    assert_eq!(ImageUsageFlags::from(usage.usage), ImageUsageFlags::SAMPLED);
}

#[test]
fn a_bindings_texture_facts_are_the_records() {
    let layout = TextureDescriptor {
        texture_type: TextureType::CubeArray,
        width: 16,
        height: 16,
        array_length: 2,
        mipmap_level_count: 3,
        usage: TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
        ..TextureDescriptor::default()
    }
    .check(APPLE)
    .unwrap();
    let record = ViewRecord::of_texture(
        &layout,
        TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
        TextureSwizzleChannels::DEFAULT,
    );
    let facts = texture_facts(&record);
    assert_eq!(facts.texture_type, TextureType::CubeArray);
    assert_eq!((facts.array_length, facts.mipmap_level_count), (2, 3));
    assert_eq!(facts.usage, record.usage);
}

/// A view's binding facts state its first level's place in its image and the image's base-level
/// size, from which a slot's `SpecKey::OffsetBaseAxes` is planned: a view at level 2 of an 18 x 16
/// texture states the width's bit (its own 4 a power of two, its parent's 18 not), the texture
/// itself none.
#[test]
fn a_view_s_facts_state_its_parent_s_base() {
    let layout = TextureDescriptor {
        texture_type: TextureType::Type2D,
        width: 18,
        height: 16,
        mipmap_level_count: 5,
        usage: TextureUsage::SHADER_READ,
        ..TextureDescriptor::default()
    }
    .check(APPLE)
    .unwrap();
    let mut record = ViewRecord::of_texture(
        &layout,
        TextureUsage::SHADER_READ,
        TextureSwizzleChannels::DEFAULT,
    );
    let own = texture_facts(&record);
    assert_eq!(own.parent_relative_level, 0);
    assert_eq!(own.offset_base_axes(), 0);
    record.base_level = 2;
    record.level_count = 3;
    let view = texture_facts(&record);
    assert_eq!(view.parent_relative_level, 2);
    assert_eq!(view.parent_base_size, [18, 16, 1]);
    assert_eq!(view.offset_base_axes(), 0b01);
}

/// A texture whose alpha reads 1 takes `ShaderWrite` and `RenderTarget` usage, keeps its Metal
/// swizzle and usage, and is read through the swizzle that reads its alpha as 1 (after its own
/// swizzle): its sampled view composes it, the plan's facts state it, and a storage view the
/// function swizzles takes none, so a write keeps the alpha it writes. Its views keep it.
#[test]
fn a_texture_whose_alpha_reads_one_is_read_so_and_written_plain() {
    let (d, device) = lavapipe();
    let facts = facts(&d, &device);
    let usage = TextureUsage::SHADER_READ
        | TextureUsage::SHADER_WRITE
        | TextureUsage::RENDER_TARGET
        | TextureUsage::PIXEL_FORMAT_VIEW;
    for (own, read) in [
        ("RGBA", "RGB1"),
        ("ARGB", "1RGB"),
        ("RRR1", "RRR1"),
        ("AAAA", "1111"),
    ] {
        let t = TextureDescriptor {
            usage: if own == "RGBA" {
                usage
            } else {
                TextureUsage::SHADER_READ
            },
            swizzle: swizzle(own),
            alpha_reads_one: true,
            ..TextureDescriptor::default()
        };
        let layout = t.check(APPLE).unwrap();
        let image = plan_image(&layout, t.image_usage(), false, &facts).unwrap();
        let r = ViewRecord::of_descriptor(&layout, &t);
        assert_eq!(r.swizzle, swizzle(own), "{own}");
        assert_eq!(r.sampled_swizzle(), swizzle(read), "{own}");
        assert_eq!(texture_facts(&r).swizzle, swizzle(read), "{own}");
        let plan = desc(
            Format::R8G8B8A8_UNORM,
            ImageViewType::TYPE_2D,
            ComponentMapping::IDENTITY,
        );
        let sampled = binding_view(&image, &r, &plan, false).unwrap();
        assert_eq!(
            sampled.components,
            swizzle(read).compose(ComponentMapping::IDENTITY),
            "{own}"
        );
        if own == "RGBA" {
            assert_eq!(r.usage, usage);
            let mut shader = plan;
            shader.shader_swizzle = true;
            let storage = binding_view(&image, &r, &shader, true).unwrap();
            assert_eq!(storage.components, ComponentMapping::IDENTITY);
        }
    }
    // Without it, the record reads as its swizzle says.
    let t = TextureDescriptor::default();
    let r = ViewRecord::of_descriptor(&t.check(APPLE).unwrap(), &t);
    assert!(!r.alpha_reads_one);
    assert_eq!(r.sampled_swizzle(), TextureSwizzleChannels::DEFAULT);
    // A view keeps it, and its texture's address-mode map, whatever swizzle it takes.
    let map =
        AddressModeMap::IDENTITY.with(SamplerAddressMode::Repeat, SamplerAddressMode::ClampToEdge);
    let t = TextureDescriptor {
        usage: TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
        alpha_reads_one: true,
        address_mode_map: map,
        ..TextureDescriptor::default()
    };
    let source = ViewRecord::of_descriptor(&t.check(APPLE).unwrap(), &t);
    let request = ViewRequest {
        pixel_format: PixelFormat::RGBA8Unorm,
        texture_type: TextureType::Type2D,
        levels: 0..1,
        slices: 0..1,
        swizzle: Some(swizzle("BGRA")),
    };
    let view = check_view(&source, &request, device.profile()).unwrap();
    assert!(view.alpha_reads_one);
    assert_eq!(view.address_mode_map, map);
    assert_eq!(view.sampled_swizzle(), swizzle("BGR1"));
}
