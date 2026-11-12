//! A record of the driver's data names a pass without its attachments' formats
//! (`PassShape::without_formats`, `GraphicsContent::record_parts`): two passes that differ only
//! in a colour or depth/stencil attachment's format name one record, and two that differ in which
//! attachments they have, their samples, their fetch inputs or their view mask name two.

use crate::color_list::ColorList;
use crate::render_target::PassShape;
use metal2vulkan::vk::Format;

/// A pass of `samples` samples with `colors` and the depth/stencil attachment `depth`.
fn shape(samples: u32, colors: &[Option<Format>], depth: Option<Format>) -> PassShape {
    PassShape {
        colors: ColorList::of(colors.iter().copied()).unwrap(),
        depth_stencil: depth,
        samples,
        color_resolves: ColorList::of(colors.iter().map(|_| None)).unwrap(),
        depth_stencil_resolve: None,
        fetch_inputs: false,
        no_inputs: ColorList::new(),
        storage: ColorList::new(),
        view_mask: 0,
        locations: ColorList::new(),
    }
}

/// How a record names `shape`.
fn named(shape: &PassShape) -> String {
    format!("{:?}", shape.without_formats())
}

#[test]
fn a_record_names_a_pass_without_its_formats() {
    let rgba8 = Some(Format::R8G8B8A8_UNORM);
    let bgra8 = Some(Format::B8G8R8A8_UNORM);
    let half = Some(Format::R16G16B16A16_SFLOAT);
    let first = shape(1, &[rgba8], None);
    for other in [bgra8, half] {
        assert_eq!(named(&first), named(&shape(1, &[other], None)), "{other:?}");
    }
    assert_eq!(
        named(&shape(4, &[rgba8, None, half], Some(Format::D32_SFLOAT))),
        named(&shape(
            4,
            &[bgra8, None, rgba8],
            Some(Format::D24_UNORM_S8_UINT)
        )),
        "every attachment's format left out, its presence kept"
    );
    for apart in [
        shape(1, &[rgba8, rgba8], None),
        shape(1, &[None, rgba8], None),
        shape(1, &[rgba8], Some(Format::D32_SFLOAT)),
        shape(4, &[rgba8], None),
        PassShape {
            fetch_inputs: true,
            ..shape(1, &[rgba8], None)
        },
        shape(1, &[rgba8], None).with_view_mask(0b11),
    ] {
        assert_ne!(named(&first), named(&apart), "{apart:?}");
    }
}
