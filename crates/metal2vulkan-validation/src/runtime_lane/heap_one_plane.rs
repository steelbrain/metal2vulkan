//! A heap's texture of one of Apple's one-plane YCbCr formats, on the device the cfg selects,
//! under the validation layer: one Metal lays out in rows (a 16 x 16 2D texture, 32-byte rows) is
//! made with no notice, and the shapes Metal lays out no one way (a 13 x 7 one, whose 26-byte rows
//! Metal's blits lay 16 bytes apart, and a mipmapped one, whose first level Metal addresses past
//! its answer) are made and announce `heap-one-plane-layout` once ([`ONE_PLANE_LAYOUTS`]). The
//! runtime's `tests::heap_one_plane_rows` holds their bytes in the heap to Metal's.

use super::degrade_notices::{counts, on_device};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::{
    Degradation, HeapDescriptor, HeapType, ResourceOptions, TextureDescriptor,
};

/// The kind a heap's one-plane YCbCr texture laid out otherwise than Metal's blits lay it takes.
pub(super) const ONE_PLANE_LAYOUTS: [Degradation; 1] = [Degradation::HeapOnePlaneLayout];

#[test]
fn a_heap_s_one_plane_ycbcr_texture_metal_lays_no_one_way_is_announced() {
    on_device(|device, messages| {
        let options = ResourceOptions::private();
        let heap = device
            .new_heap(&HeapDescriptor::new(HeapType::Placement, 64 << 10, options))
            .unwrap();
        let texture = |width, height, mipmapped| {
            let pf = PixelFormat::YCBCR84221P;
            let mut descriptor = TextureDescriptor::texture_2d(pf, width, height, mipmapped);
            descriptor.resource_options = options;
            descriptor
        };
        heap.new_texture_at_offset(&texture(16, 16, false), 0)
            .unwrap();
        let rows = counts(&messages.take_notices());
        let kind = Degradation::HeapOnePlaneLayout;
        assert!(!rows.contains_key(&kind), "32-byte rows: {rows:?}");
        heap.new_texture_at_offset(&texture(13, 7, false), 0)
            .unwrap();
        heap.new_texture_at_offset(&texture(16, 16, true), 0)
            .unwrap();
        let heard = counts(&messages.take_notices());
        assert_eq!(heard.get(&kind), Some(&1), "{heard:?}");
    });
}
