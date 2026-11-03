//! A pushed set 0's raw writes, and a written set's (`binding::push_set`, `binding::write_sets`):
//! each points at its own write's info, where it lies, so nothing copies the infos into arrays of
//! their own, and only the places written are handed over.

use crate::binding::{fill_raw, Info, SetWrite, WRITES_AT_ONCE};
use ash::vk::{self, Handle};
use std::mem::MaybeUninit;

#[test]
fn a_raw_write_points_at_its_own_info() {
    let buffer = vk::DescriptorBufferInfo::default()
        .buffer(vk::Buffer::from_raw(7))
        .offset(256)
        .range(64);
    let image = vk::DescriptorImageInfo::default()
        .image_view(vk::ImageView::from_raw(9))
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let write = |binding, ty, info| SetWrite {
        binding,
        element: 0,
        ty,
        info,
    };
    let writes = [
        write(0, vk::DescriptorType::UNIFORM_BUFFER, Info::Buffer(buffer)),
        write(1, vk::DescriptorType::SAMPLED_IMAGE, Info::Image(image)),
        write(
            2,
            vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            Info::Texel(vk::BufferView::from_raw(11)),
        ),
    ];
    let mut places = [const { MaybeUninit::uninit() }; WRITES_AT_ONCE];
    let raw = fill_raw(writes.iter().map(SetWrite::raw), &mut places);
    assert_eq!(raw.len(), writes.len(), "one raw write each, no more");
    for (raw, write) in raw.iter().zip(&writes) {
        assert_eq!(
            (raw.dst_binding, raw.dst_array_element),
            (write.binding, write.element)
        );
        assert_eq!(raw.descriptor_type.as_raw(), write.ty.as_raw());
        assert_eq!(raw.descriptor_count, 1);
        match &write.info {
            Info::Buffer(info) => assert!(std::ptr::eq(raw.p_buffer_info, info)),
            Info::Image(info) => assert!(std::ptr::eq(raw.p_image_info, info)),
            Info::Texel(view) => assert!(std::ptr::eq(raw.p_texel_buffer_view, view)),
        }
    }
}
