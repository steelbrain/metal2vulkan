//! The split before an image use over bytes written in the same `VkCommandBuffer`
//! (`alias_split.rs`): which uses meet a noted write, what a joining encoder starts from, and the
//! devices that split.

use crate::alias_split::{splits_at_aliases, AliasWrites};
use crate::encoder_fold::FoldOrder;
use crate::resource::ResourceKey;
use crate::tracker::{Access, ImagePlace, Summary, Use, UseRange};
use ash::vk;
use metal2vulkan::device::DriverFact;
use std::collections::BTreeSet;
use std::ops::Range;

fn copy_read() -> Access {
    Access::new(
        vk::PipelineStageFlags2::COPY,
        vk::AccessFlags2::TRANSFER_READ,
    )
}

fn copy_write() -> Access {
    Access::new(
        vk::PipelineStageFlags2::COPY,
        vk::AccessFlags2::TRANSFER_WRITE,
    )
}

/// A blit's write of `bytes` of the buffer or heap of `key`.
fn bytes_written(key: ResourceKey, bytes: Range<u64>) -> Use<ResourceKey> {
    Use::writes(key, true, UseRange::Bytes(bytes), copy_write())
}

/// A copy's read of the first level of an image of `key` that lies at `bytes` of its memory.
fn image_read(key: ResourceKey, bytes: Range<u64>) -> Use<ResourceKey> {
    let range = UseRange::Image {
        aspects: vk::ImageAspectFlags::COLOR,
        levels: 0..1,
        layers: 0..1,
        place: ImagePlace::bytes(bytes),
    };
    Use::reads(key, true, range, copy_read())
}

/// The measured form: a heap buffer's write of a 64x64 `RGBA8Unorm` texture's 16384 bytes, then
/// the copy out of the texture's image over them, in one `VkCommandBuffer`. Its use of the image
/// meets the write, so it goes in a new one; a use of an image placed apart, of another key's, a
/// read of the bytes through a buffer, and any use after the new one begins meet none.
#[test]
fn an_image_use_over_bytes_written_in_the_command_buffer_meets_the_write() {
    let heap = ResourceKey::new();
    let other = ResourceKey::new();
    let mut writes = AliasWrites::default();
    assert!(
        !writes.meets(&[image_read(heap, 0..16384)]),
        "nothing written"
    );
    writes.note(&[
        Use::reads(other, true, UseRange::Bytes(0..16384), copy_read()),
        bytes_written(heap, 0..16384),
    ]);
    assert!(writes.meets(&[image_read(heap, 0..16384)]));
    assert!(
        writes.meets(&[image_read(heap, 8192..24576)]),
        "overlapping"
    );
    assert!(
        !writes.meets(&[image_read(heap, 16384..32768)]),
        "placed apart"
    );
    assert!(!writes.meets(&[image_read(other, 0..16384)]), "another key");
    let buffer_read = Use::reads(heap, true, UseRange::Bytes(0..16384), copy_read());
    assert!(!writes.meets(&[buffer_read]), "a read through a buffer");
    // A read of the bytes noted no write.
    let mut read_only = AliasWrites::default();
    read_only.note(&[Use::reads(
        heap,
        true,
        UseRange::Bytes(0..16384),
        copy_read(),
    )]);
    assert!(!read_only.meets(&[image_read(heap, 0..16384)]));
    // A write through an image notes nothing: the measured direction alone.
    let mut through_image = AliasWrites::default();
    let image_written = Use {
        write: copy_write(),
        read: Access::NONE,
        ..image_read(heap, 0..16384)
    };
    through_image.note(&[image_written]);
    assert!(!through_image.meets(&[image_read(heap, 0..16384)]));
    // The use goes in a new `VkCommandBuffer`, which nothing wrote.
    writes.begun();
    assert!(!writes.meets(&[image_read(heap, 0..16384)]));
}

/// An encoder that joins a group of encoders starts from the bytes the group's earlier encoders
/// wrote into its shared `VkCommandBuffer`, as its order holds them: the folded form measured.
#[test]
fn a_joining_encoder_meets_what_the_group_wrote() {
    let heap = ResourceKey::new();
    let fill = ResourceKey::new();
    let first = Summary::outside_encoders(vec![
        Use::reads(fill, true, UseRange::Bytes(0..16384), copy_read()),
        bytes_written(heap, 0..16384),
    ]);
    let order = FoldOrder::after(&first);
    let written: Vec<_> = order.byte_writes().collect();
    assert_eq!(written, [(heap, 0..16384)]);
    let joined = AliasWrites::joining(Some(&order));
    assert!(joined.meets(&[image_read(heap, 0..16384)]));
    assert!(
        !joined.meets(&[image_read(fill, 0..16384)]),
        "the fill was read"
    );
    assert!(!AliasWrites::joining(None).meets(&[image_read(heap, 0..16384)]));
}

/// Past the writes it keeps apart, a write of a new key makes every image use meet one, until the
/// next `VkCommandBuffer`: the noted writes stay bounded.
#[test]
fn writes_of_many_keys_meet_every_image_use() {
    let mut writes = AliasWrites::default();
    let keys: Vec<ResourceKey> = (0..40).map(|_| ResourceKey::new()).collect();
    for key in &keys {
        writes.note(&[bytes_written(*key, 0..64)]);
    }
    let unrelated = ResourceKey::new();
    assert!(writes.meets(&[image_read(unrelated, 0..64)]));
    writes.begun();
    assert!(!writes.meets(&[image_read(unrelated, 0..64)]));
    // A key's writes join into one range.
    writes.note(&[bytes_written(keys[0], 0..64)]);
    writes.note(&[bytes_written(keys[0], 4096..4160)]);
    assert!(writes.meets(&[image_read(keys[0], 1024..1088)]));
}

/// The devices that split: those whose driver has the fact (NVIDIA's on the RTX 3090), no other.
#[test]
fn a_device_splits_where_its_driver_s_image_uses_miss_buffer_writes() {
    let fact = DriverFact::ImageUsesMissBufferWritesInOneCommandBuffer;
    assert!(splits_at_aliases(&BTreeSet::from([fact])));
    assert!(!splits_at_aliases(&BTreeSet::new()));
    let others = BTreeSet::from([DriverFact::CompressedImagesHideAliasedWrites]);
    assert!(!splits_at_aliases(&others));
}
