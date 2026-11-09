//! The work host calls stage for the device's next submission (`texture/first_move.rs`).

use crate::texture::first_move::{Filled, FirstMoves, Mine, TexelCopy};
use ash::vk;
use ash::vk::Handle;
use std::any::Any;
use std::sync::{Arc, Weak};

/// A host call's copy staged into an image makes a later CPU copy of its texels wait for the
/// submission that makes it; a decode alone staged (`TexelCopy::decode_only`: of what a copy on
/// the host or the staging queue wrote into a carrier, into its decoded twin) does not, as it
/// writes the twin alone (`FirstMoves::texels_wait`).
#[test]
fn a_staged_decode_alone_leaves_the_texels_to_cpu_copies() {
    let firsts = FirstMoves::default();
    let decoding: Arc<u32> = Arc::new(0);
    let copied: Arc<u32> = Arc::new(1);
    let keeper = |image: &Arc<u32>| -> Weak<dyn Any + Send + Sync> {
        let weak: Weak<u32> = Arc::downgrade(image);
        weak
    };
    let at = |image: &Arc<u32>| Arc::as_ptr(image).cast::<()>();
    // A decode's shape, its pass aside: no copy of texels.
    let alone = TexelCopy::new(vk::Buffer::null(), 0, Vec::new(), Box::new(()));
    assert!(!firsts.register_texels(keeper(&decoding), alone));
    let copy = (vk::Image::null(), vk::BufferImageCopy::default());
    let staged = TexelCopy::new(vk::Buffer::null(), 16, vec![copy], Box::new(()));
    assert!(!firsts.register_texels(keeper(&copied), staged));
    assert!(!firsts.texels_wait(at(&decoding)));
    assert!(firsts.texels_wait(at(&copied)));
}

/// A buffer of the test's whose fill waits: it lies in no memory a submission holds.
struct NewBuffer;

impl Filled for NewBuffer {
    fn memory(&self) -> Option<Box<dyn Any + Send + Sync>> {
        Some(Box::new(()))
    }
}

/// A handle handed out takes its own first work alone (`FirstMoves::take_mine`): the fills whose
/// destination is its `VkBuffer`, or the copies into its image, leaving every other resource's
/// for the device's next submission; a take not submitted gives its work back.
#[test]
fn a_hand_out_takes_its_own_first_work_alone() {
    let firsts = FirstMoves::default();
    let (mine, other) = (vk::Buffer::from_raw(1), vk::Buffer::from_raw(2));
    let buffers: Vec<Arc<NewBuffer>> = (0..3).map(|_| Arc::new(NewBuffer)).collect();
    let keeper = |buffer: &Arc<NewBuffer>| -> Weak<dyn Filled> {
        let weak: Weak<NewBuffer> = Arc::downgrade(buffer);
        weak
    };
    let source = (vk::Buffer::from_raw(3), 0);
    for (buffer, raw) in buffers.iter().zip([mine, other, mine]) {
        assert!(!firsts.register_bytes(keeper(buffer), (raw, 0), source, 16, Box::new(())));
    }
    let images: Vec<Arc<u32>> = vec![Arc::new(0), Arc::new(1)];
    let at = |image: &Arc<u32>| Arc::as_ptr(image).cast::<()>();
    for image in &images {
        let weak: Weak<u32> = Arc::downgrade(image);
        let copy = (vk::Image::null(), vk::BufferImageCopy::default());
        let staged = TexelCopy::new(vk::Buffer::null(), 16, vec![copy], Box::new(()));
        assert!(!firsts.register_texels(weak, staged));
    }
    let buffer = |raw| Mine {
        buffer: Some(raw),
        image: None,
    };
    // Given back where its submission is not made.
    let taken = firsts.take_mine(buffer(mine));
    assert!(taken.has_first() && !taken.has_staged());
    drop(taken);
    assert!(firsts.fills_wait(mine));
    // Made: its fills went, the other buffer's and the images' copies wait.
    let held = firsts.take_mine(buffer(mine)).submitted(None);
    assert_eq!(held.len(), 2, "both fills of its VkBuffer, in order");
    assert!(!firsts.fills_wait(mine));
    assert!(firsts.fills_wait(other));
    assert!(firsts.texels_wait(at(&images[0])) && firsts.texels_wait(at(&images[1])));
    // An image's copies alone.
    let image = Mine {
        buffer: None,
        image: Some(at(&images[0])),
    };
    let taken = firsts.take_mine(image);
    assert!(!taken.has_first() && taken.has_staged());
    assert_eq!(taken.submitted(None).len(), 1);
    assert!(!firsts.texels_wait(at(&images[0])));
    assert!(firsts.texels_wait(at(&images[1])));
    assert!(firsts.fills_wait(other));
    // The device's next submission takes the rest.
    assert_eq!(firsts.take(|_| true).submitted(None).len(), 2);
    assert!(firsts.none_waiting());
}

/// A resource's staged work goes as the resource goes (`FirstMoves::forget`), not at the device's
/// next submission, with what it holds: a buffer's staged bytes and an image's staged copy each
/// hold a staging buffer, which holds the device, and a device the program let go of makes no
/// submission to take them. Another resource's work waits still, and the next submission takes
/// it.
#[test]
fn a_resource_s_staged_work_goes_with_it() {
    let firsts = FirstMoves::default();
    // What each piece of work holds, as a staging buffer holds its device.
    let device: Arc<()> = Arc::new(());
    let (gone, kept) = (vk::Buffer::from_raw(1), vk::Buffer::from_raw(2));
    let buffers: Vec<Arc<NewBuffer>> = (0..2).map(|_| Arc::new(NewBuffer)).collect();
    let source = (vk::Buffer::from_raw(3), 0);
    for (buffer, raw) in buffers.iter().zip([gone, kept]) {
        let weak: Weak<NewBuffer> = Arc::downgrade(buffer);
        let held = Box::new(Arc::clone(&device));
        assert!(!firsts.register_bytes(weak, (raw, 0), source, 16, held));
    }
    let images: Vec<Arc<u32>> = vec![Arc::new(0), Arc::new(1)];
    let at = |image: &Arc<u32>| Arc::as_ptr(image).cast::<()>();
    for image in &images {
        let weak: Weak<u32> = Arc::downgrade(image);
        let copy = (vk::Image::null(), vk::BufferImageCopy::default());
        let held = Box::new(Arc::clone(&device));
        let staged = TexelCopy::new(vk::Buffer::null(), 16, vec![copy], held);
        assert!(!firsts.register_texels(weak, staged));
    }
    assert_eq!(Arc::strong_count(&device), 5);
    firsts.forget(Arc::as_ptr(&buffers[0]).cast());
    assert_eq!(
        Arc::strong_count(&device),
        4,
        "the gone buffer's staging held"
    );
    assert!(!firsts.fills_wait(gone));
    assert!(firsts.fills_wait(kept));
    firsts.forget(at(&images[0]));
    assert_eq!(
        Arc::strong_count(&device),
        3,
        "the gone image's staging held"
    );
    assert!(!firsts.texels_wait(at(&images[0])));
    assert!(firsts.texels_wait(at(&images[1])));
    // An address no work waits for lets nothing go.
    firsts.forget(std::ptr::null());
    assert_eq!(Arc::strong_count(&device), 3);
    // The device's next submission takes the rest, and holds it until it completed.
    let held = firsts.take(|_| true).submitted(None);
    assert_eq!(held.len(), 2);
    assert!(firsts.none_waiting());
    drop(held);
    assert_eq!(Arc::strong_count(&device), 1);
}
