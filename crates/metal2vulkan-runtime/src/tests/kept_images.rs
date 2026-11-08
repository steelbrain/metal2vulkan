//! A queue's kept images (`texture/kept_image.rs`): a keep takes one of its view and extent that
//! no lease holds, the one taken last, whatever made it, and says whether another holder (a
//! command buffer's merge) still has it; at most two free of one view and extent stay, the one
//! taken least recently going first, and none untaken over 256 takes; one let go of while another
//! holder has it goes only once that holder lets go.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch};
use metal2vulkan::bind::ShadowTexelView;

const EXTENT: [u32; 2] = [64, 32];
const OTHER: [u32; 2] = [64, 16];
const TEXEL: ShadowTexelView = ShadowTexelView::R32Uint;

#[test]
fn a_queue_keeps_its_kept_images_for_its_next_keeps() {
    let scratch = Scratch::new("kept-images");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let place = queue.serving();
    let images = place.kept_images();
    let live = || device.work_counts().shadow_merge_kept_images_live;
    let make = || device.new_kept_image(TEXEL, EXTENT).unwrap();
    // None kept: a keep makes one, held by its lease; held, the next keep makes another.
    assert!(images.take(TEXEL, EXTENT).is_none());
    let first = make();
    let lease = images.keep(first.clone());
    assert!(images.take(TEXEL, EXTENT).is_none());
    let second = make();
    let second_lease = images.keep(second.clone());
    assert_eq!(live(), 2);
    // Its lease gone, the first is taken by the next keep of its view and extent alone, and
    // `first`, held here as a command buffer's merge holds it, makes it followed.
    drop(lease);
    assert!(images.take(TEXEL, OTHER).is_none());
    assert!(images.take(ShadowTexelView::R32G32Uint, EXTENT).is_none());
    let taken = images.take(TEXEL, EXTENT).unwrap();
    assert_eq!(taken.image.raw(), first.raw());
    assert!(taken.followed);
    // Its holders gone, the next take is not followed; of two free, the one taken last.
    drop((taken, first, second));
    drop(second_lease);
    let taken = images.take(TEXEL, EXTENT).unwrap();
    assert!(!taken.followed);
    let last = taken.image.raw();
    drop(taken);
    assert_eq!(images.take(TEXEL, EXTENT).unwrap().image.raw(), last);
    // A third free one of the extent: the one taken least recently goes at the next take, and
    // only once the holder that still has it lets go.
    let held = images.take(TEXEL, EXTENT).unwrap();
    let other = images.take(TEXEL, EXTENT).unwrap();
    let third = make();
    let third_lease = images.keep(third.clone());
    assert_eq!(live(), 3);
    // The first of the three, taken least recently, held as a command buffer's merge holds it.
    let holder = held.image.clone();
    drop((held, other, third, third_lease));
    assert!(images.take(TEXEL, OTHER).is_none());
    assert_eq!(live(), 3, "the image let go of is held still");
    drop(holder);
    assert_eq!(live(), 2);
    // Untaken over 256 takes, the free images go.
    for _ in 0..256 {
        assert!(images.take(TEXEL, OTHER).is_none());
    }
    assert_eq!(live(), 0);
}
