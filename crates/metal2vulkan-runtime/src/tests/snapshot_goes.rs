//! A reader's snapshot (`render/snapshot.rs`) is held by the command buffers whose draws take it
//! (`Texture::snapshot_texels`), the texture holding it weakly: once the command buffer that took
//! it is released, its image parks in the device's recycler while the texture and its view live on,
//! the next command buffer's reader takes that image back rather than making one, and a parked one
//! no reader takes again goes within the recycler's time (`recycle.rs`, `idle_sweep.rs`:
//! `images_evicted`).
//!
//! The texture is a guest's image cache: a private [`SIDE`]-square [`FORMAT`] render target that
//! shaders read, rendered and read through a view of its own format at a resource index, which
//! an upload fills with [`SOLID`] first. Each
//! pass loads the view, fills the whole target with [`SOLID`] (`tob_solid`) and then reads it
//! through [`READ`] at the fragments' own pixels (`tob_read`: each fragment writes the view's texel
//! under it), with no scissor, so the area rule takes the snapshot on a device that takes the
//! snapshot path; the reader covers a small share of the target, so learned coverage keeps it
//! (`render/coverage.rs`). Every pixel then holds [`SOLID`], whichever answer the reader takes:
//! what the fill wrote (a snapshot's) or what the texture held before the pass (a copy's). A
//! device without the path reads through copies, and takes no snapshot.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect`,
//! `tob_solid` and `tob_read` of the owned fixture `validation/fixtures/public/engine/tob_renders.ll`
//! (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::bytes_of;
use super::snapshot_reader_keeps::{
    corners, download, settled, state_into, target, upload, FORMAT, SIDE,
};
use crate::framebuffer_fetch::through_storage;
use crate::{
    CommandQueue, IdPlacement, LoadAction, RenderPassDescriptor, RenderPipelineState, ResourceId,
    StoreAction, Texture, WorkCounts,
};
use metal2vulkan::metal::PrimitiveType;
use metal2vulkan::pipeline::ColorAttachmentDescriptor;
use std::time::{Duration, Instant};

/// The rectangle the reader covers, `[x0, y0, x1, y1]`.
const READ: [u64; 4] = [600, 600, 664, 664];

/// The red the fill writes, which the reader writes back.
const SOLID: u8 = 96;

/// The pipeline states: the fill and the reader.
struct States {
    fill: RenderPipelineState,
    read: RenderPipelineState,
}

/// One pass over `view` (the module docs), in a command buffer of its own, completed: the
/// device's counts before and after it.
fn pass(queue: &CommandQueue, states: &States, view: &Texture) -> (WorkCounts, WorkCounts) {
    let device = queue.device();
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(view.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    let red = f32::from(SOLID) / 255.0;
    let rgba = bytes_of(&[red, 0.0, 0.0, 1.0], f32::to_le_bytes);
    encoder.set_fragment_bytes(&rgba, 0).unwrap();
    encoder.set_fragment_texture(Some(view), 0).unwrap();
    encoder.set_render_pipeline_state(&states.fill).unwrap();
    (encoder.set_vertex_bytes(&corners([0, 0, SIDE, SIDE]), 0)).unwrap();
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    encoder.set_render_pipeline_state(&states.read).unwrap();
    encoder.set_vertex_bytes(&corners(READ), 0).unwrap();
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    (before, device.work_counts())
}

/// What a pass moved: its readers through snapshots, the images it took (made or recycled) and
/// those of them the recycler handed back.
#[derive(Debug, PartialEq, Eq)]
struct Moved {
    snapshotted: u64,
    taken: u64,
    recycled: u64,
}

fn moved((before, after): &(WorkCounts, WorkCounts)) -> Moved {
    let delta = |count: fn(&WorkCounts) -> u64| count(after) - count(before);
    Moved {
        snapshotted: delta(|c| c.reading_draws_snapshotted),
        taken: delta(|c| c.object_churn.vk_images_made + c.object_churn.images_recycled),
        recycled: delta(|c| c.object_churn.images_recycled),
    }
}

#[test]
fn a_reader_s_snapshot_goes_with_its_command_buffers_not_with_the_texture() {
    let scratch = Scratch::new("snapshot-goes");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let color = || ColorAttachmentDescriptor::new(FORMAT);
    let states = States {
        fill: state_into(&device, "tob_solid", color()),
        read: state_into(&device, "tob_read", color()),
    };
    let texture = target(&device);
    let at = IdPlacement::At(ResourceId(40));
    let view = texture.new_texture_view_with_placement(FORMAT, at).unwrap();
    // `BGRA8Unorm`'s red is its third byte.
    let solid: Vec<u8> = (0..SIDE * SIDE).flat_map(|_| [0, 0, SOLID, 255]).collect();
    upload(&queue, &texture, &solid);
    // A pass first, so every pipeline its draws take is made before the passes counted.
    pass(&queue, &states, &view);
    settled(&device);
    let first = pass(&queue, &states, &view);
    let snapshots = device.shared().color_write_masks.is_some()
        && through_storage(device.compile_device().profile());
    let churn = || device.work_counts().object_churn;
    // Each wait ends as soon as the count moves: the command buffer's release, which may end on a
    // handler thread after `pass` returned, parks the snapshot's image; the recycler's idle sweep
    // destroys one parked past its time.
    let until_moved = |count: fn(&crate::ObjectChurn) -> u64, from: u64| {
        let until = Instant::now() + WAIT;
        while snapshots && count(&churn()) == from && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(1));
        }
        count(&churn()) - from
    };
    let parked_first = until_moved(|c| c.images_parked, first.0.object_churn.images_parked);
    let second = pass(&queue, &states, &view);
    let parked_second = until_moved(|c| c.images_parked, second.0.object_churn.images_parked);
    let evicted = until_moved(|c| c.images_evicted, second.1.object_churn.images_evicted);
    let reds: Vec<u8> = (download(&queue, &texture).chunks(4))
        .map(|texel| texel[2])
        .collect();
    let (first, second) = (moved(&first), moved(&second));
    eprintln!(
        "SNAPSHOT-GOES {}: snapshot path {snapshots}; first {first:?}, parked by its release \
         {parked_first}; second {second:?}, parked by its release {parked_second}, evicted since \
         {evicted}",
        device.name()
    );
    let off = (reds.iter()).position(|&red| red != SOLID);
    assert_eq!(off, None, "a pixel off the fill's red, by row");
    if !snapshots {
        assert_eq!(
            (first.snapshotted, second.snapshotted),
            (0, 0),
            "readers through snapshots on a device without the path"
        );
        return;
    }
    assert_eq!(
        first.snapshotted, 1,
        "the first pass's reader takes the snapshot"
    );
    assert!(
        parked_first >= 1,
        "the snapshot's image parks once its command buffer went, the texture alive"
    );
    let again = Moved {
        snapshotted: 1,
        taken: 1,
        recycled: 1,
    };
    assert_eq!(
        second, again,
        "the next command buffer's snapshot is the recycler's image, not one the texture kept"
    );
    assert!(
        parked_second >= 1 && evicted >= 1,
        "the last snapshot's image parks and goes within the recycler's time"
    );
}
