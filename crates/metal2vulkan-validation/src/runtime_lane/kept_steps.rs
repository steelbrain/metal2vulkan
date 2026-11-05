//! A host step that keeps the chunks it records (the runtime's `render/executions/kept.rs`), on
//! the page of [`replayed_frames`](super::replayed_frames), whose execution's records the GPU
//! writes each frame, on the device the cfg selects, under the validation layer.
//!
//! [`a_kept_host_step_records_again_where_one_input_changes`] runs a page of two chunks' commands
//! three frames under one set of inputs: the third runs both chunks the second kept, recording
//! none (`WorkCounts::host_step_chunks_kept`). It then changes one input of what a kept chunk's
//! recording read, alone, and runs frames more: the first records both chunks again
//! (`WorkCounts::host_step_chunks_recorded_again`) and draws what a page that kept nothing draws
//! under the new inputs (where the change shows in the pixels, other pixels than before); the
//! next runs the new recording kept, but where the records changed, which the changed frame
//! records inline and the next one records apart and keeps, as two steps read them alike
//! (`WorkCounts::kept_contexts_made`). Records that change at each step record inline each,
//! keeping nothing, each record read once. The inputs, a case each: the records the GPU wrote (the encoding pass binds the indices' zero
//! half, so every index address moves); the buffer
//! the commands inherit, and its offset; the pipeline state they inherit; the encoder's scissor
//! rectangle; the resources the command buffer made resident; the buffer an index address names,
//! at the same address (the indices found among the device's live buffers, released and made
//! again, all zeros, pinned where they were); and the pages of the buffer the commands inherit
//! (a no-copy buffer rebound to other host memory), where the device imports host memory.

use super::host_memory::{not_importable, Mapping, GUEST_PAGE};
use super::replayed_frames::{draw_state, indices, pixels_hash, Inputs, Page, ZEROS_AT};
use super::{layer_reported_nothing_at_all, validated_device, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::indirect_commands::COMMAND_BYTES;
use metal2vulkan::metal::ColorWriteMask;
use metal2vulkan_runtime::{
    AddressPlacement, Buffer, BufferContents, BufferDescriptor, CommandBufferStatus, CommandQueue,
    Deallocator, Device, GpuAddress, HostMemory, RenderPipelineState, ResourceOptions, ScissorRect,
    WorkCounts,
};
use std::ptr::NonNull;

/// The commands each page encodes and executes: two chunks' (the runtime's
/// `kept_commands::CHUNK` is 256).
const DRAWS: u32 = 512;

/// The chunks of [`DRAWS`] commands.
const CHUNKS: u64 = 2;

/// The target's side ([`replayed_frames`](super::replayed_frames)'s).
const SIDE: u64 = 64;

/// Where the identity case pins the indices' buffer and the one made again after it: a 32-KiB
/// granule's first page (`address.rs`), high in the space, far from the ranges the runtime picks.
const PINNED: GpuAddress = GpuAddress(0x7000_0000_0000);

/// What a frame moved of the kept chunks' counters: chunks run kept, chunks recorded again though
/// a recording of them was kept, and chunks recorded apart to keep (`WorkCounts::kept_contexts_made`).
#[derive(Debug, PartialEq)]
struct Moved {
    kept: u64,
    again: u64,
    apart: u64,
}

/// A frame that keeps nothing and runs nothing kept: one that records inline.
const INLINE: Moved = Moved {
    kept: 0,
    again: 0,
    apart: 0,
};

/// A frame that records every chunk apart and keeps it.
const KEEPS: Moved = Moved {
    kept: 0,
    again: 0,
    apart: CHUNKS,
};

/// A frame that runs every chunk kept.
const KEPT: Moved = Moved {
    kept: CHUNKS,
    again: 0,
    apart: 0,
};

/// One frame of `page` on `queue` under `inputs`, waited for: what it moved, and the target's
/// pixels after it.
fn frame(device: &Device, queue: &CommandQueue, page: &Page, inputs: &Inputs) -> (Moved, Vec<u8>) {
    let before = device.work_counts();
    let command_buffer = page.frame_with(queue, inputs);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let after = device.work_counts();
    (moved(&before, &after), page.pixels())
}

fn moved(before: &WorkCounts, after: &WorkCounts) -> Moved {
    Moved {
        kept: after.host_step_chunks_kept - before.host_step_chunks_kept,
        again: after.host_step_chunks_recorded_again - before.host_step_chunks_recorded_again,
        apart: after.kept_contexts_made - before.kept_contexts_made,
    }
}

/// Frames 0 to 2 of `page` under `base`: the first records inline, the second, which reads the
/// records alike, records apart and keeps every chunk, and the third runs them kept. Its pixels.
fn steady(
    device: &Device,
    queue: &CommandQueue,
    case: &str,
    page: &Page,
    base: &Inputs,
) -> Vec<u8> {
    let expected = [INLINE, KEEPS, KEPT];
    let mut pixels = Vec::new();
    for (at, expected) in expected.into_iter().enumerate() {
        let (counts, drawn) = frame(device, queue, page, base);
        assert_eq!(counts, expected, "{case}: frame {at}");
        pixels = drawn;
    }
    pixels
}

/// A fresh page's first frame under `inputs`, which keeps nothing: its pixels.
fn fresh(device: &Device, queue: &CommandQueue, case: &str, inputs: &Inputs) -> Vec<u8> {
    let fresh = Page::new(device, DRAWS);
    let (counts, pixels) = frame(device, queue, &fresh, inputs);
    assert_eq!(counts, INLINE, "{case}: a fresh page");
    pixels
}

/// Frames 3 on of `page` under `changed`, one input changed from frame 2's, which drew `before`:
/// frame 3 records every chunk again and draws what a fresh page draws, other pixels than
/// `before` where the change `shows`. Where the change is the `records`', frame 3 records inline,
/// frame 4 records apart and keeps, as two steps read the records alike, and frame 5 runs kept;
/// else frame 3 records apart and keeps at once, and frame 4 runs kept.
fn changed(
    device: &Device,
    queue: &CommandQueue,
    case: &str,
    page: &Page,
    inputs: &Inputs,
    (before, shows, records): (&[u8], bool, bool),
) {
    let (counts, pixels) = frame(device, queue, page, inputs);
    let apart = if records { 0 } else { CHUNKS };
    let expected = Moved {
        kept: 0,
        again: CHUNKS,
        apart,
    };
    assert_eq!(counts, expected, "{case}: the changed frame");
    assert_eq!(
        pixels_hash(&pixels),
        pixels_hash(&fresh(device, queue, case, inputs)),
        "{case}: the changed frame draws what a page that kept nothing draws"
    );
    assert_eq!(
        pixels != before,
        shows,
        "{case}: whether the change shows in the pixels"
    );
    let expected = match records {
        true => &[KEEPS, KEPT][..],
        false => &[KEPT][..],
    };
    for (after, expected) in (1..).zip(expected) {
        let (counts, drawn) = frame(device, queue, page, inputs);
        assert_eq!(&counts, expected, "{case}: frame {after} after the change");
        assert_eq!(
            drawn, pixels,
            "{case}: frame {after} after the change draws as the changed one"
        );
    }
}

/// Frames 3 to 6 of `page`, whose records change at each: the indices' zero half and their first
/// half in turn. The first forgets the kept recordings; each records inline, keeping nothing, from
/// its records copied once (`WorkCounts::indirect_record_bytes_copied`), the copy the next step
/// compares with, and draws what a fresh page draws under its inputs.
fn changing(device: &Device, queue: &CommandQueue, page: &Page) {
    let case = "records that change at each step";
    let zeros = Inputs {
        indices: Some((page.indices(), ZEROS_AT)),
        ..Inputs::default()
    };
    let first = Inputs::default();
    for at in 3..7 {
        let inputs = if at % 2 == 1 { &zeros } else { &first };
        let before = device.work_counts().indirect_record_bytes_copied;
        let (counts, pixels) = frame(device, queue, page, inputs);
        let copied = device.work_counts().indirect_record_bytes_copied - before;
        let again = if at == 3 { CHUNKS } else { 0 };
        let expected = Moved { again, ..INLINE };
        assert_eq!(counts, expected, "{case}: frame {at}");
        // The step reads each record once, into the copy the next step compares with.
        assert_eq!(
            copied,
            u64::from(DRAWS) * COMMAND_BYTES as u64,
            "{case}: frame {at} copies each record once"
        );
        assert_eq!(
            pixels_hash(&pixels),
            pixels_hash(&fresh(device, queue, case, inputs)),
            "{case}: frame {at} draws what a page that kept nothing draws"
        );
    }
}

/// An input a case changes, of the page's own resources and those [`Others`] holds.
#[derive(Clone, Copy, Debug)]
enum Change {
    Records,
    InheritedBuffer,
    InheritedOffset,
    State,
    Scissor,
    Resident,
}

/// What the cases change an input to.
struct Others {
    buffer: Buffer,
    masked: RenderPipelineState,
}

impl Change {
    /// What the case is named.
    fn case(self) -> &'static str {
        match self {
            Change::Records => "the records the GPU wrote",
            Change::InheritedBuffer => "the buffer the commands inherit",
            Change::InheritedOffset => "the inherited buffer's offset",
            Change::State => "the pipeline state the commands inherit",
            Change::Scissor => "the encoder's scissor rectangle",
            Change::Resident => "the resources the command buffer made resident",
        }
    }

    /// Whether the change shows in the pixels: the fixture's vertex function reads no buffer, so
    /// the inherited buffer's and the residency's do not.
    fn shows(self) -> bool {
        matches!(self, Change::Records | Change::State | Change::Scissor)
    }

    /// Whether the change is the records'.
    fn records(self) -> bool {
        matches!(self, Change::Records)
    }

    /// `page`'s inputs with this one changed.
    fn inputs<'a>(self, page: &'a Page, others: &'a Others) -> Inputs<'a> {
        let none = Inputs::default();
        match self {
            // The indices' zero half: every record's index address moves, inside the same buffer.
            Change::Records => Inputs {
                indices: Some((page.indices(), ZEROS_AT)),
                ..none
            },
            Change::InheritedBuffer => Inputs {
                inherited: Some((&others.buffer, 0)),
                ..none
            },
            Change::InheritedOffset => Inputs {
                inherited: Some((page.inherited(), 16)),
                ..none
            },
            Change::State => Inputs {
                draw: Some(&others.masked),
                ..none
            },
            Change::Scissor => Inputs {
                scissor: Some(ScissorRect {
                    x: 0,
                    y: 0,
                    width: SIDE / 2,
                    height: SIDE,
                }),
                ..none
            },
            Change::Resident => Inputs {
                resident: Some(&others.buffer),
                ..none
            },
        }
    }
}

/// The page's inputs with its indices in `buffer`, which the command buffer leaves unresident.
fn live(buffer: &Buffer) -> Inputs<'_> {
    Inputs {
        indices: Some((buffer, 0)),
        indices_unresident: true,
        ..Inputs::default()
    }
}

/// A buffer holding `bytes` pinned at [`PINNED`].
fn pinned(device: &Device, bytes: &[u8]) -> Buffer {
    let descriptor = BufferDescriptor::new(
        bytes.len() as u64,
        ResourceOptions::shared(),
        BufferContents::Copy(bytes),
    );
    let placement = AddressPlacement::At(PINNED);
    (device.new_buffer_with_descriptor(descriptor.with_placement(placement))).unwrap()
}

/// A deallocator that releases `mapping`.
fn releasing(mapping: Mapping) -> Deallocator {
    Box::new(move |_: NonNull<u8>, _: u64| drop(mapping))
}

/// A no-copy buffer of a guest page over `mapping`, which its deallocator releases; `None`,
/// printed, where the device imports no host memory.
fn over(device: &Device, mapping: Mapping) -> Option<Buffer> {
    let pointer = mapping.pointer();
    // SAFETY: the mapping holds the buffer's length until the deallocator releases it.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(pointer),
            GUEST_PAGE as u64,
            ResourceOptions::default(),
            Some(releasing(mapping)),
        )
    };
    match made {
        Ok(buffer) => Some(buffer),
        Err(error) if not_importable(&error) => {
            eprintln!("KEPT-STEPS: the device imports no host memory: {error:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

#[test]
fn a_kept_host_step_records_again_where_one_input_changes() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let others = Others {
            buffer: device.new_buffer(256, ResourceOptions::shared()).unwrap(),
            masked: draw_state(&device, ColorWriteMask::NONE),
        };
        let changes = [
            Change::Records,
            Change::InheritedBuffer,
            Change::InheritedOffset,
            Change::State,
            Change::Scissor,
            Change::Resident,
        ];
        for change in changes {
            let case = change.case();
            let page = Page::new(&device, DRAWS);
            let before = steady(&device, &queue, case, &page, &Inputs::default());
            let inputs = change.inputs(&page, &others);
            changed(
                &device,
                &queue,
                case,
                &page,
                &inputs,
                (&before, change.shows(), change.records()),
            );
        }

        let page = Page::new(&device, DRAWS);
        steady(
            &device,
            &queue,
            "records that change",
            &page,
            &Inputs::default(),
        );
        changing(&device, &queue, &page);

        // The indices' buffer, found among the device's live buffers, released and made again at
        // its address with other indices: the records are the same bytes, their address names
        // another buffer.
        let case = "the buffer an index address names";
        let page = Page::new(&device, DRAWS);
        let first = pinned(&device, &indices(37));
        let before = steady(&device, &queue, case, &page, &live(&first));
        drop(first);
        let zeros = vec![0; indices(37).len()];
        let second = pinned(&device, &zeros);
        changed(
            &device,
            &queue,
            case,
            &page,
            &live(&second),
            (&before, true, false),
        );

        // The buffer the commands inherit, over host memory, rebound to other pages.
        let case = "the pages of the buffer the commands inherit";
        if let Some(buffer) = over(&device, Mapping::anonymous(GUEST_PAGE)) {
            let page = Page::new(&device, DRAWS);
            let inherited = Inputs {
                inherited: Some((&buffer, 0)),
                ..Inputs::default()
            };
            let before = steady(&device, &queue, case, &page, &inherited);
            let mapping = Mapping::anonymous(GUEST_PAGE);
            let pointer = mapping.pointer();
            // SAFETY: the mapping holds the buffer's length until the deallocator releases it, and
            // no other thread encodes or commits meanwhile.
            unsafe {
                buffer.rebind_bytes_no_copy(
                    HostMemory::new(pointer),
                    Some(releasing(mapping)),
                    None,
                )
            }
            .unwrap();
            changed(
                &device,
                &queue,
                case,
                &page,
                &inherited,
                (&before, false, false),
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}
