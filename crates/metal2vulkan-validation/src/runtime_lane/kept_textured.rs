//! A host step that keeps the chunks of draws that read textures through the heap (the runtime's
//! `render/executions/kept.rs`, **Draws that read the heap**), on the page of
//! [`replayed_textured`](super::replayed_textured) at two chunks' commands, on the device the cfg
//! selects, under the validation layer. Each command's draw reads texel (0, 0) of its own texture
//! through the texture's resource ID in an argument buffer: set 1's heap set and the device's
//! texture table.
//!
//! - [`a_host_step_keeps_draws_that_read_textures_through_the_heap`]: frame 0 records inline,
//!   frame 1 records apart and keeps both chunks, frame 2 runs them kept, drawing what frame 0
//!   drew. The host then writes every texture's texels, and frame 3 runs the chunks kept and draws
//!   what a fresh page of those texels draws: a kept recording binds the textures, not their
//!   texels. On a device without `descriptorBindingUpdateUnusedWhilePending` each command buffer
//!   binds a heap set of its own, so no step keeps a chunk, every replayed draw counted
//!   (`WorkCounts::replayed_heap_draws_unshared`).
//! - [`a_kept_heap_draw_reads_what_an_earlier_command_buffer_wrote`]: between kept steps, a
//!   command buffer committed before the frame spins (the owned fixture
//!   `kernel_spin_on_an_unraised_word`, [`Spin`]) and then copies new texels into every texture
//!   from the buffer the spin wrote (a blit, so the copies wait for the spin); on the frame's
//!   queue, then on a queue of its own (`new_command_queue_unshared`). The frame runs the chunks
//!   kept and draws the new texels: a run notes the uses its recording noted
//!   (`EncoderCore::run_kept`), so the tracker orders it after the copies as it orders a fresh
//!   recording. A run that noted none would start while the other queue still spins.
//! - [`a_sampler_state_made_between_steps_records_kept_heap_draws_again`]: a sampler state made
//!   after a kept step changes the argument samplers a heap draw lists (`render/heap_part.rs`), so
//!   the next step records both chunks again under the new key and keeps them.
//! - [`a_texture_swapped_at_a_token_records_kept_heap_draws_again`]: the execution's pass makes
//!   another texture resident in place of one command's, at its place among the resident ones,
//!   and the command's entry names it: the resident list differs, so the next step records both
//!   chunks again, drawing what a fresh page drawing so draws.
//! - [`a_texture_that_takes_a_gone_texture_s_id_is_read_by_kept_heap_draws`]: a command's
//!   texture goes and a new one takes its resource ID, which the command's entry still names, made
//!   resident at its place: the next step records both chunks again (the resident list names
//!   another texture) and draws the new texture's texels, as a fresh page of it does.

use super::replayed_frames::pixels_hash;
use super::replayed_textured::{page_texture, Page};
use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary as _;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::VkFeature;
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::DeviceOptions;
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState,
    Device, Origin, PipelineOptions, Size, Texture, WorkCounts,
};
use std::path::Path;
use std::time::{Duration, Instant};

/// The commands each page encodes and executes: two chunks' (the runtime's
/// `kept_commands::CHUNK` is 256).
const COMMANDS: usize = 512;

/// The chunks of [`COMMANDS`] commands.
const CHUNKS: u64 = 2;

/// The blue of the texels a case writes over the page's (`replayed_textured`'s are 0x80).
const SHADES: [u8; 3] = [0x20, 0x40, 0xc0];

/// What a frame moved of the kept chunks' counters: chunks run kept, chunks recorded again though
/// a recording of them was kept, and chunks recorded apart to keep
/// (`WorkCounts::kept_contexts_made`).
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

/// A frame that records every chunk again, a recording of it kept, and keeps the new one.
const AGAIN: Moved = Moved {
    kept: 0,
    again: CHUNKS,
    apart: CHUNKS,
};

/// Whether `device` binds one version of a heap set in every command buffer: it has
/// `descriptorBindingUpdateUnusedWhilePending` (the runtime's `heap_sets.rs`), without which no
/// heap draw is kept.
fn shared_heap(device: &Device) -> bool {
    let enabled = &device.compile_device().profile().enabled().features;
    enabled.contains(&VkFeature::DescriptorBindingUpdateUnusedWhilePending)
}

/// What a frame moves where the device keeps heap draws, else nothing ([`shared_heap`]).
fn on(shared: bool, moved: Moved) -> Moved {
    match shared {
        true => moved,
        false => INLINE,
    }
}

fn moved(before: &WorkCounts, after: &WorkCounts) -> Moved {
    Moved {
        kept: after.host_step_chunks_kept - before.host_step_chunks_kept,
        again: after.host_step_chunks_recorded_again - before.host_step_chunks_recorded_again,
        apart: after.kept_contexts_made - before.kept_contexts_made,
    }
}

/// The heap draws of a replay kept from being plain between `before` and `after`, by why: an
/// unshared set, writes of their own, a set that announces, a sampler table, a texture the tables
/// bind at a heap slot (which no host step's tables bind).
fn unkept(before: &WorkCounts, after: &WorkCounts) -> [u64; 5] {
    [
        after.replayed_heap_draws_unshared - before.replayed_heap_draws_unshared,
        after.replayed_heap_draws_writing - before.replayed_heap_draws_writing,
        after.replayed_heap_draws_noted - before.replayed_heap_draws_noted,
        after.replayed_heap_draws_over_sampler_tables
            - before.replayed_heap_draws_over_sampler_tables,
        after.replayed_heap_draws_table_textures - before.replayed_heap_draws_table_textures,
    ]
}

/// What the draws between `before` and `after` walked once for each instance or core of the uses
/// of what the encoder made resident, gathered and noted, the heap parts they worked out, and
/// their looks at the textures made resident for those a carrier holds
/// (`WorkCounts::resident_uses_gathered`, `resident_uses_noted`, `draw_heap_parts_made`,
/// `draw_resident_textures_looked_at`).
fn walked(before: &WorkCounts, after: &WorkCounts) -> [u64; 4] {
    [
        after.resident_uses_gathered - before.resident_uses_gathered,
        after.resident_uses_noted - before.resident_uses_noted,
        after.draw_heap_parts_made - before.draw_heap_parts_made,
        after.draw_resident_textures_looked_at - before.draw_resident_textures_looked_at,
    ]
}

/// Wait for `command_buffer`, which must complete.
fn completed(command_buffer: &CommandBuffer) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// One frame of `page` on `queue`, `swap` as [`Page::frame_with`] takes it, waited for: what it
/// moved, the heap draws kept from being plain by why ([`unkept`]), and the pixels.
fn frame(
    device: &Device,
    queue: &CommandQueue,
    page: &Page,
    swap: Option<(usize, &Texture)>,
) -> (Moved, [u64; 5], Vec<u8>) {
    let before = device.work_counts();
    completed(&page.frame_with(queue, swap));
    let after = device.work_counts();
    (
        moved(&before, &after),
        unkept(&before, &after),
        page.pixels(),
    )
}

/// Frames 0 to 2 of `page`: the first records inline, the second records apart and keeps every
/// chunk, the third runs them kept, each drawing what the first drew. Where the device keeps no
/// heap draw, each records inline, counting every replayed draw's unshared set. The second walks
/// the uses of what the encoder made resident, works the heap parts out and looks at the textures
/// made resident no more often than the first ([`walked`]): its chunks take them from one to the next, as the draws of one
/// encoder do (the runtime's `render/executions/kept.rs`). Its pixels.
fn steady(device: &Device, queue: &CommandQueue, case: &str, page: &Page) -> Vec<u8> {
    let shared = shared_heap(device);
    let mut first = None;
    let mut walks = Vec::new();
    for (at, expected) in [INLINE, KEEPS, KEPT].into_iter().enumerate() {
        let before = device.work_counts();
        let (counts, unkept, pixels) = frame(device, queue, page, None);
        walks.push(walked(&before, &device.work_counts()));
        assert_eq!(counts, on(shared, expected), "{case}: frame {at}");
        assert_eq!(
            unkept[1..],
            [0, 0, 0, 0],
            "{case}: frame {at}'s unkept heap draws"
        );
        assert_eq!(
            unkept[0] == 0,
            shared,
            "{case}: frame {at}'s draws over unshared sets"
        );
        match &first {
            None => first = Some(pixels),
            Some(first) => assert!(pixels == *first, "{case}: frame {at} draws other pixels"),
        }
    }
    assert!(
        walks[1]
            .iter()
            .zip(&walks[0])
            .all(|(apart, inline)| apart <= inline),
        "{case}: frame 1 gathered, noted, worked out and looked at {:?}, frame 0 {:?}",
        walks[1],
        walks[0]
    );
    first.unwrap()
}

/// A fresh page's first frame, its textures' texels of `shade` where it is given, its commands
/// changed by `change` before: its pixels, which no kept chunk drew.
fn fresh(
    device: &Device,
    queue: &CommandQueue,
    shade: Option<u8>,
    change: impl FnOnce(&mut Page) -> Option<(usize, Texture)>,
) -> Vec<u8> {
    let mut page = Page::new(device, COMMANDS, 3 * COMMANDS);
    if let Some(shade) = shade {
        page.shade(shade);
    }
    let swap = change(&mut page);
    let swap = swap.as_ref().map(|(at, texture)| (*at, texture));
    let (counts, _, pixels) = frame(device, queue, &page, swap);
    assert_eq!(counts, INLINE, "a fresh page's first frame");
    pixels
}

/// A device under the validation layer, a queue of it and a page of [`COMMANDS`] commands, for
/// `case`, which ends with the layer having reported nothing.
fn run(case: impl FnOnce(&Device, &CommandQueue, &mut Page)) {
    run_with(|_| {}, case)
}

/// [`run`], `adjust` applied to the options of each device it makes (`at_plan` gives them
/// their queue plan).
fn run_with(adjust: fn(&mut DeviceOptions), case: impl FnOnce(&Device, &CommandQueue, &mut Page)) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
        println!(
            "KEPT-TEXTURED {}: descriptorBindingUpdateUnusedWhilePending {}",
            device.name(),
            shared_heap(&device)
        );
        let queue = device.new_command_queue().unwrap();
        let mut page = Page::new(&device, COMMANDS, 3 * COMMANDS);
        case(&device, &queue, &mut page);
    }
    layer_reported_nothing_at_all(&messages);
}

#[test]
fn a_host_step_keeps_draws_that_read_textures_through_the_heap() {
    run(|device, queue, page| {
        let case = "texels the host wrote";
        steady(device, queue, case, page);
        page.shade(SHADES[0]);
        let (counts, unkept, pixels) = frame(device, queue, page, None);
        assert_eq!(
            counts,
            on(shared_heap(device), KEPT),
            "{case}: the frame after the write"
        );
        assert_eq!(unkept[1..], [0, 0, 0, 0], "{case}: unkept heap draws");
        assert_eq!(
            pixels_hash(&pixels),
            pixels_hash(&fresh(device, queue, Some(SHADES[0]), |_: &mut Page| None)),
            "{case}: the frame after the write draws the new texels"
        );
    });
}

/// About how long each of a [`Spin`]'s dispatches runs on the GPU.
const SLICE: Duration = Duration::from_millis(8);

/// The dispatches a [`Spin`] makes: about 40 ms of GPU time, far past what the frame's command
/// buffer takes to reach its draws.
const DISPATCHES: u32 = 5;

/// The least time a dispatch of the spin's kernel is timed over.
const MEASURED: Duration = Duration::from_millis(20);

/// The owned fixture `kernel_spin_on_an_unraised_word` (one thread loading a word nothing raises
/// until its cap of loads runs out, then storing how many it made) at the cap that takes about
/// [`SLICE`] of the device's GPU time, dispatched [`DISPATCHES`] times.
struct Spin {
    state: ComputePipelineState,
    cap: u32,
    word: Buffer,
}

impl Spin {
    /// Time the cap: single dispatches of growing caps in command buffers of their own on
    /// `queue`, from commit to completion, until one takes [`MEASURED`], scaled to [`SLICE`]. The
    /// first, not timed, makes the pipeline's `VkPipeline`.
    fn new(device: &Device, queue: &CommandQueue) -> Spin {
        let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/public/engine/kernel_spin_on_an_unraised_word.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = library
            .new_function_with_name("spin_on_an_unraised_word")
            .unwrap();
        let state = device
            .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
            .unwrap();
        let mut spin = Spin {
            state,
            cap: 1 << 10,
            word: buffer_with(device, &[0; 4]).unwrap(),
        };
        let loads = buffer_with(device, &[0; 4]).unwrap();
        let run = |spin: &Spin| {
            let command_buffer = queue.command_buffer().unwrap();
            spin.encode(&command_buffer, &loads, 1);
            let at = Instant::now();
            command_buffer.commit().unwrap();
            completed(&command_buffer);
            at.elapsed()
        };
        run(&spin);
        loop {
            let took = run(&spin);
            if took >= MEASURED || spin.cap >= 1 << 30 {
                let scaled = f64::from(spin.cap) * SLICE.as_secs_f64() / took.as_secs_f64();
                spin.cap = scaled.clamp(1.0, f64::from(u32::MAX)) as u32;
                return spin;
            }
            spin.cap <<= 2;
        }
    }

    /// Encode, in an encoder of its own, `dispatches` dispatches of one thread spinning to the
    /// cap, each storing its count at the start of `loads`.
    fn encode(&self, command_buffer: &CommandBuffer, loads: &Buffer, dispatches: u32) {
        let mut compute = command_buffer.compute_command_encoder().unwrap();
        compute.set_compute_pipeline_state(&self.state).unwrap();
        compute.set_buffer(Some(&self.word), 0, 0).unwrap();
        compute.set_buffer(Some(loads), 0, 1).unwrap();
        compute.set_bytes(&self.cap.to_le_bytes(), 2).unwrap();
        for _ in 0..dispatches {
            compute
                .dispatch_threadgroups(Size::new(1, 1, 1), Size::new(1, 1, 1))
                .unwrap();
        }
        compute.end_encoding().unwrap();
    }
}

/// Commit, on `queue`, a command buffer that spins and then copies every texel of `page`'s
/// textures, of `shade`, from the buffer the spin wrote, so the copies follow the spin. It, not
/// waited for.
fn spun_copies(
    device: &Device,
    queue: &CommandQueue,
    page: &Page,
    (spin, shade): (&Spin, u8),
) -> CommandBuffer {
    // The spin's count, then the texels, 64 bytes a texture.
    let bytes = [vec![0; 64], page.texels(shade)].concat();
    let source = buffer_with(device, &bytes).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    spin.encode(&command_buffer, &source, DISPATCHES);
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for (at, texture) in (0u64..).zip(page.textures()) {
        (blit.copy_from_buffer_to_texture(
            &source,
            64 + 64 * at,
            16,
            64,
            Size::new(4, 4, 1),
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        ))
        .unwrap();
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
}

#[test]
fn a_kept_heap_draw_reads_what_an_earlier_command_buffer_wrote() {
    run_with(at_plan, |device, queue, page| {
        let shared = shared_heap(device);
        let spin = Spin::new(device, queue);
        steady(device, queue, "copies between kept steps", page);
        let other = device.new_command_queue_unshared(64).unwrap();
        for (writer, shade, case) in [
            (queue, SHADES[1], "copies on the frame's queue"),
            (&other, SHADES[2], "copies on another queue"),
        ] {
            let copies = spun_copies(device, writer, page, (&spin, shade));
            let (counts, unkept, pixels) = frame(device, queue, page, None);
            completed(&copies);
            assert_eq!(
                counts,
                on(shared, KEPT),
                "{case}: the frame after the copies"
            );
            assert_eq!(unkept[1..], [0, 0, 0, 0], "{case}: unkept heap draws");
            assert_eq!(
                pixels_hash(&pixels),
                pixels_hash(&fresh(device, queue, Some(shade), |_: &mut Page| None)),
                "{case}: the frame after the copies draws the copied texels"
            );
        }
    });
}

#[test]
fn a_sampler_state_made_between_steps_records_kept_heap_draws_again() {
    run(|device, queue, page| {
        let case = "a sampler state made between steps";
        let shared = shared_heap(device);
        let before = steady(device, queue, case, page);
        let _sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        for (after, expected) in [(0, AGAIN), (1, KEPT)] {
            let (counts, _, pixels) = frame(device, queue, page, None);
            assert_eq!(
                counts,
                on(shared, expected),
                "{case}: frame {after} after it"
            );
            assert!(
                pixels == before,
                "{case}: frame {after} after it draws other pixels"
            );
        }
    });
}

#[test]
fn a_texture_swapped_at_a_token_records_kept_heap_draws_again() {
    run(|device, queue, page| {
        let case = "a texture swapped at a token";
        let shared = shared_heap(device);
        steady(device, queue, case, page);
        // The last command's: its draw is the last at its pixels.
        let at = COMMANDS - 1;
        let swapped = page_texture(device, at, SHADES[1]);
        page.name(at, &swapped);
        let mut drawn = None;
        for (after, expected) in [(0, AGAIN), (1, KEPT)] {
            let (counts, _, pixels) = frame(device, queue, page, Some((at, &swapped)));
            assert_eq!(
                counts,
                on(shared, expected),
                "{case}: frame {after} after it"
            );
            match &drawn {
                None => drawn = Some(pixels),
                Some(drawn) => {
                    assert!(pixels == *drawn, "{case}: frame {after} draws other pixels")
                }
            }
        }
        let swap = |fresh: &mut Page| {
            fresh.name(at, &swapped);
            Some((at, swapped.clone()))
        };
        assert_eq!(
            pixels_hash(&drawn.unwrap()),
            pixels_hash(&fresh(device, queue, None, swap)),
            "{case}: the frames after it draw what a fresh page drawing so draws"
        );
    });
}

#[test]
fn a_texture_that_takes_a_gone_texture_s_id_is_read_by_kept_heap_draws() {
    run(|device, queue, page| {
        let case = "a texture at a gone texture's ID";
        let shared = shared_heap(device);
        let before = steady(device, queue, case, page);
        // The last command's: its draw is the last at its pixels, which so show its texels.
        let at = COMMANDS - 1;
        page.replace_texture(device, at, SHADES[0]);
        let mut drawn = None;
        for (after, expected) in [(0, AGAIN), (1, KEPT)] {
            let (counts, _, pixels) = frame(device, queue, page, None);
            assert_eq!(
                counts,
                on(shared, expected),
                "{case}: frame {after} after it"
            );
            match &drawn {
                None => drawn = Some(pixels),
                Some(drawn) => {
                    assert!(pixels == *drawn, "{case}: frame {after} draws other pixels")
                }
            }
        }
        let drawn = drawn.unwrap();
        assert!(drawn != before, "{case}: the new texture's texels show");
        let replace = |fresh: &mut Page| {
            fresh.replace_texture(device, at, SHADES[0]);
            None
        };
        assert_eq!(
            pixels_hash(&drawn),
            pixels_hash(&fresh(device, queue, None, replace)),
            "{case}: the frames after it draw what a fresh page of the new texture draws"
        );
    });
}
