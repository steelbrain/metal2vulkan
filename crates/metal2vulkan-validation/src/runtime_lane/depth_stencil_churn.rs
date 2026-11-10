//! `Depth32Float_Stencil8` textures made, used and dropped by many threads at once, under load: the
//! shape of a consumer whose tasks each make depth/stencil textures, page them plane by plane
//! through a transfer buffer over host memory with blit copies (depth with `DepthFromDepthStencil`,
//! stencil with `StencilFromDepthStencil`), render into some, and drop them as their tasks go,
//! while another queue keeps the GPU busy with long copies.
//!
//! Each thread is a task with a command queue of its own. Each round makes a texture of a side
//! from 4 to 32 texels and one or two levels (and one more it drops unused at once, as a
//! texture a validation refuses goes), a transfer buffer over fresh host pages whose deallocator
//! unmaps them (a `Shared` buffer where the device refuses host memory, typed), uploads both
//! planes of every level, runs a pass that loads or clears level 0's depth and stencil and
//! stores them, and copies both planes back into the transfer buffer: in one command buffer, or
//! the copies back in a second one on the next task's queue. A round then ends in one of three
//! ways: it waits and checks every byte read back; it drops everything just after the commits,
//! the runtime keeping what the GPU still uses; or it hands everything to a dropping thread,
//! which waits and checks every other one and drops the rest at once.
//!
//! The lane passes when every waited command buffer completed without error with the bytes
//! Metal's copies give, nothing crashed, and the validation layer reported nothing.

use super::at_plan;
use super::host_memory::on_device_with;
use super::host_memory::{no_copy, not_importable, Mapping, GUEST_PAGE};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CommandQueue, Deallocator, Device, LoadAction, Origin,
    RenderPassDescriptor, ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(120);

/// The tasks, and the rounds each runs.
const TASKS: usize = 8;
const ROUNDS: usize = 24;

/// The bytes of each of the load's copies.
const LOAD_BYTES: u64 = 32 << 20;

/// How each round ends.
#[derive(Clone, Copy, Debug)]
enum Ending {
    /// Wait for the round's command buffers and check every byte read back.
    Wait,
    /// Drop everything just after the commits.
    Drop,
    /// Hand everything to the dropping thread.
    HandOff,
}

/// One round's objects and what its copies back must read.
struct Round {
    what: String,
    command_buffers: Vec<CommandBuffer>,
    texture: Texture,
    buffer: Buffer,
    /// Where the copies back start in the transfer buffer, and the bytes they must leave there.
    back_at: usize,
    expected: Vec<u8>,
}

impl Round {
    /// Wait for the command buffers, check that each completed without error and that the
    /// copies back hold the expected bytes, then drop everything.
    fn check(self) {
        for command_buffer in &self.command_buffers {
            command_buffer
                .wait_until_completed_for(WAIT)
                .unwrap_or_else(|_| panic!("{}: a command buffer did not complete", self.what));
            assert_eq!(command_buffer.error(), None, "{}", self.what);
        }
        let contents = self
            .buffer
            .contents()
            .expect("a host-visible transfer buffer");
        // SAFETY: the transfer buffer holds `length` bytes at `contents`, and its command
        // buffers completed.
        let bytes =
            unsafe { std::slice::from_raw_parts(contents.as_ptr(), self.buffer.length() as usize) };
        let read = &bytes[self.back_at..self.back_at + self.expected.len()];
        if read != self.expected.as_slice() {
            let first = read
                .iter()
                .zip(&self.expected)
                .position(|(a, b)| a != b)
                .unwrap_or_default();
            panic!(
                "{}: the copies back differ from byte {first} of {} (read {:02x?}, expected \
                 {:02x?})",
                self.what,
                self.expected.len(),
                &read[first..(first + 16).min(read.len())],
                &self.expected[first..(first + 16).min(self.expected.len())]
            );
        }
        drop(self.texture);
    }
}

/// A level's side, bytes of depth and bytes of stencil, and where its planes start in a
/// transfer region: every depth level first, then every stencil level, each at a multiple of 4.
struct Level {
    side: u64,
    depth_at: usize,
    stencil_at: usize,
}

/// The levels of a texture of `side` and `levels` levels, and the region's length.
fn levels_of(side: u64, levels: u64) -> (Vec<Level>, usize) {
    let sides: Vec<u64> = (0..levels).map(|level| (side >> level).max(1)).collect();
    let mut at = 0usize;
    let mut depth = Vec::new();
    for &s in &sides {
        depth.push(at);
        at += (s * s * 4) as usize;
    }
    let mut out = Vec::new();
    for (&s, depth_at) in sides.iter().zip(depth) {
        at = at.next_multiple_of(4);
        out.push(Level {
            side: s,
            depth_at,
            stencil_at: at,
        });
        at += (s * s) as usize;
    }
    (out, at.next_multiple_of(16))
}

/// Level `level`'s depth texel `i` of task `task`'s round `round`, in `[0, 1)`.
fn depth_value(task: usize, round: usize, level: usize, i: usize) -> f32 {
    ((task * 31 + round * 7 + level * 3 + i) % 1024) as f32 / 1024.0
}

/// Level `level`'s stencil texel `i` of task `task`'s round `round`.
fn stencil_value(task: usize, round: usize, level: usize, i: usize) -> u8 {
    ((task * 17 + round * 5 + level + i) % 256) as u8
}

/// A transfer buffer of `len` bytes over fresh host pages whose deallocator unmaps them, or a
/// `Shared` buffer where the device refuses host memory (typed); and whether it is over host
/// memory.
fn transfer_buffer(device: &Device, len: usize) -> (Buffer, bool) {
    let len = len.next_multiple_of(GUEST_PAGE);
    let memory = Mapping::anonymous(len);
    let pointer = memory.pointer();
    let release: Deallocator = Box::new(move |_: NonNull<u8>, _: u64| drop(memory));
    // SAFETY: the mapping stays until the deallocator runs, at the buffer's last release.
    match unsafe {
        no_copy(
            device,
            pointer,
            len,
            ResourceOptions::default(),
            Some(release),
        )
    } {
        Ok(buffer) => (buffer, true),
        Err(error) if not_importable(&error) => (
            device
                .new_buffer(len as u64, ResourceOptions::shared())
                .unwrap(),
            false,
        ),
        Err(error) => panic!("a transfer buffer of {len} bytes: {error:?}"),
    }
}

/// Write `bytes` at `at` in `buffer`'s contents.
fn write(buffer: &Buffer, at: usize, bytes: &[u8]) {
    assert!(at + bytes.len() <= buffer.length() as usize);
    let contents = buffer.contents().expect("a host-visible transfer buffer");
    // SAFETY: the range lies in the buffer, and no command buffer uses it yet.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr().add(at), bytes.len())
    };
}

/// Copies of every level's planes between `texture` and `buffer` from `base`: into the texture
/// where `upload`, out of it elsewhere.
fn page(
    command_buffer: &CommandBuffer,
    texture: &Texture,
    buffer: &Buffer,
    base: usize,
    levels: &[Level],
    upload: bool,
) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let planes = [
        (BlitOption::DepthFromDepthStencil, 4u64),
        (BlitOption::StencilFromDepthStencil, 1u64),
    ];
    for (option, texel) in planes {
        for (index, level) in levels.iter().enumerate() {
            let at = base
                + match option {
                    BlitOption::DepthFromDepthStencil => level.depth_at,
                    _ => level.stencil_at,
                };
            let row = level.side * texel;
            let image = row * level.side;
            let size = Size::new(level.side, level.side, 1);
            let origin = Origin::new(0, 0, 0);
            if upload {
                blit.copy_from_buffer_to_texture(
                    buffer,
                    at as u64,
                    row,
                    image,
                    size,
                    texture,
                    0,
                    index as u64,
                    origin,
                    option,
                )
                .unwrap();
            } else {
                blit.copy_from_texture_to_buffer(
                    texture,
                    0,
                    index as u64,
                    origin,
                    size,
                    buffer,
                    at as u64,
                    row,
                    image,
                    option,
                )
                .unwrap();
            }
        }
    }
    blit.end_encoding().unwrap();
}

/// A pass over level 0 of `texture`'s depth and stencil that clears them (to `clear`'s values)
/// or loads them, and stores them.
fn pass(command_buffer: &CommandBuffer, texture: &Texture, clear: Option<(f64, u32)>) {
    let mut descriptor = RenderPassDescriptor::new();
    let load = match clear {
        Some(_) => LoadAction::Clear,
        None => LoadAction::Load,
    };
    let depth = &mut descriptor.depth_attachment;
    depth.texture = Some(texture.clone());
    depth.load_action = load;
    depth.store_action = StoreAction::Store;
    let stencil = &mut descriptor.stencil_attachment;
    stencil.texture = Some(texture.clone());
    stencil.load_action = load;
    stencil.store_action = StoreAction::Store;
    if let Some((clear_depth, clear_stencil)) = clear {
        descriptor.depth_attachment.clear_depth = clear_depth;
        descriptor.stencil_attachment.clear_stencil = clear_stencil;
    }
    command_buffer
        .render_command_encoder(&descriptor)
        .unwrap()
        .end_encoding()
        .unwrap();
}

/// What the tasks did, for the lane's line.
#[derive(Default)]
struct Counts {
    made: AtomicUsize,
    checked: AtomicUsize,
    dropped: AtomicUsize,
    over_host_memory: AtomicUsize,
}

/// Task `task`'s rounds on `queues[task]` (and the next task's queue for some copies back).
fn task(
    device: &Device,
    queues: &[CommandQueue],
    task: usize,
    hand: mpsc::Sender<Round>,
    counts: &Counts,
) {
    for round in 0..ROUNDS {
        let side = 4 * (1 + ((task + round) % 8) as u64);
        let levels = 1 + (round % 2) as u64;
        let rendered = round % 3 != 1;
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::Depth32FloatStencil8, side, side, false);
        descriptor.mipmap_level_count = levels;
        descriptor.usage = match rendered {
            true => TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            false => TextureUsage::SHADER_READ,
        };
        descriptor.resource_options = ResourceOptions::private();
        // A texture made and dropped unused, as one a validation refuses goes.
        drop(device.new_texture(&descriptor).unwrap());
        let texture = device.new_texture(&descriptor).unwrap();
        counts.made.fetch_add(2, Ordering::Relaxed);

        let (levels, region) = levels_of(side, levels);
        let back_at = region;
        let (buffer, host) = transfer_buffer(device, 2 * region);
        if host {
            counts.over_host_memory.fetch_add(1, Ordering::Relaxed);
        }
        let mut uploaded = vec![0u8; region];
        for (index, level) in levels.iter().enumerate() {
            let texels = (level.side * level.side) as usize;
            for i in 0..texels {
                let depth = depth_value(task, round, index, i).to_le_bytes();
                uploaded[level.depth_at + 4 * i..level.depth_at + 4 * i + 4]
                    .copy_from_slice(&depth);
                uploaded[level.stencil_at + i] = stencil_value(task, round, index, i);
            }
        }
        write(&buffer, 0, &uploaded);
        write(&buffer, back_at, &vec![0xee; region]);
        let clear = (rendered && round % 4 == 0).then_some((0.5, (round % 256) as u32));
        let mut expected = uploaded.clone();
        if let Some((clear_depth, clear_stencil)) = clear {
            let level = &levels[0];
            for i in 0..(level.side * level.side) as usize {
                expected[level.depth_at + 4 * i..level.depth_at + 4 * i + 4]
                    .copy_from_slice(&(clear_depth as f32).to_le_bytes());
                expected[level.stencil_at + i] = clear_stencil as u8;
            }
        }
        // Bytes between the planes stay as written.
        for (index, byte) in expected.iter_mut().enumerate() {
            let in_plane = levels.iter().any(|level| {
                let texels = (level.side * level.side) as usize;
                (level.depth_at..level.depth_at + 4 * texels).contains(&index)
                    || (level.stencil_at..level.stencil_at + texels).contains(&index)
            });
            if !in_plane {
                *byte = 0xee;
            }
        }

        let first = queues[task].command_buffer().unwrap();
        page(&first, &texture, &buffer, 0, &levels, true);
        if rendered {
            pass(&first, &texture, clear);
        }
        let mut command_buffers = vec![first];
        if round % 4 == 3 {
            let other = &queues[(task + 1) % queues.len()];
            command_buffers.push(other.command_buffer().unwrap());
        }
        page(
            command_buffers.last().unwrap(),
            &texture,
            &buffer,
            back_at,
            &levels,
            false,
        );
        for command_buffer in &command_buffers {
            command_buffer.commit().unwrap();
        }
        let ending = match round % 3 {
            0 => Ending::Wait,
            1 => Ending::Drop,
            _ => Ending::HandOff,
        };
        let done = Round {
            what: format!(
                "task {task} round {round} ({side}x{side}, {} levels, {ending:?}{}{})",
                levels.len(),
                if rendered { ", rendered" } else { "" },
                if command_buffers.len() > 1 {
                    ", copied back on the next queue"
                } else {
                    ""
                }
            ),
            command_buffers,
            texture,
            buffer,
            back_at,
            expected,
        };
        match ending {
            Ending::Wait => {
                done.check();
                counts.checked.fetch_add(1, Ordering::Relaxed);
            }
            Ending::Drop => {
                drop(done);
                counts.dropped.fetch_add(1, Ordering::Relaxed);
            }
            Ending::HandOff => hand.send(done).unwrap(),
        }
    }
}

/// Depth/stencil textures made, paged, rendered and dropped by eight tasks at once while another
/// queue copies 32 MiB at a time (the module docs).
#[test]
fn depth_stencil_textures_churn_across_tasks_under_load() {
    on_device_with(at_plan, |device| {
        let queues: Vec<CommandQueue> = (0..TASKS)
            .map(|_| device.new_command_queue().unwrap())
            .collect();
        let counts = Counts::default();
        let finished = AtomicBool::new(false);
        let loads = AtomicUsize::new(0);
        let (hand, handed) = mpsc::channel::<Round>();
        std::thread::scope(|scope| {
            // The load: long copies on a queue of its own until the tasks finish.
            let load_queue = device.new_command_queue().unwrap();
            let (finished, loads) = (&finished, &loads);
            scope.spawn(move || {
                let from = device
                    .new_buffer(LOAD_BYTES, ResourceOptions::private())
                    .unwrap();
                let to = device
                    .new_buffer(LOAD_BYTES, ResourceOptions::private())
                    .unwrap();
                let mut last: Option<CommandBuffer> = None;
                while !finished.load(Ordering::Acquire) {
                    let command_buffer = load_queue.command_buffer().unwrap();
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    for _ in 0..4 {
                        blit.copy_from_buffer_to_buffer(&from, 0, &to, 0, LOAD_BYTES)
                            .unwrap();
                    }
                    blit.end_encoding().unwrap();
                    command_buffer.commit().unwrap();
                    if let Some(previous) = last.replace(command_buffer) {
                        previous.wait_until_completed_for(WAIT).unwrap();
                        assert_eq!(previous.error(), None, "a load copy");
                    }
                    loads.fetch_add(1, Ordering::Relaxed);
                }
                if let Some(last) = last {
                    last.wait_until_completed_for(WAIT).unwrap();
                }
            });
            // The dropping thread: every other round handed to it waited and checked, the rest
            // dropped at once.
            let counts = &counts;
            scope.spawn(move || {
                for (index, round) in handed.into_iter().enumerate() {
                    if index % 2 == 0 {
                        round.check();
                        counts.checked.fetch_add(1, Ordering::Relaxed);
                    } else {
                        drop(round);
                        counts.dropped.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
            let tasks: Vec<_> = (0..TASKS)
                .map(|index| {
                    let (hand, queues) = (hand.clone(), &queues);
                    scope.spawn(move || task(device, queues, index, hand, counts))
                })
                .collect();
            drop(hand);
            let outcomes: Vec<_> = tasks.into_iter().map(|task| task.join()).collect();
            finished.store(true, Ordering::Release);
            for outcome in outcomes {
                if let Err(panic) = outcome {
                    std::panic::resume_unwind(panic);
                }
            }
        });
        eprintln!(
            "D32S8 churn on {}: {} textures made by {TASKS} tasks of {ROUNDS} rounds, {} rounds \
             waited and checked, {} dropped in flight, {} over host memory; {} load command \
             buffers of {} MiB copies",
            device.name(),
            counts.made.load(Ordering::Relaxed),
            counts.checked.load(Ordering::Relaxed),
            counts.dropped.load(Ordering::Relaxed),
            counts.over_host_memory.load(Ordering::Relaxed),
            loads.load(Ordering::Relaxed),
            4 * (LOAD_BYTES >> 20),
        );
    });
}
