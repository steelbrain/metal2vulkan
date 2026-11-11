//! The draws of one render pass around a fetching draw, on the device the cfg selects, under the
//! validation layer. A device without rasterization-order color access fetches through the pixel
//! interlock, which leaves the fetched color out of the draw's subpass while the plain draws
//! around it need it in theirs, so the pass runs as several Vulkan render pass instances
//! (`render-pass-split`), each after the first loading and each before the last storing every
//! attachment, a barrier from everything before to everything after between them. Metal runs it
//! as one pass: each draw reads the stencil and the colors every earlier draw of the pass left.
//!
//! Each pass (`sfo_split_fetch`) clears an 8-bit color, then draws in one encoder, over that color
//! (`Load`, `Store`), a two-float color and a stencil both loaded and stored `DontCare`, of private
//! or memoryless storage ([`Storage`]): a fullscreen `sfo_plane` writing code 7, the stencil 1
//! (`Replace`) and the second color, the pass's own number in its first float (the draw's base
//! instance); [`IMAGES`] draws of [`RECTS`] rectangles each (`sfo_image`, a code per rectangle,
//! the stencil incremented, `IncrementClamp`); one draw of [`FILTERS`] overlapping rectangles
//! where the stencil is above 2 that fetches both colors (`sfo_filter`, a map that does not
//! commute, plus the pass's number); and a fullscreen `sfo_cover` writing 200 where the stencil is
//! still 1. So a pixel holds [`expected`] only where every draw read the stencil and the colors
//! the draws before it left, through the instances. [`RUNS`] passes of each target and storage over 224x128 and 61x37,
//! the target kept across passes (fetched in place once the device remade its image with
//! `STORAGE`, `fetched-target-storage`) or new each pass (fetched through a scratch image,
//! `fetch-through-scratch`). Every pixel of every pass must hold its code; the wrong ones are
//! counted and printed per pass with their 8x4 groups, and per pass the instance splits, storage
//! forms, storage placeholders and memoryless backings it counted (`WorkCounts`), the path that
//! ran. A device whose fetch takes the storage form (the pixel interlock without
//! rasterization-order color access) must split every pass.
//!
//! Those passes run one at a time. [`split_passes_in_flight_together_read_only_what_their_own_draws_left`]
//! runs them in flight together, each over textures of its own, so that nothing orders one after
//! another but the backing a queue's passes share for their second color, and counts that
//! backing's takes while another command buffer that took it had not completed.

use super::framebuffer_fetch::fetches;
use super::memoryless::memoryless_options;
use super::{
    at_plan, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    CompareFunction, PixelFormat, PrimitiveType, StencilOperation, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBuffer, DepthStencilState, Device, Library, LoadAction,
    Origin, PipelineOptions, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
};
use std::collections::BTreeSet;

/// The plain draws of rectangles between the plane and the fetch.
const IMAGES: u32 = 4;

/// The rectangles of each plain draw: draw `d` (from 0) draws `k` from `RECTS (d + 1)`.
const RECTS: u32 = 16;

/// The fetching draw's first rectangle and its count.
const FILTER_BASE: u32 = RECTS * (IMAGES + 1);
const FILTERS: u32 = 16;

/// The passes of each target, storage and size.
const RUNS: usize = 16;

/// `sfo_corner`: the top left corner of rectangle `k`, in pixels from the target's top left.
fn corner(k: u32) -> (u64, u64) {
    (u64::from((k * 37 + 1) % 61), u64::from((k * 11 + 1) % 23))
}

/// Whether rectangle `k` covers pixel `(x, y)`.
fn covers(k: u32, (x, y): (u64, u64)) -> bool {
    let (left, top) = corner(k);
    left <= x && top <= y
}

/// `sfo_code`.
fn code(k: u32) -> u32 {
    (k * 29 + 11) & 255
}

/// The code pixel `(x, y)` holds after a pass whose plane wrote `plane` (the module docs).
fn expected(pixel: (u64, u64), plane: u32) -> u8 {
    let (mut stencil, mut color) = (1u32, 7u32);
    for k in RECTS..FILTER_BASE {
        if covers(k, pixel) {
            color = code(k);
            stencil = (stencil + 1).min(255);
        }
    }
    for k in FILTER_BASE..FILTER_BASE + FILTERS {
        if covers(k, pixel) && 2 < stencil {
            color = (match k.count_ones() % 2 {
                1 => color * 3 + 1,
                _ => color * 5 + 3,
            } + plane)
                & 255;
        }
    }
    if stencil == 1 {
        color = 200;
    }
    color as u8
}

/// Every pixel's code of a target of `sides` after a pass whose plane wrote `plane`. Every corner
/// lies within the first 61 columns and 23 rows, so a pixel past them holds what the last of them
/// holds.
fn want(sides: (u64, u64), plane: u32) -> Vec<u8> {
    let reach: Vec<u8> = (0..23)
        .flat_map(|y| (0..61).map(move |x| expected((x, y), plane)))
        .collect();
    (0..sides.1)
        .flat_map(|y| (0..sides.0).map(move |x| (x.min(60), y.min(22))))
        .map(|(x, y)| reach[(y * 61 + x) as usize])
        .collect()
}

/// How the second color and the stencil are stored.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Storage {
    Private,
    Memoryless,
}

/// The pass's pipelines and stencil states, in draw order: plane, image, filter, cover.
struct Draws {
    pipelines: [RenderPipelineState; 4],
    states: [DepthStencilState; 4],
}

impl Draws {
    fn new(device: &Device, library: &Library) -> Draws {
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let pipeline = |vertex: &str, fragment: &str| {
            let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
            descriptor.fragment = Some(function(fragment));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
            descriptor.color_attachments[1] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RG16Float));
            descriptor.stencil_format = Some(PixelFormat::Stencil8);
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap()
        };
        let state = |compare: CompareFunction, pass: StencilOperation| {
            let mut descriptor = DepthStencilDescriptor::default();
            for face in [
                &mut descriptor.front_face_stencil,
                &mut descriptor.back_face_stencil,
            ] {
                face.stencil_compare_function = compare;
                face.depth_stencil_pass_operation = pass;
            }
            device.new_depth_stencil_state(&descriptor).unwrap()
        };
        Draws {
            pipelines: [
                pipeline("sfo_fan", "sfo_plane"),
                pipeline("sfo_steps", "sfo_image"),
                pipeline("sfo_steps", "sfo_filter"),
                pipeline("sfo_fan", "sfo_cover"),
            ],
            states: [
                state(CompareFunction::Always, StencilOperation::Replace),
                state(CompareFunction::Always, StencilOperation::IncrementClamp),
                state(CompareFunction::Less, StencilOperation::Keep),
                state(CompareFunction::Equal, StencilOperation::Keep),
            ],
        }
    }

    /// Draw `which` (0 to 3) with stencil reference `reference`: `instances` of its primitive
    /// from base instance `base`.
    fn draw(
        &self,
        encoder: &mut RenderCommandEncoder,
        which: usize,
        (reference, base, instances): (u32, u32, u32),
    ) {
        encoder
            .set_render_pipeline_state(&self.pipelines[which])
            .unwrap();
        encoder
            .set_depth_stencil_state(&self.states[which])
            .unwrap();
        encoder.set_stencil_reference_value(reference);
        let vertices = match which {
            1 | 2 => 6,
            _ => 3,
        };
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, vertices, instances, base)
            .unwrap();
    }
}

/// A texture of `format`, `sides` and `usage`, of `options`.
fn texture(
    device: &Device,
    format: PixelFormat,
    (width, height): (u64, u64),
    (usage, options): (TextureUsage, ResourceOptions),
) -> Texture {
    let mut d = TextureDescriptor::texture_2d(format, width, height, false);
    d.usage = usage;
    d.resource_options = options;
    device.new_texture(&d).unwrap()
}

/// Encode one pass over `target` of `sides` into `command_buffer`, with the second color and
/// stencil `attachments`, its plane writing `plane`: the target cleared to code 0 first, then the
/// draws.
fn encode_pass(
    command_buffer: &CommandBuffer,
    draws: &Draws,
    (target, attachments): (&Texture, &(Texture, Texture)),
    (sides, plane): ((u64, u64), u32),
) {
    let mut clear = RenderPassDescriptor::new();
    let color = &mut clear.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 1.0);
    color.store_action = StoreAction::Store;
    let encoder = command_buffer.render_command_encoder(&clear).unwrap();
    encoder.end_encoding().unwrap();
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    let cover = &mut descriptor.color_attachments[1];
    cover.texture = Some(attachments.0.clone());
    cover.load_action = LoadAction::DontCare;
    cover.store_action = StoreAction::DontCare;
    let stencil = &mut descriptor.stencil_attachment;
    stencil.texture = Some(attachments.1.clone());
    stencil.load_action = LoadAction::DontCare;
    stencil.store_action = StoreAction::DontCare;
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    let size = [sides.0 as f32, sides.1 as f32];
    encoder
        .set_vertex_bytes(&size.map(f32::to_le_bytes).concat(), 0)
        .unwrap();
    draws.draw(&mut encoder, 0, (1, plane, 1));
    for d in 0..IMAGES {
        draws.draw(&mut encoder, 1, (0, RECTS * (d + 1), RECTS));
    }
    draws.draw(&mut encoder, 2, (2, FILTER_BASE, FILTERS));
    draws.draw(&mut encoder, 3, (1, 0, 1));
    encoder.end_encoding().unwrap();
}

/// Encode a copy of `target` of `sides` into `command_buffer`, to a shared buffer made for it.
fn encode_readback(
    command_buffer: &CommandBuffer,
    target: &Texture,
    (width, height): (u64, u64),
) -> Buffer {
    let bytes = width * height * 4;
    let buffer = target
        .device()
        .new_buffer(bytes, ResourceOptions::shared())
        .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(width, height, 1),
        &buffer,
        0,
        width * 4,
        bytes,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    buffer
}

/// One pass over `target` of `sides`, its plane writing `plane`, in a command buffer of a queue of
/// its own, waited for; every pixel's code after it.
fn pass(
    device: &Device,
    draws: &Draws,
    (target, attachments): (&Texture, &(Texture, Texture)),
    (sides, plane): ((u64, u64), u32),
) -> Vec<u8> {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    encode_pass(
        &command_buffer,
        draws,
        (target, attachments),
        (sides, plane),
    );
    let buffer = encode_readback(&command_buffer, target, sides);
    command_buffer.commit().unwrap();
    completed(&command_buffer);
    codes(&buffer, sides)
}

/// `command_buffer` completed without error.
fn completed(command_buffer: &CommandBuffer) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// Every pixel's code a readback of a target of `sides` holds: its red, green and blue bytes, which
/// must agree, and alpha 255.
fn codes(buffer: &Buffer, (width, height): (u64, u64)) -> Vec<u8> {
    let texels = read_buffer(buffer, 0, width * height * 4).unwrap();
    texels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&[b, g, r, a]| match (b == g && g == r, a) {
            (true, 255) => b,
            // A pixel whose channels disagree holds no code; 1 is none any pixel expects
            // (`expected` gives 7, 200, a rectangle's code or a map's).
            _ => 1,
        })
        .collect()
}

/// The 8x4 groups of `sides` that hold a wrong pixel, by their top left pixel.
fn wrong_groups(wrong: &[bool], (width, _): (u64, u64)) -> BTreeSet<(u64, u64)> {
    wrong
        .iter()
        .enumerate()
        .filter(|(_, &w)| w)
        .map(|(i, _)| (i as u64 % width / 8 * 8, i as u64 / width / 4 * 4))
        .collect()
}

/// Every pass's pixels hold the codes of its draws in order (the module docs).
#[test]
fn a_split_pass_s_draws_read_the_stencil_and_colors_the_draws_before_left() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
    if !fetches(&device) {
        return;
    }
    let ll = engine_dir().join("sfo_split_fetch.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let draws = Draws::new(&device, &library);
    let fragment = device.compile_device().profile().fragment();
    let interlocked = fragment.pixel_interlock && !fragment.rasterization_order_color_access;
    let target_usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let mut wrong_passes = Vec::new();
    for sides in [(224u64, 128u64), (61, 37)] {
        let mut kept = None;
        for storage in [Storage::Private, Storage::Memoryless] {
            let options = match storage {
                Storage::Private => ResourceOptions::private(),
                Storage::Memoryless => memoryless_options(),
            };
            let attachments = (
                texture(
                    &device,
                    PixelFormat::RG16Float,
                    sides,
                    (TextureUsage::RENDER_TARGET, options),
                ),
                texture(
                    &device,
                    PixelFormat::Stencil8,
                    sides,
                    (TextureUsage::RENDER_TARGET, options),
                ),
            );
            for fresh in [false, true] {
                for run in 0..RUNS {
                    let target = match (fresh, &kept) {
                        (false, Some(target)) => Texture::clone(target),
                        _ => texture(
                            &device,
                            PixelFormat::BGRA8Unorm,
                            sides,
                            (target_usage, ResourceOptions::private()),
                        ),
                    };
                    if !fresh {
                        kept = Some(target.clone());
                    }
                    let counted = device.work_counts();
                    let plane = run as u32 + 1;
                    let want = want(sides, plane);
                    let codes = pass(&device, &draws, (&target, &attachments), (sides, plane));
                    let counts = device.work_counts();
                    let splits = [
                        counts.instance_splits_shape - counted.instance_splits_shape,
                        counts.instance_splits_interlocked_writes
                            - counted.instance_splits_interlocked_writes,
                        counts.instance_splits_barrier - counted.instance_splits_barrier,
                        counts.instance_splits_attached_read
                            - counted.instance_splits_attached_read,
                    ];
                    let inline = counts.inline_split_instances - counted.inline_split_instances;
                    let made = [
                        counts.storage_forms_taken - counted.storage_forms_taken,
                        counts.storage_placeholder_attachments
                            - counted.storage_placeholder_attachments,
                        counts.kept_backings_made - counted.kept_backings_made,
                        counts.pooled_backings_made - counted.pooled_backings_made,
                    ];
                    let wrong: Vec<bool> = codes.iter().zip(&want).map(|(c, w)| c != w).collect();
                    let count = wrong.iter().filter(|&&w| w).count();
                    let groups = wrong_groups(&wrong, sides);
                    let first: Vec<_> = groups.iter().take(6).collect();
                    eprintln!(
                        "SPLIT-FETCH on {}: {}x{} {storage:?} fresh {fresh}, pass {run}: {count} \
                         of {} pixels wrong, in {} 8x4 groups, first {first:?}; splits (shape, \
                         interlocked writes, barrier, attached read) {splits:?}, {inline} \
                         instance(s) after them inline; storage forms, placeholder attachments, \
                         kept and pooled backings {made:?}",
                        device.name(),
                        sides.0,
                        sides.1,
                        codes.len(),
                        groups.len(),
                    );
                    assert!(
                        !interlocked || splits.iter().sum::<u64>() > 0,
                        "{}: a pass fetching through the interlock did not split",
                        device.name()
                    );
                    if count > 0 {
                        wrong_passes.push((sides, storage, fresh, run, count, groups.len()));
                    }
                }
            }
        }
    }
    assert!(
        wrong_passes.is_empty(),
        "{}: passes whose draws did not read what the draws before them left (sides, storage, \
         fresh, pass, pixels, groups): {wrong_passes:?}",
        device.name()
    );
    drop(draws);
    drop(device);
    layer_reported_nothing_at_all(&messages);
}

/// Where an in-flight round commits its passes.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Queues {
    /// Every pass on one queue.
    One,
    /// The passes on two queues in turn, nothing ordering one queue's after the other's.
    Two,
}

/// How an in-flight round's targets are fetched.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Route {
    /// New targets, each fetched through a scratch image of its pass (`fetch-through-scratch`).
    Scratch,
    /// Targets that took their storage form in a pass before the rounds, fetched in place.
    InPlace,
}

/// When an in-flight round commits its passes.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Commit {
    /// Each as soon as it is encoded.
    Each,
    /// All of them once every one is encoded, back to back.
    All,
}

/// The passes of an in-flight round, each over textures of its own.
const IN_FLIGHT: usize = 16;

/// The in-flight rounds of each size, storage, route, queues and commit.
const ROUNDS: usize = 2;

/// Passes in flight together: every pass of a round over a target, a second color and a stencil
/// of its own, its readback in its command buffer, committed with no wait between them ([`Commit`])
/// on one queue or two ([`Queues`]), the targets new ([`Route::Scratch`]) or fetched in place
/// ([`Route::InPlace`]). Nothing orders one pass after another but the backing the passes of a
/// queue share for their second color (its private form stored `DontCare`, `kept_bytes.rs`, or its
/// memoryless form, `attachment_stand_in::pooled_backing`), which the hazard tracker orders as any
/// texture. Each pass's plane writes a number of its own into that color, which its fetching draw
/// reads, so a pass that read what another left in a shared backing holds codes of the other's
/// number. Each pass counts the backings it was handed (`WorkCounts::pooled_backings_made`,
/// `pooled_backings_taken`) and those taken while another command buffer that took them had not
/// completed (`pooled_backings_taken_in_flight`), its counts read once it was encoded (a read of
/// the counts lowers what its encoders left); a round prints them beside each pass's wrong pixels
/// and 8x4 groups. Every pixel of every pass must hold its code, and the queues' passes must have
/// taken kept backings.
#[test]
fn split_passes_in_flight_together_read_only_what_their_own_draws_left() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
    if !fetches(&device) {
        return;
    }
    let ll = engine_dir().join("sfo_split_fetch.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let draws = Draws::new(&device, &library);
    let fragment = device.compile_device().profile().fragment();
    let interlocked = fragment.pixel_interlock && !fragment.rasterization_order_color_access;
    let queues = [
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    ];
    let target = |sides| {
        let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        texture(
            &device,
            PixelFormat::BGRA8Unorm,
            sides,
            (usage, ResourceOptions::private()),
        )
    };
    let mut wrong_passes = Vec::new();
    let mut taken = 0;
    for sides in [(224u64, 128u64), (61, 37)] {
        for storage in [Storage::Private, Storage::Memoryless] {
            let options = match storage {
                Storage::Private => ResourceOptions::private(),
                Storage::Memoryless => memoryless_options(),
            };
            let attachments: Vec<(Texture, Texture)> = (0..IN_FLIGHT)
                .map(|_| {
                    let usage = (TextureUsage::RENDER_TARGET, options);
                    (
                        texture(&device, PixelFormat::RG16Float, sides, usage),
                        texture(&device, PixelFormat::Stencil8, sides, usage),
                    )
                })
                .collect();
            // Each in-place target's storage form taken by a pass of its own, waited for.
            let in_place: Vec<Texture> = (0..IN_FLIGHT).map(|_| target(sides)).collect();
            for (kept, attachments) in in_place.iter().zip(&attachments) {
                pass(&device, &draws, (kept, attachments), (sides, 0));
            }
            for route in [Route::Scratch, Route::InPlace] {
                for (on, commit) in [Queues::One, Queues::Two]
                    .into_iter()
                    .flat_map(|on| [(on, Commit::Each), (on, Commit::All)])
                {
                    for round in 0..ROUNDS {
                        let targets: Vec<Texture> = match route {
                            Route::Scratch => (0..IN_FLIGHT).map(|_| target(sides)).collect(),
                            Route::InPlace => in_place.clone(),
                        };
                        let mut encoded = Vec::with_capacity(IN_FLIGHT);
                        for (k, (target, attachments)) in
                            targets.iter().zip(&attachments).enumerate()
                        {
                            let queue = match on {
                                Queues::One => &queues[0],
                                Queues::Two => &queues[k % 2],
                            };
                            let command_buffer = queue.command_buffer().unwrap();
                            let counted = device.work_counts();
                            let plane = (round * IN_FLIGHT + k + 1) as u32;
                            let textures = (target, attachments);
                            encode_pass(&command_buffer, &draws, textures, (sides, plane));
                            let buffer = encode_readback(&command_buffer, target, sides);
                            let counts = device.work_counts();
                            let splits =
                                counts.instance_splits_shape - counted.instance_splits_shape;
                            assert!(
                                !interlocked || splits > 0,
                                "{}: a pass fetching through the interlock did not split",
                                device.name()
                            );
                            let backings = [
                                counts.pooled_backings_made - counted.pooled_backings_made,
                                counts.pooled_backings_taken - counted.pooled_backings_taken,
                                counts.pooled_backings_taken_in_flight
                                    - counted.pooled_backings_taken_in_flight,
                            ];
                            if commit == Commit::Each {
                                command_buffer.commit().unwrap();
                            }
                            encoded.push((command_buffer, buffer, backings, plane));
                        }
                        if commit == Commit::All {
                            for (command_buffer, ..) in &encoded {
                                command_buffer.commit().unwrap();
                            }
                        }
                        let mut wrong_counts = Vec::with_capacity(IN_FLIGHT);
                        let mut round_backings = [0u64; 3];
                        for (k, (command_buffer, buffer, backings, plane)) in
                            encoded.iter().enumerate()
                        {
                            completed(command_buffer);
                            let codes = codes(buffer, sides);
                            let want = want(sides, *plane);
                            let wrong: Vec<bool> =
                                codes.iter().zip(&want).map(|(c, w)| c != w).collect();
                            let count = wrong.iter().filter(|&&w| w).count();
                            for (sum, add) in round_backings.iter_mut().zip(backings) {
                                *sum += add;
                            }
                            wrong_counts.push(count);
                            if count > 0 {
                                let groups = wrong_groups(&wrong, sides);
                                let first: Vec<_> = groups.iter().take(6).collect();
                                eprintln!(
                                    "IN-FLIGHT on {}: {}x{} {storage:?} {route:?} {on:?} \
                                     {commit:?}, round {round}, pass {k}: {count} of {} pixels \
                                     wrong, in {} 8x4 groups, first {first:?}; backings made, \
                                     taken, taken in flight {backings:?}",
                                    device.name(),
                                    sides.0,
                                    sides.1,
                                    codes.len(),
                                    groups.len(),
                                );
                                wrong_passes.push((sides, storage, route, on, commit, round, k));
                            }
                        }
                        taken += round_backings[1];
                        eprintln!(
                            "IN-FLIGHT on {}: {}x{} {storage:?} {route:?} {on:?} {commit:?}, \
                             round {round}: wrong pixels per pass {wrong_counts:?}; backings \
                             made, taken, taken in flight {round_backings:?}",
                            device.name(),
                            sides.0,
                            sides.1,
                        );
                    }
                }
            }
        }
    }
    assert!(
        taken > 0,
        "{}: no pass took a backing its queue kept",
        device.name()
    );
    assert!(
        wrong_passes.is_empty(),
        "{}: passes in flight together that did not read what their own draws left (sides, \
         storage, route, queues, commit, round, pass): {wrong_passes:?}",
        device.name()
    );
    drop(draws);
    drop(device);
    layer_reported_nothing_at_all(&messages);
}
