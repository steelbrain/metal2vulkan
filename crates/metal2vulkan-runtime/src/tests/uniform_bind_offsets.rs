//! Constant buffers read through uniform views at bind offsets the device's uniform descriptor
//! alignment does and does not divide, in the vertex, fragment and kernel stages: each read is
//! the bound bytes', whether the slot's `SpecKey::RangeAtBindOffset` specialization reads the
//! Metal offset alone (the range starts at the bind offset: offset 0 or 256 on every device, and
//! inline bytes) or the residual plus it (offsets 4, 20, 36, 260 and 516, which no device's
//! alignment of at least 16 divides, and every offset of a slot a state once found bound at
//! such an offset, `argument_table::AtBindViews`). A slot bound at offsets that move keeps one
//! shape and one pipeline. A state's creation makes its predicted key's residual form, every slot
//! read at its residual, and a draw whose own key is not made draws with it at once while its own
//! is made on the device's pool (`pipeline/render/residual.rs`); a compute state's creation makes
//! its predicted region's, which a dispatch takes alike (`pipeline/dispatch_making.rs`). A device
//! whose driver is not measured to load a uniform read at a constant offset ahead of the shader
//! (`DriverFact::ConstantOffsetUniformReadsPreloaded`, which no driver carries today) reads every
//! uniform view at its residual. On a device whose profile takes
//! `DriverWorkaround::PushReadsLengthenThreadPayload` a storage view bound at offset 0 reads at its
//! bind offset alone, until a command binds it elsewhere; a state's creation plans a storage view
//! bound to nothing as one bound at offset 0, and makes that key and its residual form both
//! ([`storage_views_keyed`]): `ubo_offsets_kernel`'s output, bound at offset 0, takes the
//! creation's key, and a draw binding a storage view elsewhere takes the residual form at once.
//! There a kernel's constants also have a storage twin, read where their key holds, which a
//! dispatch at an aligned offset or of inline bytes takes, the creation's key.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are those of the owned
//! fixture `validation/fixtures/public/engine/ubo_bind_offsets.ll` (`llvm-as`), for draws moved
//! along a buffer of `cof_offsets.ll` beside it, for a storage view of `buffer_end_byte_load.ll`,
//! and for draws of storage views `examples/runtime_examples.ll`'s `colored_vertex`.

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    BlitOption, Buffer, CommandQueue, ComputePipelineState, Device, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, Size,
    StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};

/// The target's side, in pixels.
const SIDE: u64 = 4;

/// The constants' floats: float `i` is `i / 255`, whose `RGBA8Unorm` code is `i`.
const FLOATS: usize = 256;

/// Where a draw reads its constants: the fragment's at byte `.0`, the vertex's at byte `.1`, of
/// the constants buffer, or both as inline bytes from those offsets of it.
#[derive(Clone, Copy, Debug)]
pub(super) enum Bound {
    Buffer(u64, u64),
    Bytes(u64, u64),
}

impl Bound {
    /// The pixel the draw writes: `c[0], c[1], c[2]` at the fragment's offset, `k[1]` at the
    /// vertex's.
    fn pixel(self) -> [u8; 4] {
        let (Bound::Buffer(f, v) | Bound::Bytes(f, v)) = self;
        let code = |at: u64| (at / 4) as u8;
        [code(f), code(f + 4), code(f + 8), code(v + 4)]
    }
}

fn library(device: &Device) -> crate::Library {
    let ll = metal2vulkan_devtools::engine_dir().join("ubo_bind_offsets.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    device.new_library_with_compile_library(library)
}

/// The fixture's render state, one `RGBA8Unorm` target.
fn state(device: &Device) -> RenderPipelineState {
    let library = library(device);
    let vertex = library
        .new_function_with_name("ubo_offsets_vertex")
        .unwrap();
    let fragment = library
        .new_function_with_name("ubo_offsets_fragment")
        .unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The threadgroup a kernel state of [`kernel_state`] requires, and so the one its creation
/// predicts a region for: 32 by 1 by 1, which the Vulkan floor's
/// `maxComputeWorkGroupInvocations` of 128 allows on any venue and the kernel runs whole.
const PREDICTED_THREADS: [u32; 3] = [32, 1, 1];

/// A compute state of `function` whose descriptor requires [`PREDICTED_THREADS`] per threadgroup.
///
/// A creation predicts a region only where the descriptor requires a threadgroup, since nothing
/// else in the request says what a dispatch will ask for (`pipeline/dispatch_making.rs`), so a
/// test of the prediction states one.
fn kernel_state(device: &Device, function: &crate::Function) -> ComputePipelineState {
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.required_threads_per_threadgroup = Some(PREDICTED_THREADS);
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

fn constants() -> Vec<u8> {
    (0..FLOATS)
        .flat_map(|i| (i as f32 / 255.0).to_le_bytes())
        .collect()
}

/// The bytes of `buffer`, which no command uses now.
fn bytes_of(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// A pass that clears a new `RGBA8Unorm` target of [`SIDE`] by [`SIDE`] and stores it, and the
/// target.
fn pass(device: &Device) -> (RenderPassDescriptor, Texture) {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    d.usage = TextureUsage::RENDER_TARGET;
    let target = device.new_texture(&d).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    (pass, target)
}

/// `device`'s work counts once its pipeline pool holds no piece (`WorkCounts::pipeline_pool_busy`):
/// a draw that took its key's residual form does not wait for its own key's making.
pub(super) fn idle(device: &Device) -> crate::WorkCounts {
    let until = std::time::Instant::now() + WAIT;
    loop {
        let counts = device.work_counts();
        if counts.pipeline_pool_busy == 0 {
            return counts;
        }
        assert!(
            std::time::Instant::now() < until,
            "the device's pipeline pool still holds {} pieces after {WAIT:?}",
            counts.pipeline_pool_busy
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// Draw each of `cases` into a target of its own in one command buffer, and check every pixel.
pub(super) fn draw(
    device: &Device,
    queue: &CommandQueue,
    state: &RenderPipelineState,
    cases: &[Bound],
) {
    let data = constants();
    let constants = device
        .new_buffer_with_bytes(&data, ResourceOptions::shared())
        .unwrap();
    let per = SIDE * SIDE * 4;
    let back = device
        .new_buffer(per * cases.len() as u64, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    for (n, case) in cases.iter().enumerate() {
        let (pass, target) = pass(device);
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(state).unwrap();
        match *case {
            Bound::Buffer(f, v) => {
                encoder.set_fragment_buffer(Some(&constants), f, 0).unwrap();
                encoder.set_vertex_buffer(Some(&constants), v, 1).unwrap();
            }
            Bound::Bytes(f, v) => {
                let from = |at: u64| &data[at as usize..at as usize + 16];
                encoder.set_fragment_bytes(from(f), 0).unwrap();
                encoder.set_vertex_bytes(from(v), 1).unwrap();
            }
        }
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &target,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            &back,
            per * n as u64,
            SIDE * 4,
            per,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let bytes = bytes_of(&back);
    for (case, pixels) in cases.iter().zip(bytes.chunks(per as usize)) {
        for (p, pixel) in pixels.chunks(4).enumerate() {
            assert_eq!(pixel, case.pixel(), "{case:?}: pixel {p}");
        }
    }
}

/// A draw reads its vertex and fragment constants at their bind offsets in either form. Draws
/// whose ranges start at their bind offsets share one pipeline; each other combination of the two
/// slots' forms, on a state of its own (a state's slot found at such an offset once is read in
/// the other form from then on), makes one of its own (the path ran), but for both slots read at
/// their residuals, the creation's residual form. A device that reads every view at its residual
/// makes none of them: every draw takes the creation's. Each making a draw began is made on the
/// device's pool, which is waited for before the pipelines made are counted.
#[test]
fn vertex_and_fragment_constants_read_at_every_bind_offset() {
    let scratch = Scratch::new("uniform-bind-offsets");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let [state, fragment, vertex, both] = [(); 4].map(|()| self::state(&device));
    let made = || idle(&device).variant_pipelines_made;
    // The pipelines each other form of a key makes: none where every view is read at its
    // residual (`pipeline::reads_views_at_bind`).
    let forms = u64::from(crate::pipeline::reads_views_at_bind(&device));
    let start = made();
    // Offsets every device's alignment divides, and inline bytes: one pipeline, which the state's
    // creation did not make (it made the residual form).
    draw(
        &device,
        &queue,
        &state,
        &[
            Bound::Buffer(0, 0),
            Bound::Buffer(256, 512),
            Bound::Bytes(4, 20),
            Bound::Bytes(260, 36),
        ],
    );
    let aligned = made();
    assert_eq!(aligned - start, forms, "pipelines made");
    // Offsets no device's alignment divides, on one slot, the other, and both, each on a state of
    // its own: a pipeline for each of the first two, and none for both, the creation's residual
    // form, and the states keep them.
    draw(&device, &queue, &fragment, &[Bound::Buffer(4, 0)]);
    draw(&device, &queue, &vertex, &[Bound::Buffer(0, 4)]);
    draw(
        &device,
        &queue,
        &both,
        &[
            Bound::Buffer(4, 20),
            Bound::Buffer(20, 36),
            Bound::Buffer(260, 516),
        ],
    );
    let other = made();
    assert_eq!(
        other - aligned,
        2 * forms,
        "one pipeline per other combination"
    );
    // Offsets some devices' alignments divide, and offsets every device's divides, on slots
    // found at another before: read in the other form, in the pipelines made.
    let moved = [
        Bound::Buffer(16, 32),
        Bound::Buffer(64, 48),
        Bound::Buffer(128, 192),
        Bound::Buffer(0, 0),
        Bound::Buffer(256, 512),
    ];
    draw(&device, &queue, &both, &moved);
    assert_eq!(
        made(),
        other,
        "pipelines made for a state's slots found moved before"
    );
    // On the state of aligned offsets, they are read in either form.
    draw(&device, &queue, &state, &moved);
}

/// A fragment constant slot bound at offsets moving through the multiples of 16 up to 256 in one
/// encoder reads each draw's bytes; a second pass through them makes no shape, settles nothing,
/// and makes and looks up no pipeline. The first makes two shapes at most: where the device's
/// view base alignment divides the first offset, 256, and not the next, which leaves the slot out
/// of those that tell it apart from then on (`argument_table::AtBindViews`). A first command
/// buffer draws the offsets once and completes, and the device's pipeline pool is waited for, so
/// their pipelines are made before the counted passes: a draw of a pipeline its device's pool still
/// makes looks the key up (ANV makes them so while the draws record, and a draw that takes its
/// key's residual form meanwhile does not wait for them).
#[test]
fn constants_bound_at_moving_offsets_keep_one_shape() {
    let scratch = Scratch::new("uniform-moving-offsets");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let state = state(&device);
    let data = constants();
    let constants = device
        .new_buffer_with_bytes(&data, ResourceOptions::shared())
        .unwrap();
    // One draw per pixel, each through a scissor of its own.
    let offsets: Vec<u64> = std::iter::once(256)
        .chain((1..16).map(|n| 16 * n))
        .collect();
    let per = SIDE * SIDE * 4;
    // `passes` passes through the offsets in one encoder, each pass's starting work counts, the
    // counts after the last, and the target's bytes once the command buffer completed.
    let run = |passes: usize| {
        let (pass, target) = pass(&device);
        let back = device.new_buffer(per, ResourceOptions::shared()).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder.set_vertex_buffer(Some(&constants), 0, 1).unwrap();
        let mut counts = Vec::new();
        for _ in 0..passes {
            counts.push(device.work_counts());
            for (p, &offset) in (0..).zip(&offsets) {
                let (x, y) = (p % SIDE, p / SIDE);
                let rect = ScissorRect {
                    x,
                    y,
                    width: 1,
                    height: 1,
                };
                encoder.set_scissor_rect(rect).unwrap();
                encoder
                    .set_fragment_buffer(Some(&constants), offset, 0)
                    .unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                    .unwrap();
            }
        }
        counts.push(device.work_counts());
        encoder.end_encoding().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &target,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            &back,
            0,
            SIDE * 4,
            per,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let bytes = bytes_of(&back);
        for (p, (pixel, &offset)) in bytes.chunks(4).zip(&offsets).enumerate() {
            let case = Bound::Buffer(offset, 0);
            assert_eq!(pixel, case.pixel(), "{case:?}: pixel {p}");
        }
        counts
    };
    let made_first = run(1);
    idle(&device);
    let shapes = made_first[1].draw_shapes_made - made_first[0].draw_shapes_made;
    assert!(shapes <= 2, "{shapes} shapes made on the first pass");
    let counts = run(2);
    let (second, after) = (&counts[1], &counts[2]);
    let made = |c: &crate::WorkCounts| {
        [
            c.draw_shapes_made,
            c.draws_settled,
            c.variant_pipelines_made,
            c.graphics_pipelines_looked_up,
        ]
    };
    assert_eq!(
        made(after),
        made(second),
        "shapes made, draws settled, pipelines made and looked up on the second pass"
    );
}

/// The render state of the owned fixture `cof_offsets.metal` that `runtime_lane::constant_offsets`
/// draws: a triangle placed by the vertex function's constant reference `q` at buffer 0 (corner
/// `q.xy`, legs `q.zw`), in the colour of the fragment function's constant reference at buffer 0,
/// one `RGBA8Unorm` target.
fn moved_state(device: &Device) -> RenderPipelineState {
    let ll = metal2vulkan_devtools::engine_dir().join("cof_offsets.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let library = device.new_library_with_compile_library(library);
    let vertex = library.new_function_with_name("cof_constant").unwrap();
    let fragment = library.new_function_with_name("cof_flat").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A vertex constant slot moved along one buffer draw by draw in one encoder, with no other call
/// between draws (the shape `runtime_lane::constant_offsets` measures), on a device that reads
/// every uniform view at its residual (`pipeline::reads_views_at_bind`), binds one descriptor
/// range at every offset (`BindState::uniform_views_at_residual`,
/// `bind::instantiate::shared_range`): each draw is moved (`WorkCounts::draws_moved`), and where
/// its set 0 is pushed it pushes no write of it, only the slot's residual; each reads its own
/// bytes, a triangle over its own pixel alone, so every pixel takes the fragment constant's colour
/// once each draw read its own. A first pass through the offsets settles the slot's shape; the
/// second is counted (in a build with debug assertions every second draw again takes every step,
/// `render/draw_again.rs`). Where the range started at the bind offset, every draw of it pushed
/// set 0 again.
#[test]
fn constants_moved_along_one_buffer_keep_their_set_zero() {
    let scratch = Scratch::new("uniform-moved-set-zero");
    let device = device(&scratch.0);
    if crate::pipeline::reads_views_at_bind(&device) {
        eprintln!("SKIPPED: the device reads uniform views at their bind offsets");
        return;
    }
    let queue = device.new_command_queue().unwrap();
    let state = moved_state(&device);
    // Draw `p`'s triangle at byte 256 p, its corner 1/8 below and left of its pixel's centre in
    // normalized device coordinates (a pixel is 2 / SIDE wide, y up), its legs 1/2 long: it covers
    // that centre (1/8 + 1/8 from the corner) and no other (each 2 / SIDE away). The colour: codes
    // 9, 99, 199 and 255, each `code / 255`.
    let pixels = SIDE * SIDE;
    let floats = |values: [f32; 4]| values.map(f32::to_le_bytes).concat();
    let mut places = vec![0u8; (256 * pixels) as usize];
    for p in 0..pixels {
        let side = SIDE as f32;
        let x = -1.0 + (2 * (p % SIDE) + 1) as f32 / side;
        let y = 1.0 - (2 * (p / SIDE) + 1) as f32 / side;
        let at = (256 * p) as usize;
        places[at..at + 16].copy_from_slice(&floats([x - 0.125, y - 0.125, 0.5, 0.5]));
    }
    let codes = [9u8, 99, 199, 255];
    let colour = floats(codes.map(|code| f32::from(code) / 255.0));
    let shared = ResourceOptions::shared();
    let places = device.new_buffer_with_bytes(&places, shared).unwrap();
    let per = SIDE * SIDE * 4;
    let (pass, target) = pass(&device);
    let back = device.new_buffer(per, shared).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    encoder.set_fragment_bytes(&colour, 0).unwrap();
    let mut counts = Vec::new();
    for _ in 0..2 {
        counts.push(device.work_counts());
        for p in 0..pixels {
            encoder
                .set_vertex_buffer(Some(&places), 256 * p, 0)
                .unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
    }
    counts.push(device.work_counts());
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        SIDE * 4,
        per,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    for (p, pixel) in bytes_of(&back).chunks(4).enumerate() {
        assert_eq!(pixel, codes, "pixel {p}");
    }
    let (second, after) = (&counts[1], &counts[2]);
    if after.draws_pushing_set_zero > second.draws_pushing_set_zero {
        assert_eq!(
            after.set_zero_writes_pushed - second.set_zero_writes_pushed,
            0,
            "set-0 writes pushed on the second pass"
        );
    }
    // What the second pass's draws did besides, where some were not moved.
    let besides = |c: &crate::WorkCounts| {
        [
            ("rebound", c.draws_rebound),
            ("as before", c.draws_as_before),
            ("unkept", c.draws_unkept),
            ("unlike kept", c.draws_unlike_kept),
            ("none kept", c.draws_none_kept),
            ("rebind stopped", c.draws_rebind_stopped),
            (
                "stopped reshaped",
                c.draws_rebind_stopped_arguments_reshaped,
            ),
            (
                "stopped unsettled",
                c.draws_rebind_stopped_arguments_unsettled,
            ),
            ("stopped slots", c.draws_rebind_stopped_arguments_slots),
            ("stopped other", c.draws_rebind_stopped_arguments_other),
            ("reshaped by offset", c.draws_rebind_reshaped_buffer_offset),
            ("settled", c.draws_settled),
        ]
    };
    let deltas: Vec<_> = (besides(after).into_iter().zip(besides(second)))
        .map(|((name, a), (_, b))| (name, a - b))
        .filter(|&(_, delta)| delta > 0)
        .collect();
    // Every draw of the second pass draws its kept draw again, moved, but every second one from
    // the first, which takes every step in a build with debug assertions.
    assert_eq!(
        after.draws_moved - second.draws_moved,
        super::draw_again::drawn_alone(pixels),
        "draws moved on the second pass; besides: {deltas:?}"
    );
}

/// A state's creation makes its predicted key's residual form, each slot read at its residual
/// (`pipeline/render/residual.rs`). With the device's pool held, so that a draw waiting for a
/// making there could not complete, a draw binding both slots at offset 0, which every device's
/// alignment divides, and a draw binding the fragment slot at 4, which none divides, both draw at
/// once with the creation's pipeline and read their bytes: their own keys' makings wait in the
/// pool, nothing waiting for them. Once the pool runs they are made, and a later draw of the
/// second key (the state reads the fragment slot at its residual from then on) takes its own. A
/// device that reads every view at its residual makes no such key
/// (`first_draws_make_no_second_pipeline_where_views_are_read_at_their_residual`).
///
/// The kept form, which the driver fact chooses
/// (`DriverFact::ConstantOffsetUniformReadsPreloaded`): no driver carries the fact today, so this
/// test skips on every venue and the test beside it runs on each. It is what a driver measured
/// with the fact would be held to.
#[test]
fn draws_take_the_residual_form_while_their_own_keys_are_made() {
    let scratch = Scratch::new("uniform-residual-form");
    let device = device(&scratch.0);
    if !crate::pipeline::reads_views_at_bind(&device) {
        eprintln!("SKIPPED: the device reads every uniform view at its residual");
        return;
    }
    let queue = device.new_command_queue().unwrap();
    let before = device.work_counts();
    let state = state(&device);
    let created = device.work_counts();
    assert_eq!(
        created.creation_pipelines_made - before.creation_pipelines_made,
        1,
        "the creation's residual form"
    );
    let hold = crate::pipeline::pool::held::hold();
    draw(
        &device,
        &queue,
        &state,
        &[Bound::Buffer(0, 0), Bound::Buffer(4, 0)],
    );
    let drawn = device.work_counts();
    drop(hold);
    assert_eq!(
        drawn.residual_form_draws - created.residual_form_draws,
        2,
        "draws that took the residual form"
    );
    assert_eq!(
        drawn.residual_form_makings - created.residual_form_makings,
        2,
        "makings begun that nothing waits for"
    );
    assert_eq!(
        drawn.creation_pipelines_drawn - created.creation_pipelines_drawn,
        1,
        "the creation's pipeline, drawn"
    );
    assert_eq!(
        drawn.variant_pipelines_made, created.variant_pipelines_made,
        "pipelines made while the pool was held"
    );
    let made = idle(&device);
    assert_eq!(
        made.variant_pipelines_made - drawn.variant_pipelines_made,
        2,
        "the draws' own keys, made on the pool"
    );
    draw(&device, &queue, &state, &[Bound::Buffer(0, 0)]);
    let later = device.work_counts();
    assert_eq!(
        later.residual_form_draws, made.residual_form_draws,
        "a later draw of a key made took the residual form"
    );
    assert_eq!(
        later.variant_pipelines_made, made.variant_pipelines_made,
        "pipelines made for a later draw of a key made"
    );
}

/// Where the device reads every uniform view at its residual (`pipeline::reads_views_at_bind`:
/// MoltenVK), a state's draws at every bind offset take the one `VkPipeline` its creation made,
/// as Metal's first draw compiles nothing (`pipeline/render/residual.rs`). With the device's pool
/// held, draws binding both slots at offset 0, the fragment slot at 4, and both as inline bytes
/// take the creation's pipeline: none takes a residual form or makes, begins or hands over a
/// pipeline, and their command buffer completes while the pool is still held; once it runs,
/// nothing more is made. Elsewhere the draws take the residual form while their own keys are
/// made (`draws_take_the_residual_form_while_their_own_keys_are_made`).
#[test]
fn first_draws_make_no_second_pipeline_where_views_are_read_at_their_residual() {
    let scratch = Scratch::new("uniform-one-form");
    let device = device(&scratch.0);
    if crate::pipeline::reads_views_at_bind(&device) {
        eprintln!("SKIPPED: the device reads uniform views at their bind offsets");
        return;
    }
    let queue = device.new_command_queue().unwrap();
    let before = idle(&device);
    let state = state(&device);
    let created = idle(&device);
    assert_eq!(
        (
            created.creation_pipelines_made - before.creation_pipelines_made,
            created.variant_pipelines_made - before.variant_pipelines_made,
        ),
        (1, 1),
        "pipelines the creation made"
    );
    let hold = crate::pipeline::pool::held::hold();
    draw(
        &device,
        &queue,
        &state,
        &[
            Bound::Buffer(0, 0),
            Bound::Buffer(4, 0),
            Bound::Bytes(0, 16),
        ],
    );
    let drawn = device.work_counts();
    drop(hold);
    assert_eq!(
        (
            drawn.creation_pipelines_drawn - created.creation_pipelines_drawn,
            drawn.residual_form_draws - created.residual_form_draws,
            drawn.residual_form_makings - created.residual_form_makings,
            drawn.pipelines_pooled - created.pipelines_pooled,
            drawn.variant_pipelines_made - created.variant_pipelines_made,
        ),
        (1, 0, 0, 0, 0),
        "the creation's pipeline drawn; residual-form draws, their makings, pieces pooled and \
         pipelines made"
    );
    let after = idle(&device);
    assert_eq!(
        after.variant_pipelines_made, created.variant_pipelines_made,
        "pipelines made once the pool ran"
    );
}

/// Whether a compute state's creation-time prediction (`pipeline/dispatch_making.rs`) can serve a
/// `dispatchThreads:`, the call Metal programs reach for. The prediction's region is a
/// `dispatchThreadgroups:` of one threadgroup of the size the descriptor requires; this dispatches
/// `dispatchThreads:` over a grid of exactly that size, so the threads per threadgroup, the
/// bindings and the specialization are the predicted ones and only the call differs. The counts
/// say which: a prediction drawn, or a region handed to the pool and a second `VkPipeline` made.
/// Where the kernel's key reads its constants at their bind offset (on a device that reads
/// uniform views so) the key is not the prediction's: the dispatch takes the prediction while its
/// own key is made on the pool. Its output bound at offset 0 is the prediction's on every device:
/// where storage views declare the key the creation makes the region's residual form beside it
/// ([`storage_views_keyed`]).
#[test]
fn a_dispatch_threads_over_the_predicted_threadgroup_reads_what_the_prediction_serves() {
    let scratch = Scratch::new("uniform-kernel-threads");
    let device = device(&scratch.0);
    let own = u64::from(keys_its_own(&device));
    let keyed = u64::from(storage_views_keyed(&device));
    let queue = device.new_command_queue().unwrap();
    let function = library(&device)
        .new_function_with_name("ubo_offsets_kernel")
        .unwrap();
    let before = idle(&device);
    let state = kernel_state(&device, &function);
    let created = idle(&device);
    assert_eq!(
        created.creation_pipelines_made - before.creation_pipelines_made,
        1 + keyed,
        "the creation made its predicted region's pipeline, and its residual form's where storage          views declare the key"
    );
    let [width, height, depth] = PREDICTED_THREADS;
    let threads = Size::new(u64::from(width), u64::from(height), u64::from(depth));
    let data = constants();
    let constants = device
        .new_buffer_with_bytes(&data, ResourceOptions::shared())
        .unwrap();
    let output = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_buffer(Some(&constants), 0, 0).unwrap();
    encoder.set_buffer(Some(&output), 0, 1).unwrap();
    // The grid is one whole threadgroup of the predicted size: no edge region.
    encoder.dispatch_threads(threads, threads).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(WAIT)
        .expect("the dispatch's command buffer waits for no making");
    assert_eq!(command_buffer.error(), None);
    let after = idle(&device);
    assert_eq!(
        (
            after.creation_pipelines_drawn - created.creation_pipelines_drawn,
            after.variant_pipelines_made - created.variant_pipelines_made,
            after.dispatch_pipelines_pooled - created.dispatch_pipelines_pooled,
        ),
        (1, own, own),
        "a `dispatchThreads:` over the predicted threadgroup takes the prediction, and makes a \
         second pipeline on the pool only where its key reads a view at its bind offset"
    );
}

/// A compute state's creation makes the `VkPipeline` of its predicted region, a threadgroup of
/// the size its descriptor requires (`pipeline/dispatch_making.rs`), as Metal compiles at
/// creation and its first dispatch compiles nothing, and where storage views declare the key
/// ([`storage_views_keyed`]) the region's residual form beside it. With the device's pool held, a
/// dispatch of
/// that size reading its constants at offset 0 and one reading them as inline bytes are recorded
/// at once, and their command buffer completes while the pool is still held: where their key
/// reads no view at its bind offset alone (`keys_its_own`) it is the creation's, and nothing is
/// made, begun or handed over; elsewhere they take the creation's pipeline, its residual form,
/// while their own key, handed to the pool once, is made there. The kernel reads its constants
/// either way.
#[test]
fn a_compute_state_s_creation_makes_its_first_dispatch_s_pipeline() {
    let scratch = Scratch::new("uniform-kernel-creation");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let function = library(&device)
        .new_function_with_name("ubo_offsets_kernel")
        .unwrap();
    let before = idle(&device);
    let state = kernel_state(&device, &function);
    let created = idle(&device);
    let keyed = u64::from(storage_views_keyed(&device));
    assert_eq!(
        (
            created.creation_pipelines_made - before.creation_pipelines_made,
            created.variant_pipelines_made - before.variant_pipelines_made,
        ),
        (1 + keyed, 1 + keyed),
        "pipelines the creation made"
    );
    let [width, height, depth] = PREDICTED_THREADS;
    let threads = Size::new(u64::from(width), u64::from(height), u64::from(depth));
    let data = constants();
    let constants = device
        .new_buffer_with_bytes(&data, ResourceOptions::shared())
        .unwrap();
    let outputs: Vec<Buffer> = (0..2)
        .map(|_| device.new_buffer(16, ResourceOptions::shared()).unwrap())
        .collect();
    let hold = crate::pipeline::pool::held::hold();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_buffer(Some(&constants), 0, 0).unwrap();
    encoder.set_buffer(Some(&outputs[0]), 0, 1).unwrap();
    encoder
        .dispatch_threadgroups(Size::new(1, 1, 1), threads)
        .unwrap();
    encoder.set_bytes(&data[8..24], 0).unwrap();
    encoder.set_buffer(Some(&outputs[1]), 0, 1).unwrap();
    encoder
        .dispatch_threadgroups(Size::new(1, 1, 1), threads)
        .unwrap();
    encoder.end_encoding().unwrap();
    let dispatched = device.work_counts();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(WAIT)
        .expect("the dispatches' command buffer waits for no making");
    assert_eq!(command_buffer.error(), None);
    drop(hold);
    let at_bind = keys_its_own(&device);
    // Where a view is read at its bind offset: the first dispatch hands its key's making to the
    // pool and takes the residual form, and the second finds that making in flight and takes the
    // residual form too.
    let (residual, makings, pooled) = match at_bind {
        true => (2, 1, 1),
        false => (0, 0, 0),
    };
    assert_eq!(
        (
            dispatched.creation_pipelines_drawn - created.creation_pipelines_drawn,
            dispatched.residual_form_draws - created.residual_form_draws,
            dispatched.residual_form_makings - created.residual_form_makings,
            dispatched.dispatch_pipelines_pooled - created.dispatch_pipelines_pooled,
            dispatched.variant_pipelines_made - created.variant_pipelines_made,
        ),
        (1, residual, makings, pooled, 0),
        "the creation's pipeline dispatched; residual-form dispatches, their makings, regions \
         pooled and pipelines made"
    );
    let after = idle(&device);
    assert_eq!(
        after.variant_pipelines_made - created.variant_pipelines_made,
        u64::from(at_bind),
        "pipelines made once the pool ran"
    );
    for (at, output) in [0, 8].into_iter().zip(&outputs) {
        let words = bytes_of(output);
        let read = |w: usize| f32::from_le_bytes(words[w * 4..w * 4 + 4].try_into().unwrap());
        let want = |i: usize| i as f32 / 255.0;
        assert_eq!(
            [read(0), read(1)],
            [want(at / 4), want(at / 4 + 2)],
            "constants from byte {at}"
        );
    }
}

/// A slot bound to nothing is read at its residual, its residual 0 (`RangeAtBindOffset` 0, as
/// `bind::plan::slots::range_at_bind_offset` has it): the key of a draw that leaves both constant
/// slots unbound is the creation's, its residual form. With the device's pool held, such a draw
/// takes the creation's pipeline itself, neither through the residual form nor by a making of its
/// own: nothing is made, handed to the pool or begun, and its command buffer completes while the
/// pool is still held. (Its pixels read unbound slots, which Metal leaves undefined: unchecked.)
#[test]
fn a_draw_that_leaves_constant_slots_unbound_takes_the_creations_key() {
    let scratch = Scratch::new("uniform-unbound");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let state = state(&device);
    let created = idle(&device);
    let hold = crate::pipeline::pool::held::hold();
    let (pass, _target) = pass(&device);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    let drawn = device.work_counts();
    assert_eq!(
        drawn.creation_pipelines_drawn - created.creation_pipelines_drawn,
        1,
        "the creation's pipeline, drawn"
    );
    assert_eq!(
        (
            drawn.residual_form_draws - created.residual_form_draws,
            drawn.residual_form_makings - created.residual_form_makings,
            drawn.pipelines_pooled - created.pipelines_pooled,
            drawn.variant_pipelines_made - created.variant_pipelines_made,
        ),
        (0, 0, 0, 0),
        "residual-form draws, their makings, pieces pooled and pipelines made"
    );
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(WAIT)
        .expect("the draw's command buffer waits for no making");
    assert_eq!(command_buffer.error(), None);
    assert_eq!(
        device.work_counts().pipeline_pool_busy,
        0,
        "nothing waits in the pool"
    );
    drop(hold);
}

/// Whether a dispatch of `ubo_offsets_kernel` with its constants at offset 0 or as inline bytes,
/// and its output at offset 0, keys apart from its creation's predicted region: on a device that
/// reads uniform views at their bind offsets (`pipeline::reads_views_at_bind`), whose constants
/// the prediction, binding nothing, reads at their residuals. Its output at offset 0 is the
/// prediction's on every device ([`storage_views_keyed`]).
fn keys_its_own(device: &Device) -> bool {
    crate::pipeline::reads_views_at_bind(device)
}

/// Whether a storage view declares `SpecKey::RangeAtBindOffset` on `device`, its profile taking
/// `DriverWorkaround::PushReadsLengthenThreadPayload`: there a state's creation plans a storage
/// view bound to nothing as one bound at offset 0, the key of a command that binds it there, and
/// makes that key's general form (its residual form, for a compute state) beside it
/// (`pipeline/predicted.rs`, `pipeline/dispatch_making.rs`), one pipeline more for a state whose
/// functions read a storage view.
pub(super) fn storage_views_keyed(device: &Device) -> bool {
    use metal2vulkan::device::DriverWorkaround;
    (device.compile_device().profile().workarounds())
        .contains(&DriverWorkaround::PushReadsLengthenThreadPayload)
}

/// A kernel reads its constants at their bind offset in either form.
#[test]
fn kernel_constants_read_at_every_bind_offset() {
    let scratch = Scratch::new("uniform-bind-offsets-kernel");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let function = library(&device)
        .new_function_with_name("ubo_offsets_kernel")
        .unwrap();
    let state = device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap();
    let data = constants();
    let constants = device
        .new_buffer_with_bytes(&data, ResourceOptions::shared())
        .unwrap();
    // `None`: inline bytes from offset 8.
    let offsets = [
        Some(0),
        Some(4),
        Some(16),
        Some(20),
        Some(64),
        Some(260),
        None,
    ];
    let outputs: Vec<Buffer> = offsets
        .iter()
        .map(|_| device.new_buffer(16, ResourceOptions::shared()).unwrap())
        .collect();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    for (offset, output) in offsets.iter().zip(&outputs) {
        match offset {
            Some(at) => encoder.set_buffer(Some(&constants), *at, 0).unwrap(),
            None => encoder.set_bytes(&data[8..24], 0).unwrap(),
        }
        encoder.set_buffer(Some(output), 0, 1).unwrap();
        encoder
            .dispatch_threadgroups(Size::new(1, 1, 1), Size::new(1, 1, 1))
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    for (offset, output) in offsets.iter().zip(&outputs) {
        let at = offset.unwrap_or(8) as usize / 4;
        let words = bytes_of(output);
        let read = |w: usize| f32::from_le_bytes(words[w * 4..w * 4 + 4].try_into().unwrap());
        let want = |i: usize| i as f32 / 255.0;
        assert_eq!(
            [read(0), read(1)],
            [want(at), want(at + 2)],
            "offset {offset:?}"
        );
    }
}

/// A kernel's constants bound at offsets the view base alignment divides (0 and 128, a multiple
/// of every device's) and as inline bytes take the key its creation predicted and made, and make
/// nothing; on a profile that takes `DriverWorkaround::PushReadsLengthenThreadPayload` that key
/// reads them through their storage twin (`DescriptorView::UniformWithStorageTwin`) at their bind
/// offset. Bound then at offset 4, which no alignment divides, they are read through the uniform
/// view at their residual: there a key of its own (the output still read at its bind offset 0),
/// taken through the creation's residual form at once while it is made on the pool, one
/// pipeline; elsewhere the creation's. With the device's pool held, each dispatch reads its bytes.
#[test]
fn kernel_constants_at_aligned_offsets_take_the_creations_key() {
    let scratch = Scratch::new("uniform-twin-kernel");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let function = library(&device)
        .new_function_with_name("ubo_offsets_kernel")
        .unwrap();
    let keyed = u64::from(storage_views_keyed(&device));
    let state = kernel_state(&device, &function);
    let created = idle(&device);
    let [width, height, depth] = PREDICTED_THREADS;
    let threads = Size::new(u64::from(width), u64::from(height), u64::from(depth));
    let data = constants();
    let constants = device
        .new_buffer_with_bytes(&data, ResourceOptions::shared())
        .unwrap();
    // `None`: inline bytes from offset 8.
    let offsets = [Some(0), Some(128), None, Some(4)];
    let outputs: Vec<Buffer> = offsets
        .iter()
        .map(|_| device.new_buffer(16, ResourceOptions::shared()).unwrap())
        .collect();
    let took = |from: &crate::WorkCounts, to: &crate::WorkCounts| {
        (
            to.residual_form_draws - from.residual_form_draws,
            to.dispatch_pipelines_pooled - from.dispatch_pipelines_pooled,
            to.variant_pipelines_made - from.variant_pipelines_made,
        )
    };
    let hold = crate::pipeline::pool::held::hold();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    let mut counts = vec![device.work_counts()];
    for (offset, output) in offsets.iter().zip(&outputs) {
        match offset {
            Some(at) => encoder.set_buffer(Some(&constants), *at, 0).unwrap(),
            None => encoder.set_bytes(&data[8..24], 0).unwrap(),
        }
        encoder.set_buffer(Some(output), 0, 1).unwrap();
        encoder
            .dispatch_threadgroups(Size::new(1, 1, 1), threads)
            .unwrap();
        counts.push(device.work_counts());
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(WAIT)
        .expect("the dispatches' command buffer waits for no making");
    assert_eq!(command_buffer.error(), None);
    drop(hold);
    assert_eq!(
        took(&counts[0], &counts[3]),
        (0, 0, 0),
        "dispatches at 0 and 128 and of inline bytes: residual-form dispatches, regions pooled, \
         pipelines made"
    );
    assert_eq!(
        took(&counts[3], &counts[4]),
        (keyed, keyed, 0),
        "a dispatch at 4: residual-form dispatches, regions pooled, pipelines made"
    );
    assert_eq!(
        idle(&device).variant_pipelines_made - created.variant_pipelines_made,
        keyed,
        "pipelines made once the pool ran"
    );
    for (offset, output) in offsets.iter().zip(&outputs) {
        let at = offset.unwrap_or(8) as usize / 4;
        let words = bytes_of(output);
        let read = |w: usize| f32::from_le_bytes(words[w * 4..w * 4 + 4].try_into().unwrap());
        let want = |i: usize| i as f32 / 255.0;
        assert_eq!(
            [read(0), read(1)],
            [want(at), want(at + 2)],
            "offset {offset:?}"
        );
    }
}

/// A kernel reads a `device` buffer through a storage view at bind offset 0, then at offsets past
/// it and at 0 again (`engine/buffer_end_byte_load.ll`: `out[0] = in[5]`), and each dispatch reads
/// the bound bytes, whichever form its key takes. On a device whose profile takes
/// `DriverWorkaround::PushReadsLengthenThreadPayload` the storage views declare
/// `SpecKey::RangeAtBindOffset`: the first dispatch's key reads both slots at their bind offsets,
/// the key its creation predicted and made beside its residual form ([`storage_views_keyed`]),
/// so it makes nothing; a bind of the source at another offset leaves its slot out from then on
/// (`argument_table::AtBindViews`), so every later dispatch, at offset 0 too, reads it at its
/// residual beside the output's at its bind offset, one pipeline more, made on the pool while
/// those dispatches take the residual form. On every other device the key is not declared, and
/// every dispatch takes the creation's pipeline: none is made.
#[test]
fn a_storage_view_reads_its_bytes_at_bind_offset_0_and_past_it() {
    let scratch = Scratch::new("storage-bind-offsets");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let ll = metal2vulkan_devtools::engine_dir().join("buffer_end_byte_load.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let function = (device.new_library_with_compile_library(library))
        .new_function_with_name("buffer_end_byte_load")
        .unwrap();
    let state = kernel_state(&device, &function);
    let row = storage_views_keyed(&device);
    let data: Vec<u8> = (0..=255).collect();
    let source = device
        .new_buffer_with_bytes(&data, ResourceOptions::shared())
        .unwrap();
    let [width, height, depth] = PREDICTED_THREADS;
    let threads = Size::new(u64::from(width), u64::from(height), u64::from(depth));
    let dispatch = |offsets: &[u64]| -> Vec<u8> {
        let outputs: Vec<Buffer> = offsets
            .iter()
            .map(|_| device.new_buffer(16, ResourceOptions::shared()).unwrap())
            .collect();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&state).unwrap();
        for (at, output) in offsets.iter().zip(&outputs) {
            encoder.set_buffer(Some(&source), *at, 0).unwrap();
            encoder.set_buffer(Some(output), 0, 1).unwrap();
            encoder
                .dispatch_threadgroups(Size::new(1, 1, 1), threads)
                .unwrap();
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        outputs.iter().map(|o| bytes_of(o)[0]).collect()
    };
    let made = || idle(&device).variant_pipelines_made;
    let start = made();
    assert_eq!(dispatch(&[0]), [5], "the source at offset 0");
    assert_eq!(made() - start, 0, "pipelines made at offset 0");
    assert_eq!(
        dispatch(&[64, 4, 0, 128]),
        [69, 9, 5, 133],
        "the source at offsets past 0, and at 0 again"
    );
    assert_eq!(
        made() - start,
        u64::from(row),
        "pipelines made once the source moved"
    );
}

/// A state whose functions read storage views bound at offset 0, the commonest bind, makes nothing
/// at its first draw or dispatch: its creation made the key such a command takes. On a profile
/// that takes `DriverWorkaround::PushReadsLengthenThreadPayload` ([`storage_views_keyed`]) that
/// key reads the views at their bind offsets, and the creation makes its general form (a compute
/// state's predicted region's residual form) beside it, two pipelines, which a command binding a
/// view elsewhere takes at once while its own is made; elsewhere the two are one key, and the
/// creation makes one. A render state of `colored_vertex` and a compute state of
/// `buffer_end_byte_load` (`out[0] = in[5]`) are each created, counted as soon as the creation
/// returns, and with the device's pool held, so a making could not land meanwhile, commanded
/// with every view at offset 0 (nothing made, handed to the pool or taken through another form,
/// the creation's pipeline taken), then with one view at another offset (on that profile the
/// general or residual form taken and its own key handed to the pool). Every command reads its
/// bytes.
#[test]
fn storage_views_at_offset_0_make_nothing_at_their_first_use() {
    let scratch = Scratch::new("storage-first-use");
    let device = device(&scratch.0);
    let keyed = u64::from(storage_views_keyed(&device));
    let queue = device.new_command_queue().unwrap();
    let shared = ResourceOptions::shared();
    let floats = |values: &[f32]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    // What a command took: the creation's pipelines, other forms, pieces pooled and pipelines made.
    let took = |from: &crate::WorkCounts, to: &crate::WorkCounts| {
        (
            to.creation_pipelines_drawn - from.creation_pipelines_drawn,
            to.residual_form_draws - from.residual_form_draws,
            (to.pipelines_pooled - from.pipelines_pooled)
                + (to.dispatch_pipelines_pooled - from.dispatch_pipelines_pooled),
            to.variant_pipelines_made - from.variant_pipelines_made,
        )
    };

    // A render state: the triangle's positions from byte 16 of `positions` or from its start.
    let before = idle(&device);
    let state = storage_views_state(&device);
    let created = device.work_counts();
    assert_eq!(
        created.creation_pipelines_made - before.creation_pipelines_made,
        1 + keyed,
        "the render creation's pipelines"
    );
    const MOVED: u64 = 16;
    let triangle: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
    let mut positions = vec![0.0f32; 4];
    positions.extend_from_slice(&triangle);
    let positions = device
        .new_buffer_with_bytes(&floats(&positions), shared)
        .unwrap();
    let start = device
        .new_buffer_with_bytes(&floats(&triangle), shared)
        .unwrap();
    let codes = [9u8, 99, 199, 255];
    let colour = codes.map(|code| f32::from(code) / 255.0);
    let colours = device
        .new_buffer_with_bytes(&floats(&colour.repeat(3)), shared)
        .unwrap();
    let (pass, target) = pass(&device);
    let per = SIDE * SIDE * 4;
    let back = device.new_buffer(per, shared).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let hold = crate::pipeline::pool::held::hold();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    encoder.set_vertex_buffer(Some(&start), 0, 0).unwrap();
    encoder.set_vertex_buffer(Some(&colours), 0, 1).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    let at_zero = device.work_counts();
    encoder
        .set_vertex_buffer(Some(&positions), MOVED, 0)
        .unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    let moved = device.work_counts();
    encoder.end_encoding().unwrap();
    drop(hold);
    assert_eq!(
        took(&created, &at_zero),
        (1, 0, 0, 0),
        "a draw of storage views at offset 0: the creation's pipeline, nothing else"
    );
    let moved = took(&at_zero, &moved);
    assert_eq!(
        (moved.1, moved.2),
        (keyed, keyed),
        "a draw of a storage view moved: its general form at once, its own key pooled, where \
         storage views declare the key"
    );
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        SIDE * 4,
        per,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    for (p, pixel) in bytes_of(&back).chunks(4).enumerate() {
        assert_eq!(pixel, codes, "pixel {p}");
    }

    // A compute state: its source at offset 0, then at 64.
    let ll = metal2vulkan_devtools::engine_dir().join("buffer_end_byte_load.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let function = (device.new_library_with_compile_library(library))
        .new_function_with_name("buffer_end_byte_load")
        .unwrap();
    let before = idle(&device);
    let state = kernel_state(&device, &function);
    let created = device.work_counts();
    assert_eq!(
        created.creation_pipelines_made - before.creation_pipelines_made,
        1 + keyed,
        "the compute creation's pipelines"
    );
    let data: Vec<u8> = (0..=255).collect();
    let source = device.new_buffer_with_bytes(&data, shared).unwrap();
    let outputs: Vec<Buffer> = (0..2)
        .map(|_| device.new_buffer(16, shared).unwrap())
        .collect();
    let [width, height, depth] = PREDICTED_THREADS;
    let threads = Size::new(u64::from(width), u64::from(height), u64::from(depth));
    let command_buffer = queue.command_buffer().unwrap();
    let hold = crate::pipeline::pool::held::hold();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    let mut counts = vec![created];
    for (at, output) in [0, 64].into_iter().zip(&outputs) {
        encoder.set_buffer(Some(&source), at, 0).unwrap();
        encoder.set_buffer(Some(output), 0, 1).unwrap();
        encoder
            .dispatch_threadgroups(Size::new(1, 1, 1), threads)
            .unwrap();
        counts.push(device.work_counts());
    }
    encoder.end_encoding().unwrap();
    drop(hold);
    assert_eq!(
        took(&counts[0], &counts[1]),
        (1, 0, 0, 0),
        "a dispatch of storage views at offset 0: the creation's pipeline, nothing else"
    );
    let moved = took(&counts[1], &counts[2]);
    assert_eq!(
        (moved.1, moved.2),
        (keyed, keyed),
        "a dispatch of a storage view moved: its residual form at once, its own key pooled, where \
         storage views declare the key"
    );
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let read: Vec<u8> = outputs.iter().map(|o| bytes_of(o)[0]).collect();
    assert_eq!(read, [5, 69], "the source at offsets 0 and 64");
}

/// The render state of `examples/runtime_examples.ll`'s `colored_vertex`, which reads each
/// vertex's `float2` position from buffer 0 and its `float4` color from buffer 1, both `device`
/// buffers read through storage views, and `colored_fragment`, which returns that color, one
/// `RGBA8Unorm` target.
fn storage_views_state(device: &Device) -> RenderPipelineState {
    let library = super::encoder_holds::library(device, "examples/runtime_examples.ll");
    let vertex = library.new_function_with_name("colored_vertex").unwrap();
    let fragment = library.new_function_with_name("colored_fragment").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A draw kept to draw again is one through its key's own `VkPipeline`, never its general form's
/// (`render/draw_again.rs`, `pipeline/render/residual.rs`): a draw again records through the kept
/// draw's pipeline, where every step of it takes its key's own once that is made. A vertex
/// function reads two `device` buffers (`colored_vertex`), its positions bound at offset 16 and its
/// colors at 0, drawn four times in one encoder, the first with the device's pool held. On a
/// profile that takes `DriverWorkaround::PushReadsLengthenThreadPayload` both storage views declare
/// `SpecKey::RangeAtBindOffset`, and the creation made the key that reads both at their bind
/// offsets and its general form, which reads both at their residuals ([`storage_views_keyed`]);
/// the positions at 16 leave their slot out (`argument_table::AtBindViews`), so the draw's key
/// reads them at their residual and the colors at their bind offset, neither the creation's: the
/// first draw takes the general form while its own is made, and keeps nothing
/// (`WorkCounts::draws_unkept_making`); once the pool ran, the second takes every step through
/// its own and is kept, and the third and fourth draw it again (in a build with debug assertions
/// the third takes every step too, and its pipeline is compared with the kept draw's). Elsewhere
/// the first draw's key is the creation's, which it keeps, and the three after it draw it again.
/// Every draw reads its bytes.
#[test]
fn a_draw_through_its_general_form_keeps_nothing() {
    let scratch = Scratch::new("general-form-unkept");
    let device = device(&scratch.0);
    let row = storage_views_keyed(&device);
    let queue = device.new_command_queue().unwrap();
    let state = storage_views_state(&device);
    let floats = |values: &[f32]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let shared = ResourceOptions::shared();
    // Sixteen bytes before the positions, then one triangle over the whole target.
    const MOVED: u64 = 16;
    let positions = floats(&[0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 3.0, -1.0, -1.0, 3.0]);
    let positions = device.new_buffer_with_bytes(&positions, shared).unwrap();
    let codes = [9u8, 99, 199, 255];
    let colour = codes.map(|code| f32::from(code) / 255.0);
    let colours = device
        .new_buffer_with_bytes(&floats(&colour.repeat(3)), shared)
        .unwrap();
    let (pass, target) = pass(&device);
    let per = SIDE * SIDE * 4;
    let back = device.new_buffer(per, shared).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    encoder
        .set_vertex_buffer(Some(&positions), MOVED, 0)
        .unwrap();
    encoder.set_vertex_buffer(Some(&colours), 0, 1).unwrap();
    let before = idle(&device);
    let hold = crate::pipeline::pool::held::hold();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    let first = device.work_counts();
    drop(hold);
    let made = idle(&device);
    for _ in 0..3 {
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }
    let after = device.work_counts();
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        SIDE * 4,
        per,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    for (p, pixel) in bytes_of(&back).chunks(4).enumerate() {
        assert_eq!(pixel, codes, "pixel {p}");
    }
    assert_eq!(
        (
            first.residual_form_draws - before.residual_form_draws,
            first.draws_unkept_making - before.draws_unkept_making,
        ),
        (u64::from(row), u64::from(row)),
        "the first draw through its general form, and kept nothing for it"
    );
    // The draws again after it: the last three, or on the row's profile the last two.
    let hits = 3 - u64::from(row);
    assert_eq!(
        after.draws_as_before - made.draws_as_before,
        super::draw_again::drawn_alone(hits),
        "draws again that recorded their Vulkan draw alone"
    );
}
