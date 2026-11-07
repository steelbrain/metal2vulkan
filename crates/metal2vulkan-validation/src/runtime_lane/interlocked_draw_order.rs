//! The order of framebuffer fetches across the draws of one render pass, on the device the cfg
//! selects, under the validation layer. Metal orders a fragment function's read of its color
//! attachment after every earlier fragment of the pixel, across draws. In the pixel interlock
//! (`metal2vulkan_runtime`'s `framebuffer_fetch`) the runtime records a barrier between two draws
//! of a render pass instance where the later one reaches interlocked color storage the earlier one
//! wrote, except on a driver measured to order such draws by the interlock alone
//! (`DriverFact::InterlockOrdersDraws`), which records nothing there, and on one whose such draws
//! race under load (`DriverFact::InterlockedDrawsRaceUnderLoad`), where the later draw records into
//! a new render pass instance instead. Vulkan orders them by the interlock alone, its ordered
//! interlock's operations happening "before interlocked operations in fragment shader invocations
//! that execute later in rasterization order and cover at least one sample in the same pixel"
//! (Fragment Operations, "Interlocked Operations"), rasterization order following primitive order
//! through a subpass, submission order first ("Primitive Order"), and the critical section's
//! accesses coherent. This test is that measurement, and the check of every device's answer either
//! way.
//!
//! Each pass clears an `R32Uint` target to [`CLEAR`] and draws fullscreen triangles
//! (`ido_draw_order`'s `ido_full`), each `ido_triple` (`3c + 1`) or `ido_quintuple` (`5c + 3`), in
//! the order [`steps`] gives: sixteen of one pipeline then the two alternating over the whole
//! target; two sweeps of [`BANDS`] draws each held by its scissor to one band of rows, a band's
//! function by a xorshift's low bit; then sixteen over the whole target again by the xorshift. The
//! maps do not commute, so a texel holds [`expected`] for its band only where each draw read what
//! the draw before it wrote there: a read of an older value, or two draws in another order, leaves
//! another. [`RUNS`] passes at one sample over 1920x1080, each read back whole, then [`RUNS`] at 4
//! samples over 960x540, each sample copied out by `ido_samples`; then both over 64x36, whose
//! banded draws cover two rows each, so many draws run at once. Every texel of every pass must
//! hold its band's composition; the wrong ones are counted and printed per pass, with the GPU time
//! of each pass and what the runtime records between the draws: nothing, a barrier, or a split
//! (`WorkCounts::instance_splits_interlocked_writes`), which every draw after a pass's first must
//! take where the runtime splits there.

use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::DriverFact;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, Device, Library, LoadAction, Origin, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, Size, StoreAction,
    Texture, TextureDescriptor,
};

/// The bands of rows the banded draws are held to, each a whole number of rows at both sizes.
const BANDS: u64 = 18;

/// The passes of each sample count.
const RUNS: usize = 6;

/// The target's clear value.
const CLEAR: u32 = 7;

/// One draw: its function (`true` for `ido_triple`, `false` for `ido_quintuple`), and the band
/// its scissor holds it to, or `None` for the whole target.
type Step = (bool, Option<u64>);

/// The draws of a pass in order (the module docs).
fn steps() -> Vec<Step> {
    let mut x = 0x9e37_79b9u32;
    let mut bit = move || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x & 1 == 1
    };
    let mut steps: Vec<Step> = vec![(true, None); 16];
    steps.extend((0..16).map(|i| (i % 2 == 0, None)));
    for _ in 0..2 {
        steps.extend((0..BANDS).map(|band| (bit(), Some(band))));
    }
    steps.extend((0..16).map(|_| (bit(), None)));
    steps
}

/// What a texel of `band` holds after `steps` in order, from [`CLEAR`].
fn expected(steps: &[Step], band: u64) -> u32 {
    steps
        .iter()
        .filter(|(_, held)| held.is_none_or(|held| held == band))
        .fold(CLEAR, |c, &(triple, _)| match triple {
            true => c.wrapping_mul(3).wrapping_add(1),
            false => c.wrapping_mul(5).wrapping_add(3),
        })
}

/// The two pipelines of `samples` samples: `ido_triple`'s, then `ido_quintuple`'s.
fn pipelines(device: &Device, library: &Library, samples: u32) -> [RenderPipelineState; 2] {
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    ["ido_triple", "ido_quintuple"].map(|name| {
        let mut descriptor = RenderPipelineDescriptor::new(function("ido_full"));
        descriptor.fragment = Some(function(name));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
        descriptor.raster_sample_count = samples;
        device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    })
}

/// A private `R32Uint` render target of `samples` samples and sides.
fn target(device: &Device, samples: u32, (width, height): (u64, u64)) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::R32Uint, width, height, false);
    if samples > 1 {
        d.texture_type = TextureType::Type2DMultisample;
        d.sample_count = u64::from(samples);
    }
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d).unwrap()
}

/// One pass over `target` of the draws [`steps`] names, and its GPU time in milliseconds.
fn draw(device: &Device, target: &Texture, pipelines: &[RenderPipelineState; 2]) -> f64 {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.clear_color = ClearColor::new(f64::from(CLEAR), 0.0, 0.0, 0.0);
    color.store_action = StoreAction::Store;
    let (width, height) = (target.width(), target.height());
    let rows = height / BANDS;
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    for (triple, band) in steps() {
        let pipeline = &pipelines[usize::from(!triple)];
        encoder.set_render_pipeline_state(pipeline).unwrap();
        let scissor = match band {
            Some(band) => ScissorRect::new(0, band * rows, width, rows),
            None => ScissorRect::new(0, 0, width, height),
        };
        encoder.set_scissor_rect(scissor).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    (command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap()) * 1e3
}

/// Every texel of a single-sampled `target` of `sides`, read back through a buffer.
fn single_texels(device: &Device, target: &Texture, (width, height): (u64, u64)) -> Vec<u32> {
    let bytes = width * height * 4;
    let buffer = device.new_buffer(bytes, ResourceOptions::shared()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
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
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    words(&read_buffer(&buffer, 0, bytes).unwrap())
}

/// Every sample of a 4-sample `target` of `sides`, copied out by `ido_samples`.
fn sample_texels(
    device: &Device,
    library: &Library,
    target: &Texture,
    (width, height): (u64, u64),
) -> Vec<u32> {
    let function = library.new_function_with_name("ido_samples").unwrap();
    let state = device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap();
    let bytes = width * height * 4 * 4;
    let buffer = buffer_with(device, &vec![0u8; bytes as usize]).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut compute = command_buffer.compute_command_encoder().unwrap();
    compute.set_compute_pipeline_state(&state).unwrap();
    compute.set_texture(Some(target), 0).unwrap();
    compute.set_buffer(Some(&buffer), 0, 0).unwrap();
    compute
        .dispatch_threads(Size::new(width, height, 1), Size::new(8, 8, 1))
        .unwrap();
    compute.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    words(&read_buffer(&buffer, 0, bytes).unwrap())
}

/// `bytes` as little-endian words.
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

/// Every pass's texels hold the composition of its draws in order (the module docs).
#[test]
fn consecutive_fetching_draws_read_each_others_writes() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let ll = engine_dir().join("ido_draw_order.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let facts = &device.properties().driver_facts;
    let ordered = facts.contains(&DriverFact::InterlockOrdersDraws);
    let races = facts.contains(&DriverFact::InterlockedDrawsRaceUnderLoad);
    // What the runtime records between two such draws of an instance: nothing where the driver
    // orders them by the interlock alone, else a new instance where its draws race under load,
    // else a barrier.
    let between = match (ordered, races) {
        (true, _) => "nothing",
        (false, true) => "a split",
        (false, false) => "a barrier",
    };
    // Every draw reads and writes the target in the interlock, so each after a pass's first
    // splits the pass where the runtime splits there.
    let want_splits = match !ordered && races {
        true => steps().len() as u64 - 1,
        false => 0,
    };
    let steps = steps();
    let mut wrong_passes = Vec::new();
    for (samples, sides) in [
        (1u32, (1920u64, 1080u64)),
        (4, (960, 540)),
        (1, (64, 36)),
        (4, (64, 36)),
    ] {
        let rows = sides.1 / BANDS;
        let per_texel = u64::from(samples);
        let want: Vec<u32> = (0..BANDS).map(|band| expected(&steps, band)).collect();
        let pipelines = pipelines(&device, &library, samples);
        let target = target(&device, samples, sides);
        for run in 0..RUNS {
            let counted = device.work_counts();
            let gpu = draw(&device, &target, &pipelines);
            let splits = device.work_counts().instance_splits_interlocked_writes
                - counted.instance_splits_interlocked_writes;
            let texels = match samples {
                1 => single_texels(&device, &target, sides),
                _ => sample_texels(&device, &library, &target, sides),
            };
            let wrong = texels
                .iter()
                .enumerate()
                .filter(|&(i, &t)| {
                    let row = i as u64 / per_texel / sides.0;
                    t != want[(row / rows).min(BANDS - 1) as usize]
                })
                .count();
            eprintln!(
                "INTERLOCKED-ORDER on {}: {samples} sample(s), pass {run}: {wrong} of {} texels \
                 wrong, GPU {gpu:.2} ms; between draws: {between}, {splits} splits",
                device.name(),
                texels.len(),
            );
            assert_eq!(
                splits,
                want_splits,
                "{}: {samples} sample(s), pass {run}: the splits between draws ({between})",
                device.name()
            );
            if wrong > 0 {
                wrong_passes.push((samples, run, wrong));
            }
        }
    }
    assert!(
        wrong_passes.is_empty(),
        "{}: passes whose draws did not read each other's writes (samples, pass, texels): \
         {wrong_passes:?}",
        device.name()
    );
    drop(device);
    layer_reported_nothing_at_all(&messages);
}
