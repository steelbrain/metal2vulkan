//! An instance that follows a pass's inline first instance only because that one runs no
//! secondary (`render/inline_first.rs`, `Instance::continues`): its draws' shadowed textures are
//! copied in before the first instance, as where the first instance took them, so a draw in it
//! that reads its own attachment at its own pixel, with no barrier, reads the texels from before
//! the pass (`texture/attached_reads.rs`), Metal's answer there (probe `selfsample`).
//!
//! The pass's first draw fills a box (with the pipeline made at the state's creation, its key's
//! residual form, while its own key's making waits in the pool, `pipeline/render/residual.rs`), so
//! its first instance records inline; the second, a line that reads the attachment, has a pipeline
//! only its draw makes (a line, of another topology class than the state's creation predicts,
//! `pipeline/predicted.rs`), which the device's pool's test hold (`pool::held`) keeps being made,
//! so it records into an instance after the inline one.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect`,
//! `tob_solid` and `tob_read` of the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use crate::pipeline::pool::held;
use crate::{
    BlitOption, Device, DeviceKind, DeviceOptions, LoadAction, Origin, PipelineOptions, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ShaderCacheOptions, Size,
    StoreAction, TextureDescriptor,
};
use metal2vulkan::device::PipelineWaitPolicy;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The longest the command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// The target's side, in pixels.
const SIDE: u64 = 32;

/// The target's red before the pass, and the red the first draw writes.
const BEFORE: u8 = 64;
const DRAWN: u8 = 128;

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path =
            std::env::temp_dir().join(format!("m2v-inline-continues-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache
/// in `cache`.
fn device(cache: &Path) -> Device {
    let mut options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    // The held making is waited for until it is made, whatever the hold's length, under either
    // value; under `PipelineWaitPolicy::Exact` the state's creation makes its own pipeline before
    // it returns, held pool or not (`tests/deferred_calls.rs` takes the default's form).
    options.policy.pipeline_waits = PipelineWaitPolicy::Exact;
    Device::system_default_with(&options).unwrap()
}

/// The fixture's state of `tob_rect` and `fragment`, one `RGBA8Unorm` target.
fn state(device: &Device, fragment: &str) -> RenderPipelineState {
    let ll = metal2vulkan_devtools::engine_dir().join("tob_renders.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let library = device.new_library_with_compile_library(library);
    let vertex = library.new_function_with_name("tob_rect").unwrap();
    let fragment = library.new_function_with_name(fragment).unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

fn floats(v: [f32; 4]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// The corners, in normalized device coordinates, of the target's pixels `[x0, x1) x [y0, y1)`.
fn corners([x0, y0, x1, y1]: [u64; 4]) -> Vec<u8> {
    let x = |p: u64| p as f32 / SIDE as f32 * 2.0 - 1.0;
    let y = |p: u64| 1.0 - p as f32 / SIDE as f32 * 2.0;
    floats([x(x0), y(y1), x(x1), y(y0)])
}

/// The first draw fills `[4, 28)` on both axes; the second draws the lines of the rectangle
/// `[8, 16)`, inside it, each fragment the texel under it in the target's copy.
#[test]
fn a_draw_after_an_inline_first_instance_reads_the_copy_from_before_the_pass() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let fill = state(&device, "tob_solid");
    let read = state(&device, "tob_read");
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&d).unwrap();
    let before = [BEFORE, 0, 0, 255].repeat((SIDE * SIDE) as usize);
    let region = Region::new_2d(0, 0, SIDE, SIDE);
    target
        .replace_region(region, 0, 0, &before, SIDE * 4, 0)
        .unwrap();
    let back = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let counted = device.work_counts();
    let hold = held::hold();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&fill).unwrap();
    encoder
        .set_vertex_bytes(&corners([4, 4, 28, 28]), 0)
        .unwrap();
    let red = f32::from(DRAWN) / 255.0;
    encoder
        .set_fragment_bytes(&floats([red, 0.0, 0.0, 1.0]), 0)
        .unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.set_render_pipeline_state(&read).unwrap();
    encoder
        .set_vertex_bytes(&corners([8, 8, 16, 16]), 0)
        .unwrap();
    encoder.set_fragment_texture(Some(&target), 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Line, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    let drawn = device.work_counts();
    let pooled = drawn.pipelines_pooled - counted.pipelines_pooled;
    let residual = drawn.residual_form_makings - counted.residual_form_makings;
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
        SIDE * SIDE * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    drop(hold);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let after = device.work_counts();
    // The path ran: the first instance inline, the line's draw after it, its making held. The
    // fill reads its bytes at their bind offset on a device that reads views so, which makes that
    // key apart from the creation's residual form, nothing waiting for it
    // (`pipeline/render/residual.rs`).
    assert_eq!(
        residual,
        u64::from(crate::pipeline::reads_views_at_bind(&device)),
        "the fill's own key's making, which nothing waits for"
    );
    assert_eq!(pooled - residual, 1, "the line's making waits in the pool");
    assert_eq!(after.inline_passes - counted.inline_passes, 1);
    assert_eq!(after.inline_passes_split - counted.inline_passes_split, 1);
    let contents = back.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses now.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), back.length() as usize) };
    let red = |x: u64, y: u64| bytes[((y * SIDE + x) * 4) as usize];
    let mut read_before = 0;
    for y in 0..SIDE {
        for x in 0..SIDE {
            let inside = (4..28).contains(&x) && (4..28).contains(&y);
            match (inside, red(x, y)) {
                (false, value) => assert_eq!(value, BEFORE, "pixel ({x}, {y}) outside the fill"),
                (true, value) if value == BEFORE => read_before += 1,
                (true, value) => assert_eq!(value, DRAWN, "pixel ({x}, {y}) inside the fill"),
            }
        }
    }
    // The line's pixels read the copy from before the pass: the texels before the fill.
    assert!(
        read_before > 0,
        "the line read what the inline first instance drew"
    );
}
