//! A render pass's end works out the tiles its draws reach only where something reads them
//! (`render/shadows.rs`, `clear_reach.rs`): the count of the tiles its draws' scissors cover
//! against their bounds (`WorkCounts::shadow_attachment_tiles_covered`) only for a shadowed color
//! attachment it loads within those scissors, which the count is of, and the grid of the tiles
//! its draws reach only for a `DontCare` load that clears them. A pass with neither works out
//! neither, whatever else of its shadows it plans: each pass here draws a box that reads its own
//! attachment at its own pixel, through a copy (`texture/attached_reads.rs`), so its end plans
//! that copy (`WorkCounts::attachment_reads_through_copies`) with no shadowed attachment.
//!
//! The assertion is a count of the work done on this thread (`clear_reach::made`), never a time:
//! a pass end that works the tiles out again where nothing reads them turns it red at once.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect` and
//! `tob_read` of the owned fixture `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{device, Host, Scratch, WAIT};
use crate::clear_reach::made;
use crate::{
    CommandQueue, Device, Error, LoadAction, PipelineOptions, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, ScissorRect, StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::path::Path;

/// The target's side, in pixels: two tiles of 32 each way.
const SIDE: u64 = 64;

/// The fixture's state of `tob_rect` and `tob_read`, one `RGBA8Unorm` target.
fn state(device: &Device) -> RenderPipelineState {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/tob_renders.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let library = device.new_library_with_compile_library(library);
    let vertex = library.new_function_with_name("tob_rect").unwrap();
    let fragment = library.new_function_with_name("tob_read").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The target's descriptor: rendered into and read.
fn descriptor() -> TextureDescriptor {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d
}

/// What one pass's end worked out, and the counts it moved.
#[derive(Debug, PartialEq, Eq)]
struct Ended {
    /// The tile counts and the grids of reached tiles worked out (`clear_reach::made`).
    covered: u64,
    grids: u64,
    /// `WorkCounts::attachment_reads_through_copies`: the draw's read of its own attachment.
    read_copies: u64,
    /// `WorkCounts::shadow_attachment_loads_within` and `shadow_attachment_tiles_bounded`.
    loads_within: u64,
    tiles_bounded: u64,
}

/// A pass over `target`, loaded by `load` and stored: one draw over the whole target that reads
/// the target at its own pixel, inside a scissor of the tile at the origin.
fn end_one_pass(
    device: &Device,
    queue: &CommandQueue,
    state: &RenderPipelineState,
    target: &Texture,
    load: LoadAction,
) -> Ended {
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = load;
    color.store_action = StoreAction::Store;
    let before = device.work_counts();
    let [covered, grids] = made::so_far();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    // The corners of the whole target, in normalized device coordinates.
    let corners: Vec<u8> = [-1.0f32, -1.0, 1.0, 1.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    encoder.set_vertex_bytes(&corners, 0).unwrap();
    encoder.set_fragment_texture(Some(target), 0).unwrap();
    let scissor = ScissorRect {
        x: 8,
        y: 8,
        width: 8,
        height: 8,
    };
    encoder.set_scissor_rect(scissor).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    let [covered_after, grids_after] = made::so_far();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let after = device.work_counts();
    Ended {
        covered: covered_after - covered,
        grids: grids_after - grids,
        read_copies: after.attachment_reads_through_copies - before.attachment_reads_through_copies,
        loads_within: after.shadow_attachment_loads_within - before.shadow_attachment_loads_within,
        tiles_bounded: after.shadow_attachment_tiles_bounded
            - before.shadow_attachment_tiles_bounded,
    }
}

/// A pass whose color attachment is not shadowed counts no tile, loaded or cleared, and makes
/// the grid of the tiles its draws reach only where a `DontCare` load clears them; a shadowed
/// one it loads counts the tiles once, where the device shadows a texture over host pages.
#[test]
fn a_pass_end_works_out_its_tiles_only_where_they_are_read() {
    let scratch = Scratch::new("pass-end-tiles");
    let host = Host::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let state = state(&device);
    let mut private = descriptor();
    private.resource_options = ResourceOptions::private();
    let plain = device.new_texture(&private).unwrap();
    assert!(
        plain.shadow().is_none(),
        "a texture of its own is not shadowed"
    );
    let plain_pass = |load| end_one_pass(&device, &queue, &state, &plain, load);
    // Loaded and cleared: the read's copy is planned, and no tile is worked out.
    for load in [LoadAction::Load, LoadAction::Clear] {
        let ended = plain_pass(load);
        let none = Ended {
            covered: 0,
            grids: 0,
            read_copies: 1,
            loads_within: 0,
            tiles_bounded: 0,
        };
        assert_eq!(ended, none, "{load:?}");
    }
    // A `DontCare` load the draw's tiles bound: their grid, once (`clear_reach::ReachClears`).
    let ended = plain_pass(LoadAction::DontCare);
    assert_eq!((ended.covered, ended.grids, ended.read_copies), (0, 1, 1));
    // A texture over host pages, which a device that cannot render into them in place shadows
    // (the pass copies it in and back): the count of its tiles, once, of the one tile the
    // scissor reaches.
    // SAFETY: the memory outlives the buffer (dropped after the device), and only the buffer's
    // commands use it while it lives.
    let made = unsafe {
        let memory = host.memory();
        device.new_buffer_with_bytes_no_copy(
            memory,
            SIDE * SIDE * 4,
            ResourceOptions::default(),
            None,
        )
    };
    let buffer = match made {
        Ok(buffer) => buffer,
        Err(Error::Unsupported(gap)) => {
            eprintln!("PASS-END-TILES: the device imports no host memory: {gap:?}");
            return;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    let over_pages = buffer.new_texture(&descriptor(), 0, SIDE * 4).unwrap();
    let ended = end_one_pass(&device, &queue, &state, &over_pages, LoadAction::Load);
    let counted = u64::from(over_pages.shadow().is_some());
    eprintln!("PASS-END-TILES: over host pages, shadowed {counted}: {ended:?}");
    let within = Ended {
        covered: counted,
        grids: 0,
        read_copies: 1,
        loads_within: counted,
        tiles_bounded: counted,
    };
    assert_eq!(ended, within);
}
