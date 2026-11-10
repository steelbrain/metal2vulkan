//! Render passes and kernel writes into ASTC textures, of every form (LDR, sRGB and HDR, Apple's
//! 4x2 and 8x4 footprints among them, in their own block or in a carrier), on the device the cfg
//! selects, under the validation layer.
//!
//! Plain Metal (an Apple M3 Max under macOS 27.0.1, the conformance probe
//! `astc_passes_writes_footprints`) completes a pass with an ASTC color attachment, whatever its
//! load and store actions, with a draw and without, leaving the texture's blocks as they were and
//! rendering the pass's other attachments as it would; a render pipeline whose attachment format
//! is an ASTC format ends the process. The runtime drops the attachment from the pass, which is
//! Metal's answer, and refuses the pipeline ([`astc_attachments_leave_the_blocks_as_metal_s`]).
//!
//! Plain Metal completes a kernel's `write` and `read_write` access of an ASTC texture and stores
//! bytes no format defines into its blocks. The runtime binds the slot's neutral instead: the
//! dispatch completes, the texture's blocks and texels stay as they were, and the device announces
//! `astc-storage-neutral` once ([`ASTC_STORAGE`], [`astc_kernel_writes_leave_the_blocks`]).

use super::astc_textures::{block_count, read_whole, reader, row_bytes, SIDE};
use super::at_plan;
use super::degrade_notices::on_device_with;
use super::degrade_notices::{counts, on_device};
use super::WAIT;
use crate::air_library::AirLibrary;
use metal2vulkan::bind::AstcDecode;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, CommandBufferStatus, CommandQueue, Degradation, Device, Error, Library, LoadAction,
    PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size,
    StoreAction, Texture, TextureDescriptor,
};

/// The kind a dispatch or draw binding an ASTC texture at a storage slot announces, once
/// ([`astc_kernel_writes_leave_the_blocks`]).
pub(super) const ASTC_STORAGE: [Degradation; 1] = [Degradation::AstcStorageNeutral];

/// Every ASTC format Metal makes: LDR, sRGB and HDR, of every footprint.
fn formats() -> impl Iterator<Item = PixelFormat> {
    PixelFormat::ALL
        .iter()
        .copied()
        .filter(|&pf| AstcDecode::of(pf).is_some())
}

/// A `SIDE` x `SIDE` texture of `pf`, `Shared`, made with `usage`.
fn shared(device: &Device, pf: PixelFormat, usage: TextureUsage) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(pf, SIDE.into(), SIDE.into(), false);
    descriptor.usage = usage;
    descriptor.resource_options = ResourceOptions::shared();
    device.new_texture(&descriptor).unwrap()
}

/// The whole of a `SIDE` x `SIDE` level.
fn whole() -> Region {
    Region::new_2d(0, 0, SIDE.into(), SIDE.into())
}

/// A texture of `pf` and `usage` holding blocks of a pattern seeded by `seed`, and the blocks.
fn patterned(
    device: &Device,
    pf: PixelFormat,
    usage: TextureUsage,
    seed: u8,
) -> (Texture, Vec<u8>) {
    let decode = AstcDecode::of(pf).unwrap();
    let texture = shared(device, pf, usage);
    let blocks: Vec<u8> = (0..16 * block_count(decode))
        .map(|i| (i as u8).wrapping_mul(29).wrapping_add(seed))
        .collect();
    texture
        .replace_region(whole(), 0, 0, &blocks, row_bytes(decode), 0)
        .unwrap();
    (texture, blocks)
}

/// The blocks of `texture`, a `SIDE` x `SIDE` texture of `pf`, as `get_bytes` copies them out.
fn blocks_of(texture: &Texture, pf: PixelFormat) -> Vec<u8> {
    let decode = AstcDecode::of(pf).unwrap();
    let mut back = vec![0; 16 * block_count(decode)];
    texture
        .get_bytes(&mut back, row_bytes(decode), 0, whole(), 0, 0)
        .unwrap();
    back
}

/// A pipeline of the owned fixture `color_groups`'s full-screen triangle (`cg_cover`) and
/// `cg_five`, which writes colors 0 to 4, of a color attachment of `pf` at `index` alone.
fn cover_pipeline(
    device: &Device,
    library: &Library,
    (index, pf): (usize, PixelFormat),
) -> Result<RenderPipelineState, Error> {
    let function = |name| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("cg_cover"));
    descriptor.fragment = Some(function("cg_five"));
    descriptor.color_attachments[index] = Some(ColorAttachmentDescriptor::new(pf));
    device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// One pass over `astc` at color 0 (`load` and `store`, its clear a value of every channel) and
/// an `RGBA8Unorm` texture at color 1 cleared to green and stored, drawing `pipeline`'s triangle
/// where it is given, committed and waited for; the color 1 texture's bytes.
fn pass(
    device: &Device,
    queue: &CommandQueue,
    astc: Option<(&Texture, LoadAction, StoreAction)>,
    pipeline: Option<&RenderPipelineState>,
) -> Vec<u8> {
    let other = shared(
        device,
        PixelFormat::RGBA8Unorm,
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
    );
    let mut descriptor = RenderPassDescriptor::new();
    if let Some((texture, load, store)) = astc {
        let color = &mut descriptor.color_attachments[0];
        color.texture = Some(texture.clone());
        color.load_action = load;
        color.store_action = store;
        color.clear_color = ClearColor::new(0.5, 4.0, -2.0, 1.0);
    }
    let color = &mut descriptor.color_attachments[1];
    color.texture = Some(other.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 1.0, 0.0, 1.0);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    if let Some(pipeline) = pipeline {
        encoder.set_render_pipeline_state(pipeline).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let mut bytes = vec![0; 4 * (SIDE * SIDE) as usize];
    other
        .get_bytes(&mut bytes, 4 * u64::from(SIDE), 0, whole(), 0, 0)
        .unwrap();
    bytes
}

/// Every ASTC format's texture at color 0 of a pass, cleared, loaded or left (`DontCare`), stored
/// or not, with a draw and without: the pass completes, the texture's blocks come back unchanged,
/// and color 1 holds what the same pass without the ASTC attachment leaves (the module docs). A
/// render pipeline of an ASTC color attachment is refused.
#[test]
fn astc_attachments_leave_the_blocks_as_metal_s() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let queue = device.new_command_queue().unwrap();
        let bitcode = llvm_assemble(&engine_dir().join("color_groups.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        // Its color 0 output has no attachment, as Metal's pipeline of a pass with an ASTC color
        // 0 has none.
        let pipeline = cover_pipeline(device, &library, (1, PixelFormat::RGBA8Unorm)).unwrap();
        let actions = [
            (LoadAction::Clear, StoreAction::Store, false),
            (LoadAction::Load, StoreAction::Store, true),
            (LoadAction::DontCare, StoreAction::DontCare, true),
            (LoadAction::Clear, StoreAction::Store, true),
        ];
        // What color 1 holds after each pass without the ASTC attachment.
        let alone: Vec<Vec<u8>> = actions
            .iter()
            .map(|&(_, _, draw)| pass(device, &queue, None, draw.then_some(&pipeline)))
            .collect();
        assert_ne!(alone[0], alone[1], "on {name}: the draw changes color 1");
        let usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let mut wrong = Vec::new();
        let mut checked = 0;
        for (f, pf) in formats().enumerate() {
            let (texture, blocks) = patterned(device, pf, usage, f as u8);
            for (&(load, store, draw), alone) in actions.iter().zip(&alone) {
                let what = format!("{pf:?} {load:?} {store:?} draw {draw}");
                let drawing = draw.then_some(&pipeline);
                let other = pass(device, &queue, Some((&texture, load, store)), drawing);
                if blocks_of(&texture, pf) != blocks {
                    wrong.push(format!("{what}: the pass changed the blocks"));
                }
                if &other != alone {
                    wrong.push(format!("{what}: color 1 is not the pass's without it"));
                }
                checked += 1;
            }
        }
        // Metal ends the process at a pipeline of an ASTC attachment; the runtime refuses it.
        for pf in [
            PixelFormat::ASTC4x4LDR,
            PixelFormat::ASTC8x8SRGB,
            PixelFormat::ASTC6x6HDR,
            PixelFormat::ASTC4x2HDR,
        ] {
            let made = cover_pipeline(device, &library, (0, pf)).map(drop);
            if made.is_ok() {
                wrong.push(format!("{pf:?}: a pipeline of the attachment was made"));
            }
        }
        eprintln!("ASTC-PASSES {name}: {checked} passes over an ASTC attachment");
        assert_eq!(checked, 4 * 48, "on {name}");
        assert!(wrong.is_empty(), "on {name}:\n{}", wrong.join("\n"));
        let counts = counts(&messages.take_notices());
        assert_eq!(counts.get(&ASTC_STORAGE[0]), None, "on {name}");
    });
}

/// Every ASTC format's texture bound to a kernel's `write` slot, then to a `read_write` one: each
/// dispatch completes, the texture's blocks come back unchanged and its texels read as before,
/// and the device announces [`ASTC_STORAGE`] once (the module docs).
#[test]
fn astc_kernel_writes_leave_the_blocks() {
    on_device_with(at_plan, |device, messages| {
        let name = device.name().to_owned();
        let queue = device.new_command_queue().unwrap();
        let bitcode = llvm_assemble(&engine_dir().join("format_scout.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let kernel = |name| {
            let function = library.new_function_with_name(name).unwrap();
            let source = StageSource::Function(function.function_descriptor());
            device
                .new_compute_pipeline_state(
                    &ComputePipelineDescriptor::new(source),
                    &PipelineOptions::default(),
                )
                .unwrap()
        };
        let writers = [kernel("scout_write_f"), kernel("scout_read_write_f")];
        let state = reader(device);
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let mut wrong = Vec::new();
        let mut checked = 0;
        for (f, pf) in formats().enumerate() {
            let (texture, blocks) = patterned(device, pf, usage, 0x80 ^ f as u8);
            let before = read_whole(device, &state, &texture);
            for (access, writer) in ["write", "read_write"].into_iter().zip(&writers) {
                let command_buffer = queue.command_buffer().unwrap();
                let mut encoder = command_buffer.compute_command_encoder().unwrap();
                encoder.set_compute_pipeline_state(writer).unwrap();
                encoder.set_texture(Some(&texture), 0).unwrap();
                let side = u64::from(SIDE);
                encoder
                    .dispatch_threads(Size::new(side, side, 1), Size::new(side, side, 1))
                    .unwrap();
                encoder.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                if blocks_of(&texture, pf) != blocks {
                    wrong.push(format!("{pf:?} {access}: the dispatch changed the blocks"));
                }
                let after = read_whole(device, &state, &texture);
                let bits = |texels: &[[f32; 4]]| -> Vec<[u32; 4]> {
                    texels.iter().map(|t| t.map(f32::to_bits)).collect()
                };
                if bits(&after) != bits(&before) {
                    wrong.push(format!("{pf:?} {access}: the dispatch changed the texels"));
                }
                checked += 1;
            }
        }
        eprintln!("ASTC-WRITES {name}: {checked} dispatches over an ASTC texture");
        assert_eq!(checked, 2 * 48, "on {name}");
        assert!(wrong.is_empty(), "on {name}:\n{}", wrong.join("\n"));
        let counts = counts(&messages.take_notices());
        assert_eq!(counts.get(&ASTC_STORAGE[0]), Some(&1), "on {name}");
    });
}
