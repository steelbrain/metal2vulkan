//! A tile function's explicit imageblock standing alone in a render pass of fewer imageblock bytes
//! per sample than its state's, on the device the cfg selects, under the validation layer. Metal's
//! validation stops the state's setting; plain Metal (Metal probes, on two M4s under macOS 27.0.1)
//! runs it, keeping the imageblock's bytes that lie below the pass's bytes rounded up to 8 and
//! reading every other word as zero, the dispatch's own writes included.
//!
//! The owned fixture `rdw_tile_kept`'s functions run in passes of one `R32Uint` attachment of
//! 32x16 pixels in 16x16 tiles, one thread per pixel of the tile. Their imageblock is one `uint4`
//! a cell, after the attachment's four bytes rounded up to 8 and then to the block's 16-byte
//! alignment, so the state's imageblock sample length is 32 and the block's word `k` lies at byte
//! `16 + 4k`. A pass of no imageblock sample length gives the attachment's 4 bytes.
//!
//! - **Kept across dispatches:** `fill` writes word `k` of every cell as `seed + k`, and `dump`,
//!   the next dispatch, reads the words back. A pass of 32 bytes keeps all four words, one of 20
//!   or 24 words 0 and 1 (bytes 16 to 23), and one of 16 or of the attachment's 4 none.
//! - **A dispatch's own writes:** `fill_dump` writes each word at a dynamic index and reads it
//!   back after an imageblock barrier (probe 7's `within` case); the words past the kept ones
//!   read zero all the same.

use super::tile_states::on_device_with;
use super::{at_plan, buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{StageSource, TileRenderPipelineDescriptor};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, Device, Library, LoadAction, PipelineOptions,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size,
    StoreAction, TextureDescriptor,
};

/// The pass's extent.
const AREA: [u32; 2] = [32, 16];

/// The pass's tile side.
const TILE: u32 = 16;

/// The pass's pixels.
const PIXELS: u32 = AREA[0] * AREA[1];

/// The states' imageblock bytes per sample: the attachment's 4 bytes rounded up to 8 and then to
/// 16, then the 16-byte cell.
const SAMPLE_LENGTH: u64 = 32;

/// The passes' imageblock sample lengths, each with the words of a cell it keeps (probes 7 and
/// 10: 16, 20 and 24 measured, and the attachment's own bytes, a length of 0).
const PASSES: [(u64, usize); 5] = [(32, 4), (24, 2), (20, 2), (16, 0), (0, 0)];

/// The fixture's tile functions on one device.
struct Fixture {
    device: Device,
    library: Library,
}

impl Fixture {
    fn new(device: &Device) -> Fixture {
        let bitcode = llvm_assemble(&engine_dir().join("rdw_tile_kept.ll")).unwrap();
        Fixture {
            library: device.new_library_with_air(&bitcode).unwrap(),
            device: device.clone(),
        }
    }

    /// The tile state of `rdw_tile_kept_<kernel>` over one `R32Uint` attachment.
    fn state(&self, kernel: &str) -> RenderPipelineState {
        let function = self
            .library
            .new_function_with_name(&format!("rdw_tile_kept_{kernel}"))
            .unwrap();
        let mut descriptor = TileRenderPipelineDescriptor::new(StageSource::Function(
            function.function_descriptor(),
        ));
        descriptor.color_attachment_formats[0] = Some(PixelFormat::R32Uint);
        self.device
            .new_tile_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// A buffer of four words a pixel, each `0xABABABAB` until written.
    fn out(&self) -> Buffer {
        buffer_with(&self.device, &vec![0xAB; (4 * PIXELS * 4) as usize]).unwrap()
    }

    /// Encode one pass whose imageblock sample length is `sample_length` over a new attachment by
    /// `encode`, and run it.
    fn run(&self, sample_length: u64, encode: impl FnOnce(&mut RenderCommandEncoder)) {
        let mut descriptor = TextureDescriptor::texture_2d(
            PixelFormat::R32Uint,
            AREA[0].into(),
            AREA[1].into(),
            false,
        );
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::shared();
        let target = self.device.new_texture(&descriptor).unwrap();
        let mut pass = RenderPassDescriptor::new();
        pass.color_attachments[0].texture = Some(target);
        pass.color_attachments[0].load_action = LoadAction::Clear;
        pass.color_attachments[0].store_action = StoreAction::Store;
        pass.tile_width = TILE.into();
        pass.tile_height = TILE.into();
        pass.imageblock_sample_length = sample_length;
        let queue = self.device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encode(&mut encoder);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }

    /// A tile dispatch of `state`, one thread per pixel of the tile, with the render area and
    /// `seed` at tile buffer 1 and `out` at tile buffer 0; the buffers it reads go to `held`.
    fn dispatch(
        &self,
        encoder: &mut RenderCommandEncoder,
        state: &RenderPipelineState,
        seed: u32,
        out: Option<&Buffer>,
        held: &mut Vec<Buffer>,
    ) {
        let words = [AREA[0], AREA[1], seed, 0];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let argument = buffer_with(&self.device, &bytes).unwrap();
        encoder.set_render_pipeline_state(state).unwrap();
        encoder.set_tile_buffer(Some(&argument), 0, 1).unwrap();
        encoder.set_tile_buffer(out, 0, 0).unwrap();
        encoder
            .dispatch_threads_per_tile(Size::new(TILE.into(), TILE.into(), 1))
            .unwrap();
        held.push(argument);
    }
}

/// The words of `buffer`.
fn words(buffer: &Buffer) -> Vec<u32> {
    read_buffer(buffer, 0, buffer.length())
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

/// Every pixel's four words: word `k` is `first + k` among the `kept` first, else zero.
fn cells(first: u32, kept: usize) -> Vec<u32> {
    let cell: Vec<u32> = (0..4u32)
        .map(|k| if (k as usize) < kept { first + k } else { 0 })
        .collect();
    (0..PIXELS).flat_map(|_| cell.clone()).collect()
}

/// The words a dispatch writes keep across dispatches below the pass's bytes rounded up to 8 and
/// read zero past them (the module docs).
#[test]
fn a_pass_of_fewer_imageblock_bytes_keeps_the_words_below_them() {
    on_device_with(at_plan, |device| {
        let f = Fixture::new(device);
        let name = device.name().to_owned();
        let fill = f.state("fill");
        let dump = f.state("dump");
        assert_eq!(fill.imageblock_sample_length(), Ok(SAMPLE_LENGTH));
        for (sample_length, kept) in PASSES {
            let out = f.out();
            let seed = 0x100 * (sample_length as u32 + 1);
            f.run(sample_length, |encoder| {
                let mut held = Vec::new();
                f.dispatch(encoder, &fill, seed, None, &mut held);
                f.dispatch(encoder, &dump, 0, Some(&out), &mut held);
            });
            assert_eq!(
                words(&out),
                cells(seed, kept),
                "a pass of {sample_length} imageblock bytes on {name}"
            );
        }
    });
}

/// A dispatch reads the words past the kept ones as zero after its own writes (the module docs).
#[test]
fn a_dispatchs_own_writes_past_the_kept_words_read_zero() {
    on_device_with(at_plan, |device| {
        let f = Fixture::new(device);
        let name = device.name().to_owned();
        let fill_dump = f.state("fill_dump");
        for (sample_length, kept) in PASSES {
            let out = f.out();
            let seed = 0x1000 * (sample_length as u32 + 1);
            f.run(sample_length, |encoder| {
                let mut held = Vec::new();
                f.dispatch(encoder, &fill_dump, seed, Some(&out), &mut held);
            });
            assert_eq!(
                words(&out),
                cells(seed + 16, kept),
                "a pass of {sample_length} imageblock bytes on {name}"
            );
        }
    });
}
