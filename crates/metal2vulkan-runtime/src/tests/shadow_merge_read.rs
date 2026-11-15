//! A merge of a shadowed `BGRA8Unorm` or `RGBA8Unorm` texture's writes reads the texels the
//! commands left from its shadow (`shadow_merge.rs`): through the shadow's own view, rebuilding
//! each byte from its unsigned normalized value (`bind::ShadowMergeRead`), where the shadow has no
//! unsigned-integer view, and through that view where it has one. Either way, the bytes are the
//! ones a copy of the texels into the merge's scratch gives (the form every other format takes).
//!
//! Each leg makes a [`WIDTH`]x[`HEIGHT`] texture over a `Shared` buffer, rows of [`ROW`] bytes
//! (not a multiple of 256, so a shadow wherever the device lays no image's rows so), whose texels
//! hold every byte value in each of their four bytes in each row ([`before`]), and the bytes
//! between its rows [`GAP`]. A compute dispatch writes the first [`WRITTEN`] texels of row 0
//! through it (`invert_texels`, of a source whose channels are 0 or 255, so its writes are exact:
//! Vulkan converts 0.0 and 1.0 to 0 and 255 exactly), texels that held, in turn, the word the
//! dispatch writes, that word with byte 0 changed, with every byte changed, and with byte 2
//! changed. Once the command buffer completed, those texels hold the dispatch's words and every
//! other byte what it held: a byte rebuilt wrong in any texel the dispatch left would differ from
//! the one kept, and the merge would write it there. The texture's merge fetched its texels
//! (`WorkCounts::shadow_merges_fetched`) and none went back as a copy.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The function is the owned fixture
//! `validation/fixtures/public/examples/runtime_examples.ll` (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    CommandBufferStatus, ComputePipelineState, Device, Library, PipelineOptions, Region,
    ResourceOptions, Size, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The texture: `WIDTH` texels a row (more than one workgroup of the merge's kernel, 256
/// texels), `HEIGHT` rows, rows of `ROW` bytes.
const WIDTH: u64 = 260;
const HEIGHT: u64 = 4;
const ROW: u64 = 1056;
/// The bytes between the rows, which no merge reads or writes.
const GAP: u8 = 0xee;
/// The texels of row 0 the dispatch writes.
const WRITTEN: u64 = 16;

/// Texel `(x, y)`'s bytes before the dispatch, memory order: byte `j` is `(x + 61 j + 17 y) mod
/// 256`, so each row holds each byte value in each byte.
fn before(x: u64, y: u64) -> [u8; 4] {
    [0, 1, 2, 3].map(|j| ((x + 61 * j + 17 * y) % 256) as u8)
}

/// Source texel `x`'s bytes, memory order: byte `j` 255 where bit `j` of `x` is set, else 0.
fn source(x: u64) -> [u8; 4] {
    [0, 1, 2, 3].map(|j| if (x >> j) & 1 == 1 { 255 } else { 0 })
}

/// The bytes `invert_texels` writes from source texel `x`: its three colour channels inverted
/// and its alpha kept, memory order, in both formats (alpha is byte 3 of each).
fn written(x: u64) -> [u8; 4] {
    let s = source(x);
    [255 - s[0], 255 - s[1], 255 - s[2], s[3]]
}

/// What written texel `x` held before the dispatch: its written word, with byte 0, every byte
/// or byte 2 changed by `x mod 4`.
fn written_before(x: u64) -> [u8; 4] {
    let change: [u8; 4] = match x % 4 {
        0 => [0; 4],
        1 => [0x5a, 0, 0, 0],
        2 => [0xa5, 0x5a, 0x3c, 0xc3],
        _ => [0, 0, 0x5a, 0],
    };
    let w = written(x);
    [0, 1, 2, 3].map(|j| w[j] ^ change[j])
}

/// The library of the fixture at `path` under `validation/fixtures/public/`.
fn library(device: &Device, path: &str) -> Library {
    let ll = metal2vulkan_devtools::fixtures_dir().join(path);
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    device.new_library_with_compile_library(library)
}

fn invert(device: &Device) -> ComputePipelineState {
    let function = library(device, "examples/runtime_examples.ll")
        .new_function_with_name("invert_texels")
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

/// The leg of a texture of `format` (the module docs).
fn leg(format: PixelFormat) {
    let scratch = Scratch::new(&format!("shadow-merge-read-{format:?}"));
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let span = (ROW * HEIGHT) as usize;
    let buffer = device
        .new_buffer(ROW * HEIGHT, ResourceOptions::shared())
        .unwrap();
    let bytes = buffer.contents().unwrap().as_ptr();
    let mut want = vec![GAP; span];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let at = (y * ROW + x * 4) as usize;
            let texel = match (y, x < WRITTEN) {
                (0, true) => written_before(x),
                _ => before(x, y),
            };
            want[at..at + 4].copy_from_slice(&texel);
        }
    }
    // SAFETY: the buffer's bytes, which no command uses now.
    unsafe { std::ptr::copy_nonoverlapping(want.as_ptr(), bytes, span) };
    for x in 0..WRITTEN {
        let at = (x * 4) as usize;
        want[at..at + 4].copy_from_slice(&written(x));
    }

    let counts = device.work_counts();
    let mut descriptor = TextureDescriptor::texture_2d(format, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    descriptor.resource_options = buffer.resource_options();
    let target = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    let mut described = TextureDescriptor::texture_2d(format, WRITTEN, 1, false);
    described.usage = TextureUsage::SHADER_READ;
    described.resource_options = ResourceOptions::shared();
    let source_texture = device.new_texture(&described).unwrap();
    let texels: Vec<u8> = (0..WRITTEN).flat_map(source).collect();
    source_texture
        .replace_region(
            Region::new_2d(0, 0, WRITTEN, 1),
            0,
            0,
            &texels,
            WRITTEN * 4,
            0,
        )
        .unwrap();
    let invert = invert(&device);

    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&invert).unwrap();
    encoder.set_texture(Some(&source_texture), 0).unwrap();
    encoder.set_texture(Some(&target), 1).unwrap();
    encoder
        .dispatch_threads(Size::new(WRITTEN, 1, 1), Size::new(WRITTEN, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(
        command_buffer.wait_until_completed_for(WAIT).is_ok(),
        "{format:?}: the command buffer never completed"
    );
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let after = device.work_counts();

    // SAFETY: the command buffer completed; nothing writes the bytes now.
    let now = unsafe { std::slice::from_raw_parts(bytes, span) };
    let differing: Vec<usize> = (0..span).filter(|&i| now[i] != want[i]).collect();
    assert!(
        differing.is_empty(),
        "{format:?}: {} bytes differ, the first at {:?}",
        differing.len(),
        &differing[..differing.len().min(16)],
    );
    let shadowed = after.shadowed_textures - counts.shadowed_textures;
    let fetched = after.shadow_merges_fetched - counts.shadow_merges_fetched;
    let copied_back = after.shadow_texels_copied_back - counts.shadow_texels_copied_back;
    eprintln!(
        "SHADOW-MERGE-READ: {format:?}: {shadowed} shadow made, {fetched} merges fetched, \
         {copied_back} texels copied back"
    );
    if shadowed > 0 {
        assert!(fetched > 0, "{format:?}: the merge copied the texels");
        assert_eq!(copied_back, 0, "{format:?}: texels copied back");
    }
}

#[test]
fn a_bgra8_shadow_merges_every_byte_as_a_copy_would() {
    leg(PixelFormat::BGRA8Unorm);
}

#[test]
fn an_rgba8_shadow_merges_every_byte_as_a_copy_would() {
    leg(PixelFormat::RGBA8Unorm);
}
