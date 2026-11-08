//! Textures of 1- and 2-byte texels a shader writes over host pages, in place where the device's
//! check passed for such writes (`in_place::Access::NarrowWrites`): the x86 macOS guests' 32x32
//! `R8Unorm` and `A8Unorm` textures at 256 bytes a row, and the same shape of `RG8Unorm` and
//! `R16Unorm`, of `ShaderRead` and `ShaderWrite` usage, over a no-copy buffer of host pages from
//! its first byte. Before the check stored such texels every one was a shadow
//! (`host_texture_shadows_store_width`).
//!
//! - **The narrow verdicts.** Once the device's check settled, its report answers the narrow
//!   accesses of every class and row ends (`Pointer narrow writes: ...`), none pending. Printed
//!   with the check's time.
//! - **The path its verdict gives.** Each texture takes the verdict of its class, its texels' row
//!   ends and its rows' padding for 1- and 2-byte texels (`narrow buffer writes` where its padding
//!   is free, `in_place::Padding`): in place where it passed (`vk_image` answers one;
//!   `host_textures_in_place`, `host_textures_in_place_written`), or a shadow its refusal counts
//!   where the device makes no image at its pitch; a shadow counted by the check's answer where it
//!   did not pass; never one counted by its width. Its placement query, asked before, answers in
//!   place exactly where it is.
//! - **Writes reach the bytes.** A kernel then writes a color at every texel (`a8r_write`), each
//!   format's of 1.0 and 0.0 alone, which every device converts exactly (an `A8Unorm` texture's
//!   alpha 1.0 and red 0.0, so a write of red reads 0): every texel of the pages holds the
//!   format's bytes of it, the bytes past the last row the pattern, and the bytes between the
//!   rows the pattern where the padding is kept. Where it is free the writes may change it, as an
//!   x86 Mac's GPU does; printed, with whether they kept it.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects; the device's shader cache is a fresh
//! directory, so its check runs. Printed (`NARROW-WRITES`).

use super::blit_holds::{device, over_host, Host, Scratch, BYTES, WAIT};
use super::encoder_holds::library;
use crate::in_place::{Access, Padding, PagesClass, RowEnds, Verdict};
use crate::linear_plan::BufferLayout;
use crate::{
    Buffer, ComputePipelineState, PipelineOptions, Size, Texture, TextureDescriptor,
    TexturePlacement, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Duration;

/// The guests' shape: width, height, and bytes a row.
const WIDTH: u64 = 32;
const HEIGHT: u64 = 32;
const PITCH: u64 = 256;

/// The byte each texel holds before the kernel writes it: neither 0x00 nor 0xff.
const BEFORE: u8 = 0x5a;

/// How long the device's in-place check may take to settle.
const SETTLE: Duration = Duration::from_secs(120);

/// Each format: its texel's bytes, the color the kernel writes, and the bytes of it a texel holds.
const FORMATS: [(PixelFormat, u64, [f32; 4], &[u8]); 4] = [
    (PixelFormat::R8Unorm, 1, [1.0, 0.0, 0.0, 0.0], &[0xff]),
    (PixelFormat::A8Unorm, 1, [0.0, 0.0, 0.0, 1.0], &[0xff]),
    (
        PixelFormat::RG8Unorm,
        2,
        [0.0, 1.0, 0.0, 0.0],
        &[0x00, 0xff],
    ),
    (
        PixelFormat::R16Unorm,
        2,
        [1.0, 0.0, 0.0, 0.0],
        &[0xff, 0xff],
    ),
];

/// The byte of the pattern every byte that is no texel holds at `offset`: a count no row pitch
/// or texel lines up with.
fn pattern(offset: u64) -> u8 {
    (offset.wrapping_mul(7) % 251) as u8 ^ 0x81
}

/// The host bytes of the no-copy buffer: the texels [`BEFORE`], every other byte [`pattern`].
fn fill(host: &Host, texel: u64) -> Vec<u8> {
    let bytes: Vec<u8> = (0..BYTES as u64)
        .map(
            |at| match at < HEIGHT * PITCH && at % PITCH < WIDTH * texel {
                true => BEFORE,
                false => pattern(at),
            },
        )
        .collect();
    let (words, _) = bytes.as_chunks::<4>();
    let words: Vec<u32> = words.iter().map(|&word| u32::from_le_bytes(word)).collect();
    host.set(0, &words);
    bytes
}

/// A copy of the no-copy buffer's bytes, which no command writes meanwhile.
fn read(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's contents span its length, and no command runs over them while the test
    // reads them (it waits for its command buffers); they are copied out at once.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// The guests' descriptor of `format`: 32x32, `ShaderRead` and `ShaderWrite`, the buffer's
/// options.
fn descriptor(buffer: &Buffer, format: PixelFormat) -> TextureDescriptor {
    let mut d = TextureDescriptor::texture_2d(format, WIDTH, HEIGHT, false);
    d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    d.resource_options = buffer.resource_options();
    d
}

/// The shadows over host pages counted by why between `before` and `after`: refused an image,
/// by the check's answer, and by their texels' width.
fn shadows(before: &WorkCounts, after: &WorkCounts) -> (u64, u64, u64) {
    let refused = |counts: &WorkCounts| {
        counts.host_texture_shadows_refused_rows
            + counts.host_texture_shadows_refused_detached
            + counts.host_texture_shadows_refused_image
            + counts.host_texture_shadows_refused_pitch
            + counts.host_texture_shadows_refused_placement
            + counts.host_texture_shadows_refused_source
            + counts.host_texture_shadows_refused_dma_buf
    };
    let by_check = |counts: &WorkCounts| {
        counts.host_texture_shadows_check_pending
            + counts.host_texture_shadows_check_failed
            + counts.host_texture_shadows_unchecked
    };
    let width = |counts: &WorkCounts| counts.host_texture_shadows_store_width;
    (
        refused(after) - refused(before),
        by_check(after) - by_check(before),
        width(after) - width(before),
    )
}

/// The kernel that writes the color at `buffer(0)` at every texel of `texture(0)`.
fn writer(device: &crate::Device) -> ComputePipelineState {
    let function = library(device, "engine/a8r_targets.ll")
        .new_function_with_name("a8r_write")
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

/// `kernel` writing `color` at every texel of `texture`, waited for.
fn write(
    queue: &crate::CommandQueue,
    kernel: &ComputePipelineState,
    texture: &Texture,
    color: [f32; 4],
) {
    let bytes: Vec<u8> = color.iter().flat_map(|c| c.to_le_bytes()).collect();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(kernel).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_bytes(&bytes, 0).unwrap();
    encoder
        .dispatch_threads(Size::new(WIDTH, HEIGHT, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

#[test]
fn narrow_written_textures_take_the_path_their_narrow_check_answered() {
    let host = Host::new();
    let scratch = Scratch::new("narrow-writes");
    let device = device(&scratch.0);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(buffer) = over_host(&device, &host, &runs) else {
        return;
    };
    let name = device.name();
    assert!(device.settle_in_place_checks(SETTLE), "the check settled");
    let report = device.in_place_check_report();
    let narrow: Vec<&String> = report
        .iter()
        .filter(|line| line.contains("narrow"))
        .collect();
    for line in &narrow {
        println!("NARROW-WRITES on {name}: {line}");
    }
    println!(
        "NARROW-WRITES on {name}: the check took {:?}",
        device.in_place_check_took()
    );
    // Three classes, two narrow accesses, two row ends; none pending.
    assert_eq!(narrow.len(), 12, "{report:?}");
    assert!(
        narrow.iter().all(|line| !line.ends_with("Pending")),
        "{narrow:?}"
    );
    let queue = device.new_command_queue().unwrap();
    let kernel = writer(&device);
    let class = match buffer.over_dma_buf() {
        true => PagesClass::DmaBuf,
        false => PagesClass::Pointer,
    };
    let mut lapses = Vec::new();
    for (format, texel, color, due) in FORMATS {
        let before_bytes = fill(&host, texel);
        let layout = BufferLayout {
            offset: 0,
            row_pitch: PITCH,
            slice_pitch: PITCH * HEIGHT,
            row_bytes: WIDTH * texel,
            rows: HEIGHT,
            slices: 1,
        };
        let start = buffer.contents().map_or(0, |at| at.as_ptr().addr() as u64);
        let checks = &device.shared().in_place;
        let line = checks.line();
        let rows = RowEnds::of(start, &layout, line);
        let padding = Padding::of(start, &layout, line, !buffer.copied());
        let access = Access::written(texel, padding).unwrap();
        assert!(
            matches!(access, Access::NarrowWrites | Access::NarrowBufferWrites),
            "{format:?}: {access:?}"
        );
        let verdict = checks.verdict(class, access, rows);
        let d = descriptor(&buffer, format);
        let asked = buffer.texture_placement(&d, 0, PITCH).unwrap();
        let before = device.work_counts();
        let texture = buffer.new_texture(&d, 0, PITCH).unwrap();
        let after = device.work_counts();
        let aliased = after.host_textures_in_place - before.host_textures_in_place;
        let written = after.host_textures_in_place_written - before.host_textures_in_place_written;
        let (refused, by_check, width) = shadows(&before, &after);
        // SAFETY: the handle is only compared, while the texture lives.
        let image = unsafe { texture.vk_image() };
        let path = match (image.is_some(), aliased, written, refused, by_check) {
            (true, 1, 1, 0, 0) => "in place",
            (false, 0, 0, 1, 0) => "refused",
            (false, 0, 0, 0, 1) => "shadow",
            _ => "inconsistent",
        };
        let what = format!("{format:?} {WIDTH}x{HEIGHT}/{PITCH}");
        println!(
            "NARROW-WRITES on {name}: {what} over {class:?}: {access:?} {rows:?} {verdict:?}; \
             asked {asked:?}; made {path} (in place {aliased}, written {written}, refused \
             {refused}, by the check {by_check}, by width {width})"
        );
        let right = match verdict {
            Verdict::Passed => path == "in place" || path == "refused",
            _ => path == "shadow",
        };
        if !right || width != 0 {
            lapses.push(format!(
                "{what}: {path} by width {width}, the verdict {verdict:?}"
            ));
        }
        if (asked == TexturePlacement::InPlace) != (path == "in place") {
            lapses.push(format!("{what}: asked {asked:?}, made {path}"));
        }
        write(&queue, &kernel, &texture, color);
        let now = read(&buffer);
        let mut texels_right = true;
        let mut padding_kept = true;
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let at = (y * PITCH + x * texel) as usize;
                let read = &now[at..at + texel as usize];
                if read != due {
                    texels_right = false;
                    lapses.push(format!("{what}: texel {x},{y} reads {read:?}"));
                }
            }
            let between = (y * PITCH + WIDTH * texel) as usize..((y + 1) * PITCH) as usize;
            if now[between.clone()] != before_bytes[between] {
                padding_kept = false;
            }
        }
        let free = padding == Padding::Free && path == "in place";
        if !padding_kept && !free {
            lapses.push(format!("{what}: the rows' padding changed"));
        }
        let past = (HEIGHT * PITCH) as usize..now.len();
        if now[past.clone()] != before_bytes[past] {
            lapses.push(format!("{what}: the bytes past the last row changed"));
        }
        println!(
            "NARROW-WRITES on {name}: {what}: every texel {due:?} {texels_right}, padding \
             {padding:?}, kept {padding_kept}"
        );
        drop(texture);
    }
    drop(buffer);
    assert!(lapses.is_empty(), "{}", lapses.join("\n"));
}
