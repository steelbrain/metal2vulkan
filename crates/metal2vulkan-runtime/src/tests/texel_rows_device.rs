//! A shadowed 2D texture's rows read and written in place on a device (`texture/texel_rows.rs`).
//!
//! - **In place.** A texture over a no-copy buffer at a pitch of its row plus 16 bytes, whose
//!   image the device refuses, is a shadow whose rows a kernel that asks its size, reads and
//!   writes it binds as a texel buffer: its texels are inverted in the pages, the bytes between
//!   its rows and past it keep theirs, and its commands copy nothing into or out of the shadow.
//!   Each test runs over a buffer imported over its pages in place and over one that is a copy of
//!   them (the caller's gate refusing the import, `copied.rs`), whose writes merge back into the
//!   pages at completion; lavapipe lays the rows over the pages in place, and refuses an image at
//!   that pitch over the copy.
//! - **The shadow still serves a sample.** A kernel that samples the texture after the rows'
//!   writes, in the same command buffer and in the next, reads the inverted texels through the
//!   shadow, copied in.
//! - **Its shadow made at use.** The texture is made with its shadow left unmade
//!   (`texture/lazy_shadow.rs`): one only written through its rows makes none and copies nothing;
//!   the first kernel that samples one makes it at its binding, copied in after an earlier
//!   encoder's writes of the rows in the same command buffer, and reads what a texture of the
//!   device's own holding the texels gives; a `use_resource` of one makes it at the call
//!   (`WorkCounts::shadows_left_unmade`, `shadows_made_at_use`, `shadowed_textures`).
//! - **The image route's answers.** The rows read and write what the image route does, bit for
//!   bit: every `RGBA8Unorm` code read, and a sweep of floats written (each code's value, the
//!   midpoints between codes, and values outside `[0, 1]`); every `RGBA16Float` code read, and
//!   each code's value, a value between it and the next, and its negation written. That is the
//!   evidence behind the route's lack of a precision note: a texel buffer's texels go through the
//!   device's conversion as an image's do.
//! - **Pipeline creation.** The time the first dispatch of a kernel over the rows takes to make its
//!   rows variant, and a second state's, against the same over the image route: printed, for a
//!   device's run to compare with Metal's range.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`). A device that
//! imports no host memory, or that lays the rows in place over both buffers (no shadow), says so
//! and passes; lavapipe must take the rows over the copy.

use crate::{
    Buffer, CommandBufferStatus, ComputePipelineState, Device, DeviceKind, DeviceOptions,
    HostMemory, ImportGate, Region, ResourceOptions, Size, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, ResourceUsage, TextureUsage};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(120);

/// Host memory of whole pages, freed when dropped (after the device).
struct Pages {
    pointer: NonNull<u8>,
    layout: Layout,
}

impl Pages {
    fn new(length: usize, seed: u8) -> Pages {
        let layout = Layout::from_size_align(length.next_multiple_of(1 << 16), 1 << 16).unwrap();
        // SAFETY: the layout's size is not zero.
        let pointer = NonNull::new(unsafe { std::alloc::alloc(layout) }).unwrap();
        let pages = Pages { pointer, layout };
        let bytes: Vec<u8> = (0..layout.size())
            .map(|at| (at as u8).wrapping_mul(29).wrapping_add(seed))
            .collect();
        pages.write(0, &bytes);
        pages
    }

    fn bytes(&self) -> Vec<u8> {
        // SAFETY: the allocation's bytes, which no command writes now.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.layout.size()) }.to_vec()
    }

    fn write(&self, at: usize, bytes: &[u8]) {
        assert!(at + bytes.len() <= self.layout.size());
        // SAFETY: the allocation's bytes, which no command uses now.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.pointer.as_ptr().add(at),
                bytes.len(),
            )
        };
    }

    /// A `Shared` no-copy buffer over the pages, lying as `over` says.
    fn buffer(&self, device: &Device, over: Over) -> Buffer {
        // SAFETY: the pages outlive the buffer and the device (dropped after both).
        let memory = unsafe { HostMemory::new(self.pointer) };
        let memory = match over {
            Over::Pages => memory,
            Over::Copy => {
                let refused: ImportGate = Arc::new(|_| false);
                memory.with_import_gate(refused)
            }
        };
        // SAFETY: as above.
        unsafe {
            device.new_buffer_with_bytes_no_copy(
                memory,
                self.layout.size() as u64,
                ResourceOptions::shared(),
                None,
            )
        }
        .expect("a no-copy buffer over the pages")
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `new`.
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

/// How a test's buffer lies over its pages: imported over them in place, or a copy of them, the
/// caller's gate refusing the import (`copied.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Over {
    Pages,
    Copy,
}

/// Both, each test's runs.
const OVER: [Over; 2] = [Over::Pages, Over::Copy];

/// Whether a test that took the rows over no buffer of `taken` (`OVER`'s) passes: on a device
/// other than lavapipe, which may lay every such texture in place; lavapipe refuses an image at
/// the tests' pitch over the copy, so it must take them there.
fn untaken_passes(taken: &[Over]) -> bool {
    if cfg!(m2v_cpu_icd) {
        assert!(
            taken.contains(&Over::Copy),
            "lavapipe refuses an image at this pitch over a copy, and the shadow's rows are read \
             in place"
        );
    }
    taken.is_empty()
}

fn device() -> Option<Device> {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    if device.shared().host_import.is_err() {
        eprintln!("TEXEL-ROWS: the device imports no host memory here; nothing to test");
        return None;
    }
    let exportable = crate::device::lock(&device.shared().memory)
        .fd_buffer_types
        .is_ok();
    if exportable {
        assert!(device.settle_in_place_checks(Duration::from_secs(120)));
    }
    Some(device)
}

/// The state of `kernel` of `examples/texel_rows.metal`.
fn state(device: &Device, kernel: &str) -> ComputePipelineState {
    let function = super::encoder_holds::library(device, "examples/texel_rows.ll")
        .new_function_with_name(kernel)
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &crate::PipelineOptions::default())
        .unwrap()
}

/// A texture of `format` (texels of `texel` bytes), `width` x `height`, `ShaderRead` and
/// `ShaderWrite`, over `buffer`'s bytes from its first, rows `pitch` bytes apart, and whether
/// its rows are read and written in place. `over` is how the buffer lies, for the line.
fn texture_over(
    device: &Device,
    (buffer, over): (&Buffer, Over),
    (format, width, height): (PixelFormat, u64, u64),
    pitch: u64,
) -> (Texture, bool) {
    let mut descriptor = TextureDescriptor::texture_2d(format, width, height, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    descriptor.resource_options = ResourceOptions::shared();
    let before = device.work_counts();
    let texture = buffer.new_texture(&descriptor, 0, pitch).unwrap();
    let after = device.work_counts();
    let rows = after.texel_rows_textures - before.texel_rows_textures;
    eprintln!(
        "TEXEL-ROWS: {format:?} {width}x{height} at a pitch of {pitch} over the {over:?}: a \
         shadow {}, rows {rows}, refused pitch {} image {} placement {} source {}",
        texture.shadow().is_some(),
        after.host_texture_shadows_refused_pitch - before.host_texture_shadows_refused_pitch,
        after.host_texture_shadows_refused_image - before.host_texture_shadows_refused_image,
        after.host_texture_shadows_refused_placement
            - before.host_texture_shadows_refused_placement,
        after.host_texture_shadows_refused_source - before.host_texture_shadows_refused_source,
    );
    (texture, rows == 1)
}

/// A `Shared` texture of the device's own of `format`, `width` x `height`, `ShaderRead` and
/// `ShaderWrite`, holding `bytes` (rows tight, `texel` bytes a texel): the image route.
fn own_texture(
    device: &Device,
    (format, width, height): (PixelFormat, u64, u64),
    texel: u64,
    bytes: &[u8],
) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, width, height, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let region = Region::new_2d(0, 0, width, height);
    texture
        .replace_region(region, 0, 0, bytes, width * texel, 0)
        .unwrap();
    texture
}

/// A buffer of the device's own holding `bytes`.
fn own_buffer(device: &Device, bytes: &[u8]) -> Buffer {
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    let at = buffer.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), at.as_ptr(), bytes.len()) };
    buffer
}

/// `buffer`'s first `length` bytes as the CPU reads them.
fn contents(buffer: &Buffer, length: usize) -> Vec<u8> {
    let at = buffer.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command writes now.
    unsafe { std::slice::from_raw_parts(at.as_ptr(), length) }.to_vec()
}

/// One dispatch of `state` over a `grid`, `columns` (the grid's width) at buffer 1, `texture` at
/// texture 0 and `buffer` at buffer 0 where given.
struct Dispatch<'a> {
    state: &'a ComputePipelineState,
    texture: &'a Texture,
    buffer: Option<&'a Buffer>,
    grid: [u64; 2],
}

/// Commit one command buffer of `dispatches`, each in an encoder of its own, wait for it, and
/// answer its status.
fn run(device: &Device, dispatches: &[Dispatch<'_>]) -> CommandBufferStatus {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    for d in dispatches {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(d.state).unwrap();
        encoder.set_texture(Some(d.texture), 0).unwrap();
        if let Some(buffer) = d.buffer {
            encoder.set_buffer(Some(buffer), 0, 0).unwrap();
        }
        let columns = (d.grid[0] as u32).to_le_bytes();
        encoder.set_bytes(&columns, 1).unwrap();
        encoder
            .dispatch_threads(Size::new(d.grid[0], d.grid[1], 1), Size::new(8, 8, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    command_buffer.status()
}

/// `bytes` laid in rows `pitch` apart, `width` texels of `texel` bytes each, `height` of them,
/// with each row's own bytes `f(x, y, byte)`.
fn rows_of(
    bytes: &mut [u8],
    (width, height, texel, pitch): (u64, u64, u64, u64),
    f: impl Fn(u64, u64, u64) -> u8,
) {
    for y in 0..height {
        for b in 0..width * texel {
            bytes[(y * pitch + b) as usize] = f(b / texel, y, b % texel);
        }
    }
}

/// The texels of `bytes` laid in rows `pitch` apart, rows tight.
fn tight(bytes: &[u8], (width, height, texel, pitch): (u64, u64, u64, u64)) -> Vec<u8> {
    (0..height)
        .flat_map(|y| {
            let start = (y * pitch) as usize;
            bytes[start..start + (width * texel) as usize].to_vec()
        })
        .collect()
}

fn shadow_copies(before: &WorkCounts, after: &WorkCounts) -> u64 {
    after.shadow_copies - before.shadow_copies
}

const SIDE: u64 = 64;

/// A kernel that asks the size, reads and writes a shadowed texture binds its rows: its texels
/// inverted in the pages with no shadow copy, the bytes between and past the rows kept; a sample
/// of it afterwards, in the same command buffer and the next, reads the inversion through the
/// shadow.
#[test]
fn a_kernel_reads_and_writes_a_shadowed_texture_s_rows_in_place() {
    let Some(device) = device() else { return };
    let taken: Vec<Over> = (OVER.into_iter())
        .filter(|&over| reads_and_writes(&device, over))
        .collect();
    if untaken_passes(&taken) {
        eprintln!("TEXEL-ROWS: this device lays the rows in place; nothing to test");
    }
}

/// [`a_kernel_reads_and_writes_a_shadowed_texture_s_rows_in_place`] over a buffer lying as
/// `over` says; `false` where the texture's rows are not read in place.
fn reads_and_writes(device: &Device, over: Over) -> bool {
    let pitch = SIDE * 4 + 16;
    let pages = Pages::new((pitch * SIDE) as usize, 7);
    let buffer = pages.buffer(device, over);
    let shape = (PixelFormat::BGRA8Unorm, SIDE, SIDE);
    let (texture, rows) = texture_over(device, (&buffer, over), shape, pitch);
    if !rows {
        return false;
    }
    let invert = state(device, "rows_invert");
    let sample = state(device, "rows_sample");
    let original = pages.bytes();
    let layout = (SIDE, SIDE, 4, pitch);
    let inverted: Vec<u8> = (tight(&original, layout).as_chunks::<4>().0.iter())
        .flat_map(|t| [255 - t[0], 255 - t[1], 255 - t[2], t[3]])
        .collect();
    // The grid crosses the texture's edges: the kernel's own test stores nothing past them.
    let before = device.work_counts();
    let status = run(
        device,
        &[Dispatch {
            state: &invert,
            texture: &texture,
            buffer: None,
            grid: [SIDE + 8, SIDE + 8],
        }],
    );
    let after = device.work_counts();
    assert_eq!(status, CommandBufferStatus::Completed);
    assert_eq!(shadow_copies(&before, &after), 0, "no shadow copy");
    assert!(
        after.texel_rows_bound > before.texel_rows_bound,
        "the rows bound"
    );
    let now = pages.bytes();
    assert!(
        tight(&now, layout) == inverted,
        "the texels inverted in the pages"
    );
    let mut kept = now.clone();
    rows_of(&mut kept, layout, |x, y, b| {
        original[(y * pitch + x * 4 + b) as usize]
    });
    assert!(kept == original, "the bytes between and past the rows kept");
    // A sample after the rows' writes, in the same command buffer: the shadow copied in.
    let out = device
        .new_buffer(SIDE * SIDE * 16, ResourceOptions::shared())
        .unwrap();
    let before = device.work_counts();
    let status = run(
        device,
        &[
            Dispatch {
                state: &invert,
                texture: &texture,
                buffer: None,
                grid: [SIDE, SIDE],
            },
            Dispatch {
                state: &sample,
                texture: &texture,
                buffer: Some(&out),
                grid: [SIDE, SIDE],
            },
        ],
    );
    let after = device.work_counts();
    assert_eq!(status, CommandBufferStatus::Completed);
    assert!(
        shadow_copies(&before, &after) > 0,
        "the sample reads the shadow"
    );
    let sampled = contents(&out, (SIDE * SIDE * 16) as usize);
    // Inverted twice: the original texels, as the device samples them from a texture of its own
    // holding them (its conversion of a code, not Metal's).
    let shape = (PixelFormat::BGRA8Unorm, SIDE, SIDE);
    let want = sampled_from(
        device,
        &sample,
        &own_texture(device, shape, 4, &tight(&original, layout)),
    );
    let once = sampled_from(device, &sample, &own_texture(device, shape, 4, &inverted));
    assert_eq!(
        first_difference(&sampled, &want, 16),
        None,
        "the sample reads the rows' writes (the first write's alone: {})",
        first_difference(&sampled, &once, 16).is_none()
    );
    // And the next command buffer's sample reads them too.
    let status = run(
        device,
        &[Dispatch {
            state: &sample,
            texture: &texture,
            buffer: Some(&out),
            grid: [SIDE, SIDE],
        }],
    );
    assert_eq!(status, CommandBufferStatus::Completed);
    assert!(contents(&out, sampled.len()) == want, "the next sample too");
    drop((texture, buffer));
    true
}

/// A dispatch of `invert` (`rows_invert`) over `texture`'s texels.
fn inverting<'a>(invert: &'a ComputePipelineState, texture: &'a Texture) -> Dispatch<'a> {
    Dispatch {
        state: invert,
        texture,
        buffer: None,
        grid: [SIDE, SIDE],
    }
}

/// Shadows left unmade, made at use, and made in all, from `before` to `after`.
fn unmade_made(before: &WorkCounts, after: &WorkCounts) -> (u64, u64, u64) {
    (
        after.shadows_left_unmade - before.shadows_left_unmade,
        after.shadows_made_at_use - before.shadows_made_at_use,
        after.shadowed_textures - before.shadowed_textures,
    )
}

/// A texture whose rows are in place is made with its shadow left unmade: a kernel that only
/// writes it through its rows makes none and copies nothing; the first kernel that samples one
/// makes the shadow at its binding, its copy in after an earlier encoder's writes of the rows in
/// the same command buffer, and reads what a texture of the device's own holding those texels
/// gives; a `use_resource` of one makes its shadow at the call.
#[test]
fn a_shadow_is_made_at_the_first_reach_of_its_image() {
    let Some(device) = device() else { return };
    let taken: Vec<Over> = (OVER.into_iter())
        .filter(|&over| made_at_use(&device, over))
        .collect();
    if untaken_passes(&taken) {
        eprintln!("TEXEL-ROWS: this device lays the rows in place; nothing to test");
    }
}

/// [`a_shadow_is_made_at_the_first_reach_of_its_image`] over a buffer lying as `over` says;
/// `false` where the texture's rows are not read in place.
fn made_at_use(device: &Device, over: Over) -> bool {
    let pitch = SIDE * 4 + 16;
    let pages = Pages::new((pitch * SIDE) as usize, 11);
    let buffer = pages.buffer(device, over);
    let shape = (PixelFormat::BGRA8Unorm, SIDE, SIDE);
    let invert = state(device, "rows_invert");
    let sample = state(device, "rows_sample");
    let original = pages.bytes();
    let layout = (SIDE, SIDE, 4, pitch);
    let inverted: Vec<u8> = (tight(&original, layout).as_chunks::<4>().0.iter())
        .flat_map(|t| [255 - t[0], 255 - t[1], 255 - t[2], t[3]])
        .collect();
    // Written through its rows alone: no shadow made, nothing copied, the texels inverted in the
    // pages; dropped unmade.
    let before = device.work_counts();
    let (written, rows) = texture_over(device, (&buffer, over), shape, pitch);
    if !rows {
        return false;
    }
    let made = device.work_counts();
    assert_eq!(
        unmade_made(&before, &made),
        (1, 0, 0),
        "{over:?}: left unmade"
    );
    assert!(written.shadow().is_none(), "{over:?}: no shadow");
    let status = run(device, &[inverting(&invert, &written)]);
    let after = device.work_counts();
    assert_eq!(status, CommandBufferStatus::Completed);
    assert_eq!(
        unmade_made(&made, &after),
        (0, 0, 0),
        "{over:?}: written, none made"
    );
    assert_eq!(shadow_copies(&made, &after), 0, "{over:?}: nothing copied");
    assert!(written.shadow().is_none(), "{over:?}: still no shadow");
    assert!(
        tight(&pages.bytes(), layout) == inverted,
        "{over:?}: the texels inverted in the pages"
    );
    drop(written);
    // Inverted back through its rows, then sampled, in one command buffer: the sample's binding
    // makes the shadow, whose copy in follows the first encoder's writes.
    let before = device.work_counts();
    let (sampled, _) = texture_over(device, (&buffer, over), shape, pitch);
    let out = device
        .new_buffer(SIDE * SIDE * 16, ResourceOptions::shared())
        .unwrap();
    let made = device.work_counts();
    assert_eq!(
        unmade_made(&before, &made),
        (1, 0, 0),
        "{over:?}: left unmade"
    );
    let read = Dispatch {
        state: &sample,
        texture: &sampled,
        buffer: Some(&out),
        grid: [SIDE, SIDE],
    };
    let status = run(device, &[inverting(&invert, &sampled), read]);
    let after = device.work_counts();
    assert_eq!(status, CommandBufferStatus::Completed);
    assert_eq!(
        unmade_made(&made, &after),
        (0, 1, 1),
        "{over:?}: made at the sample"
    );
    assert!(shadow_copies(&made, &after) > 0, "{over:?}: copied in");
    assert!(sampled.shadow().is_some(), "{over:?}: a shadow now");
    let answer = contents(&out, (SIDE * SIDE * 16) as usize);
    let want = sampled_from(
        device,
        &sample,
        &own_texture(device, shape, 4, &tight(&original, layout)),
    );
    let stale = sampled_from(device, &sample, &own_texture(device, shape, 4, &inverted));
    assert_eq!(
        first_difference(&answer, &want, 16),
        None,
        "{over:?}: the sample reads the rows' writes (what the pages held before them: {})",
        first_difference(&answer, &stale, 16).is_none()
    );
    // A `use_resource` makes it at the call.
    let before = device.work_counts();
    let (used, _) = texture_over(device, (&buffer, over), shape, pitch);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    let made = device.work_counts();
    (encoder.use_resource(used.as_resource(), ResourceUsage::READ)).unwrap();
    let after = device.work_counts();
    assert_eq!(
        unmade_made(&before, &made),
        (1, 0, 0),
        "{over:?}: left unmade"
    );
    assert_eq!(
        unmade_made(&made, &after),
        (0, 1, 1),
        "{over:?}: made at use_resource"
    );
    assert!(used.shadow().is_some(), "{over:?}: a shadow now");
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    drop((sampled, used, buffer));
    true
}

/// What `sample` (`rows_sample`) answers for each texel of `texture`, one command buffer of its
/// own.
fn sampled_from(device: &Device, sample: &ComputePipelineState, texture: &Texture) -> Vec<u8> {
    let length = SIDE * SIDE * 16;
    let out = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let status = run(
        device,
        &[Dispatch {
            state: sample,
            texture,
            buffer: Some(&out),
            grid: [SIDE, SIDE],
        }],
    );
    assert_eq!(status, CommandBufferStatus::Completed);
    contents(&out, length as usize)
}

/// The rows' texels and the image route's, read through `rows_read` from two textures of
/// `format` holding the same `texels` (rows tight), as bits; and their stores of `values`
/// through `rows_write`, as bytes. The rows' texture lies over a buffer lying as `over` says.
/// `None` where the device lays the rows in place.
fn both_routes(
    (device, over): (&Device, Over),
    (format, width, height, texel): (PixelFormat, u64, u64, u64),
    texels: &[u8],
    values: &[f32],
) -> Option<[(Vec<u8>, Vec<u8>); 2]> {
    let pitch = width * texel + 16;
    let pages = Pages::new((pitch * height) as usize, 3);
    let layout = (width, height, texel, pitch);
    let mut bytes = pages.bytes();
    rows_of(&mut bytes, layout, |x, y, b| {
        texels[((y * width + x) * texel + b) as usize]
    });
    pages.write(0, &bytes);
    let buffer = pages.buffer(device, over);
    let (rows, taken) = texture_over(device, (&buffer, over), (format, width, height), pitch);
    if !taken {
        return None;
    }
    let image = own_texture(device, (format, width, height), texel, texels);
    let read = state(device, "rows_read");
    let write = state(device, "rows_write");
    let length = (width * height * 16) as usize;
    let mut reads = Vec::new();
    for texture in [&rows, &image] {
        let out = device
            .new_buffer(length as u64, ResourceOptions::shared())
            .unwrap();
        let status = run(
            device,
            &[Dispatch {
                state: &read,
                texture,
                buffer: Some(&out),
                grid: [width, height],
            }],
        );
        assert_eq!(status, CommandBufferStatus::Completed);
        reads.push(contents(&out, length));
    }
    let input: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let input = own_buffer(device, &input);
    let mut stores = Vec::new();
    for texture in [&rows, &image] {
        let status = run(
            device,
            &[Dispatch {
                state: &write,
                texture,
                buffer: Some(&input),
                grid: [width, height],
            }],
        );
        assert_eq!(status, CommandBufferStatus::Completed);
        let mut stored = vec![0; (width * height * texel) as usize];
        let region = Region::new_2d(0, 0, width, height);
        texture
            .get_bytes(&mut stored, width * texel, 0, region, 0, 0)
            .unwrap();
        stores.push(stored);
    }
    assert!(
        tight(&pages.bytes(), layout) == stores[0],
        "the rows' stores are in the pages"
    );
    drop((rows, buffer));
    let [a, b] = [reads, stores].map(|mut x| (x.swap_remove(0), x.swap_remove(0)));
    Some([a, b])
}

/// The first index at which `a` and `b` differ, as `what` reads it.
fn first_difference(a: &[u8], b: &[u8], unit: usize) -> Option<usize> {
    (a.chunks(unit).zip(b.chunks(unit))).position(|(x, y)| x != y)
}

/// Every `RGBA8Unorm` code read through the rows is the image route's texel, bit for bit, and a
/// sweep of floats stores the same codes.
#[test]
fn rgba8_unorm_rows_answer_the_image_route_s_texels() {
    let Some(device) = device() else { return };
    // 256 texels: every code in every channel, each channel a different order of them.
    let (width, height) = (64, 4);
    let texels: Vec<u8> = (0..256u32)
        .flat_map(|i| [i, 255 - i, (i * 7) & 255, (i * 13 + 5) & 255].map(|c| c as u8))
        .collect();
    // Each code's value, the midpoint above it, just under that midpoint, and values outside
    // [0, 1].
    let values: Vec<f32> = (0..256u32)
        .flat_map(|k| {
            let c = k as f32;
            [
                c / 255.0,
                (c + 0.5) / 255.0,
                (c + 0.499) / 255.0,
                (c - 128.0) / 64.0,
            ]
        })
        .collect();
    let shape = (PixelFormat::RGBA8Unorm, width, height, 4);
    let mut taken = Vec::new();
    for over in OVER {
        let Some([(rows_read, image_read), (rows_store, image_store)]) =
            both_routes((&device, over), shape, &texels, &values)
        else {
            continue;
        };
        assert_eq!(first_difference(&rows_read, &image_read, 16), None, "reads");
        assert_eq!(
            first_difference(&rows_store, &image_store, 4),
            None,
            "stores"
        );
        eprintln!(
            "TEXEL-ROWS: RGBA8Unorm over the {over:?}: 1024 codes read and 1024 floats stored \
             alike"
        );
        taken.push(over);
    }
    if untaken_passes(&taken) {
        eprintln!("TEXEL-ROWS: this device lays the rows in place; nothing to compare");
    }
}

/// A `half`'s value as a `float`.
fn half(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((h >> 10) & 0x1f);
    let mantissa = f32::from(h & 0x3ff);
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        31 if mantissa == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        e => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Every `RGBA16Float` code read through the rows is the image route's texel, bit for bit, and a
/// sweep of floats stores the same codes.
#[test]
fn rgba16_float_rows_answer_the_image_route_s_texels() {
    let Some(device) = device() else { return };
    // 16384 texels: every code once.
    let (width, height) = (128, 128);
    let texels: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
    // Each code's value, a value between it and the next, and its negation.
    let values: Vec<f32> = (0..16384u32)
        .flat_map(|t| {
            let h = (t * 4) as u16;
            let (v, next) = (half(h), half(h.wrapping_add(1)));
            [v, v + (next - v) * 0.5, v + (next - v) * 0.25, -v]
        })
        .collect();
    let shape = (PixelFormat::RGBA16Float, width, height, 8);
    let mut taken = Vec::new();
    for over in OVER {
        let Some([(rows_read, image_read), (rows_store, image_store)]) =
            both_routes((&device, over), shape, &texels, &values)
        else {
            continue;
        };
        assert_eq!(first_difference(&rows_read, &image_read, 16), None, "reads");
        assert_eq!(
            first_difference(&rows_store, &image_store, 8),
            None,
            "stores"
        );
        eprintln!(
            "TEXEL-ROWS: RGBA16Float over the {over:?}: 65536 codes read and 65536 floats \
             stored alike"
        );
        taken.push(over);
    }
    if untaken_passes(&taken) {
        eprintln!("TEXEL-ROWS: this device lays the rows in place; nothing to compare");
    }
}

/// The first dispatch of a new state over a texture, timed: the pipeline and its variant made.
fn first_dispatch(device: &Device, kernel: &str, texture: &Texture) -> Duration {
    let state = state(device, kernel);
    let start = Instant::now();
    let status = run(
        device,
        &[Dispatch {
            state: &state,
            texture,
            buffer: None,
            grid: [SIDE, SIDE],
        }],
    );
    let took = start.elapsed();
    assert_eq!(status, CommandBufferStatus::Completed);
    took
}

/// The first dispatch over the rows makes the rows variant; printed against the image route's,
/// cold (the first state) and warm (a second state of the same function).
#[test]
fn a_rows_variant_s_creation_is_timed() {
    let Some(device) = device() else { return };
    let pitch = SIDE * 4 + 16;
    let pages = Pages::new((pitch * SIDE) as usize, 5);
    let shape = (PixelFormat::RGBA8Unorm, SIDE, SIDE);
    // The first buffer whose texture's rows are read in place.
    let found = OVER.into_iter().find_map(|over| {
        let buffer = pages.buffer(&device, over);
        let (rows, taken) = texture_over(&device, (&buffer, over), shape, pitch);
        taken.then_some((rows, buffer, over))
    });
    let Some((rows, buffer, over)) = found else {
        untaken_passes(&[]);
        eprintln!("TEXEL-ROWS: this device lays the rows in place; nothing to time");
        return;
    };
    let image = own_texture(
        &device,
        (PixelFormat::RGBA8Unorm, SIDE, SIDE),
        4,
        &vec![0; (SIDE * SIDE * 4) as usize],
    );
    let base_cold = first_dispatch(&device, "rows_invert", &image);
    let rows_cold = first_dispatch(&device, "rows_invert", &rows);
    let base_warm = first_dispatch(&device, "rows_invert", &image);
    let rows_warm = first_dispatch(&device, "rows_invert", &rows);
    eprintln!(
        "TEXEL-ROWS: first dispatch of a new state, base cold {base_cold:?} warm {base_warm:?}, \
         rows (over the {over:?}) cold {rows_cold:?} warm {rows_warm:?}"
    );
    assert!(rows_cold < Duration::from_secs(20), "within the ceiling");
    drop((rows, buffer));
}
