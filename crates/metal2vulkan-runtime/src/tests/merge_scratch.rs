//! A command buffer's merges lend their scratch to its later encoders' merges
//! (`merge_scratch.rs`): one command buffer of hundreds of passes over a render target over a
//! buffer's bytes, each loading and storing it beside a cleared and stored `Depth32Float` depth
//! attachment and blending one rectangle over it, as a WebGPU frame split at every indexed draw
//! renders. The targets: over a `Shared` buffer, rows packed; over a `Private` buffer, packed rows
//! of a multiple of 256 bytes (a `-T` stencilprobe's, rounded up to 256); over a `Shared` buffer
//! of packed rows of no multiple of 256 bytes (an IOSurface's, `4 * width`); over a no-copy
//! buffer's host pages, rows likewise, in place where the device's in-place check passed for
//! render targets over such pages (`in_place.rs`) and, under `HostPageTexturePolicy::Shadow`
//! (`blit_holds::shadow_device`), a shadow merged back; and over a `Private` buffer whose rows
//! hold bytes between them, a shadow on every device (an image over them would hold those
//! bytes), copied back.
//!
//! With the device's rule forced (`DeviceShared::linear_rule_forced`), each pass renders a target
//! that is its own linear image apart, through the image it keeps, its changed words merged back
//! (`linear_attachment.rs`, `linear_merge.rs`), as NVIDIA's devices render it beside a depth
//! attachment: the passes form a run or two merged once each (`pass_runs.rs`,
//! `tests::linear_runs`), each merge taking one scratch, and the runtime's device memory grows by
//! no scratch a pass, over host pages as over the runtime's memory. A target that is a shadow merged back (host pages under the shadow
//! policy, and rows the device's linear images do not lay out, `shadow_merge.rs`) is kept and merged back
//! about its first pass, and kept before the second and merged back after the last, its later
//! passes one run (`pass_runs.rs`, `tests::shadow_runs`). Either way the texels equal those the same passes leave in a private
//! texture of the same first texels, and the bytes between the rows keep theirs.
//!
//! Each test waits for the device's in-place check to end before it makes anything ([`settled`]):
//! the check's own passes and memory are counted with the test's, and a target over host pages
//! takes its place for good only once the check's verdict for it is known.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{device, shadow_device, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::in_place::Verdict;
use crate::{
    BlitOption, Buffer, ClearColor, CommandQueue, Device, Error, HostMemory, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::{BlendFactor, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::alloc::Layout;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;
use std::time::Duration;

/// The targets' height and texel bytes (`RGBA8Unorm`).
pub(super) const H: u64 = 96;
pub(super) const TEXEL: u64 = 4;

/// The passes of the command buffer.
const PASSES: usize = 240;

/// The byte the bytes between a target's rows hold, which no pass writes.
const BETWEEN: u8 = 0xa5;

/// The state that blends a half-transparent rectangle (`tob_rect`, corners from vertex buffer 0)
/// of the color in fragment buffer 0 (`tob_solid`) over an `RGBA8Unorm` color, beside a
/// `Depth32Float` depth attachment it does not test.
pub(super) fn blending(device: &Device) -> RenderPipelineState {
    let library = library(device, "engine/tob_renders.ll");
    let function = |name: &str| {
        let function = library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(function("tob_solid"));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    color.blending_enabled = true;
    color.source_rgb_blend_factor = BlendFactor::SourceAlpha;
    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    color.source_alpha_blend_factor = BlendFactor::One;
    color.destination_alpha_blend_factor = BlendFactor::Zero;
    descriptor.color_attachments[0] = Some(color);
    descriptor.depth_format = Some(PixelFormat::Depth32Float);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// What a target's bytes hold first, of `width` texels a row, `row` bytes apart: each texel's
/// bytes a counting pattern of its place among the texels (so a target of any rows holds the
/// same texels), which no pass draws, and [`BETWEEN`] between the rows.
pub(super) fn first_bytes(width: u64, row: u64) -> Vec<u8> {
    (0..row * H)
        .map(|i| match i % row < width * TEXEL {
            true => ((i / row * width * TEXEL + i % row) * 7 % 251) as u8,
            false => BETWEEN,
        })
        .collect()
}

/// The texels of `bytes`, rows `row` bytes apart, packed.
pub(super) fn texels(bytes: &[u8], width: u64, row: u64) -> Vec<u8> {
    (bytes.chunks(row as usize))
        .flat_map(|r| &r[..(width * TEXEL) as usize])
        .copied()
        .collect()
}

/// A `Shared` buffer holding `bytes`.
pub(super) fn staged(device: &Device, bytes: &[u8]) -> Buffer {
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    write(&buffer, bytes);
    buffer
}

/// Write `bytes` at the start of `buffer`, host-visible, which no command uses now.
fn write(buffer: &Buffer, bytes: &[u8]) {
    let contents = buffer.contents().unwrap();
    // SAFETY: inside the buffer's length, which no command uses now.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
}

/// The first `length` bytes of `buffer`, host-visible, which no command uses now.
pub(super) fn read(buffer: &Buffer, length: u64) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: inside the buffer's length, which no command writes now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) }.to_vec()
}

/// Run one blit encoder's commands, `encode` encoding them, in a command buffer that completes.
pub(super) fn blit(queue: &CommandQueue, encode: impl FnOnce(&mut crate::BlitCommandEncoder)) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// The longest a test waits for the device's in-place check to end ([`settled`]).
const CHECK_WAIT: Duration = Duration::from_secs(120);

/// Whether `device` runs an in-place check of textures over host pages (`in_place_check.rs`):
/// under `HostPageTexturePolicy::InPlace`.
fn checks(device: &Device) -> bool {
    device.compile_device().policy().host_page_textures == HostPageTexturePolicy::InPlace
}

/// `device` once its in-place check of textures over host pages ended, where it runs one, waited
/// for before the test makes anything: for a test that reads the device's counts across windows
/// of its own commands ([`counts_alone`]). The check renders passes over linear images of its own,
/// on a thread of its own, which the device's counts take as they take the test's
/// (`WorkCounts::linear_passes_through_copies`) and the rule a test forces
/// (`DeviceShared::linear_rule_forced`) renders apart as it renders the test's; and it allocates
/// and frees memory of its own. Once every verdict is known it runs no more. And every texture the
/// test makes then takes its place for good, in place where the check passed for it, where one
/// made before a verdict is a shadow that may move in place at a later commit (`in_place.rs`).
pub(super) fn settled(device: Device) -> Device {
    if checks(&device) {
        assert!(
            device.settle_in_place_checks(CHECK_WAIT),
            "the device's in-place check ended"
        );
    }
    device
}

/// The device's counts at the start of a window of the test's own commands: only once its
/// in-place check ended, where it runs one ([`settled`]), so that the window's counts are the
/// test's work alone.
pub(super) fn counts_alone(device: &Device) -> WorkCounts {
    if checks(device) {
        let verdicts = device.shared().in_place.verdicts();
        assert!(
            !verdicts.contains(&Verdict::Pending),
            "a window of the device's counts opened while its in-place check runs"
        );
    }
    device.work_counts()
}

/// Page-aligned host memory of `length` bytes (whole pages of any host), freed when dropped:
/// declared before the device's buffer over it, so dropped after it.
pub(super) struct Host(NonNull<u8>, Layout);

impl Host {
    pub(super) fn new(length: u64) -> Host {
        let layout =
            Layout::from_size_align(length.next_multiple_of(64 << 10) as usize, 64 << 10).unwrap();
        // SAFETY: the layout's size is not zero.
        let pointer = unsafe { std::alloc::alloc_zeroed(layout) };
        Host(NonNull::new(pointer).unwrap(), layout)
    }

    /// The memory, for a buffer over it.
    ///
    /// # Safety
    ///
    /// As for `HostMemory::new`: it outlives every buffer over it.
    pub(super) unsafe fn memory(&self) -> HostMemory {
        // SAFETY: the caller's contract.
        unsafe { HostMemory::new(self.0) }
    }

    /// Write `bytes` at its start, which no command uses now.
    pub(super) fn write(&self, bytes: &[u8]) {
        assert!(bytes.len() <= self.1.size());
        // SAFETY: inside the allocation, which no command uses now.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.0.as_ptr(), bytes.len()) };
    }

    /// Its first `length` bytes, which no command writes now.
    pub(super) fn bytes(&self, length: u64) -> Vec<u8> {
        assert!(length as usize <= self.1.size());
        // SAFETY: inside the allocation, which no command writes now.
        unsafe { std::slice::from_raw_parts(self.0.as_ptr(), length as usize) }.to_vec()
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `new`.
        unsafe { std::alloc::dealloc(self.0.as_ptr(), self.1) };
    }
}

/// Where a target's bytes lie.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Over {
    /// A `Shared` buffer.
    Shared,
    /// A `Private` buffer: written and read through blits.
    Private,
    /// A no-copy buffer over host pages.
    HostPages,
}

/// A target of `width` texels a row, `row` bytes apart, over a buffer of `over`'s holding
/// [`first_bytes`]; `None`, printed, where the device imports no host memory.
pub(super) fn target(
    queue: &CommandQueue,
    over: Over,
    (width, row): (u64, u64),
    host: &Host,
) -> Option<(Buffer, Texture)> {
    let device = queue.device();
    let first = first_bytes(width, row);
    let length = row * H;
    let buffer = match over {
        Over::Shared => device
            .new_buffer(length, ResourceOptions::shared())
            .unwrap(),
        Over::Private => {
            let buffer = device
                .new_buffer(length, ResourceOptions::private())
                .unwrap();
            let source = staged(device, &first);
            blit(queue, |b| {
                (b.copy_from_buffer_to_buffer(&source, 0, &buffer, 0, length)).unwrap()
            });
            buffer
        }
        Over::HostPages => {
            // SAFETY: the memory outlives the buffer, and only its commands use it meanwhile.
            let made = unsafe {
                device.new_buffer_with_bytes_no_copy(
                    HostMemory::new(host.0),
                    host.1.size() as u64,
                    ResourceOptions::default(),
                    None,
                )
            };
            match made {
                Ok(buffer) => buffer,
                Err(Error::Unsupported(gap)) => {
                    eprintln!("MERGE-SCRATCH: the device imports no host memory: {gap:?}");
                    return None;
                }
                Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
            }
        }
    };
    if over != Over::Private {
        write(&buffer, &first);
    }
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, width, H, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = buffer.resource_options();
    let texture = buffer.new_texture(&descriptor, 0, row).unwrap();
    Some((buffer, texture))
}

/// The bytes of `buffer` (`length` of them), read through a blit where it is `Private`.
pub(super) fn bytes(queue: &CommandQueue, over: Over, buffer: &Buffer, length: u64) -> Vec<u8> {
    if over != Over::Private {
        return read(buffer, length);
    }
    let out = (queue.device())
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    blit(queue, |b| {
        (b.copy_from_buffer_to_buffer(buffer, 0, &out, 0, length)).unwrap()
    });
    read(&out, length)
}

/// Pass `i`'s rectangle, in normalized device coordinates (`[x0, y0, x1, y1]`), and its color:
/// rectangles over a grid of places, each overlapping its neighbours, half transparent.
pub(super) fn drawn(i: usize) -> ([f32; 4], [f32; 4]) {
    let (x, y) = ((i % 12) as f32, ((i / 12) % 6) as f32);
    let x0 = -1.0 + 0.125 * x;
    let y0 = -1.0 + 0.25 * y;
    let corners = [x0, y0, (x0 + 0.5).min(1.0), (y0 + 0.75).min(1.0)];
    let color = [
        (i % 5) as f32 / 4.0,
        (i % 7) as f32 / 6.0,
        (i % 3) as f32 / 2.0,
        0.5,
    ];
    (corners, color)
}

/// One command buffer of [`PASSES`] passes over `texture`, each loading and storing it beside a
/// `Depth32Float` depth attachment it clears and stores, blending its rectangle ([`drawn`]);
/// committed and completed. The device's work counts after the passes were encoded, before the
/// commit.
fn passes(queue: &CommandQueue, state: &RenderPipelineState, texture: &Texture) -> WorkCounts {
    let device = queue.device();
    let (width, height) = (texture.width(), texture.height());
    let mut d = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, width, height, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    let depth = device.new_texture(&d).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    for i in 0..PASSES {
        let mut descriptor = RenderPassDescriptor::new();
        let color = &mut descriptor.color_attachments[0];
        color.texture = Some(texture.clone());
        color.load_action = LoadAction::Load;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        descriptor.depth_attachment.texture = Some(depth.clone());
        descriptor.depth_attachment.load_action = LoadAction::Clear;
        descriptor.depth_attachment.store_action = StoreAction::Store;
        let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
        encoder.set_render_pipeline_state(state).unwrap();
        let (corners, rgba) = drawn(i);
        encoder
            .set_vertex_bytes(&bytes_of(&corners, f32::to_le_bytes), 0)
            .unwrap();
        encoder
            .set_fragment_bytes(&bytes_of(&rgba, f32::to_le_bytes), 0)
            .unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    let encoded = device.work_counts();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    encoded
}

/// The texels [`passes`] leave in a private texture of `width` texels a row holding
/// [`first_bytes`]'s texels first.
fn reference(queue: &CommandQueue, state: &RenderPipelineState, width: u64) -> Vec<u8> {
    let device = queue.device();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, width, H, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let texture = device.new_texture(&d).unwrap();
    let packed = width * TEXEL;
    let source = staged(device, &first_bytes(width, packed));
    let size = Size::new(width, H, 1);
    blit(queue, |b| {
        let origin = Origin::new(0, 0, 0);
        b.copy_from_buffer_to_texture(
            &source,
            0,
            packed,
            packed * H,
            size,
            &texture,
            0,
            0,
            origin,
            BlitOption::None,
        )
        .unwrap()
    });
    passes(queue, state, &texture);
    let out = device
        .new_buffer(packed * H, ResourceOptions::shared())
        .unwrap();
    blit(queue, |b| {
        let origin = Origin::new(0, 0, 0);
        b.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            origin,
            size,
            &out,
            0,
            packed,
            packed * H,
            BlitOption::None,
        )
        .unwrap()
    });
    read(&out, packed * H)
}

/// [`passes`] over a target over `over`'s buffer of `width` texels a row, `row` bytes apart, on
/// the device `device_of` makes, the rule forced: its texels equal the private texture's, the
/// bytes between its rows keep theirs, and its merges take few scratches (a linear image's, one
/// a run; a shadow's merged back, one keep and one merge for the run of passes), the device's
/// memory growing by less than an eighth of what a scratch a pass would take.
fn one_scratch(device_of: fn(&Path) -> Device, over: Over, (width, row): (u64, u64)) {
    let cache = Scratch::new(&format!("merge-scratch-{over:?}-{width}"));
    let host = Host::new(row * H);
    let device = settled(device_of(&cache.0));
    let queue = device.new_command_queue().unwrap();
    let state = blending(&device);
    let expected = reference(&queue, &state, width);
    let Some((buffer, texture)) = target(&queue, over, (width, row), &host) else {
        return;
    };
    let linear = texture.linear_image().unwrap();
    let shadow = texture.shadow().is_some();
    // A shadow merged back at each encoder's end: one over bytes another writer may write (all
    // but a tracked `Private` buffer's, whose shadow is copied back).
    let merged = texture.shadow().is_some_and(|copies| copies.merges_back());
    // The check's verdict for render targets over host pages, for texels that lie in lines as
    // the target's do: `RGBA8Unorm` rows `row` bytes apart from the buffer's start.
    let renders_in_place = {
        use crate::in_place::{Access, PagesClass, RowEnds};
        let layout = crate::linear_plan::BufferLayout {
            offset: 0,
            row_pitch: row,
            slice_pitch: row * H,
            row_bytes: width * 4,
            rows: H,
            slices: 1,
        };
        let start = buffer.contents().map_or(0, |at| at.as_ptr().addr() as u64);
        let checks = &device.shared().in_place;
        let rows = RowEnds::of(start, &layout, checks.line());
        checks.verdict(PagesClass::Pointer, Access::Renders, rows) == Verdict::Passed
    };
    println!(
        "MERGE-SCRATCH {over:?} {width}x{H}, rows of {row}: linear image {linear}, shadow {shadow} \
         (merged back {merged}), the in-place check passed for render targets over host pages \
         {renders_in_place}"
    );
    if matches!(over, Over::HostPages) {
        assert!(
            !linear || renders_in_place,
            "a linear image over host pages where the check did not pass for render targets"
        );
    }
    let before = counts_alone(&device);
    let forced = &device.shared().linear_rule_forced;
    forced.store(true, Ordering::Relaxed);
    let encoded = passes(&queue, &state, &texture);
    let after = device.work_counts();
    forced.store(false, Ordering::Relaxed);
    let delta = |of: fn(&WorkCounts) -> u64| of(&after) - of(&before);
    let apart = delta(|c| c.linear_passes_through_copies);
    let made = delta(|c| c.linear_merge_scratches_made);
    let reused = delta(|c| c.linear_merge_scratches_reused);
    let recorded = delta(|c| c.linear_merges_recorded);
    let taken_over = delta(|c| c.linear_merges_taken_over);
    let shadow_reused = delta(|c| c.shadow_merge_scratches_reused);
    let kept = delta(|c| c.shadow_texels_kept);
    let grown = encoded.device_memory_allocated - before.device_memory_allocated;
    // A merge of every texel's rows: its kept and changed parts.
    let scratch = 2 * width * TEXEL * H;
    println!(
        "MERGE-SCRATCH {over:?}: {apart} passes apart, linear merges {recorded} recorded and \
         {taken_over} taken over, linear scratches {made} made and {reused} reused, shadow \
         scratches {shadow_reused} reused, shadow texels kept {kept}; device memory grew {grown} \
         bytes while encoding ({} merges' scratch)",
        grown / scratch
    );
    match (linear, merged) {
        // One run (`pass_runs.rs`), or two where the second pass's box reaches past the first's
        // (then one over the whole target), merged once each.
        (true, _) => {
            assert_eq!(apart, PASSES as u64, "every pass renders the texture apart");
            assert!(
                recorded <= 2,
                "the passes merge once a run: {recorded} merges"
            );
            assert_eq!(
                recorded + taken_over,
                PASSES as u64,
                "every later pass takes its run's"
            );
            assert_eq!(
                made + reused,
                recorded,
                "each run's merge takes one scratch"
            );
        }
        // One run of passes (`pass_runs.rs`): kept twice, every texel, before the first pass,
        // which merges in place, and before the second, which opens the run.
        (false, true) => assert_eq!(kept, 2 * width * H, "the passes keep twice, as one run"),
        (false, false) => println!("MERGE-SCRATCH {over:?}: no merge"),
    }
    assert!(
        grown < scratch * PASSES as u64 / 8,
        "encoding the passes grew the device's memory by {grown} bytes, {} merges' scratch",
        grown / scratch
    );
    let pages = bytes(&queue, over, &buffer, row * H);
    let first = first_bytes(width, row);
    let between = (0..pages.len())
        .filter(|&i| first[i] == BETWEEN && (i as u64 % row) >= width * TEXEL)
        .filter(|&i| pages[i] != BETWEEN)
        .count();
    assert_eq!(between, 0, "bytes between the rows changed");
    let got = texels(&pages, width, row);
    let differ = (0..got.len()).filter(|&i| got[i] != expected[i]).count();
    assert_eq!(differ, 0, "the texels differ from the private texture's");
    drop((texture, buffer));
}

/// Over a `Shared` buffer, rows packed (1024 bytes).
#[test]
fn many_passes_over_a_shared_buffer_take_one_scratch() {
    one_scratch(device, Over::Shared, (256, 1024));
}

/// Over a `Private` buffer, packed rows of 1024 bytes, a multiple of 256 (a `-T` stencilprobe's
/// rows rounded up to 256 bytes, with no bytes between them).
#[test]
fn many_passes_over_a_private_buffer_take_one_scratch() {
    one_scratch(device, Over::Private, (256, 1024));
}

/// Over a `Private` buffer, rows of 200 texels rounded up to 1024 bytes: 224 bytes between them.
#[test]
fn many_passes_over_padded_private_rows_hold_their_texels() {
    one_scratch(device, Over::Private, (200, 1024));
}

/// Over a `Shared` buffer of packed rows of 300 texels, 1200 bytes, of no multiple of 256 (an
/// IOSurface's).
#[test]
fn many_passes_over_packed_rows_take_one_scratch() {
    one_scratch(device, Over::Shared, (300, 1200));
}

/// Over a no-copy buffer's host pages, packed rows of 1200 bytes, under the shadow policy: a
/// shadow merged back on every device.
#[test]
fn many_passes_over_host_pages_take_one_scratch() {
    one_scratch(shadow_device, Over::HostPages, (300, 1200));
}

/// Over a no-copy buffer's host pages, packed rows of 1200 bytes, under the default policy: a
/// linear image over the pages where the device's check passed for render targets there, its
/// passes one run or two; a shadow merged back elsewhere.
#[test]
fn many_passes_over_host_pages_in_place_take_one_scratch() {
    one_scratch(device, Over::HostPages, (300, 1200));
}

/// The encoders of [`two_ways_merges_take_few_scratches`]'s command buffer.
const TWO_WAYS: usize = 64;

/// One command buffer of [`TWO_WAYS`] compute encoders, each dispatching `tob_two_ways` once over
/// a shadowed texture over a `Private` buffer whose rows hold bytes between them (200 texels in
/// rows of 1024 bytes, a shadow copied back) and that buffer: it writes texel (0, 0) through the
/// texture and 4 bytes of the buffer, texel `1 + i` of the first row, another way. Each encoder
/// merges the texture's write through a copy (`shadow_merge.rs`, **Two ways**), whose kernel
/// reads bytes no copy writes (those between the rows), so its scratch is filled with zero before
/// its keep: every merge after the first takes a scratch an earlier encoder was done with, and
/// encoding them grows the device's memory by less than an eighth of what one a merge would take.
/// The bytes hold both writes of each encoder, and every other byte its first.
#[test]
fn two_ways_merges_take_few_scratches() {
    let (width, row) = (200, 1024);
    let cache = Scratch::new("merge-scratch-two-ways");
    let host = Host::new(row * H);
    let device = settled(device(&cache.0));
    let queue = device.new_command_queue().unwrap();
    let library = library(&device, "engine/tob_renders.ll");
    let function = library.new_function_with_name("tob_two_ways").unwrap();
    let kernel = device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap();
    let Some((buffer, _)) = target(&queue, Over::Private, (width, row), &host) else {
        return;
    };
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, width, H, false);
    descriptor.usage = TextureUsage::SHADER_WRITE | TextureUsage::SHADER_READ;
    descriptor.resource_options = buffer.resource_options();
    let texture = buffer.new_texture(&descriptor, 0, row).unwrap();
    assert!(texture.shadow().is_some(), "the texture is a shadow");
    let before = counts_alone(&device);
    let command_buffer = queue.command_buffer().unwrap();
    for i in 0..TWO_WAYS {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&kernel).unwrap();
        encoder.set_texture(Some(&texture), 0).unwrap();
        encoder.set_buffer(Some(&buffer), 0, 0).unwrap();
        let off = (TEXEL * (1 + i as u64)) as u32;
        encoder.set_bytes(&off.to_le_bytes(), 1).unwrap();
        (encoder.dispatch_threads(Size::new(1, 1, 1), Size::new(1, 1, 1))).unwrap();
        encoder.end_encoding().unwrap();
    }
    let encoded = device.work_counts();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let after = device.work_counts();
    let delta = |of: fn(&WorkCounts) -> u64| of(&after) - of(&before);
    let zeroed = delta(|c| c.shadow_merge_scratches_zeroed);
    let reused = delta(|c| c.shadow_merge_scratches_reused);
    let grown = encoded.device_memory_allocated - before.device_memory_allocated;
    // A merge through a copy of every texel's bytes: its kept, changed and current parts.
    let scratch = 3 * row * H;
    println!(
        "MERGE-SCRATCH two ways: {zeroed} merges' scratch zeroed, {reused} reused; device memory \
         grew {grown} bytes while encoding ({} merges' scratch)",
        grown / scratch
    );
    assert!(
        zeroed >= TWO_WAYS as u64,
        "every encoder's merge through a copy needs zero"
    );
    assert!(
        reused >= TWO_WAYS as u64 - 1,
        "every merge after the first takes a scratch: {reused}"
    );
    assert!(
        grown < scratch * TWO_WAYS as u64 / 8,
        "encoding the merges grew the device's memory by {grown} bytes, {} merges' scratch",
        grown / scratch
    );
    let mut want = first_bytes(width, row);
    want[..4].copy_from_slice(&[0xff, 0, 0, 0xff]);
    for i in 0..TWO_WAYS {
        let at = (TEXEL * (1 + i as u64)) as usize;
        want[at..at + 4].copy_from_slice(&[0x5a, 0x5b, 0x5c, 0x5d]);
    }
    let got = bytes(&queue, Over::Private, &buffer, row * H);
    let differ: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    assert!(
        differ.is_empty(),
        "{} bytes differ, the first at {:?}",
        differ.len(),
        differ.first()
    );
    drop((texture, buffer));
}
