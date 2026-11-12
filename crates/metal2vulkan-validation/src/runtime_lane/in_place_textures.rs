//! Device tests of textures over host pages in place (`HostPageTexturePolicy::InPlace`) on the
//! device the cfg selects: a texture over a no-copy buffer's pages or a buffer in fresh pages is an
//! image aliasing the pages where the device's check passed for their class and each of the
//! texture's accesses (one that reads, one that renders, one a shader writes), for texels that lie
//! in the device's lines of memory as its own do (`in_place_rows`), and a shadow elsewhere,
//! counted by why.
//!
//! - [`sampled_textures_over_host_pages_take_the_path_their_check_answered`], under the
//!   validation layer: once the device's check answered (printed, `IN-PLACE-CHECK`, with its
//!   time), a 1920x1080 `BGRA8Unorm` texture of packed rows and a 64x64 `RGBA16Float` one of
//!   padded rows, over a no-copy buffer of anonymous pages and over a buffer in fresh pages, once
//!   the device's submission checks answered whether its long buffers take them (a device made
//!   again where load split their tries; skipped, printed, only on a device whose held imports
//!   cost its submissions or which imports no host memory): where the class passed, each is in
//!   place (`vk_image` answers one; `host_textures_in_place`, `fresh_page_textures_in_place`) or
//!   a shadow its refusal counts; where it did not, a shadow counted by the check's answer.
//!   Either way a blit reads the pages' bytes, and a CPU write made after the command buffer's
//!   commit and before it runs (held by an event the CPU signals once it wrote) is what the blit
//!   reads: the image follows the buffer's rule (Host Write Ordering Guarantees). Printed
//!   (`IN-PLACE`): each texture's path.
//! - [`written_textures_over_host_pages_take_the_path_their_checks_answered`], under the
//!   validation layer: once the device's check answered, a 64x64 `RGBA16Float` texture of
//!   `ShaderRead` and `ShaderWrite` usage over a no-copy buffer of anonymous pages, with padded
//!   rows, with packed ones, and a 61x64 one at a 496-byte pitch (the pitch a driver that lays rows
//!   tight lays them at), every byte beside the texels a signaling NaN's half: where the check
//!   passed for an image a shader writes over the class (`Pointer writes: Passed`), each is in
//!   place (`vk_image` answers one; `host_textures_in_place`, `host_textures_in_place_written`) or
//!   a shadow its refusal counts; where it did not, a shadow counted by the check's answer, on a
//!   discrete GPU as on any other; where the rows' padding is free (`in_place::Padding`), the
//!   verdict is the one that lets it change (`Pointer buffer writes: Passed`). A kernel then
//!   writes every texel, with no usage shadow (`undeclared-usage-shadow`): every texel of the
//!   pages holds the value, the bytes past the last row hold the pattern, and so do the bytes
//!   between the rows but of a texture in place whose padding is free. Printed
//!   (`IN-PLACE-WRITTEN`): each texture's path.
//! - [`writes_and_passes_beyond_the_usage_reach_the_pages_where_their_checks_passed`], under the
//!   validation layer: an in-place 64x64 `RGBA16Float` texture of `ShaderRead` usage alone, with
//!   padded rows and with packed ones (where the device takes either in place), written by a
//!   kernel (no `ShaderWrite`) and rendered into by a pass that clears and stores (no
//!   `RenderTarget`): where the image in place took the usage it lacks (the first set of them the
//!   device makes the image with, the storage usage before the attachment usage, each where the
//!   device's check passed for a shader's writes, `Pointer buffer writes` where the rows' padding
//!   is free, or for renders over the class; a texture of the same shape that declares the set is
//!   placed in place, `Buffer::texture_placement`), the write or pass
//!   reaches the image in place: no shadow announced
//!   (`undeclared-usage-shadow`), no texel copied in (`shadow_texels_copied_in`,
//!   `shadow_texels_copied_in_images`); elsewhere through its shadow (or stand-in), announced.
//!   The image stays the texture's (`vk_image` the same handle), and its texels stay in the pages
//!   at the buffer's pitch: every texel of the pages then holds the value, and the bytes between
//!   the rows hold the pattern but where a shader's writes in place may change rows' padding that
//!   is free. Printed (`IN-PLACE-USAGE`): each form's path, the texels copied in, and the
//!   placement query's answers for the texture's shape declaring each set.
//! - [`textures_that_render_or_write_take_the_path_their_checks_answered`], under the validation
//!   layer: a 64x64 `R32Uint` texture over a no-copy buffer of anonymous pages, of each usage that
//!   renders or writes, once the device's check answered: each is in place where the check passed
//!   over the class for each of its accesses (`Pointer renders: Passed` for one that renders,
//!   `Pointer buffer writes: Passed` for one a shader writes, `ShaderWrite` or `ShaderAtomic`, its
//!   rows' padding free; `vk_image` answers one, `host_textures_in_place`, and
//!   `host_textures_in_place_written` for one a shader writes), else a shadow counted by the check's answer or the device's refusal; under
//!   `HostPageTexturePolicy::Shadow` every one is the exact form's (`host_texture_shadows_exact`).
//!   Each shadow that renders also counts in `host_texture_shadows_render_target`, and each one a
//!   shader writes in `host_texture_shadows_shader_write`.
//! - [`a_texture_made_before_its_verdict_moves_in_place_once_it_passes`], under the validation
//!   layer: on a device whose check has not run, a 256x64 `BGRA8Unorm` texture of `ShaderRead` and
//!   `RenderTarget` usage over a no-copy buffer of anonymous pages, of packed rows, is a shadow
//!   made before its verdict (`host_texture_shadows_check_pending`; `vk_image` answers none), and a
//!   blit reads the pages' bytes through it. Once the check answered, a texture of the same shape
//!   over other pages takes the path its verdicts give, and by the end of the next command buffer
//!   (a blit that reads the first texture, committed while no other is open) the first has moved
//!   to it: where the second is in place, the first is in place, moved at that commit or at one
//!   of the check's own passes, each committed while no other command buffer was open
//!   (`host_textures_moved_in_place`, `host_textures_in_place`; `vk_image` answers one), and
//!   its shadow's image went once the command buffer that took the move completed
//!   (`host_texture_shadows_freed_after_move`, one per move, the in-place counts as before); a
//!   blit then reads a CPU write made after
//!   its commit and before it runs with no shadow copy (`shadow_copies`), and a pass that clears
//!   and stores it writes the clear into the pages, every byte past its texels kept; else it stays
//!   a shadow, no move counted. Printed (`IN-PLACE-MOVE`): each texture's path and the counts.
//! - [`a_texture_an_attached_read_copied_before_its_verdict_moves_in_place_too`], under the
//!   validation layer: the same with a 64x64 `RGBA8Unorm` texture that a pass renders and a draw
//!   of it reads at its own pixel (`tob_read`, no barrier) before its verdict, which takes the
//!   texture's usage shadow of the pages' bytes (`attachment_reads_through_copies`). Once the
//!   check answered, the next such pass's commit moves it where a texture of the same shape made
//!   then is in place (`host_textures_moved_in_place`, its shadow's image freed); a pass after it
//!   copies back and keeps no texel (`shadow_texels_copied_back`, `shadow_texels_kept`), and its
//!   read still draws the texels from before the pass, through its copy, each pass's pages then
//!   all the red the CPU wrote before it. Printed (`IN-PLACE-ATTACHED-MOVE`): the paths and counts.
//! - [`the_guests_surface_lies_in_place_where_its_checks_passed`], under the validation layer: the
//!   guest's 3664x2039 `BGRA8Unorm` surface over a no-copy buffer of anonymous pages, at a 14848-byte
//!   pitch (the next multiple of 256, at which RADV lays linear rows) and at 14656 bytes, of
//!   `ShaderRead` usage, with `RenderTarget`, and with `ShaderWrite`: each is in place only where
//!   the check passed for its access and row ends (`Pointer`, `Pointer renders`, `Pointer buffer
//!   writes` where its padding is free), and where it passed, in place or refused by the device's
//!   image (`host_texture_shadows_refused_*`). A blit then reads the pages' texels through it, a
//!   pass that clears and stores it, or a kernel that writes every texel, leaves its texels in the
//!   pages; the bytes between the rows and past the last hold the pattern, but rows' padding that
//!   is free under a shader's writes. Printed (`IN-PLACE-SURFACE`): each case's placement query
//!   answer, path, verdict and refusals.
//! - [`a_failed_check_keeps_the_shadow_and_no_record`], under the validation layer: a device made
//!   to fail its check (`DeviceOptions::fail_in_place_check`) answers failed or unchecked for every
//!   class and makes such a texture a shadow counted as the failed check's, never in place, and
//!   keeps no record: the next device on the same shader cache recalls nothing and runs the check,
//!   whose record the device after it recalls (`in_place_checks_recalled`): every verdict it
//!   measured, passed or failed, and those it could not check for a cause that lasts for the
//!   driver, with no check run where it recalls all; one it could not check for a cause that may
//!   pass is no measurement, which that device checks again, answering as before.
//! - [`in_place_and_shadow_costs`], without the validation layer, report only (built in release for
//!   the numbers): a `BGRA8Unorm` surface over a no-copy buffer of anonymous pages, read by a
//!   fragment function each frame, under each setting: a 3840x2160 one through a 3840x32 strip,
//!   and a 1920x1080 one through a 1920x32 strip, whole, and whole in each of three passes of one
//!   command buffer; the median and fastest GPU time of a frame's command buffer
//!   (`IN-PLACE-COST`). The shadow copies the whole surface in once a command buffer, into the
//!   device's own memory, where the image in place reads the pages at each pass (over the bus, on
//!   a discrete card).

use super::degrade_notices::counts;
use super::first_use_writes::CHECK_WAIT;
use super::host_memory::{counting, no_copy, not_importable, Mapping, GUEST_PAGE};
use super::in_place_rows;
use super::no_copy_textures::{floats, staged};
use super::pitched_surfaces::reading;
use super::{layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, BufferContents, BufferDescriptor, ClearColor, CommandBuffer, CommandQueue,
    ComputePipelineState, Degradation, Device, DeviceOptions, EventRef, LoadAction, Origin,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, SharedEvent, Size,
    StoreAction, Texture, TextureDescriptor, TexturePlacement, WorkCounts,
};

/// The bytes of a buffer of the runtime's own that lies in fresh pages (past half a pooled block).
const FRESH: u64 = 40 << 20;

/// The lane's options with textures over host pages taken in place where the check passes: the
/// lane's devices otherwise take the exact answers, under which every one is a shadow.
pub(super) fn in_place(options: &mut DeviceOptions) {
    options.policy.host_page_textures = HostPageTexturePolicy::InPlace;
}

/// Whether the device's check passed for `what` (`Pointer`, `Fresh renders`, `DmaBuf`, ...), for
/// a 2D texture of `(w, h, bpr)` bytes (a row's texels, its rows, their pitch) at the start of
/// `buffer` (`in_place_rows`).
pub(super) fn passed(device: &Device, what: &str, buffer: &Buffer, (w, h, bpr): Rows) -> bool {
    let first = buffer.contents().map_or(0, |at| at.as_ptr() as u64);
    let on_lines = in_place_rows::on_lines(device, first, w, bpr, h);
    in_place_rows::passed(device, what, on_lines)
}

/// Whether the device's check passed for a shader's writes over `class` (`Pointer`, ...) of a 2D
/// texture of `(w, h, bpr)` bytes at the start of `buffer`, which holds no IOSurface plane: those
/// that let its rows' padding change where it is free (`in_place_rows::padding_free`), else those
/// that keep it.
pub(super) fn writes_passed(
    device: &Device,
    class: &str,
    buffer: &Buffer,
    (w, h, bpr): Rows,
) -> bool {
    let first = buffer.contents().map_or(0, |at| at.as_ptr() as u64);
    let held = buffer.length() >= h * bpr;
    let free = in_place_rows::padding_free(device, first, bpr, h, held);
    passed(
        device,
        &in_place_rows::writes(class, free),
        buffer,
        (w, h, bpr),
    )
}

/// A 2D texture's rows: the bytes of a row's texels, its rows, and their pitch.
pub(super) type Rows = (u64, u64, u64);

/// The verdicts the device's check gives, each a line of its report before what it found
/// changed: seven classes (`Pointer`, `Fresh`, `DmaBuf`, `Pooled`, `ExportedFd`, `ExportedDmaBuf`,
/// `FdBlocks`), seven accesses, two row ends. The report gives every class on every device, one
/// the device has nothing of to check reading `Unchecked`: `FdBlocks` does on a device that
/// allocates no block of its buffers exportable, every discrete device among them (it has no
/// unified memory), because it makes no such block, not because a check failed. Held to the
/// report's own count ([`verdicts_in`]).
const VERDICTS: usize = 98;

/// The verdicts of a device's check `report`, counted from its own lines: the report gives one line
/// per class for a texture that reads, each named by its class alone (`Pointer: Passed`), then one
/// per class, in that order, for each other access and row end (`Pointer renders: Passed`), then
/// what the check found changed. So the classes are the first lines' names, and each further run
/// of as many lines that names them in order, under a suffix no run had, is one more access or row
/// end; the first that does not is past the verdicts.
fn verdicts_in(report: &[String]) -> usize {
    let key = |line: &String| line.split_once(": ").map(|(key, _)| key.to_owned());
    let classes: Vec<String> = (report.iter())
        .map_while(|line| key(line).filter(|key| !key.contains([' ', ','])))
        .collect();
    assert!(!classes.is_empty(), "no class in the report: {report:?}");
    let mut suffixes = std::collections::HashSet::new();
    let runs = (report.chunks_exact(classes.len()))
        .map_while(|run| {
            let tails: Option<Vec<String>> = (run.iter().zip(&classes))
                .map(|(line, class)| key(line)?.strip_prefix(class.as_str()).map(str::to_owned))
                .collect();
            let tails = tails?;
            tails
                .iter()
                .all(|tail| *tail == tails[0])
                .then(|| tails[0].clone())
        })
        .take_while(|suffix| suffixes.insert(suffix.clone()))
        .count();
    classes.len() * runs
}

/// The devices made again where load beside the submission checks split their import tax's tries
/// ([`answered_device`]).
const DEVICES: usize = 3;

/// The lane's device in place, once its submission checks answered whether a held import costs a
/// submission, which places its long buffers in fresh pages or not (`fresh_pages.rs`): a device
/// whose tries all split under the load beside it keeps the answer unknown for its life and
/// records nothing, so the next device measures again, up to [`DEVICES`] of them. Each one's
/// verdicts are printed.
fn answered_device(messages: &Messages, cache: &std::path::Path) -> Device {
    let mut made = 0;
    loop {
        made += 1;
        let device = validated_device_with(TEST_DEVICE, messages, cache, in_place).unwrap();
        assert!(
            device.settle_submit_checks(CHECK_WAIT),
            "the submission checks answered"
        );
        let report = device.submit_check_report();
        eprintln!("IN-PLACE-SUBMIT-CHECKS on {}: {report}", device.name());
        let split = report.contains("import tax Unchecked") && report.contains("tries split");
        if !split || made == DEVICES {
            return device;
        }
    }
}

/// The device's check answered, its verdicts and time printed.
fn settled(device: &Device) {
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "the check answered"
    );
    for line in device.in_place_check_report() {
        eprintln!("IN-PLACE-CHECK on {}: {line}", device.name());
    }
    eprintln!(
        "IN-PLACE-CHECK on {}: took {:?}",
        device.name(),
        device.in_place_check_took()
    );
}

/// The shadows over host pages refused an image in place between `before` and `after`.
pub(super) fn refused(before: &WorkCounts, after: &WorkCounts) -> u64 {
    let of = |counts: &WorkCounts| {
        counts.host_texture_shadows_refused_rows
            + counts.host_texture_shadows_refused_detached
            + counts.host_texture_shadows_refused_image
            + counts.host_texture_shadows_refused_pitch
            + counts.host_texture_shadows_refused_placement
            + counts.host_texture_shadows_refused_source
            + counts.host_texture_shadows_refused_dma_buf
    };
    of(after) - of(before)
}

/// A texture of `format`, `w` by `h`, `ShaderRead` alone, over `buffer` with rows `bpr` apart.
pub(super) fn sampled(
    buffer: &Buffer,
    format: PixelFormat,
    (w, h, bpr): (u64, u64, u64),
) -> Texture {
    let mut d = TextureDescriptor::texture_2d(format, w, h, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = buffer.resource_options();
    buffer.new_texture(&d, 0, bpr).unwrap()
}

/// A texture of `format`, `w` by `h`, of `ShaderRead` and `RenderTarget` usage, over `buffer`
/// with rows `bpr` apart.
fn rendered(buffer: &Buffer, format: PixelFormat, (w, h, bpr): (u64, u64, u64)) -> Texture {
    let mut d = TextureDescriptor::texture_2d(format, w, h, false);
    d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    d.resource_options = buffer.resource_options();
    buffer.new_texture(&d, 0, bpr).unwrap()
}

/// A texture of `format`, `w` by `h`, of `ShaderRead` and `ShaderWrite` usage, over `buffer` with
/// rows `bpr` apart.
fn written(buffer: &Buffer, format: PixelFormat, (w, h, bpr): (u64, u64, u64)) -> Texture {
    let mut d = TextureDescriptor::texture_2d(format, w, h, false);
    d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    d.resource_options = buffer.resource_options();
    buffer.new_texture(&d, 0, bpr).unwrap()
}

/// Encodes `kernel` (`write_at_level` of the level-writes lane) writing every texel of `texture`'s
/// level 0, `w` by `h`, in threadgroups of 8 by 8 (the last ones of the grid cut to it).
fn encode_writes(
    command_buffer: &CommandBuffer,
    kernel: &ComputePipelineState,
    texture: &Texture,
    (w, h): (u64, u64),
) {
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(kernel).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_bytes(&0u16.to_le_bytes(), 0).unwrap();
    encoder
        .dispatch_threads(Size::new(w, h, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
}

/// The texels of `texture` (`w` by `h`, `texel` bytes each) as a blit reads them into tight rows,
/// in a command buffer held by `hold` until the CPU signals it: `write` runs once the command
/// buffer is committed, before the signal.
fn read_held(
    queue: &CommandQueue,
    texture: &Texture,
    (w, h, texel): (u64, u64, u64),
    hold: &SharedEvent,
    value: u64,
    write: impl FnOnce(),
) -> Vec<u8> {
    let device = queue.device();
    let row = w * texel;
    let out = device
        .new_buffer(row * h, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::from(hold), value)
        .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(w, h, 1),
        &out,
        0,
        row,
        row * h,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    write();
    hold.set_signaled_value(value);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's `row * h` bytes, which no command writes any longer.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), (row * h) as usize) }.to_vec()
}

/// `bytes`' texels of `w` by `h` (`texel` bytes each) laid out `bpr` apart, as tight rows.
fn texels_of(bytes: &[u8], (w, h, texel): (u64, u64, u64), bpr: u64) -> Vec<u8> {
    (0..h)
        .flat_map(|y| {
            let at = (y * bpr) as usize;
            bytes[at..at + (w * texel) as usize].to_vec()
        })
        .collect()
}

#[test]
fn sampled_textures_over_host_pages_take_the_path_their_check_answered() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let mut lapses = Vec::new();
    {
        let device = answered_device(&messages, cache.path());
        settled(&device);
        let queue = device.new_command_queue().unwrap();
        let hold = device.new_shared_event().unwrap();
        // Whether the device imports host memory at all, as the Pointer class finds.
        let mut imports = true;
        let mut value = 0;
        let shapes = [
            (PixelFormat::BGRA8Unorm, 4u64, 1920u64, 1080u64, 7680u64),
            (PixelFormat::RGBA16Float, 8, 64, 64, 1024),
        ];
        for class in ["Pointer", "Fresh"] {
            for (format, texel, w, h, bpr) in shapes {
                let len = ((bpr * h) as usize + GUEST_PAGE).next_multiple_of(GUEST_PAGE);
                // The pages: anonymous ones mapped here, or the fresh pages of a long buffer.
                let mapping = (class == "Pointer").then(|| Mapping::anonymous(len));
                let buffer = match &mapping {
                    // SAFETY: the mapping outlives the buffer and the texture (dropped first).
                    Some(mapping) => match unsafe {
                        no_copy(
                            &device,
                            mapping.pointer(),
                            len,
                            ResourceOptions::default(),
                            None,
                        )
                    } {
                        Ok(buffer) => buffer,
                        Err(error) if not_importable(&error) => {
                            eprintln!("IN-PLACE: the device imports no host memory: {error:?}");
                            imports = false;
                            continue;
                        }
                        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
                    },
                    None => device.new_buffer(FRESH, ResourceOptions::shared()).unwrap(),
                };
                // A long buffer of the device's own lies in fresh pages where its submission
                // checks found a held import adds nothing to a submission (`fresh_pages.rs`),
                // made then or parked by an earlier one of its length: none does on a device
                // whose held imports cost its submissions, or which imports no host memory, and
                // those alone skip the class.
                if class == "Fresh" && !device.fresh_pages_pay() {
                    let report = device.submit_check_report();
                    let taxed = report.contains("import tax Holds");
                    assert!(
                        taxed || !imports,
                        "a long buffer took no fresh pages on {}, which imports host memory and \
                         whose held imports were not found to cost its submissions: {report}",
                        device.name()
                    );
                    eprintln!(
                        "IN-PLACE on {}: its long buffers take no fresh pages (held imports cost \
                         its submissions {taxed}, it imports host memory {imports}): {report}",
                        device.name()
                    );
                    continue;
                }
                let contents = buffer.contents().unwrap();
                let write = |bytes: &[u8]| {
                    // SAFETY: the buffer's first `len` bytes, which no command reads meanwhile
                    // (the blit waits for the event the CPU signals after this).
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            contents.as_ptr(),
                            bytes.len(),
                        )
                    }
                };
                let first = counting(len, 3);
                write(&first);
                let before = device.work_counts();
                let texture = sampled(&buffer, format, (w, h, bpr));
                let after = device.work_counts();
                let aliased = after.host_textures_in_place - before.host_textures_in_place;
                let fresh =
                    after.fresh_page_textures_in_place - before.fresh_page_textures_in_place;
                let refused = refused(&before, &after);
                // SAFETY: the handle is only compared, while the texture lives.
                let image = unsafe { texture.vk_image() };
                let what = format!("{class} {format:?} {w}x{h}/{bpr}");
                let path = match (image.is_some(), aliased, refused) {
                    (true, 1, 0) => "in place",
                    (false, 0, 1) => "refused",
                    (false, 0, 0) => "shadow",
                    _ => "inconsistent",
                };
                eprintln!(
                    "IN-PLACE on {}: {what}: {path} (in place {aliased}, fresh {fresh}, refused \
                     {refused})",
                    device.name()
                );
                let checked = passed(&device, class, &buffer, (w * texel, h, bpr));
                let right = match (checked, path) {
                    (true, "in place") => (class == "Fresh") == (fresh == 1),
                    (true, "refused") => true,
                    (false, "shadow") => true,
                    _ => false,
                };
                if !right {
                    lapses.push(format!("{what}: {path}, the check passed: {checked}"));
                }
                // The bytes the CPU wrote before, then those it wrote after the commit.
                value += 1;
                let read = read_held(&queue, &texture, (w, h, texel), &hold, value, || {});
                if read != texels_of(&first, (w, h, texel), bpr) {
                    lapses.push(format!("{what}: a blit read other than the pages"));
                }
                let later = counting(len, 101);
                value += 1;
                let read = read_held(&queue, &texture, (w, h, texel), &hold, value, || {
                    write(&later)
                });
                if read != texels_of(&later, (w, h, texel), bpr) {
                    lapses.push(format!(
                        "{what}: a blit read other than the CPU wrote after its commit"
                    ));
                }
                drop((texture, buffer));
                drop(mapping);
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
    assert!(lapses.is_empty(), "{}", lapses.join("\n"));
}

/// The shadow images of moved textures freed since `before` once they reach `moved`, or [`WAIT`]
/// passed: freed as the command buffer that took a move is let go, which another thread may do
/// (the check's own commits).
fn freed_shadows(device: &Device, before: &WorkCounts, moved: u64) -> u64 {
    let started = std::time::Instant::now();
    loop {
        let freed = device.work_counts().host_texture_shadows_freed_after_move
            - before.host_texture_shadows_freed_after_move;
        if freed >= moved || started.elapsed() > WAIT {
            return freed;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

/// The path of a texture over host pages: its image in place (`vk_image` answers one) or a shadow.
fn path_of(texture: &Texture) -> &'static str {
    // SAFETY: the handle is only compared, while the texture lives.
    match unsafe { texture.vk_image() } {
        Some(_) => "in place",
        None => "shadow",
    }
}

#[test]
fn a_texture_made_before_its_verdict_moves_in_place_once_it_passes() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let mut lapses = Vec::new();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), in_place).unwrap();
        let queue = device.new_command_queue().unwrap();
        let hold = device.new_shared_event().unwrap();
        let (format, texel, w, h) = (PixelFormat::BGRA8Unorm, 4u64, 256u64, 64u64);
        let bpr = w * texel;
        let shape = (w, h, texel);
        let len = ((bpr * h) as usize + GUEST_PAGE).next_multiple_of(GUEST_PAGE);
        let mappings = [Mapping::anonymous(len), Mapping::anonymous(len)];
        let mut buffers = Vec::new();
        for mapping in &mappings {
            // SAFETY: the mappings outlive the buffers and the textures (dropped first).
            match unsafe {
                no_copy(
                    &device,
                    mapping.pointer(),
                    len,
                    ResourceOptions::default(),
                    None,
                )
            } {
                Ok(buffer) => buffers.push(buffer),
                Err(error) if not_importable(&error) => {
                    eprintln!("IN-PLACE-MOVE: the device imports no host memory: {error:?}");
                    return;
                }
                Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
            }
        }
        let write = |buffer: &Buffer, bytes: &[u8]| {
            let contents = buffer.contents().unwrap();
            // SAFETY: the buffer's first `len` bytes, which no command uses meanwhile (a blit
            // that reads them waits for the event the CPU signals after this).
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) }
        };
        let pages = |buffer: &Buffer| -> Vec<u8> {
            let contents = buffer.contents().unwrap();
            // SAFETY: the buffer's first `len` bytes, which no command writes any longer.
            unsafe { std::slice::from_raw_parts(contents.as_ptr(), len) }.to_vec()
        };
        // A blit's read of `texture`, its command buffer committed while no other is open.
        let read = |texture: &Texture| -> Vec<u8> {
            let out = device
                .new_buffer(bpr * h, ResourceOptions::shared())
                .unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_texture_to_buffer(
                texture,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(w, h, 1),
                &out,
                0,
                bpr,
                bpr * h,
                BlitOption::None,
            )
            .unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
            let contents = out.contents().unwrap();
            // SAFETY: the buffer's `bpr * h` bytes, which no command writes any longer.
            unsafe { std::slice::from_raw_parts(contents.as_ptr(), (bpr * h) as usize) }.to_vec()
        };
        let first = counting(len, 7);
        write(&buffers[0], &first);
        let before = device.work_counts();
        let early = rendered(&buffers[0], format, (w, h, bpr));
        let made = device.work_counts();
        let pending =
            made.host_texture_shadows_check_pending - before.host_texture_shadows_check_pending;
        let early_path = path_of(&early);
        if (early_path, pending) != ("shadow", 1) {
            lapses.push(format!(
                "made before its verdict: {early_path}, counted pending {pending}"
            ));
        }
        if read(&early) != texels_of(&first, shape, bpr) {
            lapses.push("before the verdict: a blit read other than the pages".to_owned());
        }
        let before = device.work_counts();
        settled(&device);
        // A quiet commit after the verdicts moves the first texture where a texture made now is
        // in place: the check's own (its passes are committed while no other command buffer is
        // open) or the program's next.
        let moved_by_the_check =
            device.work_counts().host_textures_moved_in_place - before.host_textures_moved_in_place;
        let late = rendered(&buffers[1], format, (w, h, bpr));
        let late_path = path_of(&late);
        let moved_read = read(&early);
        let after = device.work_counts();
        let moved = after.host_textures_moved_in_place - before.host_textures_moved_in_place;
        // Counted in place as it moved, beside the second where that is in place.
        let late_in_place = u64::from(late_path == "in place");
        let in_place = after.host_textures_in_place - before.host_textures_in_place - late_in_place;
        let path = path_of(&early);
        // Its shadow's image goes once the command buffer that took the move completed, on the
        // thread that lets that command buffer go: by the read's wait where its commit took it.
        let freed = freed_shadows(&device, &before, moved);
        eprintln!(
            "IN-PLACE-MOVE on {}: made before its verdict: {early_path} (pending {pending}); one \
             made after it: {late_path}; the first after the next commit: {path} (moved {moved}, \
             {moved_by_the_check} of them at the check's commits, in place {in_place}, shadows \
             freed {freed})",
            device.name()
        );
        if freed != moved {
            lapses.push(format!(
                "after the move: {freed} shadow images freed, where {moved} moved"
            ));
        }
        if moved_read != texels_of(&first, shape, bpr) {
            lapses.push("at the move: a blit read other than the pages".to_owned());
        }
        let expected = match late_path {
            "in place" => ("in place", 1, 1),
            _ => ("shadow", 0, 0),
        };
        if (path, moved, in_place) != expected {
            lapses.push(format!(
                "after the next commit: {path} (moved {moved}, in place {in_place}), where one \
                 made after the verdict is {late_path}"
            ));
        }
        // A CPU write made after the commit and before the blit runs, read with no copy.
        let later = counting(len, 101);
        let before = device.work_counts();
        let held = read_held(&queue, &early, shape, &hold, 1, || {
            write(&buffers[0], &later)
        });
        let after = device.work_counts();
        let copies = after.shadow_copies - before.shadow_copies;
        if held != texels_of(&later, shape, bpr) {
            lapses.push("after the move: a blit read other than the CPU wrote".to_owned());
        }
        eprintln!(
            "IN-PLACE-MOVE on {}: a held read after it took {copies} shadow copies",
            device.name()
        );
        if path == "in place" && copies != 0 {
            lapses.push(format!("in place, a read took {copies} shadow copies"));
        }
        // A pass that clears and stores it: the clear in the pages, the bytes past them kept.
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(early.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 1.0, 0.0, 1.0);
        let command_buffer = queue.command_buffer().unwrap();
        let encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let cleared = pages(&buffers[0]);
        let green = [0x00, 0xff, 0x00, 0xff];
        let texels = (bpr * h) as usize;
        if !cleared[..texels].chunks(4).all(|t| t == green) {
            lapses.push("the pass's clear is not in the pages".to_owned());
        }
        if cleared[texels..] != later[texels..] {
            lapses.push("the pass changed a byte past the texels".to_owned());
        }
        drop((early, late, buffers));
        drop(mappings);
        for notice in messages.take_notices() {
            eprintln!("IN-PLACE-MOVE notice: {}", notice.text);
        }
    }
    layer_reported_nothing_at_all(&messages);
    assert!(lapses.is_empty(), "{}", lapses.join("\n"));
}

/// One pass over `target`, loaded and stored: `fill` draws [`DRAWN_RED`] over the block
/// `[w/4, w/2) x [h/4, h/2)`, then `read` (`tob_read`, at its own pixel, with no barrier) draws
/// the target's texels there, an attached read; committed and waited for.
fn fill_then_read_own(
    queue: &CommandQueue,
    target: &Texture,
    (fill, read): (&RenderPipelineState, &RenderPipelineState),
) {
    let mut d = RenderPassDescriptor::new();
    let color = &mut d.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&d).unwrap();
    // The block's corners in normalized device coordinates: x0, y1, x1, y0.
    let block = floats([-0.5, 0.0, 0.0, 0.5]);
    encoder.set_render_pipeline_state(fill).unwrap();
    encoder.set_vertex_bytes(&block, 0).unwrap();
    let red = f32::from(DRAWN_RED) / 255.0;
    encoder
        .set_fragment_bytes(&floats([red, 0.0, 0.0, 1.0]), 0)
        .unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.set_render_pipeline_state(read).unwrap();
    encoder.set_vertex_bytes(&block, 0).unwrap();
    encoder.set_fragment_texture(Some(target), 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// The red [`fill_then_read_own`]'s fill draws.
const DRAWN_RED: u8 = 128;

#[test]
fn a_texture_an_attached_read_copied_before_its_verdict_moves_in_place_too() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let mut lapses = Vec::new();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), in_place).unwrap();
        let queue = device.new_command_queue().unwrap();
        let (format, w, h) = (PixelFormat::RGBA8Unorm, 64u64, 64u64);
        let bpr = w * 4;
        let texels = (bpr * h) as usize;
        let len = texels.next_multiple_of(GUEST_PAGE);
        let mappings = [Mapping::anonymous(len), Mapping::anonymous(len)];
        let mut buffers = Vec::new();
        for mapping in &mappings {
            // SAFETY: the mappings outlive the buffers and the textures (dropped first).
            match unsafe {
                no_copy(
                    &device,
                    mapping.pointer(),
                    len,
                    ResourceOptions::default(),
                    None,
                )
            } {
                Ok(buffer) => buffers.push(buffer),
                Err(error) if not_importable(&error) => {
                    eprintln!(
                        "IN-PLACE-ATTACHED-MOVE: the device imports no host memory: {error:?}"
                    );
                    return;
                }
                Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
            }
        }
        // Every texel of the pages red `red`, opaque, as the CPU writes it.
        let paint = |red: u8| {
            let contents = buffers[0].contents().unwrap();
            let bytes = [red, 0, 0, 255].repeat(texels / 4);
            // SAFETY: the texels, which no command uses meanwhile.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), texels) }
        };
        // Whether every texel of the pages is red `red`: the fill's block too, which the read
        // drew over with the texels from before the pass, Metal's answer at a draw's own pixel.
        let all_red = |red: u8| {
            let contents = buffers[0].contents().unwrap();
            // SAFETY: the texels, which no command writes any longer.
            let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), texels) };
            bytes.chunks(4).all(|texel| texel == [red, 0, 0, 255])
        };
        let (fill, read) = (
            reading(&device, format, "tob_solid"),
            reading(&device, format, "tob_read"),
        );
        let before = device.work_counts();
        let early = rendered(&buffers[0], format, (w, h, bpr));
        let made = device.work_counts();
        let pending =
            made.host_texture_shadows_check_pending - before.host_texture_shadows_check_pending;
        let early_path = path_of(&early);
        if (early_path, pending) != ("shadow", 1) {
            lapses.push(format!(
                "made before its verdict: {early_path}, counted pending {pending}"
            ));
        }
        // Before the verdict: the attached read takes the texture's usage shadow of its bytes.
        paint(64);
        let counted = device.work_counts();
        fill_then_read_own(&queue, &early, (&fill, &read));
        let through = device.work_counts().attachment_reads_through_copies
            - counted.attachment_reads_through_copies;
        if through != 1 || !all_red(64) {
            lapses.push(format!(
                "before the verdict: {through} reads through copies, the pages all red 64 {}",
                all_red(64)
            ));
        }
        let before = device.work_counts();
        settled(&device);
        let late = rendered(&buffers[1], format, (w, h, bpr));
        let late_path = path_of(&late);
        // The next quiet commit moves it where a texture made now is in place.
        paint(32);
        fill_then_read_own(&queue, &early, (&fill, &read));
        let after = device.work_counts();
        let moved = after.host_textures_moved_in_place - before.host_textures_moved_in_place;
        let path = path_of(&early);
        let freed = freed_shadows(&device, &before, moved);
        if !all_red(32) {
            lapses.push("at the move: the pages are not all red 32".to_owned());
        }
        // In place, a pass writes the pages: no texel copied back or kept, the attached read
        // still through its copy, the texels from before the pass at its own pixels.
        paint(16);
        let counted = device.work_counts();
        fill_then_read_own(&queue, &early, (&fill, &read));
        let after = device.work_counts();
        let back = (after.shadow_texels_copied_back - counted.shadow_texels_copied_back)
            + (after.shadow_texels_kept - counted.shadow_texels_kept);
        let through =
            after.attachment_reads_through_copies - counted.attachment_reads_through_copies;
        eprintln!(
            "IN-PLACE-ATTACHED-MOVE on {}: made before its verdict: {early_path} (pending \
             {pending}); one made after it: {late_path}; the first after the next commit: {path} \
             (moved {moved}, shadows freed {freed}); a pass after it: {back} texels copied back \
             or kept, {through} reads through copies",
            device.name()
        );
        if !all_red(16) || through != 1 {
            lapses.push(format!(
                "after the move: the pages all red 16 {}, {through} reads through copies",
                all_red(16)
            ));
        }
        let expected = match late_path {
            "in place" => ("in place", 1, 0),
            _ => ("shadow", 0, back),
        };
        if (path, moved, back) != expected || freed != moved {
            lapses.push(format!(
                "after the next commit: {path} (moved {moved}, freed {freed}, {back} texels \
                 copied back or kept), where one made after the verdict is {late_path}"
            ));
        }
        drop((early, late, buffers));
        drop(mappings);
        for notice in messages.take_notices() {
            eprintln!("IN-PLACE-ATTACHED-MOVE notice: {}", notice.text);
        }
    }
    layer_reported_nothing_at_all(&messages);
    assert!(lapses.is_empty(), "{}", lapses.join("\n"));
}

/// The `RGBA16Float` texel the kernel writes and the pass clears to: `half4(0.25, 0.5, 0.75, 1)`.
const VALUE: [u8; 8] = [0x00, 0x34, 0x00, 0x38, 0x00, 0x3a, 0x00, 0x3c];

#[test]
fn written_textures_over_host_pages_take_the_path_their_checks_answered() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let mut lapses = Vec::new();
    {
        // Each texture would announce a usage shadow, not only the device's first.
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            in_place(options);
            options
                .announce_each
                .insert(Degradation::UndeclaredUsageShadow);
        })
        .unwrap();
        settled(&device);
        let queue = device.new_command_queue().unwrap();
        let kernel = super::level_writes_by_usage::kernel(&device, "write_at_level");
        let h = 64u64;
        // Padded rows, packed ones, and rows of a width no multiple of 4 at the 16-byte pitch a
        // driver that lays rows tight lays them at: a device whose linear images take no explicit
        // pitch takes the last two.
        for (w, bpr) in [(64u64, 1024u64), (64, 512), (61, 496)] {
            // A page past the image, whose bytes no write reaches.
            let len = ((bpr * h) as usize + GUEST_PAGE).next_multiple_of(GUEST_PAGE);
            let mapping = Mapping::anonymous(len);
            // Every byte that is no texel a signaling NaN's half (`0x7d01`), whose bits a driver
            // that read and wrote it back as a float would quiet.
            let mut pattern = counting(len, 13);
            for (offset, byte) in pattern.iter_mut().enumerate() {
                let offset = offset as u64;
                if offset >= h * bpr || offset % bpr >= w * 8 {
                    *byte = [0x01, 0x7d][(offset % 2) as usize];
                }
            }
            mapping.write(0, &pattern);
            // SAFETY: the mapping outlives the buffer and the texture (dropped first).
            let buffer = match unsafe {
                no_copy(
                    &device,
                    mapping.pointer(),
                    len,
                    ResourceOptions::default(),
                    None,
                )
            } {
                Ok(buffer) => buffer,
                Err(error) if not_importable(&error) => {
                    eprintln!("IN-PLACE-WRITTEN: the device imports no host memory: {error:?}");
                    return;
                }
                Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
            };
            let checked = writes_passed(&device, "Pointer", &buffer, (w * 8, h, bpr));
            // Rows whose padding is free (`in_place::Padding`): in place, a shader's writes may
            // change it, as Metal's do on an x86 Mac's GPU.
            let first = buffer.contents().map_or(0, |at| at.as_ptr() as u64);
            let free = in_place_rows::padding_free(&device, first, bpr, h, true);
            let before = device.work_counts();
            let texture = written(&buffer, PixelFormat::RGBA16Float, (w, h, bpr));
            let after = device.work_counts();
            let aliased = after.host_textures_in_place - before.host_textures_in_place;
            let writes =
                after.host_textures_in_place_written - before.host_textures_in_place_written;
            let by_check = |counts: &WorkCounts| {
                counts.host_texture_shadows_check_pending
                    + counts.host_texture_shadows_check_failed
                    + counts.host_texture_shadows_unchecked
            };
            let unchecked = by_check(&after) - by_check(&before);
            let refused = refused(&before, &after);
            // SAFETY: the handle is only compared, while the texture lives.
            let image = unsafe { texture.vk_image() };
            let what = format!("RGBA16Float {w}x{h}/{bpr}");
            let path = match (image.is_some(), aliased, writes, refused, unchecked) {
                (true, 1, 1, 0, 0) => "in place",
                (false, 0, 0, 1, 0) => "refused",
                (false, 0, 0, 0, 1) => "shadow",
                _ => "inconsistent",
            };
            eprintln!(
                "IN-PLACE-WRITTEN on {} ({:?}): {what}: {path} (in place {aliased}, written \
                 {writes}, refused {refused}, by the check {unchecked})",
                device.name(),
                device.properties().kind
            );
            let right = matches!(
                (checked, path),
                (true, "in place") | (true, "refused") | (false, "shadow")
            );
            if !right {
                lapses.push(format!("{what}: {path}, the check passed: {checked}"));
            }
            messages.take_notices();
            let command_buffer = queue.command_buffer().unwrap();
            encode_writes(&command_buffer, &kernel, &texture, (w, h));
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None, "{what}");
            let notices = counts(&messages.take_notices());
            if notices.contains_key(&Degradation::UndeclaredUsageShadow) {
                lapses.push(format!("{what}: a usage shadow for a declared write"));
            }
            // SAFETY: as above.
            if unsafe { texture.vk_image() } != image {
                lapses.push(format!("{what}: the image changed"));
            }
            let now = mapping.read(0..len);
            for y in 0..h {
                for x in 0..w {
                    let at = (y * bpr + x * 8) as usize;
                    if now[at..at + 8] != VALUE {
                        lapses.push(format!("{what}: texel {x},{y}"));
                    }
                }
                let between = (y * bpr + w * 8) as usize..((y + 1) * bpr) as usize;
                let kept = !(free && path == "in place");
                if kept && now[between.clone()] != pattern[between] {
                    lapses.push(format!("{what}: row {y}'s padding"));
                }
            }
            let past = (h * bpr) as usize..len;
            if now[past.clone()] != pattern[past] {
                lapses.push(format!("{what}: the bytes past the last row"));
            }
            drop((texture, buffer));
            drop(mapping);
        }
    }
    layer_reported_nothing_at_all(&messages);
    assert!(lapses.is_empty(), "{}", lapses.join("\n"));
}

#[test]
fn writes_and_passes_beyond_the_usage_reach_the_pages_where_their_checks_passed() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        // Each form announces its shadow, not only the device's first.
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            in_place(options);
            options
                .announce_each
                .insert(Degradation::UndeclaredUsageShadow);
        })
        .unwrap();
        settled(&device);
        let queue = device.new_command_queue().unwrap();
        let (w, h) = (64u64, 64u64);
        // Padded rows, then packed ones: a device whose linear images take no padded pitch takes
        // the second.
        for (form, bpr) in [
            ("kernel", 1024u64),
            ("pass", 1024),
            ("kernel", 512),
            ("pass", 512),
        ] {
            let len = ((bpr * h) as usize).next_multiple_of(GUEST_PAGE);
            let form_at = format!("{form}, rows of {bpr} bytes");
            let form = form_at.as_str();
            let mapping = Mapping::anonymous(len);
            let pattern = counting(len, 9);
            mapping.write(0, &pattern);
            // SAFETY: the mapping outlives the buffer and the texture (dropped first).
            let buffer = unsafe {
                no_copy(
                    &device,
                    mapping.pointer(),
                    len,
                    ResourceOptions::default(),
                    None,
                )
            }
            .unwrap();
            if !passed(&device, "Pointer", &buffer, (w * 8, h, bpr)) {
                eprintln!(
                    "IN-PLACE-USAGE on {}: {form}: no image in place to write",
                    device.name()
                );
                continue;
            }
            let texture = sampled(&buffer, PixelFormat::RGBA16Float, (w, h, bpr));
            // SAFETY: the handle is only compared, while the texture lives.
            let Some(image) = (unsafe { texture.vk_image() }) else {
                eprintln!(
                    "IN-PLACE-USAGE on {}: {form}: refused in place",
                    device.name()
                );
                continue;
            };
            // The image in place takes the first set of the usages the texture lacks that the
            // device makes it in place with, the most first and the storage usage before the
            // attachment usage, each where the device's check passed for what it lets reach the
            // pages: a shader's writes (those that let the rows' padding change, where it is
            // free), or renders. Whether the device makes it in place with a set is what the
            // placement query answers for a texture of the same shape that declares it: the
            // check's verdicts for its accesses, and the image's making up to the bind.
            let kernel = form.starts_with("kernel");
            let placed = |usage: TextureUsage| {
                let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, w, h, false);
                d.usage = TextureUsage::SHADER_READ | usage;
                d.resource_options = buffer.resource_options();
                buffer.texture_placement(&d, 0, bpr).unwrap()
            };
            let answers = [
                TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET,
                TextureUsage::SHADER_WRITE,
                TextureUsage::RENDER_TARGET,
            ]
            .map(placed);
            let [both, writes, renders] = answers.map(|answer| answer == TexturePlacement::InPlace);
            let reaches = match kernel {
                true => both || writes,
                false => both || (renders && !writes),
            };
            // The check's verdicts the query asked: a texture that declares the usage alone is in
            // place only where its check passed.
            assert!(
                !writes || writes_passed(&device, "Pointer", &buffer, (w * 8, h, bpr)),
                "{form}: a written texture in place where its check did not pass"
            );
            assert!(
                !renders || passed(&device, "Pointer renders", &buffer, (w * 8, h, bpr)),
                "{form}: a rendered texture in place where its check did not pass"
            );
            let first = buffer.contents().map_or(0, |at| at.as_ptr() as u64);
            let free = kernel && in_place_rows::padding_free(&device, first, bpr, h, true);
            let before = device.work_counts();
            messages.take_notices();
            let command_buffer = queue.command_buffer().unwrap();
            match kernel {
                true => {
                    let kernel = super::level_writes_by_usage::kernel(&device, "write_at_level");
                    let mut encoder = command_buffer.compute_command_encoder().unwrap();
                    encoder.set_compute_pipeline_state(&kernel).unwrap();
                    encoder.set_texture(Some(&texture), 0).unwrap();
                    encoder.set_bytes(&0u16.to_le_bytes(), 0).unwrap();
                    encoder
                        .dispatch_threadgroups(Size::new(w / 8, h / 8, 1), Size::new(8, 8, 1))
                        .unwrap();
                    encoder.end_encoding().unwrap();
                }
                false => {
                    let mut pass = RenderPassDescriptor::new();
                    let color = &mut pass.color_attachments[0];
                    color.texture = Some(texture.clone());
                    color.load_action = LoadAction::Clear;
                    color.store_action = StoreAction::Store;
                    color.clear_color = ClearColor::new(0.25, 0.5, 0.75, 1.0);
                    let encoder = command_buffer.render_command_encoder(&pass).unwrap();
                    encoder.end_encoding().unwrap();
                }
            }
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None, "{form}");
            let notices = counts(&messages.take_notices());
            let shadowed = notices.contains_key(&Degradation::UndeclaredUsageShadow);
            let after = device.work_counts();
            // A shadow's copies in from a buffer's texels, and from an image's (the usage shadow
            // copies the image in place, image to image).
            let copied = (after.shadow_texels_copied_in - before.shadow_texels_copied_in)
                + (after.shadow_texels_copied_in_images - before.shadow_texels_copied_in_images);
            // In place, the write or pass reaches the texels with no shadow and no copy; else it
            // takes the shadow (or stand-in), announced.
            assert_eq!(shadowed, !reaches, "{form}: {notices:?}");
            if reaches {
                assert_eq!(copied, 0, "{form}: texels copied in");
            }
            // SAFETY: as above.
            assert_eq!(unsafe { texture.vk_image() }, Some(image), "{form}");
            let now = mapping.read(0..len);
            for y in 0..h {
                for x in 0..w {
                    let at = (y * bpr + x * 8) as usize;
                    assert_eq!(now[at..at + 8], VALUE, "{form}: texel {x},{y}");
                }
                // A shader's writes in place may change rows' padding that is free.
                let between = (y * bpr + w * 8) as usize..((y + 1) * bpr) as usize;
                if !(reaches && free) {
                    assert_eq!(
                        now[between.clone()],
                        pattern[between],
                        "{form}: row {y}'s padding"
                    );
                }
            }
            let path = match reaches {
                true => "in place, no shadow",
                false => "through its shadow",
            };
            eprintln!(
                "IN-PLACE-USAGE on {}: {form}: {path} (texels copied in {copied}); declaring \
                 ShaderWrite and RenderTarget {:?}, ShaderWrite {:?}, RenderTarget {:?}",
                device.name(),
                answers[0],
                answers[1],
                answers[2],
            );
            drop((texture, buffer));
            drop(mapping);
        }
    }
    layer_reported_nothing_at_all(&messages);
}

#[test]
fn textures_that_render_or_write_take_the_path_their_checks_answered() {
    use TextureUsage as U;
    let usages = [
        U::SHADER_READ | U::RENDER_TARGET,
        U::SHADER_READ | U::SHADER_WRITE,
        U::SHADER_READ | U::SHADER_ATOMIC,
        U::SHADER_WRITE | U::SHADER_ATOMIC,
        U::SHADER_WRITE | U::RENDER_TARGET,
        U::SHADER_ATOMIC | U::RENDER_TARGET,
        U::UNKNOWN,
    ];
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    for policy in [
        HostPageTexturePolicy::InPlace,
        HostPageTexturePolicy::Shadow,
    ] {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy.host_page_textures = policy;
        })
        .unwrap();
        if policy == HostPageTexturePolicy::InPlace {
            settled(&device);
        }
        let (w, h, bpr) = (64u64, 64u64, 256u64);
        let len = ((bpr * h) as usize).next_multiple_of(GUEST_PAGE);
        let mapping = Mapping::anonymous(len);
        // SAFETY: the mapping outlives the buffer and the textures (dropped first).
        let buffer = match unsafe {
            no_copy(
                &device,
                mapping.pointer(),
                len,
                ResourceOptions::default(),
                None,
            )
        } {
            Ok(buffer) => buffer,
            Err(error) if not_importable(&error) => {
                eprintln!("IN-PLACE-SHADOWS: the device imports no host memory: {error:?}");
                return;
            }
            Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
        };
        // The verdicts its `R32Uint` texels take.
        let rows = (w * 4, h, bpr);
        let renders_in_place = passed(&device, "Pointer renders", &buffer, rows);
        let writes_in_place = writes_passed(&device, "Pointer", &buffer, rows);
        for usage in usages {
            let mut d = TextureDescriptor::texture_2d(PixelFormat::R32Uint, w, h, false);
            d.usage = usage;
            d.resource_options = buffer.resource_options();
            let before = device.work_counts();
            let texture = match buffer.new_texture(&d, 0, bpr) {
                Ok(texture) => texture,
                // A usage the device refuses over a buffer counts nothing.
                Err(error) => {
                    eprintln!(
                        "IN-PLACE-SHADOWS on {}: {usage:?}: refused: {error:?}",
                        device.name()
                    );
                    continue;
                }
            };
            let after = device.work_counts();
            let delta = |counts: fn(&WorkCounts) -> u64| counts(&after) - counts(&before);
            let counted = (
                delta(|c| c.host_texture_shadows_render_target),
                delta(|c| c.host_texture_shadows_shader_write),
                delta(|c| c.host_texture_shadows_exact),
            );
            // An `Unknown` usage is every usage, as Metal reports it.
            let unknown = usage == U::UNKNOWN;
            let renders = unknown || usage.contains(U::RENDER_TARGET);
            let writes = unknown || usage.intersects(U::SHADER_WRITE | U::SHADER_ATOMIC);
            // SAFETY: the handle is only read, while the texture lives.
            let aliased = unsafe { texture.vk_image() }.is_some();
            eprintln!(
                "IN-PLACE-SHADOWS on {}: {policy:?} {usage:?}: (render target, shader write, \
                 exact) {counted:?}, in place {aliased}",
                device.name()
            );
            let shadow = (u64::from(renders), u64::from(writes));
            match policy {
                HostPageTexturePolicy::Shadow => {
                    assert_eq!(counted, (shadow.0, shadow.1, 1), "{policy:?} {usage:?}");
                    assert!(!aliased, "{usage:?}");
                }
                // In place where the check passed for each of its accesses, else a shadow
                // counted by the check's answer or the device's refusal.
                _ => {
                    let in_place = delta(|c| c.host_textures_in_place);
                    let written = delta(|c| c.host_textures_in_place_written);
                    if aliased {
                        let checked =
                            (!renders || renders_in_place) && (!writes || writes_in_place);
                        assert!(checked, "{usage:?} in place without its checks");
                        let due = ((0, 0, 0), 1, u64::from(writes));
                        assert_eq!((counted, in_place, written), due, "{usage:?}");
                    } else {
                        let due = ((shadow.0, shadow.1, 0), 0, 0);
                        assert_eq!((counted, in_place, written), due, "{usage:?}");
                    }
                }
            }
            drop(texture);
        }
        messages.take_notices();
        drop((buffer, device));
        drop(mapping);
    }
    layer_reported_nothing_at_all(&messages);
}

/// The guest's surface: 3664x2039 `BGRA8Unorm` (16-pixel rows of 14656 bytes), at that pitch and
/// at 14848 bytes, the next multiple of 256.
const SURFACE_PITCHES: [(u64, u64, u64); 2] = [(3664, 2039, 14848), (3664, 2039, 14656)];

/// What a pass that clears to (0.25, 0.5, 0.75, 1) and a kernel that writes it (`write_at_level`)
/// leave in a `BGRA8Unorm` texel: B, G, R and A, each within 1 of it.
const SURFACE_TEXEL: [u8; 4] = [191, 128, 64, 255];

#[test]
fn the_guests_surface_lies_in_place_where_its_checks_passed() {
    use TextureUsage as U;
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), in_place).unwrap();
    settled(&device);
    let queue = device.new_command_queue().unwrap();
    let kernel = super::level_writes_by_usage::kernel(&device, "write_at_level");
    let accesses = [
        ("reads", U::SHADER_READ),
        ("renders", U::SHADER_READ | U::RENDER_TARGET),
        ("writes", U::SHADER_READ | U::SHADER_WRITE),
    ];
    for (w, h, bpr) in SURFACE_PITCHES {
        for (access, usage) in accesses {
            let case = format!("{w}x{h} at {bpr}, {access}");
            let len = ((bpr * h) as usize).next_multiple_of(GUEST_PAGE);
            let mapping = Mapping::anonymous(len);
            let pattern = counting(len, 5);
            mapping.write(0, &pattern);
            // SAFETY: the mapping outlives the buffer and the texture (dropped first).
            let buffer = match unsafe {
                no_copy(
                    &device,
                    mapping.pointer(),
                    len,
                    ResourceOptions::default(),
                    None,
                )
            } {
                Ok(buffer) => buffer,
                Err(error) if not_importable(&error) => {
                    eprintln!("IN-PLACE-SURFACE: the device imports no host memory: {error:?}");
                    return;
                }
                Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
            };
            let rows = (w * 4, h, bpr);
            let checked = match access {
                "reads" => passed(&device, "Pointer", &buffer, rows),
                "renders" => passed(&device, "Pointer renders", &buffer, rows),
                _ => writes_passed(&device, "Pointer", &buffer, rows),
            };
            let first = buffer.contents().map_or(0, |at| at.as_ptr() as u64);
            let free =
                access == "writes" && in_place_rows::padding_free(&device, first, bpr, h, true);
            let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, w, h, false);
            d.usage = usage;
            d.resource_options = buffer.resource_options();
            let placement = buffer.texture_placement(&d, 0, bpr).unwrap();
            let before = device.work_counts();
            let texture = buffer.new_texture(&d, 0, bpr).unwrap();
            let after = device.work_counts();
            // SAFETY: the handle is only read, while the texture lives.
            let aliased = unsafe { texture.vk_image() }.is_some();
            let delta = |counts: fn(&WorkCounts) -> u64| counts(&after) - counts(&before);
            let pitch_refused = delta(|c| c.host_texture_shadows_refused_pitch);
            eprintln!(
                "IN-PLACE-SURFACE on {}: {case}: placement {placement:?}, in place {aliased} \
                 (verdict passed {checked}, padding free {free}, refused for its pitch \
                 {pitch_refused}, shadows refused {})",
                device.name(),
                refused(&before, &after)
            );
            // In place only where its check passed; where the check passed, in place or refused
            // by the device's image at that pitch.
            assert!(!aliased || checked, "{case}: in place without its check");
            if checked {
                assert!(
                    aliased || refused(&before, &after) == 1,
                    "{case}: {placement:?}"
                );
            }
            if aliased {
                assert_eq!(delta(|c| c.host_textures_in_place), 1, "{case}");
            }
            // The access itself: a blit reads it; a pass clears and stores it; a kernel writes
            // every texel. Then the pages hold the texels the access leaves, and the bytes
            // between the rows hold the pattern but where a shader's writes may change rows'
            // padding that is free.
            let command_buffer = queue.command_buffer().unwrap();
            let read = device
                .new_buffer(w * 4 * h, ResourceOptions::shared())
                .unwrap();
            match access {
                "renders" => {
                    let mut pass = RenderPassDescriptor::new();
                    let color = &mut pass.color_attachments[0];
                    color.texture = Some(texture.clone());
                    color.load_action = LoadAction::Clear;
                    color.store_action = StoreAction::Store;
                    color.clear_color = ClearColor::new(0.25, 0.5, 0.75, 1.0);
                    let encoder = command_buffer.render_command_encoder(&pass).unwrap();
                    encoder.end_encoding().unwrap();
                }
                "writes" => encode_writes(&command_buffer, &kernel, &texture, (w, h)),
                _ => {
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit.copy_from_texture_to_buffer(
                        &texture,
                        0,
                        0,
                        Origin::new(0, 0, 0),
                        Size::new(w, h, 1),
                        &read,
                        0,
                        w * 4,
                        w * 4 * h,
                        BlitOption::None,
                    )
                    .unwrap();
                    blit.end_encoding().unwrap();
                }
            }
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None, "{case}");
            let now = mapping.read(0..len);
            if access == "reads" {
                let contents = read.contents().unwrap();
                // SAFETY: the buffer's bytes, which no command writes any longer.
                let got =
                    unsafe { std::slice::from_raw_parts(contents.as_ptr(), (w * 4 * h) as usize) };
                assert!(
                    got == texels_of(&pattern, (w, h, 4), bpr).as_slice(),
                    "{case}: the blit's texels"
                );
            }
            for y in 0..h {
                let at = (y * bpr) as usize;
                let row = &now[at..at + (w * 4) as usize];
                if access != "reads" {
                    let off = row.as_chunks::<4>().0.iter().position(|texel| {
                        texel
                            .iter()
                            .zip(SURFACE_TEXEL)
                            .any(|(&got, due)| got.abs_diff(due) > 1)
                    });
                    assert_eq!(off, None, "{case}: row {y}'s texels");
                }
                let between = at + (w * 4) as usize..((y + 1) * bpr) as usize;
                if !free {
                    assert!(
                        now[between.clone()] == pattern[between],
                        "{case}: row {y}'s padding"
                    );
                }
            }
            let past = (h * bpr) as usize..len;
            assert!(
                now[past.clone()] == pattern[past],
                "{case}: past the last row"
            );
            let notices = counts(&messages.take_notices());
            eprintln!(
                "IN-PLACE-SURFACE on {}: {case}: done, notices {notices:?}",
                device.name()
            );
            drop((texture, read, buffer));
            drop(mapping);
        }
    }
    layer_reported_nothing_at_all(&messages);
}

#[test]
fn a_check_report_s_verdicts_are_counted_from_its_lines() {
    let report: Vec<String> = [
        "Pointer: Passed",
        "Fresh: Unchecked",
        "Pointer renders: Passed",
        "Fresh renders: Failed",
        "Pointer, rows on lines: Passed",
        "Fresh, rows on lines: Passed",
        "Pointer renders, rows on lines: Passed",
        "Fresh renders, rows on lines: Passed",
        "Fresh renders: no pages to check: none",
        "Fresh renders, rows on lines: changed byte 3",
    ]
    .map(str::to_owned)
    .to_vec();
    assert_eq!(verdicts_in(&report), 8);
    assert_eq!(verdicts_in(&report[..8]), 8);
    assert_eq!(verdicts_in(&report[..5]), 4);
}

#[test]
fn a_failed_check_keeps_the_shadow_and_no_record() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            in_place(options);
            options.fail_in_place_check = true;
        })
        .unwrap();
        settled(&device);
        let report = device.in_place_check_report();
        assert_eq!(
            verdicts_in(&report),
            VERDICTS,
            "the check's verdicts, one per class, access and row end: {report:?}"
        );
        assert!(
            report
                .iter()
                .take(VERDICTS)
                .all(|line| !line.ends_with("Passed")),
            "{report:?}"
        );
        let len = (7680 * 1080usize).next_multiple_of(GUEST_PAGE);
        let mapping = Mapping::anonymous(len);
        // SAFETY: the mapping outlives the buffer and the texture (dropped first).
        if let Ok(buffer) = unsafe {
            no_copy(
                &device,
                mapping.pointer(),
                len,
                ResourceOptions::default(),
                None,
            )
        } {
            let before = device.work_counts();
            let texture = sampled(&buffer, PixelFormat::BGRA8Unorm, (1920, 1080, 7680));
            let after = device.work_counts();
            // SAFETY: as above.
            assert!(unsafe { texture.vk_image() }.is_none());
            assert_eq!(after.host_textures_in_place, before.host_textures_in_place);
            let failed =
                after.host_texture_shadows_check_failed - before.host_texture_shadows_check_failed;
            let unchecked =
                after.host_texture_shadows_unchecked - before.host_texture_shadows_unchecked;
            assert_eq!(failed + unchecked, 1, "{report:?}");
            let counted = after.in_place_checks_failed + after.in_place_checks_unchecked;
            assert_eq!(after.in_place_checks_run, counted);
            drop((texture, buffer));
        }
        drop(mapping);
        messages.take_notices();
    }
    // Nothing kept: the next device on the cache recalls nothing and checks. Its verdicts, one
    // per class, access and row ends: what the check found changed where one failed follows them
    // in the checking process's report alone, which no record keeps.
    let verdicts = {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), in_place).unwrap();
        assert_eq!(device.work_counts().in_place_checks_recalled, 0);
        settled(&device);
        assert!(device.in_place_check_took().is_some());
        let mut report = device.in_place_check_report();
        report.truncate(VERDICTS);
        report
    };
    // Its record is recalled by the device after it: each verdict it measured, and each it could
    // not check for a cause that lasts, which that device checks no more; one it could not check
    // for a cause that may pass is no measurement, checked again.
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), in_place).unwrap();
        let counts = device.work_counts();
        // Each class's verdicts, for a texture that reads, one that renders and one a shader
        // writes (of 4-, 8- or 16-byte texels and of 1- or 2-byte ones, each with its padding
        // kept or free), each for texels that share a line with bytes beside them and for texels
        // that share none: those the check measured, and of those it could not check, the ones
        // whose cause lasts.
        let measured = (verdicts.iter())
            .filter(|line| line.ends_with(": Passed") || line.ends_with(": Failed"))
            .count() as u64;
        let recalled = counts.in_place_checks_recalled;
        assert!(
            (measured..=VERDICTS as u64).contains(&recalled),
            "{recalled} recalled, {measured} measured: {verdicts:?}"
        );
        let report = match recalled == VERDICTS as u64 {
            true => {
                assert_eq!(counts.in_place_checks_run, 0);
                assert!(device.settle_in_place_checks(std::time::Duration::ZERO));
                assert_eq!(device.in_place_check_took(), None);
                device.in_place_check_report()
            }
            false => {
                settled(&device);
                let mut report = device.in_place_check_report();
                report.truncate(VERDICTS);
                report
            }
        };
        assert_eq!(report, verdicts);
        eprintln!(
            "IN-PLACE-RECORD on {}: recalled {recalled} of {measured} measured: {verdicts:?}",
            device.name()
        );
    }
    messages.take_notices();
    layer_reported_nothing_at_all(&messages);
}

#[test]
fn in_place_and_shadow_costs() {
    // A surface's size and pitch, the rows each pass reads from its top, and the passes a command
    // buffer reads it in.
    let cases = [
        ((3840u64, 2160u64, 15360u64), 32u64, 1usize),
        ((1920, 1080, 7680), 32, 1),
        ((1920, 1080, 7680), 1080, 1),
        ((1920, 1080, 7680), 1080, 3),
    ];
    for policy in [
        HostPageTexturePolicy::InPlace,
        HostPageTexturePolicy::Shadow,
    ] {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.layers.clear();
            options.policy.host_page_textures = policy;
        })
        .unwrap();
        if policy == HostPageTexturePolicy::InPlace {
            settled(&device);
        }
        let queue = device.new_command_queue().unwrap();
        let reads = reading(&device, PixelFormat::BGRA8Unorm, "tob_read");
        let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
        for (surface, rows, passes) in cases {
            if !surface_costs(
                &device,
                &queue,
                (&reads, &corners),
                policy,
                surface,
                rows,
                passes,
            ) {
                return;
            }
        }
    }
}

/// [`in_place_and_shadow_costs`]' case of a `BGRA8Unorm` surface of `(w, h, bpr)` over a no-copy
/// buffer of anonymous pages on `device`: each frame the CPU writes a row of the pages, and one
/// command buffer renders `passes` passes, each reading the surface's top `rows` rows through a
/// fragment function's `read` into a private target of its size. Printed (`IN-PLACE-COST`): the
/// form, and the median and fastest GPU time of a frame's command buffer. False where the device
/// imports no host memory.
fn surface_costs(
    device: &Device,
    queue: &CommandQueue,
    (reads, corners): (&RenderPipelineState, &Buffer),
    policy: HostPageTexturePolicy,
    (w, h, bpr): (u64, u64, u64),
    rows: u64,
    passes: usize,
) -> bool {
    let len = ((bpr * h) as usize).next_multiple_of(GUEST_PAGE);
    let mapping = Mapping::anonymous(len);
    mapping.write(0, &counting(len, 5));
    // SAFETY: the mapping outlives the buffer and the texture (dropped first).
    let buffer = match unsafe {
        no_copy(
            device,
            mapping.pointer(),
            len,
            ResourceOptions::default(),
            None,
        )
    } {
        Ok(buffer) => buffer,
        Err(error) => {
            eprintln!("IN-PLACE-COST: the device imports no host memory: {error:?}");
            return false;
        }
    };
    let texture = sampled(&buffer, PixelFormat::BGRA8Unorm, (w, h, bpr));
    // SAFETY: the handle is only compared.
    let form = match unsafe { texture.vk_image() } {
        Some(_) => "in place",
        None => "shadow",
    };
    // The CPU writes a row each frame, as a guest's surface changes.
    let write =
        |frame: usize| mapping.write(((frame as u64 % h) * bpr) as usize, &[frame as u8; 64]);
    let times = frame_times(
        device,
        queue,
        (reads, corners),
        &texture,
        write,
        rows,
        passes,
    );
    eprintln!(
        "IN-PLACE-COST on {}: {policy:?} ({form}): a frame's GPU time median {:.1} us, fastest \
         {:.1} us, over {} frames of {passes} pass(es) each reading the top {rows} rows of a \
         {w}x{h} surface",
        device.name(),
        times[times.len() / 2] * 1e6,
        times[0] * 1e6,
        times.len(),
    );
    drop((texture, buffer));
    drop(mapping);
    true
}

/// The GPU times of [`FRAMES`] frames, fastest first, each frame `write(frame)` by the CPU and one
/// command buffer of `queue` that renders `passes` passes, each reading the top `rows` rows of the
/// `BGRA8Unorm` `texture` through `reads` (a fragment function's `read`, over `corners`) into a
/// private target of its size.
fn frame_times(
    device: &Device,
    queue: &CommandQueue,
    (reads, corners): (&RenderPipelineState, &Buffer),
    texture: &Texture,
    mut write: impl FnMut(usize),
    rows: u64,
    passes: usize,
) -> Vec<f64> {
    let (w, h) = (texture.width(), texture.height());
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, w, h, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    let target = device.new_texture(&d).unwrap();
    let mut times = Vec::with_capacity(FRAMES);
    for frame in 0..FRAMES {
        write(frame);
        let command_buffer = queue.command_buffer().unwrap();
        for _ in 0..passes {
            let mut pass = RenderPassDescriptor::new();
            let color = &mut pass.color_attachments[0];
            color.texture = Some(target.clone());
            color.load_action = LoadAction::DontCare;
            color.store_action = StoreAction::Store;
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(reads).unwrap();
            encoder
                .set_scissor_rect(ScissorRect::new(0, 0, w, rows))
                .unwrap();
            encoder.set_vertex_buffer(Some(corners), 0, 0).unwrap();
            encoder.set_fragment_texture(Some(texture), 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
        }
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
        times.push(gpu);
    }
    times.sort_by(f64::total_cmp);
    times
}

/// The frames [`frame_times`] times.
const FRAMES: usize = 40;

/// The consumer's desktop surface over a buffer of the runtime's own memory: 3662x2038
/// `BGRA8Unorm`, rows 14848 bytes apart (14648 bytes of texels, the pitch rounded up to 256).
const OWN_SURFACE: (u64, u64, u64) = (3662, 2038, 14848);

/// A texture over a buffer of the runtime's own memory whose rows lie apart, the consumer's
/// desktop surface's shape ([`OWN_SURFACE`]): over a buffer that declares the bytes between rows
/// unobserved (`BufferDescriptor::padding_unobserved`) and over one that does not. The undeclared
/// one is a shadow (`shadowed_textures`); the declared one a linear image aliasing the buffer
/// wherever the device lays one at the pitch (`unobserved_padding_aliases`), else a shadow. Each
/// frame the CPU writes a row of the buffer, and one command buffer reads the whole surface
/// through a fragment function's `read` (the in-place cost lane's frame). Printed
/// (`UNOBSERVED-PADDING-COST`): each form's median and fastest GPU time of a frame.
#[test]
fn unobserved_padding_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    let queue = device.new_command_queue().unwrap();
    let reads = reading(&device, PixelFormat::BGRA8Unorm, "tob_read");
    let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
    let (w, h, bpr) = OWN_SURFACE;
    for declared in [false, true] {
        let descriptor =
            BufferDescriptor::new(bpr * h, ResourceOptions::shared(), BufferContents::Zero);
        let descriptor = match declared {
            true => descriptor.with_padding_unobserved(),
            false => descriptor,
        };
        let buffer = device.new_buffer_with_descriptor(descriptor).unwrap();
        let contents = buffer.contents().unwrap();
        let before = device.work_counts();
        let texture = sampled(&buffer, PixelFormat::BGRA8Unorm, (w, h, bpr));
        let after = device.work_counts();
        let aliases = after.unobserved_padding_aliases - before.unobserved_padding_aliases;
        let shadows = after.shadowed_textures - before.shadowed_textures;
        assert_eq!(
            aliases + shadows,
            1,
            "declared {declared}: one form, counted"
        );
        assert!(declared || shadows == 1, "undeclared: a shadow");
        let form = match aliases {
            1 => "an aliasing image",
            _ => "a shadow",
        };
        let write = |frame: usize| {
            let at = (frame as u64 % h) * bpr;
            // SAFETY: 64 of the buffer's mapped bytes, inside it, which no command uses between
            // frames.
            unsafe { std::ptr::write_bytes(contents.as_ptr().add(at as usize), frame as u8, 64) };
        };
        let times = frame_times(&device, &queue, (&reads, &corners), &texture, write, h, 1);
        eprintln!(
            "UNOBSERVED-PADDING-COST on {}: declared {declared} ({form}): a frame's GPU time \
             median {:.1} us, fastest {:.1} us, over {} frames each reading a {w}x{h} surface \
             with rows {bpr} bytes apart",
            device.name(),
            times[times.len() / 2] * 1e6,
            times[0] * 1e6,
            times.len(),
        );
    }
    messages.take_notices();
    layer_reported_nothing_at_all(&messages);
}

/// A guest's device caches on an Iris Xe (a consumer boot's 388 layouts, every row padded up to a
/// 64-byte line): `BGRA8Unorm` textures that render, each over a `Private` buffer of the runtime's
/// own, (width, rows, row bytes, count): eight of its most common small shapes at their counts
/// and its large ones.
const DEVICE_CACHES: [(u64, u64, u64, usize); 11] = [
    (172, 67, 704, 15),
    (136, 68, 576, 14),
    (104, 52, 448, 13),
    (164, 82, 704, 11),
    (168, 81, 704, 10),
    (88, 84, 384, 9),
    (148, 74, 640, 8),
    (58, 28, 256, 6),
    (820, 410, 3328, 4),
    (1640, 820, 6592, 2),
    (920, 436, 3712, 1),
];

/// [`DEVICE_CACHES`] over buffers that declare the bytes between rows unobserved
/// (`BufferDescriptor::padding_unobserved`) and over ones that do not, under the in-place policy
/// with the device's check settled. The undeclared ones are shadows (`shadowed_textures`); a
/// declared one is a `DRM_FORMAT_MOD_LINEAR` image at its pitch, at any size, where the device's
/// linear image lays its rows out otherwise and the device's check passed for its pooled memory
/// (`pooled_pitched_images`), else the device's linear image or a shadow. Each frame one command
/// buffer clears and stores every cache and then reads each whole through a fragment function's
/// `read`. Printed (`POOLED-PITCHED-COST`): each form's caches in place and as shadows, and the
/// median and fastest GPU time of a frame.
#[test]
fn pooled_pitched_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        in_place(options);
    })
    .unwrap();
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "the check settles"
    );
    let queue = device.new_command_queue().unwrap();
    let reads = reading(&device, PixelFormat::BGRA8Unorm, "tob_read");
    let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, 1640, 820, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    let target = device.new_texture(&d).unwrap();
    for declared in [false, true] {
        let before = device.work_counts();
        let caches: Vec<(Buffer, Texture)> = (DEVICE_CACHES.iter())
            .flat_map(|&(w, h, bpr, count)| (0..count).map(move |_| (w, h, bpr)))
            .map(|(w, h, bpr)| {
                let descriptor = BufferDescriptor::new(
                    bpr * h,
                    ResourceOptions::private(),
                    BufferContents::Zero,
                );
                let descriptor = match declared {
                    true => descriptor.with_padding_unobserved(),
                    false => descriptor,
                };
                let buffer = device.new_buffer_with_descriptor(descriptor).unwrap();
                let texture = rendered(&buffer, PixelFormat::BGRA8Unorm, (w, h, bpr));
                (buffer, texture)
            })
            .collect();
        let after = device.work_counts();
        let pitched = after.pooled_pitched_images - before.pooled_pitched_images;
        let shadows = after.shadowed_textures - before.shadowed_textures;
        assert!(
            declared || shadows == caches.len() as u64,
            "undeclared: shadows"
        );
        let mut times = Vec::with_capacity(FRAMES);
        for frame in 0..FRAMES {
            let command_buffer = queue.command_buffer().unwrap();
            for (_, texture) in &caches {
                let mut pass = RenderPassDescriptor::new();
                let color = &mut pass.color_attachments[0];
                color.texture = Some(texture.clone());
                color.load_action = LoadAction::Clear;
                color.clear_color = ClearColor::new(frame as f64 / 64.0, 0.5, 0.25, 1.0);
                color.store_action = StoreAction::Store;
                let encoder = command_buffer.render_command_encoder(&pass).unwrap();
                encoder.end_encoding().unwrap();
            }
            for (_, texture) in &caches {
                let mut pass = RenderPassDescriptor::new();
                let color = &mut pass.color_attachments[0];
                color.texture = Some(target.clone());
                color.load_action = LoadAction::DontCare;
                color.store_action = StoreAction::Store;
                let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
                encoder.set_render_pipeline_state(&reads).unwrap();
                encoder
                    .set_scissor_rect(ScissorRect::new(0, 0, texture.width(), texture.height()))
                    .unwrap();
                encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
                encoder.set_fragment_texture(Some(texture), 0).unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
                encoder.end_encoding().unwrap();
            }
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
            let gpu =
                command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
            times.push(gpu);
        }
        times.sort_by(f64::total_cmp);
        eprintln!(
            "POOLED-PITCHED-COST on {}: {} device caches, declared {declared}: {pitched} in place \
             at their pitch, {shadows} shadows; a frame's GPU time median {:.1} us, fastest {:.1} \
             us over {} frames",
            device.name(),
            caches.len(),
            times[times.len() / 2] * 1e6,
            times[0] * 1e6,
            times.len(),
        );
    }
    messages.take_notices();
    layer_reported_nothing_at_all(&messages);
}

/// The device caches of [`pooled_pitched_costs`] made over buffers that declare their padding
/// unobserved on `device`: each buffer and its texture, that renders and reads.
fn device_caches(device: &Device) -> Vec<(Buffer, Texture)> {
    (DEVICE_CACHES.iter())
        .flat_map(|&(w, h, bpr, count)| (0..count).map(move |_| (w, h, bpr)))
        .map(|(w, h, bpr)| {
            let descriptor =
                BufferDescriptor::new(bpr * h, ResourceOptions::private(), BufferContents::Zero)
                    .with_padding_unobserved();
            let buffer = device.new_buffer_with_descriptor(descriptor).unwrap();
            let texture = rendered(&buffer, PixelFormat::BGRA8Unorm, (w, h, bpr));
            (buffer, texture)
        })
        .collect()
}

/// [`DEVICE_CACHES`] over buffers that declare their padding unobserved, made as a guest's
/// WindowServer makes them, before the device's check of its pooled memory answered (a device
/// whose check has not run), under the validation layer: none takes an image at its pitch
/// (`pooled_pitched_images`), and each is a shadow (`shadowed_textures`) but where the device's
/// own linear image lays its rows out at their pitch, which takes no verdict; one command buffer
/// clears each to a value of its own. Once the check answered, the caches made again take
/// the image at their pitch where the device makes it and its check passed
/// (`pooled_pitched_images`), and the next command buffer, which blits each early cache into a
/// buffer and is committed while no other is open, moves every early one whose shape takes it
/// to it (`pooled_textures_moved_in_place`, one per cache made after that took it, counted at its
/// pitch then), its shadow's image freed once that command buffer completed
/// (`pooled_texture_shadows_freed_after_move`); every blit reads its cache's clear. Printed
/// (`POOLED-PITCHED-MOVE`): the caches in place at their pitch and the shadows, before the check
/// answered and after the move, and the check's time.
#[test]
fn device_caches_made_before_the_check_move_to_their_pitch_once_it_passes() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let mut lapses = Vec::new();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), in_place).unwrap();
        let queue = device.new_command_queue().unwrap();
        let start = device.work_counts();
        let early = device_caches(&device);
        let made = device.work_counts();
        let pitched_early = made.pooled_pitched_images - start.pooled_pitched_images;
        let shadows_early = made.shadowed_textures - start.shadowed_textures;
        // In place (`vk_image` answers one): an image at its pitch, or the device's own linear
        // image where it lays the rows out at their pitch, which takes no verdict.
        let in_place = |caches: &[(Buffer, Texture)]| {
            (caches.iter())
                .filter(|(_, texture)| path_of(texture) == "in place")
                .count()
        };
        let linear_early = in_place(&early);
        // Each cache cleared to a value of its own (red 2i / 255 of the i-th, read within one of
        // 2i: the device's own conversion), before the check.
        let command_buffer = queue.command_buffer().unwrap();
        for (i, (_, texture)) in early.iter().enumerate() {
            let mut pass = RenderPassDescriptor::new();
            let color = &mut pass.color_attachments[0];
            color.texture = Some(texture.clone());
            color.load_action = LoadAction::Clear;
            color.clear_color = ClearColor::new((2 * i) as f64 / 255.0, 0.0, 1.0, 1.0);
            color.store_action = StoreAction::Store;
            let encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.end_encoding().unwrap();
        }
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let answered = device.in_place_check_took();
        settled(&device);
        let before_late = device.work_counts();
        let late = device_caches(&device);
        let after_late = device.work_counts();
        let pitched_late = after_late.pooled_pitched_images - before_late.pooled_pitched_images;
        // The next command buffer reads every early cache and takes the move.
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let reads: Vec<Buffer> = (early.iter())
            .map(|(_, texture)| {
                let (w, h) = (texture.width(), texture.height());
                let out = device
                    .new_buffer(w * h * 4, ResourceOptions::shared())
                    .unwrap();
                blit.copy_from_texture_to_buffer(
                    texture,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                    Size::new(w, h, 1),
                    &out,
                    0,
                    w * 4,
                    w * h * 4,
                    BlitOption::None,
                )
                .unwrap();
                out
            })
            .collect();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        drop(command_buffer);
        let after = device.work_counts();
        let moved = after.pooled_textures_moved_in_place - start.pooled_textures_moved_in_place;
        // The moves are counted at their pitch as they move in, at whichever commit took them.
        let counted = after.pooled_pitched_images
            - start.pooled_pitched_images
            - pitched_early
            - pitched_late;
        let started = std::time::Instant::now();
        let freed = loop {
            let freed = device.work_counts().pooled_texture_shadows_freed_after_move
                - start.pooled_texture_shadows_freed_after_move;
            if freed >= moved || started.elapsed() > WAIT {
                break freed;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        };
        for (i, ((_, texture), out)) in early.iter().zip(&reads).enumerate() {
            let length = (texture.width() * texture.height() * 4) as usize;
            // SAFETY: the buffer's mapped bytes, which no command writes any longer.
            let read =
                unsafe { std::slice::from_raw_parts(out.contents().unwrap().as_ptr(), length) };
            let red = (2 * i) as u8;
            let cleared =
                |t: &[u8]| t[0] == 0xff && t[1] == 0 && t[2].abs_diff(red) <= 1 && t[3] == 0xff;
            if !read.chunks(4).all(cleared) {
                lapses.push(format!(
                    "cache {i} ({}x{}): a blit after the check read other than its clear",
                    texture.width(),
                    texture.height()
                ));
            }
        }
        let (in_place, in_place_late) = (in_place(&early), in_place(&late));
        eprintln!(
            "POOLED-PITCHED-MOVE on {}: {} device caches made before the check answered ({}): \
             {pitched_early} in place at their pitch ({linear_early} in place), {shadows_early} \
             shadows; made after it: \
             {pitched_late} in place at their pitch ({in_place_late} in place); the early ones \
             after the next commit: {moved} moved ({counted} counted at their pitch, {freed} \
             shadows freed), {in_place} in place, {} shadows",
            device.name(),
            early.len(),
            match answered {
                Some(took) => format!("it had answered, in {took:?}"),
                None => format!("it took {:?}", device.in_place_check_took()),
            },
            early.len() - in_place,
        );
        let unplaced = (early.len() - linear_early) as u64;
        if answered.is_none() && (pitched_early, shadows_early) != (0, unplaced) {
            lapses.push(format!(
                "before the check answered: {pitched_early} at their pitch, {shadows_early} \
                 shadows, where none is at its pitch and the {unplaced} not in place are shadows"
            ));
        }
        if (moved, counted, freed) != (pitched_late - pitched_early, moved, moved) {
            lapses.push(format!(
                "after the next commit: {moved} moved, {counted} counted, {freed} shadows freed, \
                 where {pitched_late} made after the check take the image at their pitch and \
                 {pitched_early} made before it did"
            ));
        }
        if in_place != in_place_late {
            lapses.push(format!(
                "after the move: {in_place} of the early caches in place, where {in_place_late} \
                 made after the check are"
            ));
        }
        drop((early, late, reads));
        messages.take_notices();
    }
    layer_reported_nothing_at_all(&messages);
    assert!(lapses.is_empty(), "{}", lapses.join("\n"));
}
