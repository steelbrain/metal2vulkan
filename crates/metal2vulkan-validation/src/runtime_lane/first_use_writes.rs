//! A CPU write to bytes of a texture over host pages that the texture's first use does not read
//! stays, on the device the cfg selects, under the validation layer: Metal never writes such
//! bytes, so it keeps every such write (an M3 Max probe: every CPU write between rows, and to
//! texels the command buffer did not read, stayed in each of 10 rounds).
//!
//! A texture that only reads (`ShaderRead` usage) is made over a no-copy buffer of anonymous pages
//! at each of [`LAYOUTS`] (a 1080p surface's packed rows, and a browser tile's padded ones), and
//! its first use is a pass whose draw reads it inside a scissor of the top half of its rows. Around
//! that command buffer's commit, another thread writes a byte of a texel in each row of the bottom
//! half, the byte after each such row's texels where the rows are padded, and bytes past the last
//! row: one row after another, [`SPACING`] apart, starting at each of [`STARTS`] from the commit
//! (a fresh texture each), so the writes cover the commit, the submission and the first use. Every
//! write stays, and the pass reads the texels of the top half as they were.
//!
//! Under both settings of `HostPageTexturePolicy`, each held to the path it took:
//!
//! - **`InPlace`** (the default), once the device's check of the first move over host pages
//!   answered (`Device::settle_in_place_checks`): where it passed for pages imported by their
//!   pointer, the texture is an image aliasing the pages (`vk_image` answers one;
//!   `WorkCounts::host_textures_in_place`), its first move the image's own layout transition from
//!   `UNDEFINED`, which Vulkan lets read and write every byte bound to it and the device's check
//!   showed leaving them alone; or a shadow its refusal counts (`host_texture_shadows_refused_*`:
//!   the driver lays no image at the plane's pitch). Where the check did not pass, a shadow counted
//!   by why.
//! - **`Shadow`**: a shadow, whose first use copies the texels in and writes none of the pages'
//!   bytes (`host_texture_shadows_exact`).
//!
//! Printed (`FIRST-USE-WRITES`): per setting, layout and start, the form, the writes undone and
//! the texels read that differ; and the check's verdicts (`IN-PLACE-CHECK`).

use super::host_memory::{counting, no_copy, not_importable, Mapping, GUEST_PAGE};
use super::in_place_rows;
use super::no_copy_textures::{floats, staged};
use super::pitched_surfaces::reading;
use super::{layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, IoSurfacePlane, LoadAction, Origin, RenderPassDescriptor,
    ResourceOptions, ScissorRect, Size, StoreAction, TextureDescriptor,
};
use std::ptr::NonNull;
use std::time::{Duration, Instant};

/// A surface's `BGRA8Unorm` plane: its name, width, height and bytes per row.
const LAYOUTS: [(&str, u64, u64, u64); 2] = [
    ("1920x1080/7680", 1920, 1080, 7680),
    ("1307x512/5248", 1307, 512, 5248),
];

/// When the writes start, in microseconds from the commit (before it where negative): the commit,
/// the submission and the first use take a time of their own on each device, so the writes start
/// at several times and some of them overlap each.
const STARTS: [i64; 6] = [-2000, -1000, -300, 0, 500, 2000];

/// The time between two rows' writes.
const SPACING: Duration = Duration::from_micros(4);

/// The bytes past the last row the pages hold, and the CPU writes the first and last of.
const PAST: u64 = 64;

/// The time before the first of a round's two steps (the commit, the writes' start).
const LEAD: Duration = Duration::from_millis(1);

fn spin_until(at: Instant) {
    while Instant::now() < at {
        std::hint::spin_loop();
    }
}

/// The pages' first byte, for the writing thread: the test keeps the mapping alive until the
/// thread is joined, and the bytes it writes are no other thread's.
#[derive(Clone, Copy)]
struct Pages(NonNull<u8>);

// SAFETY: see the type's docs; the thread writes bytes no other thread reads or writes meanwhile.
unsafe impl Send for Pages {}

/// How long a lane waits for the device's check of the first move over host pages.
pub(crate) const CHECK_WAIT: Duration = Duration::from_secs(300);

#[test]
fn a_cpu_write_to_bytes_a_first_use_does_not_read_stays() {
    let mut lapses = Vec::new();
    for policy in [
        HostPageTexturePolicy::InPlace,
        HostPageTexturePolicy::Shadow,
    ] {
        first_use_writes_under(policy, &mut lapses);
    }
    assert!(
        lapses.is_empty(),
        "a first use undid a CPU write, or changed or misread bytes, or took another path:\n{}",
        lapses.join("\n")
    );
}

/// The lane's rounds on a device of `policy`, their lapses pushed to `lapses`.
fn first_use_writes_under(policy: HostPageTexturePolicy, lapses: &mut Vec<String>) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy.host_page_textures = policy;
        })
        .unwrap();
        // The check answers before any texture is made, so each takes the path its verdict says.
        let in_place = policy == HostPageTexturePolicy::InPlace;
        if in_place {
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
        let queue = device.new_command_queue().unwrap();
        let reads = reading(&device, PixelFormat::BGRA8Unorm, "tob_read");
        let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
        'layouts: for (name, w, h, bpr) in LAYOUTS {
            let row = w * 4;
            let bound = bpr * h;
            let len = ((bound + PAST) as usize).next_multiple_of(GUEST_PAGE);
            let top = h / 2;
            // The bytes the thread writes, in order: a texel's in each row of the bottom half, the
            // byte after the row's texels where the rows are padded, and bytes past the last row.
            let mut targets = Vec::new();
            for y in top..h {
                targets.push((y * bpr + (y % w) * 4) as usize);
                if bpr > row {
                    targets.push((y * bpr + row) as usize);
                }
            }
            targets.extend([bound as usize, (bound + PAST - 1) as usize]);
            for start in STARTS {
                let mapping = Mapping::anonymous(len);
                let pages = counting(len, 7);
                mapping.write(0, &pages);
                // SAFETY: the mapping outlives the buffer and the texture (dropped first).
                let made = unsafe {
                    no_copy(
                        &device,
                        mapping.pointer(),
                        len,
                        ResourceOptions::default(),
                        None,
                    )
                };
                let buffer = match made {
                    Ok(buffer) => buffer,
                    Err(error) if not_importable(&error) => {
                        eprintln!("FIRST-USE-WRITES: the device imports no host memory: {error:?}");
                        break 'layouts;
                    }
                    Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
                };
                // The verdict its texels take, at the start of the pages.
                let first = buffer.contents().map_or(0, |at| at.as_ptr() as u64);
                let on_lines = in_place_rows::on_lines(&device, first, row, bpr, h);
                let pointer_passed = in_place_rows::passed(&device, "Pointer", on_lines);
                let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, w, h, false);
                d.usage = TextureUsage::SHADER_READ;
                d.resource_options = buffer.resource_options();
                let before = device.work_counts();
                let texture = buffer
                    .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, bpr, false))
                    .unwrap();
                let after = device.work_counts();
                // The path the texture took, held to the setting and the check's verdict.
                let aliased = after.host_textures_in_place - before.host_textures_in_place;
                let refused = [
                    after.host_texture_shadows_refused_pitch
                        - before.host_texture_shadows_refused_pitch,
                    after.host_texture_shadows_refused_image
                        - before.host_texture_shadows_refused_image,
                    after.host_texture_shadows_refused_placement
                        - before.host_texture_shadows_refused_placement,
                ]
                .iter()
                .sum::<u64>();
                let exact = after.host_texture_shadows_exact - before.host_texture_shadows_exact;
                // SAFETY: the handle is only compared, while the texture lives.
                let form = match unsafe { texture.vk_image() } {
                    Some(_) => "aliased",
                    None => "shadowed",
                };
                let took = match (in_place, pointer_passed) {
                    (false, _) => exact == 1 && form == "shadowed",
                    (true, true) => {
                        (aliased == 1 && form == "aliased") || (refused == 1 && form == "shadowed")
                    }
                    (true, false) => aliased == 0 && form == "shadowed",
                };
                if !took {
                    lapses.push(format!(
                        "{policy:?}, {name}: the texture is {form} (in place {aliased}, refused \
                         {refused}, exact {exact}), the pointer check passed: {pointer_passed}"
                    ));
                }
                let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, w, h, false);
                d.usage = TextureUsage::RENDER_TARGET;
                d.resource_options = ResourceOptions::private();
                let target = device.new_texture(&d).unwrap();
                let read = device
                    .new_buffer(row * h, ResourceOptions::shared())
                    .unwrap();
                let command_buffer = queue.command_buffer().unwrap();
                let mut pass = RenderPassDescriptor::new();
                let color = &mut pass.color_attachments[0];
                color.texture = Some(target.clone());
                color.load_action = LoadAction::Clear;
                color.store_action = StoreAction::Store;
                color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
                let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
                encoder.set_render_pipeline_state(&reads).unwrap();
                encoder
                    .set_scissor_rect(ScissorRect::new(0, 0, w, top))
                    .unwrap();
                encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
                encoder.set_fragment_texture(Some(&texture), 0).unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
                encoder.end_encoding().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_texture_to_buffer(
                    &target,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                    Size::new(w, h, 1),
                    &read,
                    0,
                    row,
                    row * h,
                    BlitOption::None,
                )
                .unwrap();
                blit.end_encoding().unwrap();
                let expected: Vec<u8> = {
                    let mut expected = pages.clone();
                    for &at in &targets {
                        expected[at] = pages[at] ^ 0xA5;
                    }
                    expected
                };
                let base = Instant::now() + LEAD;
                let shift = Duration::from_micros(start.unsigned_abs());
                let (writes_at, commit_at) = match start < 0 {
                    true => (base, base + shift),
                    false => (base + shift, base),
                };
                let at = Pages(mapping.pointer());
                std::thread::scope(|scope| {
                    let (targets, expected) = (&targets, &expected);
                    let writer = scope.spawn(move || {
                        let pages = at;
                        spin_until(writes_at);
                        for (k, &byte) in targets.iter().enumerate() {
                            spin_until(writes_at + SPACING * k as u32);
                            // SAFETY: `byte` lies inside the mapping, which outlives the thread,
                            // and no other thread touches it meanwhile.
                            unsafe { pages.0.as_ptr().add(byte).write_volatile(expected[byte]) };
                        }
                    });
                    spin_until(commit_at);
                    command_buffer.commit().unwrap();
                    command_buffer.wait_until_completed_for(WAIT).unwrap();
                    writer.join().unwrap();
                });
                assert_eq!(command_buffer.error(), None);
                let now = mapping.read(0..len);
                let undone: Vec<usize> = (targets.iter().copied())
                    .filter(|&at| now[at] != expected[at])
                    .collect();
                let changed = (0..len).filter(|&at| now[at] != expected[at]).count() - undone.len();
                let contents = read.contents().unwrap();
                // SAFETY: the buffer's `row * h` bytes, which no command writes any longer.
                let got =
                    unsafe { std::slice::from_raw_parts(contents.as_ptr(), (row * h) as usize) };
                let differ = (0..top)
                    .flat_map(|y| (0..w).map(move |x| (y, x)))
                    .filter(|&(y, x)| {
                        let at = (y * bpr + x * 4) as usize;
                        let out = (y * row + x * 4) as usize;
                        got[out..out + 4] != pages[at..at + 4]
                    })
                    .count();
                let what = format!("{policy:?}, {name}, writes from {start} us of the commit");
                eprintln!(
                    "FIRST-USE-WRITES on {}: {what}: {form}: {} of {} writes undone, {changed} \
                     other bytes changed, {differ} texels read differ",
                    device.name(),
                    undone.len(),
                    targets.len(),
                );
                if let Some(&at) = undone.first() {
                    lapses.push(format!(
                        "{what}: {} of {} CPU writes undone, the first at byte {at}",
                        undone.len(),
                        targets.len()
                    ));
                }
                if changed != 0 {
                    lapses.push(format!("{what}: {changed} bytes no one wrote changed"));
                }
                if differ != 0 {
                    lapses.push(format!(
                        "{what}: {differ} texels of the top half read differ from the pages'"
                    ));
                }
                drop((texture, buffer, target, read));
                drop(mapping);
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
}
