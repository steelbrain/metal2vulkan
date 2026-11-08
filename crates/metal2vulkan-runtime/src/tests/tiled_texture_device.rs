//! `Buffer::new_tiled_texture`, Metal's private `newTiledTextureWithDescriptor:offset:bytesPerRow:`,
//! as the x86 macOS guests' WindowServer asks for it: a `BGRA8Unorm` 2D texture of 54x54 and 6
//! levels, and one of 64x60 and 7, `ShaderRead` alone, its descriptor `Managed`, over a new
//! 16384-byte `Shared` buffer from offset 0 at 256 bytes a row. On a Mac its first level aliases
//! the bytes and its others lie in its own memory; here it is one image of every level whose
//! first is a copy of the bytes (`shadowed-texture`):
//!
//! - **CPU writes reach the first level.** Bytes the CPU wrote before a command buffer are what
//!   its blit out of the first level reads, texel for texel at the buffer's pitch, and what its
//!   kernel samples there, a later command buffer reading a later write.
//! - **GPU writes reach the bytes.** A blit into the first level leaves its texels in the
//!   buffer's rows once the command buffer completed, and the bytes between the rows as they
//!   were.
//! - **Mipmaps.** `generate_mipmaps` fills the later levels from the first (a checkerboard's
//!   red averages to 127 or 128 a texel at level 1, read back on the CPU), and a kernel's sample
//!   at a level-1 LOD through a sampler with a nearest mip filter reads level 1's texels.
//! - **The later levels.** Blits into and out of each level past the first, and the CPU's
//!   `replace_region` and `get_bytes` there, round-trip, and leave the buffer's bytes as they
//!   were.
//! - **One level** is `new_texture`'s: the same placement and the same path over a no-copy
//!   buffer, in place where the device's check passed.
//! - **Announced and counted.** Each texture is announced (`shadowed-texture`, at every
//!   occurrence where the device's options ask), the notice naming its levels, counted in
//!   `WorkCounts::shadowed_textures` and `shadows_zeroed`, and its placement query answers
//!   `Shadow(Mipmapped)`; its copies in move its first level's texels alone
//!   (`WorkCounts::shadow_texels_copied_in`, a multiple of them).
//! - **The later levels alone copy nothing.** A blit into level 1 and a render pass into it take
//!   no copy of the first level (`WorkCounts::shadow_copies`), and `generate_mipmaps` copies the
//!   first level in, which it reads, and nothing back, keeps nothing and merges nothing, as its
//!   writes are the later levels'.
//! - **Merges fetch.** A blit into part of the first level over a `Shared` buffer merges back
//!   through a view of the first level (`WorkCounts::shadow_merges_fetched`), the bytes outside
//!   the part kept.
//! - **Detached.** Over a no-copy buffer's pages, detached (`detach_backing`), the texture keeps
//!   its own image (its `VkImage` the same): `get_bytes` of the first level reads the bytes the
//!   CPU last wrote before the detach (after the last copy in), the later levels read and take
//!   writes, and a write of the first level is refused (`NotImplemented(DetachedTexelAccess)`), as
//!   a one-level texture's CPU access is. A rebind to new pages gives the first level their bytes
//!   and leaves the later levels as written.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Printed (`TILED-TEXTURE`): each
//! shape's sampled and counted values.

use super::blit_holds::{device, over_host, Host, Scratch, WAIT};
use super::draw_again::{device as hearing_device, Heard};
use super::encoder_holds::{library, pass};
use crate::{
    BlitOption, Buffer, ClearColor, CommandQueue, ComputePipelineState, CpuCacheMode, Degradation,
    Device, Error, HazardTrackingMode, Origin, PipelineOptions, Region, ResourceOptions,
    RuntimeGap, SamplerState, Size, StorageMode, Texture, TextureDescriptor, TexturePlacement,
    TextureShadowReason,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, SamplerMipFilter, TextureUsage};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The guest's shapes: level 0's width and height, and the level count.
const SHAPES: [((u64, u64), u64); 2] = [((54, 54), 6), ((64, 60), 7)];

/// The guest's buffer: its length and the texture's `bytesPerRow`.
const LENGTH: u64 = 16384;
const PITCH: u64 = 256;

/// The byte the buffer holds before a test writes its texels' rows.
const BETWEEN: u8 = 0xee;

/// How long the device's in-place check may take to settle.
const SETTLE: Duration = Duration::from_secs(120);

/// The guest's descriptor of `shape`: `BGRA8Unorm`, `ShaderRead`, `Managed` (the texture takes
/// the buffer's options, as `new_texture`'s does), GPU-optimized contents allowed.
fn descriptor(((width, height), levels): ((u64, u64), u64)) -> TextureDescriptor {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
    d.mipmap_level_count = levels;
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    d.allow_gpu_optimized_contents = true;
    d
}

/// A new `Shared` buffer of the guest's length, every byte [`BETWEEN`].
fn buffer(device: &Device) -> Buffer {
    let buffer = device
        .new_buffer(LENGTH, ResourceOptions::shared())
        .unwrap();
    write(&buffer, |bytes| bytes.fill(BETWEEN));
    buffer
}

/// A `Shared` buffer of `length` bytes holding `bytes` from its first.
fn holding(device: &Device, length: u64, bytes: &[u8]) -> Buffer {
    let buffer = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    write(&buffer, |all| all[..bytes.len()].copy_from_slice(bytes));
    buffer
}

/// `edit` given `buffer`'s bytes, which no command uses meanwhile.
fn write(buffer: &Buffer, edit: impl FnOnce(&mut [u8])) {
    let contents = buffer.contents().unwrap();
    // SAFETY: a `Shared` buffer's contents span its length, and no command runs over them while
    // the test writes them (each test waits for its command buffers), nor does another borrow.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(contents.as_ptr(), buffer.length() as usize) };
    edit(bytes);
}

/// A copy of `buffer`'s bytes, which no command writes meanwhile.
fn read(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: as `write`'s; the bytes are copied out at once.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// The texel `(x, y)` of a pattern seeded `seed`, its bytes in `BGRA8Unorm`'s order.
fn texel(seed: u8, x: u64, y: u64) -> [u8; 4] {
    let (x, y) = (x as u8, y as u8);
    [
        x.wrapping_mul(7) ^ seed,
        y.wrapping_mul(13) ^ seed,
        x ^ y.rotate_left(3),
        0xff ^ seed,
    ]
}

/// Write `of(x, y)` into the `width` x `height` texels at `PITCH` bytes a row of `buffer`.
fn write_rows(buffer: &Buffer, (width, height): (u64, u64), of: impl Fn(u64, u64) -> [u8; 4]) {
    write(buffer, |bytes| {
        for y in 0..height {
            for x in 0..width {
                let at = (y * PITCH + x * 4) as usize;
                bytes[at..at + 4].copy_from_slice(&of(x, y));
            }
        }
    });
}

/// The packed `width` x `height` texels of `of(x, y)`.
fn packed((width, height): (u64, u64), of: impl Fn(u64, u64) -> [u8; 4]) -> Vec<u8> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .flat_map(|(x, y)| of(x, y))
        .collect()
}

/// The level-0 pattern whose level 1 `generate_mipmaps` makes known: blue the
/// checkerboard of 0 and 255 a texel, which a 2x2 box averages to 127 or 128; green one value a 2x2
/// block, which it keeps; red 0x40 and alpha 255.
fn checker(x: u64, y: u64) -> [u8; 4] {
    let blue = if (x + y).is_multiple_of(2) { 0 } else { 0xff };
    let green = ((x / 2) * 9 + (y / 2) * 5) as u8;
    [blue, green, 0x40, 0xff]
}

/// The device's queue, the sampling kernels, and a sampler with a nearest mip filter.
struct Rig {
    device: Device,
    queue: CommandQueue,
    /// `sample_level`: `t.sample(s, float2(gid) * 0.25, level(1.0))` into `out[gid]`.
    level_one: ComputePipelineState,
    /// `sample_base`: an implicit-LOD sample outside a fragment function, which reads level 0.
    base: ComputePipelineState,
    sampler: SamplerState,
}

impl Rig {
    fn new(device: Device) -> Rig {
        let sampling = library(&device, "engine/sampling.ll");
        let state = |name: &str| {
            let function = sampling.new_function_with_name(name).unwrap();
            (device
                .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default()))
            .unwrap()
        };
        let (level_one, base) = (state("sample_level"), state("sample_base"));
        let mut d = SamplerDescriptor::new();
        d.mip_filter = SamplerMipFilter::Nearest;
        let sampler = device.new_sampler_state(&d).unwrap();
        Rig {
            queue: device.new_command_queue().unwrap(),
            device,
            level_one,
            base,
            sampler,
        }
    }

    /// Run `encode` in one command buffer, committed and completed.
    fn run(&self, encode: impl FnOnce(&crate::CommandBuffer)) {
        let command_buffer = self.queue.command_buffer().unwrap();
        encode(&command_buffer);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    }

    /// Encode `state` sampling `texture` at 4 points into `out`.
    fn sample(
        &self,
        command_buffer: &crate::CommandBuffer,
        state: &ComputePipelineState,
        (texture, out): (&Texture, &Buffer),
    ) {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(state).unwrap();
        encoder.set_texture(Some(texture), 0).unwrap();
        encoder.set_sampler_state(Some(&self.sampler), 0).unwrap();
        encoder.set_buffer(Some(out), 0, 0).unwrap();
        (encoder.dispatch_threads(Size::new(4, 1, 1), Size::new(4, 1, 1))).unwrap();
        encoder.end_encoding().unwrap();
    }

    /// `out`'s four `float4` samples.
    fn samples(out: &Buffer) -> [[f32; 4]; 4] {
        let bytes = read(out);
        std::array::from_fn(|point| {
            std::array::from_fn(|channel| {
                let at = point * 16 + channel * 4;
                f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
            })
        })
    }

    /// A 64-byte `Shared` buffer for four `float4` samples.
    fn out(&self) -> Buffer {
        (self.device.new_buffer(64, ResourceOptions::shared())).unwrap()
    }
}

/// The `BGRA8Unorm` texel a sample read, its channels red, green, blue, alpha over 255 each.
fn texel_of([r, g, b, a]: [f32; 4]) -> [u8; 4] {
    [b, g, r, a].map(|c| (c * 255.0).round() as u8)
}

/// The rig of the device the cfg selects, its shader cache in `scratch`.
fn rig(scratch: &Scratch) -> Rig {
    Rig::new(device(&scratch.0))
}

#[test]
fn cpu_writes_reach_the_first_level() {
    let scratch = Scratch::new("tiled-texture-cpu-writes");
    let rig = rig(&scratch);
    for shape in SHAPES {
        let ((width, height), levels) = shape;
        let case = format!("{width}x{height} of {levels} levels");
        let target = buffer(&rig.device);
        let texture = target
            .new_tiled_texture(&descriptor(shape), 0, PITCH)
            .unwrap();
        assert_eq!(texture.mipmap_level_count(), levels, "{case}");
        assert!(texture.shadow().is_some(), "{case}: a shadow");
        // A blit out of the first level reads the bytes the CPU wrote, texel for texel.
        write_rows(&target, (width, height), |x, y| texel(1, x, y));
        let row = width * 4;
        let read_back = (rig
            .device
            .new_buffer(row * height, ResourceOptions::shared()))
        .unwrap();
        let before = rig.device.work_counts();
        rig.run(|command_buffer| {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            let size = Size::new(width, height, 1);
            let origin = Origin::new(0, 0, 0);
            blit.copy_from_texture_to_buffer(
                &texture,
                0,
                0,
                origin,
                size,
                &read_back,
                0,
                row,
                0,
                BlitOption::None,
            )
            .unwrap();
            blit.end_encoding().unwrap();
        });
        let after = rig.device.work_counts();
        let copied_in = after.shadow_texels_copied_in - before.shadow_texels_copied_in;
        assert!(
            copied_in > 0 && copied_in.is_multiple_of(width * height),
            "{case}: {copied_in} texels copied in, of a first level of {}",
            width * height
        );
        let expected = packed((width, height), |x, y| texel(1, x, y));
        assert_eq!(
            read(&read_back),
            expected,
            "{case}: the blit out of level 0"
        );
        // A kernel samples the bytes the CPU wrote before its command buffer, and a later one
        // a later write. One value over the level: a sample's texel is the level's at any point.
        let out = rig.out();
        for color in [[0x10, 0x80, 0xf0, 0xff], [0xc0, 0x30, 0x08, 0x7f]] {
            write_rows(&target, (width, height), |_, _| color);
            rig.run(|command_buffer| rig.sample(command_buffer, &rig.base, (&texture, &out)));
            let samples = Rig::samples(&out);
            eprintln!("TILED-TEXTURE: {case}: level 0 sampled {samples:?} after {color:?}");
            for sample in samples {
                assert_eq!(texel_of(sample), color, "{case}: level 0 of {color:?}");
            }
        }
        eprintln!("TILED-TEXTURE: {case}: {copied_in} texels copied in for the blit out");
    }
}

#[test]
fn a_blit_into_the_first_level_reaches_the_buffer() {
    let scratch = Scratch::new("tiled-texture-first-level-blit");
    let rig = rig(&scratch);
    for shape in SHAPES {
        let ((width, height), levels) = shape;
        let case = format!("{width}x{height} of {levels} levels");
        let target = buffer(&rig.device);
        let texture = target
            .new_tiled_texture(&descriptor(shape), 0, PITCH)
            .unwrap();
        let row = width * 4;
        let pattern = packed((width, height), |x, y| texel(2, x, y));
        let source = holding(&rig.device, row * height, &pattern);
        let before = rig.device.work_counts();
        rig.run(|command_buffer| {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_texture(
                &source,
                0,
                row,
                0,
                Size::new(width, height, 1),
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
            .unwrap();
            blit.end_encoding().unwrap();
        });
        let after = rig.device.work_counts();
        let back = after.shadow_texels_copied_back - before.shadow_texels_copied_back;
        let got = read(&target);
        for y in 0..height {
            for x in 0..PITCH / 4 {
                let at = (y * PITCH + x * 4) as usize;
                let expected = match x < width {
                    true => texel(2, x, y),
                    false => [BETWEEN; 4],
                };
                assert_eq!(got[at..at + 4], expected, "{case}: bytes {at}");
            }
        }
        let end = (height * PITCH) as usize;
        assert!(got[end..].iter().all(|&b| b == BETWEEN), "{case}: past");
        eprintln!("TILED-TEXTURE: {case}: the blit into level 0 copied back {back} texels");
    }
}

#[test]
fn generated_mipmaps_fill_the_level_a_level_one_sample_reads() {
    let scratch = Scratch::new("tiled-texture-mipmaps");
    let rig = rig(&scratch);
    for shape in SHAPES {
        let ((width, height), levels) = shape;
        let case = format!("{width}x{height} of {levels} levels");
        let target = buffer(&rig.device);
        let texture = target
            .new_tiled_texture(&descriptor(shape), 0, PITCH)
            .unwrap();
        write_rows(&target, (width, height), checker);
        let out = rig.out();
        rig.run(|command_buffer| {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.generate_mipmaps(&texture).unwrap();
            blit.end_encoding().unwrap();
            rig.sample(command_buffer, &rig.level_one, (&texture, &out));
        });
        // Level 1 as the CPU reads it: each texel the 2x2 box of level 0's.
        let (w, h) = (width / 2, height / 2);
        let mut level_one = vec![0u8; (w * h * 4) as usize];
        let region = Region::new_2d(0, 0, w, h);
        (texture.get_bytes(&mut level_one, w * 4, 0, region, 1, 0)).unwrap();
        for y in 0..h {
            for x in 0..w {
                let at = ((y * w + x) * 4) as usize;
                let [blue, green, red, alpha] = level_one[at..at + 4].try_into().unwrap();
                let [_, kept, _, _] = checker(2 * x, 2 * y);
                assert!(
                    (127..=128).contains(&blue),
                    "{case}: level 1 blue {blue} at {x},{y}"
                );
                assert_eq!([green, red, alpha], [kept, 0x40, 0xff], "{case}: {x},{y}");
            }
        }
        // A sample at LOD 1 reads level 1's averaged blue, which level 0 holds nowhere.
        let samples = Rig::samples(&out);
        eprintln!("TILED-TEXTURE: {case}: level 1 sampled {samples:?}");
        for sample in samples {
            let [blue, _, red, alpha] = texel_of(sample);
            assert!((127..=128).contains(&blue), "{case}: blue {blue}");
            assert_eq!([red, alpha], [0x40, 0xff], "{case}");
        }
    }
}

#[test]
fn the_later_levels_round_trip_apart_from_the_buffer() {
    let scratch = Scratch::new("tiled-texture-later-levels");
    let rig = rig(&scratch);
    for shape in SHAPES {
        let ((width, height), levels) = shape;
        let case = format!("{width}x{height} of {levels} levels");
        let target = buffer(&rig.device);
        let texture = target
            .new_tiled_texture(&descriptor(shape), 0, PITCH)
            .unwrap();
        write_rows(&target, (width, height), |x, y| texel(3, x, y));
        let kept = read(&target);
        for level in 1..levels {
            let (w, h) = ((width >> level).max(1), (height >> level).max(1));
            let row = w * 4;
            let seed = 0x40 + level as u8;
            let pattern = packed((w, h), |x, y| texel(seed, x, y));
            let source = holding(&rig.device, row * h, &pattern);
            let read_back = (rig.device.new_buffer(row * h, ResourceOptions::shared())).unwrap();
            let size = Size::new(w, h, 1);
            let origin = Origin::new(0, 0, 0);
            rig.run(|command_buffer| {
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_buffer_to_texture(
                    &source,
                    0,
                    row,
                    0,
                    size,
                    &texture,
                    0,
                    level,
                    origin,
                    BlitOption::None,
                )
                .unwrap();
                blit.copy_from_texture_to_buffer(
                    &texture,
                    0,
                    level,
                    origin,
                    size,
                    &read_back,
                    0,
                    row,
                    0,
                    BlitOption::None,
                )
                .unwrap();
                blit.end_encoding().unwrap();
            });
            assert_eq!(read(&read_back), pattern, "{case}: level {level} by blits");
            let mut cpu = vec![0u8; pattern.len()];
            let region = Region::new_2d(0, 0, w, h);
            (texture.get_bytes(&mut cpu, row, 0, region, level, 0)).unwrap();
            assert_eq!(cpu, pattern, "{case}: level {level} read on the CPU");
            // The CPU's write there, read back by a blit.
            let written: Vec<u8> = pattern.iter().map(|b| !b).collect();
            (texture.replace_region(region, level, 0, &written, row, 0)).unwrap();
            rig.run(|command_buffer| {
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_texture_to_buffer(
                    &texture,
                    0,
                    level,
                    origin,
                    size,
                    &read_back,
                    0,
                    row,
                    0,
                    BlitOption::None,
                )
                .unwrap();
                blit.end_encoding().unwrap();
            });
            assert_eq!(
                read(&read_back),
                written,
                "{case}: level {level} written on the CPU"
            );
        }
        assert_eq!(read(&target), kept, "{case}: the buffer's bytes");
    }
}

#[test]
fn one_level_lies_where_new_texture_s_does() {
    // The pages, declared before the device that uses them, so dropped after it.
    let host = Host::new();
    let scratch = Scratch::new("tiled-texture-one-level");
    let device = device(&scratch.0);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(buffer) = over_host(&device, &host, &runs) else {
        return;
    };
    assert!(device.settle_in_place_checks(SETTLE), "the check settled");
    let mut d = descriptor(((64, 64), 1));
    d.resource_options = buffer.resource_options();
    let asked = buffer.texture_placement(&d, 0, PITCH).unwrap();
    let tiled = buffer.tiled_texture_placement(&d, 0, PITCH).unwrap();
    assert_eq!(tiled, asked, "the query");
    let in_place = |device: &Device| device.work_counts().host_textures_in_place;
    let start = in_place(&device);
    let plain = buffer.new_texture(&d, 0, PITCH).unwrap();
    let middle = in_place(&device);
    let made = buffer.new_tiled_texture(&d, 64 * PITCH, PITCH).unwrap();
    let end = in_place(&device);
    assert_eq!(end - middle, middle - start, "the in-place counts");
    assert_eq!(made.aliases_buffer(), plain.aliases_buffer(), "the path");
    assert_eq!(made.mipmap_level_count(), 1);
    eprintln!(
        "TILED-TEXTURE on {}: one level answered {tiled:?}, in place {}",
        device.name(),
        made.aliases_buffer()
    );
}

#[test]
fn each_tiled_texture_is_announced_and_counted() {
    let scratch = Scratch::new("tiled-texture-notices");
    let heard: Heard = Arc::new(Mutex::new(Vec::new()));
    let device = hearing_device(&scratch.0, &[Degradation::ShadowedTexture], &heard);
    for shape in SHAPES {
        let ((width, height), levels) = shape;
        let case = format!("{width}x{height} of {levels} levels");
        let target = buffer(&device);
        let d = descriptor(shape);
        let placement = target.tiled_texture_placement(&d, 0, PITCH).unwrap();
        let levels = levels as u32;
        let mipmapped = TextureShadowReason::Mipmapped { levels };
        assert_eq!(placement, TexturePlacement::Shadow(mipmapped), "{case}");
        // The public call stops the levels.
        assert!(target.new_texture(&d, 0, PITCH).is_err(), "{case}");
        let before = device.work_counts();
        let notices = heard.lock().unwrap().len();
        let texture = target.new_tiled_texture(&d, 0, PITCH).unwrap();
        let after = device.work_counts();
        assert_eq!(
            after.shadowed_textures - before.shadowed_textures,
            1,
            "{case}"
        );
        assert_eq!(after.shadows_zeroed - before.shadows_zeroed, 1, "{case}");
        let new: Vec<_> = heard.lock().unwrap()[notices..].to_vec();
        let shadowed: Vec<_> = new
            .iter()
            .filter(|n| n.degradation == Degradation::ShadowedTexture)
            .collect();
        assert_eq!(shadowed.len(), 1, "{case}: {new:?}");
        let notice = shadowed[0];
        assert!(
            notice
                .object
                .contains(&format!("{width}x{height} and {levels} levels")),
            "{case}: {}",
            notice.object
        );
        assert!(
            notice.reason.contains(&format!("{levels} levels")),
            "{case}: {}",
            notice.reason
        );
        eprintln!(
            "TILED-TEXTURE: {case}: {}: {}",
            notice.object, notice.reason
        );
        drop(texture);
    }
}

/// What a run of commands copied of a shadow (`WorkCounts`, after less before).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Copies {
    /// `shadow_copies`: every copy in, copy back, keep and merge.
    copies: u64,
    /// `shadow_texels_copied_in`.
    texels_in: u64,
    /// `shadow_texels_copied_back`.
    texels_back: u64,
    /// `shadow_texels_kept`.
    texels_kept: u64,
    /// `shadow_merges_fetched`.
    fetched: u64,
}

/// What `run` copied of the device's shadows.
fn copies_of(device: &Device, run: impl FnOnce()) -> Copies {
    let before = device.work_counts();
    run();
    let after = device.work_counts();
    Copies {
        copies: after.shadow_copies - before.shadow_copies,
        texels_in: after.shadow_texels_copied_in - before.shadow_texels_copied_in,
        texels_back: after.shadow_texels_copied_back - before.shadow_texels_copied_back,
        texels_kept: after.shadow_texels_kept - before.shadow_texels_kept,
        fetched: after.shadow_merges_fetched - before.shadow_merges_fetched,
    }
}

/// `level`'s texels of `texture` (of the level-0 extent `(width, height)`), read on the CPU.
fn level_bytes(texture: &Texture, (width, height): (u64, u64), level: u64) -> Vec<u8> {
    let (w, h) = ((width >> level).max(1), (height >> level).max(1));
    let mut bytes = vec![0u8; (w * h * 4) as usize];
    let region = Region::new_2d(0, 0, w, h);
    (texture.get_bytes(&mut bytes, w * 4, 0, region, level, 0)).unwrap();
    bytes
}

/// Encode a blit of `source`'s packed rows into `size` of `texture`'s `level` at `origin`.
fn blit_into(
    command_buffer: &crate::CommandBuffer,
    (source, texture): (&Buffer, &Texture),
    level: u64,
    (origin, size): (Origin, Size),
) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let row = size.width * 4;
    blit.copy_from_buffer_to_texture(
        source,
        0,
        row,
        0,
        size,
        texture,
        0,
        level,
        origin,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
}

#[test]
fn uses_of_the_later_levels_alone_copy_nothing_of_the_first() {
    let scratch = Scratch::new("tiled-texture-later-levels-alone");
    let rig = rig(&scratch);
    // Each shape's counts, printed before any is asserted.
    let mut counted = Vec::new();
    for shape in SHAPES {
        let ((width, height), levels) = shape;
        let case = format!("{width}x{height} of {levels} levels");
        let target = buffer(&rig.device);
        let texture = target
            .new_tiled_texture(&descriptor(shape), 0, PITCH)
            .unwrap();
        write_rows(&target, (width, height), checker);
        let kept = read(&target);
        // A blit into level 1 alone.
        let (w, h) = (width / 2, height / 2);
        let pattern = packed((w, h), |x, y| texel(6, x, y));
        let source = holding(&rig.device, w * h * 4, &pattern);
        let at = (Origin::new(0, 0, 0), Size::new(w, h, 1));
        let one = copies_of(&rig.device, || {
            rig.run(|command_buffer| blit_into(command_buffer, (&source, &texture), 1, at));
        });
        assert_eq!(level_bytes(&texture, (width, height), 1), pattern, "{case}");
        // `generate_mipmaps`, which reads level 0 and writes the later levels alone.
        let mips = copies_of(&rig.device, || {
            rig.run(|command_buffer| {
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.generate_mipmaps(&texture).unwrap();
                blit.end_encoding().unwrap();
            });
        });
        eprintln!("TILED-TEXTURE: {case}: a level-1 blit {one:?}");
        eprintln!("TILED-TEXTURE: {case}: generate_mipmaps {mips:?}");
        // Level 1 as `generate_mipmaps` left it: its first texel's blue the checkerboard's 2x2
        // average.
        let blue = level_bytes(&texture, (width, height), 1)[0];
        assert!((127..=128).contains(&blue), "{case}: level 1 blue {blue}");
        assert_eq!(read(&target), kept, "{case}: the buffer's bytes");
        counted.push((case, width * height, one, mips));
    }
    let none = Copies {
        copies: 0,
        texels_in: 0,
        texels_back: 0,
        texels_kept: 0,
        fetched: 0,
    };
    for (case, first, one, mips) in counted {
        // The level-1 blit: no copy of level 0 in, back, kept or merged.
        assert_eq!(one, none, "{case}: the level-1 blit");
        // `generate_mipmaps`: level 0 copied in, which it reads; nothing back, kept or merged.
        assert!(
            mips.texels_in > 0 && mips.texels_in.is_multiple_of(first),
            "{case}: generate_mipmaps copied in {mips:?}"
        );
        assert_eq!(
            (mips.texels_back, mips.texels_kept, mips.fetched),
            (0, 0, 0),
            "{case}: generate_mipmaps {mips:?}"
        );
    }
}

#[test]
fn a_render_pass_into_a_later_level_copies_nothing_of_the_first() {
    let scratch = Scratch::new("tiled-texture-later-level-pass");
    let rig = rig(&scratch);
    for shape in SHAPES {
        let ((width, height), levels) = shape;
        let case = format!("{width}x{height} of {levels} levels");
        let target = buffer(&rig.device);
        let mut d = descriptor(shape);
        d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let texture = target.new_tiled_texture(&d, 0, PITCH).unwrap();
        write_rows(&target, (width, height), |x, y| texel(7, x, y));
        let kept = read(&target);
        let mut into_one = pass(&texture);
        into_one.color_attachments[0].level = 1;
        into_one.color_attachments[0].clear_color = ClearColor::new(1.0, 0.0, 1.0, 1.0);
        let copied = copies_of(&rig.device, || {
            rig.run(|command_buffer| {
                let encoder = command_buffer.render_command_encoder(&into_one).unwrap();
                encoder.end_encoding().unwrap();
            });
        });
        eprintln!("TILED-TEXTURE: {case}: a pass into level 1 {copied:?}");
        assert_eq!(copied.copies, 0, "{case}: the pass into level 1 {copied:?}");
        let level_one = level_bytes(&texture, (width, height), 1);
        assert!(
            level_one.chunks(4).all(|t| t == [0xff, 0, 0xff, 0xff]),
            "{case}: level 1 cleared"
        );
        assert_eq!(read(&target), kept, "{case}: the buffer's bytes");
    }
}

#[test]
fn a_merge_of_the_first_level_fetches_through_its_view() {
    let scratch = Scratch::new("tiled-texture-merge-fetches");
    let rig = rig(&scratch);
    for shape in SHAPES {
        let ((width, height), levels) = shape;
        let case = format!("{width}x{height} of {levels} levels");
        let target = buffer(&rig.device);
        let texture = target
            .new_tiled_texture(&descriptor(shape), 0, PITCH)
            .unwrap();
        write_rows(&target, (width, height), |x, y| texel(8, x, y));
        // A blit into an 8x8 part of level 0: kept and merged back, the `Shared` buffer's bytes
        // being another writer's (`shadow.rs`, **Merged back**).
        let (x0, y0, side) = (4, 6, 8);
        let pattern = packed((side, side), |x, y| texel(9, x, y));
        let source = holding(&rig.device, side * side * 4, &pattern);
        let at = (Origin::new(x0, y0, 0), Size::new(side, side, 1));
        let merged = copies_of(&rig.device, || {
            rig.run(|command_buffer| blit_into(command_buffer, (&source, &texture), 0, at));
        });
        eprintln!("TILED-TEXTURE: {case}: a blit into part of level 0 {merged:?}");
        assert_eq!(merged.fetched, 1, "{case}: {merged:?}");
        let got = read(&target);
        for y in 0..height {
            for x in 0..PITCH / 4 {
                let byte = (y * PITCH + x * 4) as usize;
                let inside = (x0..x0 + side).contains(&x) && (y0..y0 + side).contains(&y);
                let expected = match (x < width, inside) {
                    (true, true) => texel(9, x - x0, y - y0),
                    (true, false) => texel(8, x, y),
                    (false, _) => [BETWEEN; 4],
                };
                assert_eq!(got[byte..byte + 4], expected, "{case}: bytes {byte}");
            }
        }
    }
}

#[test]
fn detached_bytes_leave_the_first_level_s_last_texels_and_the_later_levels() {
    // The pages, declared before the device that uses them, so dropped after it.
    let (host, other) = (Host::new(), Host::new());
    let scratch = Scratch::new("tiled-texture-detached");
    let rig = rig(&scratch);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(no_copy) = over_host(&rig.device, &host, &runs) else {
        return;
    };
    let shape = SHAPES[0];
    let ((width, height), levels) = shape;
    let mut d = descriptor(shape);
    d.resource_options = no_copy.resource_options();
    let texture = no_copy.new_tiled_texture(&d, 0, PITCH).unwrap();
    // A one-level texture over the same bytes, whose CPU access a detach refuses.
    let mut one = descriptor(((width, height), 1));
    one.resource_options = no_copy.resource_options();
    let plain = no_copy.new_texture(&one, 0, PITCH).unwrap();
    // The later levels, written on the CPU; level 0 copied in by a blit out of it, then written
    // again on the CPU: its last texels are those.
    for level in 1..levels {
        let (w, h) = ((width >> level).max(1), (height >> level).max(1));
        let pattern = packed((w, h), |x, y| texel(0x20 + level as u8, x, y));
        let region = Region::new_2d(0, 0, w, h);
        (texture.replace_region(region, level, 0, &pattern, w * 4, 0)).unwrap();
    }
    write_rows(&no_copy, (width, height), |x, y| texel(10, x, y));
    let row = width * 4;
    let read_back = (rig
        .device
        .new_buffer(row * height, ResourceOptions::shared()))
    .unwrap();
    rig.run(|command_buffer| {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(width, height, 1),
            &read_back,
            0,
            row,
            0,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
    });
    assert_eq!(
        read(&read_back),
        packed((width, height), |x, y| texel(10, x, y))
    );
    let last = packed((width, height), |x, y| texel(11, x, y));
    write_rows(&no_copy, (width, height), |x, y| texel(11, x, y));
    let image = texture.raw_image().unwrap();
    assert_eq!(no_copy.detach_backing(), Ok(true), "the detach");
    assert_eq!(
        texture.raw_image().unwrap(),
        image,
        "its own image, no stand-in"
    );
    assert_eq!(
        level_bytes(&texture, (width, height), 0),
        last,
        "level 0's last texels"
    );
    let refused = Err(Error::NotImplemented(RuntimeGap::DetachedTexelAccess));
    let whole = Region::new_2d(0, 0, width, height);
    assert_eq!(
        texture.replace_region(whole, 0, 0, &last, row, 0),
        refused,
        "a write of level 0"
    );
    let mut plain_bytes = vec![0u8; last.len()];
    assert_eq!(
        plain.get_bytes(&mut plain_bytes, row, 0, whole, 0, 0),
        refused,
        "the one-level texture"
    );
    for level in 1..levels {
        let (w, h) = ((width >> level).max(1), (height >> level).max(1));
        let pattern = packed((w, h), |x, y| texel(0x20 + level as u8, x, y));
        let got = level_bytes(&texture, (width, height), level);
        assert_eq!(got, pattern, "level {level} kept");
        let written: Vec<u8> = pattern.iter().map(|b| !b).collect();
        let region = Region::new_2d(0, 0, w, h);
        (texture.replace_region(region, level, 0, &written, w * 4, 0)).unwrap();
        let got = level_bytes(&texture, (width, height), level);
        assert_eq!(got, written, "level {level} written while detached");
    }
    // Pages again: level 0 their bytes, the later levels as written.
    let moved = packed((width, height), |x, y| texel(12, x, y));
    // SAFETY: `other` outlives the buffer (dropped after the device), which alone uses it.
    unsafe { no_copy.rebind_bytes_no_copy(other.memory(), None, None) }.unwrap();
    write_rows(&no_copy, (width, height), |x, y| texel(12, x, y));
    let level_one = (width / 2, height / 2);
    let one_back = (rig.device)
        .new_buffer(level_one.0 * level_one.1 * 4, ResourceOptions::shared())
        .unwrap();
    rig.run(|command_buffer| {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for (level, (w, h), out) in [(0, (width, height), &read_back), (1, level_one, &one_back)] {
            blit.copy_from_texture_to_buffer(
                &texture,
                0,
                level,
                Origin::new(0, 0, 0),
                Size::new(w, h, 1),
                out,
                0,
                w * 4,
                0,
                BlitOption::None,
            )
            .unwrap();
        }
        blit.end_encoding().unwrap();
    });
    assert_eq!(read(&read_back), moved, "level 0 over the new pages");
    let one_pattern = packed(level_one, |x, y| texel(0x21, x, y));
    let one_written: Vec<u8> = one_pattern.iter().map(|b| !b).collect();
    assert_eq!(read(&one_back), one_written, "level 1 through the rebind");
    eprintln!(
        "TILED-TEXTURE on {}: detached, level 0 read its last texels and levels 1 to {} kept",
        rig.device.name(),
        levels - 1
    );
    drop((texture, plain, no_copy));
    drop(rig);
}
