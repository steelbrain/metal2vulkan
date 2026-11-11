//! `Buffer::new_tiled_texture`, Metal's private `newTiledTextureWithDescriptor:offset:bytesPerRow:`,
//! of more than one level: one image of every level, a copy of the buffer's bytes taken at the
//! texture's first use and again at the first use after `Texture::did_modify_data`
//! (`shadow.rs`'s **Copies taken at use**), as two x86 Macs take it (an AMD Radeon Pro 5300M and an
//! Intel UHD Graphics 630 under macOS 26.7.1). The shapes: the x86 macOS guests' compositor's
//! `BGRA8Unorm` 2D textures of 54x54 and 6 levels and of 64x60 and 7, `ShaderRead` alone, the
//! descriptor `Managed`, over a new 16384-byte `Shared` buffer from offset 0 at 256 bytes a row;
//! and the Macs' probe's 256x256 of 9 levels, `RGBA8Unorm` and `BGRA8Unorm`, rows packed:
//!
//! - **The first use takes the bytes written after the making.** Bytes written after the making
//!   are what the first command buffer's blits read out of the first level, and every later
//!   level is made from the one above (a uniform grey in every level; a gradient's level each the
//!   2x2 box of the one above, to within one a byte), the one copy in counted
//!   (`WorkCounts::shadow_texels_copied_in`: the first level's texels once, nothing copied back).
//!   So does a first use that is a kernel's sample at LOD 1, the case that read zeros before.
//! - **No later write reaches a level.** A CPU write after the first use reaches no level, and the
//!   next use copies nothing in.
//! - **`did_modify_data`** takes the copy again at the next use, not at the call: a write after
//!   the call is in every level, and a later write without one is in none.
//! - **A blit into the first level** reaches neither the bytes nor the later levels.
//! - **Every format's later levels** are made as `generate_mipmaps` makes them: an integer
//!   format's (`RGBA8Uint`) by plain Metal's texel picks, exactly, whether the first use is a blit
//!   or a pass into level 1 (which clears that level after the picks); an extended-range format's
//!   (`BGR10XR`, which no blit filters) by the compute filter, each level within the filter's
//!   bound of the one above (`bind::MipmapFilterChain::outside_bound`), the take counted in
//!   `WorkCounts::mipmaps_filtered`.
//! - **The first level on the CPU is the bytes**, as on the Macs: `get_bytes` reads them before
//!   the first use, after a GPU write into the level (which the GPU reads) and after a CPU write
//!   of them (which no level sees); a `replace_region` there writes them, the rows' padding left,
//!   and the next use takes the copy again, every level made from it.
//! - **Mipmaps and the later levels.** `generate_mipmaps` makes level 1 from the first; blits into
//!   and out of each level past the first, and the CPU's `replace_region` and `get_bytes` there,
//!   round-trip, and leave the buffer's bytes as they were.
//! - **A pass into a later level** as the first use takes the copy first and clears that level.
//! - **One level** is `new_texture`'s: the same placement and the same path over a no-copy
//!   buffer, in place where the device's check passed.
//! - **Announced and counted.** Each texture is announced (`shadowed-texture`, at every
//!   occurrence where the device's options ask), the notice naming its levels, counted in
//!   `WorkCounts::shadowed_textures` and `shadows_zeroed`, and its placement query answers
//!   `Shadow(Mipmapped)`.
//! - **Detached.** Over a no-copy buffer's pages, detached (`detach_backing`), the texture keeps
//!   its own image (its `VkImage` the same): its blits read every level as the last take left it,
//!   and the CPU's calls on a later level read it so and write it; on the first, the bytes, they
//!   are refused (`NotImplemented(DetachedTexelAccess)`). A rebind to new pages changes no level
//!   until a `did_modify_data`, after which the next use takes their bytes.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Printed (`TILED-TEXTURE`): each
//! shape's read and counted values.

use super::blit_holds::{device, over_host, Host, Scratch, WAIT};
use super::draw_again::{device as hearing_device, Heard};
use super::encoder_holds::{library, pass};
use crate::{
    BlitOption, Buffer, ClearColor, CommandQueue, ComputePipelineState, CpuCacheMode, Degradation,
    Device, Error, HazardTrackingMode, Origin, PipelineOptions, Region, ResourceOptions,
    RuntimeGap, SamplerState, Size, StorageMode, Texture, TextureDescriptor, TexturePlacement,
    TextureShadowReason,
};
use metal2vulkan::bind::{MipmapFilterChain, SamplerDescriptor};
use metal2vulkan::metal::{PixelFormat, SamplerMipFilter, TextureUsage};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// A texture's level 0 width and height, and its level count.
pub(super) type Shape = ((u64, u64), u64);

/// The guest's compositor's shapes.
pub(super) const SHAPES: [Shape; 2] = [((54, 54), 6), ((64, 60), 7)];

/// The guest's buffer: its length and the texture's `bytesPerRow`.
pub(super) const LENGTH: u64 = 16384;
pub(super) const PITCH: u64 = 256;

/// The Macs' probe's shape, rows packed.
const SQUARE: Shape = ((256, 256), 9);
const SQUARE_PITCH: u64 = 256 * 4;

/// How long the device's in-place check may take to settle.
const SETTLE: Duration = Duration::from_secs(120);

/// A descriptor of `shape` in `format`: `ShaderRead`, `Managed` (the texture takes the buffer's
/// options, as `new_texture`'s does), GPU-optimized contents allowed.
fn descriptor_of(format: PixelFormat, ((width, height), levels): Shape) -> TextureDescriptor {
    let mut d = TextureDescriptor::texture_2d(format, width, height, false);
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

/// The guest's descriptor of `shape`: `BGRA8Unorm`.
pub(super) fn descriptor(shape: Shape) -> TextureDescriptor {
    descriptor_of(PixelFormat::BGRA8Unorm, shape)
}

/// A new `Shared` buffer of `length` bytes, every byte zero.
pub(super) fn zeroed(device: &Device, length: u64) -> Buffer {
    let buffer = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    write(&buffer, |bytes| bytes.fill(0));
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
pub(super) fn write(buffer: &Buffer, edit: impl FnOnce(&mut [u8])) {
    let contents = buffer.contents().unwrap();
    // SAFETY: a `Shared` buffer's contents span its length, and no command runs over them while
    // the test writes them (each test waits for its command buffers), nor does another borrow.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(contents.as_ptr(), buffer.length() as usize) };
    edit(bytes);
}

/// A copy of `buffer`'s bytes, which no command writes meanwhile.
pub(super) fn read(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: as `write`'s; the bytes are copied out at once.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// The texel `(x, y)` of a pattern seeded `seed`.
fn texel(seed: u8, x: u64, y: u64) -> [u8; 4] {
    let (x, y) = (x as u8, y as u8);
    [
        x.wrapping_mul(7) ^ seed,
        y.wrapping_mul(13) ^ seed,
        x ^ y.rotate_left(3),
        0xff ^ seed,
    ]
}

/// The Macs' probe's gradient: `(x, y, 0x80, 0xff)`.
fn gradient(x: u64, y: u64) -> [u8; 4] {
    [x as u8, y as u8, 0x80, 0xff]
}

/// Write `of(x, y)` into the `width` x `height` texels at `pitch` bytes a row of `buffer`.
pub(super) fn write_rows(
    buffer: &Buffer,
    ((width, height), pitch): ((u64, u64), u64),
    of: impl Fn(u64, u64) -> [u8; 4],
) {
    write(buffer, |bytes| {
        for y in 0..height {
            for x in 0..width {
                let at = (y * pitch + x * 4) as usize;
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

/// The level-0 pattern whose level 1 the device's filter makes known: blue the checkerboard of 0
/// and 255 a texel, which a 2x2 box averages to 127 or 128; green one value a 2x2 block, which it
/// keeps; red 0x40 and alpha 255.
pub(super) fn checker(x: u64, y: u64) -> [u8; 4] {
    let blue = if (x + y).is_multiple_of(2) { 0 } else { 0xff };
    let green = ((x / 2) * 9 + (y / 2) * 5) as u8;
    [blue, green, 0x40, 0xff]
}

/// The width and height of `level` of a level 0 of `(width, height)`.
pub(super) fn level_extent((width, height): (u64, u64), level: u64) -> (u64, u64) {
    ((width >> level).max(1), (height >> level).max(1))
}

/// Whether every byte of `got` lies within one of `expected`'s byte at its place in a texel.
pub(super) fn near_each(got: &[u8], expected: [u8; 4]) -> bool {
    (got.chunks(4)).all(|t| t.iter().zip(expected).all(|(&g, e)| g.abs_diff(e) <= 1))
}

/// Where `below` (of `size`, each axis half `above`'s) is not the 2x2 box of `above` to within one
/// a byte: the first texel and its two values.
fn off_the_box(above: &[u8], below: &[u8], (w, h): (u64, u64)) -> Option<String> {
    let aw = 2 * w;
    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                let at = |xx: u64, yy: u64| u32::from(above[((yy * aw + xx) * 4 + c) as usize]);
                let sum = at(2 * x, 2 * y)
                    + at(2 * x + 1, 2 * y)
                    + at(2 * x, 2 * y + 1)
                    + at(2 * x + 1, 2 * y + 1);
                let got = u32::from(below[((y * w + x) * 4 + c) as usize]);
                // Within one of the box's mean, sum / 4, either way.
                if 4 * got + 4 < sum || 4 * got > sum + 4 {
                    return Some(format!("texel {x},{y} channel {c}: {got}, box {sum}/4"));
                }
            }
        }
    }
    None
}

/// `level` of a first level of `(width, height)` whose packed 4-byte texels are `first`, as plain
/// Metal's picks make it: texel `i` of a level is texel `floor((2i + 1) W / 2w)` of the level above
/// on each axis, `W` and `w` the two levels' sizes on it (`mipmap_plan::PickPlan`).
fn picked(first: &[u8], (width, height): (u64, u64), level: u64) -> Vec<u8> {
    let (w, h) = level_extent((width, height), level);
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let (mut sx, mut sy) = (x, y);
            for n in (1..=level).rev() {
                let (above_w, above_h) = level_extent((width, height), n - 1);
                let (here_w, here_h) = level_extent((width, height), n);
                sx = (2 * sx + 1) * above_w / (2 * here_w);
                sy = (2 * sy + 1) * above_h / (2 * here_h);
            }
            let at = ((sy * width + sx) * 4) as usize;
            out.extend_from_slice(&first[at..at + 4]);
        }
    }
    out
}

/// The device's queue, the sampling kernels, and a sampler with a nearest mip filter.
pub(super) struct Rig {
    pub(super) device: Device,
    pub(super) queue: CommandQueue,
    /// `sample_level`: `t.sample(s, float2(gid) * 0.25, level(1.0))` into `out[gid]`.
    pub(super) level_one: ComputePipelineState,
    sampler: SamplerState,
}

impl Rig {
    fn new(device: Device) -> Rig {
        let sampling = library(&device, "engine/sampling.ll");
        let function = sampling.new_function_with_name("sample_level").unwrap();
        let level_one = (device
            .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default()))
        .unwrap();
        let mut d = SamplerDescriptor::new();
        d.mip_filter = SamplerMipFilter::Nearest;
        let sampler = device.new_sampler_state(&d).unwrap();
        Rig {
            queue: device.new_command_queue().unwrap(),
            device,
            level_one,
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
    pub(super) fn sample(
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
    pub(super) fn samples(out: &Buffer) -> [[f32; 4]; 4] {
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

    /// Every level of `texture`, of `shape`, read out by one command buffer's blits, and what
    /// that command buffer copied of the device's shadows.
    pub(super) fn levels(
        &self,
        texture: &Texture,
        ((width, height), levels): Shape,
    ) -> (Vec<Vec<u8>>, Copies) {
        let outs: Vec<(Buffer, (u64, u64))> = (0..levels)
            .map(|level| {
                let (w, h) = level_extent((width, height), level);
                let out = self.device.new_buffer(w * h * 4, ResourceOptions::shared());
                (out.unwrap(), (w, h))
            })
            .collect();
        let copies = copies_of(&self.device, || {
            self.run(|command_buffer| {
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                for (level, (out, (w, h))) in outs.iter().enumerate() {
                    blit.copy_from_texture_to_buffer(
                        texture,
                        0,
                        level as u64,
                        Origin::new(0, 0, 0),
                        Size::new(*w, *h, 1),
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
        });
        (outs.iter().map(|(out, _)| read(out)).collect(), copies)
    }
}

/// The `BGRA8Unorm` texel a sample read, its channels red, green, blue, alpha over 255 each.
pub(super) fn texel_of([r, g, b, a]: [f32; 4]) -> [u8; 4] {
    [b, g, r, a].map(|c| (c * 255.0).round() as u8)
}

/// The rig of the device the cfg selects, its shader cache in `scratch`.
pub(super) fn rig(scratch: &Scratch) -> Rig {
    Rig::new(device(&scratch.0))
}

/// What a run of commands copied of a shadow (`WorkCounts`, after less before).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Copies {
    /// `shadow_copies`: every copy in, copy back, keep and merge.
    pub(super) copies: u64,
    /// `shadow_texels_copied_in`.
    pub(super) texels_in: u64,
    /// `shadow_texels_copied_back`.
    pub(super) texels_back: u64,
}

/// What `run` copied of the device's shadows.
pub(super) fn copies_of(device: &Device, run: impl FnOnce()) -> Copies {
    let before = device.work_counts();
    run();
    let after = device.work_counts();
    Copies {
        copies: after.shadow_copies - before.shadow_copies,
        texels_in: after.shadow_texels_copied_in - before.shadow_texels_copied_in,
        texels_back: after.shadow_texels_copied_back - before.shadow_texels_copied_back,
    }
}

/// The copies of one take of a texture whose first level is `first` texels: its texels in once,
/// nothing back.
pub(super) fn one_take(first: u64) -> Copies {
    Copies {
        copies: 1,
        texels_in: first,
        texels_back: 0,
    }
}

/// No copy.
pub(super) const NONE: Copies = Copies {
    copies: 0,
    texels_in: 0,
    texels_back: 0,
};

/// The cases of the uniform grey: the compositor's shapes over the guest's buffer, and the probe's
/// square in both formats over a buffer of its packed rows.
fn grey_cases() -> Vec<(PixelFormat, Shape, u64, u64)> {
    let mut cases: Vec<_> = SHAPES
        .iter()
        .map(|&shape| (PixelFormat::BGRA8Unorm, shape, PITCH, LENGTH))
        .collect();
    for format in [PixelFormat::RGBA8Unorm, PixelFormat::BGRA8Unorm] {
        cases.push((format, SQUARE, SQUARE_PITCH, SQUARE_PITCH * 256));
    }
    cases
}

#[test]
fn the_first_use_takes_the_bytes_written_after_the_making_into_every_level() {
    let scratch = Scratch::new("tiled-texture-first-use");
    let rig = rig(&scratch);
    let grey = [128, 128, 128, 255];
    for (format, shape, pitch, length) in grey_cases() {
        let ((width, height), levels) = shape;
        let case = format!("{format:?} {width}x{height} of {levels} levels");
        let target = zeroed(&rig.device, length);
        let texture = target
            .new_tiled_texture(&descriptor_of(format, shape), 0, pitch)
            .unwrap();
        assert_eq!(texture.mipmap_level_count(), levels, "{case}");
        // Written after the making, before any use.
        write_rows(&target, ((width, height), pitch), |_, _| grey);
        let (got, copies) = rig.levels(&texture, shape);
        eprintln!("TILED-TEXTURE: {case}: the first use {copies:?}");
        for (level, bytes) in got.iter().enumerate() {
            let first = bytes.get(..4);
            eprintln!("TILED-TEXTURE: {case}: level {level} starts {first:?}");
            assert!(near_each(bytes, grey), "{case}: level {level} {first:?}");
        }
        assert_eq!(copies, one_take(width * height), "{case}");
    }
}

#[test]
fn every_later_level_is_made_from_the_one_above() {
    let scratch = Scratch::new("tiled-texture-gradient");
    let rig = rig(&scratch);
    for format in [PixelFormat::RGBA8Unorm, PixelFormat::BGRA8Unorm] {
        let ((width, height), levels) = SQUARE;
        let case = format!("{format:?} {width}x{height} of {levels} levels");
        let target = zeroed(&rig.device, SQUARE_PITCH * height);
        let texture =
            (target.new_tiled_texture(&descriptor_of(format, SQUARE), 0, SQUARE_PITCH)).unwrap();
        write_rows(&target, ((width, height), SQUARE_PITCH), gradient);
        let (got, _) = rig.levels(&texture, SQUARE);
        assert_eq!(got[0], packed((width, height), gradient), "{case}: level 0");
        for (above, pair) in got.windows(2).enumerate() {
            let level = above + 1;
            let size = level_extent((width, height), level as u64);
            let off = off_the_box(&pair[0], &pair[1], size);
            assert_eq!(off, None, "{case}: level {level}");
        }
        let last = &got[levels as usize - 1];
        eprintln!("TILED-TEXTURE: {case}: the gradient's last level {last:?}");
    }
}

#[test]
fn a_first_sample_at_level_one_reads_the_level_made_from_the_bytes() {
    let scratch = Scratch::new("tiled-texture-first-sample");
    let rig = rig(&scratch);
    for shape in SHAPES {
        let ((width, height), levels) = shape;
        let case = format!("{width}x{height} of {levels} levels");
        let target = zeroed(&rig.device, LENGTH);
        let texture = target
            .new_tiled_texture(&descriptor(shape), 0, PITCH)
            .unwrap();
        write_rows(&target, ((width, height), PITCH), checker);
        let out = rig.out();
        let copies = copies_of(&rig.device, || {
            rig.run(|command_buffer| rig.sample(command_buffer, &rig.level_one, (&texture, &out)));
        });
        // A sample at LOD 1 reads level 1's averaged blue, which level 0 holds nowhere.
        let samples = Rig::samples(&out);
        eprintln!("TILED-TEXTURE: {case}: level 1 sampled {samples:?}, {copies:?}");
        for sample in samples {
            let [blue, _, red, alpha] = texel_of(sample);
            assert!((127..=128).contains(&blue), "{case}: blue {blue}");
            assert_eq!([red, alpha], [0x40, 0xff], "{case}");
        }
        assert_eq!(copies, one_take(width * height), "{case}");
    }
}

#[test]
fn a_cpu_write_after_the_first_use_reaches_no_level() {
    let scratch = Scratch::new("tiled-texture-later-write");
    let rig = rig(&scratch);
    let ((width, height), _) = SQUARE;
    let target = zeroed(&rig.device, SQUARE_PITCH * height);
    let d = descriptor_of(PixelFormat::RGBA8Unorm, SQUARE);
    let texture = target.new_tiled_texture(&d, 0, SQUARE_PITCH).unwrap();
    write_rows(&target, ((width, height), SQUARE_PITCH), gradient);
    let (first, _) = rig.levels(&texture, SQUARE);
    write(&target, |bytes| bytes.fill(0x40));
    let (again, copies) = rig.levels(&texture, SQUARE);
    eprintln!("TILED-TEXTURE: a use after a CPU write {copies:?}");
    assert_eq!(copies, NONE, "the use after the first");
    for (level, (a, b)) in first.iter().zip(&again).enumerate() {
        assert_eq!(a, b, "level {level}");
    }
}

#[test]
fn did_modify_data_takes_the_copy_again_at_the_next_use() {
    let scratch = Scratch::new("tiled-texture-did-modify");
    let rig = rig(&scratch);
    let ((width, height), _) = SQUARE;
    let rows = ((width, height), SQUARE_PITCH);
    let target = zeroed(&rig.device, SQUARE_PITCH * height);
    let d = descriptor_of(PixelFormat::RGBA8Unorm, SQUARE);
    let texture = target.new_tiled_texture(&d, 0, SQUARE_PITCH).unwrap();
    // The Macs' probe's steps: the first use over 0x40 bytes; rows written, the call, then 0x20
    // everywhere with no call; then 0x10 with no call.
    write(&target, |bytes| bytes.fill(0x40));
    let (made, _) = rig.levels(&texture, SQUARE);
    assert!(made.iter().all(|level| near_each(level, [0x40; 4])), "made");
    write_rows(&target, rows, gradient);
    texture.did_modify_data();
    write(&target, |bytes| bytes.fill(0x20));
    let (taken, copies) = rig.levels(&texture, SQUARE);
    eprintln!("TILED-TEXTURE: the use after did_modify_data {copies:?}");
    for (level, bytes) in taken.iter().enumerate() {
        assert!(near_each(bytes, [0x20; 4]), "level {level} after the call");
    }
    assert_eq!(copies, one_take(width * height), "the use after the call");
    write(&target, |bytes| bytes.fill(0x10));
    let (kept, copies) = rig.levels(&texture, SQUARE);
    assert_eq!(copies, NONE, "a use after a write with no call");
    assert_eq!(kept, taken, "a write with no call");
}

#[test]
fn a_blit_into_the_first_level_reaches_neither_the_bytes_nor_the_later_levels() {
    let scratch = Scratch::new("tiled-texture-first-level-blit");
    let rig = rig(&scratch);
    let ((width, height), levels) = SQUARE;
    let target = zeroed(&rig.device, SQUARE_PITCH * height);
    let d = descriptor_of(PixelFormat::RGBA8Unorm, SQUARE);
    let texture = target.new_tiled_texture(&d, 0, SQUARE_PITCH).unwrap();
    write(&target, |bytes| bytes.fill(0x40));
    rig.levels(&texture, SQUARE);
    let source = holding(
        &rig.device,
        width * height * 4,
        &vec![0x20; (width * height * 4) as usize],
    );
    let at = (Origin::new(0, 0, 0), Size::new(width, height, 1));
    let copies = copies_of(&rig.device, || {
        rig.run(|command_buffer| blit_into(command_buffer, (&source, &texture), 0, at));
    });
    assert_eq!(copies, NONE, "the blit into level 0");
    let (got, _) = rig.levels(&texture, SQUARE);
    assert!(
        near_each(&got[0], [0x20; 4]),
        "level 0 as the blit wrote it"
    );
    assert_eq!(got.len(), levels as usize);
    for (level, bytes) in got.iter().enumerate().skip(1) {
        assert!(near_each(bytes, [0x40; 4]), "level {level} as taken");
    }
    assert!(
        read(&target).iter().all(|&b| b == 0x40),
        "the buffer's bytes"
    );
}

#[test]
fn an_integer_format_s_later_levels_are_plain_metal_s_picks() {
    let scratch = Scratch::new("tiled-texture-picks");
    let rig = rig(&scratch);
    for shape in SHAPES {
        let ((width, height), levels) = shape;
        let case = format!("RGBA8Uint {width}x{height} of {levels} levels");
        let target = zeroed(&rig.device, LENGTH);
        let d = descriptor_of(PixelFormat::RGBA8Uint, shape);
        let texture = target.new_tiled_texture(&d, 0, PITCH).unwrap();
        write_rows(&target, ((width, height), PITCH), |x, y| texel(5, x, y));
        let (got, copies) = rig.levels(&texture, shape);
        eprintln!("TILED-TEXTURE: {case}: the first use {copies:?}");
        assert_eq!(copies, one_take(width * height), "{case}");
        let first = packed((width, height), |x, y| texel(5, x, y));
        assert_eq!(got[0], first, "{case}: level 0");
        for level in 1..levels {
            let expected = picked(&first, (width, height), level);
            assert_eq!(got[level as usize], expected, "{case}: level {level}");
        }
    }
}

#[test]
fn a_pass_into_level_one_of_an_integer_format_follows_the_picks() {
    let scratch = Scratch::new("tiled-texture-picks-pass");
    let rig = rig(&scratch);
    let shape = SHAPES[1];
    let ((width, height), levels) = shape;
    let target = zeroed(&rig.device, LENGTH);
    let mut d = descriptor_of(PixelFormat::RGBA8Uint, shape);
    d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let texture = target.new_tiled_texture(&d, 0, PITCH).unwrap();
    write_rows(&target, ((width, height), PITCH), |x, y| texel(9, x, y));
    let mut into_one = pass(&texture);
    into_one.color_attachments[0].level = 1;
    into_one.color_attachments[0].clear_color = ClearColor::new(7.0, 9.0, 11.0, 13.0);
    // The pass is the first use: the take, its picks among it, records before it.
    let copies = copies_of(&rig.device, || {
        rig.run(|command_buffer| {
            let encoder = command_buffer.render_command_encoder(&into_one).unwrap();
            encoder.end_encoding().unwrap();
        });
    });
    assert_eq!(copies, one_take(width * height), "the pass");
    let (got, again) = rig.levels(&texture, shape);
    eprintln!("TILED-TEXTURE: a pass into level 1 of RGBA8Uint {copies:?}, then {again:?}");
    assert_eq!(again, NONE, "the use after the pass");
    let first = packed((width, height), |x, y| texel(9, x, y));
    assert_eq!(got[0], first, "level 0");
    assert!(
        got[1].chunks(4).all(|t| t == [7, 9, 11, 13]),
        "level 1 as the pass cleared it after the picks: {:?}",
        &got[1][..8]
    );
    for level in 2..levels {
        let expected = picked(&first, (width, height), level);
        assert_eq!(got[level as usize], expected, "level {level}");
    }
}

#[test]
fn an_extended_range_format_s_later_levels_take_the_compute_filter() {
    let scratch = Scratch::new("tiled-texture-filter");
    let rig = rig(&scratch);
    let shape = SHAPES[1];
    let ((width, height), levels) = shape;
    let target = zeroed(&rig.device, LENGTH);
    let d = descriptor_of(PixelFormat::BGR10XR, shape);
    let texture = target.new_tiled_texture(&d, 0, PITCH).unwrap();
    write_rows(&target, ((width, height), PITCH), |x, y| texel(13, x, y));
    let filtered = |device: &Device| device.work_counts().mipmaps_filtered;
    let before = filtered(&rig.device);
    let (got, copies) = rig.levels(&texture, shape);
    let ran = filtered(&rig.device) - before;
    eprintln!("TILED-TEXTURE: BGR10XR's first use {copies:?}, the compute filter run {ran} times");
    assert_eq!(copies, one_take(width * height), "the first use");
    assert_eq!(ran, 1, "the take's compute filter");
    assert_eq!(got[0], packed((width, height), |x, y| texel(13, x, y)));
    let extent = [width as u32, height as u32, 1];
    let chain = MipmapFilterChain::plan(
        PixelFormat::BGR10XR,
        extent,
        1,
        levels as u32,
        false,
        64,
        u32::MAX,
        [65535, 65535],
        None,
    )
    .unwrap();
    for level in 1..levels as usize {
        let outside = chain.outside_bound(level, &got[level - 1], &got[level]);
        let outside = outside.unwrap();
        assert!(outside.is_empty(), "level {level}: texels {outside:?}");
    }
}

#[test]
fn the_first_level_on_the_cpu_is_the_buffer_s_bytes() {
    let scratch = Scratch::new("tiled-texture-first-level-cpu");
    let rig = rig(&scratch);
    let shape = SHAPES[0];
    let ((width, height), _) = shape;
    let rows = ((width, height), PITCH);
    let target = zeroed(&rig.device, LENGTH);
    let texture = target
        .new_tiled_texture(&descriptor(shape), 0, PITCH)
        .unwrap();
    write_rows(&target, rows, |x, y| texel(20, x, y));
    let bytes = packed((width, height), |x, y| texel(20, x, y));
    let cpu = || level_bytes(&texture, (width, height), 0);
    assert_eq!(cpu(), bytes, "before the first use");
    let (taken, _) = rig.levels(&texture, shape);
    assert_eq!(taken[0], bytes, "taken");
    // A GPU write into the first level stays in the texture: the GPU reads it, the CPU the bytes.
    let texels = (width * height * 4) as usize;
    let grey = holding(&rig.device, texels as u64, &vec![0x33; texels]);
    let at = (Origin::new(0, 0, 0), Size::new(width, height, 1));
    rig.run(|command_buffer| blit_into(command_buffer, (&grey, &texture), 0, at));
    let (written, _) = rig.levels(&texture, shape);
    assert!(
        written[0].iter().all(|&b| b == 0x33),
        "the GPU reads its write"
    );
    assert_eq!(
        cpu(),
        bytes,
        "the CPU reads the bytes after the GPU's write"
    );
    // A CPU write of the bytes is what the CPU reads there, and no level sees it.
    write_rows(&target, rows, |x, y| texel(21, x, y));
    assert_eq!(cpu(), packed((width, height), |x, y| texel(21, x, y)));
    let (kept, copies) = rig.levels(&texture, shape);
    assert_eq!(copies, NONE, "a use after the CPU's write");
    assert_eq!(kept, written, "every level as the GPU left it");
}

#[test]
fn a_replace_region_of_the_first_level_writes_the_bytes_and_the_next_use_takes_them() {
    let scratch = Scratch::new("tiled-texture-first-level-replace");
    let rig = rig(&scratch);
    let shape = SHAPES[0];
    let ((width, height), levels) = shape;
    let target = zeroed(&rig.device, LENGTH);
    let texture = target
        .new_tiled_texture(&descriptor(shape), 0, PITCH)
        .unwrap();
    write(&target, |bytes| bytes.fill(0x40));
    let (taken, _) = rig.levels(&texture, shape);
    assert!(
        taken.iter().all(|level| near_each(level, [0x40; 4])),
        "taken"
    );
    let grey = [0x20, 0x60, 0xa0, 0xff];
    let written = packed((width, height), |_, _| grey);
    let region = Region::new_2d(0, 0, width, height);
    (texture.replace_region(region, 0, 0, &written, width * 4, 0)).unwrap();
    // The bytes hold the rows, each row's padding as it was.
    let bytes = read(&target);
    for y in 0..height {
        let row = &bytes[(y * PITCH) as usize..((y + 1) * PITCH) as usize];
        let (texels, padding) = row.split_at((width * 4) as usize);
        assert!(texels.chunks(4).all(|t| t == grey), "row {y}");
        assert!(padding.iter().all(|&b| b == 0x40), "row {y}'s padding");
    }
    let (again, copies) = rig.levels(&texture, shape);
    eprintln!("TILED-TEXTURE: the use after a replace_region of level 0 {copies:?}");
    assert_eq!(copies, one_take(width * height), "the use after the write");
    assert_eq!(again.len(), levels as usize);
    for (level, bytes) in again.iter().enumerate() {
        assert!(near_each(bytes, grey), "level {level} after the write");
    }
    let (kept, copies) = rig.levels(&texture, shape);
    assert_eq!(copies, NONE, "the use after that");
    assert_eq!(kept, again);
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

/// `level`'s texels of `texture` (of the level-0 extent `(width, height)`), read on the CPU.
fn level_bytes(texture: &Texture, (width, height): (u64, u64), level: u64) -> Vec<u8> {
    let (w, h) = level_extent((width, height), level);
    let mut bytes = vec![0u8; (w * h * 4) as usize];
    let region = Region::new_2d(0, 0, w, h);
    (texture.get_bytes(&mut bytes, w * 4, 0, region, level, 0)).unwrap();
    bytes
}

#[test]
fn generated_mipmaps_fill_the_level_a_level_one_sample_reads() {
    let scratch = Scratch::new("tiled-texture-mipmaps");
    let rig = rig(&scratch);
    for shape in SHAPES {
        let ((width, height), levels) = shape;
        let case = format!("{width}x{height} of {levels} levels");
        let target = zeroed(&rig.device, LENGTH);
        let texture = target
            .new_tiled_texture(&descriptor(shape), 0, PITCH)
            .unwrap();
        write_rows(&target, ((width, height), PITCH), checker);
        let out = rig.out();
        rig.run(|command_buffer| {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.generate_mipmaps(&texture).unwrap();
            blit.end_encoding().unwrap();
            rig.sample(command_buffer, &rig.level_one, (&texture, &out));
        });
        // Level 1 as the CPU reads it: each texel the 2x2 box of level 0's.
        let (w, h) = (width / 2, height / 2);
        let level_one = level_bytes(&texture, (width, height), 1);
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
        let samples = Rig::samples(&out);
        eprintln!("TILED-TEXTURE: {case}: level 1 sampled after generate_mipmaps {samples:?}");
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
        let target = zeroed(&rig.device, LENGTH);
        let texture = target
            .new_tiled_texture(&descriptor(shape), 0, PITCH)
            .unwrap();
        write_rows(&target, ((width, height), PITCH), |x, y| texel(3, x, y));
        let kept = read(&target);
        for level in 1..levels {
            let (w, h) = level_extent((width, height), level);
            let row = w * 4;
            let seed = 0x40 + level as u8;
            let pattern = packed((w, h), |x, y| texel(seed, x, y));
            let source = holding(&rig.device, row * h, &pattern);
            let read_back = (rig.device.new_buffer(row * h, ResourceOptions::shared())).unwrap();
            let size = Size::new(w, h, 1);
            let origin = Origin::new(0, 0, 0);
            let out = |command_buffer: &crate::CommandBuffer| {
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
            };
            rig.run(|command_buffer| {
                blit_into(command_buffer, (&source, &texture), level, (origin, size));
                out(command_buffer);
            });
            assert_eq!(read(&read_back), pattern, "{case}: level {level} by blits");
            let cpu = level_bytes(&texture, (width, height), level);
            assert_eq!(cpu, pattern, "{case}: level {level} read on the CPU");
            // The CPU's write there, read back by a blit.
            let written: Vec<u8> = pattern.iter().map(|b| !b).collect();
            let region = Region::new_2d(0, 0, w, h);
            (texture.replace_region(region, level, 0, &written, row, 0)).unwrap();
            rig.run(out);
            let got = read(&read_back);
            assert_eq!(got, written, "{case}: level {level} written on the CPU");
        }
        assert_eq!(read(&target), kept, "{case}: the buffer's bytes");
    }
}

#[test]
fn a_render_pass_into_a_later_level_takes_the_copy_first() {
    let scratch = Scratch::new("tiled-texture-later-level-pass");
    let rig = rig(&scratch);
    for shape in SHAPES {
        let ((width, height), levels) = shape;
        let case = format!("{width}x{height} of {levels} levels");
        let target = zeroed(&rig.device, LENGTH);
        let mut d = descriptor(shape);
        d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let texture = target.new_tiled_texture(&d, 0, PITCH).unwrap();
        write_rows(&target, ((width, height), PITCH), |x, y| texel(7, x, y));
        let kept = read(&target);
        let mut into_one = pass(&texture);
        into_one.color_attachments[0].level = 1;
        into_one.color_attachments[0].clear_color = ClearColor::new(1.0, 0.0, 1.0, 1.0);
        let passes = || {
            copies_of(&rig.device, || {
                rig.run(|command_buffer| {
                    let encoder = command_buffer.render_command_encoder(&into_one).unwrap();
                    encoder.end_encoding().unwrap();
                });
            })
        };
        let (first, second) = (passes(), passes());
        eprintln!("TILED-TEXTURE: {case}: passes into level 1 {first:?}, then {second:?}");
        assert_eq!(first, one_take(width * height), "{case}: the first pass");
        assert_eq!(second, NONE, "{case}: the second pass");
        let level_zero = level_bytes(&texture, (width, height), 0);
        assert_eq!(
            level_zero,
            packed((width, height), |x, y| texel(7, x, y)),
            "{case}"
        );
        let level_one = level_bytes(&texture, (width, height), 1);
        assert!(
            level_one.chunks(4).all(|t| t == [0xff, 0, 0xff, 0xff]),
            "{case}: level 1 cleared"
        );
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
        let target = zeroed(&device, LENGTH);
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

#[test]
fn detached_bytes_leave_every_level_as_taken() {
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
    let rows = ((width, height), PITCH);
    let mut d = descriptor(shape);
    d.resource_options = no_copy.resource_options();
    let texture = no_copy.new_tiled_texture(&d, 0, PITCH).unwrap();
    // Taken at the first use, then written again on the CPU, which no level sees.
    write_rows(&no_copy, rows, |x, y| texel(10, x, y));
    let (taken, _) = rig.levels(&texture, shape);
    let first = packed((width, height), |x, y| texel(10, x, y));
    assert_eq!(taken[0], first, "level 0 taken");
    write_rows(&no_copy, rows, |x, y| texel(11, x, y));
    let image = texture.raw_image().unwrap();
    assert_eq!(no_copy.detach_backing(), Ok(true), "the detach");
    assert_eq!(texture.raw_image().unwrap(), image, "its own image");
    // The first level, the bytes, is refused to the CPU while they are detached.
    let gap = Err(Error::NotImplemented(RuntimeGap::DetachedTexelAccess));
    let region = Region::new_2d(0, 0, width, height);
    let mut out = vec![0u8; first.len()];
    let read_first = texture.get_bytes(&mut out, width * 4, 0, region, 0, 0);
    assert_eq!(read_first, gap, "level 0 read while detached");
    let write_first = texture.replace_region(region, 0, 0, &first, width * 4, 0);
    assert_eq!(write_first, gap, "level 0 written while detached");
    // Every later level reads as taken and takes a CPU write.
    for level in 1..levels {
        let (w, h) = level_extent((width, height), level);
        let got = level_bytes(&texture, (width, height), level);
        assert_eq!(
            got, taken[level as usize],
            "level {level} as taken, read while detached"
        );
        let written = packed((w, h), |x, y| texel(0x20 + level as u8, x, y));
        let region = Region::new_2d(0, 0, w, h);
        (texture.replace_region(region, level, 0, &written, w * 4, 0)).unwrap();
        let got = level_bytes(&texture, (width, height), level);
        assert_eq!(got, written, "level {level} written while detached");
    }
    // Pages again: no level changes until the call, and the next use after it takes their bytes.
    // SAFETY: `other` outlives the buffer (dropped after the device), which alone uses it.
    unsafe { no_copy.rebind_bytes_no_copy(other.memory(), None, None) }.unwrap();
    write_rows(&no_copy, rows, |x, y| texel(12, x, y));
    let moved = packed((width, height), |x, y| texel(12, x, y));
    let cpu = level_bytes(&texture, (width, height), 0);
    assert_eq!(cpu, moved, "level 0 on the CPU, the new pages' bytes");
    let (rebound, copies) = rig.levels(&texture, shape);
    assert_eq!(copies, NONE, "a use after the rebind");
    assert_eq!(rebound[0], first, "level 0 through the rebind, as taken");
    let written = packed(level_extent((width, height), 1), |x, y| texel(0x21, x, y));
    assert_eq!(
        rebound[1], written,
        "level 1 through the rebind, as written"
    );
    texture.did_modify_data();
    let (again, copies) = rig.levels(&texture, shape);
    assert_eq!(copies, one_take(width * height), "the use after the call");
    assert_eq!(again[0], moved, "level 0 over the new pages");
    eprintln!(
        "TILED-TEXTURE on {}: detached, every level of {levels} read as taken",
        rig.device.name()
    );
    drop((texture, no_copy));
    drop(rig);
}
