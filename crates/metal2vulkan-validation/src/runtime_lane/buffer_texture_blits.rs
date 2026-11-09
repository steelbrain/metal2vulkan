//! Device tests of blits that name textures over buffers on the device the cfg selects, under the
//! validation layer. The blit cases of a Metal probe on the M3 Max and an M4 run on the runtime
//! over Metal's bytes: a buffer whose byte `i` is `7i + 3`, holding 4x4 `RGBA8Unorm` textures at
//! offset 256, two-slice linear arrays and 16-texel texture buffers, filled with texels that name
//! themselves. Each case runs over an ordinary buffer and a no-copy one, at the probe's 32-byte
//! rows and at 256-byte rows, so each layout takes the aliasing or the shadow form the device gives
//! it. Every byte of every buffer and ordinary texture is checked against a model of Metal's
//! measured answer: a copy moves texel rows as a copy through a temporary does, never the bytes
//! between rows. The overlapping cases, which a model alone would not prove, also match the rows
//! the probe recorded on the M4 and the M3 Max.

use super::degrade_notices::{
    assert_made_once, clear_in_question, forced, shared_in_system_memory,
};
use super::host_memory::{no_copy, not_importable, Mapping};
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device, Messages,
    ScratchCache, WAIT,
};
use crate::gpu::TEST_DEVICE;
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    AppleOnly, BlitCommandEncoder, BlitOption, Buffer, BufferTextureProblem, ClearColor,
    CommandBufferStatus, Deallocator, Degradation, Device, Error, Invalid, IoSurfacePlane,
    LoadAction, Origin, Problem, Region, RenderPassDescriptor, ResourceOptions, RuntimeGap, Size,
    StoreAction, Texture, TextureDescriptor,
};
use std::ops::Range;

/// The buffers' length: a multiple of every host page size.
pub(super) const LEN: usize = 16384;
/// Where the probe's textures start in their buffer.
pub(super) const OFF: u64 = 256;
/// The probe's texture usage.
const USAGE: TextureUsage = TextureUsage::SHADER_READ
    .union(TextureUsage::SHADER_WRITE)
    .union(TextureUsage::PIXEL_FORMAT_VIEW);

/// Whether no linear 2D image of `pf` on `device`, `width` x `height` texels of `texel` bytes,
/// holds a texture's texels alone over an ordinary buffer whose rows lie `row` bytes apart, as the
/// runtime's rule has it (`linear_plan::aliases`): where the rows lie with bytes between them, or
/// where the driver's own image, read through Vulkan apart from the runtime, starts its texels
/// past its binding, lays its rows otherwise than `row` bytes apart, or holds bytes past its last
/// texel in its subresource (lavapipe's does, its height rounded up to 4 rows). The texture is then
/// a shadow image (`shadowed-texture`).
pub(super) fn linear_rows_differ(
    device: &Device,
    pf: PixelFormat,
    [width, height]: [u32; 2],
    texel: u64,
    row: u64,
) -> bool {
    let format = metal2vulkan::metal::format::direct_format(pf).expect("a format a buffer backs");
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::from_raw(format.as_raw()))
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::LINEAR)
        .usage(
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
        )
        .initial_layout(vk::ImageLayout::PREINITIALIZED);
    let subresource = vk::ImageSubresource {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        array_layer: 0,
    };
    // SAFETY: an image of the test's own on the live device, made, read and destroyed here; none
    // of the device's objects is destroyed.
    let layout = unsafe {
        let raw = device.raw();
        let image = raw
            .device
            .create_image(&info, None)
            .expect("a linear image");
        let layout = raw.device.get_image_subresource_layout(image, subresource);
        raw.device.destroy_image(image, None);
        layout
    };
    let row_bytes = u64::from(width) * texel;
    let texels = u64::from(height - 1) * row + row_bytes;
    (height > 1 && row != row_bytes)
        || layout.offset != 0
        || layout.row_pitch != row
        || layout.size > texels
}

/// The probe's buffer pattern.
pub(super) fn pat(i: usize) -> u8 {
    i.wrapping_mul(7).wrapping_add(3) as u8
}

/// Texel (x, y) of set `s`: four bytes naming it.
pub(super) fn texel(x: u64, y: u64, s: u64) -> [u8; 4] {
    [
        0x10 | x as u8,
        0x20 | y as u8,
        (0x40 + 16 * s + 4 * y + x) as u8,
        (0xC0 + 4 * y + x) as u8,
    ]
}

/// Set `s`'s texels of a w x h texture, packed.
pub(super) fn texels(w: u64, h: u64, s: u64) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).flat_map(move |x| texel(x, y, s)))
        .collect()
}

/// The byte rows of a w x h region at (x, y) of 4-byte texels whose rows lie `bpr` apart from
/// `offset`.
pub(super) fn rows(offset: u64, bpr: u64, [x, y]: [u64; 2], [w, h]: [u64; 2]) -> Vec<Range<usize>> {
    (y..y + h)
        .map(|row| {
            let start = (offset + row * bpr + 4 * x) as usize;
            start..start + 4 * w as usize
        })
        .collect()
}

/// The bytes of `rows` of `bytes`, one after another.
fn gather(bytes: &[u8], rows: &[Range<usize>]) -> Vec<u8> {
    rows.iter()
        .flat_map(|r| bytes[r.clone()].to_vec())
        .collect()
}

/// Write `packed` into `rows` of `bytes`.
fn scatter(bytes: &mut [u8], rows: &[Range<usize>], packed: &[u8]) {
    let mut at = 0;
    for r in rows {
        bytes[r.clone()].copy_from_slice(&packed[at..at + r.len()]);
        at += r.len();
    }
}

/// `n` rows of `bpr` bytes of `bytes` from `offset` as hex, `|` between rows (the probe's form).
pub(super) fn hex_rows(bytes: &[u8], offset: u64, bpr: u64, n: u64) -> String {
    (0..n)
        .map(|row| {
            let start = (offset + row * bpr) as usize;
            bytes[start..start + bpr as usize]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("|")
}

/// The 16-bit code a pass's clear to `depth` stores, whose first element `stored` holds: Metal's
/// clear truncates `depth` scaled by 65535 (0.375, 0.625 and 0.875 are 24575.625, 40959.375 and
/// 57343.125 codes), and the device converts it its own way, which can store the code above
/// (`unorm-depth-rounding`): that code where `stored` holds it, else Metal's.
fn depth16_cleared(depth: f64, stored: &[u8]) -> Vec<u8> {
    let metal = (depth * 65535.0) as u16;
    let held = u16::from_le_bytes([stored[0], stored[1]]);
    let code = if held == metal + 1 { held } else { metal };
    code.to_le_bytes().to_vec()
}

/// Where a case's buffers come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Ordinary,
    NoCopy,
}

/// A buffer of the probe's pattern on the device, and its bytes in the model.
pub(super) struct Held {
    pub(super) buffer: Buffer,
    pub(super) model: Vec<u8>,
}

impl Held {
    /// A new buffer of `kind`, or `None` where the device imports no host memory of the kind.
    pub(super) fn new(device: &Device, kind: Kind) -> Option<Held> {
        let model: Vec<u8> = (0..LEN).map(pat).collect();
        let buffer = match kind {
            Kind::Ordinary => buffer_with(device, &model).unwrap(),
            Kind::NoCopy => {
                let mapping = Mapping::anonymous(LEN);
                mapping.write(0, &model);
                let pointer = mapping.pointer();
                let release: Deallocator = Box::new(move |_, _| drop(mapping));
                // SAFETY: the mapping stays mapped until the deallocator, which owns it, runs.
                let made = unsafe {
                    no_copy(
                        device,
                        pointer,
                        LEN,
                        ResourceOptions::default(),
                        Some(release),
                    )
                };
                match made {
                    Ok(buffer) => buffer,
                    Err(error) => {
                        assert!(not_importable(&error), "{error:?}");
                        return None;
                    }
                }
            }
        };
        Some(Held { buffer, model })
    }

    /// Write `bytes` at `offset`, on the device (no command uses the buffer) and in the model.
    pub(super) fn write(&mut self, offset: u64, bytes: &[u8]) {
        let at = offset as usize;
        self.model[at..at + bytes.len()].copy_from_slice(bytes);
        let contents = self.buffer.contents().unwrap();
        // SAFETY: the range lies in the buffer's bytes, which no command uses now.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr().add(at), bytes.len())
        };
    }

    /// Write set `s`'s texels of a w x h texture at `offset`, rows `bpr` apart.
    pub(super) fn fill(&mut self, offset: u64, bpr: u64, [w, h]: [u64; 2], s: u64) {
        for y in 0..h {
            let row: Vec<u8> = (0..w).flat_map(|x| texel(x, y, s)).collect();
            self.write(offset + y * bpr, &row);
        }
    }

    /// The device's bytes.
    pub(super) fn bytes(&self) -> Vec<u8> {
        read_buffer(&self.buffer, 0, LEN as u64).unwrap()
    }

    /// Check the device's bytes against the model's.
    fn check(&self, what: &str) {
        let bytes = self.bytes();
        if let Some(i) = (0..LEN).find(|&i| bytes[i] != self.model[i]) {
            panic!(
                "{what}: byte {i} is {:#04x}, Metal's {:#04x}",
                bytes[i], self.model[i]
            );
        }
    }

    /// A 2D texture over the buffer.
    fn texture(&self, offset: u64, bpr: u64, extent: [u64; 2]) -> Texture {
        self.texture_of(PixelFormat::RGBA8Unorm, USAGE, (offset, bpr), extent)
    }

    /// A 2D texture of `pf` and `usage` over the buffer, at `offset` with rows `bpr` apart.
    pub(super) fn texture_of(
        &self,
        pf: PixelFormat,
        usage: TextureUsage,
        (offset, bpr): (u64, u64),
        [w, h]: [u64; 2],
    ) -> Texture {
        let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
        d.usage = usage;
        d.resource_options = self.buffer.resource_options();
        self.buffer.new_texture(&d, offset, bpr).unwrap()
    }

    /// A linear 4x4 array of two slices over the buffer.
    fn array(&self, offset: u64, bpr: u64) -> Texture {
        self.array_of(USAGE, offset, bpr)
    }

    /// A linear 4x4 `RGBA8Unorm` array of two slices and `usage` over the buffer.
    pub(super) fn array_of(&self, usage: TextureUsage, offset: u64, bpr: u64) -> Texture {
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
        d.texture_type = TextureType::Type2DArray;
        d.array_length = 2;
        d.usage = usage;
        d.resource_options = self.buffer.resource_options();
        self.buffer.new_linear_texture(&d, offset, bpr, 0).unwrap()
    }

    /// A texture buffer of `w` texels over the buffer.
    fn texture_buffer(&self, offset: u64, w: u64) -> Texture {
        self.texture_buffer_of(USAGE, offset, w)
    }

    /// A texture buffer of `w` `RGBA8Unorm` texels and `usage` over the buffer.
    pub(super) fn texture_buffer_of(&self, usage: TextureUsage, offset: u64, w: u64) -> Texture {
        let options = self.buffer.resource_options();
        let d = TextureDescriptor::texture_buffer(PixelFormat::RGBA8Unorm, w, options, usage);
        self.buffer.new_texture(&d, offset, 4 * w).unwrap()
    }
}

/// An ordinary `Shared` w x h texture, and its texels in the model: set `s`'s, or zero.
struct Plain {
    texture: Texture,
    model: Vec<u8>,
    extent: [u64; 2],
}

impl Plain {
    fn new(device: &Device, [w, h]: [u64; 2], set: Option<u64>) -> Plain {
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, w, h, false);
        d.usage = USAGE;
        d.resource_options = ResourceOptions::shared();
        let texture = device.new_texture(&d).unwrap();
        let model = match set {
            Some(s) => {
                let bytes = texels(w, h, s);
                texture
                    .replace_region(Region::new_2d(0, 0, w, h), 0, 0, &bytes, 4 * w, 0)
                    .unwrap();
                bytes
            }
            None => vec![0; (4 * w * h) as usize],
        };
        Plain {
            texture,
            model,
            extent: [w, h],
        }
    }

    /// The model's rows of a region.
    fn rows(&self, at: [u64; 2], size: [u64; 2]) -> Vec<Range<usize>> {
        rows(0, 4 * self.extent[0], at, size)
    }

    fn check(&self, what: &str) {
        let [w, h] = self.extent;
        let mut bytes = vec![0; (4 * w * h) as usize];
        self.texture
            .get_bytes(&mut bytes, 4 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
            .unwrap();
        assert_eq!(bytes, self.model, "{what}: the ordinary texture");
    }
}

/// Encode one blit encoder by `encode` in a command buffer of its own, and wait for it.
fn blit(device: &Device, encode: impl FnOnce(&mut BlitCommandEncoder)) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

fn origin([x, y]: [u64; 2]) -> Origin {
    Origin::new(x, y, 0)
}

fn size([w, h]: [u64; 2]) -> Size {
    Size::new(w, h, 1)
}

/// `copyFromTexture:...toTexture:` of a region of slice 0.
fn copy(
    encoder: &mut BlitCommandEncoder,
    (from, at): (&Texture, [u64; 2]),
    extent: [u64; 2],
    (to, to_at): (&Texture, [u64; 2]),
) {
    encoder
        .copy_from_texture_to_texture(
            from,
            0,
            0,
            origin(at),
            size(extent),
            to,
            0,
            0,
            origin(to_at),
        )
        .unwrap();
}

/// Run `test` on a validated device of the lane, then fail on any validation message.
fn on_device(test: impl FnOnce(&Device)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        test(&device);
    }
    layer_reported_nothing_at_all(&messages);
}

/// The rows a Metal probe recorded (M4 and M3 Max alike) for its overlapping copies at 32-byte
/// rows: a texture onto itself a texel down and right (`b.self_overlap`), a texture into one a row
/// below over the same bytes and back (`b.t2t_overlap`, `b.t2t_overlap_rev`, five rows), and copies
/// between the texture and its own buffer a texel apart (`b.b2t_overlap`, `b.t2b_overlap`).
const PROBE_SELF_OVERLAP: &str =
    "102040c0112041c1122042c2132043c3737a81888f969da4abb2b9c0c7ced5dc|\
102144c4102040c0112041c1122042c2535a61686f767d848b9299a0a7aeb5bc|\
102248c8102144c4112145c5122146c6333a41484f565d646b727980878e959c|\
10234ccc102248c8112249c912224aca131a21282f363d444b525960676e757c";
const PROBE_T2T_OVERLAP: &str = "102040c0112041c1122042c2132043c3737a81888f969da4abb2b9c0c7ced5dc|\
102040c0112041c1122042c2132043c3535a61686f767d848b9299a0a7aeb5bc|\
102144c4112145c5122146c6132147c7333a41484f565d646b727980878e959c|\
102248c8112249c912224aca13224bcb131a21282f363d444b525960676e757c|\
10234ccc11234dcd12234ece13234fcff3fa01080f161d242b323940474e555c";
const PROBE_T2T_OVERLAP_REV: &str =
    "102144c4112145c5122146c6132147c7737a81888f969da4abb2b9c0c7ced5dc|\
102248c8112249c912224aca13224bcb535a61686f767d848b9299a0a7aeb5bc|\
10234ccc11234dcd12234ece13234fcf333a41484f565d646b727980878e959c|\
838a91989fa6adb4bbc2c9d0d7dee5ec131a21282f363d444b525960676e757c|\
838a91989fa6adb4bbc2c9d0d7dee5ecf3fa01080f161d242b323940474e555c";
const PROBE_B2T_OVERLAP: &str = "112041c1122042c2132043c3737a8188737a81888f969da4abb2b9c0c7ced5dc|\
112145c5122146c6132147c7535a6168535a61686f767d848b9299a0a7aeb5bc|\
112249c912224aca13224bcb333a4148333a41484f565d646b727980878e959c|\
11234dcd12234ece13234fcf131a2128131a21282f363d444b525960676e757c";
const PROBE_T2B_OVERLAP: &str = "102040c0102040c0112041c1122042c2132043c38f969da4abb2b9c0c7ced5dc|\
102144c4102144c4112145c5122146c6132147c76f767d848b9299a0a7aeb5bc|\
102248c8102248c8112249c912224aca13224bcb4f565d646b727980878e959c|\
10234ccc10234ccc11234dcd12234ece13234fcf2f363d444b525960676e757c";

/// Check `held`'s device rows against the probe's, where the case ran at the probe's layout.
fn probe_rows(held: &Held, probe: Option<(&str, u64)>, what: &str) {
    if let Some((rows, n)) = probe {
        assert_eq!(
            hex_rows(&held.bytes(), OFF, 32, n),
            rows,
            "{what}: the rows Metal's probe recorded"
        );
    }
}

/// The probe's blits between a 2D texture over a buffer (at `OFF`, rows `bpr` apart) and ordinary
/// textures, other textures over buffers, the texture itself, and buffers.
fn texture_blits(device: &Device, kind: Kind, bpr: u64) {
    let new = || Held::new(device, kind).unwrap();
    let probe = |recorded: &'static str, n: u64| {
        (kind == Kind::Ordinary && bpr == 32).then_some((recorded, n))
    };
    let all: [u64; 2] = [4, 4];
    let here = |at: [u64; 2], extent: [u64; 2]| rows(OFF, bpr, at, extent);
    let name = |case: &str| format!("{case} over a {kind:?} buffer, rows of {bpr} bytes");

    // An ordinary texture into the texture, whole and in part, and by `copy_texture`.
    for (case, at, extent, to_at) in [
        ("o2t", [0, 0], all, [0, 0]),
        ("o2t_part", [1, 1], [2, 2], [2, 1]),
        ("whole", [0, 0], all, [0, 0]),
    ] {
        let mut b = new();
        let t = b.texture(OFF, bpr, all);
        let o = Plain::new(device, all, Some(1));
        blit(device, |e| match case {
            "whole" => e.copy_texture(&o.texture, &t).unwrap(),
            _ => copy(e, (&o.texture, at), extent, (&t, to_at)),
        });
        let moved = gather(&o.model, &o.rows(at, extent));
        scatter(&mut b.model, &here(to_at, extent), &moved);
        b.check(&name(case));
    }
    // The texture into an ordinary texture.
    for (case, at, extent, to_at) in [
        ("t2o", [0, 0], all, [0, 0]),
        ("t2o_part", [1, 0], [2, 3], [0, 1]),
        ("whole_from", [0, 0], all, [0, 0]),
    ] {
        let mut b = new();
        b.fill(OFF, bpr, all, 0);
        let t = b.texture(OFF, bpr, all);
        let mut o = Plain::new(device, all, None);
        blit(device, |e| match case {
            "whole_from" => e.copy_texture(&t, &o.texture).unwrap(),
            _ => copy(e, (&t, at), extent, (&o.texture, to_at)),
        });
        let moved = gather(&b.model, &here(at, extent));
        let to = o.rows(to_at, extent);
        scatter(&mut o.model, &to, &moved);
        o.check(&name(case));
        b.check(&name(case));
    }
    // Into a texture over another buffer, over other bytes of the same buffer, over bytes a row
    // further on (its rows overlap the texture's), and back.
    {
        let (mut b, mut b2) = (new(), new());
        b.fill(OFF, bpr, all, 0);
        let (t, t2) = (b.texture(OFF, bpr, all), b2.texture(512, 64, all));
        blit(device, |e| copy(e, (&t, [0, 0]), all, (&t2, [0, 0])));
        let moved = gather(&b.model, &here([0, 0], all));
        scatter(&mut b2.model, &rows(512, 64, [0, 0], all), &moved);
        b2.check(&name("t2t_other"));
        b.check(&name("t2t_other"));
    }
    for (case, to_offset, back) in [
        ("t2t_same_disjoint", 4096, false),
        ("t2t_overlap", OFF + bpr, false),
        ("t2t_overlap_rev", OFF + bpr, true),
    ] {
        let mut b = new();
        b.fill(OFF, bpr, all, 0);
        let (t, t4) = (b.texture(OFF, bpr, all), b.texture(to_offset, bpr, all));
        let (from, to) = if back { (&t4, &t) } else { (&t, &t4) };
        blit(device, |e| copy(e, (from, [0, 0]), all, (to, [0, 0])));
        let (src, dst) = (rows(to_offset, bpr, [0, 0], all), here([0, 0], all));
        let (src, dst) = if back { (src, dst) } else { (dst, src) };
        let moved = gather(&b.model, &src);
        scatter(&mut b.model, &dst, &moved);
        b.check(&name(case));
        let recorded = match case {
            "t2t_overlap" => probe(PROBE_T2T_OVERLAP, 5),
            "t2t_overlap_rev" => probe(PROBE_T2T_OVERLAP_REV, 5),
            _ => None,
        };
        probe_rows(&b, recorded, &name(case));
    }
    // The texture onto itself: disjoint regions, and overlapping ones.
    for (case, extent, to_at) in [
        ("self_disjoint", [2, 2], [2, 2]),
        ("self_overlap", [3, 3], [1, 1]),
    ] {
        let mut b = new();
        b.fill(OFF, bpr, all, 0);
        let t = b.texture(OFF, bpr, all);
        blit(device, |e| copy(e, (&t, [0, 0]), extent, (&t, to_at)));
        let moved = gather(&b.model, &here([0, 0], extent));
        scatter(&mut b.model, &here(to_at, extent), &moved);
        b.check(&name(case));
        if case == "self_overlap" {
            probe_rows(&b, probe(PROBE_SELF_OVERLAP, 4), &name(case));
        }
    }
    // A buffer into the texture: another buffer, other bytes of its own, and bytes a texel past
    // its own (the rows overlap).
    for (case, other, from, from_bpr) in [
        ("b2t_other", true, 0, 16),
        ("b2t_same", false, 2048, 16),
        ("b2t_overlap", false, OFF + 4, bpr),
    ] {
        let mut b = new();
        let mut b2 = new();
        if case == "b2t_overlap" {
            b.fill(OFF, bpr, all, 0);
        }
        let source = if other { &mut b2 } else { &mut b };
        if from_bpr == 16 {
            source.fill(from, 16, all, 1);
        }
        let t = b.texture(OFF, bpr, all);
        let source = if other { &b2 } else { &b };
        blit(device, |e| {
            e.copy_from_buffer_to_texture(
                &source.buffer,
                from,
                from_bpr,
                4 * from_bpr,
                size(all),
                &t,
                0,
                0,
                origin([0, 0]),
                BlitOption::None,
            )
            .unwrap()
        });
        let moved = gather(&source.model, &rows(from, from_bpr, [0, 0], all));
        scatter(&mut b.model, &here([0, 0], all), &moved);
        b.check(&name(case));
        b2.check(&name(case));
        if case == "b2t_overlap" {
            probe_rows(&b, probe(PROBE_B2T_OVERLAP, 4), &name(case));
        }
    }
    // The texture into a buffer: another buffer, other bytes of its own, and bytes a texel past
    // its own.
    for (case, other, to, to_bpr) in [
        ("t2b_other", true, 0, 16),
        ("t2b_same", false, 2048, 16),
        ("t2b_overlap", false, OFF + 4, bpr),
    ] {
        let mut b = new();
        let mut b2 = new();
        b.fill(OFF, bpr, all, 0);
        let t = b.texture(OFF, bpr, all);
        let destination = if other { &b2 } else { &b };
        blit(device, |e| {
            e.copy_from_texture_to_buffer(
                &t,
                0,
                0,
                origin([0, 0]),
                size(all),
                &destination.buffer,
                to,
                to_bpr,
                4 * to_bpr,
                BlitOption::None,
            )
            .unwrap()
        });
        let moved = gather(&b.model, &here([0, 0], all));
        let destination = if other { &mut b2 } else { &mut b };
        scatter(
            &mut destination.model,
            &rows(to, to_bpr, [0, 0], all),
            &moved,
        );
        b.check(&name(case));
        b2.check(&name(case));
        if case == "t2b_overlap" {
            probe_rows(&b, probe(PROBE_T2B_OVERLAP, 4), &name(case));
        }
    }
    // Fills and copies of the same bytes in one encoder land in encoding order.
    {
        let mut b = new();
        b.fill(OFF, bpr, all, 0);
        let t = b.texture(OFF, bpr, all);
        let mut o = Plain::new(device, all, None);
        let filled = OFF..OFF + 4 * bpr;
        blit(device, |e| {
            e.fill_buffer(&b.buffer, filled.clone(), 0x5A).unwrap();
            copy(e, (&t, [0, 0]), all, (&o.texture, [0, 0]));
        });
        b.model[filled.start as usize..filled.end as usize].fill(0x5A);
        let moved = gather(&b.model, &here([0, 0], all));
        let to = o.rows([0, 0], all);
        scatter(&mut o.model, &to, &moved);
        o.check(&name("fill_then_copy"));
        b.check(&name("fill_then_copy"));
    }
    {
        let mut b = new();
        let t = b.texture(OFF, bpr, all);
        let o = Plain::new(device, all, Some(1));
        let filled = OFF + bpr..OFF + 2 * bpr;
        blit(device, |e| {
            copy(e, (&o.texture, [0, 0]), all, (&t, [0, 0]));
            e.fill_buffer(&b.buffer, filled.clone(), 0x5A).unwrap();
        });
        scatter(&mut b.model, &here([0, 0], all), &o.model);
        b.model[filled.start as usize..filled.end as usize].fill(0x5A);
        b.check(&name("copy_then_fill"));
    }
}

/// Probe 1's blits with a linear array (two 4x4 slices at `OFF`, rows `bpr` apart, slices
/// `bpr` x 4 rounded up to 128 bytes apart), and a copy through a view of its second slice.
fn array_blits(device: &Device, kind: Kind, bpr: u64) {
    let new = || Held::new(device, kind).unwrap();
    let all: [u64; 2] = [4, 4];
    let stride = (4 * bpr).div_ceil(128) * 128;
    let slice = |s: u64| rows(OFF + s * stride, bpr, [0, 0], all);
    let name = |case: &str| format!("{case} over a {kind:?} buffer, rows of {bpr} bytes");
    let filled = || {
        let mut b = new();
        for s in 0..2 {
            b.fill(OFF + s * stride, bpr, all, s);
        }
        b
    };
    {
        let b = filled();
        let a = b.array(OFF, bpr);
        let mut o = Plain::new(device, all, None);
        blit(device, |e| {
            e.copy_from_texture_to_texture(
                &a,
                1,
                0,
                origin([0, 0]),
                size(all),
                &o.texture,
                0,
                0,
                origin([0, 0]),
            )
            .unwrap()
        });
        o.model = gather(&b.model, &slice(1));
        o.check(&name("arr_from_slice"));
    }
    for case in ["arr_to_slice", "arr_view"] {
        let mut b = filled();
        let a = b.array(OFF, bpr);
        let o = Plain::new(device, all, Some(3));
        let view = a
            .new_texture_view_with_levels_slices(
                PixelFormat::RGBA8Unorm,
                TextureType::Type2D,
                0..1,
                1..2,
            )
            .unwrap();
        blit(device, |e| {
            let (to, to_slice) = if case == "arr_view" {
                (&view, 0)
            } else {
                (&a, 1)
            };
            e.copy_from_texture_to_texture(
                &o.texture,
                0,
                0,
                origin([0, 0]),
                size(all),
                to,
                to_slice,
                0,
                origin([0, 0]),
            )
            .unwrap()
        });
        scatter(&mut b.model, &slice(1), &o.model);
        b.check(&name(case));
    }
    {
        let b = filled();
        let mut b2 = new();
        let (a, a2) = (b.array(OFF, bpr), b2.array(1024, 64));
        blit(device, |e| {
            e.copy_texture_slices(&a, 0, 0, &a2, 0, 0, 2, 1).unwrap()
        });
        for s in 0..2 {
            let moved = gather(&b.model, &slice(s));
            scatter(
                &mut b2.model,
                &rows(1024 + s * 256, 64, [0, 0], all),
                &moved,
            );
        }
        b2.check(&name("arr_slices"));
    }
    {
        let mut b = filled();
        let a = b.array(OFF, bpr);
        blit(device, |e| {
            e.copy_from_texture_to_texture(
                &a,
                0,
                0,
                origin([0, 0]),
                size(all),
                &a,
                1,
                0,
                origin([0, 0]),
            )
            .unwrap()
        });
        let moved = gather(&b.model, &slice(0));
        scatter(&mut b.model, &slice(1), &moved);
        b.check(&name("arr_self"));
    }
}

/// Probe 1's blits with a texture buffer of 16 texels at `OFF`.
fn texture_buffer_blits(device: &Device, kind: Kind) {
    let new = || Held::new(device, kind).unwrap();
    let row: [u64; 2] = [16, 1];
    let tb = |offset: u64| rows(offset, 64, [0, 0], row);
    let name = |case: &str| format!("{case} over a {kind:?} buffer");
    let filled = || {
        let mut b = new();
        b.fill(OFF, 64, row, 0);
        b
    };
    {
        let b = filled();
        let t = b.texture_buffer(OFF, 16);
        let mut o = Plain::new(device, row, None);
        blit(device, |e| copy(e, (&t, [0, 0]), row, (&o.texture, [0, 0])));
        o.model = gather(&b.model, &tb(OFF));
        o.check(&name("tb to_o"));
    }
    {
        let mut b = filled();
        let t = b.texture_buffer(OFF, 16);
        let o = Plain::new(device, row, Some(2));
        blit(device, |e| {
            copy(e, (&o.texture, [4, 0]), [8, 1], (&t, [2, 0]))
        });
        let moved = gather(&o.model, &o.rows([4, 0], [8, 1]));
        scatter(&mut b.model, &rows(OFF, 64, [2, 0], [8, 1]), &moved);
        b.check(&name("tb from_o"));
        if kind == Kind::Ordinary {
            let recorded = "102040c0112041c1142064c4152065c5162066c6172067c7182068c8192069c9\
                            1a206aca1b206bcb1a204aca1b204bcb1c204ccc1d204dcd1e204ece1f204fcf";
            assert_eq!(
                hex_rows(&b.bytes(), OFF, 64, 1),
                recorded,
                "tb from_o: probe 1"
            );
        }
    }
    {
        let mut b = new();
        let mut b2 = new();
        b2.fill(0, 64, row, 3);
        let t = b.texture_buffer(OFF, 16);
        blit(device, |e| {
            e.copy_from_buffer_to_texture(
                &b2.buffer,
                0,
                64,
                64,
                size(row),
                &t,
                0,
                0,
                origin([0, 0]),
                BlitOption::None,
            )
            .unwrap()
        });
        let moved = gather(&b2.model, &tb(0));
        scatter(&mut b.model, &tb(OFF), &moved);
        b.check(&name("tb b2t"));
    }
    {
        let b = filled();
        let mut b2 = new();
        let t = b.texture_buffer(OFF, 16);
        blit(device, |e| {
            e.copy_from_texture_to_buffer(
                &t,
                0,
                0,
                origin([0, 0]),
                size(row),
                &b2.buffer,
                128,
                64,
                64,
                BlitOption::None,
            )
            .unwrap()
        });
        let moved = gather(&b.model, &tb(OFF));
        scatter(&mut b2.model, &tb(128), &moved);
        b2.check(&name("tb t2b"));
    }
    for case in ["tb2tb", "tb whole"] {
        let b = filled();
        let mut b2 = new();
        let (t, t2) = (b.texture_buffer(OFF, 16), b2.texture_buffer(512, 16));
        blit(device, |e| match case {
            "tb2tb" => copy(e, (&t, [0, 0]), row, (&t2, [0, 0])),
            _ => e.copy_texture(&t, &t2).unwrap(),
        });
        let moved = gather(&b.model, &tb(OFF));
        scatter(&mut b2.model, &tb(512), &moved);
        b2.check(&name(case));
    }
}

/// Every blit kind with textures over buffers moves Metal's bytes: the probe's cases, over an
/// ordinary and a no-copy buffer, at 32- and 256-byte rows.
#[test]
fn blits_of_textures_over_buffers_move_metals_bytes() {
    on_device(|device| {
        for kind in [Kind::Ordinary, Kind::NoCopy] {
            if Held::new(device, kind).is_none() {
                eprintln!(
                    "{}: no {kind:?} buffer (the device imports none)",
                    device.name()
                );
                continue;
            }
            for bpr in [32, 256] {
                texture_blits(device, kind, bpr);
                array_blits(device, kind, bpr);
            }
            texture_buffer_blits(device, kind);
        }
    });
}

/// The kind [`an_overlapping_texel_copy_is_announced_once`] takes: a copy between texels over
/// bytes of one buffer that overlap, which goes through a scratch buffer.
pub(super) const OVERLAPS: [Degradation; 1] = [Degradation::OverlappingTexelCopy];

/// A copy of a texture over a buffer onto itself a texel down and right (probe case
/// `b.self_overlap`), and of one into a texture a row further on over the same bytes
/// (`b.t2t_overlap`), each in two command buffers, move the rows the probe recorded on the M4 and
/// the M3 Max, and the device announces the overlap once ([`OVERLAPS`]), and where they force it
/// the textures' shadow ([`linear_rows_differ`]) and the `Shared` buffers' system memory.
#[test]
fn an_overlapping_texel_copy_is_announced_once() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        // What the device announced as it was made.
        messages.take_notices();
        let all = [4, 4];
        for _ in 0..2 {
            let mut b = Held::new(&device, Kind::Ordinary).unwrap();
            b.fill(OFF, 32, all, 0);
            let t = b.texture(OFF, 32, all);
            blit(&device, |e| copy(e, (&t, [0, 0]), [3, 3], (&t, [1, 1])));
            probe_rows(&b, Some((PROBE_SELF_OVERLAP, 4)), "b.self_overlap");
            let mut b = Held::new(&device, Kind::Ordinary).unwrap();
            b.fill(OFF, 32, all, 0);
            let (t, below) = (b.texture(OFF, 32, all), b.texture(OFF + 32, 32, all));
            blit(&device, |e| copy(e, (&t, [0, 0]), all, (&below, [0, 0])));
            probe_rows(&b, Some((PROBE_T2T_OVERLAP, 5)), "b.t2t_overlap");
        }
        let first = forced(&[
            (
                Degradation::ShadowedTexture,
                linear_rows_differ(&device, PixelFormat::RGBA8Unorm, [4, 4], 4, 32),
            ),
            (
                Degradation::SharedInSystemMemory,
                shared_in_system_memory(&device),
            ),
        ]);
        assert_made_once(&messages, &OVERLAPS, &first);
    }
    layer_reported_nothing_at_all(&messages);
}

/// The kind [`iosurface_planes_lie_where_metal_puts_them`] takes: a texture of an IOSurface plane
/// of more than one row under 16 bytes a row, whose later rows Metal's GPU reads as
/// memory-dependent bytes.
pub(super) const NARROW_ROWS: [Degradation; 1] = [Degradation::NarrowSurfaceRows];

/// The IOSurface stand-in (`Buffer::new_texture_with_iosurface_plane`) over an ordinary and a
/// no-copy 16-KiB buffer standing for the surface, at plain Metal's answers on the M3 Max and an
/// M4: the consumer's 4x4 `R8Unorm` plane at 4 bytes a row and `RG8Unorm` plane at 8, an `RG8Unorm`
/// plane at 9 (not a multiple of its texel), an `R8Unorm` plane at offset 4, an `RGBA8Unorm`
/// plane at 16 and at 20, an `R8Unorm` plane at 40 (each through [`iosurface_plane`]); the
/// consumer's three-plane '420v' backing ([`iosurface_420v`]); and the refusals of a surface
/// without planes at 4 bytes a row (Metal asserts) and of rows that overlap on the GPU. The
/// device announces `NarrowSurfaceRows` once, for the planes under 16 bytes a row, and where they
/// force it a plane's shadow ([`linear_rows_differ`]), the `Shared` buffers' system memory, and
/// the planes' clear's conversion but where the load operation clears as Metal's
/// (`degrade_notices::clear_in_question`).
#[test]
fn iosurface_planes_lie_where_metal_puts_them() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        // What the device announced as it was made.
        messages.take_notices();
        let planes = [
            (PixelFormat::R8Unorm, 1, 0, 4),
            (PixelFormat::RG8Unorm, 2, 0, 8),
            (PixelFormat::RG8Unorm, 2, 0, 9),
            (PixelFormat::R8Unorm, 1, 4, 4),
            (PixelFormat::RGBA8Unorm, 4, 0, 16),
            (PixelFormat::RGBA8Unorm, 4, 0, 20),
            (PixelFormat::R8Unorm, 1, 0, 40),
        ];
        for kind in [Kind::Ordinary, Kind::NoCopy] {
            let Some(held) = Held::new(&device, kind) else {
                eprintln!("{}: no {kind:?} buffer", device.name());
                continue;
            };
            for (pf, texel, offset, bpr) in planes {
                iosurface_plane(&device, kind, (pf, texel), offset, bpr);
            }
            iosurface_420v(&device, kind);
            let refused = |pf, [w, h]: [u64; 2], plane| {
                let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
                d.resource_options = held.buffer.resource_options();
                held.buffer
                    .new_texture_with_iosurface_plane(&d, plane)
                    .err()
            };
            // As `refused`, with render target usage beside the descriptor's own.
            let refused_target = |pf, [w, h]: [u64; 2], plane| {
                let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
                d.resource_options = held.buffer.resource_options();
                d.usage |= TextureUsage::RENDER_TARGET;
                held.buffer
                    .new_texture_with_iosurface_plane(&d, plane)
                    .err()
            };
            let flat = IoSurfacePlane::new(0, 4, false);
            assert_eq!(
                refused(PixelFormat::R8Unorm, [4, 4], flat),
                Some(Error::Invalid(Invalid::new(
                    "new_texture_with_iosurface_plane",
                    Problem::BufferTexture(BufferTextureProblem::RowPitchAlignment {
                        bytes_per_row: 4,
                        alignment: 16,
                    }),
                )))
            );
            // A plane whose GPU rows share bytes (20 bytes of texels 16 apart): made for a
            // plain descriptor, its shadow's copies reproducing the sharing, and refused where a
            // render pass would store into it, what Metal leaves there being unmeasured.
            let overlapping = IoSurfacePlane::new(0, 20, true);
            assert_eq!(
                refused_target(PixelFormat::RGBA8Unorm, [5, 4], overlapping),
                Some(Error::NotImplemented(RuntimeGap::OverlappingSurfaceRows))
            );
            // A surface made with `kIOSurfaceClientDirection`, SPI, is refused whatever the
            // value: 2, the read-only one the consumer reports, and others.
            for direction in [0, 1, 2, 3] {
                let directed = IoSurfacePlane::new(0, 16, true).with_client_direction(direction);
                assert_eq!(
                    refused(PixelFormat::RGBA8Unorm, [4, 4], directed),
                    Some(Error::AppleOnly(AppleOnly::SurfaceClientDirection)),
                    "client direction {direction}"
                );
            }
        }
        let shadowed = planes
            .iter()
            .any(|&(pf, texel, _, bpr)| linear_rows_differ(&device, pf, [4, 4], texel, bpr));
        // The planes' clear to (0.2, 0.4, 0.6, 0.8), each scaled `float` a hair off a whole code,
        // where the device's clear may store either code nearest it, by its load operation of
        // the whole plane.
        let mut first = forced(&[
            (Degradation::ShadowedTexture, shadowed),
            (
                Degradation::SharedInSystemMemory,
                shared_in_system_memory(&device),
            ),
        ]);
        first.extend(clear_in_question(&device, true));
        assert_made_once(&messages, &NARROW_ROWS, &first);
    }
    layer_reported_nothing_at_all(&messages);
}

/// The kind [`an_iosurface_plane_whose_gpu_rows_share_bytes_is_announced`] takes: a texture of an
/// IOSurface plane whose GPU rows share bytes, which the runtime holds leaving the earlier row's
/// bytes where two rows write them.
pub(super) const COLLIDING_ROWS: [Degradation; 1] = [Degradation::CollidingSurfaceRowWrites];

/// A plane of a planar surface whose GPU rows share bytes is made, and the device announces
/// [`COLLIDING_ROWS`] once for it: a 5x4 `RGBA8Unorm` plane at 20 bytes a row, whose pitch covers
/// its own 20-byte row while Metal's GPU reads and writes its rows at the pitch rounded down to
/// 16, so consecutive rows share 4 bytes, one texel. That is the rule
/// `linear_plan::plane_rows_share_bytes` states, the GPU layout sharing bytes where the CPU's
/// shares none. The runtime holds the plane as a shadow whose copies reproduce the sharing,
/// leaving the earlier row's bytes where two rows meet, which is one of the rows a kernel of
/// Metal's leaves there.
///
/// What Metal leaves of the bytes two rows share has no one rule: the shared bytes take whichever
/// row Metal's own command wrote last, and that order follows the command, the GPU pitch and the
/// row count (the kind's own declaration, over a 423-process measurement). So this holds the
/// announcement, which is what says the runtime chose one of Metal's answers, and not the shared
/// bytes themselves.
///
/// Both controls, which a plane merely named would pass: the same plane with render target usage
/// is refused (`NotImplemented(OverlappingSurfaceRows)`), what Metal's pass leaves of the shared
/// bytes being unmeasured, and announces nothing; and the same plane at 32 bytes a row, whose GPU
/// rows lie 32 apart and share nothing, is made announcing no kind of its own. Each runs over an
/// ordinary buffer standing for the surface and a no-copy one, and the second plane announces
/// nothing, a device announcing each kind once. The CPU reads and writes the plane's own rows.
/// A blit out of the plane into a buffer is served, announcing nothing of its own: it reads each
/// row at its own GPU row, a shared byte once for each row that holds it
/// ([`a_blit_out_of_a_plane_whose_gpu_rows_share_bytes_reads_each_row_at_its_own_pitch`] holds
/// its bytes over more shapes). A blit into it stays refused, a write having to settle which row
/// leaves the shared bytes where Metal settles it no one way, so a kernel is the only GPU writer
/// the plane takes.
#[test]
fn an_iosurface_plane_whose_gpu_rows_share_bytes_is_announced() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        eprintln!("device {}", device.name());
        // What the device announced as it was made.
        messages.take_notices();
        let (w, h) = (5u64, 4u64);
        let row = 4 * w;
        let read = TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE);
        for kind in [Kind::Ordinary, Kind::NoCopy] {
            let Some(mut held) = Held::new(&device, kind) else {
                eprintln!("{}: no {kind:?} buffer", device.name());
                continue;
            };
            // The first control: 32 bytes a row, where the GPU's rows lie 32 apart, a row of 20
            // bytes each, and share nothing.
            let apart = surface_plane(&held, [w, h], 32, read)
                .unwrap_or_else(|e| panic!("{kind:?} at 32 bytes a row: {e}"));
            let first = forced(&[
                (
                    Degradation::ShadowedTexture,
                    linear_rows_differ(
                        &device,
                        PixelFormat::RGBA8Unorm,
                        [w as u32, h as u32],
                        4,
                        32,
                    ),
                ),
                (
                    Degradation::SharedInSystemMemory,
                    shared_in_system_memory(&device),
                ),
            ]);
            assert_made_once(&messages, &[], &first);
            // The second control: with render target usage the plane is refused, so nothing is
            // made and nothing announced.
            assert_eq!(
                surface_plane(&held, [w, h], 20, read.union(TextureUsage::RENDER_TARGET)).err(),
                Some(Error::NotImplemented(RuntimeGap::OverlappingSurfaceRows)),
                "{kind:?}: a render target over rows that share bytes"
            );
            assert_made_once(&messages, &[], &forced(&[]));
            // The plane itself: 20 bytes of texels a row, the GPU's rows 16 apart.
            let shared = surface_plane(&held, [w, h], 20, read)
                .unwrap_or_else(|e| panic!("{kind:?} at 20 bytes a row: {e}"));
            // The CPU reads and writes the plane's own rows, which lie a row apart here.
            let n = (row * h) as usize;
            let region = Region::new_2d(0, 0, w, h);
            let mut back = vec![0u8; n];
            shared.get_bytes(&mut back, row, 0, region, 0, 0).unwrap();
            assert_eq!(back, held.model[..n], "{kind:?}: get_bytes");
            let written: Vec<u8> = (0..n).map(|i| 0x80u8.wrapping_add(i as u8)).collect();
            shared
                .replace_region(region, 0, 0, &written, row, 0)
                .unwrap();
            held.write(0, &written);
            held.check(&format!("{kind:?}: replace_region"));
            // A blit out of the plane reads each buffer row at its own GPU row, 16 apart here.
            let out = buffer_with(&device, &vec![0xee; n]).unwrap();
            blit(&device, |e| {
                e.copy_from_texture_to_buffer(
                    &shared,
                    0,
                    0,
                    origin([0, 0]),
                    size([w, h]),
                    &out,
                    0,
                    row,
                    row * h,
                    BlitOption::None,
                )
                .unwrap_or_else(|e| panic!("{kind:?}: a blit out of rows that share bytes: {e}"))
            });
            assert_eq!(
                read_buffer(&out, 0, n as u64).unwrap(),
                gather(&held.model, &gpu_rows(0, row, row, h)),
                "{kind:?}: a blit's read of rows that share bytes"
            );
            // A blit into the plane writes the GPU's rows as bytes, which no row copy holds.
            let mut gap = None;
            blit(&device, |e| {
                gap = e
                    .copy_from_buffer_to_texture(
                        &out,
                        0,
                        row,
                        row * h,
                        size([w, h]),
                        &shared,
                        0,
                        0,
                        origin([0, 0]),
                        BlitOption::None,
                    )
                    .err();
            });
            assert_eq!(
                gap,
                Some(Error::NotImplemented(RuntimeGap::OverlappingSurfaceRows)),
                "{kind:?}: a blit into rows that share bytes"
            );
            // The plane of the first buffer kind announces [`COLLIDING_ROWS`]; the second's
            // announces nothing, a device announcing each kind once. Both are the set a step
            // announces on its first occurrence, which holds exactly that. No image holds rows
            // that share bytes, so such a plane is a shadow whatever the device.
            let made = forced(&[
                (COLLIDING_ROWS[0], true),
                (Degradation::ShadowedTexture, true),
            ]);
            assert_made_once(&messages, &[], &made);
            drop((apart, shared));
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// The kind [`a_plane_past_its_surface_is_made_and_announced`] takes: a colour texture of a plane
/// of an IOSurface with planes whose texels run past the buffer standing for its surface, those
/// texels in a tail of the texture's own.
pub(super) const SURFACE_TAILS: [Degradation; 1] = [Degradation::SurfaceTexelsTail];

/// A 64x66 `RGBA8Unorm` plane of a planar surface at 256 bytes a row over the 16-KiB ordinary
/// buffer standing for the surface, its last two rows past the surface, is made, which plain
/// Metal makes too and lays out in no one place past the surface (an Apple M3 Max under macOS
/// 27.0.1, `metal2vulkan-conformance/probes/surface_texture_envelope.m`). The device announces
/// [`SURFACE_TAILS`] once for it, with the shadow its tail takes and the `Shared` buffer's system
/// memory where the device forces it. The CPU reads its rows inside the surface as the buffer's
/// bytes and its rows past it as zero, what it writes past the surface reads back, and no byte of
/// the buffer changes. `tests::buffer_tail_device` in the runtime holds every path's bytes over
/// more shapes.
#[test]
fn a_plane_past_its_surface_is_made_and_announced() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        eprintln!("device {}", device.name());
        // What the device announced as it was made.
        messages.take_notices();
        let held = Held::new(&device, Kind::Ordinary).expect("an ordinary buffer");
        let (w, row) = (64u64, 256u64);
        let inside = LEN as u64 / row;
        let h = inside + 2;
        let read = TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE);
        let texture = surface_plane(&held, [w, h], row, read)
            .unwrap_or_else(|e| panic!("a plane past its surface: {e}"));
        let first = forced(&[
            (Degradation::ShadowedTexture, true),
            (
                Degradation::SharedInSystemMemory,
                shared_in_system_memory(&device),
            ),
        ]);
        assert_made_once(&messages, &SURFACE_TAILS, &first);
        let get = |y: u64| {
            let mut bytes = vec![0xEE; row as usize];
            let region = Region::new_2d(0, y, w, 1);
            (texture.get_bytes(&mut bytes, row, 0, region, 0, 0)).unwrap();
            bytes
        };
        let last = (inside - 1) as usize * row as usize;
        assert_eq!(get(inside - 1), held.model[last..last + row as usize]);
        assert_eq!(get(inside), vec![0; row as usize], "the first row past");
        let written: Vec<u8> = (0..row as usize).map(|i| (i * 5 + 1) as u8).collect();
        let past = Region::new_2d(0, inside + 1, w, 1);
        (texture.replace_region(past, 0, 0, &written, row, 0)).unwrap();
        assert_eq!(get(inside + 1), written, "a row written past the surface");
        assert_eq!(held.bytes(), held.model, "the buffer's bytes");
        assert_made_once(&messages, &[], &forced(&[]));
    }
    layer_reported_nothing_at_all(&messages);
}

/// The kind [`a_mipmapped_plane_lies_apart_and_is_announced`] takes: a texture over a plane of an
/// IOSurface with planes whose texels Metal lays out in no one place, its first level's among
/// them, in memory of its own.
pub(super) const SURFACE_APART: [Degradation; 1] = [Degradation::SurfaceTexelsApart];

/// A 64x64 `RGBA8Unorm` 2D texture of two levels over a plane of a planar surface at 256 bytes a
/// row over the 16-KiB ordinary buffer standing for the surface is made, which plain Metal makes
/// too and lays out in no one place, its first level among them (an Apple M3 Max under macOS
/// 27.0.1, `metal2vulkan-conformance/probes/surface_texture_envelope.m`'s `bgra1x2.mips2`). The
/// device announces [`SURFACE_APART`] once for it, with the `Shared` buffer's system memory where
/// the device forces it. The texture lies in memory of its own: its first level reads zero as
/// made where the buffer holds the probe's pattern, what the CPU writes into either level reads
/// back, and no byte of the buffer changes. `tests::buffer_tail_device` in the runtime holds a 3D
/// and a multisample one too.
#[test]
fn a_mipmapped_plane_lies_apart_and_is_announced() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        eprintln!("device {}", device.name());
        // What the device announced as it was made.
        messages.take_notices();
        let held = Held::new(&device, Kind::Ordinary).expect("an ordinary buffer");
        let (w, row) = (64u64, 256u64);
        let h = LEN as u64 / row;
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, w, h, false);
        d.mipmap_level_count = 2;
        d.usage = TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE);
        d.resource_options = held.buffer.resource_options();
        let plane = IoSurfacePlane::new(0, row, true);
        let texture = (held.buffer.new_texture_with_iosurface_plane(&d, plane))
            .unwrap_or_else(|e| panic!("a mipmapped plane: {e}"));
        let first = forced(&[(
            Degradation::SharedInSystemMemory,
            shared_in_system_memory(&device),
        )]);
        assert_made_once(&messages, &SURFACE_APART, &first);
        let get = |level: u64, y: u64| {
            let bytes_per_row = (w >> level) * 4;
            let mut bytes = vec![0xEE; bytes_per_row as usize];
            let region = Region::new_2d(0, y, w >> level, 1);
            (texture.get_bytes(&mut bytes, bytes_per_row, 0, region, level, 0)).unwrap();
            bytes
        };
        assert_eq!(get(0, 5), vec![0; row as usize], "level 0 as made");
        let written: Vec<u8> = (0..row as usize).map(|i| (i * 5 + 1) as u8).collect();
        let region = Region::new_2d(0, 5, w, 1);
        (texture.replace_region(region, 0, 0, &written, row, 0)).unwrap();
        assert_eq!(get(0, 5), written, "level 0 as written");
        let half = &written[..row as usize / 2];
        let region = Region::new_2d(0, 3, w / 2, 1);
        (texture.replace_region(region, 1, 0, half, row / 2, 0)).unwrap();
        assert_eq!(get(1, 3), half, "level 1 as written");
        assert_eq!(held.bytes(), held.model, "the buffer's bytes");
        assert_made_once(&messages, &[], &forced(&[]));
    }
    layer_reported_nothing_at_all(&messages);
}

/// A texture of a `w` x `h` `RGBA8Unorm` plane of a planar surface at offset 0 of `held`'s buffer,
/// `bpr` bytes a row, of `usage` and the buffer's own resource options.
fn surface_plane(
    held: &Held,
    [w, h]: [u64; 2],
    bpr: u64,
    usage: TextureUsage,
) -> Result<Texture, Error> {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, w, h, false);
    d.usage = usage;
    d.resource_options = held.buffer.resource_options();
    held.buffer
        .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, bpr, true))
}

/// A plane whose GPU rows share bytes, copied out into a buffer: every row of the buffer takes
/// the plane's bytes at its own GPU row, so a byte two rows share is read once for each row that
/// holds it, which is what Metal's `copyFromTexture:toBuffer:` reads (an Apple M3 Max under macOS
/// 27.0.1, one shape a process,
/// `metal2vulkan-conformance/probes/overlapping_plane_row_reads.m`). Every byte of the buffer is
/// `7i + 3` of its own address, which names each byte over the plane's span, so a read off Metal's
/// names the byte it took instead of merely differing.
///
/// The copy leaves the plane's own bytes alone and `get_bytes` still reads the plane's rows, not
/// the GPU's. The copy the other way stays refused: a write must settle which of two rows leaves
/// the shared bytes behind and Metal settles it no one way, while a read settles nothing -- it
/// only says where each byte comes from, and both rows that hold a shared byte hold the same byte.
///
/// The shapes share 4 bytes at a GPU pitch of 16 and of 32 and 12 bytes at a pitch of 16, over two
/// to five rows, and include a `bytesPerRow` well past the row whose multiple of 16 still falls
/// below it (the CPU's rows 28 bytes apart, the GPU's 16).
#[test]
fn a_blit_out_of_a_plane_whose_gpu_rows_share_bytes_reads_each_row_at_its_own_pitch() {
    on_device(|device| {
        for kind in [Kind::Ordinary, Kind::NoCopy] {
            let Some(held) = Held::new(device, kind) else {
                eprintln!("{}: no {kind:?} buffer", device.name());
                continue;
            };
            for [w, h, bpr] in [
                [5u64, 4, 20],
                [5, 2, 20],
                [5, 4, 28],
                [9, 4, 36],
                [7, 5, 28],
            ] {
                let what = format!("{kind:?} {w}x{h} at {bpr}");
                let (row, pitch) = (4 * w, (bpr / 16 * 16).max(16));
                let n = (row * h) as usize;
                assert!(pitch < row, "{what}: the GPU rows share bytes");
                let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, w, h, false);
                d.usage = TextureUsage::SHADER_READ;
                d.resource_options = held.buffer.resource_options();
                let t = held
                    .buffer
                    .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(OFF, bpr, true))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let out = buffer_with(device, &vec![0xee; n]).unwrap();
                blit(device, |e| {
                    e.copy_from_texture_to_buffer(
                        &t,
                        0,
                        0,
                        origin([0, 0]),
                        size([w, h]),
                        &out,
                        0,
                        row,
                        row * h,
                        BlitOption::None,
                    )
                    .unwrap()
                });
                assert_eq!(
                    read_buffer(&out, 0, n as u64).unwrap(),
                    gather(&held.model, &gpu_rows(OFF, bpr, row, h)),
                    "{what}: a blit's read of rows that share bytes"
                );
                held.check(&format!("{what}: the plane after the read"));
                let mut back = vec![0u8; n];
                t.get_bytes(&mut back, row, 0, Region::new_2d(0, 0, w, h), 0, 0)
                    .unwrap();
                assert_eq!(
                    back,
                    gather(&held.model, &plane_rows(OFF, bpr, row, h)),
                    "{what}: get_bytes after the read"
                );
                // The write stays refused, the read having settled nothing for it.
                let from = buffer_with(device, &vec![0u8; n]).unwrap();
                let queue = device.new_command_queue().unwrap();
                let command_buffer = queue.command_buffer().unwrap();
                let mut encoder = command_buffer.blit_command_encoder().unwrap();
                let into = encoder.copy_from_buffer_to_texture(
                    &from,
                    0,
                    row,
                    row * h,
                    size([w, h]),
                    &t,
                    0,
                    0,
                    origin([0, 0]),
                    BlitOption::None,
                );
                assert_eq!(
                    into.err(),
                    Some(Error::NotImplemented(RuntimeGap::OverlappingSurfaceRows)),
                    "{what}: a blit into rows that share bytes"
                );
                encoder.end_encoding().unwrap();
            }
        }
    });
}

/// A depth or stencil texture over a plane of a surface with planes, which Metal makes and
/// samples or reads and draws (an M4 under macOS 27.0.1, TTG2's format scout: `Depth16Unorm`,
/// `Depth32Float`, `Depth32Float_Stencil8`, `Stencil8`, `X32_Stencil8`, each for reading and for
/// rendering), is made: a blit reads the plane's bytes, one aspect's unit a texel (its depth,
/// else its stencil), and a pass stores its clear in the plane in Metal's tiles. A blit of the
/// stencil of `Depth32Float_Stencil8` is the runtime's gap naming the format; over a surface
/// without planes it is Metal's stop (`NotLinear`).
#[test]
fn a_planar_surface_s_depth_or_stencil_texture_reads_and_stores_the_plane() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        eprintln!("device {}", device.name());
        let (w, h, bpr) = (4u64, 4u64, 64u64);
        for kind in [Kind::Ordinary, Kind::NoCopy] {
            let Some(mut held) = Held::new(&device, kind) else {
                eprintln!("{}: no {kind:?} buffer", device.name());
                continue;
            };
            for (pf, unit) in [
                (PixelFormat::Depth16Unorm, 2),
                (PixelFormat::Depth32Float, 4),
                (PixelFormat::Depth32FloatStencil8, 4),
                (PixelFormat::Stencil8, 1),
                (PixelFormat::X32Stencil8, 1),
            ] {
                for usage in [TextureUsage::SHADER_READ, TextureUsage::RENDER_TARGET] {
                    let what = format!("{kind:?} {pf:?} {usage:?}");
                    let made = |planar| {
                        let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
                        d.usage = usage;
                        d.resource_options = held.buffer.resource_options();
                        held.buffer.new_texture_with_iosurface_plane(
                            &d,
                            IoSurfacePlane::new(0, bpr, planar),
                        )
                    };
                    let gap = Some(Error::NotImplemented(RuntimeGap::SurfaceDepthStencil {
                        pixel_format: pf,
                    }));
                    let t = made(true).unwrap_or_else(|e| panic!("{what}: {e}"));
                    // A blit reads the plane's rows, one unit a texel.
                    let row = w * unit;
                    let n = (row * h) as usize;
                    let out = buffer_with(&device, &vec![0xee; n]).unwrap();
                    let mut stencil = None;
                    blit(&device, |e| {
                        let to_buffer = |e: &mut BlitCommandEncoder, option| {
                            e.copy_from_texture_to_buffer(
                                &t,
                                0,
                                0,
                                origin([0, 0]),
                                size([w, h]),
                                &out,
                                0,
                                row,
                                row * h,
                                option,
                            )
                        };
                        to_buffer(e, BlitOption::None).unwrap();
                        if pf == PixelFormat::Depth32FloatStencil8 {
                            stencil = to_buffer(e, BlitOption::StencilFromDepthStencil).err();
                        }
                    });
                    let plane = plane_rows(0, bpr, row, h);
                    let read = read_buffer(&out, 0, n as u64).unwrap();
                    assert_eq!(read, gather(&held.model, &plane), "{what}: a blit's read");
                    if pf == PixelFormat::Depth32FloatStencil8 {
                        assert_eq!(stencil, gap, "{what}: the stencil's blit");
                    }
                    // A pass stores its clear in Metal's tiles from the plane's offset rounded
                    // down to 128: here every element alike, end to end from byte 0.
                    if usage == TextureUsage::RENDER_TARGET {
                        let depth = !matches!(pf, PixelFormat::Stencil8 | PixelFormat::X32Stencil8);
                        let stencil =
                            !matches!(pf, PixelFormat::Depth16Unorm | PixelFormat::Depth32Float);
                        let pass = || {
                            let mut pass = RenderPassDescriptor::new();
                            if depth {
                                let a = &mut pass.depth_attachment;
                                a.texture = Some(t.clone());
                                a.load_action = LoadAction::Clear;
                                a.store_action = StoreAction::Store;
                                a.clear_depth = 0.375;
                            }
                            if stencil {
                                let a = &mut pass.stencil_attachment;
                                a.texture = Some(t.clone());
                                a.load_action = LoadAction::Clear;
                                a.store_action = StoreAction::Store;
                                a.clear_stencil = 0x5a;
                            }
                            pass
                        };
                        let queue = device.new_command_queue().unwrap();
                        let command_buffer = queue.command_buffer().unwrap();
                        let encoder = command_buffer
                            .render_command_encoder(&pass())
                            .unwrap_or_else(|e| panic!("{what}: a pass: {e}"));
                        encoder.end_encoding().unwrap();
                        command_buffer.commit().unwrap();
                        command_buffer.wait_until_completed_for(WAIT).unwrap();
                        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                        let value: Vec<u8> = match pf {
                            PixelFormat::Depth16Unorm => depth16_cleared(0.375, &held.bytes()),
                            PixelFormat::Depth32Float | PixelFormat::Depth32FloatStencil8 => {
                                0.375f32.to_le_bytes().to_vec()
                            }
                            _ => vec![0x5a],
                        };
                        for at in (0..n).step_by(unit as usize) {
                            held.model[at..at + unit as usize].copy_from_slice(&value);
                        }
                        held.check(&format!("{what}: a pass's store"));
                    }
                    assert_eq!(
                        made(false).err(),
                        Some(Error::Invalid(Invalid::new(
                            "new_texture_with_iosurface_plane",
                            Problem::BufferTexture(BufferTextureProblem::NotLinear {
                                pixel_format: pf
                            }),
                        ))),
                        "{what}"
                    );
                }
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// Where Metal's pass puts element `(x, y)` of a `w`x`h` depth or stencil texture of `unit`-byte
/// elements over a plane of a surface with planes, from the plane's offset rounded down to 128
/// (an M4 under macOS 27.0.1): tiles of Metal's largest sides for the unit, or a side's power of
/// two where it is smaller, a row of them after another, Morton order inside, x's bit lowest.
fn tiled_at(unit: u64, [w, h]: [u64; 2], x: u64, y: u64) -> u64 {
    let [lw, lh] = match unit {
        1 => [128, 128],
        2 => [128, 64],
        _ => [64, 64],
    };
    let [tw, th] = [w.next_power_of_two().min(lw), h.next_power_of_two().min(lh)];
    let tile = (y / th) * w.div_ceil(tw) + x / tw;
    let [bx, by] = [tw.trailing_zeros(), th.trailing_zeros()];
    let (mut inside, mut k) = (0, 0);
    for i in 0..bx.max(by) {
        if i < bx {
            inside |= (((x % tw) >> i) & 1) << k;
            k += 1;
        }
        if i < by {
            inside |= (((y % th) >> i) & 1) << k;
            k += 1;
        }
    }
    (tile * tw * th + inside) * unit
}

/// A pass on a depth or stencil texture over a plane of a surface with planes writes the bytes
/// of Metal's tiles alone, as Metal's does (no byte outside the elements changed): extents of
/// partial tiles, one tile and several, whose elements leave the bytes between them as they were; a
/// view of the texture in its own format renders into the same bytes; and the stencil of
/// `Depth32Float_Stencil8`, which is not in the plane, loads as 0 whatever a pass stored, through
/// the texture and its `X32_Stencil8` view alike (a draw testing it equal to 0 draws every texel,
/// equal to the stored value none), the depth loaded and stored again unchanged and no byte changed
/// by the view's pass. A shadow of such a texture is zeroed as it is made for
/// `Depth32Float_Stencil8` (`WorkCounts::shadows_zeroed`), and never for
/// `Depth16Unorm` or `Depth32Float`, whose making starts no image on the host
/// (`images_started_on_the_host`).
#[test]
fn a_planar_surface_s_depth_or_stencil_texture_s_passes_write_metal_s_elements_alone() {
    use super::render_state::{library, stencil, Probe, WHITE};
    use metal2vulkan::metal::{CompareFunction, StencilOperation};
    use CompareFunction::{Always, Equal};
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        eprintln!("device {}", device.name());
        let probe = Probe {
            queue: device.new_command_queue().unwrap(),
            library: library(&device),
            device: device.clone(),
        };
        for kind in [Kind::Ordinary, Kind::NoCopy] {
            let Some(mut held) = Held::new(&device, kind) else {
                eprintln!("{}: no {kind:?} buffer", device.name());
                continue;
            };
            for (pf, unit) in [
                (PixelFormat::Depth16Unorm, 2u64),
                (PixelFormat::Depth32Float, 4),
                (PixelFormat::Depth32FloatStencil8, 4),
                (PixelFormat::Stencil8, 1),
                (PixelFormat::X32Stencil8, 1),
            ] {
                for [w, h] in [[19u64, 7u64], [65, 3]] {
                    let what = format!("{kind:?} {pf:?} {w}x{h}");
                    let bpr = (w * unit).next_multiple_of(16);
                    let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
                    d.usage = match pf {
                        PixelFormat::Depth32FloatStencil8 => {
                            TextureUsage::RENDER_TARGET | TextureUsage::PIXEL_FORMAT_VIEW
                        }
                        _ => TextureUsage::RENDER_TARGET,
                    };
                    d.resource_options = held.buffer.resource_options();
                    let before = device.work_counts();
                    let t = held
                        .buffer
                        .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, bpr, true))
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    // A shadow's image is zeroed as it is made only where its copies leave an
                    // aspect out: always the stencil of `Depth32Float_Stencil8`, which reads 0;
                    // never a one-aspect depth format's, nor is any image started on the host
                    // for one. (`Stencil8` and `X32Stencil8` leave out the depth of an image of
                    // both aspects where the device holds them in one: ANV's `X32Stencil8`.)
                    let counted = device.work_counts();
                    let shadowed = counted.shadowed_textures - before.shadowed_textures;
                    let zeroed = counted.shadows_zeroed - before.shadows_zeroed;
                    let hosted =
                        counted.images_started_on_the_host - before.images_started_on_the_host;
                    match pf {
                        PixelFormat::Depth32FloatStencil8 => assert_eq!(
                            zeroed, shadowed,
                            "{what}: the shadows zeroed as made ({shadowed} shadows)"
                        ),
                        PixelFormat::Depth16Unorm | PixelFormat::Depth32Float => {
                            assert_eq!(
                                (zeroed, hosted),
                                (0, 0),
                                "{what}: the shadows zeroed as made, and the images started on \
                                 the host ({shadowed} shadows)"
                            )
                        }
                        _ => {}
                    }
                    let depth = !matches!(pf, PixelFormat::Stencil8 | PixelFormat::X32Stencil8);
                    let has_stencil =
                        !matches!(pf, PixelFormat::Depth16Unorm | PixelFormat::Depth32Float);
                    // A clear of the texture, then of its view in its own format.
                    let view = t.new_texture_view(pf).unwrap();
                    for (target, clear_depth, clear_stencil) in
                        [(&t, 0.625, 0x5au32), (&view, 0.875, 0x33)]
                    {
                        let mut pass = RenderPassDescriptor::new();
                        if depth {
                            let a = &mut pass.depth_attachment;
                            a.texture = Some(target.clone());
                            a.load_action = LoadAction::Clear;
                            a.store_action = StoreAction::Store;
                            a.clear_depth = clear_depth;
                        }
                        if has_stencil {
                            let a = &mut pass.stencil_attachment;
                            a.texture = Some(target.clone());
                            a.load_action = LoadAction::Clear;
                            a.store_action = StoreAction::Store;
                            a.clear_stencil = clear_stencil;
                        }
                        let command_buffer = probe.queue.command_buffer().unwrap();
                        let encoder = command_buffer
                            .render_command_encoder(&pass)
                            .unwrap_or_else(|e| panic!("{what}: a pass: {e}"));
                        encoder.end_encoding().unwrap();
                        command_buffer.commit().unwrap();
                        command_buffer.wait_until_completed_for(WAIT).unwrap();
                        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                        let value: Vec<u8> = match pf {
                            PixelFormat::Depth16Unorm => {
                                let at = tiled_at(unit, [w, h], 0, 0) as usize;
                                depth16_cleared(clear_depth, &held.bytes()[at..])
                            }
                            PixelFormat::Depth32Float | PixelFormat::Depth32FloatStencil8 => {
                                (clear_depth as f32).to_le_bytes().to_vec()
                            }
                            _ => vec![clear_stencil as u8],
                        };
                        for y in 0..h {
                            for x in 0..w {
                                let at = tiled_at(unit, [w, h], x, y) as usize;
                                held.model[at..at + unit as usize].copy_from_slice(&value);
                            }
                        }
                        held.check(&format!("{what}: a pass's store"));
                    }
                    if pf != PixelFormat::Depth32FloatStencil8 {
                        continue;
                    }
                    // The stencil loads as 0, through the texture and through its
                    // `X32_Stencil8` view: drawn where it equals 0, not where it
                    // equals the stored 0x33; the depth loads and stores unchanged, and the view's
                    // store reaches no byte.
                    let x32 = t.new_texture_view(PixelFormat::X32Stencil8).unwrap();
                    for (target, with_depth) in [(&t, true), (&x32, false)] {
                        for (reference, drawn) in [(0u32, true), (0x33, false)] {
                            let what = format!(
                                "{what} through {:?}, stencil equal to {reference:#x}",
                                target.pixel_format()
                            );
                            let mut c =
                                TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, w, h, false);
                            c.usage = TextureUsage::RENDER_TARGET;
                            c.resource_options = ResourceOptions::shared();
                            let color = device.new_texture(&c).unwrap();
                            let mut pass = RenderPassDescriptor::new();
                            let color_target = &mut pass.color_attachments[0];
                            color_target.texture = Some(color.clone());
                            color_target.load_action = LoadAction::Clear;
                            color_target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
                            color_target.store_action = StoreAction::Store;
                            if with_depth {
                                pass.depth_attachment.texture = Some(target.clone());
                                pass.depth_attachment.load_action = LoadAction::Load;
                                pass.depth_attachment.store_action = StoreAction::Store;
                            }
                            pass.stencil_attachment.texture = Some(target.clone());
                            pass.stencil_attachment.load_action = LoadAction::Load;
                            pass.stencil_attachment.store_action = StoreAction::Store;
                            let command_buffer = probe.queue.command_buffer().unwrap();
                            let mut encoder = command_buffer
                                .render_command_encoder(&pass)
                                .unwrap_or_else(|e| panic!("{what}: a pass: {e}"));
                            let face = stencil(Equal, StencilOperation::Keep, 0xff, 0xff);
                            encoder
                                .set_depth_stencil_state(&probe.state(Always, false, Some(face)))
                                .unwrap();
                            encoder.set_stencil_reference_value(reference);
                            let pipe = match with_depth {
                                true => WHITE.depth(pf).stencil(),
                                false => WHITE.stencil(),
                            };
                            let mut kept = Vec::new();
                            probe.draw(&mut encoder, &mut kept, pipe, 0.5, 0.0);
                            encoder.end_encoding().unwrap();
                            command_buffer.commit().unwrap();
                            command_buffer.wait_until_completed_for(WAIT).unwrap();
                            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                            drop(kept);
                            let n = (w * h * 4) as usize;
                            let mut texels = vec![0u8; n];
                            let region = Region::new_2d(0, 0, w, h);
                            color
                                .get_bytes(&mut texels, w * 4, w * h * 4, region, 0, 0)
                                .unwrap();
                            let want = vec![if drawn { 0xff } else { 0 }; n];
                            assert_eq!(texels, want, "{what}");
                            held.check(&format!("{what}: the plane"));
                        }
                    }
                    // A blit of that stencil through the view stays the runtime's gap
                    // (unmeasured), as through the texture.
                    let out = buffer_with(&device, &vec![0xee; (w * h) as usize]).unwrap();
                    let mut refused = None;
                    blit(&device, |e| {
                        refused = e
                            .copy_from_texture_to_buffer(
                                &x32,
                                0,
                                0,
                                origin([0, 0]),
                                size([w, h]),
                                &out,
                                0,
                                w,
                                w * h,
                                BlitOption::None,
                            )
                            .err();
                    });
                    assert_eq!(
                        refused,
                        Some(Error::NotImplemented(RuntimeGap::SurfaceDepthStencil {
                            pixel_format: PixelFormat::X32Stencil8,
                        })),
                        "{what}: a blit of its X32_Stencil8 view"
                    );
                }
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// The rows of a `w`x`h` plane `row` bytes wide, from `offset`, `pitch` bytes apart.
fn plane_rows(offset: u64, pitch: u64, row: u64, h: u64) -> Vec<Range<usize>> {
    (0..h)
        .map(|y| {
            let start = (offset + y * pitch) as usize;
            start..start + row as usize
        })
        .collect()
}

/// Where Metal's GPU finds a plane at `offset`, `bpr` bytes a row: from the offset rounded down
/// to 16, rows `bpr` rounded down to 16 apart, or 16 apart under 16 bytes a row.
fn gpu_rows(offset: u64, bpr: u64, row: u64, h: u64) -> Vec<Range<usize>> {
    plane_rows(offset / 16 * 16, (bpr / 16 * 16).max(16), row, h)
}

/// One plane of [`iosurface_planes_lie_where_metal_puts_them`]: a 4x4 plane of `pf` (`texel`
/// bytes a texel) at `offset`, `bpr` bytes a row, of a planar surface, over a new buffer of
/// `kind`. `get_bytes` reads the plane's rows and `replace_region` writes them; a blit's read
/// takes the GPU's rows, so it reads other bytes than `get_bytes` where the pitch or the offset is
/// not a multiple of 16 (under 16 bytes a row, Metal's GPU reads the first row there and every
/// later one as memory-dependent bytes: the runtime's reads them where its writes put them, the
/// announced near-equivalent); a blit's write and a render pass's clear land in the GPU's rows,
/// and `get_bytes` then sees them only where the two layouts meet, as Metal's did.
fn iosurface_plane(
    device: &Device,
    kind: Kind,
    (pf, texel): (PixelFormat, u64),
    offset: u64,
    bpr: u64,
) {
    let what = format!("{kind:?} {pf:?} at {offset}, {bpr} bytes a row");
    let mut held = Held::new(device, kind).unwrap();
    let (w, h) = (4u64, 4u64);
    let row = w * texel;
    let n = (row * h) as usize;
    let (cpu, gpu) = (
        plane_rows(offset, bpr, row, h),
        gpu_rows(offset, bpr, row, h),
    );
    let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
    d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET;
    d.resource_options = held.buffer.resource_options();
    let plane = IoSurfacePlane::new(offset, bpr, true);
    let t = held
        .buffer
        .new_texture_with_iosurface_plane(&d, plane)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let region = Region::new_2d(0, 0, w, h);
    let get_bytes = || {
        let mut back = vec![0u8; n];
        t.get_bytes(&mut back, row, 0, region, 0, 0).unwrap();
        back
    };
    // The CPU reads and writes the plane's rows.
    assert_eq!(get_bytes(), gather(&held.model, &cpu), "{what}: get_bytes");
    let written: Vec<u8> = (0..n).map(|i| 0x80u8.wrapping_add(i as u8)).collect();
    t.replace_region(region, 0, 0, &written, row, 0).unwrap();
    scatter(&mut held.model, &cpu, &written);
    held.check(&format!("{what}: replace_region"));
    // A blit reads the GPU's rows.
    let out = buffer_with(device, &vec![0xee; n]).unwrap();
    blit(device, |e| {
        e.copy_from_texture_to_buffer(
            &t,
            0,
            0,
            origin([0, 0]),
            size([w, h]),
            &out,
            0,
            row,
            row * h,
            BlitOption::None,
        )
        .unwrap()
    });
    let read = read_buffer(&out, 0, n as u64).unwrap();
    assert_eq!(read, gather(&held.model, &gpu), "{what}: a blit's read");
    // A blit writes the GPU's rows; `get_bytes` then reads the plane's.
    let source: Vec<u8> = (0..n).map(|i| 0x40u8.wrapping_add(i as u8)).collect();
    let from = buffer_with(device, &source).unwrap();
    blit(device, |e| {
        e.copy_from_buffer_to_texture(
            &from,
            0,
            row,
            row * h,
            size([w, h]),
            &t,
            0,
            0,
            origin([0, 0]),
            BlitOption::None,
        )
        .unwrap()
    });
    scatter(&mut held.model, &gpu, &source);
    held.check(&format!("{what}: a blit's write"));
    assert_eq!(
        get_bytes(),
        gather(&held.model, &cpu),
        "{what}: get_bytes after the blit"
    );
    // A render pass's clear stores into the GPU's rows (Metal's 0.2, 0.4, 0.6, 0.8 in 8-bit unorm
    // channels: 0x33, 0x66, 0x99, 0xcc).
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(t.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    attachment.clear_color = ClearColor::new(0.2, 0.4, 0.6, 0.8);
    command_buffer
        .render_command_encoder(&pass)
        .unwrap()
        .end_encoding()
        .unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let cleared: Vec<u8> = (0..n)
        .map(|i| [0x33, 0x66, 0x99, 0xcc][i % texel as usize])
        .collect();
    scatter(&mut held.model, &gpu, &cleared);
    held.check(&format!("{what}: a render pass's clear"));
}

/// A 4:2:2 texture over a surface's plane, as over a CoreVideo `2vuy` or `yuvs` surface (no
/// planes, 16 x 4 texels, 64 bytes a row), in either format, over an ordinary and a no-copy
/// buffer: made, as plain Metal makes it (an M4: plain and validated alike), and
/// its `get_bytes`, `replace_region` and blits into and out of a buffer move the surface's rows
/// unchanged, 2 bytes a texel, the bytes between rows untouched, as Metal's do.
#[test]
fn four_two_two_iosurface_planes_move_the_surface_s_rows() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        for kind in [Kind::Ordinary, Kind::NoCopy] {
            for pf in [PixelFormat::GBGR422, PixelFormat::BGRG422] {
                let what = format!("{kind:?} {pf:?}");
                let Some(mut held) = Held::new(&device, kind) else {
                    eprintln!("{}: no {kind:?} buffer", device.name());
                    continue;
                };
                let (w, h, bpr) = (16u64, 4u64, 64u64);
                let row = 2 * w;
                let n = (row * h) as usize;
                let rows = plane_rows(0, bpr, row, h);
                let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
                d.usage = TextureUsage::SHADER_READ;
                d.resource_options = held.buffer.resource_options();
                let plane = IoSurfacePlane::new(0, bpr, false);
                let t = held
                    .buffer
                    .new_texture_with_iosurface_plane(&d, plane)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(t.pixel_format(), pf, "{what}");
                let region = Region::new_2d(0, 0, w, h);
                let get_bytes = || {
                    let mut back = vec![0u8; n];
                    t.get_bytes(&mut back, row, 0, region, 0, 0).unwrap();
                    back
                };
                assert_eq!(get_bytes(), gather(&held.model, &rows), "{what}: get_bytes");
                let written: Vec<u8> = (0..n).map(|i| 0x80u8.wrapping_add(i as u8)).collect();
                t.replace_region(region, 0, 0, &written, row, 0).unwrap();
                scatter(&mut held.model, &rows, &written);
                held.check(&format!("{what}: replace_region"));
                let out = buffer_with(&device, &vec![0xee; n]).unwrap();
                blit(&device, |e| {
                    e.copy_from_texture_to_buffer(
                        &t,
                        0,
                        0,
                        origin([0, 0]),
                        size([w, h]),
                        &out,
                        0,
                        row,
                        row * h,
                        BlitOption::None,
                    )
                    .unwrap()
                });
                let read = read_buffer(&out, 0, n as u64).unwrap();
                assert_eq!(read, written, "{what}: a blit's read");
                let source: Vec<u8> = (0..n).map(|i| 0x40u8.wrapping_add(i as u8)).collect();
                let from = buffer_with(&device, &source).unwrap();
                blit(&device, |e| {
                    e.copy_from_buffer_to_texture(
                        &from,
                        0,
                        row,
                        row * h,
                        size([w, h]),
                        &t,
                        0,
                        0,
                        origin([0, 0]),
                        BlitOption::None,
                    )
                    .unwrap()
                });
                scatter(&mut held.model, &rows, &source);
                held.check(&format!("{what}: a blit's write"));
                assert_eq!(get_bytes(), source, "{what}: get_bytes after the blit");
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// The consumer's '420v' backing over a new buffer of `kind` ([`iosurface_planes_lie_where_metal_puts_them`]):
/// three planes, a 4x4 `R8Unorm` one at 0 and two 2x2 `RG8Unorm` ones at 256 and 512, each 16 bytes
/// a row (IOSurfaceGetPlaneCount 3: Metal's GPU and CPU find each plane where the
/// surface lays it out). A blit writes each plane, and each lands in its own rows alone.
fn iosurface_420v(device: &Device, kind: Kind) {
    let mut held = Held::new(device, kind).unwrap();
    let planes = [
        (PixelFormat::R8Unorm, 1u64, [4u64, 4u64], 0u64),
        (PixelFormat::RG8Unorm, 2, [2, 2], 256),
        (PixelFormat::RG8Unorm, 2, [2, 2], 512),
    ];
    for (index, (pf, texel, [w, h], offset)) in planes.into_iter().enumerate() {
        let what = format!("{kind:?} '420v' plane {index}");
        let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
        d.resource_options = held.buffer.resource_options();
        let plane = IoSurfacePlane::new(offset, 16, true);
        let t = held
            .buffer
            .new_texture_with_iosurface_plane(&d, plane)
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        let row = w * texel;
        let n = (row * h) as usize;
        let rows = plane_rows(offset, 16, row, h);
        let source: Vec<u8> = (0..n)
            .map(|i| (0x10 * (index as u8 + 1)).wrapping_add(i as u8))
            .collect();
        let from = buffer_with(device, &source).unwrap();
        blit(device, |e| {
            e.copy_from_buffer_to_texture(
                &from,
                0,
                row,
                row * h,
                size([w, h]),
                &t,
                0,
                0,
                origin([0, 0]),
                BlitOption::None,
            )
            .unwrap()
        });
        scatter(&mut held.model, &rows, &source);
        held.check(&format!("{what}: a blit's write"));
        let mut back = vec![0u8; n];
        t.get_bytes(&mut back, row, 0, Region::new_2d(0, 0, w, h), 0, 0)
            .unwrap();
        assert_eq!(back, source, "{what}: get_bytes");
    }
}

/// An IOSurface plane takes and reports its descriptor's resource options whatever the buffer
/// standing for the surface has, as plain Metal's on an M4 does (flat and
/// planar surfaces, plain and under API validation): `Shared`, `Managed` (the guest
/// WindowServer's, over a no-copy buffer of default options) and `Private`, each CPU cache mode
/// and each hazard tracking mode, the default resolving to tracked. Each texture's CPU and GPU
/// read and write the buffer's bytes, a `Private` one's too, as Metal's did the surface's (its
/// validation alone stops `Private`). A `Memoryless` plane, which Metal's validation stops and
/// whose use is unmeasured, is refused.
#[test]
fn iosurface_planes_take_their_descriptors_options() {
    use metal2vulkan_runtime::{CpuCacheMode, HazardTrackingMode, StorageMode};
    on_device(|device| {
        for kind in [Kind::Ordinary, Kind::NoCopy] {
            if Held::new(device, kind).is_none() {
                eprintln!("{}: no {kind:?} buffer", device.name());
                continue;
            }
            for storage in [
                StorageMode::Shared,
                StorageMode::Managed,
                StorageMode::Private,
            ] {
                for cache in [CpuCacheMode::DefaultCache, CpuCacheMode::WriteCombined] {
                    for hazard in [
                        HazardTrackingMode::Default,
                        HazardTrackingMode::Untracked,
                        HazardTrackingMode::Tracked,
                    ] {
                        let options = ResourceOptions::new(storage, cache, hazard);
                        plane_of_options(device, kind, options);
                    }
                }
            }
            let held = Held::new(device, kind).unwrap();
            let memoryless = ResourceOptions::new(
                StorageMode::Memoryless,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Default,
            );
            let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
            d.resource_options = memoryless;
            assert_eq!(
                held.buffer
                    .new_texture_with_iosurface_plane(&d, OPTIONS_PLANE)
                    .err(),
                Some(Error::NotImplemented(RuntimeGap::MemorylessSurfacePlane)),
                "{kind:?}: a memoryless plane, which plain Metal makes"
            );
        }
    });
}

/// The plane [`iosurface_planes_take_their_descriptors_options`] makes: at offset 256, 16 bytes a
/// row, of a planar surface, where Metal's GPU and CPU find the same bytes.
const OPTIONS_PLANE: IoSurfacePlane = IoSurfacePlane::new(OFF, 16, true);

/// One texture of [`iosurface_planes_take_their_descriptors_options`]: a 4x4 `RGBA8Unorm` plane of
/// `options` over a new buffer of `kind`, its options checked, its bytes read and written by the
/// CPU and by blits.
fn plane_of_options(device: &Device, kind: Kind, options: ResourceOptions) {
    use metal2vulkan_runtime::HazardTrackingMode;
    let what = format!("{kind:?} buffer, options {:#x}", options.raw());
    let mut held = Held::new(device, kind).unwrap();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
    d.resource_options = options;
    let t = held
        .buffer
        .new_texture_with_iosurface_plane(&d, OPTIONS_PLANE)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let hazard = match options.hazard_tracking_mode().unwrap() {
        HazardTrackingMode::Default => HazardTrackingMode::Tracked,
        hazard => hazard,
    };
    let (storage, cache) = (
        options.storage_mode().unwrap(),
        options.cpu_cache_mode().unwrap(),
    );
    assert_eq!(
        t.resource_options(),
        ResourceOptions::new(storage, cache, hazard),
        "{what}: resource_options"
    );
    assert_eq!(
        (
            t.storage_mode(),
            t.cpu_cache_mode(),
            t.hazard_tracking_mode()
        ),
        (storage, cache, hazard),
        "{what}: the modes"
    );
    let ([w, h], row) = ([4u64, 4u64], 16u64);
    let n = (row * h) as usize;
    let rows = plane_rows(OFF, 16, row, h);
    let region = Region::new_2d(0, 0, w, h);
    let get_bytes = || {
        let mut back = vec![0u8; n];
        t.get_bytes(&mut back, row, 0, region, 0, 0).unwrap();
        back
    };
    assert_eq!(get_bytes(), gather(&held.model, &rows), "{what}: get_bytes");
    let written: Vec<u8> = (0..n).map(|i| 0x80u8.wrapping_add(i as u8)).collect();
    t.replace_region(region, 0, 0, &written, row, 0).unwrap();
    scatter(&mut held.model, &rows, &written);
    held.check(&format!("{what}: replace_region"));
    let out = buffer_with(device, &vec![0xee; n]).unwrap();
    blit(device, |e| {
        e.copy_from_texture_to_buffer(
            &t,
            0,
            0,
            origin([0, 0]),
            size([w, h]),
            &out,
            0,
            row,
            row * h,
            BlitOption::None,
        )
        .unwrap()
    });
    assert_eq!(
        read_buffer(&out, 0, n as u64).unwrap(),
        written,
        "{what}: a blit's read"
    );
    let source: Vec<u8> = (0..n).map(|i| 0x40u8.wrapping_add(i as u8)).collect();
    let from = buffer_with(device, &source).unwrap();
    blit(device, |e| {
        e.copy_from_buffer_to_texture(
            &from,
            0,
            row,
            row * h,
            size([w, h]),
            &t,
            0,
            0,
            origin([0, 0]),
            BlitOption::None,
        )
        .unwrap()
    });
    scatter(&mut held.model, &rows, &source);
    held.check(&format!("{what}: a blit's write"));
    assert_eq!(get_bytes(), source, "{what}: get_bytes after the blit");
}
