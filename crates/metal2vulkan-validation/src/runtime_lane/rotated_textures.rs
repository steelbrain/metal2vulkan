//! Device tests of rotated textures on the device the cfg selects, under the validation layer:
//! framebuffer-only `BGRA8Unorm` textures over a buffer standing for an IOSurface plane
//! (`Buffer::new_texture_with_iosurface_plane`), of rotations 0 to 4, rendered by the owned fixture
//! `tob_renders` (a rectangle whose fragment reads, at its position, a texture whose texels name
//! themselves, so each rendered texel names the pass position it was drawn at).
//!
//! Every byte of the plane is checked against Metal's rule, measured on an M4 under macOS 27.0.1: a
//! pass's texel `(x, y)` of a `W x H` texture lands at `(H - 1 - y, x)` of an `H x W` plane for
//! rotation 1 and 4, at `(W - 1 - x, H - 1 - y)` for 2 and at `(y, W - 1 - x)` for 3; a load sees
//! the plane turned back at rotations 1 to 3, so texels the pass leaves keep their bytes, and the
//! plane transposed at rotation 4, so they land mirrored left to right, in each 32x32 tile of the
//! frame a draw reaches (our probes and the M3 Max's alike), the runtime turning
//! every tile of a pass that draws (announced once, `RotatedLoadTiles`) and none of one that does
//! not, as Metal; a resolve lands as a store does; the bytes between the plane's rows stay. A
//! rotated texture of a padded 10-bit or extended-range format that such a pass loads keeps its
//! codes with the bits they leave unused as Metal's tile store makes them. CPU access and blits
//! address the plane as the texture's own extent at the plane's rows, turned or not, a quarter
//! turn's rows overlapping where they are longer than the plane's pitch (a write's later row
//! keeping the bytes two rows share), refused where they run past the buffer; a fragment's `read()`
//! sees the stored frame unturned. A view of the texture in another 8-bit RGBA format acts as the
//! texture does; other views are refused. The creators refuse what Metal asserts on, and the
//! shadows a rotated texture is read and rendered through are announced once (`ShadowedTexture`; a
//! resolve's own notices, which differ by device, are not this lane's).
//!
//! `rotated_surface_frame_costs` times frames at the consumer's surface size, 1920x1080, rotated
//! and not, and prints them (`ROTATION-COST`).

use super::{validated_device, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{ColorWriteMask, PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBufferStatus, CommandQueue, ComputePipelineState,
    Degradation, Device, Error, HeapDescriptor, HeapType, Invalid, IoSurfacePlane, Library,
    LoadAction, Origin, PipelineOptions, Problem, Region, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, RuntimeGap, Size, StoreAction, Texture,
    TextureDescriptor, TextureProblem,
};
use std::path::Path;
use std::time::Instant;

const PF: PixelFormat = PixelFormat::BGRA8Unorm;
const TEXEL: u64 = 4;
/// The formats a rotated texture's views are measured in, each viewed as each
/// ([`views_of_rotated_textures_act_as_their_texture_does`]).
const VIEWED: [PixelFormat; 4] = [
    PixelFormat::BGRA8Unorm,
    PixelFormat::BGRA8UnormSRGB,
    PixelFormat::RGBA8Unorm,
    PixelFormat::RGBA8UnormSRGB,
];

/// The kind a pass that draws and loads a rotated texture of rotation 4 takes
/// ([`loads_of_rotation_4_land_mirrored_and_passes_without_draws_keep_every_texel`]).
pub(super) const ROTATED_LOADS: [Degradation; 1] = [Degradation::RotatedLoadTiles];
/// The bytes the plane holds where no texel lies: between rows and past the last.
const PAD: u8 = 0xab;

/// The device, a queue, and the fixture's pipelines: one reading a texture into one attachment
/// of each of [`VIEWED`], one into `PF` at 4 samples, and the kernel writing a texel and four
/// bytes.
struct Lane {
    device: Device,
    queue: CommandQueue,
    reads: Vec<(PixelFormat, RenderPipelineState)>,
    read4: RenderPipelineState,
    two_ways: ComputePipelineState,
    library: Library,
}

impl Lane {
    fn new(device: &Device) -> Lane {
        let ll =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/tob_renders.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let pipeline = |pf: PixelFormat, samples: u32| {
            let mut d = RenderPipelineDescriptor::new(function("tob_rect"));
            d.fragment = Some(function("tob_read"));
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(pf));
            d.raster_sample_count = samples;
            device
                .new_render_pipeline_state(&d, &PipelineOptions::default())
                .unwrap()
        };
        let two_ways = library.new_function_with_name("tob_two_ways").unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            reads: VIEWED.map(|pf| (pf, pipeline(pf, 1))).to_vec(),
            read4: pipeline(PF, 4),
            two_ways: device
                .new_compute_pipeline_state_with_function(&two_ways, &PipelineOptions::default())
                .unwrap(),
            device: device.clone(),
            library,
        }
    }

    /// The fixture's rectangle (`tob_rect`) in a color from fragment buffer 0 (`tob_solid`), into
    /// one attachment of `pf` under the write mask `mask`.
    fn solid(&self, pf: PixelFormat, mask: ColorWriteMask) -> RenderPipelineState {
        let function = |name: &str| {
            StageSource::Function(
                self.library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut d = RenderPipelineDescriptor::new(function("tob_rect"));
        d.fragment = Some(function("tob_solid"));
        let mut color = ColorAttachmentDescriptor::new(pf);
        color.write_mask = mask;
        d.color_attachments[0] = Some(color);
        self.device
            .new_render_pipeline_state(&d, &PipelineOptions::default())
            .unwrap()
    }

    /// One thread of `tob_two_ways`: texel (0, 0) of `texture` written (1, 0, 0, 1), and the
    /// bytes `0x5a` to `0x5d` at `off` of `bytes`.
    fn two_ways(&self, texture: &Texture, bytes: &Buffer, off: u32) {
        let at = self.staged(&off.to_le_bytes());
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut e = command_buffer.compute_command_encoder().unwrap();
        e.set_compute_pipeline_state(&self.two_ways).unwrap();
        e.set_texture(Some(texture), 0).unwrap();
        e.set_buffer(Some(bytes), 0, 0).unwrap();
        e.set_buffer(Some(&at), 0, 1).unwrap();
        e.dispatch_threads(Size::new(1, 1, 1), Size::new(1, 1, 1))
            .unwrap();
        e.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }

    /// A `Shared` buffer holding `bytes`.
    fn staged(&self, bytes: &[u8]) -> Buffer {
        let buffer = self
            .device
            .new_buffer(bytes.len() as u64, ResourceOptions::shared())
            .unwrap();
        write(&buffer, bytes);
        buffer
    }

    /// A `w x h` texture of `PF` whose texel `(x, y)` names itself: bytes `x`, `y`, `0x55`, `0xff`.
    fn naming(&self, w: u64, h: u64) -> Texture {
        self.naming_in(PF, w, h)
    }

    /// [`naming`](Self::naming) in `pf`.
    fn naming_in(&self, pf: PixelFormat, w: u64, h: u64) -> Texture {
        let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
        d.usage = TextureUsage::SHADER_READ;
        let texture = self.device.new_texture(&d).unwrap();
        let mut bytes = Vec::new();
        for y in 0..h {
            for x in 0..w {
                bytes.extend([x as u8, y as u8, 0x55, 0xff]);
            }
        }
        texture
            .replace_region(Region::new_2d(0, 0, w, h), 0, 0, &bytes, w * TEXEL, 0)
            .unwrap();
        texture
    }

    /// One pass into `target`, loaded and stored, that draws nothing.
    fn kept(&self, target: &Texture) {
        let mut descriptor = RenderPassDescriptor::new();
        let attachment = &mut descriptor.color_attachments[0];
        attachment.texture = Some(target.clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let e = command_buffer.render_command_encoder(&descriptor).unwrap();
        e.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }

    /// One pass into `target` (or into a 4-sample texture of `PF` resolved into it, `resolve`),
    /// loaded by `load`, that draws `source`'s texels over the pass texels `[0, px) x [0, py)`.
    fn pass(
        &self,
        target: &Texture,
        load: LoadAction,
        resolve: bool,
        source: &Texture,
        (px, py): (u64, u64),
    ) {
        let (w, h) = (target.width() as f32, target.height() as f32);
        let corners = [
            -1.0,
            1.0 - 2.0 * py as f32 / h,
            -1.0 + 2.0 * px as f32 / w,
            1.0,
        ];
        let corners: Vec<u8> = corners.iter().flat_map(|v: &f32| v.to_le_bytes()).collect();
        let corners = self.staged(&corners);
        let mut descriptor = RenderPassDescriptor::new();
        let attachment = &mut descriptor.color_attachments[0];
        attachment.load_action = load;
        attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        if resolve {
            let mut d = TextureDescriptor::texture_2d(PF, target.width(), target.height(), false);
            d.texture_type = TextureType::Type2DMultisample;
            d.sample_count = 4;
            d.usage = TextureUsage::RENDER_TARGET;
            d.resource_options = ResourceOptions::private();
            attachment.texture = Some(self.device.new_texture(&d).unwrap());
            attachment.resolve_texture = Some(target.clone());
            attachment.store_action = StoreAction::MultisampleResolve;
        } else {
            attachment.texture = Some(target.clone());
            attachment.store_action = StoreAction::Store;
        }
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut e = command_buffer.render_command_encoder(&descriptor).unwrap();
        let read = match resolve {
            true => &self.read4,
            false => {
                let pf = target.pixel_format();
                let (_, read) = self.reads.iter().find(|(of, _)| *of == pf).unwrap();
                read
            }
        };
        e.set_render_pipeline_state(read).unwrap();
        e.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
        e.set_fragment_texture(Some(source), 0).unwrap();
        e.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        e.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }
}

fn write(buffer: &Buffer, bytes: &[u8]) {
    let contents = buffer.contents().unwrap();
    // SAFETY: a `Shared` buffer of at least `bytes.len()` bytes, which no command uses now.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
}

/// The bytes of a `Shared` buffer, which no command uses now.
fn bytes_of(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's `length` bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// Check the validation layer reported nothing.
fn quiet(messages: &Messages) {
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// Where Metal stores the pass texel `(x, y)` of a `w x h` texture of `rotation` (the module
/// docs).
fn stored_at(rotation: u64, (w, h): (u64, u64), (x, y): (u64, u64)) -> (u64, u64) {
    match rotation {
        1 | 4 => (h - 1 - y, x),
        2 => (w - 1 - x, h - 1 - y),
        3 => (y, w - 1 - x),
        _ => (x, y),
    }
}

/// The stored texel whose bytes a pass that draws and loads a texture of `rotation` leaves at
/// the stored texel `(sx, sy)` of a plane `sw` texels wide, where the pass draws nothing: its own
/// at rotations 0 to 3, and at rotation 4, which loads the plane transposed and stores as
/// rotation 1, the one mirrored left to right (the module docs).
fn loaded_from(rotation: u64, sw: u64, (sx, sy): (u64, u64)) -> (u64, u64) {
    match rotation {
        4 => (sw - 1 - sx, sy),
        _ => (sx, sy),
    }
}

/// The plane's extent for a `w x h` texture of `rotation`: the frame's height wide for a
/// quarter turn.
fn plane_extent(rotation: u64, (w, h): (u64, u64)) -> (u64, u64) {
    match rotation {
        1 | 3 | 4 => (h, w),
        _ => (w, h),
    }
}

/// The least `bytesPerRow` Metal takes for a plane of a `w x h` texture of `rotation` whose
/// texels are `texel` bytes: it covers the stored frame's row, which the plane holds, **and** the
/// texture's own, which Metal holds the pitch to whatever the rotation (its descriptor validation
/// asserts below it, plain too). Under a quarter turn the two differ, and the texture's is the
/// longer where it is wider than tall.
fn tight_row(rotation: u64, (w, h): (u64, u64), texel: u64) -> u64 {
    let (sw, _) = plane_extent(rotation, (w, h));
    sw.max(w) * texel
}

/// A plane of a `w x h` texture of `rotation`: its rows `bpr` bytes apart, every byte `PAD`.
fn plane(lane: &Lane, rotation: u64, (w, h): (u64, u64), bpr: u64) -> Buffer {
    let (_, rows) = plane_extent(rotation, (w, h));
    lane.staged(&vec![PAD; (bpr * rows) as usize])
}

/// A framebuffer-only `PF` texture of `rotation`, `w x h`, over `plane`, rows `bpr` bytes apart.
fn rotated(plane: &Buffer, rotation: u64, (w, h): (u64, u64), bpr: u64) -> Result<Texture, Error> {
    rotated_in(plane, (PF, TextureUsage::empty()), rotation, (w, h), bpr)
}

/// [`rotated`] in `pf`, its usage taking `more` too.
fn rotated_in(
    plane: &Buffer,
    (pf, more): (PixelFormat, TextureUsage),
    rotation: u64,
    (w, h): (u64, u64),
    bpr: u64,
) -> Result<Texture, Error> {
    let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
    d.usage =
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | more;
    d.resource_options = plane.resource_options();
    d.framebuffer_only = true;
    d.rotation = rotation;
    d.is_drawable = rotation % 2 == 1;
    let surface = IoSurfacePlane::new(0, bpr, false);
    plane.new_texture_with_iosurface_plane(&d, surface)
}

/// The plane's bytes Metal leaves after a pass that drew the self-naming texels over
/// `[0, px) x [0, py)` of a `w x h` texture of `rotation`, its other stored texels `kept` (each
/// stored texel's four bytes from its position) and the bytes between rows as `before`.
fn expected(
    rotation: u64,
    (w, h): (u64, u64),
    bpr: u64,
    (px, py): (u64, u64),
    before: &[u8],
    kept: impl Fn(u64, u64) -> [u8; 4],
) -> Vec<u8> {
    let mut plane = before.to_vec();
    let (sw, sh) = plane_extent(rotation, (w, h));
    for sy in 0..sh {
        for sx in 0..sw {
            let at = (sy * bpr + sx * TEXEL) as usize;
            plane[at..at + 4].copy_from_slice(&kept(sx, sy));
        }
    }
    for y in 0..py {
        for x in 0..px {
            let (sx, sy) = stored_at(rotation, (w, h), (x, y));
            let at = (sy * bpr + sx * TEXEL) as usize;
            plane[at..at + 4].copy_from_slice(&[x as u8, y as u8, 0x55, 0xff]);
        }
    }
    plane
}

/// The first byte where `got` differs from `want`, as the stored texel and its byte.
fn first_difference(got: &[u8], want: &[u8], bpr: u64) -> Option<String> {
    first_past(got, want, bpr, 0)
}

/// The first byte where `got` differs from `want` by more than `slack`, as
/// [`first_difference`] gives it.
fn first_past(got: &[u8], want: &[u8], bpr: u64, slack: u8) -> Option<String> {
    let k = got
        .iter()
        .zip(want)
        .position(|(a, b)| a.abs_diff(*b) > slack)? as u64;
    Some(format!(
        "byte {k} (row {}, texel {}, byte {}): {:#04x} where Metal's is {:#04x}",
        k / bpr,
        (k % bpr) / TEXEL,
        k % TEXEL,
        got[k as usize],
        want[k as usize]
    ))
}

/// Every rotation, a texture wider than tall and one taller than wide, each over a plane at its
/// tight rows and at rows wider than the texture's: a pass that clears and draws everything,
/// one that loads and draws its top-left corner, and a resolve, each plane byte where Metal
/// puts it, but at rotation 4 the loaded texels of the 32x32 tiles past the first, which the
/// corner's draw does not reach: Metal keeps them in place, the runtime mirrors them as the
/// first tile's (`RotatedLoadTiles`).
#[test]
fn passes_store_load_and_resolve_rotated_textures_as_metal_turns_them() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let lane = Lane::new(&device);
    let mut twins = 0;
    for (w, h) in [(64u64, 16u64), (16, 64)] {
        let source = lane.naming(w, h);
        for rotation in 0..=4u64 {
            let (sw, _) = plane_extent(rotation, (w, h));
            for bpr in [
                tight_row(rotation, (w, h), TEXEL),
                (w.max(h) * TEXEL).next_multiple_of(16) + 16,
            ] {
                let what = format!("rotation {rotation}, {w}x{h}, rows {bpr} bytes apart");
                let plane = plane(&lane, rotation, (w, h), bpr);
                let texture = rotated(&plane, rotation, (w, h), bpr).unwrap();
                assert_eq!(texture.rotation(), rotation, "{what}");
                assert!(texture.framebuffer_only(), "{what}");
                assert_eq!(texture.is_drawable(), rotation % 2 == 1, "{what}");
                twins += usize::from(rotation != 0);
                // Cleared and drawn over everything.
                let before = bytes_of(&plane);
                let counts = device.work_counts();
                lane.pass(&texture, LoadAction::Clear, false, &source, (w, h));
                // A rotated texture's rows go back out of its staging buffer as a copy of rows
                // (`row_copies.rs`): rows that lie one after another as one region, which takes no
                // dispatch; rows that lie apart as one dispatch where the device runs it faster.
                let (dispatched, copies) = super::row_copies::copies_since(&device, &counts);
                if rotation != 0 {
                    assert!(copies > 0, "{what}: no copy of rows");
                    let regions = device.work_counts().row_copy_regions - counts.row_copy_regions;
                    match bpr == sw * TEXEL {
                        true => assert_eq!(
                            (dispatched, regions),
                            (0, copies),
                            "{what}: packed rows, one region a copy and no dispatch"
                        ),
                        false => assert_eq!(
                            dispatched,
                            copies * super::row_copies::dispatches(&device),
                            "{what}: dispatched copies of rows"
                        ),
                    }
                }
                let want = expected(rotation, (w, h), bpr, (w, h), &before, |_, _| [0; 4]);
                if let Some(at) = first_difference(&bytes_of(&plane), &want, bpr) {
                    panic!("{what}, a clear and a full draw: {at}");
                }
                // The plane's texels name their stored position; a load, and a draw over the
                // pass's top-left 8x4 texels: the rest keep their bytes.
                let mut named = bytes_of(&plane);
                let (sw, sh) = plane_extent(rotation, (w, h));
                for sy in 0..sh {
                    for sx in 0..sw {
                        let at = (sy * bpr + sx * TEXEL) as usize;
                        named[at..at + 4].copy_from_slice(&[sx as u8, sy as u8, 0x33, 0x77]);
                    }
                }
                write(&plane, &named);
                lane.pass(&texture, LoadAction::Load, false, &source, (8, 4));
                let want = expected(rotation, (w, h), bpr, (8, 4), &named, |sx, sy| {
                    let (fx, fy) = loaded_from(rotation, sw, (sx, sy));
                    [fx as u8, fy as u8, 0x33, 0x77]
                });
                if let Some(at) = first_difference(&bytes_of(&plane), &want, bpr) {
                    panic!("{what}, a load and a corner's draw: {at}");
                }
                // A 4-sample pass resolved into the texture, its whole frame drawn.
                lane.pass(&texture, LoadAction::Clear, true, &source, (w, h));
                let want = expected(rotation, (w, h), bpr, (w, h), &named, |_, _| [0; 4]);
                if let Some(at) = first_difference(&bytes_of(&plane), &want, bpr) {
                    panic!("{what}, a resolve: {at}");
                }
                eprintln!("ROTATED-PASS {what}: stores, loads and resolves as Metal's");
            }
        }
    }
    assert!(twins > 0);
    // The shadows a rotated texture is read and rendered through, announced once; a resolve's
    // notices (a multisample texture not stored, a resolve in a shader) differ by device.
    let counted = super::degrade_notices::counts(&messages.take_notices());
    assert_eq!(
        counted.get(&Degradation::ShadowedTexture),
        Some(&1),
        "{counted:?}"
    );
    quiet(&messages);
}

/// The consumer's frame: a 32x32 texture, one tile, of every rotation, its plane's texels naming
/// their stored position. A pass that loads it and draws nothing leaves every byte in place; one
/// that loads it and draws its top-left corner leaves, at rotation 4, every other texel mirrored
/// left to right (the plane's texel (0, 0) holding (31, 0)'s bytes, and (20, 30) holding (11,
/// 30)'s), and every texel in place at the others: Metal's answer, the tile reached. The pass that
/// draws at rotation 4 announces `RotatedLoadTiles` once; the passes before it, none.
#[test]
fn loads_of_rotation_4_land_mirrored_and_passes_without_draws_keep_every_texel() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let lane = Lane::new(&device);
    let (w, h) = (32u64, 32u64);
    let source = lane.naming(w, h);
    let bpr = w * TEXEL + 64;
    let announced = |messages: &Messages| {
        super::degrade_notices::counts(&messages.take_notices())
            .get(&ROTATED_LOADS[0])
            .copied()
    };
    for drawn in [false, true] {
        for rotation in 0..=4u64 {
            let what = format!("rotation {rotation}, {w}x{h}, drawn {drawn}");
            let plane = plane(&lane, rotation, (w, h), bpr);
            let texture = rotated(&plane, rotation, (w, h), bpr).unwrap();
            let mut named = bytes_of(&plane);
            let (sw, sh) = plane_extent(rotation, (w, h));
            for sy in 0..sh {
                for sx in 0..sw {
                    let at = (sy * bpr + sx * TEXEL) as usize;
                    named[at..at + 4].copy_from_slice(&[sx as u8, sy as u8, 0x33, 0x77]);
                }
            }
            write(&plane, &named);
            let corner = match drawn {
                true => {
                    lane.pass(&texture, LoadAction::Load, false, &source, (8, 4));
                    (8, 4)
                }
                false => {
                    lane.kept(&texture);
                    (0, 0)
                }
            };
            let want = expected(rotation, (w, h), bpr, corner, &named, |sx, sy| {
                let (fx, fy) = match drawn {
                    true => loaded_from(rotation, sw, (sx, sy)),
                    false => (sx, sy),
                };
                [fx as u8, fy as u8, 0x33, 0x77]
            });
            let got = bytes_of(&plane);
            if let Some(at) = first_difference(&got, &want, bpr) {
                panic!("{what}: {at}");
            }
            if drawn && rotation == 4 {
                let texel = |(x, y): (u64, u64)| {
                    let at = (y * bpr + x * TEXEL) as usize;
                    got[at..at + 4].to_vec()
                };
                assert_eq!(texel((0, 0)), [31, 0, 0x33, 0x77], "{what}");
                assert_eq!(texel((20, 30)), [11, 30, 0x33, 0x77], "{what}");
            }
            let notices = announced(&messages);
            assert_eq!(
                notices,
                (drawn && rotation == 4).then_some(1),
                "{what}: rotated-load-tiles"
            );
            eprintln!("ROTATED-LOAD {what}: as Metal's");
        }
    }
    quiet(&messages);
}

/// A texel of the rotated attachments [`unused_bits_of_rotated_textures_are_metals`] loads: the
/// format, its bytes, the write mask its draw takes, its bytes at the stored texel `(sx, sy)`
/// (codes naming the texel and channel, the bits they leave unused not Metal's), Metal's store
/// of those bytes, and its drawn texel's bytes, where the mask writes one.
struct UnusedBitsCase {
    pf: PixelFormat,
    texel: u64,
    mask: ColorWriteMask,
    named: fn((u64, u64)) -> Vec<u8>,
    stored: fn(&[u8]) -> Vec<u8>,
    drawn: Option<Vec<u8>>,
}

/// The 10-bit code naming channel `c` of the stored texel `(sx, sy)`.
fn code_of((sx, sy): (u64, u64), c: u64) -> u64 {
    (sx * 13 + sy * 97 + c * 511) % 1024
}

/// The padded words of `channels` channels at the stored texel `(sx, sy)`: each code naming its
/// texel and channel, the low six bits set.
fn padded_named((sx, sy): (u64, u64), channels: u64) -> Vec<u8> {
    (0..channels)
        .flat_map(|c| ((code_of((sx, sy), c) << 6 | 0x3f) as u16).to_le_bytes())
        .collect()
}

/// Metal's store of padded words: each word's low six bits zeroed.
fn padded_stored(bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks(2)
        .flat_map(|w| (u16::from_le_bytes([w[0], w[1]]) & !0x3f).to_le_bytes())
        .collect()
}

/// The cases: `R10Unorm_X6` and `RG10Unorm_X12` drawn in codes 341 and 682 (the color
/// `(1/3, 2/3, 0, 1)`, exactly), and `BGR10_XR` under an empty write mask, its bits 30 and 31
/// zero or one by texel (Metal's store sets them).
fn unused_bits_cases() -> [UnusedBitsCase; 3] {
    [
        UnusedBitsCase {
            pf: PixelFormat::R10UnormX6,
            texel: 2,
            mask: ColorWriteMask::ALL,
            named: |at| padded_named(at, 1),
            stored: padded_stored,
            drawn: Some((341u16 << 6).to_le_bytes().to_vec()),
        },
        UnusedBitsCase {
            pf: PixelFormat::RG10UnormX12,
            texel: 4,
            mask: ColorWriteMask::ALL,
            named: |at| padded_named(at, 2),
            stored: padded_stored,
            drawn: Some(
                [341u16 << 6, 682 << 6]
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect(),
            ),
        },
        UnusedBitsCase {
            pf: PixelFormat::BGR10XR,
            texel: 4,
            mask: ColorWriteMask::NONE,
            named: |(sx, sy)| {
                let codes = (0..3).fold(0, |word, c| word << 10 | code_of((sx, sy), c));
                let top = (sx + 2 * sy) % 4;
                ((top << 30 | codes) as u32).to_le_bytes().to_vec()
            },
            stored: |bytes| {
                let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                (word | 0xc000_0000).to_le_bytes().to_vec()
            },
            drawn: None,
        },
    ]
}

/// Rotated textures of `R10Unorm_X6`, `RG10Unorm_X12` and `BGR10_XR` ([`unused_bits_cases`]), of
/// every rotation, a frame of one 32x32 tile and one of two each way, over planes whose texels
/// name their stored position: a pass that loads the texture and draws its top-left 8x4 texels
/// stores the drawn texels where the mask writes them and leaves every other texel's codes where
/// a load and a store put them (at rotation 4 the mirror [`loaded_from`] gives), the bits they
/// leave unused as Metal's store makes them (a padded word's low six bits zeroed, `BGR10_XR`'s bits
/// 30 and 31 set), the bytes between rows kept. That is Metal's store in each tile the draw
/// reaches (on an M4 and the consumer's runs of them on an M3 Max; the
/// turns): the runtime rewrites the twin of the pass's frame after its copy in, so no pass
/// announces `extended-range-bits-kept`. In a frame of two tiles the one the draw does not reach is
/// rewritten as well, where Metal keeps its texels' bytes (`unreached-tiles-rewritten`, announced
/// by each such pass), but where capture replay found the draw's reach (a texture that is not a
/// rotated frame, once its capture form is made): that pass keeps it, Metal's answer, and announces
/// nothing.
#[test]
fn unused_bits_of_rotated_textures_are_metals() {
    use super::extended_range_blends::{EXTENDED_RANGE_BITS, UNREACHED_TILES};
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    // Each pass's announcement says which reach it took (below).
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.announce_each = UNREACHED_TILES.into_iter().collect();
    })
    .unwrap();
    let mut announced = 0;
    let lane = Lane::new(&device);
    let color: Vec<u8> = [1.0f32 / 3.0, 2.0 / 3.0, 0.0, 1.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let color = lane.staged(&color);
    for case in unused_bits_cases() {
        let pf = case.pf;
        let solid = lane.solid(pf, case.mask);
        for (w, h) in [(32u64, 32u64), (64, 32), (32, 64)] {
            for rotation in 0..=4u64 {
                let what = format!("{pf:?}, rotation {rotation}, {w}x{h}");
                let (sw, sh) = plane_extent(rotation, (w, h));
                let bpr = tight_row(rotation, (w, h), case.texel).next_multiple_of(64) + 64;
                let plane = plane(&lane, rotation, (w, h), bpr);
                let texture =
                    rotated_in(&plane, (pf, TextureUsage::empty()), rotation, (w, h), bpr)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                let mut before = bytes_of(&plane);
                let mut want = before.clone();
                let put = |bytes: &mut Vec<u8>, (sx, sy): (u64, u64), texel: &[u8]| {
                    let at = (sy * bpr + sx * case.texel) as usize;
                    bytes[at..at + texel.len()].copy_from_slice(texel);
                };
                for sy in 0..sh {
                    for sx in 0..sw {
                        put(&mut before, (sx, sy), &(case.named)((sx, sy)));
                        let from = loaded_from(rotation, sw, (sx, sy));
                        put(&mut want, (sx, sy), &(case.stored)(&(case.named)(from)));
                    }
                }
                if let Some(drawn) = &case.drawn {
                    for y in 0..4 {
                        for x in 0..8 {
                            put(&mut want, stored_at(rotation, (w, h), (x, y)), drawn);
                        }
                    }
                }
                write(&plane, &before);
                let corners: Vec<u8> = [-1.0, 1.0 - 8.0 / h as f32, -1.0 + 16.0 / w as f32, 1.0]
                    .iter()
                    .flat_map(|v: &f32| v.to_le_bytes())
                    .collect();
                let corners = lane.staged(&corners);
                let mut descriptor = RenderPassDescriptor::new();
                let attachment = &mut descriptor.color_attachments[0];
                attachment.texture = Some(texture.clone());
                attachment.load_action = LoadAction::Load;
                attachment.store_action = StoreAction::Store;
                let command_buffer = lane.queue.command_buffer().unwrap();
                let mut e = command_buffer.render_command_encoder(&descriptor).unwrap();
                e.set_render_pipeline_state(&solid).unwrap();
                e.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
                e.set_fragment_buffer(Some(&color), 0, 0).unwrap();
                e.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
                e.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                // A pass whose draws' reach capture replay found (a texture that is not a
                // rotated frame, once its capture form is made: `render/capture.rs`) rewrites the
                // tile the draw reaches alone and keeps every other texel's bytes, Metal's answer;
                // one that announces rewrites every tile.
                let counted = super::degrade_notices::counts(&messages.take_notices());
                assert_eq!(
                    counted.get(&EXTENDED_RANGE_BITS[0]),
                    None,
                    "{what}: {counted:?}"
                );
                let reached = |(x, y): (u64, u64)| x < 32 && y < 32;
                if counted.contains_key(&UNREACHED_TILES[0]) {
                    announced += 1;
                } else {
                    let mut exact = before.clone();
                    for y in 0..h {
                        for x in (0..w).filter(|&x| reached((x, y))) {
                            let (sx, sy) = stored_at(rotation, (w, h), (x, y));
                            let at = (sy * bpr + sx * case.texel) as usize;
                            let texel = at..at + case.texel as usize;
                            exact[texel.clone()].copy_from_slice(&want[texel]);
                        }
                    }
                    want = exact;
                }
                if let Some(at) = first_difference(&bytes_of(&plane), &want, bpr) {
                    panic!("{what}: {at} (texels counted as 4 bytes)");
                }
                eprintln!("ROTATED-UNUSED-BITS {what}: Metal's bytes");
            }
        }
    }
    assert!(announced > 0, "a frame of two tiles announces");
    quiet(&messages);
}

/// The plane with every 4-byte word naming itself: bytes `i`, `i >> 8`, `0xc0 | i >> 16`, `0xff`
/// for word `i`.
fn named_words(length: u64) -> Vec<u8> {
    (0..length / 4)
        .flat_map(|i| {
            [
                i as u8,
                (i >> 8) as u8,
                0xc0 | ((i >> 16) as u8 & 0x3f),
                0xff,
            ]
        })
        .collect()
}

/// The texels `w x h` self-naming texels write: texel `(x, y)` bytes `x`, `y`, `0x55`, `0xff`.
fn naming_texels(w: u64, h: u64) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).flat_map(move |x| [x as u8, y as u8, 0x55, 0xff]))
        .collect()
}

/// `plane` after `texels` (`w x h`, tight) are written row by row in order at rows `bpr` apart,
/// as plain Metal's `replaceRegion` and blits in write a rotated texture's rows.
fn rows_written(plane: &[u8], texels: &[u8], (w, h): (u64, u64), bpr: u64) -> Vec<u8> {
    let mut plane = plane.to_vec();
    let row = (w * TEXEL) as usize;
    for y in 0..h as usize {
        let at = y * bpr as usize;
        plane[at..at + row].copy_from_slice(&texels[y * row..(y + 1) * row]);
    }
    plane
}

/// `w x h` texels read at rows `bpr` apart of `plane`, as plain Metal's `getBytes` and blits out
/// read a rotated texture.
fn rows_read(plane: &[u8], (w, h): (u64, u64), bpr: u64) -> Vec<u8> {
    let row = (w * TEXEL) as usize;
    (0..h as usize)
        .flat_map(|y| plane[y * bpr as usize..y * bpr as usize + row].to_vec())
        .collect()
}

/// The uses other than a render pass's, as plain Metal addresses them: CPU access and blits, in and
/// out, address the plane as the texture's own extent at the plane's rows, turned or not, a quarter
/// turn's rows overlapping where they are longer than the plane's pitch and a write's later row
/// keeping the bytes two rows share; refused (`NotImplemented(RotatedTexture)`) where those rows
/// run past the buffer, and made over a buffer that holds them. A fragment's `read()` sees the
/// stored frame unturned (positions past it, which Metal reads as 0, are printed, not checked), and
/// a kernel's write of a texel reaches no byte while its write of the plane's bytes lands.
#[test]
fn other_uses_address_the_plane_as_metal_does() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let lane = Lane::new(&device);
    let gap = Error::NotImplemented(RuntimeGap::RotatedTexture);
    // Rotation, extent, the plane's pitch (the least Metal takes, `tight_row`, some 16 bytes
    // more), and bytes past the stored frame's last row.
    let tight = |rotation, extent| tight_row(rotation, extent, TEXEL);
    let cases = [
        (2u64, (64u64, 16u64), tight(2, (64, 16)) + 16, 0u64),
        (1, (64, 16), tight(1, (64, 16)), 0),
        (3, (64, 16), tight(3, (64, 16)) + 16, 0),
        (1, (16, 64), tight(1, (16, 64)), 0),
        (1, (16, 64), tight(1, (16, 64)), 48 * 64 * TEXEL),
    ];
    for (rotation, (w, h), bpr, past) in cases {
        let what = format!("rotation {rotation}, {w}x{h}, rows {bpr} bytes apart, {past} more");
        let (sw, sh) = plane_extent(rotation, (w, h));
        let length = bpr * sh + past;
        let named = named_words(length);
        let plane = lane.staged(&named);
        let texture = rotated(&plane, rotation, (w, h), bpr).unwrap();
        let region = Region::new_2d(0, 0, w, h);
        let reached = bpr * (h - 1) + w * TEXEL <= length;
        // getBytes.
        let mut out = vec![0u8; (w * h * TEXEL) as usize];
        let read = texture.get_bytes(&mut out, w * TEXEL, 0, region, 0, 0);
        // A blit out, and a blit in of the self-naming texels.
        let texels = naming_texels(w, h);
        let into = lane.staged(&vec![0; out.len()]);
        let from = lane.staged(&texels);
        let blit = |into_texture: bool| {
            let command_buffer = lane.queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            let size = Size::new(w, h, 1);
            let origin = Origin::new(0, 0, 0);
            let recorded = match into_texture {
                true => blit.copy_from_buffer_to_texture(
                    &from,
                    0,
                    w * TEXEL,
                    0,
                    size,
                    &texture,
                    0,
                    0,
                    origin,
                    BlitOption::None,
                ),
                false => blit.copy_from_texture_to_buffer(
                    &texture,
                    0,
                    0,
                    origin,
                    size,
                    &into,
                    0,
                    w * TEXEL,
                    0,
                    BlitOption::None,
                ),
            };
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            recorded
        };
        let blit_out = blit(false);
        write(&plane, &named);
        let blit_in = blit(true);
        let blitted = bytes_of(&plane);
        write(&plane, &named);
        let replaced = texture.replace_region(region, 0, 0, &texels, w * TEXEL, 0);
        if reached {
            read.unwrap();
            assert_eq!(out, rows_read(&named, (w, h), bpr), "{what}: getBytes");
            blit_out.unwrap();
            assert_eq!(bytes_of(&into), out, "{what}: a blit out");
            blit_in.unwrap();
            replaced.unwrap();
            let want = rows_written(&named, &texels, (w, h), bpr);
            if let Some(at) = first_difference(&blitted, &want, bpr) {
                panic!("{what}, a blit in: {at}");
            }
            if let Some(at) = first_difference(&bytes_of(&plane), &want, bpr) {
                panic!("{what}, replaceRegion: {at}");
            }
            eprintln!("ROTATED-BYTES {what}: CPU access and blits address the plane as Metal's");
        } else {
            assert_eq!(read, Err(gap.clone()), "{what}: getBytes");
            assert_eq!(blit_out, Err(gap.clone()), "{what}: a blit out");
            assert_eq!(blit_in, Err(gap.clone()), "{what}: a blit in");
            assert_eq!(replaced, Err(gap.clone()), "{what}: replaceRegion");
            eprintln!("ROTATED-BYTES {what}: rows past the buffer, refused");
        }
        // A fragment's read() at each position of a plain `w x h` target.
        write(&plane, &named);
        let mut d = TextureDescriptor::texture_2d(PF, w, h, false);
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::shared();
        let target = device.new_texture(&d).unwrap();
        lane.pass(&target, LoadAction::Clear, false, &texture, (w, h));
        target
            .get_bytes(&mut out, w * TEXEL, 0, region, 0, 0)
            .unwrap();
        let mut past_frame = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let got = &out[((y * w + x) * TEXEL) as usize..][..4];
                if x < sw && y < sh {
                    let at = (y * bpr + x * TEXEL) as usize;
                    assert_eq!(got, &named[at..at + 4], "{what}: read() at ({x}, {y})");
                } else {
                    past_frame.push(u32::from_le_bytes(got.try_into().unwrap()));
                }
            }
        }
        past_frame.sort_unstable();
        past_frame.dedup();
        eprintln!(
            "ROTATED-READ {what}: read() sees the stored {sw}x{sh} frame unturned; past it: \
             {past_frame:08x?}"
        );
        // A kernel writing texel (0, 0) and four of the plane's bytes another way: Metal drops
        // the texel's write, and the bytes' lands.
        write(&plane, &named);
        lane.two_ways(&texture, &plane, bpr as u32);
        let mut want = named.clone();
        want[bpr as usize..bpr as usize + 4].copy_from_slice(&[0x5a, 0x5b, 0x5c, 0x5d]);
        if let Some(at) = first_difference(&bytes_of(&plane), &want, bpr) {
            panic!("{what}, a kernel's write: {at}");
        }
    }
    messages.take_notices();
    quiet(&messages);
}

/// Views (`newTextureViewWithPixelFormat:`) of a rotated 64x16 texture, every rotation, each of
/// [`VIEWED`] viewed as each, the texture made with and without `PixelFormatView` usage, over a
/// plane whose rows are wider than its texels: plain Metal makes the view, of the texture's frame
/// extent, rotation and framebuffer-only flag, and it acts as the texture does in its own format. A
/// pass through it that clears and draws everything, and one that loads and draws its top-left
/// corner, leave each plane byte where the texture's would (rotation 4's mirror too, announced once
/// as `RotatedLoadTiles`), a fragment's `read()` through it sees the stored frame unturned, and
/// `getBytes` the plane's rows. An `_sRGB` format's drawn and read texels convert out and back,
/// each byte within 1. A view as another format, or of another type, of a rotated texture is
/// refused (`NotImplemented(RotatedTexture)`).
#[test]
fn views_of_rotated_textures_act_as_their_texture_does() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let lane = Lane::new(&device);
    let gap = Error::NotImplemented(RuntimeGap::RotatedTexture);
    let (w, h) = (64u64, 16u64);
    let sources = VIEWED.map(|pf| (pf, lane.naming_in(pf, w, h)));
    let mut loads = 0;
    for texture_format in VIEWED {
        for rotation in 0..=4u64 {
            let (sw, sh) = plane_extent(rotation, (w, h));
            let bpr = tight_row(rotation, (w, h), TEXEL) + 64;
            for more in [TextureUsage::empty(), TextureUsage::PIXEL_FORMAT_VIEW] {
                for view_format in VIEWED {
                    let what = format!(
                        "a {view_format:?} view of a {texture_format:?} texture of rotation \
                         {rotation}, usage {more:?} more"
                    );
                    let plane = plane(&lane, rotation, (w, h), bpr);
                    let texture =
                        rotated_in(&plane, (texture_format, more), rotation, (w, h), bpr).unwrap();
                    let view = texture.new_texture_view(view_format).unwrap();
                    assert_eq!(view.pixel_format(), view_format, "{what}");
                    assert_eq!((view.width(), view.height()), (w, h), "{what}");
                    assert_eq!(view.rotation(), rotation, "{what}");
                    assert!(view.framebuffer_only(), "{what}");
                    let slack = u8::from(matches!(
                        view_format,
                        PixelFormat::BGRA8UnormSRGB | PixelFormat::RGBA8UnormSRGB
                    ));
                    let (_, source) = sources.iter().find(|(pf, _)| *pf == view_format).unwrap();
                    // Cleared and drawn over everything.
                    let before = bytes_of(&plane);
                    lane.pass(&view, LoadAction::Clear, false, source, (w, h));
                    let want = expected(rotation, (w, h), bpr, (w, h), &before, |_, _| [0; 4]);
                    if let Some(at) = first_past(&bytes_of(&plane), &want, bpr, slack) {
                        panic!("{what}, a clear and a full draw: {at}");
                    }
                    // Loaded, and drawn over the pass's top-left 8x4 texels.
                    let mut named = bytes_of(&plane);
                    for sy in 0..sh {
                        for sx in 0..sw {
                            let at = (sy * bpr + sx * TEXEL) as usize;
                            named[at..at + 4].copy_from_slice(&[sx as u8, sy as u8, 0x33, 0x77]);
                        }
                    }
                    write(&plane, &named);
                    lane.pass(&view, LoadAction::Load, false, source, (8, 4));
                    loads += usize::from(rotation == 4);
                    let want = expected(rotation, (w, h), bpr, (8, 4), &named, |sx, sy| {
                        let (fx, fy) = loaded_from(rotation, sw, (sx, sy));
                        [fx as u8, fy as u8, 0x33, 0x77]
                    });
                    if let Some(at) = first_past(&bytes_of(&plane), &want, bpr, slack) {
                        panic!("{what}, a load and a corner's draw: {at}");
                    }
                    // getBytes, and a fragment's read() into a plain target of the view's format.
                    write(&plane, &named);
                    let region = Region::new_2d(0, 0, w, h);
                    let mut out = vec![0u8; (w * h * TEXEL) as usize];
                    view.get_bytes(&mut out, w * TEXEL, 0, region, 0, 0)
                        .unwrap();
                    assert_eq!(out, rows_read(&named, (w, h), bpr), "{what}: getBytes");
                    let mut d = TextureDescriptor::texture_2d(view_format, w, h, false);
                    d.usage = TextureUsage::RENDER_TARGET;
                    d.resource_options = ResourceOptions::shared();
                    let target = device.new_texture(&d).unwrap();
                    lane.pass(&target, LoadAction::Clear, false, &view, (w, h));
                    target
                        .get_bytes(&mut out, w * TEXEL, 0, region, 0, 0)
                        .unwrap();
                    for y in 0..h.min(sh) {
                        for x in 0..w.min(sw) {
                            let got = &out[((y * w + x) * TEXEL) as usize..][..4];
                            let at = (y * bpr + x * TEXEL) as usize;
                            let close = got
                                .iter()
                                .zip(&named[at..at + 4])
                                .all(|(a, b)| a.abs_diff(*b) <= slack);
                            assert!(close, "{what}: read() at ({x}, {y}): {got:?}");
                        }
                    }
                    eprintln!("ROTATED-VIEW {what}: as the texture's");
                }
            }
            // Views Metal is not measured to make.
            if rotation == 0 {
                continue;
            }
            let plane = plane(&lane, rotation, (w, h), bpr);
            let texture = rotated_in(
                &plane,
                (texture_format, TextureUsage::PIXEL_FORMAT_VIEW),
                rotation,
                (w, h),
                bpr,
            )
            .unwrap();
            let other_format = texture.new_texture_view(PixelFormat::R32Uint).map(|_| ());
            let other_type = texture
                .new_texture_view_with_levels_slices(
                    texture_format,
                    TextureType::Type2DArray,
                    0..1,
                    0..1,
                )
                .map(|_| ());
            assert_eq!(
                other_format,
                Err(gap.clone()),
                "rotation {rotation}: an R32Uint view"
            );
            assert_eq!(
                other_type,
                Err(gap.clone()),
                "rotation {rotation}: a 2D array view"
            );
        }
    }
    assert!(loads > 0);
    let counted = super::degrade_notices::counts(&messages.take_notices());
    assert_eq!(counted.get(&ROTATED_LOADS[0]), Some(&1), "{counted:?}");
    quiet(&messages);
}

/// What the creators refuse, as Metal asserts: a rotation past 4, a rotation without
/// framebufferOnly, and framebufferOnly anywhere but over an IOSurface plane; the shareable creator
/// makes the texture without framebufferOnly; the heap size query answers a rotated
/// framebuffer-only descriptor as a plain one.
#[test]
fn creators_refuse_what_metal_asserts_on() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let lane = Lane::new(&device);
    let refused = |made: Result<Texture, Error>, problem: TextureProblem| {
        assert!(
            matches!(&made, Err(Error::Invalid(Invalid { problem: p, .. }))
                if *p == Problem::Texture(problem)),
            "{problem:?}: {:?}",
            made.map(|_| ())
        );
    };
    let plane = plane(&lane, 1, (64, 16), 64);
    refused(rotated(&plane, 5, (64, 16), 64), TextureProblem::Rotation);
    let mut d = TextureDescriptor::texture_2d(PF, 64, 16, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.rotation = 1;
    let surface = IoSurfacePlane::new(0, 64, false);
    refused(
        plane.new_texture_with_iosurface_plane(&d, surface),
        TextureProblem::RotationWithoutFramebufferOnly,
    );
    d.rotation = 0;
    d.framebuffer_only = true;
    refused(device.new_texture(&d), TextureProblem::FramebufferOnly);
    d.resource_options = plane.resource_options();
    refused(
        plane.new_texture(&d, 0, 256),
        TextureProblem::FramebufferOnly,
    );
    d.resource_options = ResourceOptions::private();
    let shared = device.new_shared_texture(&d).unwrap();
    assert!(!shared.framebuffer_only());
    d.rotation = 1;
    refused(
        device.new_shared_texture(&d),
        TextureProblem::RotationWithoutFramebufferOnly,
    );
    // A heap's texture, and the heap size query, which Metal answers for a rotated
    // framebuffer-only descriptor as for a plain one, and asserts on for a rotation without
    // framebufferOnly.
    let heap = device
        .new_heap(&HeapDescriptor::new(
            HeapType::Automatic,
            1 << 20,
            ResourceOptions::private(),
        ))
        .unwrap();
    let mut d = TextureDescriptor::texture_2d(PF, 64, 16, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    let plain = device.heap_texture_size_and_align(&d).unwrap();
    d.framebuffer_only = true;
    refused(heap.new_texture(&d), TextureProblem::FramebufferOnly);
    for rotation in 0..=4 {
        d.rotation = rotation;
        assert_eq!(
            device.heap_texture_size_and_align(&d),
            Ok(plain),
            "{rotation}"
        );
    }
    d.framebuffer_only = false;
    d.rotation = 1;
    let asked = device.heap_texture_size_and_align(&d);
    assert!(
        matches!(&asked, Err(Error::Invalid(Invalid { problem, .. }))
            if *problem == Problem::Texture(TextureProblem::RotationWithoutFramebufferOnly)),
        "{asked:?}"
    );
    messages.take_notices();
    quiet(&messages);
}

/// Frames at the consumer's surface size, 1920x1080 `BGRA8Unorm`, over a plane of its own: a
/// pass that clears and draws, and one that loads and draws, rotation 0 against 1 and 2, each
/// timed over 30 frames after 3 warm ones (`ROTATION-COST`, the median and fastest). Measures;
/// asserts only that each frame completes.
#[test]
fn rotated_surface_frame_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    let lane = Lane::new(&device);
    let (w, h) = (1920u64, 1080u64);
    let source = lane.naming(w, h);
    for rotation in [0u64, 1, 2] {
        let bpr = tight_row(rotation, (w, h), TEXEL);
        let plane = plane(&lane, rotation, (w, h), bpr);
        let texture = rotated(&plane, rotation, (w, h), bpr).unwrap();
        for load in [LoadAction::Clear, LoadAction::Load] {
            let mut times = Vec::new();
            for k in 0..33 {
                let began = Instant::now();
                lane.pass(&texture, load, false, &source, (w / 2, h));
                if k >= 3 {
                    times.push(began.elapsed().as_secs_f64() * 1e3);
                }
            }
            times.sort_by(f64::total_cmp);
            eprintln!(
                "ROTATION-COST rotation {rotation} {load:?}+draw {w}x{h}: median {:.3} ms, \
                 fastest {:.3} ms",
                times[times.len() / 2],
                times[0]
            );
        }
    }
    messages.take_notices();
}
