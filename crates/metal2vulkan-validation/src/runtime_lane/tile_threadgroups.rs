//! Tile threadgroups up to the 32x32 tile through the runtime, on the device the cfg selects, under
//! the validation layer.
//!
//! - A tile pipeline's state answers Metal's `maxTotalThreadsPerThreadgroup`, 1024, where the
//!   device runs a threadgroup of 1024 threads, else the device's lower limit, and the
//!   descriptor's limit lowers it to the largest threadgroup of even sides within it (Metal
//!   probes, an M4 under macOS 27 and an M3 Max under macOS 26.7).
//! - `k10ts_tile_order` run by a 32x32 threadgroup in a 32x32 tile: each thread records its
//!   position, lane, simdgroup and its simdgroup's first lane's position at its thread index,
//!   as Metal places them (the probe: two 16-wide columns of 8x4 simdgroup blocks, each block's
//!   lanes its 2x2 quads in rows).
//! - Threads per tile past the state's limit run nothing, as plain Metal runs them.
//! - `ttg_tile_simd_blur`, a SIMD blur of 2x2 tiles of 32x32, each threadgroup's simdgroups the
//!   tile's rows (then its columns): every pixel is Metal's answer, twelve taps of
//!   `simd_shuffle_and_fill_down` that fill past the simdgroup's end from the next 32 texels.
//!   Where the device holds fewer threads of the blur (lavapipe's 8-wide subgroups emulate 32-lane
//!   simdgroups in its 32 KiB of shared memory: 960 threads), a 32x32 dispatch, within Metal's
//!   1024, is refused typed as the device's gap, and the blur runs 32 by the most simdgroups its
//!   state's limit takes, a power of two.
//! - Under `threadgroupSizeMatchesTileSize` (an M3 Max under macOS 26.7 and an M4
//!   under macOS 27.0.1): `k10ts_tile_grid` dispatched 8x8 per 16x16 tile over a 40x24 pass answers
//!   the tile as its threadgroup (threads per threadgroup 16x16, its position in the grid
//!   `threadgroup · 16 + position`, threads per grid 48x32) and still stages every cell; and
//!   `k10ts_tile_order` dispatched 16x16 in a 32x32 tile numbers its threads
//!   `simdgroup · 32 + lane`, each at its measured place.

use super::extended_range_targets::half_bits;
use super::tile_states::on_device_with;
use super::tile_states::{descriptor, on_device};
use super::{at_plan, read_buffer, WAIT};
use metal2vulkan::error::{BindError, Feature};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    CommandBufferStatus, Device, Error, LoadAction, PipelineOptions, Region, RenderCommandEncoder,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction, Texture,
    TextureDescriptor,
};

/// The tile's side, and the threadgroup's.
const SIDE: u64 = 32;

/// Encode a pass over `targets` of `SIDE`-square tiles, each attachment loaded and stored,
/// `encode` its commands, and run it.
fn run(device: &Device, targets: &[&Texture], encode: impl FnOnce(&mut RenderCommandEncoder)) {
    run_in_tiles(device, targets, SIDE, encode);
}

/// [`run`] in `tile`-square tiles.
fn run_in_tiles(
    device: &Device,
    targets: &[&Texture],
    tile: u64,
    encode: impl FnOnce(&mut RenderCommandEncoder),
) {
    let mut pass = RenderPassDescriptor::new();
    for (attachment, target) in pass.color_attachments.iter_mut().zip(targets) {
        attachment.texture = Some((*target).clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
    }
    pass.tile_width = tile;
    pass.tile_height = tile;
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// A shared 2D texture of `format`, `width` by `height`, of `usage`, holding `bytes` (rows of
/// `row` bytes).
fn texture(
    device: &Device,
    format: PixelFormat,
    [width, height]: [u64; 2],
    usage: TextureUsage,
    bytes: &[u8],
    row: u64,
) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, width, height, false);
    descriptor.usage = usage;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    texture
        .replace_region(Region::new_2d(0, 0, width, height), 0, 0, bytes, row, 0)
        .unwrap();
    texture
}

/// The bytes of `texture`'s `width` by `height` texels of `texel_bytes` each, in row order.
fn texels(texture: &Texture, [width, height]: [u64; 2], texel_bytes: u64) -> Vec<u8> {
    let mut bytes = vec![0u8; (width * height * texel_bytes) as usize];
    texture
        .get_bytes(
            &mut bytes,
            width * texel_bytes,
            0,
            Region::new_2d(0, 0, width, height),
            0,
            0,
        )
        .unwrap();
    bytes
}

/// The tile state of `stem`'s tile function over attachments of `formats`.
fn state(device: &Device, stem: &str, formats: &[PixelFormat]) -> RenderPipelineState {
    device
        .new_tile_render_pipeline_state(
            &descriptor(device, stem, formats),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// Metal's place of the thread of lane `lane` of simdgroup `simdgroup` in a 32x32 tile
/// threadgroup: the threadgroup is two columns 16 wide, simdgroups 0 to 15 the left
/// one's and 16 to 31 the right one's; a column's simdgroup `s` is its 8x4 block `(s mod 2, s /
/// 2)`, and lane `l` thread `l mod 4` of the block's quad `l / 4`, four quads a row, each quad's
/// threads in rows.
fn position(simdgroup: u64, lane: u64) -> (u64, u64) {
    let (column, s) = (simdgroup / 16, simdgroup % 16);
    let (quad, within) = (lane / 4, lane % 4);
    (
        column * 16 + (s % 2) * 8 + (quad % 4) * 2 + within % 2,
        (s / 2) * 4 + (quad / 4) * 2 + within / 2,
    )
}

/// Metal's `thread_index_in_threadgroup` of position (`x`, `y`) in a 32-wide tile threadgroup: its
/// quad's index in rows of 16 quads, times four, plus its place in the quad.
fn thread_index(x: u64, y: u64) -> u64 {
    ((y / 2) * (SIDE / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2
}

/// A tile state answers 1024 threads where the device runs them, as Metal's does, and the
/// descriptor's limit lowers it.
#[test]
fn a_tile_state_answers_metals_thread_limit() {
    on_device(|device| {
        let mut d = descriptor(device, "k10ts_tile_order", &[PixelFormat::R32Float]);
        let max = device
            .new_tile_render_pipeline_state(&d, &PipelineOptions::default())
            .unwrap()
            .max_total_threads_per_threadgroup();
        // The device's threadgroup on one axis, where a flattened threadgroup runs.
        let widest = device.properties().max_threads_per_threadgroup.width;
        assert_eq!(u64::from(max), widest.min(1024));
        d.max_total_threads_per_threadgroup = Some(512);
        let limited = device
            .new_tile_render_pipeline_state(&d, &PipelineOptions::default())
            .unwrap();
        assert_eq!(limited.max_total_threads_per_threadgroup(), max.min(512));
        // Metal answers the largest threadgroup of even sides within the limit.
        if max == 1024 {
            d.max_total_threads_per_threadgroup = Some(1000);
            let rounded = device
                .new_tile_render_pipeline_state(&d, &PipelineOptions::default())
                .unwrap();
            assert_eq!(rounded.max_total_threads_per_threadgroup(), 960);
        }
    });
}

/// Threads per tile past the state's limit run nothing, as plain Metal runs them (a state of 256
/// threads dispatched 32x32 per tile left every pixel and record as they were):
/// `k10ts_tile_order` limited to 256, dispatched 32x32 and then 16x16 in one pass, writes the
/// 16x16 dispatch's records and cells alone.
#[test]
fn threads_per_tile_past_the_states_limit_run_nothing() {
    on_device(|device| {
        let mut d = descriptor(device, "k10ts_tile_order", &[PixelFormat::R32Float]);
        d.max_total_threads_per_threadgroup = Some(256);
        let limited = device
            .new_tile_render_pipeline_state(&d, &PipelineOptions::default())
            .unwrap();
        assert_eq!(limited.max_total_threads_per_threadgroup(), 256);
        let area = [SIDE, SIDE];
        let start: Vec<u8> = (0..SIDE * SIDE)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let target = texture(
            device,
            PixelFormat::R32Float,
            area,
            TextureUsage::RENDER_TARGET,
            &start,
            SIDE * 4,
        );
        let records = SIDE * SIDE * 8;
        let out = device
            .new_buffer(records, ResourceOptions::shared())
            .unwrap();
        run(device, &[&target], |encoder| {
            encoder.set_render_pipeline_state(&limited).unwrap();
            encoder.set_tile_buffer(Some(&out), 0, 0).unwrap();
            encoder
                .dispatch_threads_per_tile(Size::new(SIDE, SIDE, 1))
                .unwrap();
            encoder
                .dispatch_threads_per_tile(Size::new(16, 16, 1))
                .unwrap();
        });
        let bytes = read_buffer(&out, 0, records).unwrap();
        // The 16x16 dispatch's 256 records of two words; none past them.
        assert!(
            bytes[256 * 8..].iter().all(|&b| b == 0),
            "a record past the 16x16 dispatch's"
        );
        let cells = texels(&target, area, 4);
        for (i, cell) in cells.as_chunks::<4>().0.iter().enumerate() {
            let (x, y) = (i as u64 % SIDE, i as u64 / SIDE);
            let f = f32::from_le_bytes([cell[0], cell[1], cell[2], cell[3]]);
            let added = if x < 16 && y < 16 { 1.0 } else { 0.0 };
            assert_eq!(f, i as f32 + added, "cell ({x}, {y})");
        }
    });
}

/// A 32x32 threadgroup in a 32x32 tile runs each thread at Metal's place: every record of
/// `k10ts_tile_order` (at its thread index: its position, lane and simdgroup, and its simdgroup's
/// first lane's position) and every cell, each added 1 once.
#[test]
fn a_32x32_tile_threadgroup_takes_metals_layout() {
    on_device(|device| {
        let state = state(device, "k10ts_tile_order", &[PixelFormat::R32Float]);
        let area = [SIDE, SIDE];
        let start: Vec<u8> = (0..SIDE * SIDE)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let target = texture(
            device,
            PixelFormat::R32Float,
            area,
            TextureUsage::RENDER_TARGET,
            &start,
            SIDE * 4,
        );
        let records = SIDE * SIDE * 8;
        let out = device
            .new_buffer(records, ResourceOptions::shared())
            .unwrap();
        run(device, &[&target], |encoder| {
            encoder.set_render_pipeline_state(&state).unwrap();
            encoder.set_tile_buffer(Some(&out), 0, 0).unwrap();
            encoder
                .dispatch_threads_per_tile(Size::new(SIDE, SIDE, 1))
                .unwrap();
        });
        let words: Vec<u32> = read_buffer(&out, 0, records)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        let place = |(x, y): (u64, u64)| (x | (y << 8)) as u32;
        for simdgroup in 0..SIDE {
            for lane in 0..32 {
                let (x, y) = position(simdgroup, lane);
                let index = thread_index(x, y) as usize;
                assert_eq!(
                    words[2 * index],
                    place((x, y)) | ((lane as u32) << 16) | ((simdgroup as u32) << 24),
                    "simdgroup {simdgroup} lane {lane}"
                );
                assert_eq!(
                    words[2 * index + 1],
                    place(position(simdgroup, 0)),
                    "simdgroup {simdgroup} lane {lane}: its first lane"
                );
            }
        }
        let cells = texels(&target, area, 4);
        for (i, cell) in cells.as_chunks::<4>().0.iter().enumerate() {
            let f = f32::from_le_bytes([cell[0], cell[1], cell[2], cell[3]]);
            assert_eq!(f, i as f32 + 1.0, "cell {i}");
        }
    });
}

/// Metal's quad order of a tile threadgroup of `qw` by `qh` 2x2 quads (measured on an Apple M3 Max
/// for every threadgroup of 1 to 32 threads a side): columns 8 quads wide, within a column its rows
/// of 2 quads, within a row its blocks 4 quads wide, within a block its quads row-major. Quad `k`
/// of the order is lanes `4 (k mod 8)` to `4 (k mod 8) + 3` of simdgroup `k / 8`, its threads in
/// rows.
fn quad_order(qw: u64, qh: u64) -> Vec<(u64, u64)> {
    let mut order = Vec::new();
    for c0 in (0..qw).step_by(8) {
        let c1 = (c0 + 8).min(qw);
        for r0 in (0..qh).step_by(2) {
            let r1 = (r0 + 2).min(qh);
            for b0 in (c0..c1).step_by(4) {
                let b1 = (b0 + 4).min(c1);
                for qy in r0..r1 {
                    for qx in b0..b1 {
                        order.push((qx, qy));
                    }
                }
            }
        }
    }
    order
}

/// Threadgroups of even sides other than squares and power-of-two rectangles run each thread at
/// Metal's place in a 32x32 tile: `k10ts_tile_order`'s record at each thread index (its position,
/// lane and simdgroup, and its simdgroup's first lane's position) and each cell of the
/// threadgroup, added 1 once, the cells past it kept. Among them the shapes whose simdgroup is a
/// whole row of quads (16x2, 32x2) and whose columns, rows and blocks end partly (6x6, 10x10,
/// 30x30, 24x8, 2x18, 14x4).
#[test]
fn even_tile_threadgroups_take_metals_layout() {
    on_device_with(at_plan, |device| {
        let state = state(device, "k10ts_tile_order", &[PixelFormat::R32Float]);
        let area = [SIDE, SIDE];
        let start: Vec<u8> = (0..SIDE * SIDE)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let place = |(x, y): (u64, u64)| (x | (y << 8)) as u32;
        for [w, h] in [
            [16, 2],
            [32, 2],
            [6, 6],
            [10, 10],
            [30, 30],
            [24, 8],
            [2, 18],
            [14, 4],
        ] {
            let target = texture(
                device,
                PixelFormat::R32Float,
                area,
                TextureUsage::RENDER_TARGET,
                &start,
                SIDE * 4,
            );
            let records = SIDE * SIDE * 8;
            let out = device
                .new_buffer(records, ResourceOptions::shared())
                .unwrap();
            run(device, &[&target], |encoder| {
                encoder.set_render_pipeline_state(&state).unwrap();
                encoder.set_tile_buffer(Some(&out), 0, 0).unwrap();
                encoder
                    .dispatch_threads_per_tile(Size::new(w, h, 1))
                    .unwrap();
            });
            let words: Vec<u32> = read_buffer(&out, 0, records)
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect();
            let order = quad_order(w / 2, h / 2);
            let at = |k: usize, within: u64| {
                let (qx, qy) = order[k];
                (qx * 2 + within % 2, qy * 2 + within / 2)
            };
            for (k, &(qx, qy)) in order.iter().enumerate() {
                let (simdgroup, first) = (k / 8, at(k / 8 * 8, 0));
                for within in 0..4 {
                    let (x, y) = at(k, within);
                    let lane = (k % 8) as u64 * 4 + within;
                    let index = (((qy * (w / 2) + qx) * 4) + within) as usize;
                    assert_eq!(
                        words[2 * index],
                        place((x, y)) | ((lane as u32) << 16) | ((simdgroup as u32) << 24),
                        "{w}x{h}: thread ({x}, {y})"
                    );
                    assert_eq!(
                        words[2 * index + 1],
                        place(first),
                        "{w}x{h}: thread ({x}, {y}), its first lane"
                    );
                }
            }
            let cells = texels(&target, area, 4);
            for (i, cell) in cells.as_chunks::<4>().0.iter().enumerate() {
                let (x, y) = (i as u64 % SIDE, i as u64 / SIDE);
                let f = f32::from_le_bytes([cell[0], cell[1], cell[2], cell[3]]);
                let added = if x < w && y < h { 1.0 } else { 0.0 };
                assert_eq!(f, i as f32 + added, "{w}x{h}: cell ({x}, {y})");
            }
        }
    });
}

/// The blur's source texel at (`x`, `y`): small integers, so every sum is exact in `half`.
fn source_texel(x: u64, y: u64) -> [u64; 4] {
    [(x + 2 * y) % 8, (3 * x + y) % 8, (x * y) % 8, (x ^ y) % 8]
}

/// The blur's twelve weights.
fn weight(tap: u64) -> u64 {
    tap % 3 + 1
}

/// The blur's source extent, its origin in it, and its color scale.
const SOURCE: [u64; 2] = [160, 160];
const ORIGIN: [u64; 2] = [3, 5];
const SCALE: f32 = 0.5;

/// Metal's answer of `ttg_tile_simd_blur` over 2x2 tiles of 32x32 run by threadgroups of 32 by
/// `simdgroups` threads, the simdgroups the rows (or, `transpose`, the columns): the `half` bits
/// of each pixel's four channels in row order. Lane `l` of simdgroup `g` in tile `(tx, ty)` is the
/// cell `(l, g)` (or `(g, l)`), its texel `(32 tx, simdgroups ty)` past the origin and the cell;
/// tap `i` takes lane `l + i`'s texel where that is a lane, else the texel 32 past lane
/// `l + i - 32`'s along the simdgroup (`simd_shuffle_and_fill_down`, measured by our probe on
/// every pixel of 32 simdgroups). A cell no thread owns keeps the target's 9.
fn blurred(transpose: bool, simdgroups: u64) -> Vec<u16> {
    let size = 2 * SIDE;
    let mut out = vec![half_bits(9.0); (size * size * 4) as usize];
    for (tx, ty) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        for g in 0..simdgroups {
            let cell = |lane: u64| if transpose { (g, lane) } else { (lane, g) };
            let at = |lane: u64| {
                let (cx, cy) = cell(lane);
                (ORIGIN[0] + tx * SIDE + cx, ORIGIN[1] + ty * simdgroups + cy)
            };
            for l in 0..32 {
                let mut sum = [0u64; 4];
                for i in 0..12 {
                    let texel = if l + i < 32 {
                        let (x, y) = at(l + i);
                        source_texel(x, y)
                    } else {
                        let (x, y) = at(l + i - 32);
                        if transpose {
                            source_texel(x, y + 32)
                        } else {
                            source_texel(x + 32, y)
                        }
                    };
                    for c in 0..4 {
                        sum[c] += weight(i) * texel[c];
                    }
                }
                let (cx, cy) = cell(l);
                let pixel = ((ty * SIDE + cy) * size + tx * SIDE + cx) as usize;
                for c in 0..4 {
                    let scale = if c < 3 { SCALE } else { 1.0 };
                    out[pixel * 4 + c] = half_bits(sum[c] as f32 * scale);
                }
            }
        }
    }
    out
}

/// A SIMD tile blur of 32x32 threadgroups over 2x2 tiles of 32x32, along the rows and along the
/// columns, writes Metal's answer to every pixel; where the device holds fewer threads of it, the
/// 32x32 dispatch, which Metal runs, is refused typed as the device's gap, and the blur runs in
/// threadgroups 32 by the most simdgroups its state takes.
#[test]
fn a_32x32_simd_tile_blur_is_metals() {
    on_device_with(at_plan, |device| {
        let state = state(device, "ttg_tile_simd_blur", &[PixelFormat::RGBA16Float]);
        let max = u64::from(state.max_total_threads_per_threadgroup());
        let simdgroups = 1u64 << (max / 32).clamp(1, SIDE).ilog2();
        // A tile threadgroup 32 wide is at least four simdgroups tall.
        assert!(simdgroups >= 4, "the blur's state takes {max} threads");
        eprintln!("the blur at 32x{simdgroups}: its state takes {max} threads");
        let source_bytes: Vec<u8> = (0..SOURCE[1])
            .flat_map(|y| (0..SOURCE[0]).map(move |x| (x, y)))
            .flat_map(|(x, y)| source_texel(x, y))
            .flat_map(|v| half_bits(v as f32).to_le_bytes())
            .collect();
        let source = texture(
            device,
            PixelFormat::RGBA16Float,
            SOURCE,
            TextureUsage::SHADER_READ,
            &source_bytes,
            SOURCE[0] * 8,
        );
        let weights: Vec<u8> = (0..12)
            .flat_map(|i| half_bits(weight(i) as f32).to_le_bytes())
            .collect();
        let area = [2 * SIDE, 2 * SIDE];
        for transpose in [false, true] {
            // Every pixel starts at 9 in every channel, which no blurred pixel holds.
            let nine: Vec<u8> =
                std::iter::repeat_n(half_bits(9.0).to_le_bytes(), (4 * 4 * SIDE * SIDE) as usize)
                    .flatten()
                    .collect();
            let target = texture(
                device,
                PixelFormat::RGBA16Float,
                area,
                TextureUsage::RENDER_TARGET,
                &nine,
                area[0] * 8,
            );
            // origin, fill (32 texels along the simdgroup), transpose, scale.
            let fill = if transpose { [0u16, 32] } else { [32, 0] };
            let params: Vec<u8> = [
                ORIGIN[0] as u16,
                ORIGIN[1] as u16,
                fill[0],
                fill[1],
                u16::from(transpose),
                half_bits(SCALE),
            ]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
            run(device, &[&target], |encoder| {
                encoder.set_render_pipeline_state(&state).unwrap();
                encoder.set_tile_texture(Some(&source), 0).unwrap();
                encoder.set_tile_bytes(&params, 0).unwrap();
                encoder.set_tile_bytes(&weights, 1).unwrap();
                if simdgroups < SIDE {
                    // Within Metal's 1024 but past the device's limit: the device's gap, refused
                    // typed, never run as nothing; the pass goes on unchanged.
                    match encoder.dispatch_threads_per_tile(Size::new(SIDE, SIDE, 1)) {
                        Err(Error::Bind(BindError::Unsupported(u))) => assert!(
                            matches!(
                                u.feature,
                                Feature::LocalSize { .. } | Feature::ThreadgroupMemorySize
                            ),
                            "the 32x32 blur past {max} threads: {u:?}"
                        ),
                        other => panic!("the 32x32 blur past {max} threads: {other:?}"),
                    }
                }
                encoder
                    .dispatch_threads_per_tile(Size::new(SIDE, simdgroups, 1))
                    .unwrap();
            });
            let got: Vec<u16> = texels(&target, area, 8)
                .as_chunks::<2>()
                .0
                .iter()
                .map(|h| u16::from_le_bytes([h[0], h[1]]))
                .collect();
            let want = blurred(transpose, simdgroups);
            let wrong: Vec<usize> = (0..got.len() / 4)
                .filter(|&p| got[p * 4..p * 4 + 4] != want[p * 4..p * 4 + 4])
                .collect();
            assert!(
                wrong.is_empty(),
                "transpose {transpose}: {} of {} pixels differ, the first at {:?}: {:?}, Metal's {:?}",
                wrong.len(),
                got.len() / 4,
                wrong.first().map(|&p| (p as u64 % area[0], p as u64 / area[0])),
                wrong.first().map(|&p| &got[p * 4..p * 4 + 4]),
                wrong.first().map(|&p| &want[p * 4..p * 4 + 4]),
            );
        }
    });
}

/// The tag `k10ts_tile_grid` writes at cell `c` of tile `tg`.
fn cell_tag(tg: [u64; 2], c: [u64; 2]) -> u32 {
    (0xA000_0000 | (tg[1] << 20) | (tg[0] << 16) | (c[1] << 8) | c[0]) as u32
}

/// `k10ts_tile_grid` under `threadgroupSizeMatchesTileSize`, dispatched 8x8 per 16x16 tile over a
/// 40x24 pass (the shape of our probe, where Metal's validation, which stops it, was off):
/// each record answers the tile as the threadgroup, as plain Metal does on an M3 Max and an M4
/// (threads per threadgroup 16x16, the position in the grid `threadgroup · 16 + position`,
/// threads per grid the 3x2 tiles times the tile), its thread index the 8x8 threadgroup's, and
/// the 64 threads stage every cell of their tile: each pixel is read and written once.
#[test]
fn a_tile_sized_threadgroup_answers_the_tile() {
    on_device(|device| {
        let mut d = descriptor(
            device,
            "k10ts_tile_grid",
            &[
                PixelFormat::R32Float,
                PixelFormat::R32Uint,
                PixelFormat::R32Uint,
            ],
        );
        d.threadgroup_size_matches_tile_size = true;
        let tile_sized = device
            .new_tile_render_pipeline_state(&d, &PipelineOptions::default())
            .unwrap();
        assert!(tile_sized.threadgroup_size_matches_tile_size());
        let (w, h) = (40u64, 24u64);
        let pixels = (0..h).flat_map(|y| (0..w).map(move |x| (x, y)));
        let f0: Vec<f32> = pixels.clone().map(|(x, y)| (y * 64 + x) as f32).collect();
        let v0: Vec<u32> = pixels.map(|(x, y)| (y * 1000 + x + 7) as u32).collect();
        let bytes =
            |words: Vec<u32>| -> Vec<u8> { words.iter().flat_map(|v| v.to_le_bytes()).collect() };
        let usage = TextureUsage::RENDER_TARGET;
        let targets = [
            texture(
                device,
                PixelFormat::R32Float,
                [w, h],
                usage,
                &bytes(f0.iter().map(|f| f.to_bits()).collect()),
                w * 4,
            ),
            texture(
                device,
                PixelFormat::R32Uint,
                [w, h],
                usage,
                &bytes(v0.clone()),
                w * 4,
            ),
            texture(
                device,
                PixelFormat::R32Uint,
                [w, h],
                usage,
                &bytes(vec![0x5555_5555; (w * h) as usize]),
                w * 4,
            ),
        ];
        let record_bytes = 512 * 48;
        let out = device
            .new_buffer(record_bytes, ResourceOptions::shared())
            .unwrap();
        run_in_tiles(
            device,
            &[&targets[0], &targets[1], &targets[2]],
            16,
            |encoder| {
                encoder.set_render_pipeline_state(&tile_sized).unwrap();
                encoder.set_tile_buffer(Some(&out), 0, 0).unwrap();
                encoder
                    .dispatch_threads_per_tile(Size::new(8, 8, 1))
                    .unwrap();
            },
        );
        let plane = |i: usize| -> Vec<u32> {
            texels(&targets[i], [w, h], 4)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let planes = [plane(0), plane(1), plane(2)];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let (tg, c) = ([x / 16, y / 16], [x % 16, y % 16]);
                let k = 1 + (c[0] % 2) + 2 * (c[1] % 2);
                assert_eq!(planes[0][i], (f0[i] * 2.0 + 1.0).to_bits(), "f ({x}, {y})");
                assert_eq!(planes[1][i], v0[i] * 3 + k as u32, "v ({x}, {y})");
                assert_eq!(planes[2][i], cell_tag(tg, c), "tag ({x}, {y})");
            }
        }
        let words: Vec<u32> = read_buffer(&out, 0, record_bytes)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        for tile in 0..6u64 {
            let tg = [tile % 3, tile / 3];
            for tidx in 0..64u64 {
                let slot = (tg[1] * 4 + tg[0]) * 64 + tidx;
                let r = &words[(slot * 12) as usize..(slot * 12 + 12) as usize];
                // An 8-wide threadgroup's index is its quads' order, which is
                // `simdgroup · 32 + lane` too.
                let (quad, within) = (tidx / 4, tidx % 4);
                let tid = [(quad % 4) * 2 + within % 2, (quad / 4) * 2 + within / 2];
                let mirrored = [15 - 2 * tid[0], 15 - 2 * tid[1]];
                let want = [
                    tid[0] | (tid[1] << 16),
                    tg[0] | (tg[1] << 16),
                    tg[0] * 16 + tid[0],
                    tg[1] * 16 + tid[1],
                    16 | (16 << 16),
                    48,
                    32,
                    tidx,
                    0,
                    16 | (16 << 16),
                ]
                .map(|v| v as u32);
                assert_eq!(r[..10], want, "tile {tg:?} thread {tidx}");
                assert_eq!(r[10], cell_tag(tg, mirrored), "tile {tg:?} thread {tidx}");
                assert_eq!(r[11], 0xC0DE_0000 | slot as u32);
            }
        }
    });
}

/// `k10ts_tile_order` under `threadgroupSizeMatchesTileSize`, dispatched 16x16 in a 32x32 tile:
/// each thread keeps its measured place, lane and simdgroup, and its thread index is
/// `simdgroup · 32 + lane`, not the quad order across the 16-wide threadgroup.
#[test]
fn a_tile_sized_threadgroup_numbers_its_threads_by_simdgroup() {
    on_device(|device| {
        let mut d = descriptor(device, "k10ts_tile_order", &[PixelFormat::R32Float]);
        d.threadgroup_size_matches_tile_size = true;
        let tile_sized = device
            .new_tile_render_pipeline_state(&d, &PipelineOptions::default())
            .unwrap();
        let area = [SIDE, SIDE];
        let start: Vec<u8> = (0..SIDE * SIDE)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let target = texture(
            device,
            PixelFormat::R32Float,
            area,
            TextureUsage::RENDER_TARGET,
            &start,
            SIDE * 4,
        );
        let records = 256 * 8;
        let out = device
            .new_buffer(records, ResourceOptions::shared())
            .unwrap();
        run(device, &[&target], |encoder| {
            encoder.set_render_pipeline_state(&tile_sized).unwrap();
            encoder.set_tile_buffer(Some(&out), 0, 0).unwrap();
            encoder
                .dispatch_threads_per_tile(Size::new(16, 16, 1))
                .unwrap();
        });
        let words: Vec<u32> = read_buffer(&out, 0, records)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        // A 16x16 threadgroup is one column of eight 8x4 blocks, two a row.
        let place = |simdgroup: u64, lane: u64| -> (u64, u64) {
            let (quad, within) = (lane / 4, lane % 4);
            (
                (simdgroup % 2) * 8 + (quad % 4) * 2 + within % 2,
                (simdgroup / 2) * 4 + (quad / 4) * 2 + within / 2,
            )
        };
        let packed = |(x, y): (u64, u64)| (x | (y << 8)) as u32;
        for simdgroup in 0..8 {
            for lane in 0..32 {
                let index = (simdgroup * 32 + lane) as usize;
                assert_eq!(
                    words[2 * index],
                    packed(place(simdgroup, lane))
                        | ((lane as u32) << 16)
                        | ((simdgroup as u32) << 24),
                    "simdgroup {simdgroup} lane {lane}"
                );
                assert_eq!(words[2 * index + 1], packed(place(simdgroup, 0)));
            }
        }
        let cells = texels(&target, area, 4);
        for (i, cell) in cells.as_chunks::<4>().0.iter().enumerate() {
            let (x, y) = (i as u64 % SIDE, i as u64 / SIDE);
            let f = f32::from_le_bytes([cell[0], cell[1], cell[2], cell[3]]);
            let added = if x < 16 && y < 16 { 1.0 } else { 0.0 };
            assert_eq!(f, i as f32 + added, "cell ({x}, {y})");
        }
    });
}
