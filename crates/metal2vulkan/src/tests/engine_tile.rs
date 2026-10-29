//! Tile functions through tile staging, end to end on the CPU interpreter.
//!
//! A tile function compiles under its tile pipeline's color attachment formats
//! (`LoweringInputs::tile_attachments`): its implicit imageblock, and an explicit layout aliased
//! onto it, are the threadgroup arena's imageblock region holding the tile's attachment texels,
//! loaded before the function runs and stored after it ends. Each test runs the owned fixtures
//! (`validation/fixtures/public/engine/k10ts_*.metal`, whose behavior Metal qualified on Apple
//! silicon) on every committed profile that compiles them, in both scheduling orders, and checks
//! every texel and record against the measured model: tile `t`'s cell `c` is pixel `t · tile + c`,
//! a cell past the render area holds the function's writes but is never stored, the dispatch
//! builtins are a compute dispatch's over the tiles, and an aliased layout's member `i` holds
//! attachment `i`'s texel bits at its offset. The interpreter traps a read of threadgroup memory
//! nobody wrote and any image access outside the image, so a cell the staging misses, or a pixel
//! past the render area it reads or writes, fails the run.
//!
//! - `k10ts_tile_grid`: a 40x24 render area in 16x16 tiles (partial ones on both axes) of 8x8
//!   threads, each thread owning four cells; `float`, `uint`, and `uint` attachments.
//! - `k10ts_alias_probe`: one tile of `r16Float`, `r32Float`, `rgba16Float`, and `r16Float`
//!   attachments read through an aliased layout whose offsets are not the packed ones, their
//!   texels including denormals, infinities, and NaNs with payloads, whose bits the staging keeps.
//! - `cfit_tile_memory`: tile functions over their tile's threadgroup memory, which the pass keeps
//!   across its dispatches in one buffer of every tile's: dispatches accumulate in their own tile's
//!   words, and arguments placed at overlapping offsets share them.
//! - `cfb_tile_norm`: tile functions over a normalized attachment, whose cell holds the texel's
//!   bits, each access converting them by the device's conversion of Metal's pack functions
//!   (`arith::pack`, `PrecisionNote::DevicePackConversion`): every format's reads, stores and a
//!   read after the function's own write, within the note's bound of the probes's model; and
//!   through `half` members, of probe 12's. The interpreter models no `Pow` and leaves a pack of a
//!   NaN undefined, so the sRGB formats and NaN stores are the device lane's
//!   (`runtime_lane::tile_normalized`).
//! - The refusals are typed: no tile inputs, a converting format, an attachment without a format,
//!   a multisampled pass, a threadgroup variable in a tile function; the binder specializes the
//!   render area, refuses an empty one, and binds the attachments in their staging formats, and
//!   places a tile function's threadgroup arguments where the encoder set them.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use std::collections::BTreeMap;

use crate::bind::{
    BindState, BindingPlan, Dispatch, ImageblockSize, StageTable, WriteSource, SRGB_RESOLVE_LINEAR,
    SRGB_RESOLVE_THRESHOLDS,
};
use crate::budget::Budget;
use crate::device::profile::DeviceProfile;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{BindError, DescriptorError, Error, Feature, Needs, RuntimeInput};
use crate::library::Library;
use crate::lower::select::{LowerInputs, Plan};
use crate::lower::texture::TexelClass;
use crate::metal::PixelFormat;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::layout::BindingSource;
use crate::pipeline::sym::{
    Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage, SyntheticBinding,
};
use crate::pipeline::testing::{engine_dir, llvm_assemble, typed_module_with, AssembleError};
use crate::pipeline::{
    ComputePipeline, FunctionDescriptor, StageSource, StorageAttachmentView,
    TileRenderPipelineDescriptor,
};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;
use crate::vk;

fn assemble(stem: &str) -> Option<Vec<u8>> {
    let path = engine_dir().join(format!("{stem}.ll"));
    match llvm_assemble(&path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {}: {why}", path.display()),
    }
}

fn tile_inputs(formats: &[PixelFormat]) -> LowerInputs {
    let mut inputs = LowerInputs::new();
    inputs.tile_attachments = Some(
        formats
            .iter()
            .enumerate()
            .map(|(i, f)| (i as u32, *f))
            .collect(),
    );
    inputs
}

/// The module of `entry` under `inputs` on `p`, or `None` when `p` lacks a capability its
/// attachments' storage images need (`Feature::Format`).
fn module(
    bc: &[u8],
    entry: &str,
    stem: &str,
    p: &DeviceProfile,
    inputs: &LowerInputs,
) -> Option<(Plan, TypedModule)> {
    match typed_module_with(bc, entry, p, inputs) {
        Ok(m) => Some(m),
        Err(Error::Unsupported(u)) if u.feature == Feature::Format => None,
        Err(e) => panic!("{stem}: {entry}: {e}"),
    }
}

fn attachment(color: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::TileAttachment(color),
        index: color,
    }
}

/// One tile dispatch.
struct Tiles<'a> {
    area: [u32; 2],
    tile: [u32; 2],
    threads: [u32; 2],
    /// `[[buffer(0)]]`.
    buffer: Vec<u8>,
    /// Each attachment's texel bits, row-major over the render area (the first channel; a 16-bit
    /// channel zero-extended).
    attachments: &'a [Vec<u32>],
    /// Metal's threadgroup where the function's threadgroup is the tile: the binder's region then
    /// carries it and the tiles times it as the grid (`MultiRegion` 1).
    nominal: Option<[u32; 2]>,
}

/// Whether the interpreter's image of attachment `color` holds a half channel as its `f32` value
/// (a float view of a 16-bit float attachment) rather than its bits. The image stores the four
/// 32-bit components of its sampled type, so a float view holds the value the read gives.
fn half_as_value(plan: &Plan, color: u32) -> bool {
    plan.families
        .access
        .tile
        .as_ref()
        .and_then(|t| t.slots.iter().find(|s| s.color == color))
        .is_some_and(|s| {
            s.class == TexelClass::Float
                && matches!(
                    s.pixel_format,
                    PixelFormat::R16Float | PixelFormat::RG16Float | PixelFormat::RGBA16Float
                )
        })
}

/// Run `m` over `d`'s tiles in both scheduling orders (which must agree): the buffer and each
/// attachment's first channels' bits after.
fn run(plan: &Plan, m: &TypedModule, d: &Tiles<'_>) -> (Vec<u8>, Vec<Vec<u32>>) {
    let images: Vec<Image> = d
        .attachments
        .iter()
        .enumerate()
        .map(|(i, texels)| {
            let mut img = Image::new(attachment(i as u32), [d.area[0], d.area[1], 1], 1, 1, 1);
            let value = half_as_value(plan, i as u32);
            for (t, x) in img.texels.iter_mut().zip(texels) {
                let x = if value {
                    half_bits_to_f32(*x as u16)
                } else {
                    *x
                };
                *t = [x, 0, 0, 0];
            }
            img
        })
        .collect();
    let (buffer, images) = run_images(plan, m, d, images);
    let planes = images
        .iter()
        .enumerate()
        .map(|(i, texels)| {
            let value = half_as_value(plan, i as u32);
            texels
                .iter()
                .map(|t| {
                    if value {
                        u32::from(f32_to_half_bits(t[0]))
                    } else {
                        t[0]
                    }
                })
                .collect()
        })
        .collect();
    (buffer, planes)
}

/// Run `m` over `d`'s tiles (its attachments unread) with the attachments' images `images`, in
/// both scheduling orders (which must agree): the buffer and each image's texels after.
fn run_images(
    plan: &Plan,
    m: &TypedModule,
    d: &Tiles<'_>,
    images: Vec<Image>,
) -> (Vec<u8>, Vec<Vec<[u32; 4]>>) {
    assert!(plan.facts.contains(&RuntimeFact::TileFunction));
    let mut spec = BTreeMap::new();
    spec.insert(SpecKey::LocalSize(Axis::X), u64::from(d.threads[0]));
    spec.insert(SpecKey::LocalSize(Axis::Y), u64::from(d.threads[1]));
    spec.insert(SpecKey::LocalSize(Axis::Z), 1);
    spec.insert(
        SpecKey::FlatLocalSize,
        u64::from(d.threads[0] * d.threads[1]),
    );
    spec.insert(SpecKey::ImageblockWidth, u64::from(d.tile[0]));
    spec.insert(SpecKey::ImageblockHeight, u64::from(d.tile[1]));
    spec.insert(SpecKey::RenderAreaWidth, u64::from(d.area[0]));
    spec.insert(SpecKey::RenderAreaHeight, u64::from(d.area[1]));
    let groups = [
        d.area[0].div_ceil(d.tile[0]),
        d.area[1].div_ceil(d.tile[1]),
        1,
    ];
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            // One region, every buffer bound at its start.
            (PushField::SlotBindOffset(_), PushLoc::Push { .. }) => {}
            // Metal's threadgroup the tile where the function's is, as the binder's region carries
            // it.
            (PushField::DispatchRegion, PushLoc::Push { offset }) => {
                if let Some([nx, ny]) = d.nominal {
                    let grid = [groups[0] * nx, groups[1] * ny, 1];
                    let bytes = crate::lower::dispatch::region_bytes([0; 3], grid, [nx, ny, 1])
                        .expect("a packable tile");
                    let at = offset as usize;
                    push[at..at + bytes.len()].copy_from_slice(&bytes);
                    spec.insert(SpecKey::MultiRegion, 1);
                }
            }
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let buffers = vec![Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index: 0,
        }),
        address: None,
        bytes: d.buffer.clone(),
    }];
    let res = Resources {
        buffers,
        push,
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            order,
            workgroups: groups,
            ..Config::default()
        };
        let (o, imgs) = run_with_images(m, &cfg, res.clone(), images.clone()).expect("run");
        let texels: Vec<Vec<[u32; 4]>> = imgs.into_iter().map(|img| img.texels).collect();
        outs.push((o.buffers[0].clone(), texels));
    }
    assert_eq!(outs[0], outs[1], "scheduling orders disagree");
    outs.swap_remove(0)
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cell_tag(tg: [u32; 2], c: [u32; 2]) -> u32 {
    0xA000_0000 | (tg[1] << 20) | (tg[0] << 16) | (c[1] << 8) | c[0]
}

/// A 40x24 render area in 16x16 tiles of 8x8 threads: every pixel is its own tile's cell, read
/// and written once; a record shows each thread's builtins and the tag another thread wrote in a
/// cell (past the render area for the partial tiles) across the imageblock barrier.
#[test]
fn every_tile_stages_its_pixels() {
    stage_every_pixel(false);
}

/// [`every_tile_stages_its_pixels`] under `threadgroupSizeMatchesTileSize`: the 8x8 threads
/// still stage every cell of their 16x16 tile, and the builtins answer the tile as the
/// threadgroup, as plain Metal answers them (an M3 Max and an M4): threads per
/// threadgroup 16x16, the position in the grid `threadgroup · 16 + position`, and the threads per
/// grid the 3x2 tiles times the tile, 48x32.
#[test]
fn a_tile_sized_threadgroup_stages_every_pixel_and_answers_the_tile() {
    stage_every_pixel(true);
}

/// The 40x24 pass of 16x16 tiles of 8x8 threads over `k10ts_tile_grid`, whose pipeline's
/// threadgroup is the tile where `tile_sized`.
fn stage_every_pixel(tile_sized: bool) {
    let Some(bc) = assemble("k10ts_tile_grid") else {
        return;
    };
    let (w, h) = (40u32, 24u32);
    let f0: Vec<u32> = (0..h)
        .flat_map(|y| (0..w).map(move |x| ((y * 64 + x) as f32).to_bits()))
        .collect();
    let v0: Vec<u32> = (0..h)
        .flat_map(|y| (0..w).map(move |x| y * 1000 + x + 7))
        .collect();
    let t0 = vec![0x5555_5555u32; (w * h) as usize];
    let mut inputs = tile_inputs(&[
        PixelFormat::R32Float,
        PixelFormat::R32Uint,
        PixelFormat::R32Uint,
    ]);
    inputs.threadgroup_size_matches_tile_size = tile_sized;
    // Metal's threadgroup: the tile where it is the tile, else the threads per tile.
    let group = if tile_sized { 16 } else { 8 };
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some((plan, m)) = module(&bc, "k10ts_tile_grid", &stem, &p, &inputs) else {
            continue;
        };
        assert!(plan
            .facts
            .contains(&RuntimeFact::TileThreadgroup { tile_sized }));
        let d = Tiles {
            area: [w, h],
            tile: [16, 16],
            threads: [8, 8],
            buffer: vec![0xAB; 512 * 48],
            attachments: &[f0.clone(), v0.clone(), t0.clone()],
            nominal: tile_sized.then_some([16, 16]),
        };
        let (buf, planes) = run(&plan, &m, &d);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let (tg, c) = ([x / 16, y / 16], [x % 16, y % 16]);
                let k = 1 + (c[0] % 2) + 2 * (c[1] % 2);
                let f = f32::from_bits(f0[i]) * 2.0 + 1.0;
                assert_eq!(planes[0][i], f.to_bits(), "{stem} f ({x}, {y})");
                assert_eq!(planes[1][i], v0[i] * 3 + k, "{stem} v ({x}, {y})");
                assert_eq!(planes[2][i], cell_tag(tg, c), "{stem} tag ({x}, {y})");
            }
        }
        let out = words(&buf);
        let mut records = 0;
        for slot in 0..512u32 {
            let r = &out[(slot * 12) as usize..(slot * 12 + 12) as usize];
            let (tile, tidx) = (slot / 64, slot % 64);
            let tg = [tile % 4, tile / 4];
            if tg[0] >= 3 || tg[1] >= 2 {
                assert!(r.iter().all(|x| *x == 0xABAB_ABAB), "{stem} slot {slot}");
                continue;
            }
            // Metal's tile dispatch numbers a threadgroup's 2x2 quads in row-major order, each
            // quad's threads in row-major order (measured).
            let (quad, within) = (tidx / 4, tidx % 4);
            let tid = [(quad % 4) * 2 + within % 2, (quad / 4) * 2 + within / 2];
            let mirrored = [15 - 2 * tid[0], 15 - 2 * tid[1]];
            let want = [
                tid[0] | (tid[1] << 16),
                tg[0] | (tg[1] << 16),
                tg[0] * group + tid[0],
                tg[1] * group + tid[1],
                group | (group << 16),
                3 * group,
                2 * group,
                tidx,
                0,
                16 | (16 << 16),
                cell_tag(tg, mirrored),
                0xC0DE_0000 | slot,
            ];
            assert_eq!(r, want, "{stem} slot {slot}");
            records += 1;
        }
        assert_eq!(records, 6 * 64);
        ran += 1;
    }
    assert!(ran > 0);
}

/// IEEE binary16 bits `h` as the `f32` bits of the same value, for every half: a denormal is
/// a normal `f32`, and an infinity or NaN keeps its payload in the top bits.
fn half_bits_to_f32(h: u16) -> u32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1F);
    let man = u32::from(h & 0x3FF);
    match exp {
        0 => sign | (man as f32 / 16_777_216.0).to_bits(),
        31 => sign | 0x7F80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    }
}

/// [`half_bits_to_f32`]'s inverse on the `f32` values of halves.
fn f32_to_half_bits(f: u32) -> u16 {
    let sign = ((f >> 16) & 0x8000) as u16;
    let mag = f & 0x7FFF_FFFF;
    let bits = if mag >= 0x7F80_0000 {
        0x7C00 | (((mag >> 13) & 0x3FF) as u16)
    } else if mag < 0x3880_0000 {
        (f32::from_bits(mag) * 16_777_216.0) as u16
    } else {
        ((mag - (112 << 23)) >> 13) as u16
    };
    sign | bits
}

/// A half texel of the probe for pixel `p` and channel `k`: normals, and then the values a
/// float conversion need not keep: denormals of both signs, infinities, and NaNs with payloads,
/// quiet and signalling.
fn probe_half(p: u32, k: u16) -> u16 {
    let p = p as u16;
    match p % 6 {
        0 | 1 => 0x1100 + k * 0x400 + p,
        2 => 1 + p + k,
        3 => 0x8000 | (0x3FF - p - k),
        4 => [0x7C00, 0xFC00, 0x7E00 | p, 0x7C01 + p][usize::from((p / 6 + k) % 4)],
        _ => 0xFC01 + ((p + k) & 0x1FF),
    }
}

/// A 32-bit float texel of the probe for pixel `p`: normals, denormals, infinities, and NaNs
/// with payloads.
fn probe_f32(p: u32) -> u32 {
    match p % 5 {
        0 | 1 => 0x2200_0022 | (p << 8),
        2 => 1 + p,
        3 => 0x7F80_0001 + p,
        _ => 0xFFC0_1234 + p,
    }
}

/// An aliased layout's member `i` holds attachment `i`'s texel bits at the member's own
/// offset (0, 4, 8, 16), not at a packed one (0, 2, 6, 14), and the staging keeps every bit of
/// every texel through its load and its store: half and float denormals, infinities, and NaN
/// payloads included (a float view would give them as values, and `OpQuantizeToF16` flushes a
/// half denormal).
#[test]
fn an_aliased_layout_reads_each_attachment_at_its_member() {
    let Some(bc) = assemble("k10ts_alias_probe") else {
        return;
    };
    let n = 256u32;
    let a0: Vec<u16> = (0..n).map(|p| probe_half(p, 0)).collect();
    let a1: Vec<u32> = (0..n).map(probe_f32).collect();
    let a2: Vec<[u16; 4]> = (0..n)
        .map(|p| [0, 1, 2, 3].map(|k: u16| probe_half(p, k + 1)))
        .collect();
    let a3: Vec<u16> = (0..n).map(|p| probe_half(n - 1 - p, 2)).collect();
    let inputs = tile_inputs(&[
        PixelFormat::R16Float,
        PixelFormat::R32Float,
        PixelFormat::RGBA16Float,
        PixelFormat::R16Float,
    ]);
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some((plan, m)) = module(&bc, "k10ts_alias_probe", &stem, &p, &inputs) else {
            continue;
        };
        // The interpreter's image holds one texel as four 32-bit components; the rgba16 plane is
        // checked one channel at a time through the first component.
        for channel in 0..4 {
            let d = Tiles {
                area: [16, 16],
                tile: [16, 16],
                threads: [16, 16],
                buffer: vec![0xAB; (n * 20) as usize],
                attachments: &[
                    a0.iter().map(|h| u32::from(*h)).collect(),
                    a1.clone(),
                    a2.iter().map(|c| u32::from(c[channel])).collect(),
                    a3.iter().map(|h| u32::from(*h)).collect(),
                ],
                nominal: None,
            };
            let (buf, planes) = run(&plan, &m, &d);
            let out = words(&buf);
            for p in 0..n as usize {
                let r = &out[p * 5..p * 5 + 5];
                assert_eq!(r[0], u32::from(a0[p]), "{stem} a {p}");
                assert_eq!(r[1], a1[p], "{stem} b {p}");
                // The interpreter's texel has one half value in the channel the test fills.
                let c = u32::from(a2[p][channel]);
                assert_eq!(r[2] & 0xFFFF, c, "{stem} c.x {p} channel {channel}");
                assert_eq!(r[4], u32::from(a3[p]), "{stem} d {p}");
            }
            // The function writes nothing through its cell pointer: the stored texels are the
            // loaded ones, bit for bit.
            assert_eq!(planes[0], d.attachments[0], "{stem}");
            assert_eq!(planes[1], a1, "{stem}");
            assert_eq!(planes[2], d.attachments[2], "{stem} channel {channel}");
            assert_eq!(planes[3], d.attachments[3], "{stem}");
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// A float attachment is bound through the unsigned-integer format of its channels and widths,
/// so the dispatch moves its texels' bits: Vulkan would decode a float view's texel to a value,
/// keeping neither a NaN's payload nor, in a shader that flushes its inputs, a denormal. The
/// pipeline records its aliased layout's size, a tile state's `imageblockSampleLength`.
#[test]
fn a_float_attachment_is_bound_through_its_bits() {
    let Some(bc) = assemble("k10ts_alias_probe") else {
        return;
    };
    let formats = [
        Some(PixelFormat::R16Float),
        Some(PixelFormat::R32Float),
        Some(PixelFormat::RGBA16Float),
        Some(PixelFormat::R16Float),
    ];
    let views = [
        vk::Format::R16_UINT,
        vk::Format::R32_UINT,
        vk::Format::R16G16B16A16_UINT,
        vk::Format::R16_UINT,
    ];
    let mut planned = 0;
    for stem in ["moltenvk-m3", "lavapipe", "amd-rdna2-apu-radv"] {
        let dev = device(stem);
        let d = descriptor(&bc, "k10ts_alias_probe", &formats);
        let pipe = match dev.compile_tile_pipeline(&d, &Budget::standard()) {
            Ok(p) => p,
            // A profile without the extended storage formats refuses the 16-bit views.
            Err(Error::Unsupported(u)) if u.feature == Feature::Format => continue,
            Err(e) => panic!("{stem}: {e}"),
        };
        // The aliased layout's size, a tile state's sample length whatever the formats pack to
        // (18 here), as Metal answers for this struct and these formats.
        assert!(
            pipe.dependencies
                .iter()
                .any(|x| x.fact == RuntimeFact::AliasedImageblock { data_size: 24 }),
            "{stem}"
        );
        let mut s = BindState::default();
        s.stages.insert(Stage::Kernel, StageTable::default());
        s.imageblock_size = ImageblockSize::new(16, 16);
        s.render_area = [16, 16];
        let plan = pipe.plan(&dev, &s.shape()).expect("plan");
        for (c, view) in views.into_iter().enumerate() {
            assert!(
                plan.descriptors.iter().any(|w| w.source
                    == WriteSource::TileAttachment {
                        color_index: c as u32,
                        format: view,
                        view: StorageAttachmentView::Attachment,
                        view_alias: None,
                    }),
                "{stem} {c}: {:?}",
                plan.descriptors
            );
        }
        planned += 1;
    }
    assert!(planned > 0);
}

/// A tile function's `[[threadgroup(n)]]` argument is its tile's threadgroup memory, which the
/// render pass keeps: the function compiles, recording the argument's fact, where a tile function
/// that declares a threadgroup variable is refused as Metal refuses its pipeline. The same kernels
/// compile as compute kernels.
#[test]
fn a_tile_functions_threadgroup_memory_is_its_tiles() {
    use crate::pipeline::testing::fixtures_dir;
    let p =
        crate::pipeline::testing::profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let dev = device("moltenvk-m3");
    let mut assembled = Vec::new();
    for (path, entry) in [
        (
            fixtures_dir().join("kernel_threadgroup_word.ll"),
            "threadgroup_word",
        ),
        (engine_dir().join("threadgroup_32k.ll"), "threadgroup_32k"),
    ] {
        match llvm_assemble(&path) {
            Ok(b) => assembled.push((b, entry)),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {entry}: {why}"),
        }
    }
    for (bc, entry) in &assembled {
        typed_module_with(bc, entry, &p, &LowerInputs::new())
            .unwrap_or_else(|e| panic!("{entry} as a compute kernel: {e}"));
    }
    let (word, word_entry) = &assembled[0];
    let (plan, _) = typed_module_with(word, word_entry, &p, &tile_inputs(&[]))
        .unwrap_or_else(|e| panic!("{word_entry} as a tile function: {e}"));
    assert!(plan
        .facts
        .contains(&RuntimeFact::TileThreadgroupMemory { index: 0 }));
    let pipe = dev
        .compile_tile_pipeline(&descriptor(word, word_entry, &[]), &Budget::standard())
        .unwrap_or_else(|e| panic!("{word_entry}: a tile pipeline: {e}"));
    assert!(pipe.layout.sets.iter().any(|s| s.bindings.iter().any(|b| {
        b.source
            == BindingSource::Synthetic {
                stage: Stage::Kernel,
                binding: SyntheticBinding::TileMemory,
            }
    })));
    let (variable, variable_entry) = &assembled[1];
    let refused = Err(Error::InvalidDescriptor(
        DescriptorError::TileThreadgroupVariable,
    ));
    assert_eq!(
        typed_module_with(variable, variable_entry, &p, &tile_inputs(&[])).map(|_| ()),
        refused
    );
    assert_eq!(
        dev.compile_tile_pipeline(
            &descriptor(variable, variable_entry, &[]),
            &Budget::standard()
        )
        .map(|_| ()),
        refused
    );
}

/// The tile memory buffer's slot (`SyntheticBinding::TileMemory`).
fn tile_memory_slot() -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Synthetic(SyntheticBinding::TileMemory),
        index: 0,
    }
}

/// One tile dispatch of a function of `cfit_tile_memory` over a pass's tile memory.
struct MemoryDispatch<'a> {
    /// The tiles across and down.
    grid: [u32; 2],
    threads: [u32; 2],
    /// Each tile's threadgroup memory's bytes, and each argument's offset by index.
    tile_length: u32,
    offsets: &'a [u32],
}

/// Run `m` as `d` over `memory`, every tile's threadgroup memory, in both scheduling orders
/// (which must agree): `memory` after, and `[[buffer(0)]]` of `out` bytes after.
fn run_memory(
    plan: &Plan,
    m: &TypedModule,
    d: &MemoryDispatch<'_>,
    memory: &[u8],
    out: usize,
) -> (Vec<u8>, Vec<u8>) {
    let mut spec = BTreeMap::new();
    spec.insert(SpecKey::LocalSize(Axis::X), u64::from(d.threads[0]));
    spec.insert(SpecKey::LocalSize(Axis::Y), u64::from(d.threads[1]));
    spec.insert(SpecKey::LocalSize(Axis::Z), 1);
    spec.insert(
        SpecKey::FlatLocalSize,
        u64::from(d.threads[0] * d.threads[1]),
    );
    spec.insert(SpecKey::TileThreadgroupLength, u64::from(d.tile_length));
    spec.insert(SpecKey::TileGridWidth, u64::from(d.grid[0]));
    for (index, &offset) in d.offsets.iter().enumerate() {
        spec.insert(SpecKey::TgOffset(index as u32), u64::from(offset));
    }
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            // One region, every buffer bound at its start.
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let buffers = vec![
        Buffer {
            slot: Some(tile_memory_slot()),
            address: None,
            bytes: memory.to_vec(),
        },
        Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0xAB; out],
        },
    ];
    let res = Resources {
        buffers,
        push,
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            order,
            workgroups: [d.grid[0], d.grid[1], 1],
            ..Config::default()
        };
        let (o, _) = run_with_images(m, &cfg, res.clone(), Vec::new()).expect("run");
        outs.push((o.buffers[0].clone(), o.buffers[1].clone()));
    }
    assert_eq!(outs[0], outs[1], "scheduling orders disagree");
    outs.swap_remove(0)
}

/// A pass's tile memory through `cfit_tile_memory`'s functions on the interpreter, 3x2 tiles of 4x4
/// threads and 128 bytes each: two dispatches of `add` leave twice its words in each tile's own
/// memory, `dump` reads them back by tile, and `shift` with its second argument 16 bytes on reads
/// and writes through overlapping windows, every other word kept.
#[test]
fn a_tiles_threadgroup_memory_persists_across_dispatches() {
    let Some(bc) = assemble("cfit_tile_memory") else {
        return;
    };
    const TILES: u32 = 6;
    const THREADS: u32 = 16;
    const WORDS: u32 = 32;
    let d = |offsets| MemoryDispatch {
        grid: [3, 2],
        threads: [4, 4],
        tile_length: WORDS * 4,
        offsets,
    };
    let added = |tile: u32, t: u32| 2 * (1000 * tile + t + 1);
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some((add_plan, add)) =
            module(&bc, "cfit_tile_memory_add", &stem, &p, &tile_inputs(&[]))
        else {
            continue;
        };
        let (dump_plan, dump) =
            module(&bc, "cfit_tile_memory_dump", &stem, &p, &tile_inputs(&[])).expect("dump");
        let (shift_plan, shift) =
            module(&bc, "cfit_tile_memory_shift", &stem, &p, &tile_inputs(&[])).expect("shift");
        for plan in [&add_plan, &dump_plan] {
            assert!(plan
                .facts
                .contains(&RuntimeFact::TileThreadgroupMemory { index: 0 }));
        }
        let zeros = vec![0u8; (TILES * WORDS * 4) as usize];
        let (memory, _) = run_memory(&add_plan, &add, &d(&[0]), &zeros, 0);
        let (memory, _) = run_memory(&add_plan, &add, &d(&[0]), &memory, 0);
        let out = (TILES * THREADS * 4) as usize;
        let (kept, dumped) = run_memory(&dump_plan, &dump, &d(&[0]), &memory, out);
        assert_eq!(kept, memory, "{stem}: dump wrote the tile memory");
        let want: Vec<u32> = (0..TILES)
            .flat_map(|tile| (0..THREADS).map(move |t| added(tile, t)))
            .collect();
        assert_eq!(words(&dumped), want, "{stem}: dump");
        let (shifted, _) = run_memory(&shift_plan, &shift, &d(&[0, 16]), &memory, 0);
        let want: Vec<u32> = (0..TILES)
            .flat_map(|tile| {
                (0..WORDS).map(move |w| match w {
                    0..4 => added(tile, w),
                    4..20 => added(tile, w - 4) + 0x10000,
                    _ => 0,
                })
            })
            .collect();
        assert_eq!(words(&shifted), want, "{stem}: shift");
        ran += 1;
    }
    assert!(ran > 0);
}

/// The binder places a tile function's threadgroup arguments where the encoder set them: it
/// specializes each offset, the tile's bytes (the pass's length rounded up to 16) and the tiles
/// across the render area, and binds the pass's tile memory of every tile's bytes. An argument's
/// length takes no part in the layout and bounds no access, as plain Metal's does not, so a length
/// that is not a multiple of 16 and one that ends past the tile's memory are both planned at their
/// offset (plain Metal on an M3 Max under macOS 27.0.1 takes every length, and reads zero from the
/// tile's memory's end on, which the arena's containment gives). Every offset is planned too, where
/// plain Metal begins the argument: at the offset floored to a multiple of 4, with no clamp, since
/// the containment reads zero past the tile's memory and Metal's own arithmetic wraps at 2^32. An
/// argument the encoder did not set is refused, and so is a tile memory past one storage-buffer
/// descriptor's range.
#[test]
fn a_tile_pipeline_places_its_threadgroup_arguments() {
    use crate::bind::{TileThreadgroupArgument, TileThreadgroupMemory};
    let Some(bc) = assemble("cfit_tile_memory") else {
        return;
    };
    let formats = [Some(PixelFormat::R32Uint)];
    let d = descriptor(&bc, "cfit_tile_memory_shift", &formats);
    let state = |length: u32, arguments: &[(u32, u32, u32)]| {
        let mut s = BindState::default();
        s.stages.insert(Stage::Kernel, StageTable::default());
        s.imageblock_size = ImageblockSize::new(16, 16);
        s.render_area = [40, 24];
        let mut memory = TileThreadgroupMemory::new(length);
        for &(index, length, offset) in arguments {
            memory
                .arguments
                .insert(index, TileThreadgroupArgument::new(length, offset));
        }
        s.tile_threadgroup = memory;
        s
    };
    for stem in ["moltenvk-m3", "lavapipe", "vk12-floor"] {
        let dev = device(stem);
        let pipe = dev
            .compile_tile_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        for index in [0, 1] {
            assert!(
                pipe.dependencies
                    .iter()
                    .any(|x| x.fact == RuntimeFact::TileThreadgroupMemory { index }),
                "{stem} {index}"
            );
        }
        let plan = pipe
            .plan(&dev, &state(72, &[(0, 64, 0), (1, 48, 16)]).shape())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(specialized(&pipe, &plan, SpecKey::TgOffset(0)), Some(0));
        assert_eq!(specialized(&pipe, &plan, SpecKey::TgOffset(1)), Some(16));
        assert_eq!(
            specialized(&pipe, &plan, SpecKey::TileThreadgroupLength),
            Some(80)
        );
        assert_eq!(specialized(&pipe, &plan, SpecKey::TileGridWidth), Some(3));
        assert!(
            plan.descriptors
                .iter()
                .any(|w| w.source == WriteSource::TileMemory { bytes: 6 * 80 }),
            "{stem}"
        );
        let refused = |arguments: &[(u32, u32, u32)]| {
            pipe.plan(&dev, &state(72, arguments).shape()).map(|_| ())
        };
        assert_eq!(
            refused(&[(0, 64, 0)]),
            Err(BindError::MissingThreadgroupMemory { index: 1 }),
            "{stem}"
        );
        // A length that is not a multiple of 16, and an argument that ends past the tile's
        // memory, are planned at their offset: plain Metal takes either.
        for (length, offset) in [(56, 16), (64, 16), (1, 64), (0, 0), (u32::MAX, 32)] {
            let plan = pipe
                .plan(&dev, &state(72, &[(0, 64, 0), (1, length, offset)]).shape())
                .unwrap_or_else(|e| panic!("{stem}: {length} at {offset}: {e}"));
            assert_eq!(
                specialized(&pipe, &plan, SpecKey::TgOffset(1)),
                Some(offset),
                "{stem} {length} at {offset}"
            );
            assert_eq!(
                specialized(&pipe, &plan, SpecKey::TileThreadgroupLength),
                Some(80),
                "{stem} {length} at {offset}"
            );
        }
        // Every offset is placed where plain Metal begins the argument: at the offset floored to
        // a multiple of 4, and nothing is clamped, since an access from a base past the tile's
        // memory is contained, reads zero and drops its store, and Metal's own arithmetic wraps at
        // 2^32 rather than reading zero for ever. The offsets below are the ones the planner used
        // to refuse; `(2, 0)`, `(20, 20)` and `(70, 68)` are what a floor to 16 would place
        // differently from a floor to 4.
        for (offset, at) in [
            (8, 8),
            (4, 4),
            (2, 0),
            (17, 16),
            (20, 20),
            (70, 68),
            (72, 72),
            (80, 80),
            (1 << 20, 1 << 20),
            (u32::MAX, u32::MAX - 3),
        ] {
            let plan = pipe
                .plan(&dev, &state(72, &[(0, 64, 0), (1, 16, offset)]).shape())
                .unwrap_or_else(|e| panic!("{stem}: 16 at {offset}: {e}"));
            assert_eq!(
                specialized(&pipe, &plan, SpecKey::TgOffset(1)),
                Some(at),
                "{stem} 16 at {offset}"
            );
        }
    }
    // Every tile's memory is one storage-buffer descriptor.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/lavapipe.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    desc.limits.max_storage_buffer_range = 256;
    let dev = Device::new(desc, DevicePolicy::default()).expect("device");
    let pipe = dev
        .compile_tile_pipeline(&d, &Budget::standard())
        .expect("a tile pipeline");
    match pipe.plan(&dev, &state(80, &[(0, 64, 0), (1, 64, 16)]).shape()) {
        Err(BindError::Unsupported(u)) => assert_eq!(u.feature, Feature::TileShading),
        other => panic!(
            "a tile memory past the range planned: {:?}",
            other.map(|_| ())
        ),
    }
}

fn device(stem: &str) -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

fn descriptor(
    bc: &[u8],
    entry: &str,
    formats: &[Option<PixelFormat>],
) -> TileRenderPipelineDescriptor {
    let lib = Library::from_air(bc).expect("library");
    let f = lib.function(entry).expect("function");
    let mut d =
        TileRenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    for (i, pf) in formats.iter().enumerate() {
        d.color_attachment_formats[i] = *pf;
    }
    d
}

/// The value `plan` specializes `key` to.
fn specialized(pipe: &ComputePipeline, plan: &BindingPlan, key: SpecKey) -> Option<u32> {
    let id = pipe
        .stage
        .specialization
        .constants
        .iter()
        .find(|c| c.key == key)?
        .id;
    let e = plan
        .specialization
        .entries
        .iter()
        .find(|e| e.constant_id == id)?;
    let at = e.offset as usize;
    Some(u32::from_le_bytes(
        plan.specialization.data[at..at + 4]
            .try_into()
            .expect("u32"),
    ))
}

/// `Device::compile_tile_pipeline` compiles the tile function under the descriptor's formats;
/// its binding plan specializes the tile size and the render area, binds each attachment as
/// `WriteSource::TileAttachment` in its staging format (a float attachment through the
/// unsigned-integer format of its channels), refuses an empty render area and a layered pass,
/// and answers `maxTotalThreadsPerThreadgroup` with Metal's 1024 where the device runs it, the
/// largest tile threadgroup it runs, and records no aliased imageblock, so a tile state's `imageblockSampleLength` is its
/// attachments' texels, packed. The key is the descriptor's.
#[test]
fn a_tile_pipeline_binds_its_attachments() {
    let Some(bc) = assemble("k10ts_tile_grid") else {
        return;
    };
    let formats = [
        Some(PixelFormat::R32Float),
        Some(PixelFormat::R32Uint),
        Some(PixelFormat::R32Uint),
    ];
    for stem in ["moltenvk-m3", "lavapipe", "vk12-floor"] {
        let dev = device(stem);
        let d = descriptor(&bc, "k10ts_tile_grid", &formats);
        let pipe = dev
            .compile_tile_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(pipe.key, dev.tile_pipeline_key(&d), "{stem}");
        assert!(pipe
            .dependencies
            .iter()
            .any(|x| x.fact == RuntimeFact::TileFunction));
        let sources: Vec<BindingSource> = pipe
            .layout
            .sets
            .iter()
            .flat_map(|s| s.bindings.iter().map(|b| b.source))
            .collect();
        for c in 0..3 {
            assert!(
                sources.contains(&BindingSource::TileAttachment(c)),
                "{stem} {c}"
            );
        }
        let state = |layers: u32| {
            let mut s = BindState::default();
            s.stages.insert(Stage::Kernel, StageTable::default());
            s.imageblock_size = ImageblockSize::new(16, 16);
            s.render_area = [40, 24];
            s.render_target_array_length = layers;
            s
        };
        let plan = pipe.plan(&dev, &state(0).shape()).expect("plan");
        assert_eq!(
            specialized(&pipe, &plan, SpecKey::ImageblockWidth),
            Some(16)
        );
        assert_eq!(
            specialized(&pipe, &plan, SpecKey::RenderAreaWidth),
            Some(40)
        );
        assert_eq!(
            specialized(&pipe, &plan, SpecKey::RenderAreaHeight),
            Some(24)
        );
        // The float attachment's bits move through its `uint` view; the others are `uint`.
        for c in 0..3 {
            assert!(
                plan.descriptors.iter().any(|w| w.source
                    == WriteSource::TileAttachment {
                        color_index: c,
                        format: vk::Format::R32_UINT,
                        view: StorageAttachmentView::Attachment,
                        view_alias: None,
                    }),
                "{stem} {c}"
            );
        }
        match pipe.plan(&dev, &state(2).shape()) {
            Err(BindError::Unsupported(u)) => assert_eq!(u.feature, Feature::TileShading),
            other => panic!("{stem}: a layered pass planned: {:?}", other.map(|_| ())),
        }
        // A tile dispatch is one threadgroup per tile of a threadgroup whose layout is measured.
        let tiles = |threads_per_group: [u32; 3]| Dispatch::Threadgroups {
            groups: [3, 2, 1],
            threads_per_group,
        };
        // A measured threadgroup runs where the device's local-size limits take it.
        for [w, h] in [
            [1, 1],
            [2, 2],
            [4, 4],
            [8, 8],
            [16, 16],
            [32, 32],
            [8, 4],
            [4, 2],
            [8, 2],
            [2, 32],
            [32, 4],
            [16, 32],
            [32, 16],
            [16, 2],
            [32, 2],
            [6, 6],
            [24, 24],
            [30, 2],
            [2, 14],
        ] {
            match plan.dispatch_variant(&tiles([w, h, 1])) {
                Ok(_) => {}
                Err(BindError::Unsupported(u)) => {
                    assert!(
                        matches!(u.feature, Feature::LocalSize { .. }),
                        "{stem} {w}x{h}: {u:?}"
                    )
                }
                Err(e) => panic!("{stem} {w}x{h}: {e}"),
            }
        }
        // A threadgroup with an odd side leaves the lanes of its missing threads empty, which
        // is not built, and a tile threadgroup is one deep.
        for refused in [
            tiles([16, 1, 1]),
            tiles([8, 1, 1]),
            tiles([1, 8, 1]),
            tiles([3, 3, 1]),
            tiles([5, 2, 1]),
            tiles([2, 31, 1]),
            tiles([8, 8, 2]),
            tiles([64, 16, 1]),
            Dispatch::Threads {
                threads: [40, 24, 1],
                threads_per_group: [8, 8, 1],
            },
        ] {
            match plan.dispatch_variant(&refused) {
                Err(BindError::Unsupported(u)) => {
                    assert_eq!(
                        (u.feature, u.needs),
                        (Feature::TileShading, Needs::NotImplemented),
                        "{stem} {refused:?}"
                    )
                }
                other => panic!("{stem}: {refused:?} dispatched: {other:?}"),
            }
        }
        // A tile dispatch outside a render pass (the encoder never set the area) has no tile.
        for area in [[0, 0], [40, 0], [0, 24]] {
            let mut s = state(0);
            s.render_area = area;
            assert_eq!(
                pipe.plan(&dev, &s.shape()).map(|_| ()),
                Err(BindError::InvalidRenderArea {
                    width: area[0],
                    height: area[1],
                }),
                "{stem} {area:?}"
            );
        }
        // An implicit imageblock aliases no layout: a tile state's sample length is the
        // attachments' texels, packed.
        assert!(
            !pipe
                .dependencies
                .iter()
                .any(|x| matches!(x.fact, RuntimeFact::AliasedImageblock { .. })),
            "{stem}"
        );
        // Metal's 1024, the 32x32 threadgroup the binder runs, where the
        // device's limits take it.
        let max = pipe.metal_info.max_total_threads_per_threadgroup;
        assert!(max <= 1024, "{stem}: {max}");
        if stem == "moltenvk-m3" {
            assert_eq!(max, 1024);
        }
        // Metal's own answer, before the device's limits: 1024 on every profile.
        assert_eq!(
            pipe.metal_info.metal_max_total_threads_per_threadgroup, 1024,
            "{stem}"
        );
        // The descriptor's limit lowers it, never raises it, and is part of the key; Metal's 0 is
        // no limit. A limit answers the most threads of a threadgroup of even sides within it, as
        // Metal's does (on an M3 Max and an M4: 1000 answers 960, 255 252, 3 none).
        for (limit, metal) in [
            (Some(64), 64),
            (Some(1024), 1024),
            (Some(0), 1024),
            (Some(1000), 960),
            (Some(255), 252),
            (Some(3), 0),
        ] {
            let expected = metal.min(max);
            let mut limited = d.clone();
            limited.max_total_threads_per_threadgroup = limit;
            let pipe = dev
                .compile_tile_pipeline(&limited, &Budget::standard())
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert_eq!(
                pipe.metal_info.max_total_threads_per_threadgroup, expected,
                "{stem} {limit:?}"
            );
            assert_eq!(
                pipe.metal_info.metal_max_total_threads_per_threadgroup, metal,
                "{stem} {limit:?}: Metal's own"
            );
            assert_ne!(dev.tile_pipeline_key(&limited), dev.tile_pipeline_key(&d));
        }
    }
}

/// The refusals are typed: a tile function compiled without tile inputs needs them, a format
/// whose texel converts into a member of another type (a `float` over `RGBA8Unorm`) is not
/// implemented, an attachment its layout holds without a format, or
/// a `Compiled` tile function's attachment the descriptor gives another format than it was
/// compiled for, is the descriptor's error, and a multisampled pass is not implemented.
#[test]
fn tile_refusals_are_typed() {
    let Some(bc) = assemble("k10ts_tile_grid") else {
        return;
    };
    let dev = device("moltenvk-m3");
    let b = Budget::standard();
    let lib = Library::from_air(&bc).expect("library");
    let f = lib.function("k10ts_tile_grid").expect("function");
    let compute = crate::pipeline::ComputePipelineDescriptor::new(StageSource::Function(
        FunctionDescriptor::new(f),
    ));
    assert_eq!(
        dev.compile_compute_pipeline(&compute, &b).map(|_| ()),
        Err(Error::NeedsInput(RuntimeInput::TileAttachments))
    );
    let not_implemented = |r: Result<ComputePipeline, Error>| match r {
        Err(Error::Unsupported(u)) => {
            u.feature == Feature::TileShading && u.needs == Needs::NotImplemented
        }
        _ => false,
    };
    let converting = descriptor(
        &bc,
        "k10ts_tile_grid",
        &[
            Some(PixelFormat::RGBA8Unorm),
            Some(PixelFormat::R32Uint),
            Some(PixelFormat::R32Uint),
        ],
    );
    assert!(not_implemented(dev.compile_tile_pipeline(&converting, &b)));
    let missing = descriptor(
        &bc,
        "k10ts_tile_grid",
        &[
            Some(PixelFormat::R32Float),
            None,
            Some(PixelFormat::R32Uint),
        ],
    );
    assert_eq!(
        dev.compile_tile_pipeline(&missing, &b).map(|_| ()),
        Err(Error::InvalidDescriptor(DescriptorError::ColorAttachment {
            index: 1
        }))
    );
    let mut msaa = descriptor(
        &bc,
        "k10ts_tile_grid",
        &[
            Some(PixelFormat::R32Float),
            Some(PixelFormat::R32Uint),
            Some(PixelFormat::R32Uint),
        ],
    );
    msaa.raster_sample_count = 4;
    assert!(not_implemented(dev.compile_tile_pipeline(&msaa, &b)));
    let compiled = dev
        .compile_function(
            &FunctionDescriptor::new(lib.function("k10ts_tile_grid").expect("function")),
            &tile_inputs(&[
                PixelFormat::R32Float,
                PixelFormat::R32Uint,
                PixelFormat::R32Uint,
            ]),
            &b,
        )
        .expect("compiled tile function");
    let with = |formats: [PixelFormat; 3]| {
        let mut d = TileRenderPipelineDescriptor::new(StageSource::Compiled(compiled.clone()));
        for (i, pf) in formats.into_iter().enumerate() {
            d.color_attachment_formats[i] = Some(pf);
        }
        dev.compile_tile_pipeline(&d, &b).map(|_| ())
    };
    assert_eq!(
        with([
            PixelFormat::R32Float,
            PixelFormat::R32Uint,
            PixelFormat::R32Uint
        ]),
        Ok(())
    );
    assert_eq!(
        with([
            PixelFormat::R32Float,
            PixelFormat::R32Sint,
            PixelFormat::R32Uint
        ]),
        Err(Error::InvalidDescriptor(DescriptorError::ColorAttachment {
            index: 1
        }))
    );
}

/// The authored tile fixtures (`validation/fixtures/public/kernel_*imageblock*`, whose cases Metal
/// qualified) compile under their attachments' formats on every committed profile, or are refused
/// typed where the profile lacks the storage format.
#[test]
fn the_authored_tile_fixtures_compile() {
    use crate::pipeline::testing::fixtures_dir;
    let cases: [(&str, &[PixelFormat]); 5] = [
        (
            "kernel_implicit_imageblock_half2",
            &[PixelFormat::RG16Float],
        ),
        (
            "kernel_implicit_imageblock_half4",
            &[PixelFormat::RGBA16Float],
        ),
        (
            "kernel_implicit_imageblock_three_widths",
            &[
                PixelFormat::R32Float,
                PixelFormat::R16Float,
                PixelFormat::R32Uint,
            ],
        ),
        ("kernel_imageblock_extent", &[PixelFormat::RGBA16Float]),
        (
            "kernel_aliased_imageblock_planes",
            &[
                PixelFormat::RGBA16Float,
                PixelFormat::RGBA16Float,
                PixelFormat::R16Float,
            ],
        ),
    ];
    for (stem, formats) in cases {
        let path = fixtures_dir().join(format!("{stem}.ll"));
        let bc = match llvm_assemble(&path) {
            Ok(b) => b,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}: {why}"),
        };
        let inputs = tile_inputs(formats);
        let mut compiled = 0;
        for (profile, p) in committed_profiles().expect("profiles") {
            if let Some((plan, _)) = module(&bc, stem, &profile, &p, &inputs) {
                assert!(plan.facts.contains(&RuntimeFact::TileFunction), "{stem}");
                compiled += 1;
            }
        }
        assert!(compiled > 0, "{stem}");
    }
}

/// Metal's quad order of a tile threadgroup of `qw` by `qh` 2x2 quads (measured on an Apple M3 Max
/// for every threadgroup of 1 to 32 threads a side): columns 8 quads wide, within a column its rows
/// of 2 quads, within a row its blocks 4 quads wide, within a block its quads row-major.
fn metal_quad_order(qw: u32, qh: u32) -> Vec<[u32; 2]> {
    let mut order = Vec::new();
    for c0 in (0..qw).step_by(8) {
        let c1 = (c0 + 8).min(qw);
        for r0 in (0..qh).step_by(2) {
            let r1 = (r0 + 2).min(qh);
            for b0 in (c0..c1).step_by(4) {
                let b1 = (b0 + 4).min(c1);
                for qy in r0..r1 {
                    for qx in b0..b1 {
                        order.push([qx, qy]);
                    }
                }
            }
        }
    }
    order
}

/// Metal's tile-dispatch layout of a `w x h` threadgroup of even sides (or the single thread):
/// the position of `thread_index_in_threadgroup` `index` (the threadgroup's 2x2 quads in rows of
/// `w / 2`, each quad's threads in rows).
fn metal_tile_position(index: u32, [w, _]: [u32; 2]) -> [u32; 2] {
    let (quad, within) = (index / 4, index % 4);
    let quads = w.div_ceil(2);
    [
        (quad % quads) * 2 + within % 2,
        (quad / quads) * 2 + within / 2,
    ]
}

/// The simdgroup and lane Metal gives the thread at `pos` of a `w x h` threadgroup: its quad's
/// place `k` in [`metal_quad_order`] is lanes `4 (k mod 8)` to `4 (k mod 8) + 3` of simdgroup
/// `k / 8`, its place in the quad (row-major) the lane among them.
fn metal_tile_simd(pos: [u32; 2], [w, h]: [u32; 2]) -> (u32, u32) {
    let quad = [pos[0] / 2, pos[1] / 2];
    let k = metal_quad_order(w.div_ceil(2), h.div_ceil(2))
        .iter()
        .position(|&q| q == quad)
        .expect("a quad of the threadgroup") as u32;
    (k / 8, (k % 8) * 4 + (pos[1] % 2) * 2 + pos[0] % 2)
}

/// A tile function's threads take Metal's tile-dispatch layout for the threadgroups the binder runs
/// (squares of 1 to 32 threads a side, in a 16x16 or 32x32 tile, and rectangles of even sides to
/// 32x32, among them the shapes whose simdgroups span a whole row of quads or a column's partial
/// rows and blocks): each thread records, at its `thread_index_in_threadgroup`, the position, lane, and
/// simdgroup Metal gives that index and the position of its simdgroup's first lane
/// (`simd_broadcast_first`), and adds one to its own cell. Runs on the `Direct` tier at 32 lanes
/// and on every committed profile whose simdgroups are emulated.
#[test]
fn every_tile_thread_takes_its_measured_place() {
    let Some(bc) = assemble("k10ts_tile_order") else {
        return;
    };
    let inputs = tile_inputs(&[PixelFormat::R32Float]);
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some((plan, m)) = module(&bc, "k10ts_tile_order", &stem, &p, &inputs) else {
            continue;
        };
        if !matches!(
            plan.simd.1,
            crate::device::subgroup::Tier::Direct | crate::device::subgroup::Tier::Emulated
        ) {
            continue;
        }
        for (tile, [w, h]) in [
            (16, [1, 1]),
            (16, [2, 2]),
            (16, [4, 4]),
            (16, [8, 8]),
            (16, [16, 16]),
            (32, [16, 16]),
            (32, [32, 32]),
            (32, [32, 16]),
            (32, [16, 32]),
            (32, [32, 4]),
            (32, [4, 32]),
            (32, [2, 32]),
            (16, [8, 2]),
            (16, [4, 2]),
            (16, [16, 4]),
            (32, [16, 2]),
            (32, [32, 2]),
            (16, [6, 6]),
            (16, [10, 10]),
            (32, [24, 8]),
            (32, [2, 18]),
            (32, [32, 6]),
            (16, [14, 4]),
        ] {
            let n = w * h;
            let f0: Vec<u32> = (0..tile * tile).map(|i| (i as f32).to_bits()).collect();
            let d = Tiles {
                area: [tile, tile],
                tile: [tile, tile],
                threads: [w, h],
                buffer: vec![0xAB; (n * 8) as usize],
                attachments: std::slice::from_ref(&f0),
                nominal: None,
            };
            let (buf, planes) = run(&plan, &m, &d);
            let out = words(&buf);
            for index in 0..n {
                let pos = metal_tile_position(index, [w, h]);
                let (sg, lane) = metal_tile_simd(pos, [w, h]);
                let first = metal_tile_position(
                    (0..n)
                        .find(|&i| {
                            metal_tile_simd(metal_tile_position(i, [w, h]), [w, h]) == (sg, 0)
                        })
                        .expect("a first lane"),
                    [w, h],
                );
                let at = (index * 2) as usize;
                assert_eq!(
                    out[at..at + 2],
                    [
                        pos[0] | (pos[1] << 8) | (lane << 16) | (sg << 24),
                        first[0] | (first[1] << 8)
                    ],
                    "{stem} {w}x{h} in {tile}: index {index}"
                );
            }
            for y in 0..tile {
                for x in 0..tile {
                    let i = (y * tile + x) as usize;
                    let touched = x < w && y < h;
                    let want = if touched {
                        (f32::from_bits(f0[i]) + 1.0).to_bits()
                    } else {
                        f0[i]
                    };
                    assert_eq!(planes[0][i], want, "{stem} {w}x{h} pixel ({x}, {y})");
                }
            }
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// Under `threadgroupSizeMatchesTileSize` a tile function's `thread_index_in_threadgroup` is
/// `simdgroup · 32 + lane`, not the quad order across the whole width, while each thread keeps
/// its measured place, lane and simdgroup (an M3 Max and an M4): the record at
/// each index holds the thread whose simdgroup and lane give it, for threadgroups 16 and 32
/// wide, where the two orders differ, and 8x8 in a 16x16 tile, where they agree.
#[test]
fn a_tile_sized_threadgroup_numbers_its_threads_by_simdgroup() {
    let Some(bc) = assemble("k10ts_tile_order") else {
        return;
    };
    let mut inputs = tile_inputs(&[PixelFormat::R32Float]);
    inputs.threadgroup_size_matches_tile_size = true;
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some((plan, m)) = module(&bc, "k10ts_tile_order", &stem, &p, &inputs) else {
            continue;
        };
        if !matches!(
            plan.simd.1,
            crate::device::subgroup::Tier::Direct | crate::device::subgroup::Tier::Emulated
        ) {
            continue;
        }
        assert!(plan
            .facts
            .contains(&RuntimeFact::TileThreadgroup { tile_sized: true }));
        for (tile, [w, h]) in [(16, [8, 8]), (16, [16, 16]), (32, [16, 16]), (32, [32, 16])] {
            let n = w * h;
            let f0: Vec<u32> = (0..tile * tile).map(|i| (i as f32).to_bits()).collect();
            let d = Tiles {
                area: [tile, tile],
                tile: [tile, tile],
                threads: [w, h],
                buffer: vec![0xAB; (n * 8) as usize],
                attachments: std::slice::from_ref(&f0),
                nominal: Some([tile, tile]),
            };
            let (buf, _) = run(&plan, &m, &d);
            let out = words(&buf);
            let mut first = vec![None; (n / 32).max(1) as usize];
            let mut places = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    let (sg, lane) = metal_tile_simd([x, y], [w, h]);
                    if lane == 0 {
                        first[sg as usize] = Some([x, y]);
                    }
                    places.push(([x, y], sg, lane));
                }
            }
            for ([x, y], sg, lane) in places {
                let index = sg * 32 + lane;
                let at = (index * 2) as usize;
                let head = first[sg as usize].expect("a first lane");
                assert_eq!(
                    out[at..at + 2],
                    [
                        x | (y << 8) | (lane << 16) | (sg << 24),
                        head[0] | (head[1] << 8)
                    ],
                    "{stem} {w}x{h} in {tile}: index {index}"
                );
            }
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// A tile function's explicit imageblock that stands alone is the pass's imageblock storage: each
/// function of `cfit_tile_imageblock` that reaches its cells compiles on every committed profile
/// that compiles a tile function, records its size and alignment
/// (`RuntimeFact::TileImageblock`), and plans the storage at `SyntheticBinding::Imageblock`
/// (`WriteSource::Imageblock`) beside no staged attachment; the layout aliased onto the
/// attachment records neither and stages the attachment.
#[test]
fn a_tile_functions_explicit_imageblock_alone_is_the_pass_storage() {
    let Some(bc) = assemble("cfit_tile_imageblock") else {
        return;
    };
    let formats = [Some(PixelFormat::R32Uint)];
    let mut planned = 0;
    for stem in ["moltenvk-m3", "lavapipe", "amd-rdna2-apu-radv"] {
        let dev = device(stem);
        let mut s = BindState::default();
        s.stages.insert(Stage::Kernel, StageTable::default());
        s.imageblock_size = ImageblockSize::new(16, 16);
        s.render_area = [40, 24];
        for entry in [
            "cfit_ib_tile_clear",
            "cfit_ib_tile_add",
            "cfit_ib_tile_resolve",
            "cfit_ib_tile_resolve_halves",
        ] {
            let d = descriptor(&bc, entry, &formats);
            let pipe = dev
                .compile_tile_pipeline(&d, &Budget::standard())
                .unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            assert!(
                pipe.dependencies.iter().any(|x| x.fact
                    == RuntimeFact::TileImageblock {
                        data_size: 8,
                        alignment: 4,
                    }),
                "{stem} {entry}"
            );
            assert!(
                !pipe
                    .dependencies
                    .iter()
                    .any(|x| matches!(x.fact, RuntimeFact::TileAttachmentFormat { .. })),
                "{stem} {entry}"
            );
            let plan = pipe.plan(&dev, &s.shape()).expect("plan");
            let sources: Vec<_> = plan.descriptors.iter().map(|w| &w.source).collect();
            assert!(
                sources.contains(&&WriteSource::Imageblock),
                "{stem} {entry}: {sources:?}"
            );
            assert!(
                !sources
                    .iter()
                    .any(|w| matches!(w, WriteSource::TileAttachment { .. })),
                "{stem} {entry}: {sources:?}"
            );
            planned += 1;
        }
        let d = descriptor(&bc, "cfit_ib_tile_aliased", &formats);
        let pipe = dev
            .compile_tile_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem} aliased: {e}"));
        assert!(
            !pipe
                .dependencies
                .iter()
                .any(|x| matches!(x.fact, RuntimeFact::TileImageblock { .. })),
            "{stem}"
        );
        let plan = pipe.plan(&dev, &s.shape()).expect("plan");
        assert!(
            !plan
                .descriptors
                .iter()
                .any(|w| w.source == WriteSource::Imageblock),
            "{stem}"
        );
    }
    assert_eq!(planned, 12);
}

/// A tile function's explicit imageblock standing alone in a pass that keeps fewer of its cells'
/// bytes: the plan of a shape stating them (`BindShape::kept_imageblock_bytes`) asks for the
/// variant that keeps them where they are fewer than the block's data, and none otherwise; the
/// function compiles under them (`LoweringInputs::kept_imageblock_bytes`) with its plan's cells
/// keeping them, and kept bytes that are not a cell's first whole words, or given to a function
/// whose cells are not the pass's imageblock storage, are refused.
#[test]
fn a_pass_keeping_fewer_imageblock_bytes_asks_for_the_kept_variant() {
    let Some(bc) = assemble("cfit_tile_imageblock") else {
        return;
    };
    let formats = [Some(PixelFormat::R32Uint)];
    for stem in ["moltenvk-m3", "lavapipe"] {
        let dev = device(stem);
        let entry = "cfit_ib_tile_clear";
        let d = descriptor(&bc, entry, &formats);
        let pipe = dev
            .compile_tile_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let mut s = BindState::default();
        s.stages.insert(Stage::Kernel, StageTable::default());
        s.imageblock_size = ImageblockSize::new(16, 16);
        s.render_area = [40, 24];
        // The eight-byte cell: four bytes kept ask for the variant; eight, or none, do not.
        for (kept, wanted) in [
            (Some(4), Some(4)),
            (Some(0), Some(0)),
            (Some(8), None),
            (None, None),
        ] {
            s.kept_imageblock_bytes = kept;
            let plan = pipe.plan(&dev, &s.shape()).expect("plan");
            assert_eq!(
                plan.variant.kept_imageblock_bytes, wanted,
                "{stem} {kept:?}"
            );
        }
        let mut inputs = tile_inputs(&[PixelFormat::R32Uint]);
        inputs.kept_imageblock_bytes = Some(4);
        let (plan, _) = typed_module_with(&bc, entry, dev.profile(), &inputs)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let tile = plan.families.access.tile.as_ref().expect("the tile plan");
        assert_eq!(tile.kept_bytes, Some(4), "{stem}");
        crate::pipeline::testing::compile_with(&bc, entry, dev.profile(), &inputs)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        // Not a cell's first whole words.
        for kept in [2, 8] {
            inputs.kept_imageblock_bytes = Some(kept);
            assert!(
                matches!(
                    typed_module_with(&bc, entry, dev.profile(), &inputs),
                    Err(Error::Internal(_))
                ),
                "{stem} {kept}"
            );
        }
        // The layout aliased onto the attachment, and a kernel without tile inputs.
        inputs.kept_imageblock_bytes = Some(4);
        assert!(
            matches!(
                typed_module_with(&bc, "cfit_ib_tile_aliased", dev.profile(), &inputs),
                Err(Error::Internal(_))
            ),
            "{stem} aliased"
        );
        let mut plain = LowerInputs::new();
        plain.kept_imageblock_bytes = Some(4);
        assert!(
            matches!(
                typed_module_with(&bc, entry, dev.profile(), &plain),
                Err(Error::Internal(_))
            ),
            "{stem} without tile inputs"
        );
    }
}

/// A channel of a normalized format: an unsigned or signed normalized one of its width, or an
/// sRGB colour channel of 8 bits.
#[derive(Clone, Copy, Debug)]
enum Channel {
    Unorm(u32),
    Snorm(u32),
    Srgb,
}

impl Channel {
    fn bits(self) -> u32 {
        match self {
            Channel::Unorm(bits) | Channel::Snorm(bits) => bits,
            Channel::Srgb => 8,
        }
    }

    /// The value Metal reads of code `c` (probes): the quotient by the code's scale correctly
    /// rounded (exact in `f64`, then rounded once: no such quotient lies near a tie of `f32`), an
    /// snorm one at least -1; an sRGB code's 12-bit linear value over 4095.
    fn value(self, c: u32) -> f32 {
        match self {
            Channel::Unorm(bits) => (f64::from(c) / f64::from((1u32 << bits) - 1)) as f32,
            Channel::Snorm(bits) => {
                let signed = ((c << (32 - bits)) as i32) >> (32 - bits);
                let scale = f64::from((1u32 << (bits - 1)) - 1);
                (f64::from(signed) / scale).max(-1.0) as f32
            }
            Channel::Srgb => (f64::from(SRGB_RESOLVE_LINEAR[c as usize]) / 4095.0) as f32,
        }
    }

    /// The code Metal stores of `x` (probes 9b to 9d): `x` clamped to `[0, 1]` (`[-1, 1]` for
    /// snorm), a NaN 0, times the scale exactly, rounded to nearest even; an sRGB channel's 12-bit
    /// index so, encoded by Metal's thresholds.
    fn code(self, x: f32) -> u32 {
        let x = if x.is_nan() { 0.0 } else { f64::from(x) };
        match self {
            Channel::Unorm(bits) => {
                (x.clamp(0.0, 1.0) * f64::from((1u32 << bits) - 1)).round_ties_even() as u32
            }
            Channel::Snorm(bits) => {
                let scale = f64::from((1u32 << (bits - 1)) - 1);
                let c = (x.clamp(-1.0, 1.0) * scale).round_ties_even() as i32;
                (c as u32) & ((1u32 << bits) - 1)
            }
            Channel::Srgb => {
                let index = (x.clamp(0.0, 1.0) * 4095.0).round_ties_even() as u16;
                SRGB_RESOLVE_THRESHOLDS.partition_point(|&t| t <= index) as u32
            }
        }
    }

    /// The exact value of code `c` ([`Channel::value`]'s before its rounding to `f32`) as an
    /// `f64`: `c / D` correctly rounded, which no `half` tie lies within an `f64` step of (the
    /// grids are odd, so no quotient but 0 and 1 is dyadic, and each is at least `2^-51` from a
    /// `half` tie).
    fn exact(self, c: u32) -> f64 {
        match self {
            Channel::Unorm(bits) => f64::from(c) / f64::from((1u32 << bits) - 1),
            Channel::Snorm(bits) => {
                let signed = ((c << (32 - bits)) as i32) >> (32 - bits);
                let scale = f64::from((1u32 << (bits - 1)) - 1);
                (f64::from(signed) / scale).max(-1.0)
            }
            Channel::Srgb => f64::from(SRGB_RESOLVE_LINEAR[c as usize]) / 4095.0,
        }
    }

    /// The `half` Metal reads of code `c` through a `half` member, as its `f32` (a probe, an M4 and
    /// an M3 Max): the exact value correctly rounded to nearest even once.
    fn half(self, c: u32) -> f32 {
        f32::from_bits(half_bits_to_f32(f64_to_half_bits(self.exact(c))))
    }
}

/// Whether `have`, the device's conversion's value of a `channel` read (through a `half` member
/// where `half`), lies within `PrecisionNote::DevicePackConversion`'s bound of Metal's `want`: 3
/// `f32` units in the last place (2.5 of the quotient), and one `half` step more through a `half`
/// member. A lane past the channels holds its constant exactly.
fn near(channel: Option<Channel>, have: f32, want: f32, half: bool) -> bool {
    let Some(channel) = channel else {
        return have.to_bits() == want.to_bits();
    };
    let ulp = |x: f32| f32::from_bits(x.abs().to_bits() + 1) - x.abs();
    let half_step = |x: f32| match x.abs() {
        a if a < f32::powi(2.0, -14) => f32::powi(2.0, -24),
        a => f32::powi(2.0, (a.to_bits() >> 23) as i32 - 127 - 10),
    };
    let mut bound = match channel {
        Channel::Srgb => 1.0 / 8190.0 + f32::powi(2.0, -14),
        Channel::Unorm(_) | Channel::Snorm(_) => 3.0 * ulp(want),
    };
    if half {
        bound += half_step(want);
    }
    (have - want).abs() <= bound
}

/// Whether code `a` of `channel` lies within one of code `b` (a packed code's bound under
/// `PrecisionNote::DevicePackConversion`; signed codes as their values).
fn code_near(channel: Channel, a: u32, b: u32) -> bool {
    match channel {
        Channel::Snorm(bits) => {
            let signed = |c: u32| i64::from(((c << (32 - bits)) as i32) >> (32 - bits));
            (signed(a) - signed(b)).abs() <= 1
        }
        Channel::Unorm(_) | Channel::Srgb => a.abs_diff(b) <= 1,
    }
}

/// Whether the interpreter runs `fmt`'s conversions: an sRGB one's transfer takes the device's
/// `Pow`, which it does not model.
fn interpreted(fmt: &NormFormat) -> bool {
    !fmt.channels.iter().any(|(c, _)| matches!(c, Channel::Srgb))
}

/// The IEEE binary16 bits of `x` (within `[-1, 1]`) rounded to nearest even once.
fn f64_to_half_bits(x: f64) -> u16 {
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    let bits = if a < f64::powi(2.0, -14) {
        // The denormal grid `2^-24`, whose top step is the least normal's bits.
        (a * f64::powi(2.0, 24)).round_ties_even() as u16
    } else {
        let e = ((a.to_bits() >> 52) & 0x7FF) as i32 - 1023;
        let n = (a * f64::powi(2.0, 10 - e)).round_ties_even() as i32;
        let (e, n) = if n == 2048 { (e + 1, 1024) } else { (e, n) };
        (((e + 15) << 10) | (n - 1024)) as u16
    };
    sign | bits
}

/// A normalized attachment format: its channels in Metal's order, each with its bit offset in
/// the texel, and the texel's bytes.
struct NormFormat {
    pf: PixelFormat,
    channels: &'static [(Channel, u32)],
    bytes: u32,
}

const NORM_FORMATS: [NormFormat; 15] = {
    use Channel::{Snorm, Srgb, Unorm};
    [
        NormFormat {
            pf: PixelFormat::RGBA8Unorm,
            channels: &[(Unorm(8), 0), (Unorm(8), 8), (Unorm(8), 16), (Unorm(8), 24)],
            bytes: 4,
        },
        NormFormat {
            pf: PixelFormat::BGRA8Unorm,
            channels: &[(Unorm(8), 16), (Unorm(8), 8), (Unorm(8), 0), (Unorm(8), 24)],
            bytes: 4,
        },
        NormFormat {
            pf: PixelFormat::RGBA8UnormSRGB,
            channels: &[(Srgb, 0), (Srgb, 8), (Srgb, 16), (Unorm(8), 24)],
            bytes: 4,
        },
        NormFormat {
            pf: PixelFormat::BGRA8UnormSRGB,
            channels: &[(Srgb, 16), (Srgb, 8), (Srgb, 0), (Unorm(8), 24)],
            bytes: 4,
        },
        NormFormat {
            pf: PixelFormat::RGBA8Snorm,
            channels: &[(Snorm(8), 0), (Snorm(8), 8), (Snorm(8), 16), (Snorm(8), 24)],
            bytes: 4,
        },
        NormFormat {
            pf: PixelFormat::RGB10A2Unorm,
            channels: &[
                (Unorm(10), 0),
                (Unorm(10), 10),
                (Unorm(10), 20),
                (Unorm(2), 30),
            ],
            bytes: 4,
        },
        NormFormat {
            pf: PixelFormat::BGR10A2Unorm,
            channels: &[
                (Unorm(10), 20),
                (Unorm(10), 10),
                (Unorm(10), 0),
                (Unorm(2), 30),
            ],
            bytes: 4,
        },
        NormFormat {
            pf: PixelFormat::RG16Unorm,
            channels: &[(Unorm(16), 0), (Unorm(16), 16)],
            bytes: 4,
        },
        NormFormat {
            pf: PixelFormat::RG16Snorm,
            channels: &[(Snorm(16), 0), (Snorm(16), 16)],
            bytes: 4,
        },
        NormFormat {
            pf: PixelFormat::RGBA16Unorm,
            channels: &[
                (Unorm(16), 0),
                (Unorm(16), 16),
                (Unorm(16), 32),
                (Unorm(16), 48),
            ],
            bytes: 8,
        },
        NormFormat {
            pf: PixelFormat::RGBA16Snorm,
            channels: &[
                (Snorm(16), 0),
                (Snorm(16), 16),
                (Snorm(16), 32),
                (Snorm(16), 48),
            ],
            bytes: 8,
        },
        NormFormat {
            pf: PixelFormat::RG8Unorm,
            channels: &[(Unorm(8), 0), (Unorm(8), 8)],
            bytes: 2,
        },
        NormFormat {
            pf: PixelFormat::RG8Snorm,
            channels: &[(Snorm(8), 0), (Snorm(8), 8)],
            bytes: 2,
        },
        NormFormat {
            pf: PixelFormat::R16Unorm,
            channels: &[(Unorm(16), 0)],
            bytes: 2,
        },
        NormFormat {
            pf: PixelFormat::R16Snorm,
            channels: &[(Snorm(16), 0)],
            bytes: 2,
        },
    ]
};

impl NormFormat {
    /// The texel's bits as the staging view's components (`R32Uint`, `RG32Uint`, or `R16Uint`).
    fn components(&self, texel: u64) -> [u32; 4] {
        match self.bytes {
            8 => [texel as u32, (texel >> 32) as u32, 0, 0],
            4 => [texel as u32, 0, 0, 0],
            _ => [texel as u32 & 0xFFFF, 0, 0, 0],
        }
    }

    /// The texel bits of the view's components `t`.
    fn texel(&self, t: [u32; 4]) -> u64 {
        match self.bytes {
            8 => u64::from(t[0]) | (u64::from(t[1]) << 32),
            4 => u64::from(t[0]),
            _ => u64::from(t[0] & 0xFFFF),
        }
    }

    /// The code of channel `k` of `texel`.
    fn code(&self, texel: u64, k: usize) -> u32 {
        let (channel, at) = self.channels[k];
        ((texel >> at) & ((1u64 << channel.bits()) - 1)) as u32
    }
}

/// The floats a store writes: specials but NaN (whose code is the device's), the bounds, every tie
/// of an 8-bit code and some of others, and their neighbours.
fn norm_inputs() -> Vec<f32> {
    let mut xs = vec![
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        -1.0,
        1.0,
        2.0,
        -2.0,
        0.5,
        -0.5,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        1.0f32.next_up(),
        (-1.0f32).next_down(),
        1.0f32.next_down(),
    ];
    for k in 0..255 {
        let tie = ((f64::from(k) + 0.5) / 255.0) as f32;
        xs.extend([tie.next_down(), tie, tie.next_up(), -tie]);
    }
    for k in (0..4095).step_by(7) {
        let tie = ((f64::from(k) + 0.5) / 4095.0) as f32;
        xs.extend([tie.next_down(), tie, tie.next_up()]);
    }
    xs
}

/// The cells of one tile, each a thread's.
const NORM_CELLS: usize = 256;

/// The committed profiles the conversions run on, one of each driver family the lanes run: the
/// conversion is the same integer sequence on every profile.
const NORM_PROFILES: [&str; 4] = [
    "moltenvk-m3",
    "lavapipe",
    "amd-rdna2-apu-radv",
    "intel-arl-anv",
];

/// Run `entry` of `cfb_tile_norm` over an attachment of `fmt` whose texels are `texels`, a whole
/// number of tiles in rows of one tile's width, with `buffer` at buffer 0, on each of
/// [`NORM_PROFILES`] that compiles it: each profile's buffer and texels after.
fn norm_runs(
    bc: &[u8],
    entry: &str,
    fmt: &NormFormat,
    buffer: &[u8],
    texels: &[u64],
) -> Vec<(String, Vec<u8>, Vec<u64>)> {
    let height = (texels.len() as u32).div_ceil(16);
    let inputs = tile_inputs(&[fmt.pf]);
    let mut out = Vec::new();
    for (stem, p) in committed_profiles().expect("profiles") {
        if !NORM_PROFILES.contains(&stem.as_str()) {
            continue;
        }
        let Some((plan, m)) = module(bc, entry, &stem, &p, &inputs) else {
            continue;
        };
        let mut img = Image::new(attachment(0), [16, height, 1], 1, 1, 1);
        for (t, x) in img.texels.iter_mut().zip(texels) {
            *t = fmt.components(*x);
        }
        let d = Tiles {
            area: [16, height],
            tile: [16, 16],
            threads: [16, 16],
            buffer: buffer.to_vec(),
            attachments: &[],
            nominal: None,
        };
        let (buf, images) = run_images(&plan, &m, &d, vec![img]);
        let after = images[0].iter().map(|t| fmt.texel(*t)).collect();
        out.push((stem, buf, after));
    }
    out
}

/// The floats of `bytes`.
fn floats(bytes: &[u8]) -> Vec<f32> {
    words(bytes).into_iter().map(f32::from_bits).collect()
}

/// The stores' inputs ([`norm_inputs`]) as tile buffer 0: cell `i`'s lane `k` is
/// `xs[(3i + k) mod len]`, over whole tiles, so every input is reached; and the cells.
fn norm_store_values(xs: &[f32]) -> (Vec<u8>, usize) {
    let cells = xs.len().div_ceil(3).div_ceil(NORM_CELLS) * NORM_CELLS;
    let values = (0..cells)
        .flat_map(|i| (0..4).flat_map(move |k| xs[(3 * i + k) % xs.len()].to_le_bytes()))
        .collect();
    (values, cells)
}

/// The entries of `cfb_tile_norm` that `op` (`read` or `write`, or `hread` or `hwrite` through
/// halves) a format of `n` channels through: a member of its channels, and of four, with their
/// lanes.
fn norm_entries(op: &str, n: usize) -> Vec<(usize, String)> {
    let mut entries = vec![(n, format!("cfb_tn_{op}{n}"))];
    if n < 4 {
        entries.push((4, format!("cfb_tn_{op}4")));
    }
    entries
}

/// A tile function's implicit imageblock over a normalized attachment reads as Metal does (probes):
/// a member of the format's channels or of four floats reads each channel's value, in Metal's
/// channel order, and green and blue 0 and alpha 1 past them, through the attachment's bits in its
/// unsigned-integer view, which the read keeps; each value within the device's conversion's bound
/// ([`near`]). The texels cover every code of each 8-bit channel and the extremes of the others.
#[test]
fn a_normalized_attachment_reads_as_metal_does() {
    let Some(bc) = assemble("cfb_tile_norm") else {
        return;
    };
    let cells = NORM_CELLS;
    let mut ran = 0;
    for fmt in NORM_FORMATS.iter().filter(|f| interpreted(f)) {
        let n = fmt.channels.len();
        // Texel `i`'s channel `k` holds code `(i * (2k + 1) + 37k) mod 2^bits`, and the last
        // texels the extremes, so each 8-bit channel takes every code.
        let mut texels: Vec<u64> = (0..cells as u64)
            .map(|i| {
                fmt.channels
                    .iter()
                    .enumerate()
                    .fold(0u64, |t, (k, &(channel, at))| {
                        let code = (i * (2 * k as u64 + 1) + 37 * k as u64)
                            & ((1u64 << channel.bits()) - 1);
                        t | (code << at)
                    })
            })
            .collect();
        texels[cells - 1] = u64::MAX >> (64 - 8 * fmt.bytes);
        texels[cells - 2] = fmt.channels.iter().fold(0, |t, &(channel, at)| {
            t | (1u64 << (channel.bits() - 1) << at)
        });
        for (lanes, entry) in norm_entries("read", n) {
            for (stem, buf, after) in norm_runs(&bc, &entry, fmt, &vec![0xAB; cells * 16], &texels)
            {
                let got = floats(&buf);
                for (i, &t) in texels.iter().enumerate() {
                    let want: Vec<f32> = (0..4)
                        .map(|k| match k {
                            k if k < n => fmt.channels[k].0.value(fmt.code(t, k)),
                            k if k < lanes => [0.0, 0.0, 0.0, 1.0][k],
                            _ => -7.0,
                        })
                        .collect();
                    let have = &got[4 * i..4 * i + 4];
                    let channel = |k: usize| fmt.channels.get(k).map(|c| c.0);
                    assert!(
                        (0..4).all(|k| near(channel(k), have[k], want[k], false)),
                        "{stem} {:?} {entry} texel {t:#x}: {have:?}, Metal's {want:?}",
                        fmt.pf
                    );
                }
                assert_eq!(after, texels, "{stem} {:?} {entry} keeps", fmt.pf);
                ran += 1;
            }
        }
    }
    assert!(ran > 0);
}

/// A tile function's implicit imageblock over a normalized attachment stores as Metal does
/// (probes): a member of the format's channels or of four floats stores each channel's code, in
/// Metal's channel order, its lanes past the channels dropped, each code within one of Metal's
/// (the device's conversion, [`code_near`]). The stores cover the ties between 8-bit codes, some of
/// the 12-bit grid's, and the specials.
#[test]
fn a_normalized_attachment_stores_as_metal_does() {
    let Some(bc) = assemble("cfb_tile_norm") else {
        return;
    };
    let xs = norm_inputs();
    let (values, cells) = norm_store_values(&xs);
    let lane = |i: usize, k: usize| xs[(3 * i + k) % xs.len()];
    let cleared = vec![0xCDCD_CDCD_CDCD_CDCD; cells];
    let mut ran = 0;
    for fmt in NORM_FORMATS.iter().filter(|f| interpreted(f)) {
        for (_, entry) in norm_entries("write", fmt.channels.len()) {
            for (stem, _, after) in norm_runs(&bc, &entry, fmt, &values, &cleared) {
                for (i, &t) in after.iter().enumerate() {
                    let want = fmt
                        .channels
                        .iter()
                        .enumerate()
                        .fold(0u64, |w, (k, &(channel, at))| {
                            w | (u64::from(channel.code(lane(i, k))) << at)
                        });
                    let near = (fmt.channels.iter().enumerate())
                        .all(|(k, &(c, _))| code_near(c, fmt.code(t, k), fmt.code(want, k)));
                    let unwritten = !(fmt.channels.iter())
                        .fold(0u64, |m, &(c, at)| m | (((1u64 << c.bits()) - 1) << at));
                    assert!(
                        near && (t & unwritten) == (want & unwritten),
                        "{stem} {:?} {entry} cell {i}: {t:#x}, Metal's {want:#x}: {:?}",
                        fmt.pf,
                        (0..4).map(|k| lane(i, k)).collect::<Vec<_>>()
                    );
                }
                ran += 1;
            }
        }
    }
    assert!(ran > 0);
}

/// A read after the tile function's own write of a normalized attachment reads the codes it
/// stored (a probe): the cell holds the texel's bits, each code within one of Metal's and its read
/// within the device's conversion's bound.
#[test]
fn a_normalized_cell_reads_back_the_codes_it_stored() {
    let Some(bc) = assemble("cfb_tile_norm") else {
        return;
    };
    let xs = norm_inputs();
    let (values, cells) = norm_store_values(&xs);
    let lane = |i: usize, k: usize| xs[(3 * i + k) % xs.len()];
    let fmt = &NORM_FORMATS[9];
    assert_eq!(fmt.pf, PixelFormat::RGBA16Unorm);
    let mut ran = 0;
    for (stem, buf, after) in norm_runs(&bc, "cfb_tn_round4", fmt, &values, &vec![0; cells]) {
        let got = floats(&buf);
        for i in 0..cells {
            for k in 0..4 {
                let (channel, _) = fmt.channels[k];
                let stored = fmt.code(after[i], k);
                assert!(
                    code_near(channel, stored, channel.code(lane(i, k))),
                    "{stem} cell {i} lane {k}: code {stored}"
                );
                let (have, want) = (got[4 * i + k], channel.value(stored));
                assert!(
                    near(Some(channel), have, want, false),
                    "{stem} cell {i} lane {k}: {have}, Metal's {want}"
                );
            }
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// A tile function's implicit imageblock over a normalized attachment reads through a `half`
/// member within the device's conversion's bound of Metal's (a probe on an M4 and an M3 Max: each
/// channel's exact value correctly rounded to `half` once; the device rounds its unpacked `float`
/// again, [`near`]), in Metal's channel order, and green and blue 0 and alpha 1 past them.
/// Channel 0 takes the top 256 codes, channel 1 the bottom 256 of an snorm one (an unorm one's
/// spread over its range), channel 2 the lowest, the `half` denormals, so a 16-bit channel reaches
/// the codes whose `float` value rounded again to `half` differs (unorm 65455 and 65519, snorm
/// ±32759), and every 8-bit channel every code.
#[test]
fn a_normalized_attachment_reads_through_halves_as_metal_does() {
    let Some(bc) = assemble("cfb_tile_norm") else {
        return;
    };
    let cells = NORM_CELLS;
    let (mut ran, mut twice) = (0, 0);
    for fmt in NORM_FORMATS.iter().filter(|f| interpreted(f)) {
        let n = fmt.channels.len();
        let texels: Vec<u64> = (0..cells as u32)
            .map(|i| {
                fmt.channels
                    .iter()
                    .enumerate()
                    .fold(0u64, |t, (k, &(channel, at))| {
                        let bits = channel.bits();
                        let mask = (1u32 << bits) - 1;
                        let top = match channel {
                            Channel::Snorm(_) => (1u32 << (bits - 1)) - 1,
                            _ => mask,
                        };
                        let code = match (k, channel) {
                            (0, _) => top.wrapping_sub(i),
                            (1, Channel::Snorm(_)) => i.wrapping_sub(top),
                            (1, _) => i * mask / 255,
                            (2, Channel::Snorm(_)) => i.wrapping_sub(128),
                            (2, _) => i,
                            _ => i * 7 + 37,
                        } & mask;
                        t | (u64::from(code) << at)
                    })
            })
            .collect();
        for &t in &texels {
            for (k, &(channel, _)) in fmt.channels.iter().enumerate() {
                let c = fmt.code(t, k);
                let again = f64_to_half_bits(f64::from(channel.value(c)));
                twice += usize::from(again != f64_to_half_bits(channel.exact(c)));
            }
        }
        for (lanes, entry) in norm_entries("hread", n) {
            for (stem, buf, after) in norm_runs(&bc, &entry, fmt, &vec![0xAB; cells * 16], &texels)
            {
                let got = floats(&buf);
                for (i, &t) in texels.iter().enumerate() {
                    let want: Vec<f32> = (0..4)
                        .map(|k| match k {
                            k if k < n => fmt.channels[k].0.half(fmt.code(t, k)),
                            k if k < lanes => [0.0, 0.0, 0.0, 1.0][k],
                            _ => -7.0,
                        })
                        .collect();
                    let have = &got[4 * i..4 * i + 4];
                    let channel = |k: usize| fmt.channels.get(k).map(|c| c.0);
                    assert!(
                        (0..4).all(|k| near(channel(k), have[k], want[k], true)),
                        "{stem} {:?} {entry} texel {t:#x}: {have:?}, Metal's {want:?}",
                        fmt.pf
                    );
                }
                assert_eq!(after, texels, "{stem} {:?} {entry} keeps", fmt.pf);
                ran += 1;
            }
        }
    }
    assert!(ran > 0);
    // The texels reach the codes a `float` read rounded again to `half` gets wrong.
    assert!(twice >= 4, "{twice}");
}

/// A tile function's implicit imageblock over a normalized attachment stores a `half` member as
/// Metal does (a probe): each channel's code of the `float` of the same value, in Metal's channel
/// order, its lanes past the channels dropped, each code within one of Metal's ([`code_near`]).
/// The halves are every sixteenth pattern but a NaN (whose code is the device's), the specials, and
/// the `half`s nearest each tie between 8-bit codes.
#[test]
fn a_normalized_attachment_stores_halves_as_metal_does() {
    let Some(bc) = assemble("cfb_tile_norm") else {
        return;
    };
    let nan = |h: u16| h & 0x7C00 == 0x7C00 && h & 0x3FF != 0;
    let mut hs: Vec<u16> = (0..=u16::MAX).step_by(16).filter(|&h| !nan(h)).collect();
    hs.extend([0x7C00, 0xFC00, 0x8000, 0x0001, 0x8001, 0x3BFF, 0x3C01]);
    for k in 0..255 {
        let tie = f64_to_half_bits((f64::from(k) + 0.5) / 255.0);
        hs.extend([tie - 1, tie, tie + 1, tie | 0x8000]);
    }
    let cells = hs.len().div_ceil(3).div_ceil(NORM_CELLS) * NORM_CELLS;
    let lane = |i: usize, k: usize| hs[(3 * i + k) % hs.len()];
    let values: Vec<u8> = (0..cells)
        .flat_map(|i| (0..4).flat_map(move |k| lane(i, k).to_le_bytes()))
        .collect();
    let cleared = vec![0xCDCD_CDCD_CDCD_CDCD; cells];
    let mut ran = 0;
    for fmt in NORM_FORMATS.iter().filter(|f| interpreted(f)) {
        for (_, entry) in norm_entries("hwrite", fmt.channels.len()) {
            for (stem, _, after) in norm_runs(&bc, &entry, fmt, &values, &cleared) {
                for (i, &t) in after.iter().enumerate() {
                    let want =
                        fmt.channels
                            .iter()
                            .enumerate()
                            .fold(0u64, |w, (k, &(channel, at))| {
                                let x = f32::from_bits(half_bits_to_f32(lane(i, k)));
                                w | (u64::from(channel.code(x)) << at)
                            });
                    let near = (fmt.channels.iter().enumerate())
                        .all(|(k, &(c, _))| code_near(c, fmt.code(t, k), fmt.code(want, k)));
                    assert!(
                        near,
                        "{stem} {:?} {entry} cell {i}: {t:#x}, Metal's {want:#x}",
                        fmt.pf
                    );
                }
                ran += 1;
            }
        }
    }
    assert!(ran > 0);
}

/// A normalized attachment is staged through the unsigned-integer view of its texel's bytes in
/// 32-bit words (`R32Uint`, `RG32Uint`, or `R16Uint` for two bytes), through a `float` member or
/// a `half` one, and a member of other lanes than its channels or four, or a format whose texel is
/// one byte, is not implemented.
#[test]
fn a_normalized_attachment_is_bound_through_its_bits() {
    let Some(bc) = assemble("cfb_tile_norm") else {
        return;
    };
    let not_implemented = |r: Result<ComputePipeline, Error>| match r {
        Err(Error::Unsupported(u)) => {
            u.feature == Feature::TileShading && u.needs == Needs::NotImplemented
        }
        _ => false,
    };
    let cases = [
        (
            "cfb_tn_read4",
            PixelFormat::RGBA8UnormSRGB,
            vk::Format::R32_UINT,
        ),
        (
            "cfb_tn_write4",
            PixelFormat::BGR10A2Unorm,
            vk::Format::R32_UINT,
        ),
        (
            "cfb_tn_read4",
            PixelFormat::RGBA16Snorm,
            vk::Format::R32G32_UINT,
        ),
        ("cfb_tn_write2", PixelFormat::RG8Unorm, vk::Format::R16_UINT),
        ("cfb_tn_read4", PixelFormat::R16Unorm, vk::Format::R16_UINT),
        (
            "cfb_tn_hread4",
            PixelFormat::RGBA8Unorm,
            vk::Format::R32_UINT,
        ),
        (
            "cfb_tn_hwrite2",
            PixelFormat::RG16Snorm,
            vk::Format::R32_UINT,
        ),
        ("cfb_tn_hread1", PixelFormat::R16Unorm, vk::Format::R16_UINT),
    ];
    let mut planned = 0;
    for stem in ["moltenvk-m3", "lavapipe", "amd-rdna2-apu-radv"] {
        let dev = device(stem);
        let b = Budget::standard();
        let mut s = BindState::default();
        s.stages.insert(Stage::Kernel, StageTable::default());
        s.imageblock_size = ImageblockSize::new(16, 16);
        s.render_area = [16, 16];
        for (entry, pf, view) in cases {
            let pipe = match dev.compile_tile_pipeline(&descriptor(&bc, entry, &[Some(pf)]), &b) {
                Ok(p) => p,
                // A profile without the extended storage formats refuses the 16-bit views.
                Err(Error::Unsupported(u)) if u.feature == Feature::Format => continue,
                Err(e) => panic!("{stem} {entry} {pf:?}: {e}"),
            };
            let plan = pipe.plan(&dev, &s.shape()).expect("plan");
            assert!(
                plan.descriptors.iter().any(|w| w.source
                    == WriteSource::TileAttachment {
                        color_index: 0,
                        format: view,
                        view: StorageAttachmentView::Attachment,
                        view_alias: None,
                    }),
                "{stem} {entry} {pf:?}: {:?}",
                plan.descriptors
            );
            planned += 1;
        }
        for (entry, pf) in [
            ("cfb_tn_read1", PixelFormat::RGBA8Unorm),
            ("cfb_tn_read2", PixelFormat::R16Unorm),
            ("cfb_tn_read2", PixelFormat::RGBA16Unorm),
            ("cfb_tn_read4", PixelFormat::R8Unorm),
            ("cfb_tn_read4", PixelFormat::RG8UnormSRGB),
            ("cfb_tn_hread1", PixelFormat::RGBA8Unorm),
            ("cfb_tn_hwrite2", PixelFormat::RGBA16Snorm),
            ("cfb_tn_hread2", PixelFormat::R16Unorm),
            ("cfb_tn_hread4", PixelFormat::R8Unorm),
        ] {
            let d = descriptor(&bc, entry, &[Some(pf)]);
            assert!(
                not_implemented(dev.compile_tile_pipeline(&d, &b)),
                "{stem} {entry} {pf:?}"
            );
        }
    }
    assert!(planned > 0);
}
