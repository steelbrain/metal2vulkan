//! Kernel imageblocks through workgroup staging, end to end on the CPU interpreter.
//!
//! A kernel's explicit imageblock is the threadgroup arena's last region, `width x height` cells
//! of `air.imageblock_data_size` bytes whose extent is specialized (`SpecKey::ImageblockWidth`,
//! `SpecKey::ImageblockHeight`); slice writes copy one field of a block of cells to a texture.
//! Every kernel here stages values that name their own cell, so a wrong cell stride, row stride,
//! field offset, extent, or overlap with the threadgroup regions lands a value that exists
//! nowhere else. Each kernel runs on every committed profile that compiles it without runtime
//! inputs, in both scheduling orders; the interpreter traps any read of threadgroup memory nobody
//! wrote and any access outside an image.
//!
//! - `kernel_imageblock_staging` (owned, `engine/`): neighbour reads across the imageblock
//!   barrier and the extent queries; an imageblock after a `[[threadgroup(0)]]` region; a sized
//!   two-lane slice at a nonzero field offset.
//! - The authored compute-imageblock fixtures (`validation/fixtures/public/`): one cell; a tile
//!   twice the threadgroup; texture-array slices at a layer, the second plane at a nonzero field.
//!   `kernel_imageblock_full_tile_planes` and `kernel_imageblock_tile_to_texture_array` stage a
//!   `half` plane beside two bytes of cell padding nothing writes: a leaf load reads its whole word
//!   (`memory::leaf`), which the interpreter's per-byte poison traps although the padding's bits
//!   are discarded, so they are exercised by the authored GPU cases instead.
//! - An integer slice names no signedness: its class is the numeric type of the format bound at
//!   the slot, the slot's storage-format input on every profile, which a formatless choice keeps
//!   and the binder then keys; with a signed format, it is written signed.
//! - The binder specializes the extent, refuses an unset or unanswerable one, charges the
//!   imageblock to `maxComputeSharedMemorySize` after the threadgroup regions, and plans a narrow
//!   slice against any bound format: its texel is the element's lanes and zeros.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use std::collections::BTreeMap;

use crate::bind::{BindState, BindingPlan, BoundTexture, ImageblockSize, StageTable, TextureFacts};
use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::profile::DeviceProfile;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{BindError, Error, Feature, LowerFamily, Needs, RuntimeInput};
use crate::library::Library;
use crate::lower::select::LowerInputs;
use crate::lower::select::Plan;
use crate::metal::format::storage_image_choice;
use crate::metal::{PixelFormat, TextureType, TextureUsage};
use crate::pipeline::facts::{Mechanism, RuntimeFact};
use crate::pipeline::interface::PushLoc;
use crate::pipeline::lower_inputs::{StorageImageChoice, StorageImageClass};
use crate::pipeline::sym::{
    Axis, PushField, ResourceKind, ResourceSlot, SampledType, SlotClass, SlotRef, SpecKey, Stage,
};
use crate::pipeline::testing::{
    engine_dir, fixtures_dir, llvm_assemble, typed_module, typed_module_with, AssembleError,
    Scratch,
};
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, StageSource,
};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;
use crate::vk;

fn assemble(path: std::path::PathBuf) -> Option<Vec<u8>> {
    match llvm_assemble(&path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {}: {why}", path.display()),
    }
}

fn owned() -> Option<Vec<u8>> {
    assemble(engine_dir().join("kernel_imageblock_staging.ll"))
}

fn authored(stem: &str) -> Option<Vec<u8>> {
    assemble(fixtures_dir().join(format!("{stem}.ll")))
}

/// The module of `entry` on `p`, or `None` when `p` needs a runtime input for it.
fn module(bc: &[u8], entry: &str, stem: &str, p: &DeviceProfile) -> Option<(Plan, TypedModule)> {
    match typed_module(bc, entry, p) {
        Ok(m) => Some(m),
        Err(Error::NeedsInput(_)) => None,
        Err(e) => panic!("{stem}: {entry}: {e}"),
    }
}

fn texture(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Texture,
        index,
    }
}

/// One dispatch of one threadgroup of `local` threads with an imageblock of `block` cells.
struct Dispatch<'a> {
    local: [u32; 2],
    block: [u32; 2],
    /// `[[buffer(i)]]` holds `buffers[i]`.
    buffers: &'a [Vec<u8>],
    /// `(n, bytes)` of each `[[threadgroup(n)]]` length.
    threadgroup: &'a [(u32, u64)],
}

/// Run `m` in both scheduling orders (which must agree): the buffers and images after.
fn run(
    plan: &Plan,
    m: &TypedModule,
    d: &Dispatch<'_>,
    images: &[Image],
) -> (Vec<Vec<u8>>, Vec<Image>) {
    assert!(
        plan.facts
            .iter()
            .any(|f| matches!(f, RuntimeFact::ImageblockSize { .. })),
        "the plan depends on the imageblock size"
    );
    let mut spec = BTreeMap::new();
    spec.insert(SpecKey::LocalSize(Axis::X), u64::from(d.local[0]));
    spec.insert(SpecKey::LocalSize(Axis::Y), u64::from(d.local[1]));
    spec.insert(SpecKey::LocalSize(Axis::Z), 1);
    spec.insert(SpecKey::FlatLocalSize, u64::from(d.local[0] * d.local[1]));
    spec.insert(SpecKey::ImageblockWidth, u64::from(d.block[0]));
    spec.insert(SpecKey::ImageblockHeight, u64::from(d.block[1]));
    for &(n, len) in d.threadgroup {
        spec.insert(SpecKey::TgLength(n), len);
    }
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            // One region, every buffer bound at its start.
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            // A bound value read as data, which the interpreter writes from `spec`.
            (PushField::Specialized(_), PushLoc::Push { .. }) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let buffers: Vec<Buffer> = d
        .buffers
        .iter()
        .enumerate()
        .map(|(i, b)| Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: i as u32,
            }),
            address: None,
            bytes: b.clone(),
        })
        .collect();
    let res = Resources {
        buffers,
        push,
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            order,
            ..Config::default()
        };
        let (o, imgs) = run_with_images(m, &cfg, res.clone(), images.to_vec()).expect("run");
        outs.push((o.buffers, imgs));
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

fn fl(x: f32) -> u32 {
    x.to_bits()
}

/// A texel the kernels never write: every test checks the texels outside a block keep it.
const POISON: [u32; 4] = [0xDEAD_0001, 0xDEAD_0002, 0xDEAD_0003, 0xDEAD_0004];

fn poisoned(slot: u32, extent: [u32; 2], layers: u32) -> Image {
    let mut img = Image::new(texture(slot), [extent[0], extent[1], 1], layers, 1, 1);
    img.texels.fill(POISON);
    img
}

fn texel(img: &Image, layer: u32, x: u32, y: u32) -> [u32; 4] {
    img.texels[img.index(0, layer, 0, [x, y, 0]).expect("texel")]
}

/// Neighbour reads across the imageblock barrier, and the extent the queries answer, for a 4x3
/// imageblock.
#[test]
fn imageblock_cells_are_shared_across_the_threadgroup() {
    let Some(bc) = owned() else { return };
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some((plan, m)) = module(&bc, "imageblock_neighbours", &stem, &p) else {
            continue;
        };
        let (w, h) = (4u32, 3u32);
        let d = Dispatch {
            local: [w, h],
            block: [w, h],
            buffers: &[vec![0; (w * h * 16) as usize]],
            threadgroup: &[],
        };
        let (bufs, _) = run(&plan, &m, &d, &[]);
        let out = words(&bufs[0]);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let nx = (x + 1) % w;
                let want = [fl(nx as f32 + 100.0 * y as f32), nx | (y << 16), w, h];
                assert_eq!(out[4 * i..4 * i + 4], want, "{stem} ({x}, {y})");
            }
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// The imageblock follows the `[[threadgroup(0)]]` region: neither overwrites the other.
#[test]
fn imageblock_follows_the_threadgroup_regions() {
    let Some(bc) = owned() else { return };
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some((plan, m)) = module(&bc, "imageblock_after_threadgroup", &stem, &p) else {
            continue;
        };
        let (w, h) = (4u32, 4u32);
        let n = w * h;
        let d = Dispatch {
            local: [w, h],
            block: [w, h],
            buffers: &[vec![0; (n * 4) as usize]],
            threadgroup: &[(0, u64::from(n) * 4)],
        };
        let (bufs, _) = run(&plan, &m, &d, &[]);
        let out = words(&bufs[0]);
        for i in 0..n {
            let j = (i + 1) % n;
            assert_eq!(out[i as usize], 3 * j + 1 + 1000 * j, "{stem} thread {i}");
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// A sized 3x2 block of `half2` planes at field offset 4, written at (1, 1): the first two lanes
/// carry the plane, the others 0, and every texel outside the block keeps its poison.
#[test]
fn a_sized_two_lane_slice_walks_the_cell_stride() {
    let Some(bc) = owned() else { return };
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some((plan, m)) = module(&bc, "imageblock_sized_slice", &stem, &p) else {
            continue;
        };
        let d = Dispatch {
            local: [4, 4],
            block: [4, 4],
            buffers: &[],
            threadgroup: &[],
        };
        let (_, imgs) = run(&plan, &m, &d, &[poisoned(0, [6, 5], 1)]);
        for y in 0..5 {
            for x in 0..6 {
                let want = if (1..4).contains(&x) && (1..3).contains(&y) {
                    let (cx, cy) = (x - 1, y - 1);
                    [fl(cx as f32), fl(cy as f32 * 0.5), 0, 0]
                } else {
                    POISON
                };
                assert_eq!(texel(&imgs[0], 0, x, y), want, "{stem} ({x}, {y})");
            }
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// `kernel_imageblock_word`: one thread stores 42 in its cell and reads it back.
#[test]
fn one_cell_reads_back_its_store() {
    let Some(bc) = authored("kernel_imageblock_word") else {
        return;
    };
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some((plan, m)) = module(&bc, "imageblock_word", &stem, &p) else {
            continue;
        };
        let d = Dispatch {
            local: [1, 1],
            block: [1, 1],
            buffers: &[vec![0; 4]],
            threadgroup: &[],
        };
        let (bufs, _) = run(&plan, &m, &d, &[]);
        assert_eq!(words(&bufs[0]), [42], "{stem}");
        ran += 1;
    }
    assert!(ran > 0);
}

/// A synthetic kernel whose imageblock cell is `data_size` bytes aligned to 4: each thread stores
/// a word naming its cell at offset 0 and a `ushort` at offset 4 (the two bytes after it are
/// padding, which the interpreter traps a read of, so the `ushort` is not read back), then,
/// after the imageblock barrier, reports its right-hand neighbour's word (wrapping at the
/// threadgroup's width):
///     out[i] = 1000 + j, j = y * W + (x + 1) % W
fn odd_stride_kernel(data_size: u32) -> String {
    format!(
        r#"%"struct.metal::_imageblock_base" = type {{ ptr addrspace(4) }}

define void @odd_stride(%"struct.metal::_imageblock_base" %block, ptr addrspace(1) %out, <2 x i16> %lid, <2 x i16> %tg) {{
  %cell = call ptr addrspace(4) @air.imageblock_data(<2 x i16> %lid, i32 0, i16 0)
  %x = extractelement <2 x i16> %lid, i64 0
  %y = extractelement <2 x i16> %lid, i64 1
  %x32 = zext i16 %x to i32
  %y32 = zext i16 %y to i32
  %w16 = extractelement <2 x i16> %tg, i64 0
  %w = zext i16 %w16 to i32
  %row = mul i32 %y32, %w
  %i = add i32 %row, %x32
  %v = add i32 %i, 1000
  store i32 %v, ptr addrspace(4) %cell, align 4
  %tagp = getelementptr inbounds i8, ptr addrspace(4) %cell, i64 4
  %tag = trunc i32 %i to i16
  store i16 %tag, ptr addrspace(4) %tagp, align 2
  call void @air.wg.barrier(i32 8, i32 1)
  %x1 = add i32 %x32, 1
  %xn = urem i32 %x1, %w
  %xn16 = trunc i32 %xn to i16
  %pos = insertelement <2 x i16> %lid, i16 %xn16, i64 0
  %other = call ptr addrspace(4) @air.imageblock_data(<2 x i16> %pos, i32 0, i16 0)
  %ov = load i32, ptr addrspace(4) %other, align 4
  %slot = zext i32 %i to i64
  %dst = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %slot
  store i32 %ov, ptr addrspace(1) %dst, align 4
  ret void
}}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.wg.barrier(i32, i32)

!air.kernel = !{{!0}}
!0 = !{{ptr @odd_stride, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !5, !6, !7}}
!3 = !{{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 {data_size}, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"imageblock<Tagged, layout_explicit>", !"air.arg_name", !"block"}}
!4 = !{{i32 0, i32 4, i32 0, !"uint", !"value", i32 4, i32 2, i32 0, !"ushort", !"tag"}}
!5 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!6 = !{{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"lid"}}
!7 = !{{i32 3, !"air.threads_per_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tg"}}
"#
    )
}

/// `odd_stride_kernel(8)` with its first cell pointer taken out of the imageblock parameter's own
/// value instead of from `air.imageblock_data`.
fn parameter_value_kernel() -> String {
    odd_stride_kernel(8).replace(
        "  %cell = call ptr addrspace(4) @air.imageblock_data(<2 x i16> %lid, i32 0, i16 0)\n",
        "  %cell = extractvalue %\"struct.metal::_imageblock_base\" %block, 0\n",
    )
}

/// Assemble `text` through a scratch file named after `name`, removed after.
fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new(&format!("m2v-imageblock-{name}"));
    let dir = scratch.path();
    let path = dir.join(format!("{name}.ll"));
    std::fs::write(&path, text).expect("write scratch .ll");
    assemble(path)
}

/// A cell stride is the data size rounded up to the imageblock's alignment: a 6-byte cell of
/// 4-byte alignment is 8 bytes apart (`RuntimeFact::ImageblockSize`), so every cell's word is as
/// aligned as the first's; a 6-byte stride would put every odd cell's word across two words, one
/// of them holding its left neighbour's `ushort`. A size that is already a multiple keeps its
/// stride.
#[test]
fn a_cell_stride_is_rounded_up_to_the_alignment() {
    let (Some(odd), Some(even)) = (
        assemble_text("odd", &odd_stride_kernel(6)),
        assemble_text("even", &odd_stride_kernel(8)),
    ) else {
        return;
    };
    let (w, h) = (4u32, 3u32);
    let n = (w * h) as usize;
    let expected: Vec<u32> = (0..h)
        .flat_map(|y| (0..w).map(move |x| y * w + (x + 1) % w))
        .map(|j| 1000 + j)
        .collect();
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        for bc in [&odd, &even] {
            let Some((plan, m)) = module(bc, "odd_stride", &stem, &p) else {
                continue;
            };
            assert!(
                plan.facts
                    .contains(&RuntimeFact::ImageblockSize { cell_bytes: 8 }),
                "{stem}: {:?}",
                plan.facts
            );
            let d = Dispatch {
                local: [w, h],
                block: [w, h],
                buffers: &[vec![0; 4 * n]],
                threadgroup: &[],
            };
            let (bufs, _) = run(&plan, &m, &d, &[]);
            assert_eq!(words(&bufs[0]), expected, "{stem}");
            ran += 1;
        }
    }
    assert!(ran > 0);
}

/// The imageblock parameter's own value (its `_imageblock_base` struct) has no lowered form: the
/// `extractvalue` reading it is refused `Lowering(Access)` there.
#[test]
fn the_imageblock_parameter_value_is_refused_where_it_is_read() {
    let Some(bc) = assemble_text("value", &parameter_value_kernel()) else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        match typed_module(&bc, "odd_stride", &p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    (u.feature, u.needs),
                    (
                        Feature::Lowering(LowerFamily::Access),
                        Needs::NotImplemented
                    ),
                    "{stem}"
                );
                let at = u.location.as_ref().and_then(|l| l.in_function());
                assert_eq!(
                    at.map(|(_, b, i)| (b, i)),
                    Some((Some(0), Some(0))),
                    "{stem}"
                );
            }
            other => panic!("{stem}: {:?}", other.map(|_| ())),
        }
    }
}

/// `kernel_imageblock_scaled_tile`: a 16x16 tile staged by 8x8 threads, copied whole (the
/// implicit region is the imageblock's extent, not the threadgroup's) as a four-lane and a
/// one-lane plane, whose texel is the value and three zeros.
#[test]
fn an_implicit_slice_copies_the_whole_imageblock() {
    let Some(bc) = authored("kernel_imageblock_scaled_tile") else {
        return;
    };
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some((plan, m)) = module(&bc, "kernel_imageblock_scaled_tile", &stem, &p) else {
            continue;
        };
        // Its slices are written at `thread_position_in_grid`, which differs between the
        // threadgroup's invocations: the invocation at local id 0 writes them alone.
        assert!(plan.families.texture.cooperative.is_empty(), "{stem}");
        let d = Dispatch {
            local: [8, 8],
            block: [16, 16],
            buffers: &[],
            threadgroup: &[],
        };
        let imgs = [poisoned(0, [16, 16], 1), poisoned(1, [16, 16], 1)];
        let (_, imgs) = run(&plan, &m, &d, &imgs);
        for y in 0..16u32 {
            for x in 0..16u32 {
                let (fx, fy) = (x as f32, y as f32);
                assert_eq!(
                    texel(&imgs[0], 0, x, y),
                    [fl(fx), fl(fy), fl(fx * 0.5), fl(fy * 0.25)],
                    "{stem} colour ({x}, {y})"
                );
                assert_eq!(
                    texel(&imgs[1], 0, x, y),
                    [fl(fx + 16.0 * fy), 0, 0, 0],
                    "{stem} weight ({x}, {y})"
                );
            }
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// `kernel_imageblock_float_tile_to_array`: each four-lane plane (the second at field offset 16)
/// lands at exactly its layer; the other layers keep their poison.
#[test]
fn array_slices_land_at_their_layer() {
    for (fixture, local, layers) in [("kernel_imageblock_float_tile_to_array", 16u32, [2u32, 0])] {
        let Some(bc) = authored(fixture) else { return };
        let mut ran = 0;
        for (stem, p) in committed_profiles().expect("profiles") {
            let Some((plan, m)) = module(&bc, fixture, &stem, &p) else {
                continue;
            };
            let d = Dispatch {
                local: [local, local],
                block: [local, local],
                buffers: &[],
                threadgroup: &[],
            };
            let imgs = [
                poisoned(0, [local, local], 3),
                poisoned(1, [local, local], 3),
            ];
            let (_, imgs) = run(&plan, &m, &d, &imgs);
            for y in 0..local {
                for x in 0..local {
                    let (fx, fy) = (x as f32, y as f32);
                    let (u, v) = (fx / 1024.0, fy / 1024.0);
                    let n = (x + 16 * y) as f32 / 1024.0;
                    let colour = [fl(1.0 + u), fl(1.0 + v), fl(1.0 - u), fl(1.0 - v)];
                    let other = [fl(2.0 + n), fl(3.0 + n), fl(4.0 + n), fl(5.0 + n)];
                    for layer in 0..3 {
                        let c = texel(&imgs[0], layer, x, y);
                        let o = texel(&imgs[1], layer, x, y);
                        let want_c = if layer == layers[0] { colour } else { POISON };
                        let want_o = if layer == layers[1] { other } else { POISON };
                        assert_eq!(c, want_c, "{stem} {fixture} colour {layer} ({x}, {y})");
                        assert_eq!(o, want_o, "{stem} {fixture} other {layer} ({x}, {y})");
                    }
                }
            }
            ran += 1;
        }
        assert!(ran > 0, "{fixture}");
    }
}

/// The storage-format inputs of an entry's texture slots.
fn formats(slots: &[(u32, vk::Format)]) -> LowerInputs {
    let mut inputs = LowerInputs::new();
    for &(index, f) in slots {
        inputs.storage_images.insert(
            SlotRef {
                stage: Stage::Kernel,
                class: SlotClass::Texture,
                index,
            },
            StorageImageChoice::new(f, StorageImageClass::Identity),
        );
    }
    inputs
}

/// Whether `r` is the refusal of an integer slice whose class nothing names.
fn unnamed_class(r: &Result<(Plan, TypedModule), Error>) -> bool {
    matches!(r, Err(Error::Unsupported(u))
        if u.feature == Feature::Imageblocks && u.needs == Needs::NotImplemented)
}

/// The storage-image choices the binder makes for an entry's bound pixel formats on `p`.
fn choices(p: &DeviceProfile, slots: &[(u32, PixelFormat)]) -> Option<LowerInputs> {
    let mut inputs = LowerInputs::new();
    for &(index, pf) in slots {
        let choice = storage_image_choice(pf, p).ok()?;
        inputs.storage_images.insert(kernel_texture(index), choice);
    }
    Some(inputs)
}

fn kernel_texture(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index,
    }
}

/// An integer slice names no signedness (neither its symbol nor the texture argument's ABI metadata
/// carries one): the class is the numeric type of the format bound at the slot, the slot's
/// storage-format input on every profile (`kernel_imageblock_runtime_slice_region` too). Where the
/// profile writes storage images without a format the binder's choice declares none but keeps the
/// type; a choice that keeps none names no class and is refused. With a signed format, the `short4`
/// plane is written sign-extended; the block sizes are the runtime operands, and both blocks start
/// at the imageblock's origin although the pointer names cell (2, 1). The one-lane `float` plane is
/// its value and three zeros, as Metal writes it.
#[test]
fn an_integer_slice_takes_its_class_from_the_format() {
    let (Some(bc), Some(authored)) = (owned(), authored("kernel_imageblock_runtime_slice_region"))
    else {
        return;
    };
    let entry = "imageblock_runtime_region";
    let mut ran_formatless = 0;
    let mut ran_keyed = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        for (bc, entry) in [
            (&bc, entry),
            (&authored, "kernel_imageblock_runtime_slice_region"),
        ] {
            let r = typed_module(bc, entry, &p);
            assert!(
                matches!(r, Err(Error::NeedsInput(_))),
                "{stem} {entry}: {:?}",
                r.map(|_| ())
            );
        }
        let formatless = formats(&[(0, vk::Format::UNDEFINED), (1, vk::Format::UNDEFINED)]);
        let r = typed_module_with(&bc, entry, &p, &formatless);
        assert!(unnamed_class(&r), "{stem}: {:?}", r.map(|_| ()));
        let Some(inputs) = choices(
            &p,
            &[(0, PixelFormat::RGBA16Sint), (1, PixelFormat::R32Float)],
        ) else {
            // A format this profile cannot store.
            continue;
        };
        let undeclared =
            inputs.storage_images[&kernel_texture(0)].vk_format == vk::Format::UNDEFINED;
        let (plan, m) = match typed_module_with(&bc, entry, &p, &inputs) {
            Ok(x) => x,
            // A format this profile cannot declare.
            Err(Error::Unsupported(_)) if !undeclared => continue,
            Err(e) => panic!("{stem}: {e}"),
        };
        assert!(
            plan.facts
                .contains(&RuntimeFact::StorageTextureNumericType {
                    slot: kernel_texture(0)
                }),
            "{stem}"
        );
        let (w, h) = (6u32, 4u32);
        let d = Dispatch {
            local: [w, h],
            block: [w, h],
            buffers: &[],
            threadgroup: &[],
        };
        let imgs = [poisoned(0, [8, 8], 1), poisoned(1, [8, 8], 1)];
        let (_, imgs) = run(&plan, &m, &d, &imgs);
        for y in 0..8u32 {
            for x in 0..8u32 {
                let code = if x < w / 2 && y < h / 2 {
                    let (cx, cy) = (x as i32, y as i32);
                    [3 * cx - 7, -cy, cx + 10 * cy, -300].map(|v| v as u32)
                } else {
                    POISON
                };
                assert_eq!(texel(&imgs[0], 0, x, y), code, "{stem} code ({x}, {y})");
                let weight = texel(&imgs[1], 0, x, y);
                if (1..w).contains(&x) && y < h - 1 {
                    let (cx, cy) = ((x - 1) as f32, y as f32);
                    assert_eq!(
                        weight,
                        [fl(cx + 0.25 * cy), 0, 0, 0],
                        "{stem} weight ({x}, {y})"
                    );
                } else {
                    assert_eq!(weight, POISON, "{stem} weight ({x}, {y})");
                }
            }
        }
        if undeclared {
            ran_formatless += 1;
        } else {
            ran_keyed += 1;
        }
    }
    assert!(
        ran_formatless > 0 && ran_keyed > 0,
        "formatless {ran_formatless}, keyed {ran_keyed}"
    );
}

fn device(stem: &str) -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

fn pipeline(dev: &Device, bc: &[u8], entry: &str) -> ComputePipeline {
    let lib = Library::from_air(bc).expect("library");
    let f = lib.function(entry).expect("function");
    let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    dev.compile_compute_pipeline(&d, &Budget::standard())
        .unwrap_or_else(|e| panic!("{entry}: {e}"))
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

/// The binder specializes the imageblock's extent from `BindState::imageblock_size`, refuses an
/// extent the encoder never set or `get_width` cannot answer, and charges the imageblock, after
/// the `[[threadgroup(0)]]` region, to `maxComputeSharedMemorySize`: exactly full plans, one
/// more cell does not.
#[test]
fn the_binder_specializes_and_charges_the_imageblock() {
    let Some(bc) = owned() else { return };
    for stem in ["moltenvk-m3", "lavapipe", "vk12-floor"] {
        let dev = device(stem);
        let pipe = pipeline(&dev, &bc, "imageblock_after_threadgroup");
        assert!(pipe
            .dependencies
            .iter()
            .any(|d| d.fact == RuntimeFact::ImageblockSize { cell_bytes: 8 }));
        let state = |w: u32, h: u32| {
            let mut table = StageTable::default();
            table.threadgroup_lengths.insert(0, 64);
            let mut s = BindState::default();
            s.stages.insert(Stage::Kernel, table);
            s.imageblock_size = ImageblockSize::new(w, h);
            s
        };
        for (w, h) in [(0, 0), (4, 0), (65536, 1)] {
            assert_eq!(
                pipe.plan(&dev, &state(w, h).shape()).map(|_| ()),
                Err(BindError::InvalidImageblockSize {
                    width: w,
                    height: h
                }),
                "{stem}"
            );
        }
        let plan = pipe.plan(&dev, &state(4, 3).shape()).expect("plan");
        assert_eq!(specialized(&pipe, &plan, SpecKey::ImageblockWidth), Some(4));
        assert_eq!(
            specialized(&pipe, &plan, SpecKey::ImageblockHeight),
            Some(3)
        );
        assert_eq!(specialized(&pipe, &plan, SpecKey::TgLength(0)), Some(64));
        // 64 region bytes, then 8-byte cells: `cells` of them fill the limit exactly.
        let max = dev.profile().fields().limits.max_compute_shared_memory_size;
        let cells = (max - 64) / 8;
        assert!(pipe.plan(&dev, &state(cells, 1).shape()).is_ok(), "{stem}");
        match pipe.plan(&dev, &state(cells + 1, 1).shape()) {
            Err(BindError::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::ThreadgroupMemorySize, "{stem}");
                assert!(format!("{:?}", u.needs).contains("MaxComputeSharedMemorySize"));
            }
            other => panic!("{stem}: {:?}", other.map(|_| ())),
        }
    }
}

/// On a profile that writes storage images without a format, the binder keys an integer slice's
/// slot although the formatless declaration would serve its format: the function compiled under
/// the slot's choice depends on the numeric type it keeps (`StorageTextureNumericType`), so the
/// plan of a signed bound format asks for exactly the key it was compiled under, which every signed
/// format shares. The float slot keys nothing.
#[test]
fn a_formatless_profile_keys_an_integer_slice_type() {
    let Some(bc) = owned() else { return };
    let dev = device("moltenvk-m3");
    let images = &dev.profile().fields().images;
    assert!(images.storage_read_without_format && images.storage_write_without_format);
    let budget = Budget::standard();
    let lib = Library::from_air(&bc).expect("library");
    let f = FunctionDescriptor::new(lib.function("imageblock_runtime_region").expect("function"));
    match dev.compile_function(&f, &LowerInputs::new(), &budget) {
        Err(Error::NeedsInput(RuntimeInput::StorageTextureFormat { slot })) => {
            assert_eq!(slot, kernel_texture(0));
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
    let mut key = VariantKey::new();
    let choice = storage_image_choice(PixelFormat::RGBA16Sint, dev.profile()).expect("choice");
    assert_eq!(choice.vk_format, vk::Format::UNDEFINED);
    key.storage_images.insert(kernel_texture(0), choice);
    let compiled = dev
        .compile_function(&f, &key.lowering_inputs(Stage::Kernel), &budget)
        .expect("compiles under the choice");
    let pipe = dev
        .compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
            &budget,
        )
        .expect("pipeline");
    assert!(pipe.dependencies.iter().any(|d| d.fact
        == RuntimeFact::StorageTextureNumericType {
            slot: kernel_texture(0)
        }
        && d.mechanism == Mechanism::Variant));
    let state = |code: PixelFormat| {
        let mut table = StageTable::default();
        for (index, pf) in [(0, code), (1, PixelFormat::R32Float)] {
            let mut facts = TextureFacts::new(TextureType::Type2D, pf);
            facts.usage = TextureUsage::SHADER_WRITE;
            table.textures.insert(
                index,
                BoundTexture::Texture {
                    facts,
                    resource_id: None,
                },
            );
        }
        let mut s = BindState::default();
        s.stages.insert(Stage::Kernel, table);
        s.imageblock_size = ImageblockSize::new(4, 4);
        s
    };
    // Every signed format shares the key; the float slot keys nothing.
    for pf in [PixelFormat::RGBA16Sint, PixelFormat::R32Sint] {
        let plan = pipe.plan(&dev, &state(pf).shape()).expect("plan");
        assert_eq!(plan.variant, key, "{pf:?}");
    }
    assert_eq!(
        key.storage_images[&kernel_texture(0)].numeric,
        Some(SampledType::Sint)
    );
}

/// A two-lane slice plans against a bound format of any component count, as Metal writes its
/// lanes and zeros whatever the format; an unbound texture drops the write.
#[test]
fn a_narrow_slice_plans_against_any_bound_format() {
    let Some(bc) = owned() else { return };
    let dev = device("moltenvk-m3");
    let pipe = pipeline(&dev, &bc, "imageblock_sized_slice");
    let state = |pf: Option<PixelFormat>| {
        let mut table = StageTable::default();
        if let Some(pf) = pf {
            let mut facts = TextureFacts::new(TextureType::Type2D, pf);
            facts.usage = TextureUsage::SHADER_WRITE;
            table.textures.insert(
                0,
                BoundTexture::Texture {
                    facts,
                    resource_id: None,
                },
            );
        }
        let mut s = BindState::default();
        s.stages.insert(Stage::Kernel, table);
        s.imageblock_size = ImageblockSize::new(4, 4);
        s
    };
    for pf in [
        Some(PixelFormat::RG16Float),
        None,
        Some(PixelFormat::R16Float),
        Some(PixelFormat::RGBA16Float),
        Some(PixelFormat::RGBA32Float),
    ] {
        if let Err(e) = pipe.plan(&dev, &state(pf).shape()) {
            panic!("{pf:?}: {e:?}");
        }
    }
}

/// A kernel whose 4x4 threads each store their cell `(x, y, 7, 9)`, then, after the imageblock
/// barrier and `between`, under the guard `guard` (which defines `%g`), write the whole imageblock
/// to texture 0 at their threadgroup's position times 4: the shape `texture::cooperative` shares
/// out when the guard selects local id 0 right after the barrier.
fn shared_slice_kernel(between: &str, guard: &str) -> String {
    format!(
        r#"%"struct.metal::_imageblock_base" = type {{ ptr addrspace(4) }}

define void @shared_slice(%"struct.metal::_imageblock_base" %block, ptr addrspace(1) %out, <2 x i16> %lid, <2 x i16> %tgp, ptr addrspace(1) %log) {{
entry:
  %cell = call ptr addrspace(4) @air.imageblock_data(<2 x i16> %lid, i32 0, i16 0)
  %x = extractelement <2 x i16> %lid, i64 0
  %y = extractelement <2 x i16> %lid, i64 1
  %xf = uitofp i16 %x to float
  %yf = uitofp i16 %y to float
  %c0 = insertelement <4 x float> <float 0.0, float 0.0, float 7.0, float 9.0>, float %xf, i64 0
  %c1 = insertelement <4 x float> %c0, float %yf, i64 1
  store <4 x float> %c1, ptr addrspace(4) %cell, align 16
  call void @air.wg.barrier(i32 8, i32 1)
{between}{guard}  br i1 %g, label %write, label %done

write:
  %first = call ptr addrspace(4) @air.imageblock_data(<2 x i16> zeroinitializer, i32 0, i16 0)
  %at = shl <2 x i16> %tgp, <i16 2, i16 2>
  call void @air.write_imageblock_slice_to_texture_2d.i16.v4f32(ptr addrspace(1) %out, ptr addrspace(4) %first, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i16> %at, i16 0, i1 false, i32 2)
  br label %done

done:
  ret void
}}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.wg.barrier(i32, i32)
declare void @air.write_imageblock_slice_to_texture_2d.i16.v4f32(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)

!air.kernel = !{{!0}}
!0 = !{{ptr @shared_slice, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !5, !6, !7, !8}}
!3 = !{{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 16, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"imageblock<Cell, layout_explicit>", !"air.arg_name", !"block"}}
!4 = !{{i32 0, i32 16, i32 0, !"float4", !"colour"}}
!5 = !{{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"out"}}
!6 = !{{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"lid"}}
!7 = !{{i32 3, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tgp"}}
!8 = !{{i32 4, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"log"}}
"#
    )
}

/// The guard `(lid.x | lid.y) == 0`, which holds at local id 0 alone.
const ORIGIN_GUARD: &str = "  %o = or i16 %x, %y\n  %g = icmp eq i16 %o, 0\n";

/// Run `shared_slice` (one threadgroup of 4x4 threads, a 4x4 imageblock) on every profile that
/// compiles it without inputs: whether its slice is shared (`TexturePlan::cooperative`), and the
/// texture every invocation of both scheduling orders leaves, each texel the cell's `(x, y, 7, 9)`.
fn run_shared_slice(name: &str, between: &str, guard: &str) -> Option<Vec<bool>> {
    let bc = assemble_text(name, &shared_slice_kernel(between, guard))?;
    let mut shared = Vec::new();
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some((plan, m)) = module(&bc, "shared_slice", &stem, &p) else {
            continue;
        };
        let d = Dispatch {
            local: [4, 4],
            block: [4, 4],
            buffers: &[vec![0; 16]],
            threadgroup: &[],
        };
        let (_, imgs) = run(&plan, &m, &d, &[poisoned(0, [4, 4], 1)]);
        for y in 0..4u32 {
            for x in 0..4u32 {
                assert_eq!(
                    texel(&imgs[0], 0, x, y),
                    [fl(x as f32), fl(y as f32), fl(7.0), fl(9.0)],
                    "{name} {stem} ({x}, {y})"
                );
            }
        }
        shared.push(!plan.families.texture.cooperative.is_empty());
    }
    assert!(!shared.is_empty(), "{name}: no profile compiled it");
    Some(shared)
}

/// A slice write right after the imageblock barrier, under a guard that holds at local id 0, with
/// operands every invocation holds alike (the threadgroup's position), is written by every
/// invocation, a strided share each (`texture::cooperative`): the guard's branch is taken by all,
/// and the texture holds every cell exactly as the one writing invocation leaves it.
#[test]
fn a_slice_written_by_local_id_zero_after_the_barrier_is_shared() {
    let Some(shared) = run_shared_slice("shared", "", ORIGIN_GUARD) else {
        return;
    };
    assert!(shared.iter().all(|s| *s), "{shared:?}");
}

/// Fallbacks to the writing invocation's own loop, the same texels: a guard that selects another
/// invocation (`lid.x == 1`), a store between the barrier and the guard, and a guard on a value the
/// local id alone does not decide (a loaded word).
#[test]
fn a_slice_outside_the_shared_shape_keeps_its_writer() {
    let late_store = "  %li = zext i16 %x to i64\n  %lp = getelementptr inbounds i32, ptr addrspace(1) %log, i64 %li\n  store i32 1, ptr addrspace(1) %lp, align 4\n";
    let loaded = "  %w = load i32, ptr addrspace(1) %log, align 4\n  %g = icmp eq i32 %w, 0\n";
    for (name, between, guard) in [
        ("other_id", "", "  %g = icmp eq i16 %x, 1\n"),
        ("late_store", late_store, ORIGIN_GUARD),
        ("loaded_guard", "", loaded),
    ] {
        let Some(shared) = run_shared_slice(name, between, guard) else {
            return;
        };
        assert!(shared.iter().all(|s| !*s), "{name}: {shared:?}");
    }
}

/// A kernel whose 4x4 threads each store the `half4` at their linear index in buffer 0 in their
/// cell, then, after the imageblock barrier and under the guard `guard` (which defines `%g`),
/// write the imageblock's `half4` slice to texture 0 at (0, 0).
fn half_slice_kernel(guard: &str) -> String {
    format!(
        r#"%"struct.metal::_imageblock_base" = type {{ ptr addrspace(4) }}

define void @half_slice(%"struct.metal::_imageblock_base" %block, ptr addrspace(1) %out, <2 x i16> %lid, ptr addrspace(1) %src) {{
entry:
  %cell = call ptr addrspace(4) @air.imageblock_data(<2 x i16> %lid, i32 0, i16 0)
  %x = extractelement <2 x i16> %lid, i64 0
  %y = extractelement <2 x i16> %lid, i64 1
  %row = shl i16 %y, 2
  %i = add i16 %row, %x
  %ii = zext i16 %i to i64
  %p = getelementptr inbounds <4 x half>, ptr addrspace(1) %src, i64 %ii
  %h = load <4 x half>, ptr addrspace(1) %p, align 8
  store <4 x half> %h, ptr addrspace(4) %cell, align 8
  call void @air.wg.barrier(i32 8, i32 1)
{guard}  br i1 %g, label %write, label %done

write:
  %first = call ptr addrspace(4) @air.imageblock_data(<2 x i16> zeroinitializer, i32 0, i16 0)
  call void @air.write_imageblock_slice_to_texture_2d.i16.v4f16(ptr addrspace(1) %out, ptr addrspace(4) %first, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i16> zeroinitializer, i16 0, i1 false, i32 2)
  br label %done

done:
  ret void
}}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.wg.barrier(i32, i32)
declare void @air.write_imageblock_slice_to_texture_2d.i16.v4f16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)

!air.kernel = !{{!0}}
!0 = !{{ptr @half_slice, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !5, !6, !7}}
!3 = !{{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 8, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"imageblock<Cell, layout_explicit>", !"air.arg_name", !"block"}}
!4 = !{{i32 0, i32 8, i32 0, !"half4", !"colour"}}
!5 = !{{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"out"}}
!6 = !{{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"lid"}}
!7 = !{{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"src"}}
"#
    )
}

/// Half bit patterns a widening or a narrowing could change: NaNs of each sign, quiet and
/// signaling, with payloads; infinities; zeros; denormals; the largest finite values; and
/// ordinary values.
const HALF_PATTERNS: [u16; 20] = [
    0x7C01, 0xFE55, 0x7E00, 0x7FFF, 0xFC01, 0x7C00, 0xFC00, 0x0000, 0x8000, 0x0001, 0x83FF, 0x03FF,
    0x0400, 0x7BFF, 0xFBFF, 0x3C00, 0xC000, 0x5555, 0x1234, 0xBEEF,
];

/// A `half` slice written to an `RGBA16Float` texture hands the device each cell's halves widened
/// exactly, in the shared form (the guard local id 0 after the barrier) and the writing
/// invocation's (another invocation's guard): each texel's channels are the cell's halves'
/// widenings (the interpreter keeps `f32` texels), signaling NaNs and denormals included, and a
/// NaN for each NaN, whose stored bits are the device's conversion's.
#[test]
fn a_half_slice_hands_the_device_the_cells_halves() {
    use crate::lower::arith::narrow::f32_of_half_bits;
    let lane = |i: usize, l: usize| HALF_PATTERNS[(i * 4 + l) % HALF_PATTERNS.len()];
    let mut src = Vec::new();
    for i in 0..16 {
        for l in 0..4 {
            src.extend_from_slice(&lane(i, l).to_le_bytes());
        }
    }
    let mut ran = 0;
    for (name, guard) in [
        ("shared", ORIGIN_GUARD),
        ("writer", "  %g = icmp eq i16 %x, 1\n"),
    ] {
        let Some(bc) = assemble_text(name, &half_slice_kernel(guard)) else {
            return;
        };
        for (stem, p) in committed_profiles().expect("profiles") {
            let Some(inputs) = choices(&p, &[(0, PixelFormat::RGBA16Float)]) else {
                continue;
            };
            let class = inputs.storage_images[&kernel_texture(0)].class;
            assert_eq!(class, StorageImageClass::Identity, "{name} {stem}");
            let (plan, m) = match typed_module_with(&bc, "half_slice", &p, &inputs) {
                Ok(x) => x,
                Err(Error::NeedsInput(_) | Error::Unsupported(_)) => continue,
                Err(e) => panic!("{name} {stem}: {e}"),
            };
            assert_eq!(
                plan.families.texture.cooperative.is_empty(),
                name == "writer",
                "{name} {stem}"
            );
            let d = Dispatch {
                local: [4, 4],
                block: [4, 4],
                buffers: &[src.clone()],
                threadgroup: &[],
            };
            let (_, imgs) = run(&plan, &m, &d, &[poisoned(0, [4, 4], 1)]);
            for y in 0..4u32 {
                for x in 0..4u32 {
                    let i = (y * 4 + x) as usize;
                    let got = texel(&imgs[0], 0, x, y);
                    for (l, &t) in got.iter().enumerate() {
                        let h = lane(i, l);
                        let what = format!("{name} {stem} ({x}, {y}) lane {l} {h:#06x}: {t:#x}");
                        if h & 0x7C00 == 0x7C00 && h & 0x3FF != 0 {
                            assert!(f32::from_bits(t).is_nan(), "{what}");
                        } else {
                            assert_eq!(t, f32_of_half_bits(u32::from(h)), "{what}");
                        }
                    }
                }
            }
            ran += 1;
        }
    }
    assert!(ran > 0, "no profile writes a half slice");
}

/// `kernel_imageblock_wide_access` (owned, `engine/`): a `{half v}` cell (data size 2, align 2)
/// stored through its pointer as a `half2` and sliced as one from its start (`ib_half_cell`), and
/// the `{half2 v}` twin (`ib_half2_cell`). A cell holds the widest access through its pointer
/// (`RuntimeFact::ImageblockSize { cell_bytes: 4 }` for both), so each thread's `half2` is its
/// cell's own and thread (0, 0)'s slice writes every pixel of the 8x8 block its own `(x, y)`, as
/// Metal on the M3 writes for both kernels (its imageblock is 8 bytes a pixel for either).
#[test]
fn a_cell_holds_the_widest_access_through_its_pointer() {
    let Some(bc) = assemble(engine_dir().join("kernel_imageblock_wide_access.ll")) else {
        return;
    };
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        for entry in ["ib_half_cell", "ib_half2_cell"] {
            let Some((plan, m)) = module(&bc, entry, &stem, &p) else {
                continue;
            };
            assert!(
                plan.facts
                    .contains(&RuntimeFact::ImageblockSize { cell_bytes: 4 }),
                "{stem}: {entry}: {:?}",
                plan.facts
            );
            let d = Dispatch {
                local: [8, 8],
                block: [8, 8],
                buffers: &[],
                threadgroup: &[],
            };
            let (_, imgs) = run(&plan, &m, &d, &[poisoned(0, [8, 8], 1)]);
            for y in 0..8 {
                for x in 0..8 {
                    let want = [fl(x as f32), fl(y as f32), 0, 0];
                    assert_eq!(texel(&imgs[0], 0, x, y), want, "{stem}: {entry} ({x}, {y})");
                }
            }
            ran += 1;
        }
    }
    assert!(ran > 0);
}
