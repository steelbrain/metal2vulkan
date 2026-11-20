//! Storage textures at a mipmap level.
//!
//! A storage slot read, written, slice-written, or size-queried at a level other than the
//! constant 0 is declared as its level views: an array of single-level storage views, element
//! `i` the bound texture's level `i` and every element past its last level the neutral, as many
//! elements as the device's largest texture of the slot's type has levels. The binder pushes the
//! bound level count (`PushField::TextureLevels`), which also answers `get_num_mip_levels` of a
//! storage texture. A write at a level past the count is out of bounds and dropped, as Metal
//! ignores it (MSL 4.1 §6.13); a read there is Metal's out-of-bounds texel, zero components with
//! the format's missing alpha read as 1 (its alpha from the `OutOfBoundsAlpha` specialization), on
//! every profile class; and a size query, which no rule covers, takes the last level. A 1D
//! texture's level count is always 0.
//!
//! The fixtures (`validation/fixtures/public/engine/txt_level_*.metal`) run on the CPU
//! interpreter's image model with the views bound as the binder binds them, in both scheduling
//! orders; every compiled module passes `spirv-val` under its profile (`G-val`). The binder
//! itself is checked end to end on two devices: the plan's level views and pushed count drive the
//! interpreter run.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::air::intrinsics::ImageDim;
use crate::bind::{BindState, BoundBuffer, BoundTexture, Dispatch, StageTable, TextureFacts};
use crate::bind::{BindingPlan, FieldLocation, PlanDiagnostic, WriteSource};
use crate::budget::Budget;
use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkFeature;
use crate::device::requirements::VkLimit;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{BindError, Capability, Error, Feature, LowerFamily, Unsupported};
use crate::library::{FunctionArtifact, Library};
use crate::lower::select::Plan;
use crate::lower::texture::max_levels;
use crate::metal::{PixelFormat, TextureType, TextureUsage};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::layout::BindingSource;
use crate::pipeline::sym::{
    Axis, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage,
};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, profile, typed_module, AssembleError,
};
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, StageSource,
};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;
use crate::spirv::testing::threadgroup_profiles;

// ---- Harness ----

fn fixture(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
    }
}

const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

/// A synthetic kernel's bitcode.
fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let text = format!(
        "target datalayout = \"{DATALAYOUT}\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n{body}"
    );
    let dir = std::env::temp_dir().join(format!(
        "m2v-texture-levels-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, &text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}\n{text}"),
    }
}

/// The refusal of a dynamic level on a device that cannot index storage-image arrays by a
/// non-uniform value.
fn no_non_uniform() -> Unsupported {
    Unsupported::needs(
        Feature::Lowering(LowerFamily::Texture),
        Capability::Feature(VkFeature::ShaderStorageImageArrayNonUniformIndexing),
    )
}

fn non_uniform(p: &DeviceProfile) -> bool {
    p.fields().descriptors.non_uniform_storage_image
}

fn is(e: &Error, want: &Unsupported) -> bool {
    matches!(e, Error::Unsupported(u) if (&u.feature, &u.needs) == (&want.feature, &want.needs))
}

/// The module of `entry` on `p`, compiled and validated (L3), or `None` when `p` needs a runtime
/// input for it (a storage format) or cannot index the level views by a dynamic level.
fn module(bc: &[u8], entry: &str, stem: &str, p: &DeviceProfile) -> Option<(Plan, TypedModule)> {
    match typed_module(bc, entry, p) {
        Ok(m) => {
            let a = compile(bc, entry, p).unwrap_or_else(|e| panic!("{stem}: {entry}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: {entry}: L3: {e}"));
            Some(m)
        }
        Err(Error::NeedsInput(_)) => None,
        Err(e) if is(&e, &no_non_uniform()) && !non_uniform(p) => None,
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

fn tslot(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index,
    }
}

/// A texel the kernels never write.
const POISON: [u32; 4] = [0xDEAD_0001, 0xDEAD_0002, 0xDEAD_0003, 0xDEAD_0004];

/// The `n` level views of slot `index` as the binder binds them: element `i < levels` is level
/// `i` of an `extent` texture of `layers` layers alone, every later element the 1x1 neutral.
/// Every texel holds [`POISON`].
fn views(index: u32, extent: [u32; 3], layers: u32, levels: u32, n: u32) -> Vec<Image> {
    let whole = Image::new(texture(index), extent, layers, levels, 1);
    (0..n)
        .map(|i| {
            let e = if i < levels {
                whole.level_extent(i)
            } else {
                [1, 1, 1]
            };
            let mut img = Image::new(texture(index), e, layers, 1, 1).at_element(i);
            img.texels.fill(POISON);
            img
        })
        .collect()
}

/// One dispatch: `[[buffer(i)]]` holds `buffers[i]`, slot `s` has level count `n` for each
/// `(s, n)` of `levels`, `grid` threadgroups of `local` threads, and `spec` beyond the local
/// size.
struct Run<'a> {
    buffers: &'a [Vec<u8>],
    levels: &'a [(u32, u32)],
    grid: [u32; 3],
    local: [u32; 3],
    spec: &'a [(SpecKey, u64)],
}

impl Run<'_> {
    fn new<'a>(buffers: &'a [Vec<u8>], levels: &'a [(u32, u32)], grid: [u32; 3]) -> Run<'a> {
        Run {
            buffers,
            levels,
            grid,
            local: [1, 1, 1],
            spec: &[],
        }
    }
}

/// The push bytes of `plan`: each slot's level count at its field, every bind offset 0.
fn push_bytes(plan: &Plan, levels: &[(u32, u32)]) -> Vec<u8> {
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::TextureLevels(slot), PushLoc::Push { offset }) => {
                let n = levels
                    .iter()
                    .find(|(s, _)| *s == slot.index)
                    .map(|(_, n)| *n)
                    .unwrap_or_else(|| panic!("no level count for {slot:?}"));
                push[offset as usize..offset as usize + 4].copy_from_slice(&n.to_le_bytes());
            }
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            // A bound value read as data, which the interpreter writes from `spec`.
            (PushField::Specialized(_), PushLoc::Push { .. }) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    push
}

/// Run `m` in both scheduling orders (which must agree): the buffers and images after.
fn run(
    plan: &Plan,
    m: &TypedModule,
    r: &Run<'_>,
    images: &[Image],
) -> Result<(Vec<Vec<u8>>, Vec<Image>), String> {
    let mut spec = BTreeMap::new();
    for (axis, n) in [Axis::X, Axis::Y, Axis::Z].into_iter().zip(r.local) {
        spec.insert(SpecKey::LocalSize(axis), u64::from(n));
    }
    spec.insert(
        SpecKey::FlatLocalSize,
        u64::from(r.local.iter().product::<u32>()),
    );
    spec.extend(r.spec.iter().copied());
    let buffers: Vec<Buffer> = r
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
        push: push_bytes(plan, r.levels),
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: r.grid,
            order,
            ..Config::default()
        };
        let (o, imgs) =
            run_with_images(m, &cfg, res.clone(), images.to_vec()).map_err(|e| format!("{e:?}"))?;
        outs.push((o.buffers, imgs));
    }
    assert_eq!(outs[0], outs[1], "scheduling orders disagree");
    Ok(outs.swap_remove(0))
}

/// A `constant` level buffer holding `lod` (as `uint`, or `ushort` in its low bytes), padded to
/// the 16 bytes the consumer allocates.
fn lod_bytes(lod: u32) -> Vec<u8> {
    let mut b = lod.to_le_bytes().to_vec();
    b.resize(16, 0);
    b
}

fn fl(x: f32) -> u32 {
    x.to_bits()
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Every texel of `img` (`layer`, `(x, y)`) is `want(layer, x, y)`.
fn texels(img: &Image, want: impl Fn(u32, u32, u32) -> [u32; 4], what: &str) {
    for layer in 0..img.layers {
        for y in 0..img.extent[1] {
            for x in 0..img.extent[0] {
                let i = img.index(0, layer, 0, [x, y, 0]).expect("texel");
                assert_eq!(
                    img.texels[i],
                    want(layer, x, y),
                    "{what}: element {} layer {layer} ({x}, {y})",
                    img.element
                );
            }
        }
    }
}

fn uses(a: &FunctionArtifact, name: &str) -> u32 {
    a.strategies.entries.get(name).map_or(0, |u| u.uses)
}

// ---- Writes, queries, reads, slices ----

/// `txt_level_write`: a `texture2d<half, write>` written at `gid` of the level `buffer(0)` holds.
/// Each level of a 3-level 4x4 texture takes exactly its own texels; a level past the count
/// (3, and 0 of the unbound slot whose count is 0) writes nothing.
#[test]
fn a_write_lands_on_its_level_alone() {
    let Some(bc) = fixture("txt_level_write") else {
        return;
    };
    let (mut ran, mut refused) = (0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some((plan, m)) = module(&bc, "txt_level_write", &stem, &p) else {
            refused += 1;
            continue;
        };
        let n = max_levels(&p, ImageDim::D2);
        for lod in 0..4u32 {
            let imgs = views(0, [4, 4, 1], 1, 3, n);
            let extent = imgs[lod.min(2) as usize].extent;
            let lodb = lod_bytes(lod);
            let r = Run::new(
                std::slice::from_ref(&lodb),
                &[(0, 3)],
                [extent[0], extent[1], 1],
            );
            let (_, out) = run(&plan, &m, &r, &imgs).unwrap_or_else(|e| panic!("{stem}: {e}"));
            for img in &out {
                texels(
                    img,
                    |_, x, y| {
                        if img.element == lod && lod < 3 {
                            [fl(x as f32), fl(y as f32), fl(1.0), fl(0.5)]
                        } else {
                            POISON
                        }
                    },
                    &format!("{stem} lod {lod}"),
                );
            }
        }
        // The neutral stands in: the count is 0 and no element is written.
        let imgs = views(0, [1, 1, 1], 1, 0, n);
        let lodb = lod_bytes(0);
        let r = Run::new(std::slice::from_ref(&lodb), &[(0, 0)], [1, 1, 1]);
        let (_, out) = run(&plan, &m, &r, &imgs).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(out, imgs, "{stem}: the neutral is not written");
        ran += 1;
    }
    assert!(ran > 0 && refused > 0, "ran {ran}, refused {refused}");
}

/// `txt_level_queries`: `get_num_mip_levels` of a storage texture reads the pushed count (of a 2D
/// slot and of a cube slot), and `get_width`/`get_height` at a level answer Metal's
/// `max(1, e >> (l & 255))` of the first level's extent `e` (`emit::level_extent`), past the count
/// too; a slot bound to nothing answers its neutral's.
#[test]
fn storage_level_queries_read_the_bound_levels() {
    let Some(bc) = fixture("txt_level_queries") else {
        return;
    };
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some((plan, m)) = module(&bc, "txt_level_queries", &stem, &p) else {
            continue;
        };
        let n = max_levels(&p, ImageDim::D2);
        let at = |lod: u32, t: (u32, u32), c: u32| {
            let mut imgs = views(0, [8, 4, 1], 1, t.1, n);
            if t.1 == 0 {
                imgs = views(0, [1, 1, 1], 1, 0, n);
            }
            imgs.push(Image::new(texture(1), [2, 2, 1], 6, c.max(1), 1));
            let bufs = [lod_bytes(lod), vec![0; 16]];
            let counts = [(0, t.1), (1, c)];
            let r = Run::new(&bufs, &counts, [1, 1, 1]);
            let (out, _) = run(&plan, &m, &r, &imgs).unwrap_or_else(|e| panic!("{stem}: {e}"));
            words(&out[1])
        };
        assert_eq!(at(0, (8, 3), 2), [3, 8, 4, 2], "{stem} lod 0");
        assert_eq!(at(1, (8, 3), 2), [3, 4, 2, 2], "{stem} lod 1");
        assert_eq!(at(2, (8, 3), 1), [3, 2, 1, 1], "{stem} lod 2");
        assert_eq!(
            at(7, (8, 3), 2),
            [3, 1, 1, 2],
            "{stem} lod 7, past the count"
        );
        assert_eq!(at(257, (8, 3), 2), [3, 4, 2, 2], "{stem} lod 257, as lod 1");
        assert_eq!(
            at(1, (1, 0), 0),
            [0, 1, 1, 0],
            "{stem} unbound: the neutral's"
        );
        ran += 1;
    }
    assert!(ran > 0);
}

/// `txt_level_cube_write`: a `texturecube<float, write>` written at face `gid.z` of a level lands
/// on that face of that level's view.
#[test]
fn a_cube_write_lands_on_the_face_of_its_level() {
    let Some(bc) = fixture("txt_level_cube_write") else {
        return;
    };
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some((plan, m)) = module(&bc, "txt_level_cube_write", &stem, &p) else {
            continue;
        };
        let n = max_levels(&p, ImageDim::Cube);
        for lod in [0u32, 1, 2, 3] {
            let imgs = views(0, [4, 4, 1], 6, 3, n);
            let e = imgs[lod.min(2) as usize].extent;
            let lodb = lod_bytes(lod);
            let r = Run::new(std::slice::from_ref(&lodb), &[(0, 3)], [e[0], e[1], 6]);
            let (_, out) = run(&plan, &m, &r, &imgs).unwrap_or_else(|e| panic!("{stem}: {e}"));
            for img in &out {
                texels(
                    img,
                    |face, x, y| {
                        if img.element == lod && lod < 3 {
                            [fl(x as f32), fl(y as f32), fl(face as f32), fl(1.0)]
                        } else {
                            POISON
                        }
                    },
                    &format!("{stem} lod {lod}"),
                );
            }
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// `txt_level_read_write`: a `read_write` texture read and written at one level reads and writes
/// that level's view alone; past the count the write is dropped (the read there is
/// `a_read_at_a_missing_level_is_metals_out_of_bounds_texel`'s).
#[test]
fn a_read_write_texture_reads_and_writes_its_level() {
    let Some(bc) = fixture("txt_level_read_write") else {
        return;
    };
    let mut ran = 0;
    let seed = |e: u32, x: u32, y: u32| [fl((10 * e + x) as f32), fl(y as f32), fl(0.0), fl(1.0)];
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some((plan, m)) = module(&bc, "txt_level_read_write", &stem, &p) else {
            continue;
        };
        let n = max_levels(&p, ImageDim::D2);
        for lod in [0u32, 1, 2, 5] {
            let mut imgs = views(0, [4, 4, 1], 1, 3, n);
            for img in imgs.iter_mut().take(3) {
                for y in 0..img.extent[1] {
                    for x in 0..img.extent[0] {
                        let i = img.index(0, 0, 0, [x, y, 0]).expect("texel");
                        img.texels[i] = seed(img.element, x, y);
                    }
                }
            }
            let e = imgs[lod.min(2) as usize].extent;
            let lodb = lod_bytes(lod);
            let r = Run::new(std::slice::from_ref(&lodb), &[(0, 3)], [e[0], e[1], 1]);
            let (_, out) = run(&plan, &m, &r, &imgs).unwrap_or_else(|e| panic!("{stem}: {e}"));
            for img in &out {
                texels(
                    img,
                    |_, x, y| match img.element {
                        k if k >= 3 => POISON,
                        k if k == lod => {
                            let [a, b, c, d] = seed(k, x, y).map(f32::from_bits);
                            [a, b, c, d].map(|v| fl(2.0 * v + 1.0))
                        }
                        k => seed(k, x, y),
                    },
                    &format!("{stem} lod {lod}"),
                );
            }
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// `txt_level_slice`: a 2x2 imageblock's color plane written as a slice at `(0, 0)` of a level
/// lands on that level alone; past the count it is dropped.
#[test]
fn a_slice_lands_on_its_level() {
    let Some(bc) = fixture("txt_level_slice") else {
        return;
    };
    let mut ran = 0;
    for (stem, p) in threadgroup_profiles().expect("profiles") {
        let Some((plan, m)) = module(&bc, "txt_level_slice", &stem, &p) else {
            continue;
        };
        assert!(
            plan.facts
                .iter()
                .any(|f| matches!(f, RuntimeFact::ImageblockSize { .. })),
            "{stem}: the plan depends on the imageblock size"
        );
        let n = max_levels(&p, ImageDim::D2);
        for lod in [0u32, 1, 3] {
            let imgs = views(0, [4, 4, 1], 1, 3, n);
            let lodb = lod_bytes(lod);
            let r = Run {
                local: [2, 2, 1],
                spec: &[
                    (SpecKey::ImageblockWidth, 2),
                    (SpecKey::ImageblockHeight, 2),
                ],
                ..Run::new(std::slice::from_ref(&lodb), &[(0, 3)], [1, 1, 1])
            };
            let (_, out) = run(&plan, &m, &r, &imgs).unwrap_or_else(|e| panic!("{stem}: {e}"));
            for img in &out {
                texels(
                    img,
                    |_, x, y| {
                        if img.element == lod && lod < 3 && x < 2 && y < 2 {
                            [fl(x as f32), fl(y as f32), fl(1.0), fl(0.5)]
                        } else {
                            POISON
                        }
                    },
                    &format!("{stem} lod {lod}"),
                );
            }
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// Seven kernels over a `read_write` texture at slot 0, each storing to `out[i]` (buffer 1) the
/// texel read at `p[i]` (buffer 0, `uint4`) of a level. Over `texture2d`: `read_level`
/// (`float`) and `read_level_uint` (`uint`) at `p[i].xy` of level `p[i].z`, `read_level_rgb`
/// the same with its alpha replaced by 2 (the alpha is never observed), and `read_level_two` at
/// `p[i].xy` of the constant level 2. Of level `p[i].w`: `read_level_array` at `p[i].xy` of
/// slice `p[i].z` of a `texture2d_array<float>`, `read_level_3d` at `p[i].xyz` of a
/// `texture3d<float>`, and `read_level_cube` at `p[i].xy` of face `p[i].z` of a
/// `texturecube<float>`.
const LEVEL_READS: &str = r#"define void @read_level(ptr addrspace(1) %t, ptr addrspace(1) %p, ptr addrspace(1) %out, i32 %i) {
  %w = zext i32 %i to i64
  %pp = getelementptr inbounds <4 x i32>, ptr addrspace(1) %p, i64 %w
  %r = load <4 x i32>, ptr addrspace(1) %pp, align 16
  %c = shufflevector <4 x i32> %r, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %l = extractelement <4 x i32> %r, i64 2
  %s = call ptr addrspace(2) @air.get_read_sampler()
  %v = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %c, <2 x i32> zeroinitializer, i32 %l, i32 3)
  %x = extractvalue { <4 x float>, i8 } %v, 0
  %o = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %w
  store <4 x float> %x, ptr addrspace(1) %o, align 16
  ret void
}

define void @read_level_uint(ptr addrspace(1) %t, ptr addrspace(1) %p, ptr addrspace(1) %out, i32 %i) {
  %w = zext i32 %i to i64
  %pp = getelementptr inbounds <4 x i32>, ptr addrspace(1) %p, i64 %w
  %r = load <4 x i32>, ptr addrspace(1) %pp, align 16
  %c = shufflevector <4 x i32> %r, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %l = extractelement <4 x i32> %r, i64 2
  %s = call ptr addrspace(2) @air.get_read_sampler()
  %v = call { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %c, <2 x i32> zeroinitializer, i32 %l, i32 3)
  %x = extractvalue { <4 x i32>, i8 } %v, 0
  %o = getelementptr inbounds <4 x i32>, ptr addrspace(1) %out, i64 %w
  store <4 x i32> %x, ptr addrspace(1) %o, align 16
  ret void
}

define void @read_level_rgb(ptr addrspace(1) %t, ptr addrspace(1) %p, ptr addrspace(1) %out, i32 %i) {
  %w = zext i32 %i to i64
  %pp = getelementptr inbounds <4 x i32>, ptr addrspace(1) %p, i64 %w
  %r = load <4 x i32>, ptr addrspace(1) %pp, align 16
  %c = shufflevector <4 x i32> %r, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %l = extractelement <4 x i32> %r, i64 2
  %s = call ptr addrspace(2) @air.get_read_sampler()
  %v = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %c, <2 x i32> zeroinitializer, i32 %l, i32 3)
  %x = extractvalue { <4 x float>, i8 } %v, 0
  %y = insertelement <4 x float> %x, float 2.0, i64 3
  %o = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %w
  store <4 x float> %y, ptr addrspace(1) %o, align 16
  ret void
}

define void @read_level_two(ptr addrspace(1) %t, ptr addrspace(1) %p, ptr addrspace(1) %out, i32 %i) {
  %w = zext i32 %i to i64
  %pp = getelementptr inbounds <4 x i32>, ptr addrspace(1) %p, i64 %w
  %r = load <4 x i32>, ptr addrspace(1) %pp, align 16
  %c = shufflevector <4 x i32> %r, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %s = call ptr addrspace(2) @air.get_read_sampler()
  %v = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %c, <2 x i32> zeroinitializer, i32 2, i32 3)
  %x = extractvalue { <4 x float>, i8 } %v, 0
  %o = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %w
  store <4 x float> %x, ptr addrspace(1) %o, align 16
  ret void
}

define void @read_level_array(ptr addrspace(1) %t, ptr addrspace(1) %p, ptr addrspace(1) %out, i32 %i) {
  %w = zext i32 %i to i64
  %pp = getelementptr inbounds <4 x i32>, ptr addrspace(1) %p, i64 %w
  %r = load <4 x i32>, ptr addrspace(1) %pp, align 16
  %c = shufflevector <4 x i32> %r, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %a = extractelement <4 x i32> %r, i64 2
  %l = extractelement <4 x i32> %r, i64 3
  %s = call ptr addrspace(2) @air.get_read_sampler()
  %v = call { <4 x float>, i8 } @air.read_texture_2d_array.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %c, i32 %a, <2 x i32> zeroinitializer, i32 %l, i32 3)
  %x = extractvalue { <4 x float>, i8 } %v, 0
  %o = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %w
  store <4 x float> %x, ptr addrspace(1) %o, align 16
  ret void
}

define void @read_level_3d(ptr addrspace(1) %t, ptr addrspace(1) %p, ptr addrspace(1) %out, i32 %i) {
  %w = zext i32 %i to i64
  %pp = getelementptr inbounds <4 x i32>, ptr addrspace(1) %p, i64 %w
  %r = load <4 x i32>, ptr addrspace(1) %pp, align 16
  %c = shufflevector <4 x i32> %r, <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %l = extractelement <4 x i32> %r, i64 3
  %s = call ptr addrspace(2) @air.get_read_sampler()
  %v = call { <4 x float>, i8 } @air.read_texture_3d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <3 x i32> %c, <3 x i32> zeroinitializer, i32 %l, i32 3)
  %x = extractvalue { <4 x float>, i8 } %v, 0
  %o = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %w
  store <4 x float> %x, ptr addrspace(1) %o, align 16
  ret void
}

define void @read_level_cube(ptr addrspace(1) %t, ptr addrspace(1) %p, ptr addrspace(1) %out, i32 %i) {
  %w = zext i32 %i to i64
  %pp = getelementptr inbounds <4 x i32>, ptr addrspace(1) %p, i64 %w
  %r = load <4 x i32>, ptr addrspace(1) %pp, align 16
  %c = shufflevector <4 x i32> %r, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %f = extractelement <4 x i32> %r, i64 2
  %l = extractelement <4 x i32> %r, i64 3
  %s = call ptr addrspace(2) @air.get_read_sampler()
  %v = call { <4 x float>, i8 } @air.read_texture_cube.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %c, i32 %f, i32 %l, i32 3)
  %x = extractvalue { <4 x float>, i8 } %v, 0
  %o = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %w
  store <4 x float> %x, ptr addrspace(1) %o, align 16
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare { <4 x float>, i8 } @air.read_texture_2d_array.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, i32, <2 x i32>, i32, i32)
declare { <4 x float>, i8 } @air.read_texture_3d.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x i32>, <3 x i32>, i32, i32)
declare { <4 x float>, i8 } @air.read_texture_cube.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, i32, i32, i32)

!air.kernel = !{!0, !7, !11, !12, !13, !16, !19}
!0 = !{ptr @read_level, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"p"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!7 = !{ptr @read_level_uint, !1, !8}
!8 = !{!9, !4, !10, !6}
!9 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<uint, read_write>", !"air.arg_name", !"t"}
!10 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"out"}
!11 = !{ptr @read_level_rgb, !1, !2}
!12 = !{ptr @read_level_two, !1, !2}
!13 = !{ptr @read_level_array, !1, !14}
!14 = !{!15, !4, !5, !6}
!15 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d_array<float, read_write>", !"air.arg_name", !"t"}
!16 = !{ptr @read_level_3d, !1, !17}
!17 = !{!18, !4, !5, !6}
!18 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture3d<float, read_write>", !"air.arg_name", !"t"}
!19 = !{ptr @read_level_cube, !1, !20}
!20 = !{!21, !4, !5, !6}
!21 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texturecube<float, read_write>", !"air.arg_name", !"t"}
"#;

/// `stem` with the robustness features in `drop` removed.
fn without(stem: &str, drop: &[VkFeature]) -> DeviceProfile {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    for f in drop {
        desc.features.remove(f);
    }
    Device::from_queries(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// How a profile bounds an image read out of bounds: the device reads Metal's texel
/// (`robustImageAccess2`), the shader checks reads under `robustImageAccess` alone, or the shader
/// checks reads with no robustness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Class {
    Device,
    ShaderReads,
    Shader,
}

/// Every committed profile, `moltenvk-m3` without `robustImageAccess2`, `moltenvk-m3` and
/// `lavapipe` without any robustness (as `src/tests/engine_texture_oob.rs` runs them), with each
/// one's class.
fn classed_profiles() -> Vec<(String, DeviceProfile, Class)> {
    let mut out = committed_profiles().expect("profiles");
    out.push((
        "moltenvk-m3 robust1".into(),
        without("moltenvk-m3", &[VkFeature::RobustImageAccess2]),
    ));
    let all = [
        VkFeature::RobustImageAccess,
        VkFeature::RobustImageAccess2,
        VkFeature::RobustBufferAccess,
        VkFeature::RobustBufferAccess2,
    ];
    for stem in ["moltenvk-m3", "lavapipe"] {
        out.push((format!("{stem} unrobust"), without(stem, &all)));
    }
    out.into_iter()
        .map(|(stem, p)| {
            let f = p.fields();
            let class = match () {
                _ if f.images.robust_image_access2 => Class::Device,
                _ if f.images.robust_image_access => Class::ShaderReads,
                _ => Class::Shader,
            };
            (stem, p, class)
        })
        .collect()
}

/// A level read's texel class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lanes {
    Float,
    Uint,
}

impl Lanes {
    /// A missing alpha's default.
    fn one(self) -> u32 {
        match self {
            Lanes::Float => fl(1.0),
            Lanes::Uint => 1,
        }
    }

    /// The texel at `xyz` of `layer` of level `e`: distinct per level, layer, and texel, alpha
    /// nonzero.
    fn seed(self, e: u32, layer: u32, [x, y, z]: [u32; 3]) -> [u32; 4] {
        match self {
            Lanes::Float => [
                fl((10 * e + x) as f32),
                fl((100 * layer + y) as f32),
                fl(0.5 + z as f32),
                fl(100.0 + e as f32),
            ],
            Lanes::Uint => [10 * e + x, 100 * layer + y, 7 + z, 1000 + e],
        }
    }
}

/// A level read's texture: its type, its level-0 extent and layers, and the layers of the
/// neutral the binder binds past its last level (one per view layer: 6 for a cube's 2D-array
/// view).
#[derive(Clone, Copy, Debug)]
struct Tex {
    dim: ImageDim,
    extent: [u32; 3],
    layers: u32,
    neutral_layers: u32,
}

const TEX_2D: Tex = Tex {
    dim: ImageDim::D2,
    extent: [4, 4, 1],
    layers: 1,
    neutral_layers: 1,
};
const TEX_2D_ARRAY: Tex = Tex {
    layers: 2,
    ..TEX_2D
};
const TEX_3D: Tex = Tex {
    dim: ImageDim::D3,
    extent: [4, 4, 4],
    layers: 1,
    neutral_layers: 1,
};
const TEX_CUBE: Tex = Tex {
    dim: ImageDim::Cube,
    extent: [4, 4, 1],
    layers: 6,
    neutral_layers: 6,
};

/// The `n` level views of a `tex` texture of `levels` levels: element `i < levels` its level
/// `i` holding [`Lanes::seed`], every later element the 1x1(x1) neutral holding [`POISON`], each
/// reading `robust` outside itself (Metal's texel under `robustImageAccess2`, a poison where the
/// device's texel is not Metal's, a trap without robustness).
fn seeded_views(
    lanes: Lanes,
    tex: Tex,
    levels: u32,
    n: u32,
    robust: Option<[u32; 4]>,
) -> Vec<Image> {
    let whole = Image::new(texture(0), tex.extent, tex.layers, levels.max(1), 1);
    (0..n)
        .map(|e| {
            let mut img = if e < levels {
                let mut img =
                    Image::new(texture(0), whole.level_extent(e), tex.layers, 1, 1).at_element(e);
                let [w, h, d] = img.extent;
                for layer in 0..tex.layers {
                    for z in 0..d {
                        for y in 0..h {
                            for x in 0..w {
                                let i = img.index(0, layer, 0, [x, y, z]).expect("texel");
                                img.texels[i] = lanes.seed(e, layer, [x, y, z]);
                            }
                        }
                    }
                }
                img
            } else {
                let mut img =
                    Image::new(texture(0), [1, 1, 1], tex.neutral_layers, 1, 1).at_element(e);
                img.texels.fill(POISON);
                img
            };
            img.robust = robust;
            img
        })
        .collect()
}

/// A thread's `p[i]`, the texel its read takes (level, layer, `xyz`; `None`: Metal's read is out
/// of bounds), and whether every coordinate is inside each element the read can reach (only
/// those rows run without containment, where nothing contains a coordinate).
#[derive(Clone, Copy, Debug)]
struct Row {
    p: [u32; 4],
    at: Option<(u32, u32, [u32; 3])>,
    inside: bool,
}

fn row(p: [u32; 4], at: Option<(u32, u32, [u32; 3])>, inside: bool) -> Row {
    Row { p, at, inside }
}

/// A level-read kernel: its entry, texel class, whether it observes its alpha, its texture, and
/// its rows by bound level count, for views of `n` elements.
struct LevelRead {
    entry: &'static str,
    lanes: Lanes,
    observed: bool,
    tex: Tex,
    cases: fn(u32) -> Vec<(u32, Vec<Row>)>,
}

/// `texture2d` at the level `p.z`: three levels, then nothing bound.
fn cases_2d(n: u32) -> Vec<(u32, Vec<Row>)> {
    vec![
        (
            3,
            vec![
                row([1, 2, 0, 0], Some((0, 0, [1, 2, 0])), true),
                row([1, 1, 1, 0], Some((1, 0, [1, 1, 0])), true),
                row([0, 0, 2, 0], Some((2, 0, [0, 0, 0])), true),
                row([0, 0, 3, 0], None, true),
                row([1, 0, 5, 0], None, false),
                row([0, 0, n - 1, 0], None, true),
                row([0, 0, u32::MAX, 0], None, true),
                row([3, 3, 7, 0], None, false),
                row([4, 0, 0, 0], None, false),
                row([1, 1, 2, 0], None, false),
            ],
        ),
        (
            0,
            vec![row([0, 0, 0, 0], None, true), row([0, 0, 1, 0], None, true)],
        ),
    ]
}

/// `texture2d` at the constant level 2: three levels, then two (element 2 a neutral).
fn cases_level_two(_: u32) -> Vec<(u32, Vec<Row>)> {
    vec![
        (
            3,
            vec![
                row([0, 0, 0, 0], Some((2, 0, [0, 0, 0])), true),
                row([1, 0, 0, 0], None, false),
            ],
        ),
        (
            2,
            vec![
                row([0, 0, 0, 0], None, true),
                row([1, 1, 0, 0], None, false),
            ],
        ),
    ]
}

/// `texture2d_array` (two slices) at slice `p.z` of level `p.w`.
fn cases_array(n: u32) -> Vec<(u32, Vec<Row>)> {
    vec![
        (
            3,
            vec![
                row([1, 2, 1, 0], Some((0, 1, [1, 2, 0])), true),
                row([1, 1, 0, 1], Some((1, 0, [1, 1, 0])), true),
                row([0, 0, 1, 2], Some((2, 1, [0, 0, 0])), true),
                row([0, 0, 1, 3], None, true),
                row([0, 0, 0, n - 1], None, true),
                row([0, 0, 1, u32::MAX], None, true),
                row([0, 0, 2, 0], None, false),
                row([2, 1, 5, 4], None, false),
            ],
        ),
        (
            0,
            vec![
                row([0, 0, 0, 0], None, true),
                row([0, 0, 1, 1], None, false),
            ],
        ),
    ]
}

/// `texture3d` at `p.xyz` of level `p.w`.
fn cases_3d(n: u32) -> Vec<(u32, Vec<Row>)> {
    vec![
        (
            3,
            vec![
                row([1, 2, 3, 0], Some((0, 0, [1, 2, 3])), true),
                row([1, 1, 1, 1], Some((1, 0, [1, 1, 1])), true),
                row([0, 0, 0, 2], Some((2, 0, [0, 0, 0])), true),
                row([0, 0, 0, 3], None, true),
                row([0, 0, 0, n - 1], None, true),
                row([0, 0, 1, 2], None, false),
                row([3, 3, 3, 9], None, false),
            ],
        ),
        (
            0,
            vec![row([0, 0, 0, 0], None, true), row([0, 0, 0, 2], None, true)],
        ),
    ]
}

/// `texturecube` at face `p.z` of level `p.w`.
fn cases_cube(n: u32) -> Vec<(u32, Vec<Row>)> {
    vec![
        (
            3,
            vec![
                row([1, 2, 5, 0], Some((0, 5, [1, 2, 0])), true),
                row([1, 1, 3, 1], Some((1, 3, [1, 1, 0])), true),
                row([0, 0, 0, 2], Some((2, 0, [0, 0, 0])), true),
                row([0, 0, 4, 3], None, true),
                row([0, 0, 2, n - 1], None, true),
                row([0, 0, 2, u32::MAX], None, true),
                row([1, 0, 1, 2], None, false),
            ],
        ),
        (
            0,
            vec![row([0, 0, 3, 0], None, true), row([0, 0, 0, 1], None, true)],
        ),
    ]
}

const LEVEL_READ_KERNELS: [LevelRead; 7] = [
    LevelRead {
        entry: "read_level",
        lanes: Lanes::Float,
        observed: true,
        tex: TEX_2D,
        cases: cases_2d,
    },
    LevelRead {
        entry: "read_level_uint",
        lanes: Lanes::Uint,
        observed: true,
        tex: TEX_2D,
        cases: cases_2d,
    },
    LevelRead {
        entry: "read_level_rgb",
        lanes: Lanes::Float,
        observed: false,
        tex: TEX_2D,
        cases: cases_2d,
    },
    LevelRead {
        entry: "read_level_two",
        lanes: Lanes::Float,
        observed: true,
        tex: TEX_2D,
        cases: cases_level_two,
    },
    LevelRead {
        entry: "read_level_array",
        lanes: Lanes::Float,
        observed: true,
        tex: TEX_2D_ARRAY,
        cases: cases_array,
    },
    LevelRead {
        entry: "read_level_3d",
        lanes: Lanes::Float,
        observed: true,
        tex: TEX_3D,
        cases: cases_3d,
    },
    LevelRead {
        entry: "read_level_cube",
        lanes: Lanes::Float,
        observed: true,
        tex: TEX_CUBE,
        cases: cases_cube,
    },
];

/// A read at a level the bound texture lacks is out of bounds (MSL 4.1 §6.13): Metal reads zero
/// components with the format's missing alpha read as 1, which the level views select on every
/// profile class, whatever the device's robustness and without containment too, since the
/// element they read is the last level or a neutral. The alpha is the `OutOfBoundsAlpha`
/// specialization, run with both values, and the fact is recorded on every class where the alpha is
/// observed and on none where it is not. 2D, 2D-array, 3D, and cube textures are read at a dynamic
/// level (at the count, past it, at the views' last element, at `u32::MAX`, and with nothing
/// bound), and a 2D texture at the constant level 2 (a 2-level texture's neutral). Coordinates past
/// an existing level stay as before; without containment only rows whose coordinates are inside
/// every element they can reach run.
#[test]
fn a_read_at_a_missing_level_is_metals_out_of_bounds_texel() {
    let Some(bc) = assemble("level_reads", LEVEL_READS) else {
        return;
    };
    let fact = RuntimeFact::OutOfBoundsAlpha { slot: tslot(0) };
    let mut ran: BTreeMap<&str, Vec<Class>> = BTreeMap::new();
    for (stem, p, class) in classed_profiles() {
        for k in &LEVEL_READ_KERNELS {
            let entry = k.entry;
            let (plan, m) = match typed_module(&bc, entry, &p) {
                Ok(m) => m,
                Err(Error::NeedsInput(_)) => continue,
                Err(e) if is(&e, &no_non_uniform()) && !non_uniform(&p) => continue,
                // The views may exceed a small per-stage storage image limit.
                Err(Error::Unsupported(u))
                    if u.feature != Feature::Lowering(LowerFamily::Texture) =>
                {
                    continue
                }
                Err(e) => panic!("{stem}: {entry}: {e}"),
            };
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{stem}: {entry}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {entry}: L3: {e}"));
            assert_eq!(plan.facts.contains(&fact), k.observed, "{stem}: {entry}");
            let n = max_levels(&p, k.tex.dim);
            for (levels, rows) in (k.cases)(n) {
                let rows: Vec<Row> = rows.into_iter().collect();
                for alpha in [0, k.lanes.one()] {
                    let robust = match class {
                        Class::Device => Some([0, 0, 0, alpha]),
                        Class::ShaderReads => Some([0xDEAD_BEEF; 4]),
                        Class::Shader => None,
                    };
                    let imgs = seeded_views(k.lanes, k.tex, levels, n, robust);
                    let p_rows: Vec<u8> = rows
                        .iter()
                        .flat_map(|r| r.p)
                        .flat_map(u32::to_le_bytes)
                        .collect();
                    let bufs = [p_rows, vec![0; 16 * rows.len()]];
                    let counts = [(0, levels)];
                    let spec = [(SpecKey::OutOfBoundsAlpha(tslot(0)), u64::from(alpha != 0))];
                    let r = Run {
                        spec: &spec,
                        ..Run::new(&bufs, &counts, [rows.len() as u32, 1, 1])
                    };
                    let (out, after) =
                        run(&plan, &m, &r, &imgs).unwrap_or_else(|e| panic!("{stem}: {e}"));
                    assert_eq!(after, imgs, "{stem}: {entry} writes nothing");
                    for (i, r) in rows.iter().enumerate() {
                        let mut want = match r.at {
                            Some((e, layer, xyz)) => k.lanes.seed(e, layer, xyz),
                            None => [0, 0, 0, alpha],
                        };
                        if !k.observed {
                            want[3] = fl(2.0);
                        }
                        assert_eq!(
                            words(&out[1][16 * i..16 * i + 16]),
                            want,
                            "{stem} ({class:?}): {entry} {levels} levels, p {:?}, alpha {alpha:#x}",
                            r.p
                        );
                    }
                }
            }
            ran.entry(entry).or_default().push(class);
        }
    }
    for k in &LEVEL_READ_KERNELS {
        let mut classes = ran.remove(k.entry).unwrap_or_default();
        classes.sort();
        classes.dedup();
        assert_eq!(
            classes,
            [Class::Device, Class::ShaderReads, Class::Shader],
            "{} ran on every class",
            k.entry
        );
    }
}

// ---- Declarations, facts, and refusals ----

/// A level-view slot (`txt_level_write`, written at a level from a buffer) declares one storage
/// descriptor per level the device's largest texture of its type has, with both facts and the
/// level-count push field; profiles that cannot index storage arrays by a non-uniform value refuse
/// its dynamic level, typed. A slot whose level count alone is read, or whose size is queried at a
/// level (`txt_level_queries`), keeps one descriptor and has the count, on every profile: the
/// query answers from the first level.
#[test]
fn level_views_declare_every_level_of_the_largest_texture() {
    let (Some(write), Some(queries)) = (fixture("txt_level_write"), fixture("txt_level_queries"))
    else {
        return;
    };
    let (mut ok, mut refused) = (0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let has_push = |a: &FunctionArtifact, s: u32| {
            a.interface
                .push
                .fields
                .iter()
                .any(|f| f.field == PushField::TextureLevels(tslot(s)))
        };
        let count = |a: &FunctionArtifact, index: u32| {
            a.interface
                .descriptors
                .iter()
                .find(|d| d.slot == texture(index))
                .map(|d| d.count)
        };
        match compile(&queries, "txt_level_queries", &p) {
            Ok(a) => {
                for s in [0, 1] {
                    assert_eq!(count(&a, s), Some(1), "{stem}: slot {s}");
                    assert!(
                        a.facts
                            .contains(&RuntimeFact::StorageTextureLevels { slot: tslot(s) }),
                        "{stem}: slot {s}"
                    );
                    assert!(
                        !a.facts
                            .contains(&RuntimeFact::StorageTextureLevelViews { slot: tslot(s) }),
                        "{stem}: slot {s}"
                    );
                    assert!(has_push(&a, s), "{stem}: slot {s}");
                }
            }
            Err(Error::NeedsInput(_)) => {}
            Err(e) => panic!("{stem}: the queries: {e}"),
        }
        let a = match compile(&write, "txt_level_write", &p) {
            Ok(a) => a,
            Err(e) if !non_uniform(&p) => {
                assert!(is(&e, &no_non_uniform()), "{stem}: {e}");
                refused += 1;
                continue;
            }
            Err(Error::NeedsInput(_)) => continue,
            Err(e) => panic!("{stem}: {e}"),
        };
        assert!(
            non_uniform(&p),
            "{stem}: compiled without non-uniform indexing"
        );
        assert_eq!(count(&a, 0), Some(max_levels(&p, ImageDim::D2)), "{stem}");
        let has = |f: RuntimeFact| a.facts.contains(&f);
        assert!(
            has(RuntimeFact::StorageTextureLevels { slot: tslot(0) }),
            "{stem}"
        );
        assert!(
            has(RuntimeFact::StorageTextureLevelViews { slot: tslot(0) }),
            "{stem}"
        );
        assert!(has_push(&a, 0), "{stem}");
        ok += 1;
    }
    assert!(ok > 0 && refused > 0, "ok {ok}, refused {refused}");
}

/// `texture2d<float, write>` slot 0 written with `(1, 2, 3, 4)` at `gid` of the constant level 1.
const CONSTANT_LEVEL: &str = r#"define void @k(ptr addrspace(1) %t, <2 x i32> %gid) {
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %gid, <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, i32 1, i32 2)
  ret void
}

declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;

/// A constant level indexes its view directly (no non-uniform indexing needed): level 1 of a
/// 2-level texture is written, of a 1-level texture nothing is.
#[test]
fn a_constant_level_indexes_its_view_directly() {
    let Some(bc) = assemble("constant_level", CONSTANT_LEVEL) else {
        return;
    };
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let (plan, m) = match typed_module(&bc, "k", &p) {
            Ok(m) => m,
            Err(Error::NeedsInput(_)) => continue,
            // The views may exceed a small per-stage storage image limit.
            Err(Error::Unsupported(u)) if u.feature != Feature::Lowering(LowerFamily::Texture) => {
                eprintln!("{stem}: {u}");
                continue;
            }
            Err(e) => panic!("{stem}: {e}"),
        };
        let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert!(uses(&a, "texture.write") > 0, "{stem}");
        let n = max_levels(&p, ImageDim::D2);
        for levels in [2u32, 1] {
            let imgs = views(0, [4, 4, 1], 1, levels, n);
            let counts = [(0, levels)];
            let r = Run::new(&[], &counts, [2, 2, 1]);
            let (_, out) = run(&plan, &m, &r, &imgs).unwrap_or_else(|e| panic!("{stem}: {e}"));
            for img in &out {
                texels(
                    img,
                    |_, _, _| {
                        if img.element == 1 && levels == 2 {
                            [fl(1.0), fl(2.0), fl(3.0), fl(4.0)]
                        } else {
                            POISON
                        }
                    },
                    &format!("{stem} {levels} levels"),
                );
            }
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// `texture2d<uint, read_write>` slot 0: `atomic_fetch_max` at `gid`, then the old texel written
/// at `gid` of the level `buffer(0)` holds.
const ATOMIC_AND_LEVEL: &str = r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %lod, <2 x i32> %gid) {
  %old = call <4 x i32> @air.atomic_fetch_max_explicit_texture_2d.u.v4i32(ptr addrspace(1) %t, <2 x i32> %gid, <2 x i32> zeroinitializer, <4 x i32> <i32 7, i32 7, i32 7, i32 7>, i32 0, i32 3)
  %l = load i32, ptr addrspace(1) %lod, align 4
  call void @air.write_texture_2d.u.v4i32(ptr addrspace(1) %t, <2 x i32> %gid, <4 x i32> %old, i32 %l, i32 3)
  ret void
}

declare <4 x i32> @air.atomic_fetch_max_explicit_texture_2d.u.v4i32(ptr addrspace(1), <2 x i32>, <2 x i32>, <4 x i32>, i32, i32)
declare void @air.write_texture_2d.u.v4i32(ptr addrspace(1), <2 x i32>, <4 x i32>, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<uint, read_write>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"lod"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;

/// A texture atomic's texel pointer into one level view is not modelled: a slot updated
/// atomically and accessed at a level stays the typed refusal.
#[test]
fn an_atomic_slot_with_level_views_is_refused() {
    let Some(bc) = assemble("atomic_and_level", ATOMIC_AND_LEVEL) else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let want = Unsupported::not_implemented(Feature::Lowering(LowerFamily::Texture));
    match compile(&bc, "k", &p) {
        Err(e) if is(&e, &want) => {}
        other => panic!("{:?}", other.map(|_| ())),
    }
}

/// `texture2d<float, write>` slot 0 written with `(1, 2, 3, 4)` at `gid` of the constant level
/// 40, past the levels of any texture a device can create.
const PAST_EVERY_LEVEL: &str = r#"define void @k(ptr addrspace(1) %t, <2 x i32> %gid) {
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %gid, <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, i32 40, i32 2)
  ret void
}

declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;

/// A constant level at or past the level views' count indexes them as a dynamic level does
/// (clamped, `NonUniform`), so it needs `shaderStorageImageArrayNonUniformIndexing` like one:
/// typed where the profile lacks it, and on every other profile a write that is dropped.
#[test]
fn a_constant_level_past_every_texture_is_indexed_as_a_dynamic_one() {
    let Some(bc) = assemble("past_every_level", PAST_EVERY_LEVEL) else {
        return;
    };
    let (mut ran, mut refused) = (0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some((plan, m)) = module(&bc, "k", &stem, &p) else {
            if !non_uniform(&p) {
                assert!(
                    is(&compile(&bc, "k", &p).expect_err(&stem), &no_non_uniform()),
                    "{stem}"
                );
                refused += 1;
            }
            continue;
        };
        let n = max_levels(&p, ImageDim::D2);
        let imgs = views(0, [4, 4, 1], 1, 3, n);
        let r = Run::new(&[], &[(0, 3)], [1, 1, 1]);
        let (_, out) = run(&plan, &m, &r, &imgs).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(out, imgs, "{stem}: nothing is written");
        ran += 1;
    }
    assert!(ran > 0 && refused > 0, "ran {ran}, refused {refused}");
}

/// `get_num_mip_levels` of a sampled `texture1d<float>` (slot 0), a sampled
/// `texture1d_array<float>` (slot 1), and a `texture1d<float, write>` (slot 2), stored to
/// `out[0..3]` (buffer 0).
const ONE_D_LEVELS: &str = r#"define void @k(ptr addrspace(1) %s, ptr addrspace(1) %a, ptr addrspace(1) %w, ptr addrspace(1) %out) {
  %l0 = call i32 @air.get_num_mip_levels_texture_1d(ptr addrspace(1) %s)
  %l1 = call i32 @air.get_num_mip_levels_texture_1d_array(ptr addrspace(1) %a)
  %l2 = call i32 @air.get_num_mip_levels_texture_1d(ptr addrspace(1) %w)
  store i32 %l0, ptr addrspace(1) %out, align 4
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %l1, ptr addrspace(1) %o1, align 4
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 2
  store i32 %l2, ptr addrspace(1) %o2, align 4
  ret void
}

declare i32 @air.get_num_mip_levels_texture_1d(ptr addrspace(1))
declare i32 @air.get_num_mip_levels_texture_1d_array(ptr addrspace(1))

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture1d<float>", !"air.arg_name", !"s"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture1d_array<float>", !"air.arg_name", !"a"}
!5 = !{i32 2, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.write", !"air.arg_type_name", !"texture1d<float, write>", !"air.arg_name", !"w"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;

/// Metal has no mipmapped 1D textures: `get_num_mip_levels` of a 1D texture or 1D texture array
/// always returns 0 (MSL 4.1), sampled or storage, whatever the bound image's level count, and
/// the storage slot asks the binder for no level count.
#[test]
fn one_d_textures_have_no_levels() {
    let Some(bc) = assemble("one_d_levels", ONE_D_LEVELS) else {
        return;
    };
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some((plan, m)) = module(&bc, "k", &stem, &p) else {
            continue;
        };
        assert!(
            !plan
                .facts
                .iter()
                .any(|f| matches!(f, RuntimeFact::StorageTextureLevels { .. })),
            "{stem}"
        );
        let imgs = [
            Image::new(texture(0), [4, 1, 1], 1, 3, 1),
            Image::new(texture(1), [4, 1, 1], 2, 3, 1),
            Image::new(texture(2), [4, 1, 1], 1, 1, 1),
        ];
        let bufs = [vec![0xFF; 16]];
        let r = Run::new(&bufs, &[(2, 1)], [1, 1, 1]);
        let (out, _) = run(&plan, &m, &r, &imgs).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(words(&out[0])[..3], [0, 0, 0], "{stem}");
        ran += 1;
    }
    assert!(ran > 0);
}

// ---- The binder, end to end ----

fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::from_queries(desc, DevicePolicy::default()).expect("device")
}

fn pipeline(dev: &Device, bc: &[u8], entry: &str) -> ComputePipeline {
    let lib = Library::from_air(bc).expect("library");
    let f = FunctionDescriptor::new(lib.function(entry).expect("function"));
    let d = ComputePipelineDescriptor::new(StageSource::Function(f));
    dev.compile_compute_pipeline(&d, &Budget::standard())
        .unwrap_or_else(|e| panic!("{entry}: {e}"))
}

/// A bind state of `txt_level_write`: two bytes of level at buffer 0 and, when `levels` is
/// `Some`, a 2D `RGBA16Float` texture of that many levels at texture 0.
fn state(levels: Option<u32>) -> BindState {
    let mut t = StageTable::default();
    t.buffers.insert(
        0,
        BoundBuffer::Bytes {
            len: 2,
            attribute_stride: None,
        },
    );
    if let Some(n) = levels {
        let mut facts = TextureFacts::new(TextureType::Type2D, PixelFormat::RGBA16Float);
        facts.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        facts.mipmap_level_count = n;
        t.textures.insert(
            0,
            BoundTexture::Texture {
                facts,
                resource_id: None,
            },
        );
    }
    let mut s = BindState::default();
    s.stages.insert(Stage::Kernel, t);
    s
}

/// The binder plans `txt_level_write`'s level views (one write per element: element `i` a view
/// of level `i` alone, the neutral past the last level or for an unbound slot) and pushes the
/// bound level count; running the module over exactly those writes and push bytes writes the
/// level the buffer names and nothing past the count. Plans of textures with different level
/// counts differ (the shape carries the count).
#[test]
fn the_binder_binds_each_level_and_pushes_the_count() {
    let Some(bc) = fixture("txt_level_write") else {
        return;
    };
    for stem in ["moltenvk-m3", "lavapipe"] {
        let dev = device(stem);
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        let pipe = pipeline(&dev, &bc, "txt_level_write");
        let (_, m) = typed_module(&bc, "txt_level_write", &p).expect("module");
        let n = max_levels(&p, ImageDim::D2);
        let binding = pipe
            .layout
            .sets
            .iter()
            .flat_map(|s| s.bindings.iter().map(move |b| (s.set, b)))
            .find(|(_, b)| b.source == BindingSource::Texture(tslot(0)))
            .map(|(set, b)| (set, b.binding))
            .expect("the texture binding");
        assert_ne!(state(Some(3)).shape(), state(Some(2)).shape(), "{stem}");
        for (levels, lod) in [(Some(3), 2u16), (Some(3), 3), (Some(1), 0), (None, 0)] {
            let what = format!("{stem}: {levels:?} levels, lod {lod}");
            let s = state(levels);
            let plan = pipe.plan(&dev, &s.shape()).expect("plan");
            let writes: Vec<_> = plan
                .descriptors
                .iter()
                .filter(|w| (w.set, w.binding) == binding)
                .collect();
            assert_eq!(writes.len(), n as usize, "{what}");
            let bound = levels.unwrap_or(0);
            let mut imgs = Vec::new();
            for (i, w) in writes.iter().enumerate() {
                let i = i as u32;
                assert_eq!(w.array_element, i, "{what}");
                let extent = match &w.source {
                    WriteSource::Texture { view, .. } => {
                        assert!(i < bound, "{what}: element {i}");
                        assert_eq!(view.level, Some(i), "{what}");
                        [4 >> i, 4 >> i, 1]
                    }
                    WriteSource::Neutral(_) => {
                        assert!(i >= bound, "{what}: element {i}");
                        [1, 1, 1]
                    }
                    other => panic!("{what}: {other:?}"),
                };
                let mut img = Image::new(texture(0), extent, 1, 1, 1).at_element(i);
                img.texels.fill(POISON);
                imgs.push(img);
            }
            let (push, spec) = bound_inputs(&pipe, &plan, &s, [1, 1, 1]);
            let res = Resources {
                buffers: vec![Buffer {
                    slot: Some(ResourceSlot {
                        stage: Stage::Kernel,
                        kind: ResourceKind::Buffer,
                        index: 0,
                    }),
                    address: None,
                    bytes: lod_bytes(u32::from(lod)),
                }],
                push,
                spec,
            };
            let cfg = Config {
                workgroups: [1, 1, 1],
                ..Config::default()
            };
            let (_, out) = run_with_images(&m, &cfg, res, imgs).expect("run");
            for img in &out {
                let want = if u32::from(lod) < bound && img.element == u32::from(lod) {
                    [fl(0.0), fl(0.0), fl(1.0), fl(0.5)]
                } else {
                    POISON
                };
                assert_eq!(img.texels[0], want, "{what}: element {}", img.element);
            }
        }
    }
}

/// The push bytes and specialization values `plan` instantiates for `s` (a dispatch of `groups`
/// one-thread threadgroups), each value by its key.
fn bound_inputs(
    pipe: &ComputePipeline,
    plan: &BindingPlan,
    s: &BindState,
    groups: [u32; 3],
) -> (Vec<u8>, BTreeMap<SpecKey, u64>) {
    let cmds = plan
        .instantiate_dispatch(
            s,
            &Dispatch::Threadgroups {
                groups,
                threads_per_group: [1, 1, 1],
            },
        )
        .expect("instantiate");
    let region = &cmds.regions[0];
    let mut push = Vec::new();
    if let Some(w) = region.push_constants.first() {
        push = vec![0u8; w.offset as usize + w.bytes.len()];
        push[w.offset as usize..].copy_from_slice(&w.bytes);
    }
    let ids: BTreeMap<u32, SpecKey> = pipe
        .stage
        .specialization
        .constants
        .iter()
        .map(|c| (c.id, c.key))
        .collect();
    let mut spec = BTreeMap::new();
    for e in &region.specialization.entries {
        let at = e.offset as usize;
        let mut v = [0u8; 8];
        v[..e.size as usize].copy_from_slice(&region.specialization.data[at..at + e.size as usize]);
        spec.insert(ids[&e.constant_id], u64::from_le_bytes(v));
    }
    // The values the binder pushes for the keys the function reads as data.
    for placement in &pipe.interface.push {
        spec.extend(placement.pushed_values(&push));
    }
    (push, spec)
}

/// The binder specializes a level-view read's out-of-bounds alpha from the bound texture's
/// format, as it does a checked read's, on profiles whose robustness leaves the coordinates to the
/// device: 1 for `R32Float`, which has no alpha, 0 for `RGBA32Float` and an unbound slot. Running
/// `read_level` over the binder's level views, push bytes, and specialization reads Metal's
/// out-of-bounds texel at a missing level, `(0, 0, 0, 1)` exactly for the format without alpha, and
/// the bound level below the count.
#[test]
fn the_binder_specializes_a_level_reads_alpha_from_the_bound_format() {
    let Some(bc) = assemble("level_reads", LEVEL_READS) else {
        return;
    };
    let key = SpecKey::OutOfBoundsAlpha(tslot(0));
    // p[0] reads (1, 1) of level 1, p[1] (0, 0) of the missing level 5.
    let rows: Vec<u8> = [[1u32, 1, 1, 0], [0, 0, 5, 0]]
        .iter()
        .flatten()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let texel = [fl(1.0), fl(2.0), fl(3.0), fl(4.0)];
    for stem in ["moltenvk-m3", "lavapipe"] {
        let dev = device(stem);
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        assert!(p.fields().images.robust_image_access2, "{stem}");
        let pipe = pipeline(&dev, &bc, "read_level");
        let (_, m) = typed_module(&bc, "read_level", &p).expect("module");
        assert!(pipe
            .stage
            .specialization
            .constants
            .iter()
            .any(|c| c.key == key));
        let n = max_levels(&p, ImageDim::D2);
        for (format, alpha) in [
            (Some(PixelFormat::R32Float), fl(1.0)),
            (Some(PixelFormat::RGBA32Float), 0),
            (None, 0),
        ] {
            let what = format!("{stem}: {format:?}");
            let mut t = StageTable::default();
            t.buffers.insert(
                0,
                BoundBuffer::Bytes {
                    len: 32,
                    attribute_stride: None,
                },
            );
            t.buffers.insert(
                1,
                BoundBuffer::Bytes {
                    len: 32,
                    attribute_stride: None,
                },
            );
            if let Some(f) = format {
                let mut facts = TextureFacts::new(TextureType::Type2D, f);
                facts.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
                facts.mipmap_level_count = 3;
                t.textures.insert(
                    0,
                    BoundTexture::Texture {
                        facts,
                        resource_id: None,
                    },
                );
            }
            let mut s = BindState::default();
            s.stages.insert(Stage::Kernel, t);
            let plan = pipe.plan(&dev, &s.shape()).expect("plan");
            let (push, spec) = bound_inputs(&pipe, &plan, &s, [2, 1, 1]);
            assert_eq!(spec.get(&key), Some(&u64::from(alpha != 0)), "{what}");
            let mut imgs = Vec::new();
            for w in plan
                .descriptors
                .iter()
                .filter(|w| matches!(w.ty, crate::vk::DescriptorType::STORAGE_IMAGE))
            {
                let i = w.array_element;
                let extent = match &w.source {
                    WriteSource::Texture { view, .. } => {
                        assert_eq!(view.level, Some(i), "{what}");
                        [4 >> i, 4 >> i, 1]
                    }
                    WriteSource::Neutral(_) => [1, 1, 1],
                    other => panic!("{what}: {other:?}"),
                };
                let mut img = Image::new(texture(0), extent, 1, 1, 1).at_element(i);
                img.texels.fill(if i == 1 && format.is_some() {
                    texel
                } else {
                    POISON
                });
                img.robust = Some([0, 0, 0, alpha]);
                imgs.push(img);
            }
            assert_eq!(imgs.len(), n as usize, "{what}");
            let res = Resources {
                buffers: [rows.clone(), vec![0; 32]]
                    .into_iter()
                    .enumerate()
                    .map(|(i, bytes)| Buffer {
                        slot: Some(ResourceSlot {
                            stage: Stage::Kernel,
                            kind: ResourceKind::Buffer,
                            index: i as u32,
                        }),
                        address: None,
                        bytes,
                    })
                    .collect(),
                push,
                spec,
            };
            let cfg = Config {
                workgroups: [2, 1, 1],
                ..Config::default()
            };
            let (out, _) = run_with_images(&m, &cfg, res, imgs).expect("run");
            let first = if format.is_some() {
                texel
            } else {
                [0, 0, 0, 0]
            };
            assert_eq!(words(&out.buffers[1][..16]), first, "{what}: level 1");
            assert_eq!(
                words(&out.buffers[1][16..]),
                [0, 0, 0, alpha],
                "{what}: level 5"
            );
        }
    }
}

/// The pushed level count of `plan`'s texture slot 0, as instantiated for `s`.
fn pushed_levels(plan: &BindingPlan, s: &BindState) -> u32 {
    let at = plan
        .push
        .fields
        .iter()
        .find(|f| f.field == PushField::TextureLevels(tslot(0)))
        .map(|f| match f.location {
            FieldLocation::PushWindow { offset } => offset as usize,
            other => panic!("{other:?}"),
        })
        .expect("a level-count field");
    let cmds = plan
        .instantiate_dispatch(
            s,
            &Dispatch::Threadgroups {
                groups: [1, 1, 1],
                threads_per_group: [1, 1, 1],
            },
        )
        .expect("instantiate");
    let w = cmds.regions[0].push_constants.first().expect("push bytes");
    let i = at - w.offset as usize;
    u32::from_le_bytes(w.bytes[i..i + 4].try_into().expect("u32"))
}

/// A level slot binds a texture of any usage at its levels: plain Metal writes a texture made
/// without `ShaderWrite` at every level, through every write form, as one made with it
/// (CoreAnimation's blur textures are `ShaderRead | RenderTarget | BlockWritesOnly`),
/// where the binder bound the neutral there and dropped the writes. Its shape states its levels
/// whatever the usage. A texture with more levels than the views hold is larger than the device's
/// largest 2D image: refused, naming `maxImageDimension2D`.
#[test]
fn level_slots_bind_textures_of_every_usage_the_device_can_hold() {
    let Some(bc) = fixture("txt_level_write") else {
        return;
    };
    let stem = "moltenvk-m3";
    let dev = device(stem);
    let p = profile(stem, DevicePolicy::default()).expect("profile");
    let pipe = pipeline(&dev, &bc, "txt_level_write");
    let n = max_levels(&p, ImageDim::D2);
    let with_usage = |levels: Option<u32>, usage: TextureUsage| {
        let mut s = state(levels);
        if let Some(BoundTexture::Texture { facts, .. }) = s
            .stages
            .get_mut(&Stage::Kernel)
            .and_then(|t| t.textures.get_mut(&0))
        {
            facts.usage = usage;
        }
        s
    };
    let private =
        TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET | TextureUsage::BLOCK_WRITES_ONLY;
    for usage in [
        TextureUsage::SHADER_READ,
        TextureUsage::RENDER_TARGET,
        TextureUsage::BLOCK_WRITES_ONLY,
        private,
        TextureUsage::empty(),
    ] {
        let what = format!("{usage:?}");
        let (three, one) = (with_usage(Some(3), usage), with_usage(Some(1), usage));
        assert_ne!(three.shape(), one.shape(), "{what}");

        let plan = pipe.plan(&dev, &three.shape()).expect("plan");
        let writes: Vec<_> = plan
            .descriptors
            .iter()
            .filter(|w| matches!(w.ty, crate::vk::DescriptorType::STORAGE_IMAGE))
            .collect();
        assert_eq!(writes.len(), n as usize, "{what}");
        for (i, w) in writes.iter().enumerate() {
            match &w.source {
                WriteSource::Texture { view, .. } => {
                    assert!(i < 3, "{what}: element {i}");
                    assert_eq!(view.level, Some(i as u32), "{what}");
                }
                WriteSource::Neutral(_) => assert!(i >= 3, "{what}: element {i}"),
                other => panic!("{what}: {other:?}"),
            }
        }
        assert!(
            !plan.diagnostics.iter().any(
                |d| matches!(d, PlanDiagnostic::IncompatibleBinding { slot, .. } if *slot == tslot(0))
            ),
            "{what}: {:?}",
            plan.diagnostics
        );
        assert_eq!(pushed_levels(&plan, &three), 3, "{what}");
    }

    let s = state(Some(3));
    let plan = pipe.plan(&dev, &s.shape()).expect("plan");
    assert_eq!(pushed_levels(&plan, &s), 3);

    let want = Unsupported::needs(
        Feature::Lowering(LowerFamily::Texture),
        Capability::Limit {
            limit: VkLimit::MaxImageDimension2D,
            required: 1 << n,
            available: u64::from(p.fields().limits.max_image_dimension_2d),
        },
    );
    match pipe.plan(&dev, &state(Some(n + 1)).shape()) {
        Err(BindError::Unsupported(u)) => {
            assert_eq!((u.feature, u.needs), (want.feature, want.needs))
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
}
