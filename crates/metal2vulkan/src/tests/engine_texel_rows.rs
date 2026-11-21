//! A 2D texture's rows read and written in place through a texel buffer (`bind::texel_rows`), on
//! the CPU interpreter.
//!
//! - A slot whose function only reads, writes or asks the size of its texture states
//!   `RuntimeFact::TexelRows`; one that samples it states none.
//! - Under the rows variant (`LoweringInputs::texel_rows`) the slot is a texel buffer (uniform
//!   for an `access::read` slot, storage for an `access::write` or `read_write` one) that the
//!   profile validates, and its push field holds the rows' words.
//! - A read of texel `(x, y)` reads the view's texel `first + y · row + x`, and Metal's
//!   out-of-bounds texel past the texture's width or height; a write stores there, and nothing
//!   past the texture, so the texels between the rows and before the first keep their values;
//!   a size query answers the field's size. The interpreter's texel buffer traps every access
//!   outside the view, so each access lies within it.
//!
//! The kernels are the runtime's device tests' (`examples/texel_rows.metal`). `llvm-as` assembles
//! them; without it the tests skip visibly.

use std::collections::BTreeMap;

use crate::bind::TexelRows;
use crate::device::profile::DeviceProfile;
use crate::device::DevicePolicy;
use crate::lower::select::{LowerInputs, Plan};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef};
use crate::pipeline::sym::{SpecKey, Stage};
use crate::pipeline::testing::{
    compile, compile_with, fixtures_dir, link_and_validate, llvm_assemble, profile,
    typed_module_with, AssembleError,
};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::{Dim, GlobalKind, ImageUsage, Type as SType, TypedModule};

const TEXTURE: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Texture,
    index: 0,
};

/// The texture's width and height, its rows' texels apart in the view, and the view's texel
/// that is its first: rows of 5 texels 7 apart, after 3 texels of the view.
const SIZE: [u32; 2] = [5, 4];
const ROW: u64 = 7;
const FIRST: u64 = 3;

/// The grid: past the texture's width and height, so its edges are crossed.
const GRID: [u32; 2] = [8, 6];

/// The kernels' bitcode, or `None` without `llvm-as`.
fn bitcode() -> Option<Vec<u8>> {
    let ll = fixtures_dir().join("examples/texel_rows.ll");
    match llvm_assemble(&ll) {
        Ok(x) => Some(x),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("texel_rows.ll: {why}"),
    }
}

fn lavapipe() -> DeviceProfile {
    profile("lavapipe", DevicePolicy::RESIDENT_TABLES).expect("profile")
}

/// The rows variant's inputs: slot 0's rows.
fn rows_inputs() -> LowerInputs {
    let mut inputs = LowerInputs::new();
    inputs.texel_rows.insert(TEXTURE);
    inputs
}

fn rows() -> TexelRows {
    TexelRows::new(SIZE, ROW, FIRST).expect("rows")
}

/// The view's texel `i` before any write: a value of its own in each lane, as `float` bits.
fn view_texel(i: u32) -> [u32; 4] {
    std::array::from_fn(|lane| (i as f32 * 4.0 + lane as f32 + 0.25).to_bits())
}

/// Each texel of the texture by its view index, `(x, y)` row-major.
fn texture_indices() -> Vec<((u32, u32), u32)> {
    let [w, h] = SIZE;
    (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x, y), (FIRST + u64::from(y) * ROW + u64::from(x)))))
        .map(|(at, i)| (at, i as u32))
        .collect()
}

/// The texture's binding, a texel buffer of `usage`.
fn rows_binding(m: &TypedModule, usage: ImageUsage) -> ResourceSlot {
    let (slot, pointee) = m
        .globals()
        .iter()
        .find_map(|g| match g.kind {
            GlobalKind::Resource(
                s @ ResourceSlot {
                    kind: ResourceKind::Texture,
                    index: 0,
                    ..
                },
            ) => Some((s, g.pointee)),
            _ => None,
        })
        .expect("the texture's binding");
    match m.types().get(pointee).expect("its type") {
        SType::Image(i) => {
            assert_eq!(i.dim, Dim::Buffer, "the rows' binding is a texel buffer");
            assert_eq!(i.usage, usage, "the rows' texel buffer's kind");
        }
        other => panic!("the rows' binding is {other:?}"),
    }
    slot
}

/// The view, `elements` texels of [`view_texel`], bound at `slot`.
fn view(slot: ResourceSlot) -> Image {
    let n = rows().elements() as u32;
    let mut image = Image::new(slot, [n, 1, 1], 1, 1, 1);
    for x in 0..n {
        let i = image.index(0, 0, 0, [x, 0, 0]).expect("element");
        image.texels[i] = view_texel(x);
    }
    image
}

/// Resources of a dispatch of one-thread threadgroups: `[[buffer(i)]]` holding `init[i]`, the
/// slot's push field holding the rows' words, and Metal's out-of-bounds alpha of a format with
/// alpha (0).
fn resources(plan: &Plan, init: &[Vec<u8>]) -> Resources {
    let mut spec = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(axis), 1);
    }
    spec.insert(SpecKey::FlatLocalSize, 1);
    spec.insert(SpecKey::OutOfBoundsAlpha(TEXTURE), 0);
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    let mut written = false;
    for f in &plan.push.fields {
        if let (PushField::TexelRows(TEXTURE), PushLoc::Push { offset }) = (f.field, f.loc) {
            let at = offset as usize;
            for (i, w) in rows().words().into_iter().enumerate() {
                push[at + 4 * i..at + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
            }
            written = true;
        }
    }
    assert!(written, "the rows' push field");
    let buffers = init
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
    Resources {
        buffers,
        push,
        spec,
    }
}

/// `m` over the grid with `image` bound, in both scheduling orders, which must agree: the final
/// buffers and the view.
fn run(m: &TypedModule, res: &Resources, image: &Image) -> (Vec<Vec<u8>>, Image) {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [GRID[0], GRID[1], 1],
            order,
            ..Config::default()
        };
        let (o, images) = run_with_images(m, &cfg, res.clone(), vec![image.clone()])
            .unwrap_or_else(|e| panic!("interpreter: {e:?}"));
        let image = images.into_iter().next().expect("the view");
        outs.push((o.buffers, image.texels.clone(), image));
    }
    assert_eq!(outs[0].0, outs[1].0, "the scheduling orders disagree");
    assert_eq!(outs[0].1, outs[1].1, "the scheduling orders disagree");
    let (buffers, _, image) = outs.swap_remove(0);
    (buffers, image)
}

/// The grid's columns, as the kernels' `constant uint &columns`.
fn columns() -> Vec<u8> {
    GRID[0].to_le_bytes().to_vec()
}

/// A slot whose function only reads, writes or asks the size of its texture states its rows; a
/// slot that samples it states none.
#[test]
fn reads_writes_and_sizes_state_rows_and_samples_do_not() {
    let Some(bitcode) = bitcode() else { return };
    let p = lavapipe();
    let fact = RuntimeFact::TexelRows { slot: TEXTURE };
    for entry in ["rows_invert", "rows_read", "rows_write"] {
        let a = compile(&bitcode, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert!(a.facts.contains(&fact), "{entry} states its rows");
    }
    let a = compile(&bitcode, "rows_sample", &p).expect("rows_sample");
    assert!(!a.facts.contains(&fact), "a sample states no rows");
}

/// Under the rows variant each kernel's module validates on the profile, the slot a texel
/// buffer: uniform for a slot that only reads, storage for one that writes.
#[test]
fn the_rows_variant_validates() {
    let Some(bitcode) = bitcode() else { return };
    let p = lavapipe();
    for (entry, usage) in [
        ("rows_invert", ImageUsage::Storage),
        ("rows_read", ImageUsage::Sampled),
        ("rows_write", ImageUsage::Storage),
    ] {
        let a = compile_with(&bitcode, entry, &p, &rows_inputs())
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        let (_, m) = typed_module_with(&bitcode, entry, &p, &rows_inputs()).expect(entry);
        rows_binding(&m, usage);
        let fact = RuntimeFact::TexelRows { slot: TEXTURE };
        assert!(
            a.facts.contains(&fact),
            "{entry}'s variant states its rows, as the plan holds it to"
        );
    }
}

/// A read of texel `(x, y)` reads the view's texel `first + y · row + x`, and Metal's zero texel
/// past the texture's width or height, never the next row's.
#[test]
fn a_read_takes_its_texel_by_row() {
    let Some(bitcode) = bitcode() else { return };
    let p = lavapipe();
    let (plan, m) = typed_module_with(&bitcode, "rows_read", &p, &rows_inputs()).expect("module");
    let slot = rows_binding(&m, ImageUsage::Sampled);
    let out = vec![0u8; (GRID[0] * GRID[1] * 16) as usize];
    let res = resources(&plan, &[out, columns()]);
    let (buffers, _) = run(&m, &res, &view(slot));
    let (words, _) = buffers[0].as_chunks::<4>();
    let texels: Vec<[u32; 4]> = (words.as_chunks::<4>().0.iter())
        .map(|t| t.map(u32::from_le_bytes))
        .collect();
    let inside: BTreeMap<(u32, u32), u32> = texture_indices().into_iter().collect();
    for y in 0..GRID[1] {
        for x in 0..GRID[0] {
            let got = texels[(y * GRID[0] + x) as usize];
            let want = inside.get(&(x, y)).map_or([0; 4], |&i| view_texel(i));
            assert_eq!(got, want, "texel ({x}, {y})");
        }
    }
}

/// A write of texel `(x, y)` stores the view's texel `first + y · row + x`, and nothing past the
/// texture: the texels before the first and between the rows keep their values.
#[test]
fn a_write_stores_its_texel_by_row_and_nothing_else() {
    let Some(bitcode) = bitcode() else { return };
    let p = lavapipe();
    let (plan, m) = typed_module_with(&bitcode, "rows_write", &p, &rows_inputs()).expect("module");
    let slot = rows_binding(&m, ImageUsage::Storage);
    let value = |x: u32, y: u32| -> [u32; 4] {
        std::array::from_fn(|lane| {
            (1000.0 + (y * GRID[0] + x) as f32 * 4.0 + lane as f32).to_bits()
        })
    };
    let mut input = Vec::new();
    for y in 0..GRID[1] {
        for x in 0..GRID[0] {
            for lane in value(x, y) {
                input.extend(lane.to_le_bytes());
            }
        }
    }
    let res = resources(&plan, &[input, columns()]);
    let (_, image) = run(&m, &res, &view(slot));
    let mut want: Vec<[u32; 4]> = (0..rows().elements() as u32).map(view_texel).collect();
    for ((x, y), i) in texture_indices() {
        want[i as usize] = value(x, y);
    }
    assert_eq!(image.texels, want, "the view after the writes");
}

/// A kernel that asks the size, reads and writes inverts each texel inside the texture by its
/// row, and nothing else.
#[test]
fn a_read_write_kernel_asks_the_rows_size() {
    let Some(bitcode) = bitcode() else { return };
    let p = lavapipe();
    let (plan, m) = typed_module_with(&bitcode, "rows_invert", &p, &rows_inputs()).expect("module");
    let slot = rows_binding(&m, ImageUsage::Storage);
    let res = resources(&plan, &[]);
    let (_, image) = run(&m, &res, &view(slot));
    let mut want: Vec<[u32; 4]> = (0..rows().elements() as u32).map(view_texel).collect();
    for (_, i) in texture_indices() {
        let t = view_texel(i).map(f32::from_bits);
        want[i as usize] = [
            (1.0 - t[0]).to_bits(),
            (1.0 - t[1]).to_bits(),
            (1.0 - t[2]).to_bits(),
            t[3].to_bits(),
        ];
    }
    assert_eq!(image.texels, want, "the view after the inversion");
}
