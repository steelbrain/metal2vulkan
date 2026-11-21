//! The texture operations the decoder refused previously end to end on the CPU interpreter's image
//! model, on every committed profile that compiles them without runtime inputs (and, for
//! containment, two profiles without robustness):
//!
//! - family A: the 1D-array read and width query, the size queries of multisampled arrays, texture
//!   buffers, and depth multisampled and cube textures, and the reads of depth cubes, depth cube
//!   arrays, and depth multisampled arrays (owned fixtures `unk_read_1d_array`,
//!   `unk_size_queries`, `unk_read_depth_cube`, `unk_read_depth_ms_array`).
//!
//! Previously each of these kernels was refused `UnknownIntrinsic`, so every test here fails
//! without it. A profile that lacks a capability a kernel needs (a cube array on `vk12-floor`)
//! refuses it with the typed device gap, which the tests accept and count.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use std::collections::BTreeMap;

use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, Needs};
use crate::lower::select::Plan;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, typed_module, AssembleError,
};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;

fn fixture(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

/// `stem` with the robustness features removed, so the shader contains its own accesses.
fn unrobust(stem: &str) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    for f in [
        VkFeature::RobustImageAccess,
        VkFeature::RobustImageAccess2,
        VkFeature::RobustBufferAccess,
        VkFeature::RobustBufferAccess2,
    ] {
        desc.features.remove(&f);
    }
    Device::from_queries(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// Every committed profile, and `moltenvk-m3` and `lavapipe` without robustness.
fn profiles() -> Vec<(String, DeviceProfile)> {
    let mut out = committed_profiles().expect("profiles");
    for stem in ["moltenvk-m3", "lavapipe"] {
        out.push((format!("{stem} unrobust"), unrobust(stem)));
    }
    out
}

/// The module of `entry` on `p`: every module also links and passes L3 (`G-val`). `None` when
/// `p` needs a runtime input for it, or lacks a capability it needs (a typed device gap); any
/// other refusal fails the test.
fn module(bc: &[u8], entry: &str, stem: &str, p: &DeviceProfile) -> Option<(Plan, TypedModule)> {
    match typed_module(bc, entry, p) {
        Ok(m) => {
            let a = compile(bc, entry, p).unwrap_or_else(|e| panic!("{stem}: {entry}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: {entry}: L3: {e}"));
            Some(m)
        }
        Err(Error::NeedsInput(_)) => None,
        Err(Error::Unsupported(u)) if matches!(u.needs, Needs::AnyOf(_)) => None,
        Err(e) => panic!("{stem}: {entry}: {e}"),
    }
}

/// Resources of a dispatch of 1x1x1 threadgroups, `[[buffer(i)]]` holding `init[i]`.
fn resources(plan: &Plan, init: &[Vec<u8>]) -> Resources {
    let mut spec = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(axis), 1);
    }
    spec.insert(SpecKey::FlatLocalSize, 1);
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
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

fn texture(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Texture,
        index,
    }
}

/// The output buffers of a run over `grid` threadgroups of one thread, in both scheduling orders,
/// which must agree.
fn run_both(m: &TypedModule, res: &Resources, images: &[Image], grid: [u32; 3]) -> Vec<Vec<u8>> {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: grid,
            order,
            ..Config::default()
        };
        let (o, _) = run_with_images(m, &cfg, res.clone(), images.to_vec())
            .unwrap_or_else(|e| panic!("interpreter: {e:?}"));
        outs.push(o.buffers);
    }
    assert_eq!(outs[0], outs[1], "scheduling orders disagree");
    outs.swap_remove(0)
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

/// Set texel `(x, layer)` (and `y`, `sample`) of level 0 of `img` to `t`.
fn put(img: &mut Image, layer: u32, sample: u32, xy: [u32; 2], t: [u32; 4]) {
    let i = img
        .index(0, layer, sample, [xy[0], xy[1], 0])
        .expect("texel");
    img.texels[i] = t;
}

/// A 1D-array read addresses texel `x` of slice `l` at either coordinate width and through a
/// sampled texture's read, and the width query reads the array's width: lane i reads texel
/// (i mod 5, (i / 5) mod 3) of three 5-texel, 3-slice arrays (`float`, `uint` at 16-bit
/// coordinates, `half`).
#[test]
fn one_d_array_reads_address_texel_then_slice() {
    let Some(bc) = fixture("unk_read_1d_array") else {
        return;
    };
    let (w, n) = (5u32, 3u32);
    let ft = |x: u32, l: u32| {
        [
            fl((x + 10 * l) as f32),
            fl(-(x as f32)),
            fl(0.5 * l as f32),
            fl(1.0),
        ]
    };
    let ut = |x: u32, l: u32| [x, l, 1000 + x, u32::MAX - l];
    // Every component is a `half`, so the read's rounding to `half` keeps it.
    let ht = |x: u32, l: u32| [fl(0.25 * x as f32), fl(l as f32 + 0.5), fl(-2.0), fl(0.125)];
    let mut ran = 0;
    for (stem, p) in profiles() {
        let Some((plan, m)) = module(&bc, "unk_read_1d_array", &stem, &p) else {
            continue;
        };
        let mut images = [
            Image::new(texture(0), [w, 1, 1], n, 1, 1),
            Image::new(texture(1), [w, 1, 1], n, 1, 1),
            Image::new(texture(2), [w, 1, 1], n, 1, 1),
        ];
        for l in 0..n {
            for x in 0..w {
                put(&mut images[0], l, 0, [x, 0], ft(x, l));
                put(&mut images[1], l, 0, [x, 0], ut(x, l));
                put(&mut images[2], l, 0, [x, 0], ht(x, l));
            }
        }
        let lanes = w * n + 2;
        let res = resources(&plan, &[vec![0; 64 * lanes as usize]]);
        let out = words(&run_both(&m, &res, &images, [lanes, 1, 1])[0]);
        for i in 0..lanes {
            let (x, l) = (i % w, (i / w) % n);
            let at = 16 * i as usize;
            assert_eq!(out[at..at + 4], ft(x, l), "{stem} lane {i} float");
            assert_eq!(out[at + 4..at + 8], ut(x, l), "{stem} lane {i} uint");
            assert_eq!(out[at + 8..at + 12], ht(x, l), "{stem} lane {i} half");
            assert_eq!(out[at + 12..at + 16], [w, w, w, n], "{stem} lane {i} sizes");
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// The size queries of a multisampled texture array, a texture buffer, a depth multisampled
/// texture and texture array, and a depth cube and cube array read their bound images.
#[test]
fn size_queries_of_multisampled_buffer_and_depth_cube_textures() {
    let Some(bc) = fixture("unk_size_queries") else {
        return;
    };
    let mut ran = 0;
    let mut gaps = 0;
    for (stem, p) in profiles() {
        let Some((plan, m)) = module(&bc, "unk_size_queries", &stem, &p) else {
            gaps += 1;
            continue;
        };
        let images = [
            Image::new(texture(0), [3, 2, 1], 4, 1, 2),
            Image::new(texture(1), [7, 1, 1], 1, 1, 1),
            Image::new(texture(2), [5, 3, 1], 1, 1, 4),
            Image::new(texture(3), [2, 6, 1], 3, 1, 2),
            Image::new(texture(4), [4, 4, 1], 6, 3, 1),
            Image::new(texture(5), [2, 2, 1], 12, 1, 1),
        ];
        let res = resources(&plan, &[vec![0; 80]]);
        let out = words(&run_both(&m, &res, &images, [1, 1, 1])[0]);
        assert_eq!(
            out,
            [3, 2, 4, 2, 7, 5, 3, 4, 2, 6, 3, 2, 4, 4, 2, 2, 2, 3, 1, 0],
            "{stem}"
        );
        ran += 1;
    }
    assert!(ran > 0, "ran {ran}, device gaps {gaps}");
}

/// A depth cube read of face f is layer f of the cube's 2D-array view, and a depth cube-array read
/// of face f of cube c is layer f + 6c, at either coordinate width: lane i reads face i mod 6 at
/// texel ((i / 6) mod 2, (i / 12) mod 2) of a 2x2 depth cube, and of cube (i / 24) mod 2 of a
/// depth cube array.
#[test]
fn depth_cube_reads_address_face_then_cube() {
    let Some(bc) = fixture("unk_read_depth_cube") else {
        return;
    };
    let depth = |layer: u32, x: u32, y: u32| fl(layer as f32 + 0.125 * x as f32 + 0.5 * y as f32);
    let mut ran = 0;
    for (stem, p) in profiles() {
        let Some((plan, m)) = module(&bc, "unk_read_depth_cube", &stem, &p) else {
            continue;
        };
        let mut c = Image::new(texture(0), [2, 2, 1], 6, 1, 1);
        let mut a = Image::new(texture(1), [2, 2, 1], 12, 1, 1);
        for img in [&mut c, &mut a] {
            for layer in 0..img.layers {
                for y in 0..2 {
                    for x in 0..2 {
                        // The depth is the first component; the others must not be read.
                        let d = depth(100 * (img.slot.index + 1) + layer, x, y);
                        put(img, layer, 0, [x, y], [d, 7, 7, 7]);
                    }
                }
            }
        }
        let lanes = 48;
        let res = resources(&plan, &[vec![0; 16 * lanes]]);
        let out = words(&run_both(&m, &res, &[c, a], [lanes as u32, 1, 1])[0]);
        for i in 0..lanes as u32 {
            let (face, x, y, cube) = (i % 6, (i / 6) % 2, (i / 12) % 2, (i / 24) % 2);
            let dc = depth(100 + face, x, y);
            let da = depth(200 + face + 6 * cube, x, y);
            let at = 4 * i as usize;
            assert_eq!(out[at..at + 4], [dc, dc, da, da], "{stem} lane {i}");
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// A depth multisampled-array read reads sample s of texel (x, y) in slice l, at either
/// coordinate width.
#[test]
fn depth_multisampled_array_reads_address_sample_texel_then_slice() {
    let Some(bc) = fixture("unk_read_depth_ms_array") else {
        return;
    };
    let (w, h, s, n) = (2u32, 3u32, 4u32, 2u32);
    let depth = |l: u32, sample: u32, x: u32, y: u32| {
        fl(l as f32 * 64.0 + sample as f32 * 8.0 + y as f32 * 2.0 + x as f32 * 0.5)
    };
    let mut ran = 0;
    for (stem, p) in profiles() {
        let Some((plan, m)) = module(&bc, "unk_read_depth_ms_array", &stem, &p) else {
            continue;
        };
        let mut t = Image::new(texture(0), [w, h, 1], n, 1, s);
        for l in 0..n {
            for sample in 0..s {
                for y in 0..h {
                    for x in 0..w {
                        put(&mut t, l, sample, [x, y], [depth(l, sample, x, y), 9, 9, 9]);
                    }
                }
            }
        }
        let lanes = s * w * h * n;
        let res = resources(&plan, &[vec![0; 8 * lanes as usize]]);
        let out = words(&run_both(&m, &res, &[t], [lanes, 1, 1])[0]);
        for i in 0..lanes {
            let (sample, x, y, l) = (i % s, (i / s) % w, (i / (s * w)) % h, i / (s * w * h));
            let d = depth(l, sample, x, y);
            assert_eq!(
                out[2 * i as usize..2 * i as usize + 2],
                [d, d],
                "{stem} lane {i}"
            );
        }
        ran += 1;
    }
    assert!(ran > 0);
}
