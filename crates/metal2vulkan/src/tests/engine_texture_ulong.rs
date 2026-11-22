//! Metal's `ulong` textures end to end on the CPU interpreter's image model, on every committed
//! profile that compiles them, one with `robustImageAccess` but not `robustImageAccess2`, and two
//! without any robustness: the owned fixture `unk_ulong_rw`.
//!
//! Metal binds only an `RG32Uint` texture to a `ulong` texture parameter, its low word in R and
//! its high word in G; a read returns `(v, 0, 0, 1)`, `(0, 0, 0, 1)` out of bounds, and a write
//! stores lane 0 only (measured on Apple M3 Max). The slot is an `RG32` unsigned image declared
//! `Rg32ui`, whose two words the shader joins and splits, so no 64-bit image is needed:
//!
//! - reads and writes of 1D, 1D array, 2D (16-bit coordinates), 2D array, 3D, and buffer
//!   textures, in and out of bounds on each axis, under each containment class;
//! - a sample-access read (`OpImageFetch`), a read-access read at 16-bit coordinates, and a
//!   write-access write;
//! - the interface states each slot's `ulong` texel, and the module declares no 64-bit image.
//!
//! A profile without `shaderInt64` holds the texels as their low and high words, and one without
//! `shaderStorageImageExtendedFormats` declares the storage images without a format, so no profile
//! needs a 64-bit integer for them. Previously each kernel was refused `UnknownIntrinsic`, so these
//! tests fail without it. A profile without a capability a kernel needs (six storage images, a
//! storage format) refuses it with the typed device gap, which the tests accept and count.
//!
//! Metal's `ulong` `atomic_{min,max}` (the owned fixture `unk_ulong_atomics`) compare the texel's
//! unsigned 64-bit value and return nothing. Their one exact form is a 64-bit image atomic, which
//! no runtime enables, so they are refused on every profile (`Unsupported{ImageAtomics}`, not
//! implemented).
//!
//! `llvm-as` assembles the fixtures; without them the tests skip visibly.

use std::collections::BTreeMap;

use crate::bind::{BindState, BoundBuffer, BoundTexture, StageTable, TextureFacts};
use crate::budget::Budget;
use crate::device::profile::{DeviceProfile, Robustness};
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, Feature, Needs, Unsupported};
use crate::library::Library;
use crate::lower::select::Plan;
use crate::metal::{PixelFormat, TextureType, TextureUsage};
use crate::pipeline::interface::{PushLoc, SlotInterface};
use crate::pipeline::sym::{
    Axis, PushField, ResourceKind, ResourceSlot, SampledType, SlotClass, SpecKey, Stage,
};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, typed_module, AssembleError,
};
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, StageSource,
};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;

const OP_TYPE_INT: u16 = 21;
const OP_TYPE_IMAGE: u16 = 25;
const OP_CAPABILITY: u16 = 17;
const FORMAT_UNKNOWN: u32 = 0;
const FORMAT_RG32UI: u32 = 35;
/// `Int64ImageEXT`.
const CAPABILITY_INT64_IMAGE: u32 = 5016;

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

/// `stem` with the features in `drop` removed.
fn without(stem: &str, drop: &[VkFeature]) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    for f in drop {
        desc.features.remove(f);
    }
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
        .expect("device")
        .profile()
        .clone()
}

/// Every committed profile, `moltenvk-m3` with `robustImageAccess` but not `robustImageAccess2`,
/// and `moltenvk-m3` and `lavapipe` without robustness.
fn profiles() -> Vec<(String, DeviceProfile)> {
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
    // A device that holds 64-bit values as two words (below the runtime's floor).
    out.push((
        "lavapipe without shaderInt64".into(),
        without("lavapipe", &[VkFeature::ShaderInt64]),
    ));
    out
}

/// What the interpreter's image does outside itself on `p`: the device's texel for an `RG32` view,
/// `(0, 0, 0, 1)`, and its discarded write where the device's robustness covers the access; a
/// poisoned texel where the device discards writes but the shader checks reads; a trap where the
/// shader contains both.
fn robust(p: &DeviceProfile, buffer: bool) -> Option<[u32; 4]> {
    let f = p.fields();
    match (buffer, f.robustness) {
        (true, Robustness::Rba2) => Some([0, 0, 0, 1]),
        (true, _) => None,
        (false, _) if f.images.robust_image_access2 => Some([0, 0, 0, 1]),
        (false, _) if f.images.robust_image_access => Some([0xDEAD_BEEF; 4]),
        (false, _) => None,
    }
}

/// The plan and module of `entry` on `p`; every module also links and passes L3 (`G-val`).
/// `None` when `p` lacks a capability it needs (a typed device gap); any other refusal, a
/// runtime input included (a `ulong` slot's format is Metal's one), fails the test.
fn module(bc: &[u8], entry: &str, stem: &str, p: &DeviceProfile) -> Option<(Plan, TypedModule)> {
    match typed_module(bc, entry, p) {
        Ok(m) => {
            let a = compile(bc, entry, p).unwrap_or_else(|e| panic!("{stem}: {entry}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: {entry}: L3: {e}"));
            Some(m)
        }
        Err(Error::Unsupported(u)) if matches!(u.needs, Needs::AnyOf(_)) => None,
        Err(e) => panic!("{stem}: {entry}: {e}"),
    }
}

/// Resources of a dispatch of 1x1x1 threadgroups, `[[buffer(0)]]` holding `out` bytes.
fn resources(plan: &Plan, out: usize) -> Resources {
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
    Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0xAB; out],
        }],
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

/// The output buffer and the images after a run over `lanes` threadgroups of one thread, in both
/// scheduling orders, which must agree.
fn run_both(
    m: &TypedModule,
    res: &Resources,
    images: &[Image],
    lanes: u32,
) -> (Vec<u64>, Vec<Image>) {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [lanes, 1, 1],
            order,
            ..Config::default()
        };
        let (o, imgs) = run_with_images(m, &cfg, res.clone(), images.to_vec())
            .unwrap_or_else(|e| panic!("interpreter: {e:?}"));
        let words: Vec<u64> = o.buffers[0]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| u64::from_le_bytes(*c))
            .collect();
        let texels: Vec<Vec<[u32; 4]>> = imgs.iter().map(|i| i.texels.clone()).collect();
        outs.push((words, texels, imgs));
    }
    assert_eq!(outs[0].0, outs[1].0, "scheduling orders disagree");
    assert_eq!(outs[0].1, outs[1].1, "scheduling orders disagree");
    let (words, _, imgs) = outs.swap_remove(0);
    (words, imgs)
}

/// The `RG32Uint` words of texel `n` of slot `s` and its `ulong` (low word first).
fn texel(s: u32, n: u32) -> ([u32; 4], u64) {
    let lo = 0xA000_0000 | s << 20 | n;
    let hi = 0x0B00_0000 | s << 20 | n << 4;
    ([lo, hi, 0, 1], u64::from(hi) << 32 | u64::from(lo))
}

/// One `ulong` texture of the fixture: its extent, layers, and how thread `t` addresses it.
struct Shape {
    extent: [u32; 3],
    layers: u32,
    /// `(x, y, z, layer)` of thread `t`.
    at: fn(u32) -> [u32; 4],
}

fn xyz(t: u32) -> [u32; 3] {
    if t < 10 {
        [t, t & 1, (t >> 1) & 1]
    } else {
        [t & 7, 2, 2]
    }
}

/// The fixture's six textures, slot by slot.
fn shapes() -> [Shape; 6] {
    [
        Shape {
            extent: [8, 1, 1],
            layers: 1,
            at: |t| [t, 0, 0, 0],
        },
        Shape {
            extent: [8, 1, 1],
            layers: 2,
            at: |t| [xyz(t)[0], 0, 0, xyz(t)[1]],
        },
        Shape {
            extent: [8, 2, 1],
            layers: 1,
            at: |t| [xyz(t)[0], xyz(t)[1], 0, 0],
        },
        Shape {
            extent: [8, 2, 1],
            layers: 2,
            at: |t| [xyz(t)[0], xyz(t)[1], 0, xyz(t)[1]],
        },
        Shape {
            extent: [8, 2, 2],
            layers: 1,
            at: |t| {
                let [x, y, z] = xyz(t);
                [x, y, z, 0]
            },
        },
        Shape {
            extent: [8, 1, 1],
            layers: 1,
            at: |t| [t, 0, 0, 0],
        },
    ]
}

/// The texel index of `at` in an image of `shape`, `None` outside it.
fn index(img: &Image, at: [u32; 4]) -> Option<usize> {
    img.index(0, at[3], 0, [at[0], at[1], at[2]])
}

/// Each `ulong` read joins R (the low word) and G of its `RG32` texel into `(v, 0, 0, 1)`, and
/// `(0, 0, 0, 1)` outside the texture on every axis (Metal's lanes, measured); each write stores
/// lane 0's low word in R and its high word in G at the texel, and nothing outside it.
#[test]
fn ulong_reads_join_the_two_words_and_writes_split_them() {
    let Some(bc) = fixture("unk_ulong_rw") else {
        return;
    };
    let (mut ran, mut wide, mut gaps) = (0, 0, Vec::new());
    for (stem, p) in profiles() {
        let Some((plan, m)) = module(&bc, "unk_ulong_rw", &stem, &p) else {
            gaps.push(stem);
            continue;
        };
        // Without `shaderInt64` the texels are held as their low and high words.
        wide += usize::from(!p.fields().leaf.int64);
        let mut images = Vec::new();
        for (s, shape) in shapes().iter().enumerate() {
            let s = s as u32;
            let mut img = Image::new(texture(s), shape.extent, shape.layers, 1, 1);
            for (n, t) in img.texels.iter_mut().enumerate() {
                *t = texel(s, n as u32).0;
            }
            img.robust = robust(&p, s == 5);
            images.push(img);
        }
        let lanes = 12;
        let res = resources(&plan, 6 * 32 * lanes as usize);
        let (out, after) = run_both(&m, &res, &images, lanes);
        let mut written = vec![Vec::new(); 6];
        for t in 0..lanes {
            for (s, shape) in shapes().iter().enumerate() {
                let at = (shape.at)(t);
                let got = &out[4 * (6 * t as usize + s)..][..4];
                let k = 0xF00D_0000_C0DE_0000u64 + u64::from(t);
                match index(&images[s], at) {
                    Some(n) => {
                        let v = texel(s as u32, n as u32).1;
                        assert_eq!(got, [v, 0, 0, 1], "{stem} slot {s} lane {t} {at:?}");
                        let w = v ^ k;
                        let stored = after[s].texels[n];
                        assert_eq!(
                            [stored[0], stored[1]],
                            [w as u32, (w >> 32) as u32],
                            "{stem} slot {s} lane {t}: written"
                        );
                        written[s].push(n);
                    }
                    None => assert_eq!(got, [0, 0, 0, 1], "{stem} slot {s} lane {t} {at:?}"),
                }
            }
        }
        for (s, img) in after.iter().enumerate() {
            for (n, t) in img.texels.iter().enumerate() {
                if !written[s].contains(&n) {
                    assert_eq!(*t, texel(s as u32, n as u32).0, "{stem} slot {s} texel {n}");
                }
            }
        }
        ran += 1;
    }
    // Only `runtime-floor` (six storage images, four allowed; no declared `Rg32ui` and no
    // formatless access) refuses it.
    assert!(
        ran > 0 && wide > 0,
        "ran {ran}, wide {wide}, device gaps {gaps:?}"
    );
    assert!(gaps.iter().all(|g| g == "runtime-floor"), "{gaps:?}");
}

/// A sample-access `ulong` read is a fetch of the sampled `RG32` view, a read-access one at
/// 16-bit coordinates the same, and a write-access write stores the sum's two words: in bounds
/// for threads 0 to 15, out of bounds (rows 2 and 3) from thread 16 on.
#[test]
fn ulong_sampled_reads_and_write_only_writes() {
    let Some(bc) = fixture("unk_ulong_rw") else {
        return;
    };
    let mut ran = 0;
    for (stem, p) in profiles() {
        let Some((plan, m)) = module(&bc, "unk_ulong_access", &stem, &p) else {
            continue;
        };
        let mut images = Vec::new();
        for (s, extent) in [[8, 2, 1], [8, 2, 1], [8, 2, 1]].into_iter().enumerate() {
            let s = s as u32;
            let mut img = Image::new(texture(s), extent, 1, 1, 1);
            for (n, t) in img.texels.iter_mut().enumerate() {
                *t = texel(s, n as u32).0;
            }
            img.robust = robust(&p, false);
            images.push(img);
        }
        let lanes = 20;
        let res = resources(&plan, 2 * 32 * lanes as usize);
        let (out, after) = run_both(&m, &res, &images, lanes);
        for t in 0..lanes {
            let n = (t & 7) + 8 * ((t >> 3) & 1);
            let at = 4 * 2 * t as usize;
            if t < 16 {
                let (vs, vr) = (texel(0, n).1, texel(1, n).1);
                assert_eq!(out[at..at + 4], [vs, 0, 0, 1], "{stem} lane {t} sampled");
                assert_eq!(out[at + 4..at + 8], [vr, 0, 0, 1], "{stem} lane {t} read");
                let w = vs.wrapping_add(vr);
                let stored = after[2].texels[n as usize];
                assert_eq!(
                    [stored[0], stored[1]],
                    [w as u32, (w >> 32) as u32],
                    "{stem} lane {t}"
                );
            } else {
                assert_eq!(out[at..at + 8], [0, 0, 0, 1, 0, 0, 0, 1], "{stem} lane {t}");
            }
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// The instructions of a SPIR-V module: opcode and its words.
fn insts(words: &[u32]) -> Vec<(u16, &[u32])> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = ((words[i] >> 16) as usize).max(1);
        out.push((
            (words[i] & 0xFFFF) as u16,
            &words[i..(i + n).min(words.len())],
        ));
        i += n;
    }
    out
}

/// Every `ulong` slot's interface states the `ulong` texel, every image is a 32-bit unsigned one
/// (a storage image declared `Rg32ui`, or formatless on a profile without
/// `shaderStorageImageExtendedFormats`), and no module needs a 64-bit image.
#[test]
fn ulong_slots_are_rg32_images_of_ulong_texels() {
    let Some(bc) = fixture("unk_ulong_rw") else {
        return;
    };
    let mut compiled = 0;
    for (stem, p) in profiles() {
        for (entry, storage) in [("unk_ulong_rw", 6), ("unk_ulong_access", 1)] {
            let a = match compile(&bc, entry, &p) {
                Ok(a) => a,
                Err(Error::Unsupported(u)) if matches!(u.needs, Needs::AnyOf(_)) => continue,
                Err(e) => panic!("{stem}: {entry}: {e}"),
            };
            compiled += 1;
            let textures: Vec<SampledType> = a
                .interface
                .slots
                .iter()
                .filter(|(slot, _)| slot.class == SlotClass::Texture)
                .map(|(_, i)| match i {
                    SlotInterface::Texture(t) => t.sampled_type,
                    other => panic!("{stem}: {other:?}"),
                })
                .collect();
            assert!(!textures.is_empty(), "{stem}: {entry}");
            assert!(
                textures.iter().all(|&t| t == SampledType::Uint64),
                "{stem}: {entry}: {textures:?}"
            );
            let words = &a.module.words;
            let ints: BTreeMap<u32, (u32, u32)> = insts(words)
                .into_iter()
                .filter(|(o, _)| *o == OP_TYPE_INT)
                .map(|(_, w)| (w[1], (w[2], w[3])))
                .collect();
            let images: Vec<&[u32]> = insts(words)
                .into_iter()
                .filter(|(o, _)| *o == OP_TYPE_IMAGE)
                .map(|(_, w)| w)
                .collect();
            for w in &images {
                assert_eq!(ints.get(&w[2]), Some(&(32, 0)), "{stem}: {entry}: {w:?}");
            }
            // A storage image (`Sampled` 2) is declared `Rg32ui` where the profile has
            // `shaderStorageImageExtendedFormats`, otherwise without a format.
            let declared = if p.fields().images.storage_extended_formats {
                FORMAT_RG32UI
            } else {
                FORMAT_UNKNOWN
            };
            let stores: Vec<u32> = images.iter().filter(|w| w[7] == 2).map(|w| w[8]).collect();
            assert_eq!(
                stores,
                vec![declared; storage],
                "{stem}: {entry}: {images:?}"
            );
            assert!(
                !insts(words)
                    .into_iter()
                    .any(|(o, w)| o == OP_CAPABILITY && w[1] == CAPABILITY_INT64_IMAGE),
                "{stem}: {entry}"
            );
        }
    }
    assert!(compiled > 0);
}

/// `unk_ulong_atomics`: a `ulong` texture's `atomic_max` and `atomic_min` (1D, 1D array, 2D at
/// 16-bit coordinates, 2D array, 3D, buffer) are refused on every profile,
/// `Unsupported{ImageAtomics}` not implemented, as is `unk_ulong_atomic_rw`, which also reads and
/// writes another `ulong` slot, where no device gap of that slot comes first.
#[test]
fn ulong_atomics_are_refused_on_every_device() {
    let Some(bc) = fixture("unk_ulong_atomics") else {
        return;
    };
    let refusal = Unsupported::not_implemented(Feature::ImageAtomics);
    let profiles = profiles();
    let mut refused = 0;
    for (stem, p) in &profiles {
        for entry in ["unk_ulong_atomics", "unk_ulong_atomic_rw"] {
            match compile(&bc, entry, p) {
                Err(Error::Unsupported(u)) if u == refusal => refused += 1,
                Err(Error::Unsupported(u))
                    if entry == "unk_ulong_atomic_rw" && matches!(u.needs, Needs::AnyOf(_)) => {}
                other => panic!("{stem}: {entry}: {:?}", other.map(|_| ())),
            }
        }
    }
    assert!(refused >= profiles.len(), "{refused}");
}

fn device(stem: &str) -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device")
}

fn pipeline(dev: &Device, bc: &[u8], entry: &str) -> ComputePipeline {
    let lib = Library::from_air(bc).expect("library");
    let f = FunctionDescriptor::new(lib.function(entry).expect("function"));
    let d = ComputePipelineDescriptor::new(StageSource::Function(f));
    dev.compile_compute_pipeline(&d, &Budget::standard())
        .unwrap_or_else(|e| panic!("{entry}: {e}"))
}

/// A bind state with `RG32Uint` textures of `types` at texture slots 0, 1, ... (`None`: left
/// unbound), Metal's usage for atomics, and a buffer at slot 0.
fn ulong_state(types: &[Option<TextureType>]) -> BindState {
    let mut t = StageTable::default();
    t.buffers.insert(
        0,
        BoundBuffer::Bytes {
            len: 4096,
            attribute_stride: None,
        },
    );
    for (i, ty) in types.iter().enumerate() {
        let Some(ty) = *ty else { continue };
        let mut facts = TextureFacts::new(ty, PixelFormat::RG32Uint);
        facts.usage =
            TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC;
        t.textures.insert(
            i as u32,
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

/// A texture buffer made with a swizzle at a slot the function reads as `ulong` texels is refused
/// (`Unsupported { feature: TextureSwizzle }`): the read applies no swizzle, and Metal's swizzled
/// `ulong` texel is not measured. Unswizzled, it binds.
#[test]
fn a_swizzled_texture_buffer_read_as_ulong_is_refused() {
    use crate::metal::{TextureSwizzle as S, TextureSwizzleChannels as C};
    let Some(bc) = fixture("unk_ulong_rw") else {
        return;
    };
    let dev = device("amd-780m-radv");
    let rw = pipeline(&dev, &bc, "unk_ulong_rw");
    let state = |swizzle| {
        let mut t = ulong_state(&[]);
        t.stages
            .get_mut(&Stage::Kernel)
            .expect("kernel")
            .textures
            .insert(
                5,
                BoundTexture::Buffer {
                    pixel_format: PixelFormat::RG32Uint,
                    offset: 0,
                    length: 256,
                    bytes_per_row: 256,
                    resource_id: None,
                    swizzle,
                    parts: 1,
                },
            );
        t
    };
    rw.plan(&dev, &state(C::DEFAULT).shape()).expect("plan");
    let bgr1 = C::new(S::Blue, S::Green, S::Red, S::One);
    match rw.plan(&dev, &state(bgr1).shape()) {
        Err(crate::error::BindError::Unsupported(u)) => assert_eq!(
            u,
            crate::error::Unsupported::not_implemented(Feature::TextureSwizzle)
        ),
        other => panic!("{other:?}"),
    }
}
