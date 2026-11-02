//! Metal's value of a sampler's access in a block of a sparse image that holds no mapped tile.
//!
//! Where the profile enables `shaderResourceResidency`, a sparse texture is a sparse image, and a
//! block of it that holds no mapped tile reads the device's value: RADV's is zero in every lane,
//! where Metal reads a format's missing alpha as 1, and a device without
//! `residencyNonResidentStrict` reads any value. A sampler's access whose residency the function
//! reads there, and every read or access through a sampler whose residency it does not read, sets
//! the lanes Metal reads as constants (a channel the format lacks, through the texture's swizzle)
//! from the slot's `SpecKey::ConstantLanes`, and a pixel fetch whose residency the function reads,
//! naming its texel, reads zero where its residency says it is unmapped; a sample's and a gather's
//! are the device's own sparse access. The CPU interpreter fetches but does not sample, so a
//! read and a pixel sampler's fetch run here and the sample and gather are checked by their facts,
//! notes and specializations.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::{SparseTileMap, TileMapping};
use crate::device::requirements::VkFeature;
use crate::error::Error;
use crate::library::FunctionArtifact;
use crate::lower::select::Plan;
use crate::metal::{PixelFormat, TextureSwizzle, TextureSwizzleChannels};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{
    Axis, CombinedSampler, PushField, ResourceKind, ResourceSlot, SamplerRef, SlotClass, SlotRef,
    SpecKey, Stage,
};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, typed_module, AssembleError,
};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;

const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let text = format!(
        "target datalayout = \"{DATALAYOUT}\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n{body}"
    );
    let dir = std::env::temp_dir().join(format!(
        "m2v-sparse-unmapped-{}-{}-{name}",
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

fn ok(
    a: Result<std::sync::Arc<FunctionArtifact>, Error>,
    p: &crate::device::profile::DeviceProfile,
    what: &str,
) -> std::sync::Arc<FunctionArtifact> {
    let a = a.unwrap_or_else(|e| panic!("{what}: compile: {e}"));
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
    a
}

/// Whether profile `p` makes sparse textures sparse images (it enables `shaderResourceResidency`).
fn sparse_images(p: &crate::device::profile::DeviceProfile) -> bool {
    p.fields()
        .enabled
        .features
        .contains(&VkFeature::ShaderResourceResidency)
}

fn texture_slot(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index,
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
            // A bound value read as data, which the interpreter writes from `spec`.
            (PushField::Specialized(_), PushLoc::Push { .. }) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let buffers: Vec<Buffer> = init
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

/// Run over `grid` threadgroups of one thread in both scheduling orders; both must agree.
fn run_both(m: &TypedModule, res: &Resources, images: &[Image], grid: [u32; 3]) -> Vec<Vec<u8>> {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: grid,
            order,
            ..Config::default()
        };
        let (o, _) = run_with_images(m, &cfg, res.clone(), images.to_vec()).expect("run");
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

/// The constexpr sampler words of a nearest, clamp-to-edge pixel sampler (its sample is an exact
/// fetch).
const PIXEL_NEAREST: u64 = 0x007b_ff00_0008_8049;
/// A nearest, clamp-to-zero normalized sampler (`sps2_residency_probes`' `nearest_zero`).
const NEAREST_ZERO: u64 = 34_901_797_601_017_856;
/// A linear, clamp-to-zero normalized sampler (`sps2_residency_probes`' `linear_zero`).
const LINEAR_ZERO: u64 = 34_901_797_601_020_416;

/// A kernel over `texture2d<float, sample>` slot 0 and `uint4 *out` (buffer 0) that samples,
/// gathers, or fetches (`call`, its value `%r`) through the constexpr sampler of `words` at
/// `%c`, the centre of texel `gid` in pixels (`%uv` normalized over a 2x2 texture), and stores
/// the texel to `out[2 * i]` and `(resident(), 0, 0, 0)` to `out[2 * i + 1]`, `i = gid.x + 2 *
/// gid.y`.
fn kernel(words: u64, call: &str, decl: &str) -> String {
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8

define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, <2 x i32> %gid) {{
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %f = uitofp <2 x i32> %gid to <2 x float>
  %c = fadd <2 x float> %f, <float 0.5, float 0.5>
  %uv = fmul <2 x float> %c, <float 0.5, float 0.5>
  %r = call {{ <4 x float>, i8 }} {call}
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  %st = extractvalue {{ <4 x float>, i8 }} %r, 1
  %m = and i8 %st, 1
  %n = xor i8 %m, 1
  %w = zext i8 %n to i32
  %y2 = shl i32 %y, 1
  %i = add i32 %x, %y2
  %i2 = shl i32 %i, 1
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %i2
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  %i3 = or i32 %i2, 1
  %q = getelementptr inbounds <4 x i32>, ptr addrspace(1) %out, i32 %i3
  %wv = insertelement <4 x i32> zeroinitializer, i32 %w, i64 0
  store <4 x i32> %wv, ptr addrspace(1) %q, align 16
  ret void
}}

declare {{ <4 x float>, i8 }} {decl}

!air.kernel = !{{!0}}
!air.sampler_states = !{{!6}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}}
!6 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
"#
    )
}

const SAMPLE_DECL: &str = "@air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)";
const GATHER_DECL: &str = "@air.gather_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)";

/// A sample at level 0 of `%coord` through the constexpr sampler.
fn sample(coord: &str) -> String {
    format!("@air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> {coord}, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)")
}

/// A gather of component `component` at `%uv` through the constexpr sampler.
fn gather(component: u32) -> String {
    format!("@air.gather_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 {component}, i32 0)")
}

/// The `ConstantLanes` word of an `R32Float` texture made with `swizzle`.
fn r32_lanes(swizzle: TextureSwizzleChannels) -> u64 {
    let present = crate::metal::format::channels(PixelFormat::R32Float).expect("channels");
    u64::from(crate::metal::sampler::constant_lanes(present, swizzle))
}

/// The lanes Metal reads as constants are the channels a format lacks (red, green and blue 0,
/// alpha 1) and the `Zero` and `One` swizzles, through the texture's swizzle; a format holding
/// every channel has none under its own swizzle; a depth format states none.
#[test]
fn the_constant_lanes_are_the_channels_a_format_lacks_through_its_swizzle() {
    use crate::metal::format::channels;
    use crate::metal::sampler::constant_lanes;
    use TextureSwizzle as T;
    let s = TextureSwizzleChannels::new;
    let id = TextureSwizzleChannels::DEFAULT;
    let lanes = |pf: PixelFormat, sw: TextureSwizzleChannels| {
        constant_lanes(channels(pf).expect("channels"), sw)
    };
    // R8: green and blue read 0, alpha 1.
    assert_eq!(lanes(PixelFormat::R8Unorm, id), 0b1110 | 0b1000 << 4);
    assert_eq!(lanes(PixelFormat::R32Float, id), 0b1110 | 0b1000 << 4);
    assert_eq!(lanes(PixelFormat::BC4RUnorm, id), 0b1110 | 0b1000 << 4);
    // RG: blue 0, alpha 1; RGB: alpha 1.
    assert_eq!(lanes(PixelFormat::RG8Unorm, id), 0b1100 | 0b1000 << 4);
    assert_eq!(lanes(PixelFormat::B5G6R5Unorm, id), 0b1000 | 0b1000 << 4);
    assert_eq!(lanes(PixelFormat::ETC2RGB8, id), 0b1000 | 0b1000 << 4);
    // A8: red, green and blue 0.
    assert_eq!(lanes(PixelFormat::A8Unorm, id), 0b0111);
    // Every channel: none, but a swizzle's constants.
    assert_eq!(lanes(PixelFormat::RGBA8Unorm, id), 0);
    assert_eq!(
        lanes(
            PixelFormat::RGBA8Unorm,
            s(T::Red, T::Green, T::Zero, T::One)
        ),
        0b1100 | 0b1000 << 4
    );
    // Through a swizzle: each lane takes its source's constant.
    assert_eq!(
        lanes(
            PixelFormat::R8Unorm,
            s(T::Alpha, T::Alpha, T::Alpha, T::Alpha)
        ),
        0b1111 | 0b1111 << 4
    );
    assert_eq!(
        lanes(PixelFormat::R8Unorm, s(T::Alpha, T::Red, T::Green, T::Blue)),
        0b1101 | 0b0001 << 4
    );
    assert_eq!(
        lanes(PixelFormat::A8Unorm, s(T::Alpha, T::Red, T::One, T::Alpha)),
        0b0110 | 0b0100 << 4
    );
    assert_eq!(channels(PixelFormat::Depth32Float), None);
    assert_eq!(channels(PixelFormat::Stencil8), None);
    assert_eq!(channels(PixelFormat::Invalid), None);
}

/// On the interpreter: a nearest, clamp-to-edge pixel sampler's fetch of a 2x2 sparse `R32Float`
/// texture whose column x = 1 is unmapped, through the slot's tile map, on every profile whose
/// sparse textures are sparse images. The image's texels hold what a device reads: a mapped texel
/// garbage in the lanes the format lacks, and an unmapped one the value of a block that holds no
/// mapped tile (RADV's zero in every lane, or another device's garbage). The fetch reads a mapped
/// texel's red with Metal's constants, and an unmapped texel as Metal's zero memory, alpha 1, not
/// resident; through a swizzle each lane takes its source's. A texture that is not sparse (both
/// specializations 0) reads the image as it is. The stage states `ConstantLanes` and notes no
/// `SparseResidency`; elsewhere it states neither.
#[test]
fn a_pixel_fetch_reads_metals_value_of_an_unmapped_texel() {
    use TextureSwizzle as T;
    let Some(bc) = assemble(
        "unmapped_fetch",
        &kernel(PIXEL_NEAREST, &sample("%c"), SAMPLE_DECL),
    ) else {
        return;
    };
    let mut legs = [0; 2];
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = ok(compile(&bc, "k", &p), &p, &stem);
        assert_eq!(
            a.strategies
                .entries
                .get("sampler.pixel_fetch.resident")
                .map_or(0, |u| u.uses),
            1,
            "{stem}"
        );
        let (plan, m) = typed_module(&bc, "k", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let fact = RuntimeFact::ConstantLanes {
            slot: texture_slot(0),
        };
        let images = sparse_images(&p);
        assert_eq!(plan.facts.contains(&fact), images, "{stem}");
        assert!(
            !plan.precision.contains(&PrecisionNote::SparseResidency),
            "{stem}"
        );
        assert!(
            plan.facts.contains(&RuntimeFact::SparseTexture {
                slot: texture_slot(0),
                sparse_arm: false,
                tile_map: true,
            }),
            "{stem}"
        );
        legs[usize::from(images)] += 1;
        if !images {
            continue;
        }
        let mut map = SparseTileMap::new(
            crate::metal::TextureType::Type2D,
            [2, 2, 1],
            1,
            1,
            [1; 3],
            1,
        )
        .expect("tile map");
        map.update(TileMapping::Map, 0, 0, [0; 3], [1, 2, 1])
            .expect("map");
        // Each leg: the swizzle; the device's value of mapped texel `i`, its red `r` (`i + 1`)
        // in the lane the swizzle reads it and garbage in the lanes the format lacks; the
        // device's value of an unmapped texel; and what Metal reads from each.
        type Texel = fn(u32) -> [u32; 4];
        let legs_of: [(TextureSwizzleChannels, Texel, [u32; 4], Texel, [u32; 4]); 3] = [
            (
                TextureSwizzleChannels::DEFAULT,
                |r| [r, fl(10.0), fl(20.0), fl(30.0)],
                [0; 4],
                |r| [r, 0, 0, fl(1.0)],
                [0, 0, 0, fl(1.0)],
            ),
            (
                TextureSwizzleChannels::DEFAULT,
                |r| [r, fl(10.0), fl(20.0), fl(30.0)],
                [0xDEAD_BEEF; 4],
                |r| [r, 0, 0, fl(1.0)],
                [0, 0, 0, fl(1.0)],
            ),
            (
                TextureSwizzleChannels::new(T::Alpha, T::Red, T::One, T::Zero),
                |r| [fl(77.0), r, fl(88.0), fl(99.0)],
                [0xDEAD_BEEF; 4],
                |r| [fl(1.0), r, fl(1.0), 0],
                [fl(1.0), 0, fl(1.0), 0],
            ),
        ];
        for (swizzle, mapped, device, want_mapped, want_unmapped) in legs_of {
            let red = |i: usize| fl(i as f32 + 1.0);
            let what = format!("{stem} device {:#x} {swizzle:?}", device[0]);
            // The combined binding of slot 0 and the constexpr sampler, whose image the fetch reads.
            let combined = ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::CombinedTextureSampler(CombinedSampler {
                    sampler: SamplerRef::Constexpr([PIXEL_NEAREST, 0]),
                    compare: false,
                    integer: false,
                    unnormalized: false,
                }),
                index: 0,
            };
            let mut img = Image::new(combined, [2, 2, 1], 1, 1, 1);
            for (i, t) in img.texels.iter_mut().enumerate() {
                *t = if i % 2 == 1 { device } else { mapped(red(i)) };
            }
            let mut res = resources(&plan, &[vec![0; 32 * 4]]);
            res.buffers.push(Buffer {
                slot: Some(ResourceSlot {
                    stage: Stage::Kernel,
                    kind: ResourceKind::TileMap,
                    index: 0,
                }),
                address: None,
                bytes: map.to_bytes(),
            });
            res.spec.insert(SpecKey::SparseTexture(texture_slot(0)), 1);
            res.spec
                .insert(SpecKey::ConstantLanes(texture_slot(0)), r32_lanes(swizzle));
            let out = words(&run_both(&m, &res, &[img.clone()], [2, 2, 1])[0]);
            for i in 0..4usize {
                let (texel, resident) = match i % 2 {
                    0 => (want_mapped(red(i)), 1),
                    _ => (want_unmapped, 0),
                };
                assert_eq!(out[8 * i..8 * i + 4], texel, "{what} texel {i}");
                assert_eq!(out[8 * i + 4], resident, "{what} texel {i}");
            }
            // Not sparse: both specializations 0, every texel the image's, resident.
            let mut res = resources(&plan, &[vec![0; 32 * 4]]);
            res.buffers.push(Buffer {
                slot: Some(ResourceSlot {
                    stage: Stage::Kernel,
                    kind: ResourceKind::TileMap,
                    index: 0,
                }),
                address: None,
                bytes: vec![0; 64],
            });
            for (i, t) in img.texels.iter_mut().enumerate() {
                *t = mapped(red(i));
            }
            let out = words(&run_both(&m, &res, &[img], [2, 2, 1])[0]);
            for i in 0..4usize {
                assert_eq!(
                    out[8 * i..8 * i + 4],
                    mapped(red(i)),
                    "{what} plain texel {i}"
                );
                assert_eq!(out[8 * i + 4], 1, "{what} plain texel {i}");
            }
        }
    }
    assert!(legs.iter().all(|n| *n > 0), "legs {legs:?}");
}

/// A nearest sample, a gather and a linear sample through clamp-to-zero constexpr samplers whose
/// residency the function reads. Where the profile's sparse textures are sparse images
/// (`shaderResourceResidency`), each is the device's own sparse access, its word the device's
/// residency code: the slot takes the sparse arm and no tile map, the stage states
/// `ConstantLanes` and specializes on it, and notes `SparseResidency`. Elsewhere the slot's tile
/// map answers each by Metal's footprint, noting nothing.
#[test]
fn a_sparse_device_answers_a_samples_residency_itself() {
    let cases = [
        ("nearest", kernel(NEAREST_ZERO, &sample("%uv"), SAMPLE_DECL)),
        ("gather_x", kernel(NEAREST_ZERO, &gather(0), GATHER_DECL)),
        ("gather_w", kernel(NEAREST_ZERO, &gather(3), GATHER_DECL)),
        ("linear", kernel(LINEAR_ZERO, &sample("%uv"), SAMPLE_DECL)),
    ];
    let mut legs = [0; 2];
    for (name, text) in cases {
        let Some(bc) = assemble(name, &text) else {
            return;
        };
        for (stem, p) in committed_profiles().expect("profiles") {
            let what = format!("{stem} {name}");
            let a = ok(compile(&bc, "k", &p), &p, &what);
            let images = sparse_images(&p);
            let fact = RuntimeFact::ConstantLanes {
                slot: texture_slot(0),
            };
            assert_eq!(a.facts.contains(&fact), images, "{what}");
            assert_eq!(
                a.precision.contains(&PrecisionNote::SparseResidency),
                images,
                "{what}"
            );
            let keys = a.interface.read_keys();
            assert_eq!(
                keys.contains(&SpecKey::ConstantLanes(texture_slot(0))),
                images,
                "{what}: {keys:?}"
            );
            assert!(
                a.facts.contains(&RuntimeFact::SparseTexture {
                    slot: texture_slot(0),
                    sparse_arm: images,
                    tile_map: !images,
                }),
                "{what}: {:?}",
                a.facts
            );
            legs[usize::from(images)] += 1;
        }
    }
    assert!(legs.iter().all(|n| *n > 0), "legs {legs:?}");
}

/// A kernel over `texture2d<float, sample>` slot 0 and `float4 *out` (buffer 0) whose thread `gid`
/// reads texel `gid` (`read`) or fetches it through a nearest pixel sampler (otherwise), and
/// stores the texel to `out[gid.x + 2 * gid.y]`, never reading its residency.
fn unobserved_kernel(read: bool) -> String {
    let (call, decl) = if read {
        (
            "%rs = call ptr addrspace(2) @air.get_read_sampler()
  %r = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 0)",
            "declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)",
        )
    } else {
        (
            "%r = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %c, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)",
            "declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)",
        )
    };
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {PIXEL_NEAREST}, i64 0], align 8

define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, <2 x i32> %gid) {{
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %f = uitofp <2 x i32> %gid to <2 x float>
  %c = fadd <2 x float> %f, <float 0.5, float 0.5>
  {call}
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  %y2 = shl i32 %y, 1
  %i = add i32 %x, %y2
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %i
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}}

{decl}

!air.kernel = !{{!0}}
!air.sampler_states = !{{!6}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}}
!6 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
"#
    )
}

/// On the interpreter, for accesses whose residency the function does not read: a read, and a
/// nearest pixel sampler's fetch, of a 2x2 `R32Float` texture, on every profile. Where the
/// profile's sparse textures are sparse images the stage states `ConstantLanes` (and nothing sparse
/// besides), and notes nothing: each texel the image holds as a block that holds no mapped tile
/// reads it (RADV's zero in every lane, or another device's garbage), and as a mapped texel with
/// garbage in the lanes the format lacks, reads its red with Metal's constants (green and blue 0,
/// alpha 1) where the bound texture is sparse, through a swizzle each lane its source's; a texture
/// that is not sparse (the specialization 0) reads the image as it is. Elsewhere the stage states
/// nothing of the kind.
#[test]
fn an_access_that_does_not_read_its_residency_reads_metals_constant_lanes() {
    use TextureSwizzle as T;
    let mut legs = [0; 2];
    for read in [true, false] {
        let name = if read {
            "unobserved_read"
        } else {
            "unobserved_fetch"
        };
        let Some(bc) = assemble(name, &unobserved_kernel(read)) else {
            return;
        };
        for (stem, p) in committed_profiles().expect("profiles") {
            let what = format!("{stem} {name}");
            let a = ok(compile(&bc, "k", &p), &p, &what);
            let images = sparse_images(&p);
            let fact = RuntimeFact::ConstantLanes {
                slot: texture_slot(0),
            };
            assert_eq!(a.facts.contains(&fact), images, "{what}");
            assert!(
                !a.facts
                    .iter()
                    .any(|f| matches!(f, RuntimeFact::SparseTexture { .. })),
                "{what}"
            );
            assert!(
                !a.precision.contains(&PrecisionNote::SparseResidency),
                "{what}"
            );
            legs[usize::from(images)] += 1;
            if !images {
                continue;
            }
            let (plan, m) = typed_module(&bc, "k", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let slot = if read {
                ResourceSlot {
                    stage: Stage::Kernel,
                    kind: ResourceKind::Texture,
                    index: 0,
                }
            } else {
                ResourceSlot {
                    stage: Stage::Kernel,
                    kind: ResourceKind::CombinedTextureSampler(CombinedSampler {
                        sampler: SamplerRef::Constexpr([PIXEL_NEAREST, 0]),
                        compare: false,
                        integer: false,
                        unnormalized: false,
                    }),
                    index: 0,
                }
            };
            // Each leg: the swizzle; the device's value of texel `i` (`r` its red, `i + 1`); and
            // what Metal reads.
            type Texel = fn(u32) -> [u32; 4];
            let legs_of: [(TextureSwizzleChannels, Texel, Texel); 3] = [
                (
                    TextureSwizzleChannels::DEFAULT,
                    |r| {
                        if r == fl(2.0) {
                            [0; 4]
                        } else {
                            [r, 0xDEAD_BEEF, 7, 9]
                        }
                    },
                    |r| [if r == fl(2.0) { 0 } else { r }, 0, 0, fl(1.0)],
                ),
                (
                    TextureSwizzleChannels::DEFAULT,
                    |r| [r, 0, 0, 0],
                    |r| [r, 0, 0, fl(1.0)],
                ),
                (
                    TextureSwizzleChannels::new(T::Alpha, T::Red, T::One, T::Zero),
                    |r| [0xDEAD_BEEF, r, 5, 6],
                    |r| [fl(1.0), r, fl(1.0), 0],
                ),
            ];
            for (swizzle, device, want) in legs_of {
                let red = |i: usize| fl(i as f32 + 1.0);
                let leg = format!("{what} {swizzle:?}");
                let mut img = Image::new(slot, [2, 2, 1], 1, 1, 1);
                for (i, t) in img.texels.iter_mut().enumerate() {
                    *t = device(red(i));
                }
                let mut res = resources(&plan, &[vec![0; 16 * 4]]);
                res.spec
                    .insert(SpecKey::ConstantLanes(texture_slot(0)), r32_lanes(swizzle));
                let out = words(&run_both(&m, &res, &[img.clone()], [2, 2, 1])[0]);
                for i in 0..4usize {
                    assert_eq!(out[4 * i..4 * i + 4], want(red(i)), "{leg} texel {i}");
                }
                // Not sparse: the specialization 0, every texel the image's.
                let res = resources(&plan, &[vec![0; 16 * 4]]);
                let out = words(&run_both(&m, &res, &[img], [2, 2, 1])[0]);
                for i in 0..4usize {
                    assert_eq!(
                        out[4 * i..4 * i + 4],
                        device(red(i)),
                        "{leg} plain texel {i}"
                    );
                }
            }
        }
    }
    assert!(legs.iter().all(|n| *n > 0), "legs {legs:?}");
}
