//! Rasterization-rate maps end to end (Apple's layout): `rasterization_rate_map_decoder`'s `float2`
//! maps compile to Apple's GPU decoder of Metal's parameter data on every tested profile, pass L3,
//! and record no `RasterizationRateMap` note (the decoder is Apple's arithmetic over Metal's
//! bytes). Its `float` arithmetic is the device's: Metal's on a device measured to read `float`
//! denormals as Metal does (`DriverDeviation::MetalFloatDenormals`, MoltenVK's), the device's own
//! denormals elsewhere, noted `DeviceDenormals`, and its `NaN` the device's, noted
//! `NativeNanEncoding` where the profile is not measured to answer Metal's. On the CPU interpreter,
//! which keeps `float` denormals as lavapipe does, the module computes the shader decoder's
//! arithmetic over the same bytes with denormals kept ([`kept`]): bit for bit, the sign of a zero
//! included (a `NaN` is any `NaN`), in both directions, for every layer (a layer past the map's
//! reads the words where its tables would be), inside the screen, in the half pixel before a tile
//! boundary at a multiple of 32, and past both ends, for infinities and `NaN`, over compiled maps
//! (one of no layers) and over Apple M3's measured tiles. Where no denormal arises, that is what
//! `bind::RasterizationRateMap`'s shader decoders answer (`shader_screen_to_physical` and
//! `shader_physical_to_screen`, Metal's). Stray words (a map of garbage) decode without a fault.
//!
//! A far coordinate's saturating index reads before or past the buffer, which robust descriptors
//! contain on a device and the interpreter traps on, so with them only the near points run.
//!
//! The kernel is the shape Apple's compiler emits for
//! `out[tid] = float4(dec.map_screen_to_physical_coordinates(in[tid], layer),
//! dec.map_physical_to_screen_coordinates(in[tid], layer))` (fast math, denormals off).
//! `llvm-as` assembles it; without it the tests skip visibly. The `uint2` overloads are
//! `integer`'s.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::air::canon::rate_map_data as format;
use crate::bind::{RasterizationRateMap, RateMapAxis, RateMapDescriptor, RateMapLayerDescriptor};
use crate::device::profile::NanEncoding;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::lower::arith::{denormal, nan};
use crate::lower::select::Plan;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, typed_module, AssembleError,
};
use crate::pipeline::PrecisionNote;
use crate::spirv::interp::{run, Buffer, Config, Resources};

mod integer;

const PROFILES: [&str; 4] = [
    "moltenvk-m3",
    "lunarg-desktop-baseline-2023",
    "vk12-floor",
    "lavapipe",
];

const KERNEL: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @rate_map(ptr addrspace(2) noundef align 1 "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, ptr addrspace(2) noundef readonly align 4 captures(none) dereferenceable(4) "air-buffer-no-alias" %3, i32 noundef %4) local_unnamed_addr #0 {
  %6 = zext i32 %4 to i64
  %7 = getelementptr inbounds <2 x float>, ptr addrspace(1) %1, i64 %6
  %8 = load <2 x float>, ptr addrspace(1) %7, align 8
  %9 = load i32, ptr addrspace(2) %3, align 4
  %10 = tail call fast <2 x float> @air.map_screen_to_physical_coordinates.v2f32.p2i8.i32(<2 x float> %8, ptr addrspace(2) readonly captures(none) %0, i32 %9) #2
  %11 = tail call fast <2 x float> @air.map_physical_to_screen_coordinates.v2f32.p2i8.i32(<2 x float> %8, ptr addrspace(2) readonly captures(none) %0, i32 %9) #2
  %12 = shufflevector <2 x float> %10, <2 x float> %11, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %13 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %6
  store <4 x float> %12, ptr addrspace(1) %13, align 16
  ret void
}

declare <2 x float> @air.map_screen_to_physical_coordinates.v2f32.p2i8.i32(<2 x float>, ptr addrspace(2) readonly captures(none), i32) local_unnamed_addr #1
declare <2 x float> @air.map_physical_to_screen_coordinates.v2f32.p2i8.i32(<2 x float>, ptr addrspace(2) readonly captures(none), i32) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #2 = { nounwind willreturn memory(argmem: read) }

!air.kernel = !{!9}
!air.compile_options = !{!18, !19}
!air.version = !{!22}
!air.language_version = !{!23}

!9 = !{ptr @rate_map, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !16, !17}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"metal::rasterization_rate_map_data", !"air.arg_name", !"map"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"in"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!16 = !{i32 3, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"layer"}
!17 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!18 = !{!"air.compile.denorms_disable"}
!19 = !{!"air.compile.fast_math_enable"}
!22 = !{i32 2, i32 8, i32 0}
!23 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

fn assemble(text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-rate-map-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the kernel: {why}"),
    }
}

/// Committed profile `stem`, with buffer robustness or with the in-shader containment encoding.
fn profile_with(stem: &str, robust: bool) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    if !robust {
        desc.features.remove(&VkFeature::RobustBufferAccess);
        desc.features.remove(&VkFeature::RobustBufferAccess2);
    }
    Device::from_queries(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// A shader's map of `c` over parameter data `words` in layer `layer`, to physical coordinates
/// if `to_physical`, as the translated decoder computes it on a device that keeps `float`
/// denormals: `rate_map_data`'s shader decoder (the granule of `c` plus half a pixel, separate
/// roundings, the index saturating and unclamped, a read past the words 0) with IEEE `f32`
/// arithmetic, no operand or result flushed.
fn kept(words: &[u32], layer: u32, c: [f32; 2], to_physical: bool) -> [f32; 2] {
    let entry = |table: u32, j: u32| {
        let at = table.wrapping_add(j.wrapping_mul(format::ENTRY_WORDS));
        let word = |i: u32| words.get(i as usize).copied().unwrap_or(0);
        (word(at), word(at.wrapping_add(1)))
    };
    let axis = |axis: u32, x: f32| {
        let table = format::table(layer, axis, to_physical);
        if !to_physical {
            let k = format::shader_index(x * format::INV_GRANULARITY);
            let (reciprocal, offset) = entry(table, k);
            return (x - f32::from_bits(offset)) * f32::from_bits(reciprocal);
        }
        let j = format::shader_index((x + format::SHADER_GRANULE_OFFSET) * format::INV_GRANULARITY);
        let packed = entry(table, j).1;
        let (byte, boundary) = (packed & format::NO_BOUNDARY, packed & format::BOUNDARY_MASK);
        let into = (x - j as f32 * format::GRANULARITY as f32) * 2.0;
        let j = if byte != format::NO_BOUNDARY && into >= boundary as f32 {
            j.wrapping_add(1)
        } else {
            j
        };
        let (scale, packed) = entry(table, j);
        let offset = ((packed as i32) >> format::OFFSET_SHIFT) as f32 * format::OFFSET_UNIT;
        x * f32::from_bits(scale) + offset
    };
    [axis(0, c[0]), axis(1, c[1])]
}

fn bound(index: u32, data: &[u8]) -> Buffer {
    let mut bytes = data.to_vec();
    while !bytes.len().is_multiple_of(16) {
        bytes.push(0);
    }
    Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        }),
        address: None,
        bytes,
    }
}

/// Push bytes of zeros: every slot's bind residual and the dispatch region's origin are 0.
fn push_bytes(plan: &Plan) -> Vec<u8> {
    for f in &plan.push.fields {
        assert!(
            matches!(f.loc, PushLoc::Push { .. }),
            "{:?} in the argument table",
            f.field
        );
    }
    vec![0u8; 128]
}

fn floats(c: &[[f32; 2]]) -> Vec<u8> {
    c.iter()
        .flat_map(|p| p.iter().flat_map(|v| v.to_le_bytes()))
        .collect()
}

/// The points every map is decoded at: inside, on tile and screen edges, in the half pixel before
/// each multiple of 32 (where a shader reads the next tile's line if a tile starts there), past
/// both ends, and non-finite.
fn points(screen: [u32; 2]) -> Vec<[f32; 2]> {
    let [w, h] = screen.map(|v| v as f32);
    let mut out = vec![
        [0.0, 0.0],
        [-3.5, -0.25],
        [w, h],
        [w + 17.75, h * 0.5],
        [w * 0.5, -40.0],
        [f32::INFINITY, 2.0],
        [f32::NEG_INFINITY, f32::NAN],
        [1.0e30, -1.0e30],
    ];
    for i in 0..48u32 {
        let t = i as f32 / 47.0;
        out.push([w * t + 0.37, h * (1.0 - t) + 0.61]);
        out.push([(i * 32) as f32, (i * 16) as f32 + 0.5]);
        let edge = (i * 32) as f32;
        out.push([edge - 0.5, edge - 0.75]);
        out.push([edge - 0.25, (edge - 0.5).next_down()]);
    }
    out
}

/// The largest coordinate whose saturating index reads inside the buffer at every decoded layer
/// (the tables of layer `count + 1` end below its last word for indices up to 512).
const NEAR: f32 = 16_384.0;

/// Bit for bit, the sign of a zero included; a `NaN` is any `NaN` (the module leaves a `NaN`'s
/// bits to the device, which the interpreter does not model).
fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn maps() -> Vec<(&'static str, RasterizationRateMap)> {
    let layer = |h: &[f32], v: &[f32]| RateMapLayerDescriptor::new(h.to_vec(), v.to_vec());
    let compiled = |screen: [u32; 2], layers: Vec<RateMapLayerDescriptor>| {
        let mut d = RateMapDescriptor::new(screen);
        d.layers = layers;
        RasterizationRateMap::new(&d).expect("map")
    };
    let apple_x = RateMapAxis::from_tiles(
        32,
        &[
            77, 63, 55, 49, 45, 41, 39, 36, 34, 33, 32, 32, 32, 33, 34, 34, 35, 36, 37, 38, 39, 40,
            41, 43,
        ]
        .iter()
        .map(|&s| (s, 32))
        .chain([(22, 16)])
        .collect::<Vec<_>>(),
    )
    .expect("tiles");
    let apple_y = RateMapAxis::from_tiles(32, &[(64, 32), (36, 20)]).expect("tiles");
    vec![
        ("identity", compiled([64, 48], vec![layer(&[1.0], &[1.0])])),
        ("no layers", compiled([256, 256], Vec::new())),
        ("half", compiled([1024, 100], vec![layer(&[0.5], &[0.5])])),
        (
            "two layers",
            compiled(
                [1920, 1080],
                vec![
                    // At most 4 samples per axis: the counts measured on maps of two layers.
                    layer(&[0.1, 0.4, 1.0, 0.2], &[1.0, 0.3, 0.8, 1.0]),
                    layer(&[0.25, 1.0, 0.25], &[0.75]),
                ],
            ),
        ),
        (
            "Apple M3's tiles",
            RasterizationRateMap::from_layers([1000, 100], vec![[apple_x, apple_y]]).expect("map"),
        ),
    ]
}

/// Both maps of every layer, as the shader decoder's arithmetic computes them with the device's
/// denormals ([`kept`]), on every profile, with the notes of the device's arithmetic: no
/// `RasterizationRateMap`, `DeviceDenormals` where the profile does not read `float` denormals as
/// Metal does, and `NativeNanEncoding` where it is not measured to answer Metal's `NaN`. The
/// answers are `bind::RasterizationRateMap`'s shader decoders' (Metal's) at every point of a
/// map's own layers; only a layer past a map's, whose tables are other words read as `float`s,
/// some of them denormal, answers otherwise at some points.
#[test]
fn the_decoder_answers_as_the_map() {
    let Some(bc) = assemble(KERNEL) else {
        return;
    };
    let (mut metals, mut apart) = (0, 0);
    for stem in PROFILES {
        for robust in [true, false] {
            let what = format!("{stem} robust={robust}");
            let p = profile_with(stem, robust);
            let a = compile(&bc, "rate_map", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert!(
                !a.precision.contains(&PrecisionNote::RasterizationRateMap),
                "{what}: {:?}",
                a.precision
            );
            assert_eq!(
                a.precision.contains(&PrecisionNote::DeviceDenormals),
                !denormal::device_flushes(&p),
                "{what}: {:?}",
                a.precision
            );
            assert_eq!(
                a.precision.contains(&PrecisionNote::NativeNanEncoding),
                nan::encoding(&p) == NanEncoding::Native,
                "{what}: {:?}",
                a.precision
            );
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (plan, m) = typed_module(&bc, "rate_map", &p).expect("module");
            for (name, map) in maps() {
                let words: Vec<u32> = map
                    .to_bytes()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect();
                let pts = points(map.screen_size());
                // A far coordinate's saturating index reads before or past the buffer: robust
                // descriptors contain those reads on a device, which the interpreter does not
                // model (it traps there), so with them only the near points run; the in-shader
                // containment encoding reads 0 there, as the map's shader decoders do.
                let pts: Vec<[f32; 2]> = if robust {
                    pts.into_iter()
                        .filter(|p| !p.iter().any(|&v| v > NEAR))
                        .collect()
                } else {
                    pts
                };
                for layer in 0..map.layer_count() + 2 {
                    let res = Resources {
                        buffers: vec![
                            bound(0, &map.to_bytes()),
                            bound(1, &floats(&pts)),
                            bound(2, &vec![0u8; pts.len() * 16]),
                            bound(3, &layer.to_le_bytes()),
                        ],
                        push: push_bytes(&plan),
                        spec: Default::default(),
                    };
                    let cfg = Config {
                        workgroups: [pts.len() as u32, 1, 1],
                        ..Config::default()
                    };
                    let out = run(&m, &cfg, res).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                    for (i, &c) in pts.iter().enumerate() {
                        let got: Vec<f32> = out.buffers[2][i * 16..i * 16 + 16]
                            .chunks(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .collect();
                        let want =
                            [kept(&words, layer, c, true), kept(&words, layer, c, false)].concat();
                        assert!(
                            got.iter().zip(&want).all(|(&g, &w)| same(g, w)),
                            "{what}: {name} layer {layer} at {c:?}: {got:?}, kept {want:?}"
                        );
                        let metal = [
                            map.shader_screen_to_physical(c, layer),
                            map.shader_physical_to_screen(c, layer),
                        ]
                        .concat();
                        if want.iter().zip(&metal).all(|(&g, &w)| same(g, w)) {
                            metals += 1;
                        } else {
                            assert!(
                                layer >= map.layer_count(),
                                "{what}: {name} layer {layer} at {c:?}: {want:?}, Metal's {metal:?}"
                            );
                            apart += 1;
                        }
                    }
                }
            }
        }
    }
    // Both kinds of point were decoded, so neither comparison is vacuous.
    assert!(
        metals > 0 && apart > 0,
        "{metals} points Metal's, {apart} apart"
    );
}

/// A buffer of stray words (an unbound map's neutral zeros, or garbage) decodes to some value
/// without a fault under the in-shader containment encoding: every read past the buffer is
/// contained. With robust descriptors the device contains them, which the interpreter does not
/// model (it traps there, to catch reads a module relies on).
#[test]
fn stray_words_decode_without_a_fault() {
    let Some(bc) = assemble(KERNEL) else {
        return;
    };
    let mut seed = 0x9e37_79b9u32;
    let garbage: Vec<u8> = (0..64)
        .flat_map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed.to_le_bytes()
        })
        .collect();
    for stem in PROFILES {
        let what = format!("{stem} in-shader containment");
        let p = profile_with(stem, false);
        let (plan, m) = typed_module(&bc, "rate_map", &p).expect("module");
        let pts = points([64, 64]);
        for words in [vec![0u8; 16], garbage.clone()] {
            let res = Resources {
                buffers: vec![
                    bound(0, &words),
                    bound(1, &floats(&pts)),
                    bound(2, &vec![0u8; pts.len() * 16]),
                    bound(3, &3u32.to_le_bytes()),
                ],
                push: push_bytes(&plan),
                spec: Default::default(),
            };
            let cfg = Config {
                workgroups: [pts.len() as u32, 1, 1],
                ..Config::default()
            };
            run(&m, &cfg, res).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        }
    }
}

/// A fragment function that unwarps its position through the map (the corpus's fragment shape:
/// `map_screen_to_physical_coordinates(position.xy, layer)`, the layer a buffer word) compiles on
/// every profile, passes L3, and records no `RasterizationRateMap` note.
#[test]
fn a_fragment_function_maps_its_position() {
    let text = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <4 x float> @unwarp(<4 x float> noundef %0, ptr addrspace(2) noundef align 1 "air-buffer-no-alias" %1, ptr addrspace(2) noundef readonly align 4 captures(none) dereferenceable(4) "air-buffer-no-alias" %2) local_unnamed_addr #0 {
  %4 = shufflevector <4 x float> %0, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %5 = load i32, ptr addrspace(2) %2, align 4
  %6 = tail call fast <2 x float> @air.map_screen_to_physical_coordinates.v2f32.p2i8.i32(<2 x float> %4, ptr addrspace(2) readonly captures(none) %1, i32 %5) #1
  %7 = tail call fast <2 x float> @air.map_physical_to_screen_coordinates.v2f32.p2i8.i32(<2 x float> %6, ptr addrspace(2) readonly captures(none) %1, i32 %5) #1
  %8 = shufflevector <2 x float> %6, <2 x float> %7, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x float> %8
}

declare <2 x float> @air.map_screen_to_physical_coordinates.v2f32.p2i8.i32(<2 x float>, ptr addrspace(2) readonly captures(none), i32) local_unnamed_addr #1
declare <2 x float> @air.map_physical_to_screen_coordinates.v2f32.p2i8.i32(<2 x float>, ptr addrspace(2) readonly captures(none), i32) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #1 = { nounwind willreturn memory(argmem: read) }

!air.fragment = !{!9}
!air.compile_options = !{!16, !17}
!air.version = !{!22}
!air.language_version = !{!23}

!9 = !{ptr @unwarp, !10, !12}
!10 = !{!11}
!11 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!12 = !{!13, !14, !15}
!13 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!14 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"metal::rasterization_rate_map_data", !"air.arg_name", !"map"}
!15 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"layer"}
!16 = !{!"air.compile.denorms_disable"}
!17 = !{!"air.compile.fast_math_enable"}
!22 = !{i32 2, i32 8, i32 0}
!23 = !{!"Metal", i32 4, i32 0, i32 0}
"#;
    let Some(bc) = assemble(text) else {
        return;
    };
    for stem in PROFILES {
        for robust in [true, false] {
            let what = format!("{stem} robust={robust}");
            let p = profile_with(stem, robust);
            let a = compile(&bc, "unwarp", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert!(
                !a.precision.contains(&PrecisionNote::RasterizationRateMap),
                "{what}"
            );
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        }
    }
}
