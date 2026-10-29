//! Apple's extended-range formats read and written as Metal reads and writes them: under a slot's
//! `extended_range` variant the function declares the slot of its raw unsigned texels and decodes
//! and encodes Metal's values in the shader.
//!
//! - On the CPU interpreter, a read of every code of each of the four formats, through a sampled
//!   and a storage slot, with a `float` and a `half` result, is Metal's value bit for bit (the
//!   measured rule of `metal::format::ExtendedRange`), and writing it back stores Metal's raw
//!   texel: the same codes, the bits Metal does not read written as Metal writes them, and alpha
//!   clamped to its codes.
//! - A read outside the texture is Metal's zero texel, alpha the format's (measured), on a profile
//!   whose device robustness would return the raw zero.
//! - `ARGB8Unorm`'s storage slot rotates its bytes in the shader (`RotateArgb`), and a half-float
//!   storage slot leaves each written `float` to the device's conversion, noting where it is not
//!   measured to be Metal's.
//! - The binder joins a slot bound in an extended-range format to the plan's variant, the
//!   variant's plan states the same key, and it views the image in the raw format.
//! - Samples of such a texture are Metal's (`samples`), at Metal's level and through a linear
//!   mip filter's trilinear sum (`levels`), and a 3D texture's (`volumes`).
//! - A two-plane 3D texture reads its first texel, writes its first slice and answers Metal's
//!   depth and level count (`planes_volumes`).
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::{BindState, BoundBuffer, BoundTexture, StageTable, TextureFacts, WriteSource};
use crate::budget::{Budget, SPIRV_WORDS};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::Feature;
use crate::library::{FunctionConstantValues, Library};
use crate::lower::select::{LowerInputs, Plan};
use crate::metal::format::ExtendedRange;
use crate::metal::{self, PixelFormat, TextureType};
use crate::pipeline::compile::{compile_bitcode, plan_bitcode, EntryRef};
use crate::pipeline::interface::PushLoc;
use crate::pipeline::lower_inputs::{StorageImageChoice, StorageImageClass};
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef};
use crate::pipeline::sym::{SpecKey, Stage};
use crate::pipeline::testing::{
    link_and_validate, llvm_assemble, meter, profile, typed_module_with, AssembleError,
};
use crate::pipeline::StageSource;
use crate::pipeline::{ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::TypedModule;

mod bicubic;
mod heap;
mod levels;
mod planes_volumes;
mod samples;
mod targets;
mod volumes;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// The texture's side: 32 x 32 texels, one per code.
const SIDE: u32 = 32;

const SOURCE: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Texture,
    index: 0,
};

const TARGET: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Texture,
    index: 1,
};

/// Every extended-range form: the one-plane formats and the two-plane ones, and their `_sRGB`
/// twins.
const FORMS: [ExtendedRange; 6] = [
    ExtendedRange::Bgr10 { srgb: false },
    ExtendedRange::Bgr10 { srgb: true },
    ExtendedRange::Bgra10 { srgb: false },
    ExtendedRange::Bgra10 { srgb: true },
    ExtendedRange::Planes { srgb: false },
    ExtendedRange::Planes { srgb: true },
];

/// Bitcode of the module `body` (after the AIR header), or `None` without `llvm-as`.
fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-extended-range-{}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed),
        name.replace(' ', "-")
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, format!("{HEADER}{body}")).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(x) => Some(x),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("{name}: {why}"),
    }
}

/// How the kernel reaches slot 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    /// `texture2d<T, access::read>`: a sampled image.
    Read,
    /// `texture2d<T, access::read_write>`: a storage image.
    ReadWrite,
}

/// A kernel reading slot 0 (`half` texels, or `float`) at its thread's texel `gid` moved `dx`
/// texels right (a 32 x 32 grid), storing the texel to `out[gid.x + 32 gid.y]` (buffer 0) and
/// writing it to `texture2d<T, access::write>` slot 1 at `gid`.
fn kernel(source: Source, half: bool, dx: u32) -> String {
    let (texel, vec4, size, out_ty, suffix) = if half {
        ("half", "<4 x half>", 8, "half4", "v4f16")
    } else {
        ("float", "<4 x float>", 16, "float4", "v4f32")
    };
    let (access, code) = match source {
        Source::Read => ("read", 1),
        Source::ReadWrite => ("read_write", 3),
    };
    format!(
        r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %w, ptr addrspace(1) %out, <2 x i32> %gid) {{
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %row = shl i32 %y, 5
  %n = add i32 %x, %row
  %i = zext i32 %n to i64
  %rx = add i32 %x, {dx}
  %at = insertelement <2 x i32> %gid, i32 %rx, i64 0
  %rs = call ptr addrspace(2) @air.get_read_sampler()
  %r = call {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %at, <2 x i32> zeroinitializer, i32 0, i32 {code})
  %v = extractvalue {{ {vec4}, i8 }} %r, 0
  call void @air.write_texture_2d.{suffix}(ptr addrspace(1) %w, <2 x i32> %gid, {vec4} %v, i32 0, i32 2)
  %q = getelementptr inbounds {vec4}, ptr addrspace(1) %out, i64 %i
  store {vec4} %v, ptr addrspace(1) %q, align {size}
  ret void
}}

declare ptr addrspace(2) @air.get_read_sampler()
declare {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare void @air.write_texture_2d.{suffix}(ptr addrspace(1), <2 x i32>, {vec4}, i32, i32)

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.{access}", !"air.arg_type_name", !"texture2d<{texel}, {access}>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<{texel}, write>", !"air.arg_name", !"w"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{out_ty}", !"air.arg_name", !"out"}}
!6 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}}
"#
    )
}

/// Lowering inputs with slots 0 and 1 bound in `form`: each storage slot's choice is the raw
/// format the binder views it in.
fn inputs(form: ExtendedRange, source: Source) -> LowerInputs {
    let mut i = LowerInputs::new();
    i.extended_range.insert(SOURCE, form);
    i.extended_range.insert(TARGET, form);
    let raw = StorageImageChoice::new(form.raw_format(), StorageImageClass::Identity);
    if source == Source::ReadWrite {
        i.storage_images.insert(SOURCE, raw);
    }
    i.storage_images.insert(TARGET, raw);
    i
}

/// The profiles the extended-range tests compile for: the site code's (lavapipe, MoltenVK) and a
/// routine's (the RTX 5080 Laptop GPU, whose driver keeps a `DontInline` call a call).
fn profiles() -> Vec<(&'static str, crate::device::profile::DeviceProfile)> {
    ["lavapipe", "moltenvk-m3", "nvidia-rtx5080-laptop"]
        .into_iter()
        .map(|stem| {
            (
                stem,
                profile(stem, DevicePolicy::default()).expect("profile"),
            )
        })
        .collect()
}

/// Compile the kernel `bc` under `inputs` on `p`, check that the plan's word estimate bounds the
/// emission, and validate the linked module.
fn validate(
    bc: &[u8],
    inputs: &LowerInputs,
    p: &crate::device::profile::DeviceProfile,
    what: &str,
) {
    validate_entry(bc, "k", inputs, p, what);
}

/// [`validate`] of the kernel `entry` of `bc`: its artifact.
fn validate_entry(
    bc: &[u8],
    entry: &str,
    inputs: &LowerInputs,
    p: &crate::device::profile::DeviceProfile,
    what: &str,
) -> std::sync::Arc<crate::library::FunctionArtifact> {
    let consts = FunctionConstantValues::defaults();
    let entry = EntryRef::Name(entry);
    let plan = plan_bitcode(bc, entry, &consts, p, inputs, &mut meter())
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let a = compile_bitcode(bc, entry, &consts, p, inputs, &mut meter(), SPIRV_WORDS)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let words = a.module.words.len() as u64;
    assert!(
        plan.estimate.words >= words,
        "{what}: estimate {} < emitted {words}",
        plan.estimate.words
    );
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: {e}"));
    a
}

// ---- The reference ----------------------------------------------------------------------------

/// The red, green, blue, and alpha codes of texel `n` (0 to 1023): every code in each channel.
fn codes(n: u32) -> [u32; 4] {
    [(n * 7 + 3) % 1024, 1023 - n, n, (n * 13 + 5) % 1024]
}

/// Texel `n`'s codes in `form`: [`codes`], a two-plane texel's alpha its byte (every byte).
fn codes_in(form: ExtendedRange, n: u32) -> [u32; 4] {
    let [r, g, b, a] = codes(n);
    match form {
        ExtendedRange::Planes { .. } => [r, g, b, a % 256],
        _ => [r, g, b, a],
    }
}

/// The lanes of `form`'s raw texel a write stores (the raw format's components).
fn raw_lanes(form: ExtendedRange) -> usize {
    match form {
        ExtendedRange::Bgr10 { .. } => 1,
        ExtendedRange::Planes { .. } => 2,
        _ => 4,
    }
}

/// Texel `n`'s raw texel in `form`, with bits Metal does not read set: `BGR10_XR`'s top two
/// bits, and `BGRA10_XR`'s low six of each field.
fn raw_texel(form: ExtendedRange, n: u32) -> [u32; 4] {
    let [r, g, b, a] = codes_in(form, n);
    match form {
        ExtendedRange::Bgr10 { .. } => [(n % 4) << 30 | r << 20 | g << 10 | b, 0, 0, 1],
        // The word's bits 30 and 31 set, and the byte's component past its low eight bits.
        ExtendedRange::Planes { .. } => [
            (n % 4) << 30 | r << 20 | g << 10 | b,
            (n % 3) << 8 | a,
            0,
            1,
        ],
        _ => {
            let junk = n % 64;
            [
                b << 6 | junk,
                g << 6 | (63 - junk),
                r << 6 | junk,
                a << 6 | 42,
            ]
        }
    }
}

/// Texel `n`'s raw texel as Metal writes back the value it reads.
fn written(form: ExtendedRange, n: u32) -> [u32; 4] {
    let [r, g, b, a] = codes_in(form, n);
    match form {
        ExtendedRange::Bgr10 { .. } => [0xC000_0000 | r << 20 | g << 10 | b, 0, 0, 1],
        ExtendedRange::Planes { .. } => [r << 20 | g << 10 | b, a, 0, 1],
        _ => [b << 6, g << 6, r << 6, a.clamp(384, 894) << 6],
    }
}

/// The device's value of code `c` of channel `channel` of linear `form` (red 0, alpha 3): the code
/// less 384 times the `f32` nearest `1 / 510` (`extended_range::decode`), a two-plane texel's
/// alpha byte times that of `1 / 255`. An `_sRGB` form's is the device's `Pow`'s, which the
/// interpreter does not model.
fn value(form: ExtendedRange, channel: usize, c: u32) -> f32 {
    assert!(!form.srgb(), "{form:?}: the interpreter models no Pow");
    if channel == 3 {
        return match form {
            ExtendedRange::Bgr10 { .. } => 1.0,
            // The alpha byte, an 8-bit unorm code.
            ExtendedRange::Planes { .. } => c as f32 * (1.0 / 255.0f32),
            _ => (c.clamp(384, 894) - 384) as f32 * (1.0 / 510.0f32),
        };
    }
    (c as i32 - 384) as f32 * (1.0 / 510.0f32)
}

/// `x` narrowed to `half`, nearest even, the device's narrowing of an extended-range read
/// (`extended_range::decode`; every value here is a normal `half` or zero).
fn half_bits(x: f32) -> u16 {
    let h = crate::float16::f32_to_f16_bits(x);
    assert!(h & 0x7c00 != 0 || h & 0x7fff == 0, "{x} is no normal half");
    h
}

/// The image of slot `index` of `kind`'s texels.
fn image(index: u32, texels: impl Fn(u32) -> [u32; 4]) -> Image {
    let slot = ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Texture,
        index,
    };
    let mut img = Image::new(slot, [SIDE, SIDE, 1], 1, 1, 1);
    for n in 0..SIDE * SIDE {
        let at = img.index(0, 0, 0, [n % SIDE, n / SIDE, 0]).expect("texel");
        img.texels[at] = texels(n);
    }
    img
}

/// Resources of a dispatch of one-thread threadgroups over the grid, `out` sized for `bytes` per
/// texel, with `spec`'s specializations.
fn resources(plan: &Plan, bytes: usize, spec: &[(SpecKey, u64)]) -> Resources {
    let mut map = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        map.insert(SpecKey::LocalSize(axis), 1);
    }
    map.insert(SpecKey::FlatLocalSize, 1);
    map.extend(spec.iter().copied());
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            // A bound value read as data, which the interpreter writes from `spec`.
            (PushField::Specialized(_), PushLoc::Push { .. }) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let out = Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index: 0,
        }),
        address: None,
        bytes: vec![0xab; bytes * (SIDE * SIDE) as usize],
    };
    Resources {
        buffers: vec![out],
        push,
        spec: map,
    }
}

/// `m` over the grid with `images` bound, in both scheduling orders, which must agree: `out`
/// and the images.
fn run(m: &TypedModule, res: &Resources, images: &[Image]) -> (Vec<u8>, Vec<Image>) {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [SIDE, SIDE, 1],
            order,
            ..Config::default()
        };
        let (o, imgs) = run_with_images(m, &cfg, res.clone(), images.to_vec())
            .unwrap_or_else(|e| panic!("interpreter: {e:?}"));
        outs.push((o.buffers.into_iter().next().expect("out"), imgs));
    }
    let second = outs.pop().expect("second");
    let first = outs.pop().expect("first");
    assert_eq!(first.0, second.0, "the scheduling orders disagree");
    first
}

/// Lane `c` of texel `n` of `out` (`half` or `float` lanes) as bits.
fn lane(out: &[u8], half: bool, n: u32, c: usize) -> u32 {
    if half {
        let i = n as usize * 8 + 2 * c;
        u32::from(u16::from_le_bytes([out[i], out[i + 1]]))
    } else {
        let i = n as usize * 16 + 4 * c;
        u32::from_le_bytes([out[i], out[i + 1], out[i + 2], out[i + 3]])
    }
}

/// The texels of the image of texture slot `index` among `images`.
fn texels_of(images: &[Image], index: u32) -> &[[u32; 4]] {
    &images
        .iter()
        .find(|i| i.slot.kind == ResourceKind::Texture && i.slot.index == index)
        .expect("image")
        .texels
}

/// Every code of each linear format, read through a sampled and a storage slot with a `half` and a
/// `float` result, is the device's product (`value`), narrowed to `half` to nearest even; written
/// back, it is Metal's raw texel. An `_sRGB` form's transfer takes the device's `Pow`, which the
/// interpreter does not model: its values are the device lanes'.
#[test]
fn reads_and_writes_of_every_code_are_metals() {
    for source in [Source::Read, Source::ReadWrite] {
        for half in [false, true] {
            let what = format!("{source:?} {}", if half { "half" } else { "float" });
            let Some(bc) = assemble(&what, &kernel(source, half, 0)) else {
                return;
            };
            for (stem, p) in profiles() {
                for form in FORMS.into_iter().filter(|f| !f.srgb()) {
                    let what = format!("{what} {form:?} on {stem}");
                    let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(form, source))
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    validate(&bc, &inputs(form, source), &p, &what);
                    let bytes = if half { 8 } else { 16 };
                    let images = [image(0, |n| raw_texel(form, n)), image(1, |_| [0; 4])];
                    let (out, images) = run(&m, &resources(&plan, bytes, &[]), &images);
                    for n in 0..SIDE * SIDE {
                        for (c, code) in codes_in(form, n).into_iter().enumerate() {
                            let v = value(form, c, code);
                            let want = match half {
                                true => u32::from(half_bits(v)),
                                false => v.to_bits(),
                            };
                            let got = lane(&out, half, n, c);
                            assert_eq!(
                                got, want,
                                "{what}: texel {n} channel {c} code {code}: {got:#x} vs {want:#x}"
                            );
                        }
                    }
                    let target = texels_of(&images, 1);
                    let layout = image(1, |_| [0; 4]);
                    for n in 0..SIDE * SIDE {
                        let at = layout
                            .index(0, 0, 0, [n % SIDE, n / SIDE, 0])
                            .expect("texel");
                        let lanes = raw_lanes(form);
                        assert_eq!(
                            target[at][..lanes],
                            written(form, n)[..lanes],
                            "{what}: texel {n} written"
                        );
                    }
                }
            }
        }
    }
}

/// A read outside the texture is Metal's zero texel with the format's alpha (measured on the M3
/// Max): `BGR10_XR` reads alpha 1, which the slot's `OutOfBoundsAlpha`
/// specialization states, and `BGRA10_XR` 0, never the decode of the raw zero a device's
/// robustness returns (`-0.752941`).
#[test]
fn reads_outside_the_texture_are_metals_zero() {
    let Some(bc) = assemble("outside", &kernel(Source::Read, false, SIDE)) else {
        return;
    };
    // An `_sRGB` form's decode takes the device's `Pow`, which the interpreter does not model.
    for (stem, p) in profiles() {
        for form in FORMS.into_iter().filter(|f| !f.srgb()) {
            let what = format!("{form:?} on {stem}");
            let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(form, Source::Read))
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let alpha = match form {
                ExtendedRange::Bgr10 { .. } => 1,
                _ => 0,
            };
            let spec = [(SpecKey::OutOfBoundsAlpha(SOURCE), alpha)];
            let images = [image(0, |n| raw_texel(form, n)), image(1, |_| [0; 4])];
            let (out, _) = run(&m, &resources(&plan, 16, &spec), &images);
            let one = 1.0f32.to_bits();
            for n in 0..SIDE * SIDE {
                let got: Vec<u32> = (0..4).map(|c| lane(&out, false, n, c)).collect();
                let want = [0, 0, 0, if alpha == 1 { one } else { 0 }];
                assert_eq!(got, want, "{what}: texel {n}");
            }
        }
    }
}

/// Texel `n`'s bfloat codes, channel `c` of each: spread over every code, NaNs, infinities and
/// denormals among them.
fn bfloat_codes(n: u32) -> [u32; 4] {
    std::array::from_fn(|c| (n * 64 + 16411 * c as u32 + n / 16) & 0xFFFF)
}

/// Every texel of each bfloat form, read through a sampled and a storage slot with a `half` and
/// a `float` result, is its code shifted up 16 bits (a NaN code a NaN, as the device computes
/// it, where Metal reads `0x7fc00000`), a `half` result the device's narrowing of it (to nearest
/// even, a result below the smallest normal `half` its zero where the device's quantize flushes
/// it), the channels a form lacks 0 and alpha 1 whatever the image's other components hold;
/// written back, it is that value's top 16 bits (a NaN a NaN code). A read outside the texture
/// is Metal's zero texel, alpha 1 where the form has none.
#[test]
fn bfloat_reads_and_writes_are_metals() {
    let forms = [
        (ExtendedRange::R16Bfloat, 1usize),
        (ExtendedRange::Rg16Bfloat, 2),
        (ExtendedRange::Rgba16Bfloat, 4),
    ];
    let read = |c: u32| c << 16;
    let nan32 = |b: u32| b & 0x7FFF_FFFF > 0x7F80_0000;
    let nan16 = |h: u32| h & 0x7FFF > 0x7C00;
    let bfloat_nan = |c: u32| c & 0x7FFF > 0x7F80;
    let one = 1.0f32.to_bits();
    for source in [Source::Read, Source::ReadWrite] {
        for half in [false, true] {
            let what = format!("bfloat {source:?} {}", if half { "half" } else { "float" });
            let Some(bc) = assemble(&what, &kernel(source, half, 0)) else {
                return;
            };
            for (stem, p) in profiles() {
                for (form, ch) in forms {
                    let what = format!("{what} {form:?} on {stem}");
                    let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(form, source))
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    validate(&bc, &inputs(form, source), &p, &what);
                    let bytes = if half { 8 } else { 16 };
                    let raw = |n: u32| {
                        let codes = bfloat_codes(n);
                        std::array::from_fn(|c| match c < ch {
                            true => codes[c],
                            false => 0x1234_5678 ^ n,
                        })
                    };
                    let images = [image(0, raw), image(1, |_| [0; 4])];
                    let (out, images) = run(&m, &resources(&plan, bytes, &[]), &images);
                    let target = texels_of(&images, 1);
                    let layout = image(1, |_| [0; 4]);
                    for n in 0..SIDE * SIDE {
                        let codes = bfloat_codes(n);
                        let at = layout
                            .index(0, 0, 0, [n % SIDE, n / SIDE, 0])
                            .expect("texel");
                        for c in 0..4 {
                            let v = match (c < ch, c) {
                                (true, _) => read(codes[c]),
                                (false, 3) => one,
                                (false, _) => 0,
                            };
                            let h = crate::float16::f32_to_f16_bits(f32::from_bits(v));
                            let got = lane(&out, half, n, c);
                            let held = match (half, nan32(v)) {
                                (_, true) => {
                                    let nan = if half { nan16(got) } else { nan32(got) };
                                    assert!(nan, "{what}: texel {n} channel {c}: {got:#x}");
                                    if c < ch {
                                        let w = target[at][c];
                                        assert!(bfloat_nan(w), "{what}: texel {n} {c}: {w:#x}");
                                    }
                                    continue;
                                }
                                (true, false) => {
                                    // The device's quantize may flush a `half` denormal.
                                    let flushed = u32::from(h & 0x8000);
                                    let ok =
                                        got == u32::from(h) || (h & 0x7C00 == 0 && got == flushed);
                                    assert!(ok, "{what}: texel {n} channel {c}: {got:#x}");
                                    crate::float16::f16_bits_to_f32(got as u16).to_bits()
                                }
                                (false, false) => {
                                    assert_eq!(got, v, "{what}: texel {n} channel {c}");
                                    v
                                }
                            };
                            if c < ch {
                                assert_eq!(
                                    target[at][c],
                                    held >> 16,
                                    "{what}: texel {n} channel {c} written"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    let Some(bc) = assemble("bfloat outside", &kernel(Source::Read, false, SIDE)) else {
        return;
    };
    for (stem, p) in profiles() {
        for (form, ch) in forms {
            let what = format!("{form:?} outside on {stem}");
            let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(form, Source::Read))
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let alpha = u64::from(ch < 4);
            let spec = [(SpecKey::OutOfBoundsAlpha(SOURCE), alpha)];
            let images = [image(0, bfloat_codes), image(1, |_| [0; 4])];
            let (out, _) = run(&m, &resources(&plan, 16, &spec), &images);
            for n in 0..SIDE * SIDE {
                let got: Vec<u32> = (0..4).map(|c| lane(&out, false, n, c)).collect();
                let want = [0, 0, 0, if alpha == 1 { one } else { 0 }];
                assert_eq!(got, want, "{what}: texel {n}");
            }
        }
    }
}

/// `ARGB8Unorm`'s storage slot holds its bytes in an `R8G8B8A8_UNORM` image, alpha in red: a
/// read rotates them into Metal's channels, and a write back rotates them into the image's lanes
/// as computed, which the device's conversion stores. The held lanes run from below 0 to above 1.
#[test]
fn argb8_storage_slots_rotate_their_bytes() {
    let Some(bc) = assemble("argb8", &kernel(Source::ReadWrite, false, 0)) else {
        return;
    };
    let rotate = StorageImageChoice::new(
        crate::vk::Format::R8G8B8A8_UNORM,
        StorageImageClass::RotateArgb,
    );
    let mut i = LowerInputs::new();
    i.storage_images.insert(SOURCE, rotate);
    i.storage_images.insert(TARGET, rotate);
    // The image's lanes as the view reads them (the interpreter holds `f32` bits).
    let value = |k: u32| k as f32 / 3600.0 - 0.1;
    let held = |n: u32| -> [u32; 4] { [0, 1, 2, 3].map(|c| value(4 * n + c).to_bits()) };
    for (stem, p) in profiles() {
        let (plan, m) =
            typed_module_with(&bc, "k", &p, &i).unwrap_or_else(|e| panic!("{stem}: {e}"));
        validate(&bc, &i, &p, stem);
        let images = [image(0, held), image(1, |_| [0; 4])];
        let (out, images) = run(&m, &resources(&plan, 16, &[]), &images);
        let target = texels_of(&images, 1);
        let layout = image(1, |_| [0; 4]);
        for n in 0..SIDE * SIDE {
            let [a, r, g, b] = held(n);
            let got: Vec<u32> = (0..4).map(|c| lane(&out, false, n, c)).collect();
            assert_eq!(got, [r, g, b, a], "{stem}: texel {n} read");
            let at = layout
                .index(0, 0, 0, [n % SIDE, n / SIDE, 0])
                .expect("texel");
            assert_eq!(target[at], held(n), "{stem}: texel {n} written");
        }
    }
}

/// A half-float storage slot (formatless, and declared `Rgba16f`) writes each `float` channel as
/// computed, the device's conversion rounding it and storing its NaN. The rounding is Metal's
/// toward zero where the profile takes `DriverDeviation::MetalHalfStorageWrites` (the module
/// requires its property), else the device's own (`PrecisionNote::DeviceColorConversion`); the
/// NaN is Metal's `0x7e00` where it takes `DriverDeviation::MetalHalfStorageNans` (MoltenVK's: the
/// module requires its property), else the device's (`PrecisionNote::DeviceNanEncoding`); a
/// formatless slot's notes hold under the format bound there (`tests::engine_half_storage_notes`).
/// A formatless slot may also reach a packed small float image, whose NaN is the device's on every
/// profile (noted too), or a normalized one, whose conversion is the device's on every profile
/// (noted too). The held lanes cover every tie between neighbouring halves and its
/// neighbours, past the largest, the infinities and NaNs.
#[test]
fn half_float_storage_slots_leave_their_writes_to_the_device() {
    use crate::device::requirements::VkProperty;
    use crate::error::DriverDeviation;
    use crate::pipeline::artifact::PrecisionNote;
    let Some(bc) = assemble("halves", &kernel(Source::ReadWrite, false, 0)) else {
        return;
    };
    // Lane `c` of texel `n`: a tie between neighbouring halves (or its neighbour), or a special.
    let value = |n: u32, c: u32| -> u32 {
        let k = 4 * n + c;
        let specials = [
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0xffc0_0001,
            0x477f_f000,
            0x477f_efff,
            0x4780_0000,
            0xc77f_f001,
        ];
        if (k as usize) < specials.len() {
            return specials[k as usize];
        }
        let h = ((k * 37) & 0x7bfe) as u16;
        let mid = (f64::from(half_value(h)) + f64::from(half_value(h + 1))) / 2.0;
        let sign = if k & 1 == 1 { 0x8000_0000 } else { 0 };
        let tie = (mid as f32).to_bits();
        sign | match k % 3 {
            0 => tie - 1,
            1 => tie,
            _ => tie + 1,
        }
    };
    let held = move |n: u32| -> [u32; 4] { [0, 1, 2, 3].map(|c| value(n, c)) };
    let class = StorageImageClass::Identity;
    let mut nans_seen = std::collections::BTreeSet::new();
    for choice in [
        StorageImageChoice {
            numeric: Some(crate::pipeline::sym::SampledType::Float),
            ..StorageImageChoice::new(crate::vk::Format::UNDEFINED, class)
        },
        StorageImageChoice::new(crate::vk::Format::R16G16B16A16_SFLOAT, class),
    ] {
        let mut i = LowerInputs::new();
        i.storage_images.insert(TARGET, choice);
        let anv = "intel-irisxe-rplp-anv";
        let anv = (anv, profile(anv, DevicePolicy::default()).expect("profile"));
        for (stem, p) in profiles().into_iter().chain([anv]) {
            // The kernel reads both textures: without formatless reads, the target's choice
            // declares its format and the source's `RGBA32Float`'s.
            let mut i = i.clone();
            if !p.fields().images.storage_read_without_format {
                if choice.vk_format == crate::vk::Format::UNDEFINED {
                    continue;
                }
                let held = crate::vk::Format::R32G32B32A32_SFLOAT;
                let source = StorageImageChoice::new(held, StorageImageClass::Identity);
                i.storage_images.insert(SOURCE, source);
            }
            let what = format!("{stem} {:?}", choice.vk_format);
            let (plan, m) =
                typed_module_with(&bc, "k", &p, &i).unwrap_or_else(|e| panic!("{what}: {e}"));
            validate(&bc, &i, &p, &what);
            let images = [image(0, held), image(1, |_| [0; 4])];
            let (_, images) = run(&m, &resources(&plan, 16, &[]), &images);
            let target = texels_of(&images, 1);
            let layout = image(1, |_| [0; 4]);
            for n in 0..SIDE * SIDE {
                let at = layout
                    .index(0, 0, 0, [n % SIDE, n / SIDE, 0])
                    .expect("texel");
                assert_eq!(target[at], held(n), "{what}: texel {n}");
            }
            let a = compile_bitcode(
                &bc,
                EntryRef::Name("k"),
                &FunctionConstantValues::defaults(),
                &p,
                &i,
                &mut meter(),
                SPIRV_WORDS,
            )
            .unwrap_or_else(|e| panic!("{what}: {e}"));
            let rounds = p.takes_driver_deviation(DriverDeviation::MetalHalfStorageWrites);
            let nans = p.takes_driver_deviation(DriverDeviation::MetalHalfStorageNans);
            nans_seen.insert(nans);
            let relied = &a.requirements.properties;
            assert_eq!(
                relied.contains(&VkProperty::MetalHalfStorageWrites),
                rounds,
                "{what}"
            );
            let formatless = choice.vk_format == crate::vk::Format::UNDEFINED;
            let converted = a.precision.contains(&PrecisionNote::DeviceColorConversion);
            assert!(converted || rounds, "{what}: {:?}", a.precision);
            if !formatless {
                assert_eq!(converted, !rounds, "{what}: {:?}", a.precision);
            }
            assert_eq!(
                relied.contains(&VkProperty::MetalHalfStorageNans),
                nans,
                "{what}"
            );
            assert_eq!(
                a.precision.contains(&PrecisionNote::DeviceNanEncoding),
                !nans || formatless,
                "{what}: {:?}",
                a.precision
            );
        }
    }
    assert_eq!(nans_seen.len(), 2, "both NaN facts: {nans_seen:?}");
}

/// The value of a `half`'s bits (not a NaN).
fn half_value(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f64::from(bits & 0x3ff);
    let magnitude = match exponent {
        0 => fraction * 2f64.powi(-24),
        0x1f => f64::INFINITY,
        _ => (1.0 + fraction / 1024.0) * 2f64.powi(exponent - 15),
    };
    (sign * magnitude) as f32
}

fn devices() -> Vec<(String, Device)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles");
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dir).expect("profiles") {
        let path = e.expect("entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let text = std::fs::read_to_string(&path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        out.push((
            stem,
            Device::new(desc, DevicePolicy::default()).expect("device"),
        ));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// The kernel compiled into a compute pipeline under `inputs`, or `None` where the device needs a
/// storage slot's format as an input (it accesses no storage image without one).
fn pipeline(dev: &Device, bc: &[u8], inputs: &LowerInputs) -> Option<ComputePipeline> {
    pipeline_entry(dev, bc, "k", inputs)
}

/// [`pipeline`] of the kernel `entry` of `bc`.
fn pipeline_entry(
    dev: &Device,
    bc: &[u8],
    entry: &str,
    inputs: &LowerInputs,
) -> Option<ComputePipeline> {
    let lib = Library::from_air(bc).expect("library");
    let f = lib.function(entry).expect("function");
    let compiled =
        match dev.compile_function(&FunctionDescriptor::new(f), inputs, &Budget::standard()) {
            Ok(c) => c,
            Err(crate::error::Error::NeedsInput(_)) => return None,
            Err(e) => panic!("compile: {e}"),
        };
    Some(
        dev.compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
            &Budget::standard(),
        )
        .expect("pipeline"),
    )
}

/// A kernel's state: texture slots 0 and 1 bound in `pf` (shader read and write), buffer 0.
fn state(pf: PixelFormat) -> BindState {
    let mut t = StageTable::default();
    for index in [0, 1] {
        let mut facts = TextureFacts::new(TextureType::Type2D, pf);
        facts.usage = metal::TextureUsage::SHADER_READ | metal::TextureUsage::SHADER_WRITE;
        t.textures.insert(
            index,
            BoundTexture::Texture {
                facts,
                resource_id: None,
            },
        );
    }
    t.buffers.insert(
        0,
        BoundBuffer::Buffer {
            length: 65536,
            offset: 0,
            gpu_address: 0x1_0000,
            device_address: None,
            attribute_stride: None,
        },
    );
    let mut s = BindState::default();
    s.stages.insert(Stage::Kernel, t);
    s
}

/// A slot bound in an extended-range format joins the plan's variant, the variant's pipeline
/// plans the same key for the same shape, and both slots view the image in the format's raw
/// unsigned format; a slot bound in another format leaves the variant. On every committed
/// profile, for a sampled and a storage source.
#[test]
fn a_slot_bound_in_an_extended_range_format_joins_the_variant() {
    let formats = [
        (PixelFormat::BGR10XR, ExtendedRange::Bgr10 { srgb: false }),
        (
            PixelFormat::BGR10XRSRGB,
            ExtendedRange::Bgr10 { srgb: true },
        ),
        (PixelFormat::BGRA10XR, ExtendedRange::Bgra10 { srgb: false }),
        (
            PixelFormat::BGRA10XRSRGB,
            ExtendedRange::Bgra10 { srgb: true },
        ),
        (
            PixelFormat::RGB10A82PXR10,
            ExtendedRange::Planes { srgb: false },
        ),
        (
            PixelFormat::RGB10A82PXR10SRGB,
            ExtendedRange::Planes { srgb: true },
        ),
    ];
    let mut planned = 0;
    for source in [Source::Read, Source::ReadWrite] {
        let Some(bc) = assemble("bind", &kernel(source, false, 0)) else {
            return;
        };
        for (stem, dev) in devices() {
            let Some(plain) = pipeline(&dev, &bc, &LowerInputs::new()) else {
                continue;
            };
            for (pf, form) in formats {
                let what = format!("{source:?} {pf:?} on {stem}");
                let shape = state(pf).shape();
                let plan = match plain.plan(&dev, &shape) {
                    Ok(plan) => plan,
                    // A profile without the raw format's storage refuses it for the format.
                    Err(crate::error::BindError::Unsupported(u))
                        if u.feature == Feature::Format =>
                    {
                        continue
                    }
                    Err(e) => panic!("{what}: {e}"),
                };
                assert_eq!(
                    plan.variant.extended_range,
                    BTreeMap::from([(SOURCE, form), (TARGET, form)]),
                    "{what}"
                );
                let key = plan.variant.lowering_inputs(Stage::Kernel);
                assert_eq!(key.extended_range, plan.variant.extended_range, "{what}");
                let variant = pipeline(&dev, &bc, &key).expect("the variant's inputs");
                let again = variant.plan(&dev, &shape).expect("variant plan");
                assert_eq!(again.variant, plan.variant, "{what}");
                let views: Vec<_> = again
                    .descriptors
                    .iter()
                    .filter_map(|d| match &d.source {
                        WriteSource::Texture { view, .. } => Some(view.format),
                        _ => None,
                    })
                    .collect();
                assert_eq!(views, [form.raw_format(); 2], "{what}");
                planned += 1;
                let other = variant
                    .plan(&dev, &state(PixelFormat::RGBA16Float).shape())
                    .expect("plan");
                assert!(other.variant.extended_range.is_empty(), "{what}");
            }
        }
    }
    assert!(planned > 0);
}

/// A swizzled extended-range, padded or bfloat texture joins the plan's variant
/// (`swizzled_storage`) at a sampled slot as at a storage slot: its view holds raw codes, which
/// a swizzle in the view's component mapping would move before the function decodes them, so the
/// view takes none (`shader_swizzle`) and the variant's function swizzles the decoded texel
/// under the slot's `TextureSwizzle` specialization, the swizzle's word. An
/// unswizzled texture leaves the variant. On every committed profile.
#[test]
fn a_swizzled_extended_texture_joins_the_variant_at_a_sampled_slot() {
    use crate::metal::{TextureSwizzle as S, TextureSwizzleChannels};
    let swizzle = TextureSwizzleChannels::new(S::Alpha, S::Red, S::One, S::Zero);
    let mut planned = 0;
    for source in [Source::Read, Source::ReadWrite] {
        let Some(bc) = assemble("extended swizzle", &kernel(source, false, 0)) else {
            return;
        };
        for (stem, dev) in devices() {
            let Some(plain) = pipeline(&dev, &bc, &LowerInputs::new()) else {
                continue;
            };
            for pf in [
                PixelFormat::BGR10XR,
                PixelFormat::BGR10XRSRGB,
                PixelFormat::BGRA10XR,
                PixelFormat::BGRA10XRSRGB,
                PixelFormat::R10UnormX6,
                PixelFormat::RG10UnormX12,
                PixelFormat::R16Bfloat,
                PixelFormat::RG16Bfloat,
                PixelFormat::RGBA16Bfloat,
            ] {
                let what = format!("{source:?} {pf:?} on {stem}");
                let mut state = state(pf);
                if let Some(BoundTexture::Texture { facts, .. }) = state
                    .stages
                    .get_mut(&Stage::Kernel)
                    .and_then(|t| t.textures.get_mut(&0))
                {
                    facts.swizzle = swizzle;
                }
                let shape = state.shape();
                let (plan, own) = match (
                    plain.plan(&dev, &shape),
                    plain.plan(&dev, &self::state(pf).shape()),
                ) {
                    // A profile without the raw format's storage refuses it for the format.
                    (_, Err(crate::error::BindError::Unsupported(u)))
                        if u.feature == Feature::Format =>
                    {
                        continue
                    }
                    (Ok(plan), Ok(own)) => (plan, own),
                    (swizzled, unswizzled) => panic!(
                        "{what}: swizzled {:?}, unswizzled {:?}",
                        swizzled.map(|_| ()),
                        unswizzled.map(|_| ())
                    ),
                };
                assert!(plan.variant.swizzled_storage.contains(&SOURCE), "{what}");
                assert!(!own.variant.swizzled_storage.contains(&SOURCE), "{what}");
                let key = plan.variant.lowering_inputs(Stage::Kernel);
                let variant = pipeline(&dev, &bc, &key).expect("the variant's inputs");
                let again = variant
                    .plan(&dev, &shape)
                    .unwrap_or_else(|e| panic!("{what}: variant {e}"));
                assert_eq!(again.variant, plan.variant, "{what}");
                let view = again
                    .descriptors
                    .iter()
                    .find_map(|d| match &d.source {
                        WriteSource::Texture { slot, view } if *slot == SOURCE => Some(*view),
                        _ => None,
                    })
                    .expect("the source's view");
                assert!(view.shader_swizzle, "{what}");
                let id = variant
                    .stage
                    .specialization
                    .constants
                    .iter()
                    .find_map(|c| (c.key == SpecKey::TextureSwizzle(SOURCE)).then_some(c.id))
                    .unwrap_or_else(|| panic!("{what}: no TextureSwizzle constant"));
                let e = again
                    .specialization
                    .entries
                    .iter()
                    .find(|e| e.constant_id == id)
                    .expect("TextureSwizzle entry");
                let at = e.offset as usize;
                let word = u32::from_le_bytes(
                    again.specialization.data[at..at + 4]
                        .try_into()
                        .expect("a u32"),
                );
                assert_eq!(word, swizzle.word(), "{what}");
                planned += 1;
            }
        }
    }
    assert!(planned > 0);
}

/// A storage slot of `YCbCrWrite`, a 4:4:4 two-plane YCbCr texture's write shadow, stores each
/// texel the kernel writes as its red, green and blue bits and the mark 1 in an
/// `R32G32B32A32_UINT` image, which the encode kernel reads (`bind::ycbcr_write`); the module
/// validates on the profiles. The held lanes run from below 0 to above 1. A slot that also reads
/// the texture keeps the format's refusal.
#[test]
fn ycbcr_write_slots_store_each_texel_s_bits_and_mark() {
    let Some(bc) = assemble("ycbcr-write", &kernel(Source::Read, false, 0)) else {
        return;
    };
    let shadow = StorageImageChoice::new(
        crate::vk::Format::R32G32B32A32_UINT,
        StorageImageClass::YCbCrWrite,
    );
    let mut i = LowerInputs::new();
    i.storage_images.insert(TARGET, shadow);
    let value = |k: u32| (k % 211) as f32 / 128.0 - 0.25;
    let held = |n: u32| -> [u32; 4] { [0, 1, 2, 3].map(|c| value(4 * n + c).to_bits()) };
    for (stem, p) in profiles() {
        let (plan, m) =
            typed_module_with(&bc, "k", &p, &i).unwrap_or_else(|e| panic!("{stem}: {e}"));
        validate(&bc, &i, &p, stem);
        let images = [image(0, held), image(1, |_| [0; 4])];
        let (_, images) = run(&m, &resources(&plan, 16, &[]), &images);
        let target = texels_of(&images, 1);
        let layout = image(1, |_| [0; 4]);
        for n in 0..SIDE * SIDE {
            let [r, g, b, _] = held(n);
            let at = layout
                .index(0, 0, 0, [n % SIDE, n / SIDE, 0])
                .expect("texel");
            assert_eq!(target[at], [r, g, b, 1], "{stem}: texel {n}");
        }
    }
    let Some(bc) = assemble("ycbcr-write-read", &kernel(Source::ReadWrite, false, 0)) else {
        return;
    };
    let mut both = LowerInputs::new();
    both.storage_images.insert(SOURCE, shadow);
    both.storage_images.insert(TARGET, shadow);
    for (stem, p) in profiles() {
        match typed_module_with(&bc, "k", &p, &both) {
            Err(crate::error::Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::Format, "{stem}: {u:?}")
            }
            Err(other) => panic!("{stem}: {other}"),
            Ok(_) => panic!("{stem}: a YCbCr write slot that reads is lowered"),
        }
    }
}
