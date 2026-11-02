//! A texture read through a slot of another numeric class: plain Metal reads a texture in its
//! format's own class (float, signed or unsigned integer) and hands back the words as the slot's
//! declared type, a `half` result each word's low 16 bits (measured on an Apple M3 Max and an Apple
//! M4, 28 formats, reads, nearest and linear samples and gathers). Under a slot's `cross_class`
//! variant the function declares the image of the bound class, takes that class's own result and
//! hands back its words.
//!
//! - On the CPU interpreter, a read through each foreign class is the words the native slot's
//!   read gives, a `half` result their low 16 bits (a NaN's any NaN where the module widens a
//!   `half` by the device's conversion, `PrecisionNote::NativeNanEncoding`, whose NaN encoding is
//!   the device's), on the lavapipe and MoltenVK profiles; every module validates.
//! - The binder joins a slot bound to a texture of another class to the plan's variant, the
//!   variant's plan states the same key, and the slot's binding views the texture's own format
//!   (an integer slot over an 8-bit sRGB texture joins the sRGB variant too); a slot sampled
//!   with a `ushort` result binds its neutral and says so (`PlanDiagnostic::CrossClassNeutral`),
//!   on every committed profile.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::{
    BindState, BoundBuffer, BoundTexture, ImageSource, PlanDiagnostic, StageTable, TextureFacts,
    WriteSource,
};
use crate::budget::{Budget, SPIRV_WORDS};
use crate::device::DevicePolicy;
use crate::device::{Device, DeviceDescription};
use crate::library::{FunctionConstantValues, Library};
use crate::lower::select::{LowerInputs, Plan};
use crate::metal::{self, PixelFormat, TextureType};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::compile::{compile_bitcode, plan_bitcode, EntryRef};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{
    Axis, ResourceKind, ResourceSlot, SampledType, SlotClass, SlotRef, SpecKey, Stage,
};
use crate::pipeline::testing::{
    device_nan_bits, link_and_validate, llvm_assemble, meter, profile, typed_module_with,
    AssembleError,
};
use crate::pipeline::StageSource;
use crate::pipeline::{ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::{GlobalKind, TypedModule};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// The texture's side: 16 x 16 texels.
const SIDE: u32 = 16;

const TEXTURE: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Texture,
    index: 0,
};

/// Bitcode of the module `body` (after the AIR header), or `None` without `llvm-as`.
fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-cross-class-{}-{}-{}",
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

/// A texel type a function declares at a texture parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ty {
    Float,
    Half,
    Int,
    Uint,
    Ushort,
    Short,
}

impl Ty {
    /// The MSL name, the LLVM vector, its size in bytes, and the intrinsics' suffix.
    fn spelling(self) -> (&'static str, &'static str, u32, &'static str) {
        match self {
            Ty::Float => ("float", "<4 x float>", 16, "v4f32"),
            Ty::Half => ("half", "<4 x half>", 8, "v4f16"),
            Ty::Int => ("int", "<4 x i32>", 16, "s.v4i32"),
            Ty::Uint => ("uint", "<4 x i32>", 16, "u.v4i32"),
            Ty::Ushort => ("ushort", "<4 x i16>", 8, "u.v4i16"),
            Ty::Short => ("short", "<4 x i16>", 8, "s.v4i16"),
        }
    }

    /// The type whose class is `class`, of 32-bit lanes.
    fn of(class: SampledType) -> Ty {
        match class {
            SampledType::Float => Ty::Float,
            SampledType::Sint => Ty::Int,
            _ => Ty::Uint,
        }
    }
}

/// How the kernel accesses `texture2d` slot 0 at its thread's texel `gid`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Access {
    /// `t.read(gid)` of an `access::read` texture.
    Read,
    /// `t.sample(s, coord)` through a nearest constexpr sampler at the texel's centre.
    Sample,
    /// `t.gather(s, coord)` (component 0) through a nearest constexpr sampler at the texel's
    /// centre.
    Gather,
}

/// The words of a nearest constexpr sampler of normalized coordinates, every axis clamped to
/// its edge, no mip filter (`engine_srgb_decode`'s `sampler_words(0, 0, false)`).
fn nearest_words() -> u64 {
    const BASE: u64 = 0x007b_ff00_0008_0000;
    let edge = 1;
    BASE | edge | edge << 3 | edge << 6
}

/// A kernel accessing slot 0 as `access` with `ty` texels at thread `gid` (a 16 x 16 grid),
/// storing the texel to `out[gid.x + 16 gid.y]` (buffer 0).
fn kernel(access: Access, ty: Ty) -> String {
    let (texel, vec4, size, suffix) = ty.spelling();
    let out_ty = format!("{texel}4");
    let (tex_access, call, decl) = match access {
        Access::Read => (
            "read",
            format!(
                "%rs = call ptr addrspace(2) @air.get_read_sampler()\n  %r = call {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)"
            ),
            format!(
                "declare ptr addrspace(2) @air.get_read_sampler()\ndeclare {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)"
            ),
        ),
        Access::Sample => (
            "sample",
            format!(
                "%r = call {{ {vec4}, i8 }} @air.sample_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)"
            ),
            format!(
                "declare {{ {vec4}, i8 }} @air.sample_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)"
            ),
        ),
        Access::Gather => (
            "sample",
            format!(
                "%r = call {{ {vec4}, i8 }} @air.gather_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)"
            ),
            format!(
                "declare {{ {vec4}, i8 }} @air.gather_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)"
            ),
        ),
    };
    let (state, states, state_md) = match access {
        Access::Read => (String::new(), "", ""),
        Access::Sample | Access::Gather => (
            format!(
                "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {}, i64 0], align 8\n\n",
                nearest_words()
            ),
            "!air.sampler_states = !{!6}\n",
            "!6 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
        ),
    };
    format!(
        r#"{state}define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, <2 x i32> %gid) {{
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %row = shl i32 %y, 4
  %n = add i32 %x, %row
  %i = zext i32 %n to i64
  %fx = uitofp i32 %x to float
  %fy = uitofp i32 %y to float
  %cx = fadd float %fx, 0.5
  %cy = fadd float %fy, 0.5
  %c0 = insertelement <2 x float> zeroinitializer, float %cx, i64 0
  %c1 = insertelement <2 x float> %c0, float %cy, i64 1
  %uv = fmul <2 x float> %c1, <float 0.0625, float 0.0625>
  {call}
  %v = extractvalue {{ {vec4}, i8 }} %r, 0
  %q = getelementptr inbounds {vec4}, ptr addrspace(1) %out, i64 %i
  store {vec4} %v, ptr addrspace(1) %q, align {size}
  ret void
}}

{decl}

!air.kernel = !{{!0}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.{tex_access}", !"air.arg_type_name", !"texture2d<{texel}, {tex_access}>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{out_ty}", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}}
{state_md}"#
    )
}

/// Lowering inputs with slot 0 bound to a texture of class `bound`, or none.
fn inputs(bound: Option<SampledType>) -> LowerInputs {
    let mut i = LowerInputs::new();
    if let Some(bound) = bound {
        i.cross_class.insert(TEXTURE, bound);
    }
    i
}

/// Compile the kernel `bc` under `inputs` on `p`, check that the plan's word estimate bounds the
/// emission, and validate the linked module.
fn validate(
    bc: &[u8],
    inputs: &LowerInputs,
    p: &crate::device::profile::DeviceProfile,
    what: &str,
) {
    let consts = FunctionConstantValues::defaults();
    let entry = EntryRef::Name("k");
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
}

/// Word `c` of texel `n` of a texture of class `class`: pseudo-random words of every bit
/// pattern for an integer class (their low halves among them half NaNs, infinities and
/// subnormals), finite `f32` values of every sign and scale for a float one.
fn word(class: SampledType, n: u32, c: usize) -> u32 {
    let mut x = (n * 4 + c as u32).wrapping_mul(0x9E37_79B9) ^ 0x5bd1_e995;
    x ^= x >> 15;
    x = x.wrapping_mul(0x2c1b_3c6d);
    x ^= x >> 12;
    match class {
        SampledType::Float => {
            // A finite exponent away from the subnormals and the infinities.
            let exp = 64 + (x >> 23) % 128;
            (x & 0x807f_ffff) | exp << 23
        }
        _ => x,
    }
}

/// The module's texture resource at slot 0 (a combined binding's, for a sampled slot).
fn resource(m: &TypedModule) -> ResourceSlot {
    m.globals()
        .iter()
        .find_map(|g| match g.kind {
            GlobalKind::Resource(s)
                if s.index == 0
                    && matches!(
                        s.kind,
                        ResourceKind::Texture | ResourceKind::CombinedTextureSampler(_)
                    ) =>
            {
                Some(s)
            }
            _ => None,
        })
        .expect("the texture's resource")
}

/// Resources of a dispatch of one-thread threadgroups over the grid, `out` sized for `bytes`
/// per texel.
fn resources(plan: &Plan, bytes: u32) -> Resources {
    let mut spec = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(axis), 1);
    }
    spec.insert(SpecKey::FlatLocalSize, 1);
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    let out = Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index: 0,
        }),
        address: None,
        bytes: vec![0xab; (bytes * SIDE * SIDE) as usize],
    };
    Resources {
        buffers: vec![out],
        push,
        spec,
    }
}

/// The `out` buffer of kernel `bc` with `ty` texels compiled under `inputs` on `p`, over a
/// texture of class `class`'s words ([`word`]), in both scheduling orders, which must agree.
fn run(
    bc: &[u8],
    ty: Ty,
    inputs: &LowerInputs,
    p: &crate::device::profile::DeviceProfile,
    class: SampledType,
    what: &str,
) -> Vec<u8> {
    let (plan, m) = typed_module_with(bc, "k", p, inputs).unwrap_or_else(|e| panic!("{what}: {e}"));
    validate(bc, inputs, p, what);
    let mut img = Image::new(resource(&m), [SIDE, SIDE, 1], 1, 1, 1);
    for n in 0..SIDE * SIDE {
        let at = img.index(0, 0, 0, [n % SIDE, n / SIDE, 0]).expect("texel");
        img.texels[at] = [0, 1, 2, 3].map(|c| word(class, n, c));
    }
    let res = resources(&plan, ty.spelling().2);
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [SIDE, SIDE, 1],
            order,
            ..Config::default()
        };
        let (o, _) = run_with_images(&m, &cfg, res.clone(), vec![img.clone()])
            .unwrap_or_else(|e| panic!("{what}: interpreter: {e:?}"));
        outs.push(o.buffers.into_iter().next().expect("out"));
    }
    assert_eq!(outs[0], outs[1], "{what}: the scheduling orders disagree");
    outs.swap_remove(0)
}

/// Every read through a slot of another class gives the words the slot of the texture's own
/// class gives, a `half` result their low 16 bits: Metal's rule. A `ushort` result of an integer
/// texture is the low 16 bits too, of a float texture the `half` read's bits. Where the module
/// widens a `half` from its bits by the device and records `PrecisionNote::NativeNanEncoding` or
/// requires `VkProperty::MetalNanMoves` (both profiles here, which have `shaderFloat16`;
/// `device_nan_bits`), a `half` result's NaN is the device's widening's, which the interpreter is
/// not, so a NaN pattern need only come back a NaN.
#[test]
fn accesses_through_another_class_are_the_native_words() {
    use SampledType as S;
    let pairs = [
        (Ty::Float, S::Uint),
        (Ty::Float, S::Sint),
        (Ty::Half, S::Uint),
        (Ty::Half, S::Sint),
        (Ty::Int, S::Uint),
        (Ty::Uint, S::Sint),
        (Ty::Int, S::Float),
        (Ty::Uint, S::Float),
        (Ty::Ushort, S::Sint),
        (Ty::Ushort, S::Float),
    ];
    let profiles: Vec<_> = ["lavapipe", "moltenvk-m3"]
        .into_iter()
        .map(|stem| {
            (
                stem,
                profile(stem, DevicePolicy::default()).expect("profile"),
            )
        })
        .collect();
    let mut checked = 0;
    // The interpreter models no sample or gather: theirs compile and validate in
    // `a_texture_of_another_class_joins_the_variant`, and run on the GPU lanes.
    let access = Access::Read;
    for (ty, class) in pairs {
        // A `ushort` result of a float texture is the `half` result's bits.
        let native = match (ty, class) {
            (Ty::Ushort, S::Float) => Ty::Half,
            _ => Ty::of(class),
        };
        let name = format!("{access:?} {ty:?} over {class:?}");
        let Some(cross) = assemble(&name, &kernel(access, ty)) else {
            return;
        };
        let Some(own) = assemble(&name, &kernel(access, native)) else {
            return;
        };
        for (stem, p) in &profiles {
            let what = format!("{name} on {stem}");
            let got = run(&cross, ty, &inputs(Some(class)), p, class, &what);
            let want = run(&own, native, &inputs(None), p, class, &what);
            let consts = FunctionConstantValues::defaults();
            let device_nans = ty == Ty::Half
                && device_nan_bits(
                    &compile_bitcode(
                        &cross,
                        EntryRef::Name("k"),
                        &consts,
                        p,
                        &inputs(Some(class)),
                        &mut meter(),
                        SPIRV_WORDS,
                    )
                    .unwrap_or_else(|e| panic!("{what}: {e}")),
                );
            let nan = |h: u32| h & 0x7C00 == 0x7C00 && h & 0x3FF != 0;
            let lanes = (SIDE * SIDE * 4) as usize;
            let lane_of = |b: &[u8], t: Ty, lane: usize| match t {
                Ty::Half | Ty::Ushort | Ty::Short => u32::from(u16::from_le_bytes(
                    b[2 * lane..2 * lane + 2].try_into().unwrap(),
                )),
                _ => u32::from_le_bytes(b[4 * lane..4 * lane + 4].try_into().unwrap()),
            };
            for lane in 0..lanes {
                let w = lane_of(&want, native, lane);
                let g = lane_of(&got, ty, lane);
                let w = match ty {
                    Ty::Half | Ty::Ushort | Ty::Short => w & 0xffff,
                    _ => w,
                };
                if device_nans && nan(w) {
                    assert!(nan(g), "{what}: lane {lane}: {g:#x} for the NaN {w:#x}");
                    continue;
                }
                assert_eq!(g, w, "{what}: lane {lane}: {g:#x} vs {w:#x}");
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 10 * 2);
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

/// The kernel compiled into a compute pipeline under `inputs`.
fn pipeline(dev: &Device, bc: &[u8], inputs: &LowerInputs) -> ComputePipeline {
    let lib = Library::from_air(bc).expect("library");
    let f = lib.function("k").expect("function");
    let compiled = dev
        .compile_function(&FunctionDescriptor::new(f), inputs, &Budget::standard())
        .expect("compile");
    dev.compile_compute_pipeline(
        &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
        &Budget::standard(),
    )
    .expect("pipeline")
}

/// A kernel's state: texture slot 0 bound in `pf` (shader read), buffer 0.
fn state(pf: PixelFormat) -> BindState {
    let mut t = StageTable::default();
    let mut facts = TextureFacts::new(TextureType::Type2D, pf);
    facts.usage = metal::TextureUsage::SHADER_READ;
    t.textures.insert(
        0,
        BoundTexture::Texture {
            facts,
            resource_id: None,
        },
    );
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

/// The view format of slot 0's image among a plan's writes, a combined binding's included.
fn view_format(plan: &crate::bind::BindingPlan) -> Option<crate::vk::Format> {
    plan.descriptors.iter().find_map(|d| match &d.source {
        WriteSource::Texture { view, .. } => Some(view.format),
        WriteSource::CombinedImageSampler {
            image: ImageSource::Texture { view, .. },
            ..
        } => Some(view.format),
        _ => None,
    })
}

/// A slot whose compile states `CrossClassReads` bound to a texture of another class joins the
/// plan's variant, with the texture's class; the variant's plan states the same key and views
/// the texture's own format. An integer slot over an 8-bit sRGB texture joins the sRGB variant
/// too, whose raw codes its float reads decode: its gather views the raw format, as a `float`
/// slot's gather of the texture does, on a device that does not sample through Metal's own
/// sampler, whose gather is the table's decode (Metal's uint gather of an `RGBA8Unorm_sRGB` texture
/// is its float gather's bits, the table's decode, 240 of 240 on an M4 and an M3). On every
/// committed profile.
#[test]
fn a_texture_of_another_class_joins_the_variant() {
    use SampledType as S;
    let cases = [
        (Access::Read, Ty::Float, PixelFormat::R32Uint, S::Uint),
        (Access::Read, Ty::Half, PixelFormat::RGBA8Sint, S::Sint),
        (Access::Read, Ty::Int, PixelFormat::RGBA16Uint, S::Uint),
        (Access::Read, Ty::Uint, PixelFormat::RGBA8Unorm, S::Float),
        (Access::Sample, Ty::Float, PixelFormat::RGBA8Uint, S::Uint),
        (Access::Sample, Ty::Int, PixelFormat::RGBA16Float, S::Float),
        (
            Access::Sample,
            Ty::Ushort,
            PixelFormat::RGBA8Unorm,
            S::Float,
        ),
        (Access::Read, Ty::Ushort, PixelFormat::RGBA16Sint, S::Sint),
        (
            Access::Gather,
            Ty::Uint,
            PixelFormat::RGBA8UnormSRGB,
            S::Float,
        ),
        (
            Access::Gather,
            Ty::Uint,
            PixelFormat::RG8UnormSRGB,
            S::Float,
        ),
    ];
    let mut planned = 0;
    for (access, ty, pf, class) in cases {
        let name = format!("{access:?} {ty:?} over {pf:?}");
        let Some(bc) = assemble(&name, &kernel(access, ty)) else {
            return;
        };
        for (stem, dev) in devices() {
            let what = format!("{name} on {stem}");
            let plain = pipeline(&dev, &bc, &LowerInputs::new());
            assert!(
                plain
                    .dependencies
                    .iter()
                    .any(|d| d.fact == RuntimeFact::CrossClassReads { slot: TEXTURE }),
                "{what}: no CrossClassReads"
            );
            let shape = state(pf).shape();
            let plan = match plain.plan(&dev, &shape) {
                Ok(plan) => plan,
                // A device that cannot sample the format refuses it for the format.
                Err(crate::error::BindError::Unsupported(u)) => {
                    eprintln!("{what}: {u:?}");
                    continue;
                }
                Err(e) => panic!("{what}: {e}"),
            };
            assert_eq!(
                plan.variant.cross_class,
                BTreeMap::from([(TEXTURE, class)]),
                "{what}"
            );
            // Only a format the device holds as its UNORM twin joins the sRGB variant.
            let twin = crate::metal::texel_format::srgb_twin_on(pf, dev.profile());
            assert_eq!(
                plan.variant.srgb_textures.contains(&TEXTURE),
                twin,
                "{what}"
            );
            let key = plan.variant.lowering_inputs(Stage::Kernel);
            assert_eq!(key.cross_class, plan.variant.cross_class, "{what}");
            validate(&bc, &key, dev.profile(), &what);
            let variant = pipeline(&dev, &bc, &key);
            let again = variant.plan(&dev, &shape).expect("variant plan");
            assert_eq!(again.variant, plan.variant, "{what}");
            assert!(
                again.diagnostics.is_empty(),
                "{what}: {:?}",
                again.diagnostics
            );
            let view = view_format(&again).unwrap_or_else(|| panic!("{what}: no view"));
            // Every access views the texture's own format, an sRGB one's decoding its codes; a
            // format held as its UNORM twin, the twin's.
            let usages = [crate::device::FormatUsage::SampledImage].into();
            let held = match twin {
                true => crate::metal::format::srgb_raw_twin(pf).expect("a twin"),
                false => pf,
            };
            let want = crate::metal::format::vk_format(held, &usages, dev.profile())
                .expect("the format")
                .format;
            assert_eq!(view, want, "{what}");
            planned += 1;
            // A texture of the declared class leaves the variant.
            let same = match ty {
                Ty::Float | Ty::Half => PixelFormat::RGBA8Unorm,
                Ty::Int => PixelFormat::RGBA8Sint,
                _ => PixelFormat::RGBA8Uint,
            };
            let other = variant.plan(&dev, &state(same).shape()).expect("plan");
            assert!(other.variant.cross_class.is_empty(), "{what}");
        }
    }
    assert!(planned > 0);
}

/// A slot read with a `uint` result over a texture of a depth format, whose read in another
/// class the binder does not take (`metal::texel_format::crosses_classes`), binds the slot's
/// neutral, which the plan reports as `CrossClassNeutral`, and asks for no variant. On every
/// committed profile that samples the format.
#[test]
fn an_unanswered_result_binds_the_neutral_and_says_so() {
    let Some(bc) = assemble("uint over depth", &kernel(Access::Read, Ty::Uint)) else {
        return;
    };
    let mut planned = 0;
    for (stem, dev) in devices() {
        let plain = pipeline(&dev, &bc, &LowerInputs::new());
        assert!(
            plain
                .dependencies
                .iter()
                .any(|d| d.fact == RuntimeFact::CrossClassReads { slot: TEXTURE }),
            "{stem}"
        );
        let plan = match plain.plan(&dev, &state(PixelFormat::Depth32Float).shape()) {
            Ok(plan) => plan,
            Err(crate::error::BindError::Unsupported(u)) => {
                eprintln!("{stem}: {u:?}");
                continue;
            }
            Err(e) => panic!("{stem}: {e}"),
        };
        assert!(plan.variant.cross_class.is_empty(), "{stem}");
        assert!(
            plan.diagnostics.iter().any(
                |d| matches!(d, PlanDiagnostic::CrossClassNeutral { slot, .. } if *slot == TEXTURE)
            ),
            "{stem}: {:?}",
            plan.diagnostics
        );
        assert_eq!(view_format(&plan), None, "{stem}");
        planned += 1;
    }
    assert!(planned > 0);
}

/// What a kernel does with the `texture2d<T>` its argument buffer holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    /// A read of an `access::read` texture, and a width query.
    Read,
    /// A sample through a nearest constexpr sampler.
    Sample,
}

/// Synthetic AIR written for this test: kernel `k` accesses (`held`) the `texture2d<T>` (`T` as
/// `ty` says) its argument buffer (buffer 1, `{ texture2d<T> t; }`) holds at thread `gid`, storing
/// the texel to `out[gid]` (buffer 0).
fn held_kernel(held: Held, ty: Ty) -> String {
    let (texel, vec4, size, suffix) = ty.spelling();
    let (access, state, states, call, decl) = match held {
        Held::Read => (
            "read",
            String::new(),
            "",
            format!(
                "%rs = call ptr addrspace(2) @air.get_read_sampler()\n  %r = call {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)\n  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 0)\n  %wq = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1024\n  store i32 %w, ptr addrspace(1) %wq, align 4"
            ),
            format!(
                "declare ptr addrspace(2) @air.get_read_sampler()\ndeclare {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)\ndeclare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)"
            ),
        ),
        Held::Sample => (
            "sample",
            format!(
                "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {}, i64 0], align 8\n\n",
                nearest_words()
            ),
            "!air.sampler_states = !{!10}\n!10 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
            format!(
                "%r = call {{ {vec4}, i8 }} @air.sample_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)"
            ),
            format!(
                "declare {{ {vec4}, i8 }} @air.sample_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)"
            ),
        ),
    };
    let texture = format!("texture2d<{texel}, {access}>");
    format!(
        r#"%struct.M = type {{ ptr addrspace(1) }}
{state}define void @k(ptr addrspace(1) %out, ptr addrspace(2) %m, <2 x i32> %gid) {{
  %pt = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %row = shl i32 %y, 4
  %n = add i32 %x, %row
  %i = zext i32 %n to i64
  %fx = uitofp i32 %x to float
  %fy = uitofp i32 %y to float
  %c0 = insertelement <2 x float> zeroinitializer, float %fx, i64 0
  %c1 = insertelement <2 x float> %c0, float %fy, i64 1
  %uv = fmul <2 x float> %c1, <float 0.0625, float 0.0625>
  {call}
  %v = extractvalue {{ {vec4}, i8 }} %r, 0
  %q = getelementptr inbounds {vec4}, ptr addrspace(1) %out, i64 %i
  store {vec4} %v, ptr addrspace(1) %q, align {size}
  ret void
}}

{decl}

!air.kernel = !{{!0}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !7}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{texel}4", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !8, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"M", !"air.arg_name", !"m"}}
!7 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}}
!8 = !{{i32 0, i32 8, i32 0, !"{texture}", !"t", !"air.indirect_argument", !9}}
!9 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.{access}", !"air.arg_type_name", !"{texture}", !"air.arg_name", !"t"}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The sampled 2D heap classes a module declares, by numeric type.
fn held_classes(a: &crate::library::FunctionArtifact) -> Vec<SampledType> {
    use crate::pipeline::sym::{HandleClass, ImageAccess, ImageDim};
    let mut out: Vec<SampledType> = a
        .interface
        .descriptors
        .iter()
        .filter_map(|d| match d.slot.kind {
            ResourceKind::Heap(HandleClass::Image(c))
                if c.dim == ImageDim::D2
                    && !c.arrayed
                    && !c.multisampled
                    && !c.depth_compare
                    && c.access == ImageAccess::Sampled =>
            {
                Some(c.sampled_type)
            }
            _ => None,
        })
        .collect();
    out.sort();
    out
}

/// A read and a query of a texture an argument buffer holds, through each declared type, read a
/// member of another numeric class in its own class: on every committed profile with a sampled
/// heap, the module declares the sampled class of every numeric type, stays inside the plan's word
/// estimate, passes L3 and states `RuntimeFact::HeapCrossClassNeutral` for the formats it does not
/// take alone (not `every_member`). A sample through the argument buffer, and a `ushort` read
/// (whose query still crosses), have no answer for any such member: they state the fact for every
/// member of their class, which the planner reports and the runtime announces.
#[test]
fn a_texture_an_argument_buffer_holds_is_read_in_its_own_class() {
    use crate::error::{Error, Feature, Unsupported};
    let consts = FunctionConstantValues::defaults();
    let entry = EntryRef::Name("k");
    let inputs = LowerInputs::new();
    let every = vec![SampledType::Float, SampledType::Sint, SampledType::Uint];
    let mut compiled = 0;
    for (held, ty) in [
        (Held::Read, Ty::Float),
        (Held::Read, Ty::Half),
        (Held::Read, Ty::Int),
        (Held::Read, Ty::Uint),
        (Held::Read, Ty::Ushort),
        (Held::Sample, Ty::Float),
        (Held::Sample, Ty::Uint),
    ] {
        let name = format!("held {held:?} {ty:?}");
        let Some(bc) = assemble(&name, &held_kernel(held, ty)) else {
            return;
        };
        for (stem, p) in crate::spirv::testing::committed_profiles().expect("profiles") {
            let what = format!("{name} on {stem}");
            let a = match compile_bitcode(
                &bc,
                entry,
                &consts,
                &p,
                &inputs,
                &mut meter(),
                SPIRV_WORDS,
            ) {
                Ok(a) => a,
                Err(Error::Unsupported(Unsupported {
                    feature: Feature::ArgumentBufferTextures,
                    ..
                })) => continue,
                Err(e) => panic!("{what}: {e}"),
            };
            validate(&bc, &inputs, &p, &what);
            let classes = held_classes(&a);
            let neutral = |every: bool| {
                a.facts.iter().any(|f| {
                    matches!(f, RuntimeFact::HeapCrossClassNeutral { every_member, .. }
                        if *every_member == every)
                })
            };
            match (held, ty) {
                (Held::Read, Ty::Ushort) => {
                    // Its query still queries such a member through its class.
                    assert_eq!(classes, every, "{what}");
                    assert!(neutral(true), "{what}");
                }
                (Held::Read, _) => {
                    assert_eq!(classes, every, "{what}");
                    assert!(!neutral(true), "{what}: {:?}", a.facts);
                    assert!(neutral(false), "{what}: {:?}", a.facts);
                }
                (Held::Sample, _) => assert!(neutral(true), "{what}"),
            }
            compiled += 1;
        }
    }
    assert!(compiled > 0, "no committed profile has the sampled heap");
}

/// A kernel writing the `ty` texel `src[gid.x + 16 gid.y]` (buffer 0) to `access::write` texture
/// slot 0 at its thread's texel `gid`.
fn write_kernel(ty: Ty) -> String {
    let (texel, vec4, size, suffix) = ty.spelling();
    format!(
        r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %src, <2 x i32> %gid) {{
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %row = shl i32 %y, 4
  %n = add i32 %x, %row
  %i = zext i32 %n to i64
  %q = getelementptr inbounds {vec4}, ptr addrspace(1) %src, i64 %i
  %v = load {vec4}, ptr addrspace(1) %q, align {size}
  call void @air.write_texture_2d.{suffix}(ptr addrspace(1) %t, <2 x i32> %gid, {vec4} %v, i32 0, i32 2)
  ret void
}}

declare void @air.write_texture_2d.{suffix}(ptr addrspace(1), <2 x i32>, {vec4}, i32, i32)

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<{texel}, write>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{texel}4", !"air.arg_name", !"src"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}}
"#
    )
}

/// The 32-bit words a write's lanes take (the probe's): each channel's edges and
/// past them, both signs, and float bits.
const WORDS: [u32; 32] = [
    0x0000_0000,
    0x3F00_0000,
    0x3E80_0000,
    0x3F80_0000,
    0xBF80_0000,
    0x4000_0000,
    0x437F_0000,
    0x4396_0000,
    0x477F_FF00,
    0x7FC0_0000,
    0x8000_0000,
    0x7F80_0000,
    0x0000_0001,
    0x0000_012C,
    0x0000_00FF,
    0x0000_0100,
    0x0000_7FFF,
    0x0000_8000,
    0x0000_FFFF,
    0x0001_0000,
    0x7FFF_FFFF,
    0xFFFF_FFFF,
    0xFFFF_FF80,
    0x3C00_3C00,
    0x4E6E_6B28,
    0xC2C8_0000,
    0x3F7F_FFFF,
    0x3B80_8081,
    0x0000_0080,
    0xFFFF_8000,
    0x0000_03FF,
    0x1234_5678,
];

/// The 16-bit lanes a write takes (the probe's).
const HALVES: [u16; 24] = [
    0x0000, 0x3800, 0x3C00, 0xBC00, 0x5BF8, 0x7C00, 0x7E00, 0x0001, 0x00FF, 0x0100, 0x012C, 0x7FFF,
    0x8000, 0xFFFF, 0x3400, 0xFC00, 0x03FF, 0x0080, 0xFF80, 0x4000, 0x57F8, 0x1234, 0xABCD, 0x7BFF,
];

/// What plain Metal stores in a channel of `bits` bits of an integer format (`signed`) from a
/// lane `lane` (a 32-bit word, or a 16-bit lane where `sixteen`): the lane extended by the
/// format's signedness, saturated to the channel's range, as the format's signedness reads it. The
/// 32-bit word the image holds.
fn metal_store(lane: u32, sixteen: bool, bits: u32, signed: bool) -> u32 {
    let v: i64 = match (sixteen, signed) {
        (true, true) => i64::from(lane as u16 as i16),
        (true, false) => i64::from(lane & 0xffff),
        (false, true) => i64::from(lane as i32),
        (false, false) => i64::from(lane),
    };
    let (lo, hi) = match signed {
        true => (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1),
        false => (0, (1i64 << bits) - 1),
    };
    v.clamp(lo, hi) as u32
}

/// An integer texel's write stores each lane saturated to its channel's range, as plain Metal
/// stores it (`uint` 300 into `R8Uint` stores 255): a 32-bit lane's word as the format's signedness
/// reads it, a 16-bit lane extended by that signedness, through a slot of the format's own class
/// and of another class. The widths come from the format the slot's storage choice declares, or,
/// where the slot declares none, from the binder's `PushField::TextureIntegerChannels` word; both
/// give Metal's words on the interpreter, and a 32-bit channel's lanes stay as written. Every
/// module validates, on the lavapipe and MoltenVK profiles.
#[test]
fn integer_writes_saturate_to_the_bound_formats_channels() {
    use crate::pipeline::interface::PushLoc;
    use crate::pipeline::lower_inputs::{StorageImageChoice, StorageImageClass};
    use crate::pipeline::sym::PushField;
    use crate::vk::Format as F;
    use PixelFormat as P;
    use SampledType as S;
    // The declared type, the bound format (its Vulkan format), and its class where it is not
    // the declared type's.
    let cases = [
        (Ty::Uint, P::RGBA8Uint, F::R8G8B8A8_UINT, None),
        (Ty::Uint, P::R8Uint, F::R8_UINT, None),
        (Ty::Uint, P::RGBA16Uint, F::R16G16B16A16_UINT, None),
        (Ty::Uint, P::RGB10A2Uint, F::A2B10G10R10_UINT_PACK32, None),
        (Ty::Uint, P::RGBA32Uint, F::R32G32B32A32_UINT, None),
        (Ty::Int, P::RGBA8Sint, F::R8G8B8A8_SINT, None),
        (Ty::Int, P::RGBA16Sint, F::R16G16B16A16_SINT, None),
        (Ty::Int, P::R32Sint, F::R32_SINT, None),
        (Ty::Ushort, P::RGBA8Uint, F::R8G8B8A8_UINT, None),
        (Ty::Short, P::RGBA8Sint, F::R8G8B8A8_SINT, None),
        (Ty::Short, P::RGBA16Sint, F::R16G16B16A16_SINT, None),
        (Ty::Int, P::RGBA8Uint, F::R8G8B8A8_UINT, Some(S::Uint)),
        (Ty::Uint, P::RGBA8Sint, F::R8G8B8A8_SINT, Some(S::Sint)),
        (
            Ty::Float,
            P::RGBA16Uint,
            F::R16G16B16A16_UINT,
            Some(S::Uint),
        ),
        (Ty::Short, P::RGBA8Uint, F::R8G8B8A8_UINT, Some(S::Uint)),
        (Ty::Ushort, P::RGBA8Sint, F::R8G8B8A8_SINT, Some(S::Sint)),
    ];
    let profiles: Vec<_> = ["lavapipe", "moltenvk-m3"]
        .into_iter()
        .map(|stem| {
            (
                stem,
                profile(stem, DevicePolicy::default()).expect("profile"),
            )
        })
        .collect();
    let mut checked = 0;
    for (ty, pf, format, cross) in cases {
        let name = format!("write {ty:?} over {pf:?}");
        let Some(bc) = assemble(&name, &write_kernel(ty)) else {
            return;
        };
        let sixteen = matches!(ty, Ty::Ushort | Ty::Short);
        let block = format.block().expect("block");
        let signed = block.color == Some(crate::vk::NumericFormat::Sint);
        let bits = crate::metal::format::integer_channel_bits(format).expect("integer");
        let lane = |n: u32, c: usize| match sixteen {
            true => u32::from(HALVES[(n as usize * 4 + c) * 7 % HALVES.len()]),
            false => WORDS[(n as usize * 4 + c) * 7 % WORDS.len()],
        };
        let src: Vec<u8> = (0..SIDE * SIDE)
            .flat_map(|n| (0..4).map(move |c| (n, c)))
            .flat_map(|(n, c)| match sixteen {
                true => (lane(n, c) as u16).to_le_bytes().to_vec(),
                false => lane(n, c).to_le_bytes().to_vec(),
            })
            .collect();
        // The storage choice declaring the format, and (same class only) none.
        let declared = [true, false];
        for declares in declared.into_iter().filter(|d| *d || cross.is_none()) {
            let mut inputs = LowerInputs::new();
            if let Some(class) = cross {
                inputs.cross_class.insert(TEXTURE, class);
            }
            if declares {
                inputs.storage_images.insert(
                    TEXTURE,
                    StorageImageChoice::new(format, StorageImageClass::Identity),
                );
            }
            for (stem, p) in &profiles {
                let what = format!("{name} (declared {declares}) on {stem}");
                let (plan, m) = typed_module_with(&bc, "k", p, &inputs)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                validate(&bc, &inputs, p, &what);
                let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
                let mut words = 0;
                for f in &plan.push.fields {
                    if let (PushField::TextureIntegerChannels(slot), PushLoc::Push { offset }) =
                        (f.field, f.loc)
                    {
                        assert_eq!(slot, TEXTURE, "{what}");
                        let word = crate::metal::format::integer_channels_word(pf);
                        push[offset as usize..offset as usize + 4]
                            .copy_from_slice(&word.to_le_bytes());
                        words += 1;
                    }
                }
                assert_eq!(words, usize::from(!declares), "{what}: the slot's word");
                let mut spec = BTreeMap::new();
                for axis in [Axis::X, Axis::Y, Axis::Z] {
                    spec.insert(SpecKey::LocalSize(axis), 1);
                }
                spec.insert(SpecKey::FlatLocalSize, 1);
                let res = Resources {
                    buffers: vec![Buffer {
                        slot: Some(ResourceSlot {
                            stage: Stage::Kernel,
                            kind: ResourceKind::Buffer,
                            index: 0,
                        }),
                        address: None,
                        bytes: src.clone(),
                    }],
                    push,
                    spec,
                };
                let img = Image::new(resource(&m), [SIDE, SIDE, 1], 1, 1, 1);
                let cfg = Config {
                    workgroups: [SIDE, SIDE, 1],
                    ..Config::default()
                };
                let (_, images) = run_with_images(&m, &cfg, res, vec![img])
                    .unwrap_or_else(|e| panic!("{what}: interpreter: {e:?}"));
                let img = images.into_iter().next().expect("image");
                for n in 0..SIDE * SIDE {
                    let at = img.index(0, 0, 0, [n % SIDE, n / SIDE, 0]).expect("texel");
                    let got = img.texels[at];
                    for c in 0..4 {
                        let want = metal_store(lane(n, c), sixteen, bits[c], signed);
                        assert_eq!(
                            got[c],
                            want,
                            "{what}: texel {n} lane {c}: {:#x} stored {:#x}, Metal's {want:#x}",
                            lane(n, c),
                            got[c]
                        );
                    }
                }
                checked += 1;
            }
        }
    }
    assert_eq!(checked, (11 * 2 + 5) * 2);
}

/// An `int` or `uint` write into a texture of the float class in a half-float or small-float
/// format stores the `float` of each lane's bits as written: the device's store converts it, as
/// it converts every `float` write, where Metal's integer store rounds to nearest even, so the
/// module records `PrecisionNote::DeviceColorConversion` (and requires no
/// `VkProperty::MetalHalfStorageWrites`, whose rounding is Metal's `float` write's). A 32-bit
/// float format's needs no conversion, and records nothing. On the interpreter the image holds the
/// written `f32` itself.
#[test]
fn integer_writes_into_narrow_floats_take_the_devices_conversion() {
    use crate::device::requirements::VkProperty;
    use crate::pipeline::lower_inputs::{StorageImageChoice, StorageImageClass};
    use crate::vk::Format as F;
    let cases = [
        (Ty::Uint, F::R16G16B16A16_SFLOAT, true),
        (Ty::Int, F::R16G16B16A16_SFLOAT, true),
        (Ty::Uint, F::B10G11R11_UFLOAT_PACK32, true),
        (Ty::Int, F::R32G32B32A32_SFLOAT, false),
    ];
    let mut checked = 0;
    for stem in ["lavapipe", "moltenvk-m3"] {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        for (ty, format, narrow) in cases {
            let what = format!("write {ty:?} over {format:?} on {stem}");
            let Some(bc) = assemble(&what, &write_kernel(ty)) else {
                return;
            };
            let mut inputs = LowerInputs::new();
            inputs.cross_class.insert(TEXTURE, SampledType::Float);
            inputs.storage_images.insert(
                TEXTURE,
                StorageImageChoice::new(format, StorageImageClass::Identity),
            );
            validate(&bc, &inputs, &p, &what);
            let consts = FunctionConstantValues::defaults();
            let a = compile_bitcode(
                &bc,
                EntryRef::Name("k"),
                &consts,
                &p,
                &inputs,
                &mut meter(),
                SPIRV_WORDS,
            )
            .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(
                a.precision.contains(&PrecisionNote::DeviceColorConversion),
                narrow,
                "{what}: {:?}",
                a.precision
            );
            assert!(
                !a.requirements
                    .properties
                    .contains(&VkProperty::MetalHalfStorageWrites),
                "{what}"
            );
            let (plan, m) =
                typed_module_with(&bc, "k", &p, &inputs).unwrap_or_else(|e| panic!("{what}: {e}"));
            let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
            let lane = |n: u32, c: usize| WORDS[(n as usize * 4 + c) * 7 % WORDS.len()];
            let src: Vec<u8> = (0..SIDE * SIDE)
                .flat_map(|n| (0..4).flat_map(move |c| lane(n, c).to_le_bytes()))
                .collect();
            let mut spec = BTreeMap::new();
            for axis in [Axis::X, Axis::Y, Axis::Z] {
                spec.insert(SpecKey::LocalSize(axis), 1);
            }
            spec.insert(SpecKey::FlatLocalSize, 1);
            let res = Resources {
                buffers: vec![Buffer {
                    slot: Some(ResourceSlot {
                        stage: Stage::Kernel,
                        kind: ResourceKind::Buffer,
                        index: 0,
                    }),
                    address: None,
                    bytes: src,
                }],
                push,
                spec,
            };
            let img = Image::new(resource(&m), [SIDE, SIDE, 1], 1, 1, 1);
            let cfg = Config {
                workgroups: [SIDE, SIDE, 1],
                ..Config::default()
            };
            let (_, images) = run_with_images(&m, &cfg, res, vec![img])
                .unwrap_or_else(|e| panic!("{what}: interpreter: {e:?}"));
            let img = images.into_iter().next().expect("image");
            for n in 0..SIDE * SIDE {
                let at = img.index(0, 0, 0, [n % SIDE, n / SIDE, 0]).expect("texel");
                for c in 0..4 {
                    assert_eq!(
                        img.texels[at][c],
                        lane(n, c),
                        "{what}: texel {n} lane {c} stored {:#x}",
                        img.texels[at][c]
                    );
                }
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 8);
}

/// A kernel's state for [`write_kernel`]: texture slot 0 bound in `pf` (shader write), buffer 0.
fn write_state(pf: PixelFormat) -> BindState {
    let mut s = state(pf);
    if let Some(t) = s.stages.get_mut(&Stage::Kernel) {
        let mut facts = TextureFacts::new(TextureType::Type2D, pf);
        facts.usage = metal::TextureUsage::SHADER_WRITE;
        t.textures.insert(
            0,
            BoundTexture::Texture {
                facts,
                resource_id: None,
            },
        );
    }
    s
}

/// Through the device's binder: a write kernel's plan over a texture of each integer format, its
/// variant's compile, the dispatch's push bytes and specialization, run on the interpreter, stores
/// Metal's saturated words, whether the slot declares the format (a device without formatless
/// storage writes, or a texture of another class) or reads the binder's
/// `PushField::TextureIntegerChannels` word. On every committed profile that writes the format.
#[test]
fn the_binder_saturates_each_bound_formats_channels() {
    use crate::bind::Dispatch;
    use PixelFormat as P;
    let cases = [
        (Ty::Uint, P::RGBA8Uint),
        (Ty::Int, P::RGBA16Sint),
        (Ty::Uint, P::RGB10A2Uint),
        (Ty::Uint, P::R32Uint),
        (Ty::Ushort, P::R8Uint),
        (Ty::Int, P::RGBA8Uint),
    ];
    let (mut worded, mut declared) = (0, 0);
    for (ty, pf) in cases {
        let name = format!("bound write {ty:?} over {pf:?}");
        let Some(bc) = assemble(&name, &write_kernel(ty)) else {
            return;
        };
        let sixteen = matches!(ty, Ty::Ushort | Ty::Short);
        let format = crate::metal::format::direct_format(pf).expect("format");
        let signed = format.block().expect("block").color == Some(crate::vk::NumericFormat::Sint);
        let bits = crate::metal::format::integer_channel_bits(format).expect("integer");
        let lane = |n: u32, c: usize| match sixteen {
            true => u32::from(HALVES[(n as usize * 4 + c) * 5 % HALVES.len()]),
            false => WORDS[(n as usize * 4 + c) * 5 % WORDS.len()],
        };
        let src: Vec<u8> = (0..SIDE * SIDE)
            .flat_map(|n| (0..4).map(move |c| (n, c)))
            .flat_map(|(n, c)| match sixteen {
                true => (lane(n, c) as u16).to_le_bytes().to_vec(),
                false => lane(n, c).to_le_bytes().to_vec(),
            })
            .collect();
        for (stem, dev) in devices() {
            let what = format!("{name} on {stem}");
            let s = write_state(pf);
            // A device that declares every storage image's format compiles the slot only under
            // a plan's choice.
            let lib = Library::from_air(&bc).expect("library");
            let f = FunctionDescriptor::new(lib.function("k").expect("function"));
            let budget = Budget::standard();
            if let Err(e) = dev.compile_function(&f, &LowerInputs::new(), &budget) {
                eprintln!("{what}: {e}");
                continue;
            }
            let plain = pipeline(&dev, &bc, &LowerInputs::new());
            let plan = match plain.plan(&dev, &s.shape()) {
                Ok(plan) => plan,
                Err(crate::error::BindError::Unsupported(u)) => {
                    eprintln!("{what}: {u:?}");
                    continue;
                }
                Err(e) => panic!("{what}: {e}"),
            };
            let key = plan.variant.lowering_inputs(Stage::Kernel);
            let pipe = pipeline(&dev, &bc, &key);
            let plan = pipe.plan(&dev, &s.shape()).expect("variant plan");
            let reads_word = pipe
                .dependencies
                .iter()
                .any(|d| d.fact == RuntimeFact::StorageTextureIntegerChannels { slot: TEXTURE });
            match reads_word {
                true => worded += 1,
                false => declared += 1,
            }
            let cmds = plan
                .instantiate_dispatch(
                    &s,
                    &Dispatch::Threadgroups {
                        groups: [SIDE, SIDE, 1],
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
                v[..e.size as usize]
                    .copy_from_slice(&region.specialization.data[at..at + e.size as usize]);
                spec.insert(ids[&e.constant_id], u64::from_le_bytes(v));
            }
            for placement in &pipe.interface.push {
                spec.extend(placement.pushed_values(&push));
            }
            let (_, m) = typed_module_with(&bc, "k", dev.profile(), &key)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let res = Resources {
                buffers: vec![Buffer {
                    slot: Some(ResourceSlot {
                        stage: Stage::Kernel,
                        kind: ResourceKind::Buffer,
                        index: 0,
                    }),
                    address: None,
                    bytes: src.clone(),
                }],
                push,
                spec,
            };
            let img = Image::new(resource(&m), [SIDE, SIDE, 1], 1, 1, 1);
            let cfg = Config {
                workgroups: [SIDE, SIDE, 1],
                ..Config::default()
            };
            let (_, images) = run_with_images(&m, &cfg, res, vec![img])
                .unwrap_or_else(|e| panic!("{what}: interpreter: {e:?}"));
            let img = images.into_iter().next().expect("image");
            for n in 0..SIDE * SIDE {
                let at = img.index(0, 0, 0, [n % SIDE, n / SIDE, 0]).expect("texel");
                for (c, b) in bits.iter().enumerate() {
                    let want = metal_store(lane(n, c), sixteen, *b, signed);
                    assert_eq!(
                        img.texels[at][c],
                        want,
                        "{what}: texel {n} lane {c}: {:#x}",
                        lane(n, c)
                    );
                }
            }
        }
    }
    println!("integer channels: {worded} plans read the word, {declared} declare the format");
    assert!(worded > 0 && declared > 0, "{worded} {declared}");
}
