//! The coordinates a device's gathers and samples read. A gather and a sample through a constexpr
//! normalized sampler are the device's sampler's on every device (Metal's own on MoltenVK), at the
//! coordinate as given: a subnormal coordinate reaches the device unflushed, the border axes'
//! included. A constexpr pixel sampler's sample or gather of a 1D or 2D texture that is no array
//! hands the device's unnormalized sampler the coordinate itself (a nearest sample with an offset,
//! its texel's centre moved by the offset), with no `PixelSamplerScaling`; of an array or a 3D
//! texture its nearest sample hands its texel's centre and the rest the coordinate divided by the
//! level-0 size, noted; a cube reads it as a normalized sampler. A runtime sampler's gather reads
//! only its word's coordinates bit, and its sample also its `NearestPixelSampler` specialization.
//! Every module requires the device's sampler (`VkProperty::MetalSampler` or
//! `VkProperty::DeviceSampling`) and validates on every committed profile, and the plan's word
//! estimate bounds what it emits.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkProperty;
use crate::library::FunctionArtifact;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, AssembleError,
};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Bitcode of the module `body` (after the AIR header), or `None` without `llvm-as`.
fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-coordinates-{}-{}-{}",
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

/// AIR address mode codes.
const ZERO: u64 = 0;
const EDGE: u64 = 1;
const REPEAT: u64 = 2;
const MIRROR: u64 = 3;

/// The words of a constexpr sampler: minification and magnification `filter` (0 nearest, 1
/// linear), mip filter `mip` (0 none, 1 nearest, 2 linear), axes `s`, `t`, `r` addressed by `address`'s
/// codes, and `pixel` coordinates, with the fields Apple's compiler writes for the rest (the
/// `hlf_half_address` fixture's sampler).
fn sampler(filter: u64, mip: u64, address: [u64; 3], pixel: bool) -> u64 {
    const BASE: u64 = 0x007b_ff00_0008_0000;
    BASE | address[0]
        | address[1] << 3
        | address[2] << 6
        | filter << 9
        | filter << 11
        | mip << 13
        | u64::from(pixel) << 15
}

/// What a test kernel does with texture slot 0 at `float2` buffer 0's lane `tid`, storing the
/// `float4` to buffer 1.
#[derive(Clone, Copy, Debug)]
enum Op {
    /// `texture2d<float>::gather(s, uv, offset, component::y)`.
    Gather { offset: (i32, i32) },
    /// `texture2d_array<float>::gather(s, uv, 1, int2(0), component::x)`.
    GatherArray,
    /// `texture1d<float>::sample(s, uv.x)`.
    Sample1D,
    /// `texture2d<float>::sample(s, uv)`.
    Sample2D,
    /// `texture2d<float>::sample(s, uv, offset)`.
    Sample2DOffset { offset: (i32, i32) },
    /// `texture2d_array<float>::sample(s, uv, 1)`.
    SampleArray,
    /// `texture2d<float>::sample(s, uv, level(lod))` with a dynamic level (buffer 2).
    SampleLevel,
    /// `texture3d<float>::sample(s, float3(uv, 0.5))`.
    Sample3D,
    /// `texturecube<float>::sample(s, float3(uv, 0.5))`.
    SampleCube,
}

/// A kernel performing `op` through constexpr sampler `words` (`None`: the runtime
/// `[[sampler(0)]]`).
fn kernel(op: Op, words: Option<u64>) -> String {
    kernel_of(op, words, false)
}

/// [`kernel`], its coordinate, where `flushed`, replaced in the AIR by the zero of its sign on
/// each lane whose magnitude is subnormal (integer operations on its bits, a select between them):
/// the shape a flush of the coordinate takes in the module.
fn kernel_of(op: Op, words: Option<u64>, flushed: bool) -> String {
    let s = if words.is_some() {
        "@__air_sampler_state"
    } else {
        "%s"
    };
    let (tex_ty, call, decl) = match op {
        Op::Gather { offset: (ox, oy) } => (
            "texture2d<float, sample>",
            format!("@air.gather_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {s}, <2 x float> %uv, i1 true, <2 x i32> <i32 {ox}, i32 {oy}>, i32 1, i32 0)"),
            "@air.gather_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)",
        ),
        Op::GatherArray => (
            "texture2d_array<float, sample>",
            format!("@air.gather_texture_2d_array.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {s}, <2 x float> %uv, i32 1, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)"),
            "@air.gather_texture_2d_array.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i32, i1, <2 x i32>, i32, i32)",
        ),
        Op::Sample1D => (
            "texture1d<float, sample>",
            format!("@air.sample_texture_1d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {s}, float %u, i1 false, i32 0, i1 false, float 0.0, float 0.0, i32 0)"),
            "@air.sample_texture_1d.v4f32(ptr addrspace(1), ptr addrspace(2), float, i1, i32, i1, float, float, i32)",
        ),
        Op::Sample2D => (
            "texture2d<float, sample>",
            format!("@air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {s}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)"),
            "@air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)",
        ),
        Op::Sample2DOffset { offset: (ox, oy) } => (
            "texture2d<float, sample>",
            format!("@air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {s}, <2 x float> %uv, i1 true, <2 x i32> <i32 {ox}, i32 {oy}>, i1 false, float 0.0, float 0.0, i32 0)"),
            "@air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)",
        ),
        Op::SampleArray => (
            "texture2d_array<float, sample>",
            format!("@air.sample_texture_2d_array.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {s}, <2 x float> %uv, i32 1, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)"),
            "@air.sample_texture_2d_array.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i32, i1, <2 x i32>, i1, float, float, i32)",
        ),
        Op::SampleLevel => (
            "texture2d<float, sample>",
            format!("@air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {s}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 true, float %lod, float 0.0, i32 0)"),
            "@air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)",
        ),
        Op::Sample3D => (
            "texture3d<float, sample>",
            format!("@air.sample_texture_3d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {s}, <3 x float> %uvw, i1 true, <3 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)"),
            "@air.sample_texture_3d.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, <3 x i32>, i1, float, float, i32)",
        ),
        Op::SampleCube => (
            "texturecube<float, sample>",
            format!("@air.sample_texture_cube.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {s}, <3 x float> %uvw, i1 false, float 0.0, float 0.0, i32 0)"),
            "@air.sample_texture_cube.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, float, float, i32)",
        ),
    };
    let level = matches!(op, Op::SampleLevel);
    let mut params = String::new();
    let mut mds = Vec::new();
    let mut next = 3;
    if level {
        params.push_str(", ptr addrspace(1) %lods");
        mds.push(format!("!{{i32 {next}, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"lods\"}}"));
        next += 1;
    }
    if words.is_none() {
        params.push_str(", ptr addrspace(2) %s");
        mds.push(format!("!{{i32 {next}, !\"air.sampler\", !\"air.location_index\", i32 0, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"s\"}}"));
        next += 1;
    }
    mds.push(format!("!{{i32 {next}, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"tid\"}}"));
    let first = 7;
    let arg_mds: Vec<String> = (0..mds.len()).map(|i| format!("!{}", first + i)).collect();
    let tail: String = mds
        .iter()
        .enumerate()
        .map(|(i, m)| format!("!{} = {m}\n", first + i))
        .collect();
    let load_lod = if level {
        "  %pl = getelementptr inbounds float, ptr addrspace(1) %lods, i64 %i\n  %lod = load float, ptr addrspace(1) %pl, align 4\n"
    } else {
        ""
    };
    let (state, states, state_md) = match words {
        Some(w) => (
            format!("@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {w}, i64 0], align 8\n\n"),
            "!air.sampler_states = !{!6}\n",
            "!6 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
        ),
        None => (String::new(), "", ""),
    };
    let load = match flushed {
        false => "  %uv = load <2 x float>, ptr addrspace(1) %p, align 8\n".to_owned(),
        true => [
            "  %given = load <2 x float>, ptr addrspace(1) %p, align 8",
            "  %bits = bitcast <2 x float> %given to <2 x i32>",
            "  %magnitude = and <2 x i32> %bits, <i32 2147483647, i32 2147483647>",
            "  %small = icmp ult <2 x i32> %magnitude, <i32 8388608, i32 8388608>",
            "  %sign = and <2 x i32> %bits, <i32 -2147483648, i32 -2147483648>",
            "  %kept = select <2 x i1> %small, <2 x i32> %sign, <2 x i32> %bits",
            "  %uv = bitcast <2 x i32> %kept to <2 x float>",
            "",
        ]
        .join("\n"),
    };
    format!(
        r#"{state}define void @k(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out{params}, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <2 x float>, ptr addrspace(1) %c, i64 %i
{load}  %u = extractelement <2 x float> %uv, i64 0
  %uvw = shufflevector <2 x float> %uv, <2 x float> <float 0.5, float 0.5>, <3 x i32> <i32 0, i32 1, i32 2>
{load_lod}  %r = call {{ <4 x float>, i8 }} {call}
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v, ptr addrspace(1) %q, align 16
  ret void
}}

declare {{ <4 x float>, i8 }} {decl}

!air.kernel = !{{!0}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, {arg_mds}}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{tex_ty}", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
{state_md}{tail}"#,
        arg_mds = arg_mds.join(", "),
    )
}

fn uses(a: &FunctionArtifact, name: &str) -> u32 {
    a.strategies.entries.get(name).map_or(0, |u| u.uses)
}

/// Whether compiled `a` requires the device's sampler: `VkProperty::MetalSampler` on a profile
/// that samples through Metal's own sampler, `VkProperty::DeviceSampling` on any other.
fn requires_the_sampler(a: &FunctionArtifact, p: &DeviceProfile) -> bool {
    let property = match p
        .fields()
        .enabled
        .properties
        .contains(&VkProperty::MetalSampler)
    {
        true => VkProperty::MetalSampler,
        false => VkProperty::DeviceSampling,
    };
    a.requirements.properties.contains(&property)
}

/// The kernels of the selection test, each with its access's census name.
fn cases() -> Vec<(&'static str, Op, Option<u64>, &'static str)> {
    let near = |a: [u64; 3]| Some(sampler(0, 0, a, false));
    let gather = Op::Gather { offset: (0, 0) };
    vec![
        ("gather edge", gather, near([EDGE; 3]), "sampler.gather"),
        ("gather zero", gather, near([ZERO; 3]), "sampler.gather"),
        (
            "gather repeat offset",
            Op::Gather { offset: (1, -1) },
            near([REPEAT; 3]),
            "sampler.gather",
        ),
        ("gather mirror", gather, near([MIRROR; 3]), "sampler.gather"),
        (
            "gather linear sampler",
            gather,
            Some(sampler(1, 0, [EDGE; 3], false)),
            "sampler.gather",
        ),
        (
            "gather array",
            Op::GatherArray,
            near([EDGE; 3]),
            "sampler.gather",
        ),
        ("gather runtime sampler", gather, None, "sampler.gather"),
        (
            "nearest repeat 2d",
            Op::Sample2D,
            near([REPEAT; 3]),
            "sampler.sample",
        ),
        (
            "nearest mirror s, edge t",
            Op::Sample2D,
            near([MIRROR, EDGE, EDGE]),
            "sampler.sample",
        ),
        (
            "nearest repeat 1d",
            Op::Sample1D,
            near([REPEAT; 3]),
            "sampler.sample",
        ),
        (
            "nearest repeat r only 3d",
            Op::Sample3D,
            near([EDGE, EDGE, REPEAT]),
            "sampler.sample",
        ),
        (
            "nearest repeat, nearest mip filter, level 0",
            Op::Sample2D,
            Some(sampler(0, 1, [REPEAT; 3], false)),
            "sampler.sample",
        ),
        (
            "nearest edge",
            Op::Sample2D,
            near([EDGE; 3]),
            "sampler.sample",
        ),
        (
            "nearest repeat, nearest mip filter, dynamic level",
            Op::SampleLevel,
            Some(sampler(0, 1, [REPEAT; 3], false)),
            "sampler.sample",
        ),
        (
            "nearest repeat runtime sampler",
            Op::Sample2D,
            None,
            "sampler.sample",
        ),
        (
            "nearest cube",
            Op::SampleCube,
            near([EDGE; 3]),
            "sampler.sample",
        ),
        (
            "linear cube",
            Op::SampleCube,
            Some(sampler(1, 0, [EDGE; 3], false)),
            "sampler.sample",
        ),
        (
            "nearest cube runtime sampler",
            Op::SampleCube,
            None,
            "sampler.sample",
        ),
    ]
}

/// On every committed profile, each kernel's gather or sample through a normalized or runtime
/// sampler is the device's sampler's, under its census name, requires the device's sampler, and
/// validates.
#[test]
fn normalized_gathers_and_samples_are_the_device_s_sampler_s() {
    let profiles = crate::spirv::testing::committed_profiles().expect("profiles");
    for (name, op, words, census) in cases() {
        let Some(bc) = assemble(name, &kernel(op, words)) else {
            return;
        };
        for (stem, p) in &profiles {
            let what = format!("{name} on {stem}");
            let a = compile(&bc, "k", p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(uses(&a, census), 1, "{what}");
            assert!(requires_the_sampler(&a, p), "{what}");
        }
    }
}

/// Over an owned fixture: `unk_gather`'s five gathers of cubes (three color, two depth) and its
/// four comparison gathers (of cubes, and of a depth 2D texture and array) are the device's
/// sampler's on every committed profile that compiles it.
#[test]
fn the_fixture_cube_gathers_are_the_device_s() {
    let Ok(bc) = llvm_assemble(&engine_dir().join("unk_gather.ll")) else {
        eprintln!("SKIPPED: llvm-as");
        return;
    };
    for (stem, p) in crate::spirv::testing::committed_profiles().expect("profiles") {
        let a = match compile(&bc, "unk_gather", &p) {
            Ok(a) => a,
            // A profile without cube arrays refuses the fixture.
            Err(e) => {
                eprintln!("{stem}: refused: {e}");
                continue;
            }
        };
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(uses(&a, "sampler.gather"), 5, "{stem}");
        assert_eq!(uses(&a, "sampler.gather_compare"), 4, "{stem}");
        assert!(requires_the_sampler(&a, &p), "{stem}");
    }
}

/// What a pixel sampler test kernel's module hands the device.
#[derive(Clone, Copy, Debug)]
enum Hands {
    /// The coordinate itself, its bits unchanged, to the device's unnormalized sampler, under the
    /// census name: no `Floor`.
    Itself(&'static str),
    /// The coordinate moved by its offset, to the device's unnormalized sampler, under the census
    /// name, with this many `Floor`s (a nearest sample's texel centre).
    Moved(&'static str, usize),
    /// Its texel's centre over the level-0 size to the normalized sampler, under the census name:
    /// one `Floor`.
    Centre(&'static str),
    /// The coordinate over the level-0 size, under the census name, noted `PixelSamplerScaling`:
    /// no `Floor`.
    Scaled(&'static str),
    /// As through the same sampler with normalized coordinates (a cube).
    AsNormalized,
}

/// The kernels of the pixel sampler test: each through a constexpr pixel sampler, with what its
/// module hands the device.
fn pixel_cases() -> Vec<(&'static str, Op, u64, Hands)> {
    let pixel = |filter, address| sampler(filter, 0, address, true);
    let gather = "sampler.pixel_unnormalized_gather";
    let sample = "sampler.pixel_unnormalized";
    vec![
        (
            "pixel gather edge",
            Op::Gather { offset: (0, 0) },
            pixel(0, [EDGE; 3]),
            Hands::Itself(gather),
        ),
        (
            "pixel gather zero offset",
            Op::Gather { offset: (2, -3) },
            pixel(0, [ZERO; 3]),
            Hands::Moved(gather, 0),
        ),
        (
            "pixel gather linear sampler",
            Op::Gather { offset: (0, 0) },
            pixel(1, [EDGE; 3]),
            Hands::Itself(gather),
        ),
        (
            "pixel gather array",
            Op::GatherArray,
            pixel(0, [ZERO, EDGE, EDGE]),
            Hands::Scaled("sampler.gather"),
        ),
        (
            "pixel linear 1d",
            Op::Sample1D,
            pixel(1, [EDGE; 3]),
            Hands::Itself(sample),
        ),
        (
            "pixel linear 2d",
            Op::Sample2D,
            pixel(1, [ZERO, EDGE, EDGE]),
            Hands::Itself(sample),
        ),
        (
            "pixel linear 2d offset",
            Op::Sample2DOffset { offset: (2, -3) },
            pixel(1, [ZERO; 3]),
            Hands::Moved(sample, 0),
        ),
        (
            "pixel linear 3d",
            Op::Sample3D,
            pixel(1, [EDGE; 3]),
            Hands::Scaled("sampler.sample"),
        ),
        (
            "pixel linear array",
            Op::SampleArray,
            pixel(1, [ZERO; 3]),
            Hands::Scaled("sampler.sample"),
        ),
        (
            "pixel nearest zero 2d",
            Op::Sample2D,
            pixel(0, [ZERO; 3]),
            Hands::Itself(sample),
        ),
        (
            "pixel nearest zero 2d offset",
            Op::Sample2DOffset { offset: (1, -1) },
            pixel(0, [ZERO; 3]),
            Hands::Moved(sample, 1),
        ),
        (
            "pixel nearest zero array",
            Op::SampleArray,
            pixel(0, [ZERO; 3]),
            Hands::Centre("sampler.pixel_texel"),
        ),
        (
            "pixel nearest cube",
            Op::SampleCube,
            pixel(0, [EDGE; 3]),
            Hands::AsNormalized,
        ),
        (
            "pixel linear cube",
            Op::SampleCube,
            pixel(1, [EDGE, ZERO, EDGE]),
            Hands::AsNormalized,
        ),
    ]
}

/// The combined bindings of compiled `a` that read through the device's unnormalized sampler.
fn unnormalized_bindings(a: &FunctionArtifact) -> usize {
    use crate::pipeline::sym::ResourceKind;
    a.interface
        .descriptors
        .iter()
        .filter(
            |d| matches!(d.slot.kind, ResourceKind::CombinedTextureSampler(c) if c.unnormalized),
        )
        .count()
}

/// The `GLSL.std.450` instructions numbered `glsl` the module of `a` calls.
fn glsl_count(a: &FunctionArtifact, glsl: u32) -> usize {
    const OP_EXT_INST: u32 = 12;
    let words = &a.module.words;
    let mut n = 0;
    let mut i = 5;
    while i < words.len() {
        let len = (words[i] >> 16) as usize;
        if words[i] & 0xFFFF == OP_EXT_INST && words.get(i + 4) == Some(&glsl) {
            n += 1;
        }
        i += len.max(1);
    }
    n
}

/// `GLSL.std.450 Floor`, which a texel centre, a footprint cell and a gather's block all take.
const GLSL_FLOOR: u32 = 8;

/// On every committed profile, a constexpr pixel sampler's gather and sample hand the device what
/// [`pixel_cases`] states, under its census name: of a 1D or 2D texture that is no array, `x`
/// itself (its bits unchanged where no offset moves it) through a combined binding of the device's
/// unnormalized sampler, without `PixelSamplerScaling`, a nearest sample with an offset its texel's
/// centre (one `Floor`, noted `DeviceDenormals`); an array's nearest sample its texel's centre over
/// the size; the rest the coordinate over the size (no `Floor`), noted. Each requires the device's
/// sampler and validates; a cube through a pixel sampler compiles as through the same sampler with
/// normalized coordinates, to the same census, notes, and requirements.
#[test]
fn pixel_samplers_hand_the_device_their_coordinate() {
    let profiles = crate::spirv::testing::committed_profiles().expect("profiles");
    for (name, op, words, hands) in pixel_cases() {
        let Some(bc) = assemble(name, &kernel(op, Some(words))) else {
            return;
        };
        let normalized = match hands {
            Hands::AsNormalized => assemble(name, &kernel(op, Some(words & !(1 << 15)))),
            _ => None,
        };
        for (stem, p) in &profiles {
            let what = format!("{name} on {stem}");
            let a = compile(&bc, "k", p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let scaling = a.precision.contains(&PrecisionNote::PixelSamplerScaling);
            let floors = glsl_count(&a, GLSL_FLOOR);
            let (sampled, through) = coordinates_pass_through(&a.module.words);
            assert!(sampled, "{what}: no sample or gather");
            match hands {
                Hands::Itself(census) | Hands::Moved(census, _) => {
                    assert_eq!(uses(&a, census), 1, "{what}");
                    assert_eq!(unnormalized_bindings(&a), 1, "{what}");
                    assert!(!scaling, "{what}");
                    assert_eq!(through, matches!(hands, Hands::Itself(_)), "{what}");
                    let expected = match hands {
                        Hands::Moved(_, n) => n,
                        _ => 0,
                    };
                    assert_eq!(floors, expected, "{what}");
                    assert_eq!(
                        a.precision.contains(&PrecisionNote::DeviceDenormals),
                        expected != 0,
                        "{what}"
                    );
                }
                Hands::Centre(census) => {
                    assert_eq!(uses(&a, census), 1, "{what}");
                    assert_eq!(unnormalized_bindings(&a), 0, "{what}");
                    assert!(!scaling, "{what}");
                    assert_eq!(floors, 1, "{what}");
                    assert!(
                        a.precision.contains(&PrecisionNote::DeviceDenormals),
                        "{what}"
                    );
                }
                Hands::Scaled(census) => {
                    assert_eq!(uses(&a, census), 1, "{what}");
                    assert_eq!(unnormalized_bindings(&a), 0, "{what}");
                    assert!(scaling, "{what}");
                    assert_eq!(floors, 0, "{what}");
                    assert!(!through, "{what}");
                }
                Hands::AsNormalized => {
                    let Some(nbc) = &normalized else { return };
                    let b = compile(nbc, "k", p).unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert_eq!(a.strategies.entries, b.strategies.entries, "{what}");
                    assert_eq!(a.precision, b.precision, "{what}");
                    assert_eq!(a.requirements, b.requirements, "{what}");
                }
            }
            assert!(requires_the_sampler(&a, p), "{what}");
        }
    }
}

/// The fields of the word compiled `a`'s runtime sampler slot specializes.
fn sampler_fields(a: &FunctionArtifact) -> u32 {
    use crate::pipeline::sym::SpecKey;
    a.interface
        .read_keys()
        .into_iter()
        .find_map(|k| match k {
            SpecKey::SamplerState { fields, .. } => Some(fields),
            _ => None,
        })
        .expect("a `SamplerState` specialization")
}

/// Whether compiled `a` reads its runtime sampler slot's `NearestPixelSampler` specialization.
fn reads_nearest_pixels(a: &FunctionArtifact) -> bool {
    use crate::pipeline::sym::SpecKey;
    a.interface
        .read_keys()
        .into_iter()
        .any(|k| matches!(k, SpecKey::NearestPixelSampler(_)))
}

/// On every committed profile, a runtime sampler's gather of a 2D texture or array and its 2D
/// sample read only the coordinates bit of its word (none of its address modes or filters) and
/// record `PixelSamplerScaling`. A gather hands the device the coordinate scaled behind it (no
/// `Floor`); a sample also reads the slot's `NearestPixelSampler` specialization, behind which it
/// hands the device its texel's centre (one `Floor`). Each requires the device's sampler.
#[test]
fn a_runtime_samplers_gather_and_sample_read_only_its_coordinates() {
    use crate::metal::sampler::SamplerWord;
    let profiles = crate::spirv::testing::committed_profiles().expect("profiles");
    for (name, op, nearest) in [
        ("runtime gather", Op::Gather { offset: (0, 0) }, false),
        (
            "runtime gather offset",
            Op::Gather { offset: (2, -1) },
            false,
        ),
        ("runtime gather array", Op::GatherArray, false),
        ("runtime sample", Op::Sample2D, true),
    ] {
        let Some(bc) = assemble(name, &kernel(op, None)) else {
            return;
        };
        for (stem, p) in &profiles {
            let what = format!("{name} on {stem}");
            let a = compile(&bc, "k", p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert!(requires_the_sampler(&a, p), "{what}");
            assert_eq!(sampler_fields(&a), SamplerWord::PIXEL_COORDINATES, "{what}");
            assert!(
                a.precision.contains(&PrecisionNote::PixelSamplerScaling),
                "{what}"
            );
            assert_eq!(reads_nearest_pixels(&a), nearest, "{what}");
            assert_eq!(glsl_count(&a, GLSL_FLOOR), usize::from(nearest), "{what}");
        }
    }
}

/// Whether every coordinate operand of every sample and gather in the module `words` is the
/// coordinate the kernel loaded, its bits unchanged: built from the loaded words by bit moves alone
/// (loads, bitcasts, the extraction, construction and shuffle of lanes, constants), and an array's
/// layer by its conversion to `float`. Also whether the module samples or gathers at all.
fn coordinates_pass_through(words: &[u32]) -> (bool, bool) {
    use crate::spirv_binary::{parse_bytes, Operand};
    use crate::spv::Op as O;
    use std::collections::HashMap;
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let parsed = parse_bytes(&bytes).expect("parse");
    let defs: HashMap<u32, &crate::spirv_binary::Instruction> = parsed
        .instructions
        .iter()
        .filter_map(|i| i.result_id.map(|id| (id, i)))
        .collect();
    let ids = |i: &crate::spirv_binary::Instruction| -> Vec<u32> {
        i.operands
            .iter()
            .filter_map(|o| match o {
                Operand::IdRef(w) => Some(*w),
                _ => None,
            })
            .collect()
    };
    // The value `id` is a bit move of the loaded coordinate (or a constant, or a layer).
    fn moves(
        id: u32,
        defs: &HashMap<u32, &crate::spirv_binary::Instruction>,
        ids: &dyn Fn(&crate::spirv_binary::Instruction) -> Vec<u32>,
    ) -> bool {
        let Some(i) = defs.get(&id) else {
            return false;
        };
        match i.opcode {
            // A load's operand is its pointer, not a value of the coordinate's.
            O::Load
            | O::Constant
            | O::ConstantComposite
            | O::ConstantNull
            | O::ConvertUToF
            | O::ConvertSToF => true,
            O::Bitcast
            | O::CompositeExtract
            | O::CompositeConstruct
            | O::CompositeInsert
            | O::VectorShuffle
            | O::CopyObject => ids(i).into_iter().all(|v| moves(v, defs, ids)),
            _ => false,
        }
    }
    let mut sampled = false;
    let mut all = true;
    for i in &parsed.instructions {
        let coordinate = match i.opcode {
            O::ImageSampleImplicitLod
            | O::ImageSampleExplicitLod
            | O::ImageSampleDrefImplicitLod
            | O::ImageSampleDrefExplicitLod
            | O::ImageGather
            | O::ImageDrefGather
            | O::ImageSparseSampleImplicitLod
            | O::ImageSparseSampleExplicitLod
            | O::ImageSparseGather => ids(i).get(1).copied(),
            _ => continue,
        };
        sampled = true;
        all &= coordinate.is_some_and(|c| moves(c, &defs, &ids));
    }
    (sampled, all)
}

/// The kernels of the subnormal test: through constexpr normalized samplers whose every axis
/// addresses through the border (clamp to zero, and to an opaque white border), nearest and linear,
/// of 1D, 2D and 3D textures and a 2D gather.
fn border_cases() -> Vec<(String, Op, u64)> {
    const OPAQUE_WHITE: u64 = 2 << 56;
    let mut out = Vec::new();
    for (op, what) in [
        (Op::Sample1D, "1d"),
        (Op::Sample2D, "2d"),
        (Op::Sample3D, "3d"),
        (Op::Gather { offset: (0, 0) }, "gather"),
    ] {
        for filter in [0, 1] {
            for (border, colour) in [(0, "zero"), (OPAQUE_WHITE, "white")] {
                let words = sampler(filter, 0, [ZERO; 3], false) | border;
                out.push((format!("border {what} {filter} {colour}"), op, words));
            }
        }
    }
    out
}

/// A subnormal coordinate reaches the device's sampler as given, on every committed profile: the
/// coordinate operand of each sample and gather through a constexpr normalized sampler whose axes
/// address through the border (where Metal's texture unit reads a subnormal as the zero of its
/// sign before the address mode decides, `PrecisionNote::DeviceSampling`) is the loaded coordinate,
/// its bits unchanged. The negative control: the same kernels with the coordinate flushed in the
/// AIR, whose modules read the coordinate's bits before the sample, are told apart.
#[test]
fn a_subnormal_coordinate_reaches_the_device_unflushed() {
    let profiles = crate::spirv::testing::committed_profiles().expect("profiles");
    for (name, op, words) in border_cases() {
        for flushed in [false, true] {
            let Some(bc) = assemble(&name, &kernel_of(op, Some(words), flushed)) else {
                return;
            };
            for (stem, p) in &profiles {
                let what = format!("{name} (flushed: {flushed}) on {stem}");
                let a = compile(&bc, "k", p).unwrap_or_else(|e| panic!("{what}: {e}"));
                link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: {e}"));
                let (sampled, through) = coordinates_pass_through(&a.module.words);
                assert!(sampled, "{what}: no sample or gather");
                assert_eq!(through, !flushed, "{what}");
                if !flushed {
                    assert!(requires_the_sampler(&a, p), "{what}");
                }
            }
        }
    }
}

/// The plan's word estimate bounds the emitted module of every kernel above on every committed
/// profile.
#[test]
fn the_coordinate_estimate_bounds_its_emission() {
    use crate::budget::SPIRV_WORDS;
    use crate::library::FunctionConstantValues;
    use crate::lower::select::LowerInputs;
    use crate::pipeline::compile::{compile_bitcode, plan_bitcode, EntryRef};
    use crate::pipeline::testing::meter;
    let profiles = crate::spirv::testing::committed_profiles().expect("profiles");
    let pixel = pixel_cases()
        .into_iter()
        .map(|(n, o, w, _)| (n, o, Some(w)));
    for (name, op, words) in cases()
        .into_iter()
        .map(|(n, o, w, _)| (n, o, w))
        .chain(pixel)
    {
        let Some(bc) = assemble(name, &kernel(op, words)) else {
            return;
        };
        for (stem, p) in &profiles {
            let what = format!("{name} on {stem}");
            let (entry, consts) = (EntryRef::Name("k"), FunctionConstantValues::defaults());
            let inputs = LowerInputs::new();
            let plan = plan_bitcode(&bc, entry, &consts, p, &inputs, &mut meter())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let a = compile_bitcode(&bc, entry, &consts, p, &inputs, &mut meter(), SPIRV_WORDS)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let emitted = a.module.words.len() as u64;
            eprintln!("{what}: estimate {} emitted {emitted}", plan.estimate.words);
            assert!(
                plan.estimate.words >= emitted,
                "{what}: estimate {} < emitted {emitted}",
                plan.estimate.words
            );
        }
    }
}
