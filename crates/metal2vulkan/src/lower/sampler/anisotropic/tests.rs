//! Metal's anisotropic filter emulated: where a sample takes it, the module's shape, the notes it
//! records and the specialization a runtime sampler slot reads; every module validated by
//! `spirv-val` under its profile. The interpreter leaves `Exp2` unmodelled (the spacing's 6-bit
//! power reads it), so the footprint's values are the GPU lanes' to check (`validation`'s
//! `runtime_lane::anisotropic_filter`); these are shape tests.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::air::abi::{Filter, MipFilter};
use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DevicePolicy};
use crate::library::FunctionArtifact;
use crate::metal::sampler::SamplerWord;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::sym::SpecKey;
use crate::pipeline::sym::{SlotClass, SlotRef, Stage};
use crate::pipeline::testing::{
    compile, compile_with, link_and_validate, llvm_assemble, profile, AssembleError,
};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// A public fixture's constexpr sampler words (bit 63 clear), the base of the states below.
const BASE_WORDS: u64 = 34901797601036873;

const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
const OP_IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
const OP_IMAGE_QUERY_LEVELS: u16 = 106;
/// The taps an emulation's code holds, whatever its degree: tap 0 and the loop's one (one
/// explicit-level sample each, the device blending a linear mip filter's two levels).
const WRITTEN_TAPS: u32 = 2;

/// The constexpr sampler words of linear filters, mip filter `mip` and `maxAnisotropy` `aniso`.
fn words(mip: MipFilter, aniso: u32) -> u64 {
    let mut s = super::super::constexpr::state_of([BASE_WORDS, 0]).expect("state");
    s.min_filter = Filter::Linear;
    s.mag_filter = Filter::Linear;
    s.mip_filter = mip;
    s.max_anisotropy = aniso;
    super::super::constexpr::state_words(&s)[0]
}

fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-anisotropic-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, format!("{HEADER}{body}")).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}\n{body}"),
    }
}

/// A fragment function sampling `texture2d<float>` slot 0 at its position over 16 at its
/// implicit level, through the constexpr sampler of first state word `words`.
fn constexpr_fragment(words: u64) -> String {
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8

define <4 x float> @k(<4 x float> %pos, ptr addrspace(1) %t) {{
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %uv = fmul <2 x float> %xy, <float 6.250000e-02, float 6.250000e-02>
  %r = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  ret <4 x float> %v
}}

declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.fragment = !{{!0}}
!air.sampler_states = !{{!6}}
!0 = !{{ptr @k, !1, !3}}
!1 = !{{!2}}
!2 = !{{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}}
!3 = !{{!4, !5}}
!4 = !{{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}}
!5 = !{{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!6 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
"#
    )
}

/// A kernel sampling `texture2d<float>` slot 0 with explicit gradients through runtime sampler
/// slot 0, writing the texel to buffer 0.
const RUNTIME_KERNEL: &str = r#"define void @k(ptr addrspace(1) %t, ptr addrspace(2) %s, ptr addrspace(1) %o, <2 x i32> %gid) {
  %x = extractelement <2 x i32> %gid, i64 0
  %xf = uitofp i32 %x to float
  %u = fmul float %xf, 1.562500e-02
  %uv = insertelement <2 x float> <float 0.000000e+00, float 5.000000e-01>, float %u, i64 0
  %r = call { <4 x float>, i8 } @air.sample_texture_2d_grad.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> %uv, <2 x float> <float 2.500000e-01, float 0.000000e+00>, <2 x float> <float 0.000000e+00, float 3.125000e-02>, float 0.000000e+00, i1 true, <2 x i32> zeroinitializer, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %p = getelementptr <4 x float>, ptr addrspace(1) %o, i32 %x
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

declare { <4 x float>, i8 } @air.sample_texture_2d_grad.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, <2 x float>, <2 x float>, float, i1, <2 x i32>, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"o"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;

/// Committed profile `stem` without `samplerAnisotropy`: every Vulkan sampler is isotropic there,
/// so Metal's anisotropic filter is emulated (the emulation's home), the profile the emulation's
/// shape tests compile under.
fn p(stem: &str) -> DeviceProfile {
    let mut desc = crate::pipeline::testing::description(stem).expect(stem);
    desc.features.remove(&VkFeature::SamplerAnisotropy);
    Device::new(desc, DevicePolicy::default())
        .expect(stem)
        .profile()
        .clone()
}

/// Committed profile `stem` as described, `samplerAnisotropy` included: the device's own
/// anisotropic filter.
fn full(stem: &str) -> DeviceProfile {
    profile(stem, DevicePolicy::default()).expect(stem)
}

fn ok(bc: &[u8], entry: &str, p: &DeviceProfile) -> std::sync::Arc<FunctionArtifact> {
    let a = compile(bc, entry, p).unwrap_or_else(|e| panic!("{entry}: compile: {e}"));
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{entry}: L3: {e}"));
    a
}

/// [`RUNTIME_KERNEL`]'s runtime sampler slot.
const SLOT: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Sampler,
    index: 0,
};

/// The lowering inputs of the variant naming [`SLOT`] (`LoweringInputs::anisotropic_samplers`).
fn variant() -> LoweringInputs {
    let mut inputs = LoweringInputs::new();
    inputs.anisotropic_samplers.insert(SLOT);
    inputs
}

/// [`ok`] of entry `k` under `inputs`.
fn ok_with(
    bc: &[u8],
    inputs: &LoweringInputs,
    p: &DeviceProfile,
) -> std::sync::Arc<FunctionArtifact> {
    let a = compile_with(bc, "k", p, inputs).unwrap_or_else(|e| panic!("k: compile: {e}"));
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("k: L3: {e}"));
    a
}

/// The count of instructions of opcode `op` in `a`'s module.
fn count(a: &FunctionArtifact, op: u16) -> usize {
    let words = &a.module.words;
    let mut n = 0;
    let mut i = 5;
    while i < words.len() {
        let len = ((words[i] >> 16) as usize).max(1);
        n += usize::from((words[i] & 0xffff) as u16 == op);
        i += len;
    }
    n
}

/// The loops of `a`'s module whose `OpLoopMerge` asks the driver not to unroll them
/// (`LoopControl` `DontUnroll`, `0x2`).
fn dont_unroll_loops(a: &FunctionArtifact) -> usize {
    const OP_LOOP_MERGE: u32 = 246;
    const DONT_UNROLL: u32 = 0x2;
    let words = &a.module.words;
    let mut n = 0;
    let mut i = 5;
    while i < words.len() {
        let len = ((words[i] >> 16) as usize).max(1);
        if words[i] & 0xffff == OP_LOOP_MERGE && len >= 4 {
            n += usize::from(words[i + 3] & DONT_UNROLL != 0);
        }
        i += len;
    }
    n
}

/// A fragment's implicit-level sample through a constexpr sampler of `maxAnisotropy` 3, 8 and 16
/// under a linear mip filter, on devices without `samplerAnisotropy`, is Metal's filter emulated:
/// no implicit-level sample, one explicit-level sample a tap, the level count queried, and
/// `AnisotropicFilter` recorded. With the feature (MoltenVK's here) the device's sample stays,
/// unrecorded. Whatever the degree Metal rounds the anisotropy to (4, 8, 16), the code holds tap 0
/// and one loop of the others that asks the driver not to unroll it.
#[test]
fn a_constexpr_anisotropic_sample_takes_metals_filter() {
    for aniso in [3, 8, 16] {
        let what = format!("maxAnisotropy {aniso}");
        let Some(bc) = assemble(&what, &constexpr_fragment(words(MipFilter::Linear, aniso))) else {
            return;
        };
        for stem in ["lavapipe", "amd-rdna3-radv", "intel-irisxe-anv"] {
            let what = format!("{what}, {stem}");
            let a = ok(&bc, "k", &p(stem));
            assert_eq!(count(&a, OP_IMAGE_SAMPLE_IMPLICIT_LOD), 0, "{what}");
            assert_eq!(
                count(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD),
                WRITTEN_TAPS as usize,
                "{what}"
            );
            assert_eq!(dont_unroll_loops(&a), 1, "{what}");
            assert_eq!(count(&a, OP_IMAGE_QUERY_LEVELS), 1, "{what}");
            assert!(
                a.precision.contains(&PrecisionNote::AnisotropicFilter),
                "{what}"
            );
            assert!(
                !a.precision
                    .contains(&PrecisionNote::DeviceAnisotropicFilter),
                "{what}"
            );
        }
        let a = ok(&bc, "k", &full("moltenvk-m3"));
        assert_eq!(
            count(&a, OP_IMAGE_SAMPLE_IMPLICIT_LOD),
            1,
            "{what}, MoltenVK"
        );
        assert!(
            !a.precision.contains(&PrecisionNote::AnisotropicFilter),
            "{what}"
        );
    }
    // An isotropic sampler is untouched.
    let Some(bc) = assemble(
        "isotropic",
        &constexpr_fragment(words(MipFilter::Linear, 1)),
    ) else {
        return;
    };
    let a = ok(&bc, "k", &p("lavapipe"));
    assert_eq!(count(&a, OP_IMAGE_SAMPLE_IMPLICIT_LOD), 1, "isotropic");
    assert!(!a.precision.contains(&PrecisionNote::AnisotropicFilter));
}

/// An explicit-gradient sample through a runtime sampler slot states the slot's
/// `AnisotropicSampler` fact. Compiled without the slot named, the function is the plain sample
/// alone: no `SamplerAnisotropy` specialization, no taps, no note. Under the variant naming the
/// slot it reads the slot's `SamplerAnisotropy` specialization and its word's anisotropy and mip
/// filter: the emulation (its taps' explicit-level samples) where the word states anisotropy, the
/// plain sample elsewhere, and `AnisotropicFilter` recorded under that word alone; each on a
/// device without `samplerAnisotropy`. With the feature (MoltenVK's here) nothing of it is
/// compiled, and no fact is stated.
#[test]
fn a_runtime_slot_takes_metals_filter_where_its_word_states_anisotropy() {
    let Some(bc) = assemble("runtime", RUNTIME_KERNEL) else {
        return;
    };
    let fact = RuntimeFact::AnisotropicSampler { slot: SLOT };
    let degree = |a: &FunctionArtifact| {
        a.interface
            .read_keys()
            .into_iter()
            .any(|k| matches!(k, SpecKey::SamplerAnisotropy(_)))
    };
    let noted = |a: &FunctionArtifact| {
        a.specialized
            .iter()
            .any(|n| n.note == PrecisionNote::AnisotropicFilter)
    };
    for stem in ["lavapipe", "amd-rdna3-radv"] {
        let base = ok(&bc, "k", &p(stem));
        assert!(base.facts.contains(&fact), "{stem}: {:?}", base.facts);
        assert!(!degree(&base), "{stem}: {:?}", base.interface.read_keys());
        assert!(!noted(&base), "{stem}");
        let plain = count(&base, OP_IMAGE_SAMPLE_EXPLICIT_LOD);
        let a = ok_with(&bc, &variant(), &p(stem));
        assert!(a.facts.contains(&fact), "{stem}: {:?}", a.facts);
        let keys = a.interface.read_keys();
        assert!(degree(&a), "{stem}: {keys:?}");
        let fields = keys.into_iter().find_map(|k| match k {
            SpecKey::SamplerState { fields, .. } => Some(fields),
            _ => None,
        });
        let read = SamplerWord::ANISOTROPIC | SamplerWord::MIP_FILTER;
        assert!(
            fields.is_some_and(|f| f & read == read),
            "{stem}: {fields:?}"
        );
        // The variant is the base with the emulation's arm: every sample the base takes, and
        // one a tap of the emulation's, its taps past the first in a loop the driver is asked
        // not to unroll.
        assert!(
            count(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD) >= plain + WRITTEN_TAPS as usize,
            "{stem}: {plain}"
        );
        assert_eq!(dont_unroll_loops(&a), 1, "{stem}");
        assert_eq!(dont_unroll_loops(&base), 0, "{stem}");
        assert!(a.module.words.len() > base.module.words.len(), "{stem}");
        assert!(noted(&a), "{stem}");
    }
    let a = ok(&bc, "k", &full("moltenvk-m3"));
    assert!(!degree(&a));
    assert!(!a.facts.contains(&fact));
}

/// The binder names a slot of the `AnisotropicSampler` fact in the plan's variant where the shape
/// binds a sampler whose word states anisotropy (`maxAnisotropy` above 1), whatever its filters,
/// and nowhere else; the variant's own plan for that shape states the same key, so a draw or
/// dispatch settles on it. Its compiled pipeline round trips as the cached form. On lavapipe
/// without `samplerAnisotropy`, where the emulation runs.
#[test]
fn a_bound_anisotropic_sampler_asks_for_the_variant() {
    use crate::bind::{
        BindState, BoundSampler, BoundTexture, SamplerDescriptor, StageTable, TextureFacts,
    };
    use crate::budget::Budget;
    use crate::cache::{PipelineArtifact, VariantKey};
    use crate::device::DeviceDescription;
    use crate::library::Library;
    use crate::metal;
    use crate::pipeline::artifact::ComputePipeline;
    use crate::pipeline::descriptor::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};
    let Some(bc) = assemble("binder", RUNTIME_KERNEL) else {
        return;
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/lavapipe.json");
    let text = std::fs::read_to_string(&path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    assert!(desc.features.remove(&VkFeature::SamplerAnisotropy));
    let dev = Device::new(desc, DevicePolicy::default()).expect("device");
    let pipeline = |inputs: &LoweringInputs| {
        let function = Library::from_air(&bc)
            .and_then(|lib| lib.function("k"))
            .expect("kernel");
        let compiled = dev
            .compile_function(
                &FunctionDescriptor::new(function),
                inputs,
                &Budget::standard(),
            )
            .expect("function");
        dev.compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
            &Budget::standard(),
        )
        .expect("pipeline")
    };
    let shape = |desc: SamplerDescriptor| {
        let mut table = StageTable::default();
        let mut facts =
            TextureFacts::new(metal::TextureType::Type2D, metal::PixelFormat::RGBA8Unorm);
        facts.usage = metal::TextureUsage::SHADER_READ;
        facts.mipmap_level_count = 4;
        table.textures.insert(
            0,
            BoundTexture::Texture {
                facts,
                resource_id: None,
            },
        );
        table.samplers.insert(
            0,
            BoundSampler {
                desc,
                resource_id: None,
            },
        );
        let mut state = BindState::default();
        state.stages.insert(Stage::Kernel, table);
        state.shape()
    };
    let sampler = |filter, mip, anisotropy| {
        let mut desc = SamplerDescriptor::new();
        desc.min_filter = filter;
        desc.mag_filter = filter;
        desc.mip_filter = mip;
        desc.max_anisotropy = anisotropy;
        desc
    };
    let mut key = VariantKey::new();
    key.anisotropic_samplers.insert(SLOT);
    let base = pipeline(&LoweringInputs::new());
    let named = pipeline(&key.lowering_inputs(Stage::Kernel));
    let decoded = ComputePipeline::from_bytes(&named.to_bytes(), &dev).expect("decoded");
    assert_eq!(decoded, named);
    let linear = metal::SamplerMinMagFilter::Linear;
    let nearest = metal::SamplerMinMagFilter::Nearest;
    let (none, near) = (
        metal::SamplerMipFilter::NotMipmapped,
        metal::SamplerMipFilter::Nearest,
    );
    // Anisotropic samplers, the consumer's (linear filters, a nearest mip filter) among them.
    for desc in [
        sampler(linear, near, 16),
        sampler(linear, near, 2),
        sampler(nearest, none, 4),
    ] {
        let shape = shape(desc);
        let plan = base.plan(&dev, &shape).expect("plan");
        assert_eq!(
            plan.variant.anisotropic_samplers, key.anisotropic_samplers,
            "{desc:?}"
        );
        let plan = decoded.plan(&dev, &shape).expect("variant's plan");
        assert_eq!(
            plan.variant.anisotropic_samplers, key.anisotropic_samplers,
            "{desc:?}"
        );
    }
    // Isotropic samplers and none: the base.
    for desc in [
        sampler(linear, none, 1),
        sampler(linear, near, 1),
        SamplerDescriptor::new(),
    ] {
        let plan = base.plan(&dev, &shape(desc)).expect("plan");
        assert!(plan.variant.anisotropic_samplers.is_empty(), "{desc:?}");
    }
    let plan = base
        .plan(&dev, &crate::bind::BindShape::default())
        .expect("plan");
    assert!(plan.variant.anisotropic_samplers.is_empty());
}

/// A fragment function sampling slot 0, a texture of AIR type `tex_ty`, at its implicit level
/// through the constexpr sampler of first state word `words`: `call` makes `%v` (a `float4`) of
/// `%uv` (the position over 16) or `%uvw` (`%uv` and 0.5); `decl` is the intrinsic's declaration.
fn out_of_reach_fragment(words: u64, tex_ty: &str, call: &str, decl: &str) -> String {
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8

define <4 x float> @k(<4 x float> %pos, ptr addrspace(1) %t) {{
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %uv = fmul <2 x float> %xy, <float 6.250000e-02, float 6.250000e-02>
  %uvw = shufflevector <2 x float> %uv, <2 x float> <float 5.000000e-01, float 5.000000e-01>, <3 x i32> <i32 0, i32 1, i32 2>
  {call}
  ret <4 x float> %v
}}

declare {decl}

!air.fragment = !{{!0}}
!air.sampler_states = !{{!6}}
!0 = !{{ptr @k, !1, !3}}
!1 = !{{!2}}
!2 = !{{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}}
!3 = !{{!4, !5}}
!4 = !{{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}}
!5 = !{{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{tex_ty}", !"air.arg_name", !"t"}}
!6 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
"#
    )
}

/// [`out_of_reach_fragment`]'s call and declaration of a cube's implicit-level sample.
const CUBE_CALL: &str = "%r = call { <4 x float>, i8 } @air.sample_texture_cube.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <3 x float> %uvw, i1 false, float 0.0, float 0.0, i32 0)\n  %v = extractvalue { <4 x float>, i8 } %r, 0";
const CUBE_DECL: &str = "{ <4 x float>, i8 } @air.sample_texture_cube.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, float, float, i32)";

/// An implicit-level sample out of the emulation's reach (a cube, a 3D texture, a depth texture, a
/// comparison) through a constexpr sampler of `maxAnisotropy` 8 is the device's own filter, and
/// records `DeviceAnisotropicFilter` (never `AnisotropicFilter`) on every device without
/// `samplerAnisotropy`: the sample stays at its implicit level, with no tap. With the feature
/// (MoltenVK's here) neither note is recorded. The same samples through an isotropic sampler record
/// neither.
#[test]
fn an_anisotropic_sample_out_of_reach_records_the_device_filter() {
    let anisotropic = words(MipFilter::Linear, 8);
    let isotropic = words(MipFilter::Linear, 1);
    // `compare_func::less` in a constexpr sampler's first word.
    let less = |w: u64| w & !(0xf << 16) | 1 << 16;
    let depth = |what: &str| {
        format!(
            "%r = call {{ float, i8 }} @air.{what}(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, i32 1, <2 x float> %uv,{} i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)\n  %f = extractvalue {{ float, i8 }} %r, 0\n  %v = insertelement <4 x float> <float 0.0, float 0.0, float 0.0, float 1.0>, float %f, i64 0",
            if what.starts_with("sample_compare") { " float 0.5," } else { "" }
        )
    };
    let cases: [(&str, u64, u64, &str, String, &str); 4] = [
        (
            "cube",
            anisotropic,
            isotropic,
            "texturecube<float, sample>",
            CUBE_CALL.to_owned(),
            CUBE_DECL,
        ),
        (
            "3d",
            anisotropic,
            isotropic,
            "texture3d<float, sample>",
            "%r = call { <4 x float>, i8 } @air.sample_texture_3d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <3 x float> %uvw, i1 true, <3 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)\n  %v = extractvalue { <4 x float>, i8 } %r, 0".to_owned(),
            "{ <4 x float>, i8 } @air.sample_texture_3d.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, <3 x i32>, i1, float, float, i32)",
        ),
        (
            "depth",
            anisotropic,
            isotropic,
            "depth2d<float, sample>",
            depth("sample_depth_2d.f32"),
            "{ float, i8 } @air.sample_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, i1, <2 x i32>, i1, float, float, i32)",
        ),
        (
            "compare",
            less(anisotropic),
            less(isotropic),
            "depth2d<float, sample>",
            depth("sample_compare_depth_2d.f32"),
            "{ float, i8 } @air.sample_compare_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, float, i1, <2 x i32>, i1, float, float, i32)",
        ),
    ];
    for (name, anisotropic, isotropic, tex_ty, call, decl) in &cases {
        let text = out_of_reach_fragment(*anisotropic, tex_ty, call, decl);
        let Some(bc) = assemble(name, &text) else {
            return;
        };
        for stem in ["lavapipe", "amd-rdna3-radv", "intel-irisxe-anv"] {
            let what = format!("{name}, {stem}");
            let a = ok(&bc, "k", &p(stem));
            assert!(
                a.precision
                    .contains(&PrecisionNote::DeviceAnisotropicFilter),
                "{what}: {:?}",
                a.precision
            );
            assert!(
                !a.precision.contains(&PrecisionNote::AnisotropicFilter),
                "{what}"
            );
            assert_eq!(count(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD), 0, "{what}");
            assert_eq!(dont_unroll_loops(&a), 0, "{what}");
        }
        let a = ok(&bc, "k", &full("moltenvk-m3"));
        assert!(
            !a.precision
                .contains(&PrecisionNote::DeviceAnisotropicFilter),
            "{name}, MoltenVK"
        );
        let text = out_of_reach_fragment(*isotropic, tex_ty, call, decl);
        let Some(bc) = assemble(name, &text) else {
            return;
        };
        let a = ok(&bc, "k", &p("lavapipe"));
        assert!(
            !a.precision
                .contains(&PrecisionNote::DeviceAnisotropicFilter),
            "{name}, isotropic"
        );
    }
}

/// Committed profile `stem` under `policy`, with `samplerAnisotropy` withheld or not.
fn anisotropy_of(stem: &str, policy: DevicePolicy, withheld: bool) -> DeviceProfile {
    let mut desc = crate::pipeline::testing::description(stem).expect(stem);
    if withheld {
        assert!(
            desc.features.remove(&VkFeature::SamplerAnisotropy),
            "{stem}"
        );
    }
    Device::new(desc, policy).expect(stem).profile().clone()
}

/// Without `samplerAnisotropy` every Vulkan sampler is isotropic, so Metal's anisotropic filter is
/// emulated where its reach takes the sample, and the device's isotropic filter is recorded
/// elsewhere. On lavapipe with the feature withheld, a fragment's implicit-level sample through a constexpr sampler of
/// `maxAnisotropy` 8 takes the taps (`AnisotropicFilter`), a runtime slot's sample states its
/// `AnisotropicSampler` fact and takes the taps under the variant (`AnisotropicFilter` under its
/// word), and a cube's implicit-level sample records `DeviceAnisotropicFilter`. With the feature
/// the module compiles none of it: the device's own sample, no fact, no note. An isotropic
/// sampler's sample is the device's either way.
#[test]
fn without_sampler_anisotropy_metals_filter_is_emulated() {
    let Some(constexpr) = assemble(
        "constexpr",
        &constexpr_fragment(words(MipFilter::Linear, 8)),
    ) else {
        return;
    };
    let Some(isotropic) = assemble(
        "isotropic",
        &constexpr_fragment(words(MipFilter::Linear, 1)),
    ) else {
        return;
    };
    let Some(runtime) = assemble("runtime", RUNTIME_KERNEL) else {
        return;
    };
    let cube = out_of_reach_fragment(
        words(MipFilter::Linear, 8),
        "texturecube<float, sample>",
        CUBE_CALL,
        CUBE_DECL,
    );
    let Some(cube) = assemble("cube", &cube) else {
        return;
    };
    let fact = RuntimeFact::AnisotropicSampler { slot: SLOT };
    for withheld in [false, true] {
        let what = format!("samplerAnisotropy withheld {withheld}");
        let p = anisotropy_of("lavapipe", DevicePolicy::default(), withheld);
        let device = !withheld;
        let a = ok(&constexpr, "k", &p);
        assert_eq!(
            count(&a, OP_IMAGE_SAMPLE_IMPLICIT_LOD),
            usize::from(device),
            "{what}"
        );
        assert_eq!(dont_unroll_loops(&a), usize::from(!device), "{what}");
        assert_eq!(
            a.precision.contains(&PrecisionNote::AnisotropicFilter),
            !device,
            "{what}: {:?}",
            a.precision
        );
        let i = ok(&isotropic, "k", &p);
        assert_eq!(count(&i, OP_IMAGE_SAMPLE_IMPLICIT_LOD), 1, "{what}");
        assert!(
            !i.precision.contains(&PrecisionNote::AnisotropicFilter),
            "{what}"
        );
        let base = ok(&runtime, "k", &p);
        assert_eq!(base.facts.contains(&fact), !device, "{what}");
        if !device {
            let v = ok_with(&runtime, &variant(), &p);
            assert_eq!(dont_unroll_loops(&v), 1, "{what}");
            assert!(
                (v.specialized.iter()).any(|n| n.note == PrecisionNote::AnisotropicFilter),
                "{what}"
            );
        }
        let c = ok(&cube, "k", &p);
        assert_eq!(
            c.precision
                .contains(&PrecisionNote::DeviceAnisotropicFilter),
            !device,
            "{what}: {:?}",
            c.precision
        );
        assert_eq!(dont_unroll_loops(&c), 0, "{what}");
    }
}

/// A kernel sampling `texturecube<float>` slot 0 with explicit gradients through the constexpr
/// sampler of first state word `words`, writing the texel to buffer 0.
fn constexpr_cube_kernel(words: u64) -> String {
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8

define void @k(ptr addrspace(1) %t, ptr addrspace(1) %o, i32 %gid) {{
  %xf = uitofp i32 %gid to float
  %y = fmul float %xf, 1.562500e-02
  %d = insertelement <3 x float> <float 1.000000e+00, float 0.000000e+00, float -5.000000e-01>, float %y, i64 1
  %r = call {{ <4 x float>, i8 }} @air.sample_texture_cube_grad.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <3 x float> %d, <3 x float> <float 0.000000e+00, float 2.500000e-01, float 0.000000e+00>, <3 x float> <float 0.000000e+00, float 0.000000e+00, float 3.125000e-02>, float 0.000000e+00, i32 0)
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  %p = getelementptr <4 x float>, ptr addrspace(1) %o, i32 %gid
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}}

declare {{ <4 x float>, i8 }} @air.sample_texture_cube_grad.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, <3 x float>, <3 x float>, float, i32)

!air.kernel = !{{!0}}
!air.sampler_states = !{{!6}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texturecube<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"o"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!6 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
"#
    )
}

/// `moltenvk-m3` whose kernel cube-gradient route its description does not state (a MoltenVK
/// release no measurement covers).
fn unmeasured_moltenvk() -> DeviceProfile {
    let desc = crate::pipeline::testing::description("moltenvk-m3").expect("moltenvk-m3");
    let route = crate::error::DriverDeviation::CubeGradientRoute;
    crate::device::Device::with_deviation(
        desc,
        crate::device::DevicePolicy::default(),
        route,
        false,
    )
    .expect("device")
    .profile()
    .clone()
}

/// A cube's explicit gradients through a constexpr sampler of `maxAnisotropy` 8 under a linear
/// mip filter take Metal's footprint in the shader on every device without `samplerAnisotropy`:
/// Metal's level and taps (no gradient sample, the level count queried, tap 0 and the loop's tap
/// of one explicit level each, `AnisotropicFilter` recorded and `CubeGradientFootprint` not).
/// Through an isotropic sampler a device without the feature keeps its own gradient sample, at the
/// footprint the cube's form hands it. On a measured route (`moltenvk-m3`, whose kernels hand
/// Metal's own gradients) the device's gradient sample stays, unrecorded; on MoltenVK where its
/// kernel route is not measured it stays too, Vulkan's rule's form, recorded as
/// `CubeGradientFootprint`.
#[test]
fn a_cubes_explicit_gradients_take_metals_level_and_taps() {
    const GRAD: u32 = 0x4;
    for (aniso, taps) in [(8, true), (1, false)] {
        let what = format!("maxAnisotropy {aniso}");
        let Some(bc) = assemble(
            &what,
            &constexpr_cube_kernel(words(MipFilter::Linear, aniso)),
        ) else {
            return;
        };
        let explicit = if taps { WRITTEN_TAPS as usize } else { 2 };
        let mut shader = Vec::new();
        let mut device = vec![
            ("moltenvk-m3", full("moltenvk-m3")),
            ("unmeasured MoltenVK", unmeasured_moltenvk()),
        ];
        for stem in ["lavapipe", "amd-rdna3-radv", "intel-irisxe-anv"] {
            match taps {
                true => shader.push((stem, p(stem))),
                false => device.push((stem, p(stem))),
            }
        }
        for (stem, profile) in shader {
            let what = format!("{what}, {stem}");
            let a = ok(&bc, "k", &profile);
            assert_eq!(count(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD), explicit, "{what}");
            assert_eq!(count(&a, OP_IMAGE_QUERY_LEVELS), 1, "{what}");
            assert_eq!(dont_unroll_loops(&a), usize::from(taps), "{what}");
            assert_eq!(
                a.precision.contains(&PrecisionNote::AnisotropicFilter),
                taps,
                "{what}: {:?}",
                a.precision
            );
            for note in [
                PrecisionNote::CubeGradientFootprint,
                PrecisionNote::DeviceAnisotropicFilter,
            ] {
                assert!(!a.precision.contains(&note), "{what}: {:?}", a.precision);
            }
        }
        for (stem, profile) in device {
            let what = format!("{what}, {stem}");
            let a = ok(&bc, "k", &profile);
            assert_eq!(count(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD), 1, "{what}");
            assert_eq!(count(&a, OP_IMAGE_QUERY_LEVELS), 0, "{what}");
            assert_eq!(gradient_masks(&a), vec![GRAD], "{what}");
            assert!(
                !a.precision.contains(&PrecisionNote::AnisotropicFilter),
                "{what}: {:?}",
                a.precision
            );
            // Metal's own sampler on a measured route takes Metal's own gradients; on an
            // unmeasured one, Vulkan's rule's form, the device's reading.
            match stem {
                "moltenvk-m3" => assert!(
                    !a.precision.contains(&PrecisionNote::CubeGradientFootprint),
                    "{what}: {:?}",
                    a.precision
                ),
                "unmeasured MoltenVK" => assert!(
                    a.precision.contains(&PrecisionNote::CubeGradientFootprint),
                    "{what}: {:?}",
                    a.precision
                ),
                _ => {}
            }
        }
    }
}

/// The `Grad` and `Lod` bits of each `OpImageSampleExplicitLod`'s image operands in `a`'s module.
fn gradient_masks(a: &FunctionArtifact) -> Vec<u32> {
    let words = &a.module.words;
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let len = ((words[i] >> 16) as usize).max(1);
        if (words[i] & 0xffff) as u16 == OP_IMAGE_SAMPLE_EXPLICIT_LOD && len > 5 {
            out.push(words[i + 5] & 0x6);
        }
        i += len;
    }
    out
}

/// A fragment function summing `n` implicit-level samples of `texture2d<float>` slot 0 through
/// runtime sampler slot 0, each at its position scaled by its own power of two, and where
/// `offsets` at its own constant texel offset (`(i + 1, -i)` for the `i`-th).
fn sampling_fragment(n: usize, offsets: bool) -> String {
    let mut body = String::new();
    let mut acc = "zeroinitializer".to_string();
    for i in 0..n {
        let scale = 1.0 / f64::from(1u32 << (4 + i));
        let offset = match offsets {
            true => format!("<2 x i32> <i32 {}, i32 {}>", i + 1, -(i as i32)),
            false => "<2 x i32> zeroinitializer".to_string(),
        };
        body.push_str(&format!(
            "  %uv{i} = fmul <2 x float> %xy, <float {scale:e}, float {scale:e}>\n  \
             %r{i} = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, \
             ptr addrspace(2) %s, <2 x float> %uv{i}, i1 true, {offset}, i1 false, \
             float 0.000000e+00, float 0.000000e+00, i32 0)\n  \
             %v{i} = extractvalue {{ <4 x float>, i8 }} %r{i}, 0\n  \
             %a{i} = fadd <4 x float> {acc}, %v{i}\n"
        ));
        acc = format!("%a{i}");
    }
    format!(
        r#"define <4 x float> @k(<4 x float> %pos, ptr addrspace(1) %t, ptr addrspace(2) %s) {{
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
{body}  ret <4 x float> {acc}
}}

declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.fragment = !{{!0}}
!0 = !{{ptr @k, !1, !3}}
!1 = !{{!2}}
!2 = !{{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}}
!3 = !{{!4, !5, !6}}
!4 = !{{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}}
!5 = !{{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!6 = !{{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
"#
    )
}

/// The words a sample's site may spend on the emulation's routine besides the plain sample's:
/// the word's test and branch, the call with its arguments, and the sum's conversion.
const SITE_CALL_WORDS: usize = 256;

/// The emulation of a runtime sampler slot's samples is one routine every sample calls
/// (`routine::Filter`): under the variant naming the slot, a fragment function of 1, 2 and 4 such
/// samples holds one emulation's taps (one explicit-level sample each, past the base's) and its
/// one loop, and each sample past the first adds no more than the plain sample's words and a
/// call's. Written at every site, each sample added about 2,100 words more than the plain one's.
#[test]
fn anisotropic_samples_call_one_routine() {
    let inputs = fragment_variant();
    for stem in ["lavapipe", "intel-irisxe-anv", "nvidia-rtx5080-laptop"] {
        let p = p(stem);
        let mut words = Vec::new();
        for n in [1usize, 2, 4] {
            let what = format!("{stem}, {n} samples");
            let Some(bc) = assemble(&what, &sampling_fragment(n, false)) else {
                return;
            };
            let base = ok(&bc, "k", &p);
            let a = ok_with(&bc, &inputs, &p);
            assert_eq!(
                count(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD),
                count(&base, OP_IMAGE_SAMPLE_EXPLICIT_LOD) + WRITTEN_TAPS as usize,
                "{what}"
            );
            assert_eq!(dont_unroll_loops(&a), 1, "{what}");
            words.push((n, base.module.words.len(), a.module.words.len()));
        }
        let (n0, base0, variant0) = words[0];
        for &(n, base, variant) in &words[1..] {
            let added = n - n0;
            assert!(
                variant - variant0 <= base - base0 + added * SITE_CALL_WORDS,
                "{stem}: {n} samples: the variant grew {} words from {n0}, the base {}",
                variant - variant0,
                base - base0
            );
        }
    }
}

/// The fragment of [`sampling_fragment`]'s runtime sampler slot (`LoweringInputs::anisotropic_samplers`).
fn fragment_variant() -> LoweringInputs {
    let mut inputs = LoweringInputs::new();
    inputs.anisotropic_samplers.insert(SlotRef {
        stage: Stage::Fragment,
        class: SlotClass::Sampler,
        index: 0,
    });
    inputs
}

/// The samples of one binding at different constant texel offsets call one routine where the
/// profile admits the `Offset` operand on a sample (`maintenance8` and
/// `shaderImageGatherExtended`): the call passes the offset, which the taps' samples take as that
/// operand, so a fragment of 1, 2 and 4 samples at their own offsets holds one emulation's taps and
/// its one loop, no tap sample takes `ConstOffset`, and each sample past the first adds no more
/// than the plain sample's words and a call's. Without the operand (`vk12-floor`) each offset is
/// its own routine's `ConstOffset`: a loop per sample.
#[test]
fn anisotropic_samples_at_every_offset_call_one_routine() {
    let inputs = fragment_variant();
    for (stem, passes) in [
        ("lavapipe", true),
        ("intel-irisxe-anv", true),
        ("nvidia-rtx5080-laptop", true),
        ("vk12-floor", false),
    ] {
        let p = p(stem);
        let images = &p.fields().images;
        assert_eq!(
            images.dynamic_sample_offset && images.gather_extended,
            passes,
            "{stem}"
        );
        let mut words = Vec::new();
        for n in [1usize, 2, 4] {
            let what = format!("{stem}, {n} samples at their offsets");
            let Some(bc) = assemble(&what, &sampling_fragment(n, true)) else {
                return;
            };
            let base = ok(&bc, "k", &p);
            let a = ok_with(&bc, &inputs, &p);
            let routines = if passes { 1 } else { n };
            assert_eq!(dont_unroll_loops(&a), routines, "{what}");
            assert_eq!(
                count(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD),
                count(&base, OP_IMAGE_SAMPLE_EXPLICIT_LOD) + routines * WRITTEN_TAPS as usize,
                "{what}"
            );
            let taps = if passes { 0 } else { n * WRITTEN_TAPS as usize };
            assert_eq!(
                const_offset_samples(&a),
                const_offset_samples(&base) + taps,
                "{what}: the taps' samples take ConstOffset only without the Offset operand"
            );
            words.push((n, base.module.words.len(), a.module.words.len()));
        }
        if passes {
            let (n0, base0, variant0) = words[0];
            for &(n, base, variant) in &words[1..] {
                let added = n - n0;
                assert!(
                    variant - variant0 <= base - base0 + added * SITE_CALL_WORDS,
                    "{stem}: {n} samples: the variant grew {} words from {n0}, the base {}",
                    variant - variant0,
                    base - base0
                );
            }
        }
    }
}

/// The `ConstOffset` bit of an image operands mask.
const CONST_OFFSET: u32 = 0x8;

/// The explicit-level samples of `a`'s module that take the `ConstOffset` image operand.
fn const_offset_samples(a: &FunctionArtifact) -> usize {
    let words = &a.module.words;
    let mut found = 0;
    let mut i = 5;
    while i < words.len() {
        let len = ((words[i] >> 16) as usize).max(1);
        // The operands' mask follows the result type, result, sampled image and coordinate.
        found += usize::from(
            (words[i] & 0xffff) as u16 == OP_IMAGE_SAMPLE_EXPLICIT_LOD
                && len > 5
                && words[i + 5] & CONST_OFFSET != 0,
        );
        i += len;
    }
    found
}
