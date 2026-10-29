//! A fragment's sample bias past the device's `maxSamplerLodBias`.
//!
//! Vulkan clamps an implicit sample's `Bias` operand (with the sampler's own `mipLodBias`) into
//! `±maxSamplerLodBias`; Metal applies the whole bias: an Apple M3 Max magnifies with biases of
//! -17.5, -100, and -1000 at a level of 17 before the bias (authored case `3ec31bc4`,
//! `a2_fragment_bias_clamp`), where RADV, clamping at 16, minifies. Through a sampler without a
//! bias of its own every bias is the `Bias` operand, the device's sampler's answer (MoltenVK hands
//! it to Metal whole). Through a constexpr sampler with its own bias, whose sum with the operand's
//! no measurement has read:
//!
//! - a constant bias whose sum with the sampler's is within the limit stays the `Bias` operand of
//!   an implicit sample, unchanged;
//! - a constant past it samples with the coordinate's derivatives scaled by `2^bias` alone;
//! - a dynamic bias takes both from the sample's own position: the implicit sample with `Bias`,
//!   and, only where `|bias + sampler| > maxSamplerLodBias`, a branch sampling with the scaled
//!   derivatives;
//! - The hoisted derivatives and the explicit level take the whole bias too.
//!
//! The scaling never overflows (`bias_gradients`): the pair is normalized by its largest
//! component's exponent (`Frexp`, `Ldexp`) and scaled by `Exp2` of the exponent sum clamped into
//! `[-150, 126]`, so a huge positive bias saturates at a finite level past every texture's top
//! level and a huge negative one at zero gradients (magnified), never `0 · ∞`. The interpreter
//! does not model `Exp2` (the device's precision), so these are shape tests on the emitted
//! SPIR-V, every module validated by `spirv-val` under its profile (`G-val`).
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::DevicePolicy;
use crate::library::FunctionArtifact;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, profile, AssembleError,
};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// `a2_fragment_bias_clamp`'s constexpr sampler: magnify nearest, minify linear, repeat.
const SAMPLER_WORDS: u64 = 34901797601020050;

const OP_CONSTANT: u16 = 43;
const OP_EXT_INST: u16 = 12;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
const OP_IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
const OP_F_ORD_GREATER_THAN: u16 = 186;
const OP_DPDX: u16 = 207;
const OP_DPDY: u16 = 208;
const GLSL_EXP2: u32 = 29;
const GLSL_FCLAMP: u32 = 43;
const GLSL_FREXP_STRUCT: u32 = 52;
const GLSL_LDEXP: u32 = 53;
const GLSL_LENGTH: u32 = 66;

fn assemble_file(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(name)) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-bias-reach-{}-{}-{name}",
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

/// A fragment function sampling `texture2d<float>` slot 0 at its position over 16 through a
/// constexpr sampler of state words `[SAMPLER_WORDS, half_bias]` (`half_bias` the sampler's own
/// LOD bias as `half` bits) with shader bias `bias` (an LLVM float literal).
fn biased_fragment(bias: &str, half_bias: u16) -> String {
    biased_fragment_of(SAMPLER_WORDS, bias, half_bias)
}

/// [`biased_fragment`] through a constexpr sampler of first state word `words`.
fn biased_fragment_of(words: u64, bias: &str, half_bias: u16) -> String {
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 {half_bias}], align 8

define <4 x float> @k(<4 x float> %pos, ptr addrspace(1) %t) {{
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %uv = fmul <2 x float> %xy, <float 6.250000e-02, float 6.250000e-02>
  %r = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float {bias}, float 0.000000e+00, i32 0)
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

fn p(stem: &str) -> DeviceProfile {
    profile(stem, DevicePolicy::default()).expect(stem)
}

fn max_bias(p: &DeviceProfile) -> f32 {
    f32::from_bits(p.fields().limits.max_sampler_lod_bias_bits)
}

fn ok(bc: &[u8], entry: &str, p: &DeviceProfile) -> std::sync::Arc<FunctionArtifact> {
    let a = compile(bc, entry, p).unwrap_or_else(|e| panic!("{entry}: compile: {e}"));
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{entry}: L3: {e}"));
    a
}

/// [`ok`] under lowering inputs `inputs`.
fn ok_with(
    bc: &[u8],
    entry: &str,
    p: &DeviceProfile,
    inputs: &crate::pipeline::LoweringInputs,
) -> std::sync::Arc<FunctionArtifact> {
    let a = crate::pipeline::testing::compile_with(bc, entry, p, inputs)
        .unwrap_or_else(|e| panic!("{entry}: compile: {e}"));
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{entry}: L3: {e}"));
    a
}

/// The variant whose fragment function takes Metal's anisotropic filter, emulated, through
/// runtime sampler slot 0 (`LoweringInputs::anisotropic_samplers`): a draw binding an
/// anisotropic sampler there runs on it.
fn anisotropic_slot_0() -> crate::pipeline::LoweringInputs {
    use crate::pipeline::sym::{SlotClass, SlotRef, Stage};
    let mut inputs = crate::pipeline::LoweringInputs::new();
    inputs.anisotropic_samplers.insert(SlotRef {
        stage: Stage::Fragment,
        class: SlotClass::Sampler,
        index: 0,
    });
    inputs
}

/// One instruction: its opcode and its words after the first.
struct Inst {
    op: u16,
    operands: Vec<u32>,
}

fn insts(words: &[u32]) -> Vec<Inst> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = ((words[i] >> 16) as usize).max(1);
        out.push(Inst {
            op: (words[i] & 0xFFFF) as u16,
            operands: words[i + 1..(i + n).min(words.len())].to_vec(),
        });
        i += n;
    }
    out
}

/// The module's facts the tests read: opcode counts, the 32-bit constants by id, and each
/// GLSL.std.450 instruction as (result id, instruction, operand ids).
struct Shape {
    ops: BTreeMap<u16, usize>,
    constants: BTreeMap<u32, u32>,
    glsl: Vec<(u32, u32, Vec<u32>)>,
    /// `OpFOrdGreaterThan`'s operand ids.
    greater: Vec<(u32, u32)>,
}

fn shape(a: &FunctionArtifact) -> Shape {
    let mut s = Shape {
        ops: BTreeMap::new(),
        constants: BTreeMap::new(),
        glsl: Vec::new(),
        greater: Vec::new(),
    };
    for i in insts(&a.module.words) {
        *s.ops.entry(i.op).or_default() += 1;
        match (i.op, i.operands.as_slice()) {
            (OP_CONSTANT, [_, id, value]) => {
                s.constants.insert(*id, *value);
            }
            (OP_EXT_INST, [_, id, _, inst, args @ ..]) => s.glsl.push((*id, *inst, args.to_vec())),
            (OP_F_ORD_GREATER_THAN, [_, _, x, y]) => s.greater.push((*x, *y)),
            _ => {}
        }
    }
    s
}

impl Shape {
    fn count(&self, op: u16) -> usize {
        self.ops.get(&op).copied().unwrap_or(0)
    }

    fn glsl_count(&self, inst: u32) -> usize {
        self.glsl.iter().filter(|(_, i, _)| *i == inst).count()
    }

    fn constant(&self, id: u32) -> Option<f32> {
        self.constants.get(&id).map(|bits| f32::from_bits(*bits))
    }

    /// The `FClamp`s between constants, with their bounds.
    fn constant_clamps(&self) -> Vec<(u32, (Option<f32>, Option<f32>))> {
        self.glsl
            .iter()
            .filter(|(_, i, a)| {
                *i == GLSL_FCLAMP
                    && (self.constants.contains_key(&a[1]) || self.constants.contains_key(&a[2]))
            })
            .map(|(id, _, a)| (*id, (self.constant(a[1]), self.constant(a[2]))))
            .collect()
    }

    /// `bias_gradients`' shape, `n` times: `Frexp` of the largest magnitude, the pair's two
    /// `Ldexp`s, and `Exp2` of exactly an `FClamp` into `[-150, 126]` (`exp2` more `Exp2`s are
    /// the axis gradients'); and no other `FClamp` between constants but Metal's reading of the
    /// bias ([`Shape::reads_metals_bias`]), so nothing clamps the bias to `maxSamplerLodBias`.
    /// (A runtime sampler's gather clamps its block index between bounds the texture's size
    /// gives, which are not constants.)
    fn scales_finitely(&self, n: usize, exp2: usize, what: &str) {
        self.scales_finitely_beside(n, exp2, 0, what);
    }

    /// [`Shape::scales_finitely`] beside `emulations` of Metal's anisotropic filter, each of whose
    /// spacings takes two `Exp2`s of its own (the log ratio's power and the level's scale,
    /// `anisotropic::footprint`) and no `Ldexp`.
    fn scales_finitely_beside(&self, n: usize, exp2: usize, emulations: usize, what: &str) {
        assert_eq!(self.glsl_count(GLSL_FREXP_STRUCT), n, "{what}: Frexp");
        assert_eq!(self.glsl_count(GLSL_LDEXP), 2 * n, "{what}: Ldexp");
        assert_eq!(
            self.glsl_count(GLSL_EXP2),
            n + exp2 + 2 * emulations,
            "{what}: Exp2"
        );
        let clamps = self.constant_clamps();
        let exponent: Vec<u32> = clamps
            .iter()
            .filter(|(_, bounds)| *bounds == (Some(-150.0), Some(126.0)))
            .map(|(id, _)| *id)
            .collect();
        assert_eq!(exponent.len(), n, "{what}: exponent clamps");
        for id in exponent {
            assert!(
                self.glsl
                    .iter()
                    .any(|(_, i, a)| *i == GLSL_EXP2 && a.as_slice() == [id]),
                "{what}: Exp2 of the clamped exponent"
            );
        }
        for (_, bounds) in clamps {
            // An emulation's level is held in `[0, last level]`, its upper bound queried.
            let level = emulations > 0 && bounds == (Some(0.0), None);
            assert!(
                level
                    || [
                        (Some(-150.0), Some(126.0)),
                        (Some(-64.0), Some(64.0)),
                        (Some(-32.0), Some(31.984375)),
                    ]
                    .contains(&bounds),
                "{what}: FClamp bounds {bounds:?}"
            );
        }
    }

    /// Metal's reading of a bias (`emit::metal_bias`), `n` times: its clamp into `±64` before the
    /// rounding and into Metal's range `[-32, 31.984375]` after the floor.
    fn reads_metals_bias(&self, n: usize, what: &str) {
        let clamps = self.constant_clamps();
        let count = |b: (f32, f32)| {
            clamps
                .iter()
                .filter(|(_, x)| *x == (Some(b.0), Some(b.1)))
                .count()
        };
        assert_eq!(count((-64.0, 64.0)), n, "{what}: the clamp into ±64");
        assert_eq!(
            count((-32.0, 31.984375)),
            n,
            "{what}: the clamp into Metal's range"
        );
    }
}

/// Through a sampler without a bias of its own every fragment bias is the `Bias` operand alone, the
/// device's sampler's: on MoltenVK (`VkProperty::MetalSampler`) the device hands it to Metal whole,
/// so Metal applies its own rule (the bias rounded to `half`, floored onto `1/64` and clamped into
/// `[-32, 31.984375]`); elsewhere the device clamps it into `±maxSamplerLodBias`
/// (`VkProperty::DeviceSampling`). A dynamic bias and a constant past the limit alike take no
/// derivatives and no branch, record no `PrecisionNote::GradFromBias`, and require the property.
#[test]
fn a_bias_without_the_samplers_own_is_the_bias_operand() {
    use crate::device::requirements::VkProperty;
    let mut fixtures = Vec::new();
    if let Some(bc) = assemble_file("a2_fragment_bias_clamp.ll") {
        fixtures.push(("dynamic", bc, "a2_fragment_bias_clamp"));
    }
    for (what, bias) in [
        ("constant past", "-2.000000e+01"),
        ("constant, huge", "1.000000e+03"),
    ] {
        if let Some(bc) = assemble(what, &biased_fragment(bias, 0)) {
            fixtures.push((what, bc, "k"));
        }
    }
    for stem in ["moltenvk-m3", "amd-rdna3-radv", "lavapipe", "vk12-floor"] {
        let prof = p(stem);
        let property = match prof
            .fields()
            .enabled
            .properties
            .contains(&VkProperty::MetalSampler)
        {
            true => VkProperty::MetalSampler,
            false => VkProperty::DeviceSampling,
        };
        for (what, bc, entry) in &fixtures {
            let what = format!("{stem}, {what}");
            let a = ok(bc, entry, &prof);
            let s = shape(&a);
            let uses = |name: &str| a.strategies.entries.get(name).map_or(0, |u| u.uses);
            assert_eq!(
                uses("sampler.sample_implicit"),
                1,
                "{what}: {:?}",
                a.strategies
            );
            assert_eq!(uses("sampler.sample_implicit.bias_gradients"), 0, "{what}");
            assert_eq!(uses("sampler.sample_bias_gradients"), 0, "{what}");
            assert_eq!(s.count(OP_IMAGE_SAMPLE_IMPLICIT_LOD), 1, "{what}");
            assert_eq!(s.count(OP_IMAGE_SAMPLE_EXPLICIT_LOD), 0, "{what}");
            assert_eq!((s.count(OP_DPDX), s.count(OP_DPDY)), (0, 0), "{what}");
            assert_eq!(s.glsl_count(GLSL_EXP2), 0, "{what}");
            s.reads_metals_bias(0, &what);
            assert!(
                !a.precision.contains(&PrecisionNote::GradFromBias),
                "{what}"
            );
            assert!(a.requirements.properties.contains(&property), "{what}");
        }
    }
}

/// Through a constexpr sampler with a bias of its own, a constant bias whose sum with the
/// sampler's is within the limit is the `Bias` operand alone (no derivatives, no note); past it,
/// the scaled derivatives alone. The sampler's own bias counts: on amd-rdna3-radv (limit 16) a
/// constexpr sampler bias of 10 with a shader bias of 4 is within, with 8 or -30 past. (MoltenVK's portability subset withholds a sampler bias, so it refuses such
/// a sampler.)
#[test]
fn a_constant_bias_is_decided_at_compile_time_with_the_samplers_own() {
    let radv = p("amd-rdna3-radv");
    assert_eq!(max_bias(&radv), 16.0);
    // Half 10.0 is 0x4900.
    for (what, bias, half_bias, past) in [
        (
            "within with the sampler's",
            "4.000000e+00",
            0x4900u16,
            false,
        ),
        ("past with the sampler's", "8.000000e+00", 0x4900, true),
        (
            "past below with the sampler's",
            "-3.000000e+01",
            0x4900,
            true,
        ),
    ] {
        let Some(bc) = assemble(what, &biased_fragment(bias, half_bias)) else {
            return;
        };
        let a = ok(&bc, "k", &radv);
        let s = shape(&a);
        let uses = |name: &str| a.strategies.entries.get(name).map_or(0, |u| u.uses);
        if past {
            assert_eq!(uses("sampler.sample_bias_gradients"), 1, "{what}");
            assert_eq!(s.count(OP_IMAGE_SAMPLE_IMPLICIT_LOD), 0, "{what}");
            assert_eq!(s.count(OP_IMAGE_SAMPLE_EXPLICIT_LOD), 1, "{what}");
            assert_eq!((s.count(OP_DPDX), s.count(OP_DPDY)), (1, 1), "{what}");
            s.scales_finitely(1, 0, what);
            s.reads_metals_bias(1, what);
            assert!(a.precision.contains(&PrecisionNote::GradFromBias), "{what}");
        } else {
            s.reads_metals_bias(1, what);
            assert_eq!(uses("sampler.sample_implicit"), 1, "{what}");
            assert_eq!(s.count(OP_IMAGE_SAMPLE_IMPLICIT_LOD), 1, "{what}");
            assert_eq!(s.count(OP_IMAGE_SAMPLE_EXPLICIT_LOD), 0, "{what}");
            assert_eq!((s.count(OP_DPDX), s.count(OP_DPDY)), (0, 0), "{what}");
            assert_eq!(s.glsl_count(GLSL_EXP2), 0, "{what}");
            assert!(
                !a.precision.contains(&PrecisionNote::GradFromBias),
                "{what}"
            );
        }
    }
}

/// The hoisted derivatives (`fragment_mixed_samplers`, a sampler selected per fragment,
/// `bias(0.5)`) are scaled by the whole bias through the same finite scaling: no clamp of the bias
/// to `maxSamplerLodBias` (formerly `FClamp(bias, ±max)` before `Exp2`). Off Metal's own sampler,
/// the runtime slot among the candidates takes them into Metal's anisotropic filter, emulated where
/// its word states anisotropy.
#[test]
fn hoisted_derivatives_take_the_whole_bias() {
    let Some(bc) = assemble_file("sampling_completion.ll") else {
        return;
    };
    for (stem, emulations) in [("moltenvk-m3", 0), ("vk12-floor", 1)] {
        let a = ok_with(
            &bc,
            "fragment_mixed_samplers",
            &p(stem),
            &anisotropic_slot_0(),
        );
        let s = shape(&a);
        assert_eq!(
            a.strategies
                .entries
                .get("sampler.sample_hoisted_grad.switch")
                .map(|u| u.uses),
            Some(1)
        );
        assert_eq!((s.count(OP_DPDX), s.count(OP_DPDY)), (1, 1), "{stem}");
        s.scales_finitely_beside(1, 0, emulations, stem);
        s.reads_metals_bias(1, stem);
        assert_eq!(
            a.specialized
                .iter()
                .any(|n| n.note == PrecisionNote::AnisotropicFilter),
            emulations > 0,
            "{stem}"
        );
    }
}

/// Outside a fragment function the bias is the explicit level unclamped: no `FClamp` at all in
/// `sample_bias_kernel` (two biases and two minimum-LOD clamps, formerly two `FClamp(bias, ±max)`).
#[test]
fn a_base_level_takes_the_whole_bias() {
    let Some(bc) = assemble_file("sampling_completion.ll") else {
        return;
    };
    for stem in ["moltenvk-m3", "vk12-floor"] {
        let a = ok(&bc, "sample_bias_kernel", &p(stem));
        let s = shape(&a);
        assert_eq!(s.glsl_count(GLSL_FCLAMP), 0, "{stem}");
        assert_eq!(s.count(OP_IMAGE_SAMPLE_EXPLICIT_LOD), 3, "{stem}");
    }
}

/// `SAMPLER_WORDS` with a maximum anisotropy of 16 (the state words' field at bit 20 holds the
/// anisotropy less one).
const ANISOTROPIC_WORDS: u64 = (SAMPLER_WORDS & !(0xf << 20)) | (15 << 20);

/// The taps Metal's anisotropic filter, emulated, holds in its code, whatever its degree: tap 0
/// and its loop's one (one explicit-level sample each, the device blending the levels).
const WRITTEN_TAPS: u32 = 2;

/// Through an anisotropic constexpr sampler of a 2D texture with a bias of its own, Metal's bias
/// shifts the footprint's minor axis alone, which neither the `Bias` operand nor `2^B` derivatives
/// give: a fragment bias, within the device's limit or past it, samples with the axis gradients
/// alone (the coordinate's derivatives, their lengths in texels of the level-0 size, and `Exp2` of
/// Metal's reading of the bias), with `PrecisionNote::GradFromBias`. With `samplerAnisotropy` the
/// device's anisotropic filter samples at them; without it they are the footprint of Metal's
/// anisotropic filter, emulated: its taps' samples at explicit levels, one a tap (the device
/// blending the sampler's linear mip filter's two levels), `AnisotropicFilter` recorded.
#[test]
fn an_anisotropic_sampler_takes_the_axis_gradients() {
    for stem in ["amd-rdna3-radv", "lavapipe"] {
        for withheld in [false, true] {
            let prof = {
                let mut desc = crate::pipeline::testing::description(stem).expect(stem);
                if withheld {
                    assert!(desc
                        .features
                        .remove(&crate::device::requirements::VkFeature::SamplerAnisotropy));
                }
                crate::device::Device::new(desc, DevicePolicy::default())
                    .expect(stem)
                    .profile()
                    .clone()
            };
            // Half 10.0 is 0x4900: the sampler's own bias.
            for (what, bias) in [("within", "2.000000e+00"), ("past", "-3.000000e+01")] {
                let what = format!("{stem}, samplerAnisotropy withheld {withheld}, {what}");
                let Some(bc) =
                    assemble(&what, &biased_fragment_of(ANISOTROPIC_WORDS, bias, 0x4900))
                else {
                    return;
                };
                let a = ok(&bc, "k", &prof);
                let s = shape(&a);
                let uses = |name: &str| a.strategies.entries.get(name).map_or(0, |u| u.uses);
                assert_eq!(uses("sampler.sample_bias_gradients"), 1, "{what}");
                assert_eq!(s.count(OP_IMAGE_SAMPLE_IMPLICIT_LOD), 0, "{what}");
                let explicit = match withheld {
                    true => WRITTEN_TAPS as usize,
                    false => 1,
                };
                assert_eq!(s.count(OP_IMAGE_SAMPLE_EXPLICIT_LOD), explicit, "{what}");
                assert_eq!((s.count(OP_DPDX), s.count(OP_DPDY)), (1, 1), "{what}");
                assert_eq!(s.glsl_count(GLSL_LENGTH), 2, "{what}");
                s.reads_metals_bias(1, &what);
                assert!(a.precision.contains(&PrecisionNote::GradFromBias), "{what}");
                assert_eq!(
                    a.precision.contains(&PrecisionNote::AnisotropicFilter),
                    withheld,
                    "{what}"
                );
            }
        }
    }
}
