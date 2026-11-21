//! The device's forms of Metal's maths: on every device the maths functions, divisions and fused multiply-adds (`NativeTranscendentals`,
//! `NativeDivision`, `NativeFma`) and `dot`, `mix` and `fast::sqrt` (`NativeDot`, `NativeMix`,
//! `NativeTranscendentals`) are the device's instructions, each recording its kinds; and on every
//! device not measured to answer Metal's, the NaN and the minimum and maximum are the device's
//! (`NativeNanEncoding`, `NativeMinMax`).
//!
//! The kernels read three `float`s from buffer 0 and store one result after them: a precise
//! `sin`, `exp`, `atan2` and `pow`, a precise and an `arcp` division, a `half` `fma`, an addition,
//! an `fmin`, a `saturate`, a `fast::sqrt`, a `dot` and a `float` and a `half` `mix`. On lavapipe
//! each records its kind; the interpreter's answers stay within the bound the kind's docs state.
//! `dot`, `mix` and `fast::sqrt` are the device's instructions on MoltenVK too, each noted.
//! `llvm-as` assembles the module; without it the tests skip visibly.

use crate::air::{self, SpecRequest};
use crate::device::profile::DeviceProfile;
use crate::device::DevicePolicy;
use crate::lower::emit::emit_module;
use crate::lower::select::{select, LowerInputs};
use crate::lower::simd::WidestPin;
use crate::memory::static_root::analyze_memory;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, meter, profile, AssembleError, Scratch,
};
use crate::pipeline::PrecisionNote;
use crate::spirv::interp::{run, Buffer, Config, Resources};

const OP_FUNCTION: u32 = 54;

/// A kernel: its name, the body defining `%r` from `%a`, `%b` and `%c`, and the note its device
/// form records.
struct Kernel {
    name: &'static str,
    body: &'static str,
    note: PrecisionNote,
}

const KERNELS: [Kernel; 14] = [
    Kernel {
        name: "native_sin",
        body: "  %r = tail call float @air.sin.f32(float %a)\n",
        note: PrecisionNote::NativeTranscendentals,
    },
    Kernel {
        name: "native_exp",
        body: "  %r = tail call float @air.exp.f32(float %a)\n",
        note: PrecisionNote::NativeTranscendentals,
    },
    Kernel {
        name: "native_atan2",
        body: "  %r = tail call float @air.atan2.f32(float %a, float %b)\n",
        note: PrecisionNote::NativeTranscendentals,
    },
    Kernel {
        name: "native_pow",
        body: "  %r = tail call float @air.pow.f32(float %a, float %b)\n",
        note: PrecisionNote::NativeTranscendentals,
    },
    Kernel {
        name: "native_fdiv",
        body: "  %r = fdiv float %a, %b\n",
        note: PrecisionNote::NativeDivision,
    },
    Kernel {
        name: "native_fdiv_arcp",
        body: "  %r = fdiv arcp float %a, %b\n",
        note: PrecisionNote::NativeDivision,
    },
    Kernel {
        name: "native_half_fma",
        body: "  %ha = fptrunc float %a to half\n  %hb = fptrunc float %b to half\n  %hc = fptrunc float %c to half\n  %hr = tail call half @air.fma.f16(half %ha, half %hb, half %hc)\n  %r = fpext half %hr to float\n",
        note: PrecisionNote::NativeFma,
    },
    Kernel {
        name: "native_fadd",
        body: "  %r = fadd float %a, %b\n",
        note: PrecisionNote::NativeNanEncoding,
    },
    Kernel {
        name: "native_fmin",
        body: "  %r = tail call float @air.fmin.f32(float %a, float %b)\n",
        note: PrecisionNote::NativeMinMax,
    },
    Kernel {
        name: "native_fast_sqrt",
        body: "  %r = tail call fast float @air.fast_sqrt.f32(float %a)\n",
        note: PrecisionNote::NativeTranscendentals,
    },
    Kernel {
        name: "native_dot",
        body: "  %u0 = insertelement <2 x float> undef, float %a, i32 0\n  %u = insertelement <2 x float> %u0, float %b, i32 1\n  %v0 = insertelement <2 x float> undef, float %c, i32 0\n  %v = insertelement <2 x float> %v0, float %a, i32 1\n  %r = tail call float @air.dot.v2f32(<2 x float> %u, <2 x float> %v)\n",
        note: PrecisionNote::NativeDot,
    },
    Kernel {
        name: "native_mix",
        body: "  %r = tail call float @air.mix.f32(float %a, float %b, float %c)\n",
        note: PrecisionNote::NativeMix,
    },
    Kernel {
        name: "native_saturate",
        body: "  %r = tail call float @air.saturate.f32(float %a)\n",
        note: PrecisionNote::NativeMinMax,
    },
    Kernel {
        name: "native_half_mix",
        body: "  %ha = fptrunc float %a to half\n  %hb = fptrunc float %b to half\n  %hc = fptrunc float %c to half\n  %hr = tail call half @air.mix.f16(half %ha, half %hb, half %hc)\n  %r = fpext half %hr to float\n",
        note: PrecisionNote::NativeMix,
    },
];

/// The module of every kernel.
fn module() -> String {
    let mut s = String::from(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n",
    );
    for k in &KERNELS {
        s.push_str(&format!(
            "define void @{}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{\nentry:\n  %p1 = getelementptr inbounds float, ptr addrspace(1) %0, i64 1\n  %p2 = getelementptr inbounds float, ptr addrspace(1) %0, i64 2\n  %p3 = getelementptr inbounds float, ptr addrspace(1) %0, i64 3\n  %a = load float, ptr addrspace(1) %0, align 4\n  %b = load float, ptr addrspace(1) %p1, align 4\n  %c = load float, ptr addrspace(1) %p2, align 4\n{}  store float %r, ptr addrspace(1) %p3, align 4\n  ret void\n}}\n\n",
            k.name, k.body
        ));
    }
    s.push_str("declare float @air.sin.f32(float)\ndeclare float @air.exp.f32(float)\ndeclare float @air.atan2.f32(float, float)\ndeclare float @air.pow.f32(float, float)\ndeclare half @air.fma.f16(half, half, half)\ndeclare float @air.fmin.f32(float, float)\ndeclare float @air.dot.v2f32(<2 x float>, <2 x float>)\ndeclare float @air.fast_sqrt.f32(float)\ndeclare float @air.mix.f32(float, float, float)\ndeclare half @air.mix.f16(half, half, half)\ndeclare float @air.saturate.f32(float)\n");
    let refs: Vec<String> = (0..KERNELS.len()).map(|i| format!("!{}", 20 + i)).collect();
    s.push_str(&format!(
        "\n!llvm.module.flags = !{{!0}}\n!air.kernel = !{{{}}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n",
        refs.join(", ")
    ));
    s.push_str("!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n!10 = !{}\n!11 = !{!12}\n!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"io\"}\n!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n");
    for (i, k) in KERNELS.iter().enumerate() {
        s.push_str(&format!("!{} = !{{ptr @{}, !10, !11}}\n", 20 + i, k.name));
    }
    s
}

/// The module's bitcode, or `None` (skipped visibly) without `llvm-as`.
fn bitcode() -> Option<Vec<u8>> {
    let scratch = Scratch::new("m2v-native-maths");
    let dir = scratch.path();
    let ll = dir.join("native.ll");
    std::fs::write(&ll, module()).expect("write");
    let assembled = llvm_assemble(&ll);
    match assembled {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected the module: {e}"),
    }
}

/// `entry` on `p` on the interpreter with inputs `x`: the stored result.
fn interpret(bitcode: &[u8], entry: &str, p: &DeviceProfile, x: [f32; 3]) -> f32 {
    let mut m = meter();
    let decoded = air::decode(bitcode, &mut m).expect("decode");
    let req = SpecRequest {
        call_stack_depth: 1,
        entry: entry.to_owned(),
        ..SpecRequest::default()
    };
    let spec = air::specialize(decoded, &req, &mut m).expect("specialize");
    let memory = analyze_memory(&spec, p, &mut m).expect("memory");
    let plan = select(
        &spec,
        &memory,
        p,
        (&LowerInputs::new(), WidestPin::Weighed),
        &mut m,
    )
    .expect("select");
    let module = emit_module(&spec, &plan, &mut m).expect("emit").module;
    let bytes = [x[0], x[1], x[2], 0.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes,
        }],
        push: vec![0; plan.push.push_range.map_or(4, |(_, end)| end as usize)],
        spec: Default::default(),
    };
    let out = run(&module, &Config::default(), res)
        .unwrap_or_else(|e| panic!("{entry}: interpreter: {e}"));
    let b = &out.buffers[0];
    f32::from_le_bytes([b[12], b[13], b[14], b[15]])
}

/// The functions a module defines, its entry included.
fn functions(words: &[u32]) -> usize {
    let mut i = 5;
    let mut n = 0;
    while let Some(&w) = words.get(i) {
        n += usize::from(w & 0xFFFF == OP_FUNCTION);
        i += (w >> 16).max(1) as usize;
    }
    n
}

/// The units in the last place of `want` (an `f32`) that `got` lies from it.
fn ulps(got: f32, want: f64) -> f64 {
    let w = want as f32;
    let ulp = f64::from(f32::from_bits(w.abs().to_bits() + 1) - w.abs());
    (f64::from(got) - want).abs() / ulp
}

/// On lavapipe each kernel records its kind, passes L3, and is its entry alone (no routine).
#[test]
fn each_kernel_records_its_kind() {
    let Some(bc) = bitcode() else {
        return;
    };
    let p = profile("lavapipe", DevicePolicy::RESIDENT_TABLES).expect("profile");
    for k in &KERNELS {
        let a = compile(&bc, k.name, &p).unwrap_or_else(|e| panic!("{}: {e:?}", k.name));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{}: L3: {e}", k.name));
        assert!(
            a.precision.contains(&k.note),
            "{}: {:?}",
            k.name,
            a.precision
        );
        assert_eq!(functions(&a.module.words), 1, "{}: no routine", k.name);
    }
}

/// On lavapipe the maths functions' device forms compute within the bounds their notes state on the
/// interpreter (which models the divisions and the `half` `fma`, not the transcendental
/// instructions).
#[test]
fn the_maths_functions_are_within_their_bounds() {
    let Some(bc) = bitcode() else {
        return;
    };
    let p = profile("lavapipe", DevicePolicy::RESIDENT_TABLES).expect("lavapipe");
    for (a, b) in [(1.0f32, 3.0f32), (-3.0, 0.7), (0.75, -4.1), (1e-30, 3e7)] {
        for div in ["native_fdiv", "native_fdiv_arcp"] {
            let q = interpret(&bc, div, &p, [a, b, 0.0]);
            assert!(
                ulps(q, f64::from(a) / f64::from(b)) <= 2.5,
                "{div}({a}, {b})"
            );
        }
    }
    // A precise division's divisor past `2^126` is scaled into Vulkan's range.
    let big = 2f32.powi(127);
    assert_eq!(interpret(&bc, "native_fdiv", &p, [big, big, 0.0]), 1.0);
    // `half` `fma(1.5, 2.25, 0.125)` is exact: 3.5.
    assert_eq!(
        interpret(&bc, "native_half_fma", &p, [1.5, 2.25, 0.125]),
        3.5
    );
}

/// On lavapipe an addition answers the device's NaN (a NaN, here the operand's payload), and so do
/// the maths functions' device divisions. `fmin` is the device's `NMin` and `saturate` its
/// `NClamp(x, 0, 1)`, answering the operand Metal does beside a NaN and between two numbers.
#[test]
fn the_nan_and_min_max_are_the_device_s() {
    let Some(bc) = bitcode() else {
        return;
    };
    let p = profile("lavapipe", DevicePolicy::RESIDENT_TABLES).expect("lavapipe");
    let nan = f32::from_bits(0x7fc0_1234);
    for name in ["native_fadd", "native_fdiv", "native_fdiv_arcp"] {
        assert!(interpret(&bc, name, &p, [nan, 1.0, 0.0]).is_nan(), "{name}");
    }
    let min = |a: f32, b: f32| interpret(&bc, "native_fmin", &p, [a, b, 0.0]);
    assert_eq!(min(nan, 2.0), 2.0);
    assert_eq!(min(2.0, nan), 2.0);
    assert_eq!(min(1.0, -3.0), -3.0);
    // `saturate` is Metal's `clamp(x, +0, 1)`, the device's `NClamp`.
    let saturate = |x: f32| interpret(&bc, "native_saturate", &p, [x, 0.0, 0.0]);
    for (x, want) in [
        (nan, 0.0f32),
        (2.0, 1.0),
        (-3.0, 0.0),
        (0.25, 0.25),
        (f32::INFINITY, 1.0),
        (f32::NEG_INFINITY, 0.0),
    ] {
        assert_eq!(saturate(x), want, "saturate({x})");
    }
    // Either zero (`NativeMinMax`'s tie).
    assert_eq!(saturate(-0.0), 0.0);
}

/// The `GLSL.std.450` instruction numbers and the opcodes of `words`' instructions.
fn instructions(words: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let (mut ext, mut ops) = (Vec::new(), Vec::new());
    let mut i = 5;
    while i < words.len() {
        let (len, op) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if op == crate::spv::Op::ExtInst as u32 {
            ext.push(words[i + 4]);
        }
        ops.push(op);
        i += len;
    }
    (ext, ops)
}

/// `dot` is the device's `OpDot` and `mix` its `FMix` (a `half` one rounded once), with no fused
/// multiply-add, and `fast::sqrt` the device's `Sqrt`, on lavapipe and on MoltenVK alike, each
/// recording its note. On the interpreter's model of the instructions each
/// answer is within the bound its note states (`fast::sqrt` within Vulkan's `1 / InverseSqrt`, 4.5
/// ulp, on lavapipe).
#[test]
fn dot_mix_and_fast_sqrt_are_the_device_s_on_every_profile() {
    let Some(bc) = bitcode() else {
        return;
    };
    let on = profile("lavapipe", DevicePolicy::RESIDENT_TABLES).expect("lavapipe");
    let metal = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("moltenvk-m3");
    let (fma, fmix, sqrt) = (
        crate::spv::GlslStd450Op::Fma as u32,
        crate::spv::GlslStd450Op::FMix as u32,
        crate::spv::GlslStd450Op::Sqrt as u32,
    );
    for (p, stem) in [(&on, "lavapipe"), (&metal, "moltenvk-m3")] {
        for (name, note) in [
            ("native_dot", PrecisionNote::NativeDot),
            ("native_mix", PrecisionNote::NativeMix),
            ("native_half_mix", PrecisionNote::NativeMix),
            ("native_fast_sqrt", PrecisionNote::NativeTranscendentals),
        ] {
            let what = format!("{name} on {stem}");
            let a = compile(&bc, name, p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) = link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (ext, ops) = instructions(&words);
            assert!(
                !ext.contains(&fma),
                "{what}: a fused multiply-add ({ext:?})"
            );
            match name {
                "native_dot" => assert!(ops.contains(&(crate::spv::Op::Dot as u32)), "{what}"),
                "native_fast_sqrt" => assert!(ext.contains(&sqrt), "{what}: {ext:?}"),
                _ => assert!(ext.contains(&fmix), "{what}: {ext:?}"),
            }
            assert!(a.precision.contains(&note), "{what}: {:?}", a.precision);
        }
    }
    for p in [&on, &metal] {
        for [a, b, c] in [
            [1.0f32, 3.0, 0.5],
            [-3.0, 0.7, 1.9],
            [0.75, -4.1, 2.5e-3],
            [1e-3, 3e7, -0.25],
        ] {
            let (a64, b64, c64) = (f64::from(a), f64::from(b), f64::from(c));
            // `dot(float2(a, b), float2(c, a))`: two lanes, `(3n - 1) * 2^-24 * sum |a_i * b_i|`.
            let sum = (a64 * c64).abs() + (b64 * a64).abs();
            let dot = interpret(&bc, "native_dot", p, [a, b, c]);
            assert!(
                (f64::from(dot) - (a64 * c64 + b64 * a64)).abs() <= 5.0 * 2f64.powi(-24) * sum,
                "dot({a}, {b}, {c}) = {dot}"
            );
            // `mix(a, b, c)`: within `2^-21 * (|a * (1 - c)| + |b * c|)`.
            let scale = (a64 * (1.0 - c64)).abs() + (b64 * c64).abs();
            let mix = interpret(&bc, "native_mix", p, [a, b, c]);
            assert!(
                (f64::from(mix) - (a64 + c64 * (b64 - a64))).abs() <= 2f64.powi(-21) * scale,
                "mix({a}, {b}, {c}) = {mix}"
            );
        }
    }
    // `half` `mix(0.5, 1.5, 0.25)` is exact: 0.75.
    for p in [&on, &metal] {
        assert_eq!(interpret(&bc, "native_half_mix", p, [0.5, 1.5, 0.25]), 0.75);
    }
    // `fast::sqrt`'s zero and `+inf` are IEEE's, as Metal's.
    let sqrt = |x: f32| interpret(&bc, "native_fast_sqrt", &on, [x, 0.0, 0.0]);
    for x in [2.0f32, 0.75, 1e-30, 3e30] {
        assert!(ulps(sqrt(x), f64::from(x).sqrt()) <= 4.5, "fast::sqrt({x})");
    }
    assert_eq!(sqrt(0.0).to_bits(), 0);
    assert_eq!(sqrt(f32::INFINITY), f32::INFINITY);
}
