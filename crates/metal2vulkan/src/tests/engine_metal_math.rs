//! Metal's math functions end to end: a call's `air` symbol variant and fast-math flags select
//! what it computes, the device's instruction where one is the function and a definition of the
//! device's instructions where none is. On every committed M1 profile each kernel compiles, links,
//! and passes L3, and its module holds the instructions of the form it takes:
//!
//! - `air.fast_tanh` is the device's `Tanh` (no `Exp2`, no division) on `float` and `half`, as
//!   the precise `air.tanh` is, where MSL Table 8.2 defines Metal's `float` one as the formula
//!   `(t - 1) / (t + 1)` of `t = exp(2x)`: noted `NativeTranscendentals` wherever the device's
//!   functions are not Metal's own, and on MoltenVK, whose `Tanh` is MSL's precise `tanh` (and a
//!   `half` one that rounded to `half`), `DeviceTranscendentals`;
//! - `air.sign` is the device's `FSign` with and without fast math (it keeps the sign of a zero
//!   where Metal answers `+0`), noted `NativeTranscendentals` wherever the device's functions are
//!   not Metal's own (on MoltenVK it is MSL's own `sign`), and both answer `-1` for `-3` on the
//!   interpreter;
//! - `air.sinpi` and `air.fast_sinpi` reduce their argument (`RoundEven`) and take the sine of the
//!   residue from the device's `Sin` on every profile.
//!
//! The values of these functions are the unit tests' (`lower::arith::tests`); no GPU is used.
//!
//! The sRGB conversions of `unorm4x8_srgb` compile end to end: A1's committed fixtures
//! `a1_srgb_decode_all_bytes` (every byte in every channel, `float` and `half` results) and
//! `a1_srgb_encode_sweep` (every multiple of 2⁻¹⁴ in `[0, 1.0078]` and every `half` up to 2.25)
//! compile on every profile above, take the MSL transfer functions by the device's `Pow` around
//! the 8-bit instruction, declare no table, and note `DevicePackConversion`. The interpreter's `Pow`
//! is one answer of Vulkan's bound (the `f64` power rounded once), not the device's, so their
//! values are the device lanes' (the validation crate's authored cases of the same fixtures).
//!
//! `llvm-as` assembles the modules; without it the tests skip visibly.

use crate::air::{self, SpecRequest};
use crate::device::profile::FloatFunctions;
use crate::lower::emit::emit_module;
use crate::lower::select::{select, LowerInputs};
use crate::memory::static_root::analyze_memory;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, exact_maths, link_and_validate, llvm_assemble, meter, profile,
    typed_module, AssembleError, Scratch,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::ir::StorageClass;

const PROFILES: [&str; 3] = ["moltenvk-m3", "lunarg-desktop-baseline-2023", "vk12-floor"];

/// `GLSL.std.450` instruction numbers.
const FSIGN: u32 = 6;
const SIN: u32 = 13;
const TANH: u32 = 21;
const EXP2: u32 = 29;
const ROUND_EVEN: u32 = 2;
const POW: u32 = 26;
const PACK_UNORM_4X8: u32 = 55;
const UNPACK_UNORM_4X8: u32 = 64;

/// One kernel: its body defines `%r` (`float`), which it stores.
struct Kernel {
    name: &'static str,
    body: &'static str,
    /// `GLSL.std.450` instructions the module must and must not contain.
    has: &'static [u32],
    lacks: &'static [u32],
    /// Whether the module takes the device's instruction for a function Metal defines otherwise,
    /// recording `NativeTranscendentals` wherever the device's functions are not Metal's own.
    native: bool,
    /// Whether the instruction on MoltenVK is not Metal's own function for the call
    /// (`DeviceTranscendentals` there): MSL's precise `tanh` for `fast::tanh`, its `float` one
    /// rounded for a `half` one.
    off_metal: bool,
    /// The stored word on the interpreter, when it models the kernel.
    want: Option<u32>,
}

const KERNELS: [Kernel; 7] = [
    Kernel {
        name: "fast_tanh_float",
        body: "  %a = fadd fast float 0x4046000000000000, -0.0\n  %r = call fast float @air.fast_tanh.f32(float %a)\n",
        has: &[TANH],
        lacks: &[EXP2],
        native: true,
        off_metal: true,
        want: None,
    },
    Kernel {
        name: "precise_tanh_float",
        body: "  %a = fadd fast float 0x4046000000000000, -0.0\n  %r = call fast float @air.tanh.f32(float %a)\n",
        has: &[TANH],
        lacks: &[EXP2],
        native: false,
        off_metal: false,
        want: None,
    },
    Kernel {
        name: "fast_tanh_half",
        body: "  %a = fadd fast float 0x4000000000000000, -0.0\n  %h = fptrunc float %a to half\n  %t = call fast half @air.fast_tanh.f16(half %h)\n  %r = fpext half %t to float\n",
        has: &[TANH],
        lacks: &[EXP2],
        native: true,
        off_metal: true,
        want: None,
    },
    Kernel {
        name: "fast_sign",
        body: "  %a = fadd fast float 0xC008000000000000, -0.0\n  %r = call fast float @air.sign.f32(float %a)\n",
        has: &[FSIGN],
        lacks: &[],
        native: true,
        off_metal: false,
        want: Some(0xBF80_0000),
    },
    Kernel {
        name: "precise_sign",
        body: "  %a = fadd float 0xC008000000000000, -0.0\n  %r = call float @air.sign.f32(float %a)\n",
        has: &[FSIGN],
        lacks: &[],
        native: true,
        off_metal: false,
        want: Some(0xBF80_0000),
    },
    Kernel {
        name: "precise_sinpi",
        body: "  %a = fadd fast float 0x3FF0000000000000, -0.0\n  %r = call fast float @air.sinpi.f32(float %a)\n",
        has: &[SIN, ROUND_EVEN],
        lacks: &[],
        native: false,
        off_metal: false,
        want: None,
    },
    Kernel {
        name: "fast_sinpi",
        body: "  %a = fadd fast float 0x3FF0000000000000, -0.0\n  %r = call fast float @air.fast_sinpi.f32(float %a)\n",
        has: &[SIN, ROUND_EVEN],
        lacks: &[],
        native: false,
        off_metal: false,
        want: None,
    },
];

fn module() -> String {
    let mut s = String::from(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n",
    );
    for k in &KERNELS {
        s.push_str(&format!(
            "define void @{}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{\nentry:\n{}  store float %r, ptr addrspace(1) %0, align 4\n  ret void\n}}\n\n",
            k.name, k.body
        ));
    }
    s.push_str("declare float @air.fast_tanh.f32(float)\ndeclare float @air.tanh.f32(float)\ndeclare half @air.fast_tanh.f16(half)\ndeclare float @air.sign.f32(float)\ndeclare float @air.sinpi.f32(float)\ndeclare float @air.fast_sinpi.f32(float)\n");
    let refs: Vec<String> = (0..KERNELS.len()).map(|i| format!("!{}", 20 + i)).collect();
    s.push_str(&format!(
        "\n!llvm.module.flags = !{{!0}}\n!air.kernel = !{{{}}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n",
        refs.join(", ")
    ));
    s.push_str("!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n!10 = !{}\n!11 = !{!12}\n!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"out\"}\n!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n");
    for (i, k) in KERNELS.iter().enumerate() {
        s.push_str(&format!("!{} = !{{ptr @{}, !10, !11}}\n", 20 + i, k.name));
    }
    s
}

fn assemble() -> Option<Vec<u8>> {
    let scratch = Scratch::new("m2v-metal-math");
    let dir = scratch.path();
    let ll = dir.join("metal_math.ll");
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

/// The `GLSL.std.450` instruction numbers of every `OpExtInst` in `words`, and whether an
/// `OpFDiv` occurs.
fn ext_insts(words: &[u32]) -> (Vec<u32>, bool) {
    let (mut ext, mut div) = (Vec::new(), false);
    let mut i = 5;
    while i < words.len() {
        let (len, op) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if op == crate::spv::Op::ExtInst as u32 {
            ext.push(words[i + 4]);
        }
        div |= op == crate::spv::Op::FDiv as u32;
        i += len;
    }
    (ext, div)
}

/// The stored word of `entry` on the interpreter.
fn interpret(bitcode: &[u8], entry: &str, stem: &str) -> u32 {
    let p = profile(stem, exact_maths()).expect("profile");
    let mut m = meter();
    let decoded = air::decode(bitcode, &mut m).expect("decode");
    let req = SpecRequest {
        call_stack_depth: 1,
        entry: entry.to_owned(),
        ..SpecRequest::default()
    };
    let spec = air::specialize(decoded, &req, &mut m).expect("specialize");
    let memory = analyze_memory(&spec, &p, &mut m).expect("memory");
    let plan = select(&spec, &memory, &p, &LowerInputs::new(), &mut m).expect("select");
    let module = emit_module(&spec, &plan, &mut m).expect("emit").module;
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0; 4],
        }],
        push: vec![0; 4],
        spec: Default::default(),
    };
    let out = run(&module, &Config::default(), res)
        .unwrap_or_else(|e| panic!("{entry}: interpreter: {e}"));
    let b = out.buffers.first().expect("buffer");
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[test]
fn metal_math_variants_select_their_forms() {
    let Some(bitcode) = assemble() else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for k in &KERNELS {
            let what = format!("{} on {stem}", k.name);
            let a = compile(&bitcode, k.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (ext, div) = ext_insts(&words);
            for i in k.has {
                assert!(
                    ext.contains(i),
                    "{what}: GLSL.std.450 {i} missing ({ext:?})"
                );
            }
            for i in k.lacks {
                assert!(
                    !ext.contains(i),
                    "{what}: GLSL.std.450 {i} present ({ext:?})"
                );
            }
            let metal = p.fields().float_functions == FloatFunctions::Metal;
            assert!(!div, "{what}: OpFDiv");
            // The device's instruction: noted `NativeTranscendentals` wherever the device's
            // functions are not Metal's own. On MoltenVK `sign` is MSL's own; `fast::tanh` is
            // MSL's precise `tanh`, and a `half` `tanh` MSL's `float` one rounded to `half`.
            if k.native {
                assert_eq!(
                    a.precision.contains(&PrecisionNote::NativeTranscendentals),
                    !metal,
                    "{what}: {:?}",
                    a.precision
                );
                assert_eq!(
                    a.precision.contains(&PrecisionNote::DeviceTranscendentals),
                    metal && k.off_metal,
                    "{what}: {:?}",
                    a.precision
                );
            }
            if let Some(want) = k.want {
                let got = interpret(&bitcode, k.name, stem);
                assert_eq!(got, want, "{what}: {got:#x} != {want:#x}");
            }
        }
    }
}

/// End to end: the sRGB fixtures through select, emit, and link on every profile above: L3 passes,
/// and each module takes the MSL transfer functions by the device's `Pow` around the 8-bit
/// instruction (`a1_srgb_decode_all_bytes`: `UnpackUnorm4x8`, its `float` and `half` results;
/// `a1_srgb_encode_sweep`: `PackUnorm4x8`, from `float` and `half` colours), declares no table
/// (no initialized `Private` variable), and records `DevicePackConversion`.
#[test]
fn srgb_fixtures_take_the_transfer_by_the_devices_pow() {
    let assembled = |stem: &str| match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected {stem}: {e}"),
    };
    let (Some(decode_bc), Some(encode_bc)) = (
        assembled("a1_srgb_decode_all_bytes"),
        assembled("a1_srgb_encode_sweep"),
    ) else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for (bc, entry, instruction) in [
            (&decode_bc, "a1_srgb_decode_all_bytes", UNPACK_UNORM_4X8),
            (&encode_bc, "a1_srgb_encode_sweep", PACK_UNORM_4X8),
        ] {
            let what = format!("{entry} on {stem}");
            let a = compile(bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (ext, _) = ext_insts(&words);
            for i in [POW, instruction] {
                assert!(
                    ext.contains(&i),
                    "{what}: GLSL.std.450 {i} missing ({ext:?})"
                );
            }
            assert!(
                a.precision.contains(&PrecisionNote::DevicePackConversion),
                "{what}: {:?}",
                a.precision
            );
            let (_, m) = typed_module(bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let tables = m
                .globals()
                .iter()
                .filter(|g| g.class == StorageClass::Private && g.init.is_some())
                .count();
            assert_eq!(tables, 0, "{what}: a table");
        }
    }
}
