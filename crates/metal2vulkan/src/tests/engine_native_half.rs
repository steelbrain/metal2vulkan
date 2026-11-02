//! Native half rounding: a `half` result the module computes in `float` rounds through the device's
//! conversion and its widening under the float controls that make it Metal's
//! (`QuantizeForm::Controlled`), else through that conversion as it is where the profile has
//! `shaderFloat16` (`QuantizeForm::Converted`), recording `PrecisionNote::NativeHalfConversion`,
//! else through the device's quantize alone (`QuantizeForm::Device`), recording
//! `NativeHalfQuantize` on every such profile. A `half` read from memory is the device's widening
//! on every profile: its conversion where the module holds 16-bit floats, `UnpackHalf2x16`
//! elsewhere.
//!
//! The kernels read two `float`s from buffer 0 and store one result after them: a `half` sum of
//! the two narrowed, and the first narrowed alone. `llvm-as` assembles the module; without it the
//! compiling tests skip visibly.

use crate::air::{self, SpecRequest};
use crate::device::requirements::VkProperty;
use crate::device::DevicePolicy;
use crate::device::DeviceProfile;
use crate::error::DriverDeviation;
use crate::lower::arith::narrow::{controlled_modes, native_rounding, quantize_form};
use crate::lower::emit::emit_module;
use crate::lower::select::{select, LowerInputs};
use crate::memory::static_root::analyze_memory;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, compile_kept, device_nan_bits, link_and_validate, llvm_assemble, meter, profile,
    AssembleError, Scratch,
};
use crate::pipeline::PrecisionNote;
use crate::spirv::builder::QuantizeForm;
use crate::spirv::ids::{ConstId, Value};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::ir::GlslOp;
use crate::spirv::ir::{
    BinOp, ConstKind, Constant, Contraction, ConvOp, Inst, Type, TypedModule, ValueOrigin,
};
use crate::spirv::testing::committed_profiles;

/// The kernels: a name and the body defining `%r` from `%a` and `%b`.
const KERNELS: [(&str, &str); 2] = [
    (
        "half_sum",
        "  %ha = fptrunc float %a to half\n  %hb = fptrunc float %b to half\n  %hs = fadd half %ha, %hb\n  %r = fpext half %hs to float\n",
    ),
    (
        "half_narrow",
        "  %ha = fptrunc float %a to half\n  %r = fpext half %ha to float\n",
    ),
];

/// The module of every kernel.
fn module() -> String {
    let mut s = String::from(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n",
    );
    for (name, body) in &KERNELS {
        s.push_str(&format!(
            "define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{\nentry:\n  %p1 = getelementptr inbounds float, ptr addrspace(1) %0, i64 1\n  %p2 = getelementptr inbounds float, ptr addrspace(1) %0, i64 2\n  %a = load float, ptr addrspace(1) %0, align 4\n  %b = load float, ptr addrspace(1) %p1, align 4\n{body}  store float %r, ptr addrspace(1) %p2, align 4\n  ret void\n}}\n\n"
        ));
    }
    let refs: Vec<String> = (0..KERNELS.len()).map(|i| format!("!{}", 20 + i)).collect();
    s.push_str(&format!(
        "!llvm.module.flags = !{{!0}}\n!air.kernel = !{{{}}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n",
        refs.join(", ")
    ));
    s.push_str("!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n!10 = !{}\n!11 = !{!12}\n!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"io\"}\n!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n");
    for (i, (name, _)) in KERNELS.iter().enumerate() {
        s.push_str(&format!("!{} = !{{ptr @{name}, !10, !11}}\n", 20 + i));
    }
    s
}

/// The module's bitcode, or `None` (skipped visibly) without `llvm-as`.
fn bitcode() -> Option<Vec<u8>> {
    assemble("native-half", &module())
}

/// `text` assembled by `llvm-as` (in a directory of its own named after `stem`), or `None`
/// (skipped visibly) without it.
fn assemble(stem: &str, text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new(&format!("m2v-{stem}"));
    let dir = scratch.path();
    let ll = dir.join(format!("{stem}.ll"));
    std::fs::write(&ll, text).expect("write");
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

/// `entry` on `p` on the interpreter (whose conversion and quantize round to nearest even) with
/// inputs `x`: the stored result.
fn interpret(bitcode: &[u8], entry: &str, p: &DeviceProfile, x: [f32; 2]) -> f32 {
    let bytes = [x[0], x[1], 0.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let b = interpret_bytes(bitcode, entry, p, bytes);
    f32::from_le_bytes([b[8], b[9], b[10], b[11]])
}

/// `entry` on `p` on the interpreter with buffer 0 holding `bytes`: the buffer after the run.
fn interpret_bytes(bitcode: &[u8], entry: &str, p: &DeviceProfile, bytes: Vec<u8>) -> Vec<u8> {
    let mut m = meter();
    let decoded = air::decode(bitcode, &mut m).expect("decode");
    let req = SpecRequest {
        call_stack_depth: 1,
        entry: entry.to_owned(),
        ..SpecRequest::default()
    };
    let spec = air::specialize(decoded, &req, &mut m).expect("specialize");
    let memory = analyze_memory(&spec, p, &mut m).expect("memory");
    let plan = select(&spec, &memory, p, &LowerInputs::new(), &mut m).expect("select");
    let module = emit_module(&spec, &plan, &mut m).expect("emit").module;
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
    let mut out = run(&module, &Config::default(), res)
        .unwrap_or_else(|e| panic!("{entry}: interpreter: {e}"));
    out.buffers.swap_remove(0)
}

/// The form a profile's `half` roundings take: the controlled conversion where the profile has
/// `shaderFloat16` and its float controls, else the conversion with `shaderFloat16`, else the
/// quantize.
fn expected(p: &DeviceProfile) -> QuantizeForm {
    let float16 = p.fields().leaf.float16;
    if float16 && controlled_modes(p).is_some() {
        QuantizeForm::Controlled
    } else if float16 {
        QuantizeForm::Converted
    } else {
        QuantizeForm::Device
    }
}

/// Every committed profile rounds by the device's own form: the controlled conversion with no
/// note, the conversion noted `NativeHalfConversion`, and the quantize noted `NativeHalfQuantize`.
#[test]
fn half_rounding_is_the_devices_on_every_profile() {
    let profiles = committed_profiles().expect("profiles");
    let mut kinds = [0usize; 2];
    for (stem, p) in &profiles {
        let form = quantize_form(p);
        assert_eq!(form, expected(p), "{stem}");
        let note = native_rounding(form);
        let want = match form {
            QuantizeForm::Controlled => None,
            QuantizeForm::Converted => Some(PrecisionNote::NativeHalfConversion),
            QuantizeForm::Device => Some(PrecisionNote::NativeHalfQuantize),
        };
        assert_eq!(note, want, "{stem}");
        match note {
            Some(PrecisionNote::NativeHalfConversion) => kinds[0] += 1,
            Some(PrecisionNote::NativeHalfQuantize) => kinds[1] += 1,
            Some(other) => panic!("{stem}: {other:?}"),
            None => {}
        }
    }
    assert!(
        kinds[0] > 0 && kinds[1] > 0,
        "profiles with each native form: {kinds:?}"
    );
}

/// On lavapipe and MoltenVK (`shaderFloat16`, no `DenormPreserve` for 16 bits) each kernel rounds
/// its `half` through the conversion, records `NativeHalfConversion` and requires no measured
/// fact; on a profile without `shaderFloat16` it quantizes and records `NativeHalfQuantize`; RADV
/// takes the controlled conversion, with no fact and no note. Each passes L3. On the interpreter,
/// whose conversion and quantize round to nearest even, the native modules answer Metal's
/// `half(float)`, but for the quantize's zero below the smallest normal half.
#[test]
fn native_half_rounding_records_its_kind_and_requires_no_fact() {
    let Some(bc) = bitcode() else {
        return;
    };
    let natives = [
        PrecisionNote::NativeHalfConversion,
        PrecisionNote::NativeHalfQuantize,
    ];
    let cases: [(&str, Option<PrecisionNote>); 4] = [
        ("lavapipe", Some(natives[0])),
        ("swiftshader", Some(natives[1])),
        ("moltenvk-m3", Some(natives[0])),
        ("amd-780m-radv", None),
    ];
    for (stem, note) in cases {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        for (name, _) in &KERNELS {
            let what = format!("{name} on {stem}");
            let a = compile(&bc, name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let noted: Vec<_> = natives.iter().filter(|n| a.precision.contains(n)).collect();
            assert_eq!(noted, note.iter().collect::<Vec<_>>(), "{what}");
        }
    }
    // Metal's `half(float)` on the interpreter's nearest-even conversion and quantize: a tie to
    // even, the largest half below 65520 and infinity from it; a half denormal the conversion's,
    // and the quantize's zero (the device's, `NativeHalfQuantize`).
    for (stem, keeps) in [("lavapipe", true), ("swiftshader", false)] {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        let run = |name: &str, x: [f32; 2]| interpret(&bc, name, &p, x);
        let tie = 1.0 + 2f32.powi(-11);
        assert_eq!(run("half_narrow", [tie, 0.0]), 1.0, "{stem}: a tie to even");
        assert_eq!(run("half_narrow", [65519.0, 0.0]), 65504.0, "{stem}");
        assert_eq!(run("half_narrow", [65520.0, 0.0]), f32::INFINITY, "{stem}");
        let denormal = 3.0 * 2f32.powi(-24);
        assert_eq!(
            run("half_narrow", [denormal + 2f32.powi(-27), 0.0]),
            if keeps { denormal } else { 0.0 },
            "{stem}: a half denormal"
        );
        assert_eq!(run("half_sum", [1.0, 2f32.powi(-11)]), 1.0, "{stem}: a sum");
        assert_eq!(run("half_sum", [1.5, 2.25]), 3.75, "{stem}: an exact sum");
    }
}

/// On a profile without `shaderFloat16` (`QuantizeForm::Device`: the floor, SwiftShader, NVIDIA's
/// Pascal) a `half` sum and a narrowing alike take the device's `OpQuantizeToF16` alone, with no
/// denormal arm (no `RoundEven`), and record `NativeHalfQuantize` and require no fact; on the
/// interpreter, whose quantize flushes what is too small for a normal half, a sum or a narrowing
/// below the smallest normal half answers the zero of its sign, where Metal answers the half
/// denormal. A result from the smallest normal half up is the exactly rounded half.
#[test]
fn a_half_below_the_smallest_normal_takes_the_devices_quantize() {
    let Some(bc) = bitcode() else {
        return;
    };
    let denormal = 3.0 * 2f32.powi(-24);
    let mut devices = 0;
    for (stem, p) in &committed_profiles().expect("profiles") {
        if quantize_form(p) != QuantizeForm::Device {
            continue;
        }
        devices += 1;
        for (name, _) in &KERNELS {
            let what = format!("{name} on {stem}");
            let a = compile(&bc, name, p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let r = rounding(&a.module.words);
            assert!(r.quantizes > 0, "{what}: {r:?}");
            assert_eq!(r.round_evens, 0, "{what}: a denormal arm, {r:?}");
            assert!(
                a.precision.contains(&PrecisionNote::NativeHalfQuantize),
                "{what}: {:?}",
                a.precision
            );
        }
        let run = |name: &str, x: [f32; 2]| interpret(&bc, name, p, x);
        // Two normal halves whose difference is the half denormal `3 * 2^-24`.
        let normal = 2f32.powi(-14);
        let sum = run("half_sum", [normal + denormal, -normal]);
        assert_eq!(
            sum.to_bits(),
            0,
            "{stem}: a sum below the smallest normal half"
        );
        let sum = run("half_sum", [-normal - denormal, normal]);
        assert_eq!(sum.to_bits(), (-0.0f32).to_bits(), "{stem}: its sign");
        let narrowed = run("half_narrow", [denormal, 0.0]);
        assert_eq!(
            narrowed.to_bits(),
            0,
            "{stem}: a narrowing below the smallest normal half"
        );
        let narrowed = run("half_narrow", [-denormal, 0.0]);
        assert_eq!(narrowed.to_bits(), (-0.0f32).to_bits(), "{stem}: its sign");
        assert_eq!(
            run("half_sum", [2f32.powi(-13) + 2f32.powi(-23), -normal]),
            2f32.powi(-14) + 2f32.powi(-23),
            "{stem}: a normal sum"
        );
        assert_eq!(
            run("half_narrow", [2f32.powi(-14) + 2f32.powi(-25), 0.0]),
            2f32.powi(-14),
            "{stem}: a normal narrowing's tie to even"
        );
    }
    assert!(devices > 0, "a profile without shaderFloat16");
}

/// The decode kernels, over one buffer of words: the two `half`s of word 0, the loop's count in
/// word 2, a `half4` in words 4 and 5, and results in words 1, 3 and 6. `@both` decodes the first
/// `half` and the `half4` before its loop, sums the first `n` `half`s in the loop (a decode each
/// trip) and stores the sum to word 1, the first `half` to word 3 and the sum of the `half4`'s
/// lanes to word 6; `@inside` stores the loop's sum alone.
fn decode_module() -> String {
    let mut s = String::from(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n",
    );
    let sum = "  %pn = getelementptr inbounds i32, ptr addrspace(1) %0, i64 2\n  %n = load i32, ptr addrspace(1) %pn, align 4\n  br label %loop\nloop:\n  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]\n  %acc = phi float [ 0.0, %entry ], [ %acc1, %loop ]\n  %iw = zext i32 %i to i64\n  %ph = getelementptr inbounds half, ptr addrspace(1) %0, i64 %iw\n  %h = load half, ptr addrspace(1) %ph, align 2\n  %x = fpext half %h to float\n  %acc1 = fadd float %acc, %x\n  %i1 = add i32 %i, 1\n  %more = icmp ult i32 %i1, %n\n  br i1 %more, label %loop, label %done\ndone:\n  %p1 = getelementptr inbounds float, ptr addrspace(1) %0, i64 1\n  store float %acc1, ptr addrspace(1) %p1, align 4\n";
    s.push_str(&format!(
        "define void @both(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{\nentry:\n  %h0 = load half, ptr addrspace(1) %0, align 2\n  %x0 = fpext half %h0 to float\n  %p4 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 4\n  %hv = load <4 x half>, ptr addrspace(1) %p4, align 8\n  %xv = fpext <4 x half> %hv to <4 x float>\n  %v0 = extractelement <4 x float> %xv, i32 0\n  %v1 = extractelement <4 x float> %xv, i32 1\n  %v2 = extractelement <4 x float> %xv, i32 2\n  %v3 = extractelement <4 x float> %xv, i32 3\n  %s01 = fadd float %v0, %v1\n  %s012 = fadd float %s01, %v2\n  %vs = fadd float %s012, %v3\n{sum}  %p3 = getelementptr inbounds float, ptr addrspace(1) %0, i64 3\n  store float %x0, ptr addrspace(1) %p3, align 4\n  %p6 = getelementptr inbounds float, ptr addrspace(1) %0, i64 6\n  store float %vs, ptr addrspace(1) %p6, align 4\n  ret void\n}}\n\n"
    ));
    s.push_str(&format!(
        "define void @inside(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{\nentry:\n{sum}  ret void\n}}\n\n"
    ));
    s.push_str("!llvm.module.flags = !{!0}\n!air.kernel = !{!20, !21}\n!air.version = !{!17}\n!air.language_version = !{!18}\n\n");
    s.push_str("!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n!10 = !{}\n!11 = !{!12}\n!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"io\"}\n!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n!20 = !{ptr @both, !10, !11}\n!21 = !{ptr @inside, !10, !11}\n");
    s
}

/// The `half` decodes of `m` in its functions' blocks: the device's widenings (an `OpFConvert`
/// from 16- to 32-bit floats, or a `GLSL.std.450 UnpackHalf2x16`) and the conversions of a
/// magnitude to `float` an integer decode would take (`OpConvertUToF`, which nothing else in the
/// decode kernels has).
fn decodes(m: &TypedModule) -> (usize, usize) {
    let width = |v: Value| -> Option<u8> {
        let ty = m.value(v).ok()?.ty;
        let ty = match m.types().get(ty).ok()? {
            Type::Vector { elem, .. } => *elem,
            _ => ty,
        };
        match m.types().get(ty).ok()? {
            Type::Float { width } => Some(*width),
            _ => None,
        }
    };
    let (mut device, mut integer) = (0, 0);
    for n in m.blocks().iter().flat_map(|b| b.insts.iter()) {
        match (&n.inst, n.result) {
            (
                Inst::Convert {
                    op: ConvOp::FConvert,
                    a,
                    ..
                },
                Some(r),
            ) if width(*a) == Some(16) && width(r) == Some(32) => device += 1,
            (
                Inst::Glsl {
                    op: GlslOp::UnpackHalf2x16,
                    ..
                },
                _,
            ) => device += 1,
            (
                Inst::Convert {
                    op: ConvOp::ConvertUToF,
                    ..
                },
                _,
            ) => integer += 1,
            _ => {}
        }
    }
    (device, integer)
}

/// A `half` read from memory is the device's widening of its bits on every profile, inside a loop
/// and out, with no integer decode: on lavapipe and NVIDIA (the device's conversion, no float
/// controls) the module records `NativeNanEncoding` and `NativeHalfConversion` (a NaN's bits and,
/// where the device flushes 16-bit denormals, a denormal); on MoltenVK, whose widening the measured
/// `MetalHalfArithmetic` reads as Metal's, it records `NativeNanEncoding` alone and requires that
/// fact; on the floor (no `shaderFloat16`) it is `UnpackHalf2x16` of each word, recording both
/// notes and requiring no fact. A `half4` widens as one vector (two words unpacked and shuffled
/// where the device unpacks). Each module passes L3 and answers the `half`s' values on the
/// interpreter (whose widening is exact). Where the device's moves of a NaN are measured to be
/// Metal's (`DriverDeviation::MetalNanMoves`, MoltenVK on the M3 Max's identity), a module that
/// moves its `half`s only so requires that in place of the NaN's note (`device_nan_bits`).
#[test]
fn a_half_decode_is_the_devices_widening_inside_loops_and_out() {
    let Some(bc) = assemble("native-half-decode", &decode_module()) else {
        return;
    };
    // Word 0: 1.5 (0x3E00) and -2.5 (0xC100); the loop sums both; word 3 takes 1.5.
    // Words 4 and 5: 2.0 (0x4000), 1.0 (0x3C00), 0.5 (0x3800) and 0.25 (0x3400); their sum goes
    // to word 6.
    let input = [
        0x3E00 | (0xC100 << 16),
        0,
        2,
        0,
        0x4000 | (0x3C00 << 16),
        0x3800 | (0x3400 << 16),
        0,
    ];
    let conversion = PrecisionNote::NativeHalfConversion;
    // Each profile, whether its widening is noted `NativeHalfConversion` (else measured), and
    // whether it unpacks words (no `shaderFloat16`).
    let cases = [
        ("lavapipe", true, false),
        ("nvidia-rtx5080-laptop", true, false),
        ("moltenvk-m3", false, false),
        ("vk12-floor", true, true),
    ];
    for (stem, native, unpacks) in cases {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        for (entry, both) in [("both", true), ("inside", false)] {
            let what = format!("{entry} on {stem}");
            let (a, plan, kept) =
                compile_kept(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert!(!kept.loops().is_empty(), "{what}: the loop");
            // The loop's `half`, and before it the first `half` and the `half4`, which the unpack
            // reads as its two words.
            let reads = match (both, unpacks) {
                (false, _) => 1,
                (true, false) => 3,
                (true, true) => 4,
            };
            let (device, integer) = decodes(&kept);
            assert_eq!((device, integer), (reads, 0), "{what}: the decodes");
            assert!(device_nan_bits(&a), "{what}: {:?}", a.precision);
            assert_eq!(
                a.precision.contains(&conversion),
                native,
                "{what}: {:?}",
                a.precision
            );
            assert_eq!(
                a.requirements
                    .properties
                    .contains(&VkProperty::MetalHalfArithmetic),
                !native,
                "{what}: {:?}",
                a.requirements.properties
            );
            let res = Resources {
                buffers: vec![Buffer {
                    slot: Some(ResourceSlot {
                        stage: Stage::Kernel,
                        kind: ResourceKind::Buffer,
                        index: 0,
                    }),
                    address: None,
                    bytes: input.iter().flat_map(|w: &u32| w.to_le_bytes()).collect(),
                }],
                push: vec![0; plan.push.push_range.map_or(4, |(_, end)| end as usize)],
                spec: Default::default(),
            };
            let out = run(&kept, &Config::default(), res)
                .unwrap_or_else(|e| panic!("{what}: interpreter: {e}"));
            let b = out.buffers.first().expect("buffer");
            let word =
                |i: usize| f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
            assert_eq!(word(1), -1.0, "{what}: the loop's sum");
            if both {
                assert_eq!(word(3), 1.5, "{what}: the first half");
                assert_eq!(word(6), 3.75, "{what}: the half4's lanes");
            }
        }
    }
}

/// Whether constant `c` of `m` is `-0.0` in every lane.
fn minus_zero(m: &TypedModule, c: ConstId) -> bool {
    match m.consts().get(c).map(|d| &d.kind) {
        Ok(ConstKind::Plain(Constant::Scalar { bits, .. })) => bits.0 == 0x8000_0000,
        Ok(ConstKind::Plain(Constant::Composite { parts, .. })) => {
            parts.iter().all(|p| minus_zero(m, *p))
        }
        _ => false,
    }
}

/// The conversions to 16 bits of 32-bit values in `m` (each `half` rounding through the
/// conversion, `narrow::converted`), and how many of them convert an `OpFAdd` decorated
/// `NoContraction` whose second operand is `-0.0`.
fn barred(m: &TypedModule) -> (usize, usize) {
    let width = |v: Value| -> Option<u8> {
        let ty = m.value(v).ok()?.ty;
        let ty = match m.types().get(ty).ok()? {
            Type::Vector { elem, .. } => *elem,
            _ => ty,
        };
        match m.types().get(ty).ok()? {
            Type::Float { width } => Some(*width),
            _ => None,
        }
    };
    let insts = || m.blocks().iter().flat_map(|b| b.insts.iter());
    let defs: std::collections::HashMap<Value, &Inst> = insts()
        .filter_map(|n| n.result.map(|r| (r, &n.inst)))
        .collect();
    let (mut all, mut barred) = (0, 0);
    for n in insts() {
        if let (
            Inst::Convert {
                op: ConvOp::FConvert,
                a,
                ..
            },
            Some(r),
        ) = (&n.inst, n.result)
        {
            if width(*a) == Some(32) && width(r) == Some(16) {
                all += 1;
                if let Some(Inst::Binary {
                    op: BinOp::FAdd,
                    contract: Contraction::Forbidden,
                    b,
                    ..
                }) = defs.get(a)
                {
                    if let Ok(ValueOrigin::Const(c)) = m.value(*b).map(|d| d.origin) {
                        if minus_zero(m, c) {
                            barred += 1;
                        }
                    }
                }
            }
        }
    }
    (all, barred)
}

/// Each rounding through the conversion converts the value itself, on lavapipe and on MoltenVK
/// alike: no `x + -0.0` decorated `NoContraction` stands between the value and the conversion to
/// keep the device's compiler from folding it into the instruction computing the value.
#[test]
fn a_native_conversion_converts_the_value_itself() {
    let Some(bc) = bitcode() else {
        return;
    };
    for stem in ["lavapipe", "moltenvk-m3"] {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        for (name, _) in &KERNELS {
            let what = format!("{name} on {stem}");
            let (a, _, kept) =
                compile_kept(&bc, name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (all, barred) = barred(&kept);
            assert!(all > 0, "{what}: a rounding through the conversion");
            assert_eq!(barred, 0, "{what}: conversions of an added -0.0, of {all}");
        }
    }
}

/// What a module's words say of its `half` rounding: its execution modes with a float width, the
/// conversions from 32- to 16-bit floats and back with how many carry `NoContraction`, its
/// `OpQuantizeToF16`s, and its 16-bit float arithmetic.
#[derive(Debug, Default)]
struct Rounding {
    modes: std::collections::BTreeSet<(u32, u32)>,
    narrowings: usize,
    widenings: usize,
    uncontracted: usize,
    quantizes: usize,
    /// `OpConstant`s of 32-bit float type with Metal's canonical NaN, `0x7FC00000`.
    canonical_nans: usize,
    /// `OpFAdd`s, `OpFSub`s and extended instructions of a 16-bit float type: a `half` operation
    /// on the device's 16-bit arithmetic (`ops::f16_binary`), between exact conversions.
    f16_ops: usize,
    /// `OpLoad`s of a 16-bit integer: a 16-bit storage unit read.
    unit_loads: usize,
    /// `OpBitcast`s of a 16-bit integer to a 16-bit float: a unit read as a `half`.
    unit_halves: usize,
    /// `GLSL.std.450 RoundEven`s: the exact denormal arm a quantize no longer takes.
    round_evens: usize,
    /// `GLSL.std.450 UnpackHalf2x16`s: a `half` widened from its bits without `shaderFloat16`.
    unpacks: usize,
}

/// [`Rounding`] of `words`, read back through the test parser.
fn rounding(words: &[u32]) -> Rounding {
    use crate::spirv_binary::{parse_bytes, Operand};
    use crate::spv::{Decoration, ExecutionMode, Op};
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let parsed = parse_bytes(&bytes).expect("parse");
    let word = |o: Option<&Operand>| match o {
        Some(Operand::Word(w) | Operand::IdRef(w)) => Some(*w),
        _ => None,
    };
    let mut widths = std::collections::HashMap::new();
    let mut int16 = std::collections::HashSet::new();
    let mut types = std::collections::HashMap::new();
    let mut no_contraction = std::collections::HashSet::new();
    let mut r = Rounding::default();
    let float_modes = [
        ExecutionMode::RoundingModeRTE,
        ExecutionMode::DenormPreserve,
        ExecutionMode::SignedZeroInfNanPreserve,
    ];
    for i in &parsed.instructions {
        if let (Some(t), Some(id)) = (i.result_type, i.result_id) {
            types.insert(id, t);
        }
        match i.opcode {
            Op::TypeFloat => {
                if let (Some(id), Some(w)) = (i.result_id, word(i.operands.first())) {
                    widths.insert(id, w);
                }
            }
            Op::TypeInt if word(i.operands.first()) == Some(16) => int16.extend(i.result_id),
            Op::TypeVector => {
                let elem = word(i.operands.first()).and_then(|e| widths.get(&e).copied());
                if let (Some(id), Some(w)) = (i.result_id, elem) {
                    widths.insert(id, w);
                }
            }
            Op::Decorate if word(i.operands.get(1)) == Some(Decoration::NoContraction as u32) => {
                no_contraction.extend(word(i.operands.first()));
            }
            Op::ExecutionMode => {
                if let (Some(m), Some(w)) = (word(i.operands.get(1)), word(i.operands.get(2))) {
                    if float_modes.iter().any(|f| *f as u32 == m) {
                        r.modes.insert((m, w));
                    }
                }
            }
            Op::QuantizeToF16 => r.quantizes += 1,
            Op::ExtInst => match i.operands.get(1) {
                Some(Operand::LiteralExtInstInteger(n))
                    if *n == GlslOp::RoundEven.to_spirv() as u32 =>
                {
                    r.round_evens += 1
                }
                Some(Operand::LiteralExtInstInteger(n))
                    if *n == GlslOp::UnpackHalf2x16.to_spirv() as u32 =>
                {
                    r.unpacks += 1
                }
                _ => {}
            },
            _ => {}
        }
    }
    for i in &parsed.instructions {
        let width = i.result_type.and_then(|t| widths.get(&t).copied());
        if i.opcode == Op::Constant
            && width == Some(32)
            && word(i.operands.first()) == Some(0x7FC0_0000)
        {
            r.canonical_nans += 1;
        }
    }
    for i in &parsed.instructions {
        let width = i.result_type.and_then(|t| widths.get(&t).copied());
        if matches!(i.opcode, Op::FAdd | Op::FSub | Op::ExtInst) && width == Some(16) {
            r.f16_ops += 1;
        }
        let of_int16 = word(i.operands.first())
            .and_then(|a| types.get(&a))
            .is_some_and(|t| int16.contains(t));
        if i.opcode == Op::Load && i.result_type.is_some_and(|t| int16.contains(&t)) {
            r.unit_loads += 1;
        }
        if i.opcode == Op::Bitcast && width == Some(16) && of_int16 {
            r.unit_halves += 1;
        }
        if i.opcode != Op::FConvert {
            continue;
        }
        let to = i.result_type.and_then(|t| widths.get(&t).copied());
        let from = word(i.operands.first())
            .and_then(|a| types.get(&a))
            .and_then(|t| widths.get(t).copied());
        let pair = match (from, to) {
            (Some(32), Some(16)) => &mut r.narrowings,
            (Some(16), Some(32)) => &mut r.widenings,
            _ => continue,
        };
        *pair += 1;
        if i.result_id.is_some_and(|id| no_contraction.contains(&id)) {
            r.uncontracted += 1;
        }
    }
    r
}

/// Where the profile has `shaderFloat16` and the float controls that make a conversion to 16 bits
/// Metal's `half(float)` by the Vulkan spec (every ANV and RADV profile), and its driver does not
/// fold the round trip (`DriverWorkaround::HalfConversionPairFolded`, NVIDIA's), each kernel
/// rounds its `half` through that conversion and its widening, every one decorated
/// `NoContraction`, with no quantize, and computes its `half` sum on the
/// device's 16-bit arithmetic, between conversions that are exact and undecorated; its entry
/// declares `RoundingModeRTE`, `DenormPreserve` and `SignedZeroInfNanPreserve` for 16 bits and the
/// modes the device's independence ties to them (`RoundingModeRTE` for 32 and 64 bits on ANV,
/// whose rounding modes are not independent; `RoundingModeRTE` and `DenormPreserve` for 64 bits on
/// RADV, independent for 32 bits only); the module requires those float-controls properties and no
/// measured fact, records no native note, and passes L3. On the interpreter (whose conversion
/// rounds to nearest even) it answers Metal's `half(float)`: a tie to even, the largest half below
/// 65520 and infinity from it, a half denormal, a zero's sign, and a sum. MoltenVK, lavapipe, the
/// floor and NVIDIA's profiles keep their forms.
#[test]
fn a_controlled_conversion_rounds_half_where_the_float_controls_allow() {
    use crate::device::requirements::{FloatControl, FloatWidth};
    use crate::spv::ExecutionMode as M;
    let Some(bc) = bitcode() else {
        return;
    };
    let mut controlled = Vec::new();
    for (stem, p) in &committed_profiles().expect("profiles") {
        if quantize_form(p) != QuantizeForm::Controlled {
            continue;
        }
        controlled.push(stem.clone());
        let fc = p.fields().float_controls;
        let want: std::collections::BTreeSet<(u32, u32)> = controlled_modes(p)
            .expect("modes")
            .iter()
            .map(|m| match *m {
                crate::spirv::ir::ExecMode::RoundingModeRte { width } => {
                    (M::RoundingModeRTE as u32, u32::from(width))
                }
                crate::spirv::ir::ExecMode::DenormPreserve { width } => {
                    (M::DenormPreserve as u32, u32::from(width))
                }
                crate::spirv::ir::ExecMode::SignedZeroInfNanPreserve { width } => {
                    (M::SignedZeroInfNanPreserve as u32, u32::from(width))
                }
                other => panic!("{stem}: {other:?}"),
            })
            .collect();
        for w in [16, 32, 64] {
            let rte = want.contains(&(M::RoundingModeRTE as u32, w));
            let dp = want.contains(&(M::DenormPreserve as u32, w));
            let tied_rte = match fc.rounding_independence {
                crate::device::profile::Independence::All => w == 16,
                crate::device::profile::Independence::Only32Bit => w != 32,
                crate::device::profile::Independence::None => true,
            };
            let tied_dp = match fc.denorm_independence {
                crate::device::profile::Independence::All => w == 16,
                _ => w != 32,
            };
            assert_eq!(rte, tied_rte, "{stem}: RoundingModeRTE {w}");
            assert_eq!(dp, tied_dp, "{stem}: DenormPreserve {w}");
        }
        for (name, _) in &KERNELS {
            let what = format!("{name} on {stem}");
            let a = compile(&bc, name, p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let r = rounding(&a.module.words);
            assert_eq!(r.quantizes, 0, "{what}: {r:?}");
            assert!(r.narrowings > 0 && r.widenings > 0, "{what}: {r:?}");
            assert_eq!(r.f16_ops, usize::from(*name == "half_sum"), "{what}: {r:?}");
            // Each 16-bit operation narrows its two operands and widens its result, exactly.
            assert_eq!(
                r.uncontracted + 3 * r.f16_ops,
                r.narrowings + r.widenings,
                "{what}: {r:?}"
            );
            // Less `SignedZeroInfNanPreserve` for 32 and 64 bits, which the entry's own `float`
            // arithmetic declares.
            let declared: std::collections::BTreeSet<(u32, u32)> = r
                .modes
                .iter()
                .copied()
                .filter(|&(m, w)| m != M::SignedZeroInfNanPreserve as u32 || w == 16)
                .collect();
            assert_eq!(declared, want, "{what}");
            for (kind, width) in [
                (FloatControl::RoundingModeRte, FloatWidth::F16),
                (FloatControl::DenormPreserve, FloatWidth::F16),
                (FloatControl::SignedZeroInfNanPreserve, FloatWidth::F16),
            ] {
                assert!(
                    a.requirements
                        .properties
                        .contains(&VkProperty::FloatControl(kind, width)),
                    "{what}: {kind:?} {width:?}"
                );
            }
            for note in [
                PrecisionNote::NativeHalfConversion,
                PrecisionNote::NativeHalfQuantize,
            ] {
                assert!(!a.precision.contains(&note), "{what}: {note:?}");
            }
        }
        let run = |name: &str, x: [f32; 2]| interpret(&bc, name, p, x);
        let tie = 1.0 + 2f32.powi(-11);
        assert_eq!(run("half_narrow", [tie, 0.0]), 1.0, "{stem}: a tie to even");
        assert_eq!(run("half_narrow", [65519.0, 0.0]), 65504.0, "{stem}");
        assert_eq!(run("half_narrow", [65520.0, 0.0]), f32::INFINITY, "{stem}");
        let denormal = 3.0 * 2f32.powi(-24);
        assert_eq!(
            run("half_narrow", [denormal + 2f32.powi(-27), 0.0]),
            denormal,
            "{stem}: a half denormal"
        );
        let tiny = run("half_narrow", [-2f32.powi(-26), 0.0]);
        assert!(tiny == 0.0 && tiny.is_sign_negative(), "{stem}: {tiny}");
        assert_eq!(run("half_sum", [1.0, 2f32.powi(-11)]), 1.0, "{stem}: a sum");
        assert_eq!(run("half_sum", [1.5, 2.25]), 3.75, "{stem}: an exact sum");
    }
    for stem in ["intel-irisxe-rplp-anv", "amd-780m-radv"] {
        assert!(
            controlled.iter().any(|s| s == stem),
            "{stem}: {controlled:?}"
        );
    }
    for stem in [
        "moltenvk-m3",
        "lavapipe",
        "vk12-floor",
        "nvidia-pascal-linux",
        "nvidia-rtx50-linux",
        "nvidia-rtx5080-laptop",
    ] {
        assert!(
            !controlled.iter().any(|s| s == stem),
            "{stem}: {controlled:?}"
        );
    }
}

/// A kernel that reads a `half` from buffer 0 and stores it widened to `float` after the two
/// `float`s the other kernels read (bytes 8 to 12), and the `half` itself after that (bytes 12 to
/// 14).
fn half_load_module() -> String {
    module()
        .replace(
            "  %a = load float, ptr addrspace(1) %0, align 4\n  %b = load float, ptr addrspace(1) %p1, align 4\n  %ha = fptrunc float %a to half\n  %r = fpext half %ha to float\n",
            "  %h = load half, ptr addrspace(1) %0, align 2\n  %p6 = getelementptr inbounds half, ptr addrspace(1) %0, i64 6\n  store half %h, ptr addrspace(1) %p6, align 2\n  %r = fpext half %h to float\n",
        )
        .replace("@half_narrow(", "@half_load(")
        .replace("ptr @half_narrow,", "ptr @half_load,")
}

/// A kernel that reads a `half` from memory and rounds nothing widens its bits by the device on
/// every committed profile, with no integer decode, and records `NativeNanEncoding` (a NaN's bits
/// the device's) and selects no canonical NaN. With `shaderFloat16` the widening is the device's
/// conversion (a 16-bit storage unit it reads is read as a 16-bit float itself): under the float
/// controls (ANV, RADV) its entry declares them (`narrow::controlled_modes`, so a half denormal
/// widens exactly) and it records no other note; where the measured `MetalHalfArithmetic` reads
/// every `half` through that widening as Metal's (MoltenVK) it requires the fact; elsewhere
/// (NVIDIA, lavapipe: the device's conversion, no float controls) it records
/// `NativeHalfConversion`, whose bound holds a denormal the device flushes; where the device's
/// moves of a NaN are measured to be Metal's (`DriverDeviation::MetalNanMoves`, MoltenVK on the
/// M3 Max's identity) it requires that in place of `NativeNanEncoding`. Without it (Pascal,
/// the LunarG baseline, SwiftShader, the floor) the widening is one `UnpackHalf2x16` of the word,
/// a 16-bit unit's zero extension where the profile has 16-bit storage, with no 16-bit float type,
/// recording `NativeHalfConversion` too and requiring no fact. Each passes L3; on the interpreter
/// a half denormal, a zero's sign, `65504` and an infinity read as their values, and a NaN as a
/// NaN; the `half` stored back unchanged keeps its bits, NaNs included.
#[test]
fn a_half_read_from_memory_is_the_devices_widening_on_every_profile() {
    use crate::spv::ExecutionMode as M;
    let Some(bc) = assemble("native-half-load", &half_load_module()) else {
        return;
    };
    let mut forms = std::collections::BTreeMap::new();
    for (stem, p) in &committed_profiles().expect("profiles") {
        let form = quantize_form(p);
        *forms.entry(format!("{form:?}")).or_insert(0usize) += 1;
        let what = format!("half_load on {stem} ({form:?})");
        let (a, _, _) =
            compile_kept(&bc, "half_load", p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let r = rounding(&a.module.words);
        assert_eq!(r.narrowings, 0, "{what}: {r:?}");
        let measured = p.takes_driver_deviation(DriverDeviation::MetalHalfArithmetic);
        let fact = a
            .requirements
            .properties
            .contains(&VkProperty::MetalHalfArithmetic);
        let nan = a.precision.contains(&PrecisionNote::NativeNanEncoding);
        let moves = p.takes_driver_deviation(DriverDeviation::MetalNanMoves);
        let moved = a
            .requirements
            .properties
            .contains(&VkProperty::MetalNanMoves);
        let conversion = a.precision.contains(&PrecisionNote::NativeHalfConversion);
        match form {
            QuantizeForm::Device => {
                assert_eq!((r.widenings, r.unpacks), (0, 1), "{what}: {r:?}");
                assert!(nan && conversion && !fact, "{what}: {:?}", a.precision);
            }
            QuantizeForm::Controlled => {
                assert_eq!((r.widenings, r.unpacks), (1, 0), "{what}: {r:?}");
                for mode in [M::RoundingModeRTE, M::DenormPreserve] {
                    assert!(
                        r.modes.contains(&(mode as u32, 16)),
                        "{what}: {mode:?} {r:?}"
                    );
                }
                assert!(nan && !conversion && !fact, "{what}: {:?}", a.precision);
            }
            QuantizeForm::Converted => {
                assert_eq!((r.widenings, r.unpacks), (1, 0), "{what}: {r:?}");
                assert_eq!((nan, moved), (!moves, moves), "{what}: {:?}", a.precision);
                assert_eq!(conversion, !measured, "{what}: {:?}", a.precision);
                assert_eq!(fact, measured, "{what}: {:?}", a.requirements.properties);
            }
        }
        assert_eq!(r.canonical_nans, 0, "{what}: {r:?}");
        // A 16-bit unit read is that unit read as a `half` where the module widens by the device,
        // with no zero extension to a word between (`narrow::from_word_halves`).
        let unit_halves = match form {
            QuantizeForm::Device => 0,
            QuantizeForm::Controlled | QuantizeForm::Converted => r.unit_loads,
        };
        assert_eq!(r.unit_halves, unit_halves, "{what}: {r:?}");
        let run = |h: u16| {
            let mut bytes = vec![0; 16];
            bytes[..2].copy_from_slice(&h.to_le_bytes());
            let b = interpret_bytes(&bc, "half_load", p, bytes);
            let wide = u32::from_le_bytes([b[8], b[9], b[10], b[11]]);
            let kept = u16::from_le_bytes([b[12], b[13]]);
            assert_eq!(kept, h, "{what}: {h:#06x} stored back");
            f32::from_bits(wide)
        };
        assert_eq!(run(0x0001).to_bits(), 2f32.powi(-24).to_bits(), "{what}");
        assert_eq!(run(0x8000).to_bits(), (-0.0f32).to_bits(), "{what}");
        assert_eq!(run(0x7BFF), 65504.0, "{what}");
        assert_eq!(run(0xFC00), f32::NEG_INFINITY, "{what}");
        for h in [0x7E01, 0xFE00, 0x7C01, 0xFDFF] {
            assert!(run(h).is_nan(), "{what}: {h:#06x}");
        }
    }
    assert_eq!(forms.len(), 3, "a profile of each form: {forms:?}");
}
