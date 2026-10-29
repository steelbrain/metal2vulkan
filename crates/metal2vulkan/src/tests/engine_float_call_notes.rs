//! The precision notes of a float operation, exported, are the ones its lowering records:
//! validation reads [`float_call_notes`] and [`float_binop_notes`] to tell which operations the
//! product computes as a near-equivalent of Metal's, so the two must never diverge from what a
//! compile records.
//!
//! Every float function of the `air` math table, on `float`, `half`, and `bfloat`, called with
//! and without the `fast_` prefix and with and without fast-math flags, and every float binary
//! operator, is compiled alone (its operands loaded from a buffer and its result stored) on one
//! committed profile per class of the facts the notes read: the `float` fused multiply-add route,
//! whether the `float` functions are Metal's, and whether the device reads `f32` denormals as
//! Metal does. Each artifact's notes are exactly what the exported function answers for that call
//! in that entry's math mode. The loads and stores around it record none, but a `half` kernel's
//! widening and rounding, which the module records once by the profile's own form: its rounding's
//! note (`narrow::native_rounding`), which the test sets aside, and a module's widening of a
//! `half` from its bits or narrowing of one to them by the device (every profile), whose NaN's
//! bits are the device's (`narrow::widens_bits`, `narrow::encodes_bits`, `NativeNanEncoding`), and
//! without `shaderFloat16` its denormals too (`UnpackHalf2x16`, `PackHalf2x16`,
//! `NativeHalfConversion`), which the test adds to the operation's notes.
//!
//! `llvm-as` assembles the modules; without it the test skips visibly.

use crate::air::intrinsics::{AirIntrinsic, Elem, Math, NumTy, Operand};
use crate::air::ir::{BinOp, FastMath};
use crate::device::profile::{FloatFunctions, NanEncoding};
use crate::device::requirements::VkProperty;
use crate::device::DeviceProfile;
use crate::error::DriverDeviation;
use crate::lower::arith::fma::{self, Route};
use crate::lower::arith::libm::Mode;
use crate::lower::arith::{denormal, float_binop_notes, float_call_notes, nan, narrow, FloatFmt};
use crate::pipeline::artifact::{EntryMath, PrecisionNote};
use crate::pipeline::testing::{
    compile, exact_maths, llvm_assemble, meter, typed_module, AssembleError, Scratch,
};
use crate::spirv::testing::committed_profiles_with;
use std::collections::{BTreeMap, BTreeSet};

/// The float elements: the type token an `air` name spells, the LLVM type, and the format.
const ELEMS: [(Elem, &str, &str, FloatFmt); 3] = [
    (Elem::F32, "f32", "float", FloatFmt::F32),
    (Elem::F16, "f16", "half", FloatFmt::Half),
    (Elem::BF16, "bf16", "bfloat", FloatFmt::BFloat),
];

/// Every fast-math flag, as `fast` spells it.
fn every_flag() -> FastMath {
    FastMath::ALL
        .iter()
        .fold(FastMath::NONE, |all, (flag, _)| all.union(*flag))
}

/// The LLVM spelling of `ty`.
fn llvm_ty(ty: NumTy) -> String {
    let elem = match ty.elem {
        Elem::F32 => "float",
        Elem::F16 => "half",
        Elem::BF16 => "bfloat",
        Elem::I32 => "i32",
        other => panic!("no operand of {other:?} in the float table"),
    };
    if ty.lanes == 1 {
        elem.to_owned()
    } else {
        format!("<{} x {elem}>", ty.lanes)
    }
}

/// The bytes of `ty`, its alignment here.
fn bytes(ty: NumTy) -> u32 {
    ty.elem.bits() / 8 * ty.lanes
}

/// What one kernel computes.
enum Op {
    /// A call of `math`, spelled `name`, with fast-math flags `fmf`.
    Call {
        name: String,
        math: Math,
        fmf: FastMath,
    },
    /// A binary operator on `fmt` with fast-math flags `fmf`.
    Binop {
        op: BinOp,
        fmt: FloatFmt,
        fmf: FastMath,
    },
    /// Nothing between the load and the store.
    Copy,
}

struct Kernel {
    name: String,
    body: String,
    /// The declaration of the function it calls.
    decl: Option<String>,
    op: Op,
}

/// The `fast` flag of a call or operator with flags `fmf`, as LLVM spells it.
fn fmf_text(fmf: FastMath) -> &'static str {
    if fmf == every_flag() {
        "fast "
    } else {
        assert_eq!(fmf, FastMath::NONE, "only every flag or none is spelled");
        ""
    }
}

/// The kernels on element `elem` (`ll` in LLVM): every float function the table spells for it,
/// every float binary operator, and the bare load and store. `p` is a profile the plain-function
/// check reads.
fn kernels(elem: Elem, token: &str, ll: &str, fmt: FloatFmt, p: &DeviceProfile) -> Vec<Kernel> {
    let mut out = Vec::new();
    let scalar = NumTy::scalar(elem);
    let load_store = |body: &str, result: &str, ty: NumTy| {
        format!(
            "{body}  store {} {result}, ptr addrspace(1) %0, align {}\n",
            llvm_ty(ty),
            bytes(ty)
        )
    };
    out.push(Kernel {
        name: format!("copy_{token}"),
        body: load_store(
            &format!(
                "  %x = load {ll}, ptr addrspace(1) %0, align {}\n",
                bytes(scalar)
            ),
            "%x",
            scalar,
        ),
        decl: None,
        op: Op::Copy,
    });
    let mut stems = 0;
    for stem in Math::float_stems() {
        // `dot` is spelled on vectors only.
        let parsed = [token.to_owned(), format!("v2{token}")]
            .into_iter()
            .find_map(|ty| {
                let name = format!("air.{stem}.{ty}");
                match AirIntrinsic::parse(&name) {
                    Some(AirIntrinsic::Math(math)) => Some((ty, math)),
                    _ => None,
                }
            });
        let (ty_token, _) = parsed.unwrap_or_else(|| panic!("air.{stem}.{token} does not parse"));
        stems += 1;
        for (prefix, fmf) in [
            ("", FastMath::NONE),
            ("", every_flag()),
            ("fast_", every_flag()),
            ("fast_", FastMath::NONE),
        ] {
            let name = format!("air.{prefix}{stem}.{ty_token}");
            let Some(AirIntrinsic::Math(math)) = AirIntrinsic::parse(&name) else {
                panic!("{name} does not parse");
            };
            let sig = math.signature();
            let params: Option<Vec<NumTy>> = sig
                .params()
                .iter()
                .map(|p| match p {
                    Operand::Num(ty) => Some(*ty),
                    _ => None,
                })
                .collect();
            let (Some(params), Operand::Num(ret)) = (params, sig.ret) else {
                // `sincos` stores through a pointer: it is selected with its store, not as a
                // float function.
                assert_eq!(
                    float_call_notes(math, fmf, p),
                    None,
                    "{name} is not a plain float function"
                );
                continue;
            };
            let mut body = String::new();
            let mut args = Vec::new();
            for (i, p) in params.iter().enumerate() {
                let buffer = if p.elem.is_float() { 0 } else { 1 };
                body.push_str(&format!(
                    "  %p{i} = load {}, ptr addrspace({}) %{buffer}, align {}\n",
                    llvm_ty(*p),
                    1,
                    bytes(*p)
                ));
                args.push(format!("{} %p{i}", llvm_ty(*p)));
            }
            body.push_str(&format!(
                "  %r = call {}{} @{name}({})\n",
                fmf_text(fmf),
                llvm_ty(ret),
                args.join(", ")
            ));
            let decl = Some(format!(
                "declare {} @{name}({})",
                llvm_ty(ret),
                params
                    .iter()
                    .map(|p| llvm_ty(*p))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            out.push(Kernel {
                name: format!(
                    "{}{stem}_{ty_token}_{}",
                    prefix,
                    if fmf.is_empty() {
                        "flagless"
                    } else {
                        "flagged"
                    }
                ),
                body: load_store(&body, "%r", ret),
                decl,
                op: Op::Call { name, math, fmf },
            });
        }
    }
    assert!(stems > 40, "the float table has {stems} stems");
    for op in BinOp::ALL.into_iter().filter(|op| op.is_float()) {
        for fast in [false, true] {
            let text = format!("{op:?}").to_lowercase();
            let fmf = if fast { every_flag() } else { FastMath::NONE };
            let body = format!(
                "  %a = load {ll}, ptr addrspace(1) %0, align {n}\n  %b = load {ll}, ptr addrspace(1) %0, align {n}\n  %r = {text} {}{ll} %a, %b\n",
                fmf_text(fmf),
                n = bytes(scalar),
            );
            out.push(Kernel {
                name: format!(
                    "{text}_{token}_{}",
                    if fast { "flagged" } else { "flagless" }
                ),
                body: load_store(&body, "%r", scalar),
                decl: None,
                op: Op::Binop { op, fmt, fmf },
            });
        }
    }
    out
}

/// The module of `kernels`, each taking a float buffer and an integer buffer.
fn module(kernels: &[Kernel]) -> String {
    let mut s = String::from(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n",
    );
    for k in kernels {
        s.push_str(&format!(
            "define void @{}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1) {{\nentry:\n{}  ret void\n}}\n\n",
            k.name, k.body
        ));
    }
    let decls: BTreeSet<&String> = kernels.iter().filter_map(|k| k.decl.as_ref()).collect();
    for d in decls {
        s.push_str(d);
        s.push('\n');
    }
    let refs: Vec<String> = (0..kernels.len()).map(|i| format!("!{}", 20 + i)).collect();
    s.push_str(&format!(
        "\n!llvm.module.flags = !{{!0}}\n!air.kernel = !{{{}}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n",
        refs.join(", ")
    ));
    s.push_str("!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n!10 = !{}\n!11 = !{!12, !13}\n!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"out\"}\n!13 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"int\", !\"air.arg_name\", !\"ints\"}\n!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n");
    for (i, k) in kernels.iter().enumerate() {
        s.push_str(&format!("!{} = !{{ptr @{}, !10, !11}}\n", 20 + i, k.name));
    }
    s
}

fn assemble(token: &str, chunk: usize, text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new(&format!("m2v-float-call-notes-{token}-{chunk}"));
    let dir = scratch.path();
    let ll = dir.join("notes.ll");
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

/// What the notes of a float operation read of a profile: its `float` fused multiply-add route,
/// whether its `float` functions are Metal's, and whether the device reads `f32` denormals as Metal
/// does.
type NoteClass = (Route, bool, bool);

/// The note class of profile `p`.
fn note_class(p: &DeviceProfile) -> NoteClass {
    (
        fma::float_route(p),
        p.fields().float_functions == FloatFunctions::Metal,
        denormal::device_flushes(p),
    )
}

/// One committed profile per note class (the first by name).
fn one_profile_per_class() -> BTreeMap<NoteClass, (String, DeviceProfile)> {
    let mut out = BTreeMap::new();
    for (stem, p) in committed_profiles_with(exact_maths()).expect("profiles") {
        out.entry(note_class(&p)).or_insert((stem, p));
    }
    let routes: BTreeSet<Route> = out.keys().map(|(route, _, _)| *route).collect();
    assert_eq!(
        routes,
        BTreeSet::from([Route::Khr, Route::Native]),
        "every float route is some committed profile's"
    );
    assert!(
        out.keys().any(|(_, metal, _)| *metal),
        "a committed profile's float functions are Metal's"
    );
    out
}

/// The mode of the `float` functions of an entry on `p` whose one float operation carries flags
/// `fmf` (`None` where the profile's are the device's): precise where the operation lacks one of
/// `nnan`, `ninf`, and `nsz` and the profile declares `SignedZeroInfNanPreserve` for `f32`, the
/// container of every format here (`arith::fp_env`, `libm::module_mode`).
fn entry_mode(p: &DeviceProfile, fmf: FastMath) -> Option<Mode> {
    let relaxed = FastMath::NNAN.union(FastMath::NINF).union(FastMath::NSZ);
    let preserves =
        !fmf.contains(relaxed) && p.fields().float_controls.f32.signed_zero_inf_nan_preserve;
    (p.fields().float_functions == FloatFunctions::Metal).then_some(if preserves {
        Mode::Precise
    } else {
        Mode::Fast
    })
}

#[test]
fn exported_float_notes_are_the_ones_a_compile_records() {
    let profiles = one_profile_per_class();
    let any = &profiles.values().next().expect("a profile").1;
    let mut compared = 0;
    for (elem, token, ll, fmt) in ELEMS {
        // A compile decodes its whole module: small modules keep the test fast.
        for (chunk, kernels) in kernels(elem, token, ll, fmt, any).chunks(16).enumerate() {
            let Some(bitcode) = assemble(token, chunk, &module(kernels)) else {
                return;
            };
            for (class, (stem, p)) in &profiles {
                for k in kernels {
                    let what = format!("{} on {stem} ({class:?})", k.name);
                    let want = match &k.op {
                        Op::Copy => Some(BTreeSet::new()),
                        Op::Call { name, math, fmf } => {
                            let notes = float_call_notes(*math, *fmf, p);
                            assert!(notes.is_some(), "{what}: {name} is not a float function");
                            notes.map(|notes| notes.in_entry(entry_mode(p, *fmf)))
                        }
                        Op::Binop { op, fmt, fmf } => float_binop_notes(*op, *fmt, *fmf, p)
                            .map(|notes| notes.in_entry(entry_mode(p, *fmf))),
                    };
                    let mut want = want.unwrap_or_else(|| panic!("{what}: no exported notes"));
                    let a =
                        compile(&bitcode, &k.name, p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                    let mut got: BTreeSet<PrecisionNote> = a.precision.iter().copied().collect();
                    if matches!(elem, Elem::F16) {
                        let form = narrow::quantize_form(p);
                        if let Some(own) = narrow::native_rounding(form) {
                            got.remove(&own);
                        }
                        let (_, m) = typed_module(&bitcode, &k.name, p)
                            .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                        let crosses = narrow::widens_bits(&m, &mut meter())
                            .and_then(|w| Ok(w || narrow::encodes_bits(&m, &mut meter())?))
                            .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                        if crosses {
                            want.insert(PrecisionNote::NativeNanEncoding);
                            if form == crate::spirv::builder::QuantizeForm::Device {
                                want.insert(PrecisionNote::NativeHalfConversion);
                            }
                        }
                    }
                    assert_eq!(got, want, "{what}");
                    compared += 1;
                }
            }
        }
    }
    eprintln!("compared {compared}");
    assert!(compared > 0);
}

/// The binary operators' notes beside the functions': a remainder and a division are the device's
/// arithmetic, and a `float` operator leaves the device its denormals where the profile does not
/// take `DriverDeviation::MetalFloatDenormals`, and a remainder and a division on every profile.
/// Metal's precise division (an `fdiv` on `float` without `arcp`, or on `half` under every flag
/// set) is the device's `OpFDiv`, noted `NativeDivision`. A `float` remainder without every flag is
/// the device's `OpFRem`, and with every flag Metal's `fast::fmod`, whose notes it records. Metal's
/// fast division (an `fdiv` on `float` with `arcp`) is the device's `OpFDiv` too, and records the
/// device's arithmetic, its division and its denormals, but where the profile's `float` functions
/// are Metal's, in an entry Metal compiles with fast math, whose quotient is Metal's own. Metal
/// answers its canonical NaN for a `float` or `half` `+ - * /` and a `float` remainder; each leaves
/// that NaN to the device, noted `NativeNanEncoding` where the profile is not measured to answer
/// Metal's (`NanEncoding::Native`), a fast `float` division but in Metal's own fast-math entry.
/// No operator records `DeviceNanEncoding`.
#[test]
fn a_float_division_is_the_devices_arithmetic_unless_it_is_metals_precise_one() {
    for (class, (stem, p)) in one_profile_per_class() {
        let (_, metal_functions, _) = class;
        let flushes = p.takes_driver_deviation(DriverDeviation::MetalFloatDenormals);
        let native_nans = nan::encoding(&p) == NanEncoding::Native;
        for (_, _, _, fmt) in ELEMS {
            for op in BinOp::ALL {
                for fmf in [FastMath::NONE, FastMath::ARCP, every_flag()] {
                    let notes = float_binop_notes(op, fmt, fmf, &p);
                    assert_eq!(notes.is_some(), op.is_float(), "{op:?}");
                    let Some(notes) = notes else { continue };
                    for mode in [Mode::Fast, Mode::Precise] {
                        let metal = metal_functions.then_some(mode);
                        let notes = notes.clone().in_entry(metal);
                        let single = fmt == FloatFmt::F32;
                        let precise = op == BinOp::FDiv
                            && (fmt == FloatFmt::Half || (single && !fmf.contains(FastMath::ARCP)));
                        let what = format!("{op:?} on {fmt:?} under {fmf:?} on {stem}, {metal:?}");
                        assert!(!notes.contains(&PrecisionNote::DeviceNanEncoding), "{what}");
                        if precise {
                            let want: BTreeSet<_> = [PrecisionNote::NativeDivision]
                                .into_iter()
                                .chain(single.then_some(PrecisionNote::DeviceDenormals))
                                .chain(native_nans.then_some(PrecisionNote::NativeNanEncoding))
                                .collect();
                            assert_eq!(notes, want, "{what}");
                            continue;
                        }
                        if op == BinOp::FRem && single && fmf == every_flag() {
                            let Some(AirIntrinsic::Math(fmod)) =
                                AirIntrinsic::parse("air.fast_fmod.f32")
                            else {
                                panic!("air.fast_fmod.f32 does not parse");
                            };
                            let want = float_call_notes(fmod, fmf, &p)
                                .expect("fast fmod is a float function")
                                .in_entry(metal);
                            assert_eq!(notes, want, "{what}");
                            continue;
                        }
                        if op == BinOp::FDiv && single {
                            let want = if metal == Some(Mode::Fast) {
                                BTreeSet::new()
                            } else {
                                [
                                    PrecisionNote::DeviceArithmetic,
                                    PrecisionNote::NativeDivision,
                                    PrecisionNote::DeviceDenormals,
                                ]
                                .into_iter()
                                .chain(native_nans.then_some(PrecisionNote::NativeNanEncoding))
                                .collect()
                            };
                            assert_eq!(notes, want, "{what}");
                            continue;
                        }
                        assert_eq!(
                            notes.contains(&PrecisionNote::DeviceArithmetic),
                            matches!(op, BinOp::FDiv | BinOp::FRem),
                            "{what}"
                        );
                        // Metal's canonical NaN: a `float` or `half` `+ - *`, and a `float`
                        // remainder.
                        let canonical = (matches!(fmt, FloatFmt::F32 | FloatFmt::Half)
                            && matches!(op, BinOp::FAdd | BinOp::FSub | BinOp::FMul))
                            || (op == BinOp::FRem && single);
                        assert_eq!(
                            notes.contains(&PrecisionNote::NativeNanEncoding),
                            native_nans && canonical,
                            "{what}"
                        );
                        assert_eq!(
                            notes.contains(&PrecisionNote::DeviceDenormals),
                            single && (op == BinOp::FRem || !flushes),
                            "{what}"
                        );
                    }
                }
            }
        }
    }
}

/// Metal's precise `fmod`, on `float` and `half`, answers the one right value
/// (`libm::Fun::correctly_rounded`: exact by MSL Tables 8.1 and 8.3), which its routine computes
/// on every device: it records no precision note of the function's value on any profile class, in
/// either mode, with or without flags; a `half` one records `NativeTranscendentals` for its zero
/// remainder's sign, its `float` remainder's, where Metal's `half` answers `+0`
/// (`FloatFn::half_zero_of_float`). Precise `sqrt` and `rsqrt`, which Metal documents correctly rounded, are the
/// device's `Sqrt` and `InverseSqrt`: off Metal's own functions they record the device's precision
/// (`NativeTranscendentals`), as precise `exp`, which Apple bounds by ulps, does.
#[test]
fn precise_fmod_records_no_precision_note_and_the_roots_the_devices() {
    let function_notes = [
        PrecisionNote::DeviceArithmetic,
        PrecisionNote::DeviceTranscendentals,
        PrecisionNote::NativeTranscendentals,
    ];
    let parse = |name: &str| match AirIntrinsic::parse(name) {
        Some(AirIntrinsic::Math(call)) => call,
        _ => panic!("{name} is a math call"),
    };
    for ((_, metal_functions, _), (stem, p)) in one_profile_per_class() {
        for name in ["air.fmod.f32", "air.fmod.f16"] {
            for fmf in [FastMath::NONE, every_flag()] {
                for math in [EntryMath::Fast, EntryMath::Precise] {
                    let notes = PrecisionNote::of_math_call(parse(name), fmf, &p, math)
                        .unwrap_or_else(|| panic!("{name} is a float function"));
                    let half_zero = name.ends_with("f16");
                    assert!(
                        function_notes.iter().all(|note| {
                            notes.contains(note)
                                == (half_zero && *note == PrecisionNote::NativeTranscendentals)
                        }),
                        "{name} under {fmf:?} on {stem}, {math:?}: {notes:?}"
                    );
                }
            }
        }
        if !metal_functions {
            for name in [
                "air.sqrt.f32",
                "air.rsqrt.f32",
                "air.sqrt.f16",
                "air.rsqrt.f16",
                "air.exp.f32",
            ] {
                let notes = PrecisionNote::of_math_call(
                    parse(name),
                    FastMath::NONE,
                    &p,
                    EntryMath::Precise,
                )
                .unwrap_or_else(|| panic!("{name} is a float function"));
                assert!(
                    notes.contains(&PrecisionNote::NativeTranscendentals),
                    "{name} on {stem}: {notes:?}"
                );
            }
        }
    }
}

/// A kernel `name` of three `float` operands `%p0` to `%p2`, loaded from the float buffer, whose
/// `compute` defines `%r`, which it stores.
fn float_kernel(name: &str, compute: String, decl: Option<String>, op: Op) -> Kernel {
    let mut body = String::new();
    for i in 0..3 {
        body.push_str(&format!(
            "  %p{i} = load float, ptr addrspace(1) %0, align 4\n"
        ));
    }
    body.push_str(&compute);
    body.push_str("  store float %r, ptr addrspace(1) %0, align 4\n");
    Kernel {
        name: name.to_owned(),
        body,
        decl,
        op,
    }
}

/// A kernel `name` that calls `float` function `symbol` of `arity` operands with flags `fmf`.
fn call_kernel(name: &str, symbol: &str, arity: usize, fmf: FastMath) -> Kernel {
    let Some(AirIntrinsic::Math(math)) = AirIntrinsic::parse(symbol) else {
        panic!("{symbol} is a math call");
    };
    let args = (0..arity)
        .map(|i| format!("float %p{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let params = vec!["float"; arity].join(", ");
    float_kernel(
        name,
        format!("  %r = call {}float @{symbol}({args})\n", fmf_text(fmf)),
        Some(format!("declare float @{symbol}({params})")),
        Op::Call {
            name: symbol.to_owned(),
            math,
            fmf,
        },
    )
}

/// On MoltenVK (the M3 Max's and the M4's profiles: `float` functions Metal's own, `f32` denormals
/// read as Metal does), an entry Metal compiles with fast math records no note for its fast
/// division, which is Metal's own there, nor for a call the device answers as Metal's own
/// function, `fast::atan2` among them; an entry Metal compiles without fast math (a division
/// without `nnan` and `ninf`, a call without flags) still records them, and `fast::sinpi`, the
/// device's near-equivalent, records its note in either. `fast::sinpi` records no
/// `DeviceDenormals` there and requires the measured fact; `mix` is the device's `FMix`, which
/// reaches Metal's own `mix` there, with no note and no fact. Off MoltenVK `mix` is the device's
/// `FMix`, with or without Metal's denormals: noted `NativeMix` and `DeviceDenormals`, requiring
/// no fact.
#[test]
fn a_fast_math_entry_on_moltenvk_records_no_note_for_metals_own_operations() {
    let every = every_flag();
    let mut kernels = vec![
        float_kernel(
            "fdiv_fast",
            "  %r = fdiv fast float %p0, %p1\n".to_owned(),
            None,
            Op::Binop {
                op: BinOp::FDiv,
                fmt: FloatFmt::F32,
                fmf: every,
            },
        ),
        // `arcp` without `nnan` and `ninf`: the entry keeps NaNs, so Metal compiles it precise.
        float_kernel(
            "fdiv_preserving",
            "  %r = fdiv reassoc nsz arcp contract afn float %p0, %p1\n".to_owned(),
            None,
            Op::Binop {
                op: BinOp::FDiv,
                fmt: FloatFmt::F32,
                fmf: FastMath::REASSOC
                    .union(FastMath::NSZ)
                    .union(FastMath::ARCP)
                    .union(FastMath::CONTRACT)
                    .union(FastMath::AFN),
            },
        ),
        call_kernel("sinpi_fast", "air.fast_sinpi.f32", 1, every),
        call_kernel("mix_fast", "air.mix.f32", 3, every),
    ];
    let own = [
        ("air.fast_sin.f32", 1),
        ("air.fast_atan2.f32", 2),
        ("air.fast_sqrt.f32", 1),
        ("air.fast_rsqrt.f32", 1),
        ("air.fast_exp.f32", 1),
        ("air.fast_powr.f32", 2),
    ];
    for (symbol, arity) in own {
        let stem = symbol
            .trim_start_matches("air.fast_")
            .trim_end_matches(".f32");
        kernels.push(call_kernel(&format!("{stem}_fast"), symbol, arity, every));
        kernels.push(call_kernel(
            &format!("{stem}_preserving"),
            symbol,
            arity,
            FastMath::NONE,
        ));
    }
    let Some(bitcode) = assemble("moltenvk", 0, &module(&kernels)) else {
        return;
    };
    let profiles = committed_profiles_with(exact_maths()).expect("profiles");
    let by_stem = |stem: &str| {
        profiles
            .iter()
            .find(|(s, _)| s == stem)
            .map(|(_, p)| p)
            .unwrap_or_else(|| panic!("committed profile {stem}"))
    };
    let notes_of = |p: &DeviceProfile, kernel: &str| {
        let a = compile(&bitcode, kernel, p).unwrap_or_else(|e| panic!("{kernel}: {e:?}"));
        let notes: BTreeSet<PrecisionNote> = a.precision.iter().copied().collect();
        (
            notes,
            a.requirements
                .properties
                .contains(&VkProperty::MetalFloatDenormals),
        )
    };
    let device_notes = [
        PrecisionNote::DeviceArithmetic,
        PrecisionNote::DeviceTranscendentals,
        PrecisionNote::DeviceDenormals,
    ];
    {
        let stem = "moltenvk-m3";
        let p = by_stem(stem);
        assert_eq!(note_class(p), (Route::Khr, true, true), "{stem}");
        let (fast, _) = notes_of(p, "fdiv_fast");
        assert!(fast.is_empty(), "fdiv_fast on {stem}: {fast:?}");
        let (preserving, _) = notes_of(p, "fdiv_preserving");
        assert!(
            preserving.contains(&PrecisionNote::DeviceArithmetic),
            "fdiv_preserving on {stem}: {preserving:?}"
        );
        for (symbol, _) in own {
            let name = symbol
                .trim_start_matches("air.fast_")
                .trim_end_matches(".f32");
            let (fast, _) = notes_of(p, &format!("{name}_fast"));
            assert!(
                device_notes.iter().all(|note| !fast.contains(note)),
                "{name}_fast on {stem}: {fast:?}"
            );
            let (preserving, _) = notes_of(p, &format!("{name}_preserving"));
            assert!(
                preserving.contains(&PrecisionNote::DeviceTranscendentals),
                "{name}_preserving on {stem}: {preserving:?}"
            );
        }
        let (sinpi, requires) = notes_of(p, "sinpi_fast");
        assert!(
            sinpi.contains(&PrecisionNote::DeviceTranscendentals)
                && !sinpi.contains(&PrecisionNote::DeviceDenormals)
                && requires,
            "sinpi_fast on {stem}: {sinpi:?}, requires the fact: {requires}"
        );
        let (mix, _) = notes_of(p, "mix_fast");
        assert!(
            !mix.contains(&PrecisionNote::DeviceDenormals)
                && !mix.contains(&PrecisionNote::NativeMix),
            "mix_fast on {stem}: {mix:?}"
        );
    }
    // Without Metal's functions, with or without Metal's denormals, `mix` is the device's `FMix`,
    // and the fast division keeps its note.
    for denormals in [true, false] {
        let Some((stem, p)) = profiles.iter().find(|(_, p)| {
            let (_, metal, flushes) = note_class(p);
            !metal && flushes == denormals
        }) else {
            panic!("no committed profile of the device's functions, Metal's denormals {denormals}");
        };
        let (mix, requires) = notes_of(p, "mix_fast");
        assert!(
            mix.contains(&PrecisionNote::NativeMix)
                && mix.contains(&PrecisionNote::DeviceDenormals)
                && !requires,
            "mix_fast on {stem}: {mix:?}, requires the fact: {requires}"
        );
        let (fast, _) = notes_of(p, "fdiv_fast");
        assert!(
            fast.contains(&PrecisionNote::DeviceArithmetic),
            "fdiv_fast on {stem}: {fast:?}"
        );
    }
}
