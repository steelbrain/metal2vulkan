//! `half` values held at 16 bits (`spirv::half_hold`): under the 16-bit float controls
//! (`QuantizeForm::Controlled`, the Iris Xe's and RADV's profiles) a kernel that computes `half`
//! in 16 bits holds its `half` values in 16-bit floats, its minimum, maximum, clamp, dot product
//! and `mix` the device's 16-bit instructions, its loop-carried `half` and its thread array of
//! `half` in 16-bit variables, and narrows no exact half; it records `NativeDot` and `NativeMix`,
//! whose bounds cover the 16-bit dot product and `mix`. Where the profile rounds `half` through
//! the conversion alone (`QuantizeForm::Converted`, lavapipe) nothing is held there. Both answer
//! the kernel's exact result on the interpreter.
//!
//! The kernel reads four `float`s from buffer 0 (`a`, `b`, `c`, `t`), narrows them to `half`, and
//! stores one `float` after them: with `v1 = (a, b, c)` and `v2 = (b, c, a)`, the clamp of `v1`
//! between their minimum and maximum dotted with `v2`, plus the first lane of `mix(v1, v2, t)`,
//! summed four times in a loop that writes each partial sum to a thread array, plus that array's
//! third element.
//!
//! Under the same controls a `half` division and square root are the device's 16-bit `OpFDiv` and
//! `Sqrt` (`lower::arith::divide`, `lower::arith::math`), recording `NativeDivision` and
//! `NativeTranscendentals`; on lavapipe the `float` ones rounded. A second kernel divides a `half2` and
//! a `half`, a divisor past `2^14` and a denormal one among them (each scaled into Vulkan's range
//! for the 16-bit quotient), and takes a square root. `llvm-as` assembles the modules; without it
//! the tests skip visibly.

use crate::air::{self, SpecRequest};
use crate::device::{DevicePolicy, DeviceProfile};
use crate::lower::arith::narrow::quantize_form;
use crate::lower::emit::emit_module;
use crate::lower::select::{select, LowerInputs};
use crate::memory::static_root::analyze_memory;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, meter, profile, AssembleError, Scratch,
};
use crate::pipeline::PrecisionNote;
use crate::spirv::builder::QuantizeForm;
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::ir::{BinOp, ConvOp, GlslOp, Inst, Type, TypedModule};

const KERNEL: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @half_hold(ptr addrspace(1) noundef "air-buffer-no-alias" %0) {
entry:
  %arr = alloca [4 x half], align 2
  %p1 = getelementptr inbounds float, ptr addrspace(1) %0, i64 1
  %p2 = getelementptr inbounds float, ptr addrspace(1) %0, i64 2
  %p3 = getelementptr inbounds float, ptr addrspace(1) %0, i64 3
  %a = load float, ptr addrspace(1) %0, align 4
  %b = load float, ptr addrspace(1) %p1, align 4
  %c = load float, ptr addrspace(1) %p2, align 4
  %t = load float, ptr addrspace(1) %p3, align 4
  %ha = fptrunc float %a to half
  %hb = fptrunc float %b to half
  %hc = fptrunc float %c to half
  %ht = fptrunc float %t to half
  %v1a = insertelement <3 x half> poison, half %ha, i64 0
  %v1b = insertelement <3 x half> %v1a, half %hb, i64 1
  %v1 = insertelement <3 x half> %v1b, half %hc, i64 2
  %v2 = shufflevector <3 x half> %v1, <3 x half> poison, <3 x i32> <i32 1, i32 2, i32 0>
  %ts0 = insertelement <3 x half> poison, half %ht, i64 0
  %ts = shufflevector <3 x half> %ts0, <3 x half> poison, <3 x i32> zeroinitializer
  %mn = call <3 x half> @air.fmin.v3f16(<3 x half> %v1, <3 x half> %v2)
  %mx = call <3 x half> @air.fmax.v3f16(<3 x half> %v1, <3 x half> %v2)
  %cl = call <3 x half> @air.clamp.v3f16(<3 x half> %v1, <3 x half> %mn, <3 x half> %mx)
  %d = call half @air.dot.v3f16(<3 x half> %cl, <3 x half> %v2)
  %mixv = call <3 x half> @air.mix.v3f16(<3 x half> %v1, <3 x half> %v2, <3 x half> %ts)
  %m0 = extractelement <3 x half> %mixv, i64 0
  %e = fadd half %m0, %d
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
  %acc = phi half [ 0xH0000, %entry ], [ %acc1, %loop ]
  %acc1 = fadd half %acc, %e
  %slot = getelementptr inbounds [4 x half], ptr %arr, i32 0, i32 %i
  store half %acc1, ptr %slot, align 2
  %i1 = add i32 %i, 1
  %done = icmp eq i32 %i1, 4
  br i1 %done, label %exit, label %loop

exit:
  %s2 = getelementptr inbounds [4 x half], ptr %arr, i32 0, i32 2
  %x2 = load half, ptr %s2, align 2
  %hr = fadd half %acc1, %x2
  %r = fpext half %hr to float
  %po = getelementptr inbounds float, ptr addrspace(1) %0, i64 4
  store float %r, ptr addrspace(1) %po, align 4
  ret void
}

declare <3 x half> @air.fmin.v3f16(<3 x half>, <3 x half>)
declare <3 x half> @air.fmax.v3f16(<3 x half>, <3 x half>)
declare <3 x half> @air.clamp.v3f16(<3 x half>, <3 x half>, <3 x half>)
declare half @air.dot.v3f16(<3 x half>, <3 x half>)
declare <3 x half> @air.mix.v3f16(<3 x half>, <3 x half>, <3 x half>)

!llvm.module.flags = !{!0}
!air.kernel = !{!20}
!air.version = !{!17}
!air.language_version = !{!18}

!0 = !{i32 7, !"air.max_device_buffers", i32 31}
!10 = !{}
!11 = !{!12}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"io"}
!17 = !{i32 2, i32 8, i32 0}
!18 = !{!"Metal", i32 4, i32 0, i32 0}
!20 = !{ptr @half_hold, !10, !11}
"#;

const DIVIDE: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @half_divide(ptr addrspace(1) noundef "air-buffer-no-alias" %0) {
entry:
  %p1 = getelementptr inbounds float, ptr addrspace(1) %0, i64 1
  %p2 = getelementptr inbounds float, ptr addrspace(1) %0, i64 2
  %p3 = getelementptr inbounds float, ptr addrspace(1) %0, i64 3
  %a = load float, ptr addrspace(1) %0, align 4
  %b = load float, ptr addrspace(1) %p1, align 4
  %c = load float, ptr addrspace(1) %p2, align 4
  %e = load float, ptr addrspace(1) %p3, align 4
  %ha = fptrunc float %a to half
  %hb = fptrunc float %b to half
  %hc = fptrunc float %c to half
  %he = fptrunc float %e to half
  %hf = fmul half %hb, %hb
  %n0 = insertelement <2 x half> poison, half %ha, i64 0
  %n = insertelement <2 x half> %n0, half %hf, i64 1
  %d0 = insertelement <2 x half> poison, half %hb, i64 0
  %d = insertelement <2 x half> %d0, half %he, i64 1
  %q = fdiv <2 x half> %n, %d
  %q1 = extractelement <2 x half> %q, i64 0
  %q3 = extractelement <2 x half> %q, i64 1
  %s = call half @air.sqrt.f16(half %q1)
  %q2 = fdiv half %ha, %hc
  %u0 = insertelement <2 x float> undef, float %a, i64 0
  %u = insertelement <2 x float> %u0, float %c, i64 1
  %w = fmul <2 x float> %u, %u
  %w0 = extractelement <2 x float> %w, i64 0
  %w1 = extractelement <2 x float> %w, i64 1
  %ws = fadd float %w0, %w1
  %r1 = fpext half %q1 to float
  %r2 = fpext half %s to float
  %r3 = fpext half %q2 to float
  %r4 = fpext half %q3 to float
  %o1 = getelementptr inbounds float, ptr addrspace(1) %0, i64 4
  %o2 = getelementptr inbounds float, ptr addrspace(1) %0, i64 5
  %o3 = getelementptr inbounds float, ptr addrspace(1) %0, i64 6
  %o4 = getelementptr inbounds float, ptr addrspace(1) %0, i64 7
  %o5 = getelementptr inbounds float, ptr addrspace(1) %0, i64 8
  store float %r1, ptr addrspace(1) %o1, align 4
  store float %r2, ptr addrspace(1) %o2, align 4
  store float %r3, ptr addrspace(1) %o3, align 4
  store float %r4, ptr addrspace(1) %o4, align 4
  store float %ws, ptr addrspace(1) %o5, align 4
  ret void
}

declare half @air.sqrt.f16(half)

!llvm.module.flags = !{!0}
!air.kernel = !{!20}
!air.version = !{!17}
!air.language_version = !{!18}

!0 = !{i32 7, !"air.max_device_buffers", i32 31}
!10 = !{}
!11 = !{!12}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"io"}
!17 = !{i32 2, i32 8, i32 0}
!18 = !{!"Metal", i32 4, i32 0, i32 0}
!20 = !{ptr @half_divide, !10, !11}
"#;

/// The inputs `a`, `b`, `c` and `t`, and the kernel's exact result of them: every intermediate
/// is a `half` with no rounding, so the 16-bit and the `float` forms answer it alike.
const INPUTS: [f32; 4] = [1.5, 0.25, 2.0, 0.5];
const RESULT: f32 = 33.25;

/// The division kernel's inputs `a`, `b`, `c` (past `2^14`) and `e` (a `half` denormal, `2^-16`),
/// and its exact results: `a / b`, `sqrt(a / b)`, `a / c` (the smallest normal `half`) and
/// `b * b / e`; then, in `float`, `a * a + c * c` of a vector inserted into an undefined one.
const DIVIDE_INPUTS: [f32; 4] = [2.0, 0.25, 32768.0, 1.0 / 65536.0];
const QUOTIENTS: [f32; 5] = [8.0, 2.828_125, 1.0 / 16384.0, 4096.0, 1_073_741_824.0];

/// The bitcode of module `src`, or `None` (skipped visibly) without `llvm-as`.
fn bitcode_of(src: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new("m2v-half-hold");
    let ll = scratch.path().join("half_hold.ll");
    std::fs::write(&ll, src).expect("write");
    match llvm_assemble(&ll) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected the module: {e}"),
    }
}

/// The kernel's bitcode, or `None` (skipped visibly) without `llvm-as`.
fn bitcode() -> Option<Vec<u8>> {
    bitcode_of(KERNEL)
}

/// The kernel's module for `p`, and its push range's end.
fn emitted(bitcode: &[u8], p: &DeviceProfile) -> (TypedModule, usize) {
    emitted_entry(bitcode, "half_hold", p)
}

/// Entry `entry`'s module for `p`, and its push range's end.
fn emitted_entry(bitcode: &[u8], entry: &str, p: &DeviceProfile) -> (TypedModule, usize) {
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
    let push = plan.push.push_range.map_or(4, |(_, end)| end as usize);
    (module, push)
}

/// The module's result on the interpreter.
fn interpret(m: &TypedModule, push: usize) -> f32 {
    let out = interpret_on(m, push, &INPUTS, 1);
    out.first().copied().expect("result")
}

/// The `n` `float`s the module stores after `inputs` on the interpreter.
fn interpret_on(m: &TypedModule, push: usize, inputs: &[f32], n: usize) -> Vec<f32> {
    let bytes = inputs
        .iter()
        .copied()
        .chain(std::iter::repeat_n(0.0, n))
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
        push: vec![0; push],
        spec: Default::default(),
    };
    let out = run(m, &Config::default(), res).unwrap_or_else(|e| panic!("interpreter: {e}"));
    let b = &out.buffers[0];
    (inputs.len()..inputs.len() + n)
        .map(|i| f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]))
        .collect()
}

/// The width of float scalar or vector type `t`, or of the elements of arrays of them.
fn float_width(m: &TypedModule, t: crate::spirv::ids::TypeId) -> Option<u8> {
    match m.types().get(t).expect("type") {
        Type::Float { width } => Some(*width),
        Type::Vector { elem, .. } | Type::Array { elem, .. } => float_width(m, *elem),
        _ => None,
    }
}

/// What the module holds and computes at each width.
#[derive(Debug, Default)]
struct Shape {
    /// `f32 -> f16` conversions, and of them those of a widened 16-bit float (exact).
    narrowings: usize,
    exact_narrowings: usize,
    /// `f16 -> f32` conversions.
    widenings: usize,
    /// The 16-bit dot products, `mix`es, and minimums, maximums and clamps.
    dots: usize,
    mixes: usize,
    bounds: usize,
    /// The divisions and square roots at 16 bits and at 32.
    divides: [usize; 2],
    sqrts: [usize; 2],
    /// The undefined `f32` values, and the widenings of undefined 16-bit ones.
    wide_undefs: usize,
    undef_widenings: usize,
    /// The variables (`Function` and `Private`) holding floats at 32 bits and at 16.
    wide_vars: usize,
    narrow_vars: usize,
}

fn shape(m: &TypedModule) -> Shape {
    let width = |v| float_width(m, m.value(v).expect("value").ty);
    let widened: Vec<_> = m
        .blocks()
        .iter()
        .flat_map(|b| &b.insts)
        .filter_map(|n| match n.inst {
            Inst::Convert {
                op: ConvOp::FConvert,
                a,
                ..
            } if width(a) == Some(16) => n.result,
            _ => None,
        })
        .collect();
    let undefs: Vec<_> = m
        .blocks()
        .iter()
        .flat_map(|b| &b.insts)
        .filter_map(|n| matches!(n.inst, Inst::Undef).then_some(n.result).flatten())
        .collect();
    let mut s = Shape::default();
    for n in m.blocks().iter().flat_map(|b| &b.insts) {
        let r16 = n.result.is_some_and(|r| width(r) == Some(16));
        match &n.inst {
            Inst::Undef if n.result.is_some_and(|r| width(r) == Some(32)) => s.wide_undefs += 1,
            Inst::Convert {
                op: ConvOp::FConvert,
                a,
                ..
            } if undefs.contains(a) => s.undef_widenings += 1,
            Inst::Convert {
                op: ConvOp::FConvert,
                a,
                ..
            } if r16 && width(*a) == Some(32) => {
                s.narrowings += 1;
                s.exact_narrowings += usize::from(widened.contains(a));
            }
            Inst::Convert {
                op: ConvOp::FConvert,
                a,
                ..
            } if width(*a) == Some(16) => s.widenings += 1,
            Inst::Binary { op: BinOp::Dot, .. } if r16 => s.dots += 1,
            Inst::Binary {
                op: BinOp::FDiv, ..
            } => s.divides[usize::from(!r16)] += 1,
            Inst::Glsl {
                op: GlslOp::Sqrt, ..
            } => s.sqrts[usize::from(!r16)] += 1,
            Inst::Glsl { op, .. } if r16 => match op {
                GlslOp::FMix => s.mixes += 1,
                GlslOp::FMin
                | GlslOp::FMax
                | GlslOp::FClamp
                | GlslOp::NMin
                | GlslOp::NMax
                | GlslOp::NClamp => s.bounds += 1,
                _ => {}
            },
            _ => {}
        }
    }
    let pointees =
        m.vars()
            .iter()
            .map(|v| v.pointee)
            .chain(m.globals().iter().filter_map(|g| {
                (g.class == crate::spirv::ir::StorageClass::Private).then_some(g.pointee)
            }));
    for t in pointees {
        match float_width(m, t) {
            Some(32) => s.wide_vars += 1,
            Some(16) => s.narrow_vars += 1,
            _ => {}
        }
    }
    s
}

/// On the Iris Xe and RADV (the 16-bit float controls) the kernel holds its `half` values at 16
/// bits: its minimum, maximum, clamp, dot product and `mix` are 16-bit instructions, its four
/// inputs' roundings are its only narrowings (none of an exact half), its result's widening is its
/// only widening, its loop-carried sum and its thread array are 16-bit variables, and it records
/// `NativeDot` and `NativeMix` and passes L3. On lavapipe (the conversion alone) none of that is held.
/// Each answers the exact result on the interpreter.
#[test]
fn half_values_are_held_at_16_bits_under_the_float_controls() {
    let Some(bc) = bitcode() else {
        return;
    };
    for stem in ["intel-irisxe-rplp-anv", "amd-780m-radv"] {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        assert_eq!(quantize_form(&p), QuantizeForm::Controlled, "{stem}");
        let (m, push) = emitted(&bc, &p);
        let s = shape(&m);
        assert_eq!(s.exact_narrowings, 0, "{stem}: {s:?}");
        assert_eq!((s.narrowings, s.widenings), (4, 1), "{stem}: {s:?}");
        assert!(
            s.dots == 1 && s.mixes == 1 && s.bounds >= 3,
            "{stem}: {s:?}"
        );
        assert!(s.wide_vars == 0 && s.narrow_vars >= 1, "{stem}: {s:?}");
        assert_eq!(interpret(&m, push), RESULT, "{stem}");
        let a = compile(&bc, "half_hold", &p).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert!(
            a.precision.contains(&PrecisionNote::NativeDot)
                && a.precision.contains(&PrecisionNote::NativeMix),
            "{stem}: {:?}",
            a.precision
        );
    }
    let p = profile("lavapipe", DevicePolicy::default()).expect("profile");
    assert_eq!(quantize_form(&p), QuantizeForm::Converted);
    let (m, push) = emitted(&bc, &p);
    let s = shape(&m);
    assert!(
        s.dots == 0 && s.mixes == 0 && s.narrow_vars == 0,
        "lavapipe: {s:?}"
    );
    assert_eq!(interpret(&m, push), RESULT, "lavapipe");
}

/// On the Iris Xe and RADV (the 16-bit float controls) a `half` division, on two lanes and on
/// one, and a `half` square root are the device's 16-bit `OpFDiv` and `Sqrt`, none at 32 bits and
/// no exact half narrowed, and an undefined `float` vector stays one (no widening of its 16-bit
/// twin), recording `NativeDivision` and `NativeTranscendentals` and passing L3; a
/// divisor past `2^14` and a denormal one are scaled into Vulkan's range exactly. On lavapipe
/// (the conversion alone) they are the `float` instructions. Each answers the exact results on
/// the interpreter.
#[test]
fn half_division_and_sqrt_are_16_bit_under_the_float_controls() {
    let Some(bc) = bitcode_of(DIVIDE) else {
        return;
    };
    for stem in ["intel-irisxe-rplp-anv", "amd-780m-radv"] {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        assert_eq!(quantize_form(&p), QuantizeForm::Controlled, "{stem}");
        let (m, push) = emitted_entry(&bc, "half_divide", &p);
        let s = shape(&m);
        assert_eq!(s.exact_narrowings, 0, "{stem}: {s:?}");
        assert_eq!((s.divides, s.sqrts), ([2, 0], [1, 0]), "{stem}: {s:?}");
        // The `float` vector's undefined base stays an undefined `float` vector.
        assert!(
            s.wide_undefs >= 1 && s.undef_widenings == 0,
            "{stem}: {s:?}"
        );
        assert_eq!(
            interpret_on(&m, push, &DIVIDE_INPUTS, 5),
            QUOTIENTS,
            "{stem}"
        );
        let a = compile(&bc, "half_divide", &p).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert!(
            a.precision.contains(&PrecisionNote::NativeDivision)
                && a.precision.contains(&PrecisionNote::NativeTranscendentals),
            "{stem}: {:?}",
            a.precision
        );
    }
    let p = profile("lavapipe", DevicePolicy::default()).expect("profile");
    assert_eq!(quantize_form(&p), QuantizeForm::Converted);
    let (m, push) = emitted_entry(&bc, "half_divide", &p);
    let s = shape(&m);
    assert_eq!((s.divides, s.sqrts), ([0, 2], [0, 1]), "lavapipe: {s:?}");
    assert_eq!(
        interpret_on(&m, push, &DIVIDE_INPUTS, 5),
        QUOTIENTS,
        "lavapipe"
    );
}
