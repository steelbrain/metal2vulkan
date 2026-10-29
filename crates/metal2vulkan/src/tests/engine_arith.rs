//! The arith family end to end: owned kernels go through the whole new pipeline (decode →
//! specialize → analyze → select → emit → verify → serialize → link) on committed profiles and pass
//! L3, and their typed modules compute the Metal/LLVM result on the CPU SPIR-V interpreter.
//! Profiles without `shaderInt64` hold 64-bit integers as two words, a switch on one included, and
//! profiles without `shaderFloat64` refuse `double` with the typed capability; `half` and `bfloat`
//! are held widened on every profile.
//!
//! Each kernel stores one word to buffer 0 (the skeleton's static-root store). Canonicalization
//! folds integer arithmetic on constants but never float arithmetic, so each kernel makes its
//! inputs opaque as `fadd <constant>, -0.0` (exact for every non-NaN input) and a bitcast to
//! an integer where it needs one. `llvm-as` assembles the generated module; without it the test
//! skips visibly.
//!
//! Every profile here takes Metal's `half` rounding and NaN (`exact_maths`), whose answers these
//! tests check; the maths functions are the device's on every device but MoltenVK
//! (`engine_native_maths`).

use crate::air::{self, SpecRequest};
use crate::device::profile::{DeviceProfile, DriverWorkaround, FloatFunctions};
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Capability, Error, Feature, LowerFamily, Needs};
use crate::library::FunctionConstantValues;
use crate::lower::emit::emit_module;
use crate::lower::select::{select, LowerInputs};
use crate::memory::static_root::analyze_memory;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::compile::{plan_bitcode, EntryRef};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, exact_maths, link_and_validate, llvm_assemble, meter, profile, AssembleError, Scratch,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};

/// The profiles: the M1 set plus one with 64-bit floats.
const PROFILES: [&str; 4] = [
    "moltenvk-m3",
    "lunarg-desktop-baseline-2023",
    "vk12-floor",
    "amd-rdna3-radv",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Need {
    Float64,
}

struct Kernel {
    name: &'static str,
    /// The body; it defines `%r` (`i32` or `float`), which the kernel stores.
    body: String,
    /// The result type of `%r`.
    float: bool,
    /// The stored word, or `None` when the interpreter does not model the kernel
    /// (transcendentals).
    want: Option<u32>,
    needs: &'static [Need],
    note: Option<PrecisionNote>,
}

/// `%name` = an opaque `float` with the value `v`.
fn of(name: &str, v: f32) -> String {
    format!(
        "  %{name} = fadd float 0x{:016X}, -0.0\n",
        f64::from(v).to_bits()
    )
}

/// `%name` = an opaque `i32` with bits `b`. A NaN or denormal pattern is not carried through
/// `float` arithmetic, which answers Metal's canonical NaN and reads a denormal as the zero of its
/// sign, but through `llvm.umin` with `-1`.
fn oi(name: &str, b: u32) -> String {
    let f = f32::from_bits(b);
    if f.is_nan() || f.is_subnormal() {
        return format!(
            "  %{name} = call i32 @llvm.umin.i32(i32 {}, i32 -1)\n",
            b as i32
        );
    }
    format!(
        "{}  %{name} = bitcast float %{name}.f to i32\n",
        of(&format!("{name}.f"), f32::from_bits(b))
    )
}

/// `%name` = an opaque `i<bits>` with the low bits of `b`.
fn on(name: &str, bits: u8, b: u32) -> String {
    format!(
        "{}  %{name} = trunc i32 %{name}.w to i{bits}\n",
        oi(&format!("{name}.w"), b)
    )
}

fn k(name: &'static str, body: String, want: u32) -> Kernel {
    Kernel {
        name,
        body,
        float: false,
        want: Some(want),
        needs: &[],
        note: None,
    }
}

fn kernels() -> Vec<Kernel> {
    let mut v = vec![
        k(
            "add_i8_wraps",
            format!(
                "{}{}  %s = add i8 %a, %b\n  %r = zext i8 %s to i32\n",
                on("a", 8, 250),
                on("b", 8, 10)
            ),
            4,
        ),
        k(
            "sdiv_i16_sign_extends",
            format!(
                "{}  %q = sdiv i16 %a, 3\n  %r = sext i16 %q to i32\n",
                on("a", 16, 0xFFF6)
            ),
            0xFFFF_FFFD,
        ),
        k(
            "ashr_i5_keeps_the_sign",
            format!(
                "{}  %s = ashr i5 %a, 2\n  %r = zext i5 %s to i32\n",
                on("a", 5, 0x10)
            ),
            0x1C,
        ),
        k(
            "icmp_signed_and_unsigned",
            format!(
                "{}{}  %lt = icmp slt i32 %a, %b\n  %ult = icmp ult i32 %a, %b\n  %x = select i1 %lt, i32 7, i32 9\n  %y = select i1 %ult, i32 16, i32 32\n  %r = or i32 %x, %y\n",
                oi("a", 5),
                oi("b", 0xFFFF_FFFD)
            ),
            9 | 16,
        ),
        // An integer to `bool` is `x != 0`, whatever the low bit or byte (lanes 2, 0x80, 0, 1, and
        // the scalar 0x100).
        k(
            "convert_integers_to_bool_is_nonzero",
            format!(
                "{}{}{}{}{}  %v0 = insertelement <4 x i8> zeroinitializer, i8 %a, i32 0\n  %v1 = insertelement <4 x i8> %v0, i8 %b, i32 1\n  %v2 = insertelement <4 x i8> %v1, i8 %c, i32 2\n  %v3 = insertelement <4 x i8> %v2, i8 %d, i32 3\n  %bv = call <4 x i1> @air.convert.u.v4i1.u.v4i8(<4 x i8> %v3)\n  %e0 = extractelement <4 x i1> %bv, i32 0\n  %e1 = extractelement <4 x i1> %bv, i32 1\n  %e2 = extractelement <4 x i1> %bv, i32 2\n  %e3 = extractelement <4 x i1> %bv, i32 3\n  %s = call i1 @air.convert.u.i1.s.i32(i32 %w)\n  %x0 = select i1 %e0, i32 1, i32 0\n  %x1 = select i1 %e1, i32 2, i32 0\n  %x2 = select i1 %e2, i32 4, i32 0\n  %x3 = select i1 %e3, i32 8, i32 0\n  %x4 = select i1 %s, i32 16, i32 0\n  %y0 = or i32 %x0, %x1\n  %y1 = or i32 %y0, %x2\n  %y2 = or i32 %y1, %x3\n  %r = or i32 %y2, %x4\n",
                on("a", 8, 2),
                on("b", 8, 0x80),
                on("c", 8, 0),
                on("d", 8, 1),
                oi("w", 0x100)
            ),
            1 | 2 | 8 | 16,
        ),
        k(
            "fcmp_uno_sees_nan",
            format!(
                "{}{}  %u = fcmp uno float %a, %b\n  %o = fcmp olt float %b, %b\n  %x = zext i1 %u to i32\n  %y = zext i1 %o to i32\n  %y2 = shl i32 %y, 1\n  %r = or i32 %x, %y2\n",
                of("a", f32::NAN),
                of("b", 1.0)
            ),
            1,
        ),
        k(
            "fptosi_and_sitofp",
            format!(
                "{}  %i = fptosi float %a to i8\n  %n = sub i8 0, %i\n  %f = sitofp i8 %n to float\n  %r = bitcast float %f to i32\n",
                of("a", 3.7)
            ),
            (-3.0f32).to_bits(),
        ),
        k(
            "vector_shuffle_and_bitcast",
            format!(
                "{}  %v0 = insertelement <4 x i8> <i8 1, i8 2, i8 3, i8 0>, i8 %a, i64 3\n  %v1 = shufflevector <4 x i8> %v0, <4 x i8> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>\n  %r = bitcast <4 x i8> %v1 to i32\n",
                on("a", 8, 4)
            ),
            0x0102_0304,
        ),
        k(
            "dynamic_extract",
            format!(
                "{}  %e = extractelement <4 x i32> <i32 10, i32 20, i32 30, i32 40>, i32 %i\n  %r = add i32 %e, 1\n",
                oi("i", 2)
            ),
            31,
        ),
        // An element index may be any integer type, `i1` too, read as unsigned: a compare's `bool`
        // picks lane 1 or 0. The SPIR-V builder rejected the `bool` index.
        k(
            "i1_index_extract_set",
            format!(
                "{}  %c = fcmp ogt float %x, 0x3FE0000000000000\n  %r = extractelement <2 x i32> <i32 10, i32 20>, i1 %c\n",
                of("x", 0.75)
            ),
            20,
        ),
        k(
            "i1_index_extract_clear",
            format!(
                "{}  %c = fcmp ogt float %x, 0x3FE0000000000000\n  %r = extractelement <2 x i32> <i32 10, i32 20>, i1 %c\n",
                of("x", 0.25)
            ),
            10,
        ),
        k(
            "i1_index_insert_set",
            format!(
                "{}{}  %c = fcmp ogt float %x, 0x3FE0000000000000\n  %w = insertelement <2 x i32> <i32 10, i32 20>, i32 %a, i1 %c\n  %l0 = extractelement <2 x i32> %w, i32 0\n  %l1 = extractelement <2 x i32> %w, i32 1\n  %s = mul i32 %l0, 1000\n  %r = add i32 %s, %l1\n",
                of("x", 0.75),
                oi("a", 7)
            ),
            10_007,
        ),
        k(
            "i1_index_insert_clear",
            format!(
                "{}{}  %c = fcmp ogt float %x, 0x3FE0000000000000\n  %w = insertelement <2 x i32> <i32 10, i32 20>, i32 %a, i1 %c\n  %l0 = extractelement <2 x i32> %w, i32 0\n  %l1 = extractelement <2 x i32> %w, i32 1\n  %s = mul i32 %l0, 1000\n  %r = add i32 %s, %l1\n",
                of("x", 0.25),
                oi("a", 7)
            ),
            7_020,
        ),
        k(
            "aggregate_round_trip",
            format!(
                "{}{}  %s0 = insertvalue {{ i32, float }} undef, i32 %a, 0\n  %s1 = insertvalue {{ i32, float }} %s0, float %f, 1\n  %x = extractvalue {{ i32, float }} %s1, 0\n  %g = extractvalue {{ i32, float }} %s1, 1\n  %gi = fptosi float %g to i32\n  %r = add i32 %x, %gi\n",
                oi("a", 40),
                of("f", 2.0)
            ),
            42,
        ),
        k(
            "llvm_integer_intrinsics",
            format!(
                "{}{}{}  %m = call i32 @llvm.umin.i32(i32 %a, i32 %b)\n  %p = call i16 @llvm.ctpop.i16(i16 %h)\n  %pw = zext i16 %p to i32\n  %l = call i32 @llvm.ctlz.i32(i32 %m, i1 false)\n  %f = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 8)\n  %s0 = add i32 %m, %pw\n  %s1 = add i32 %s0, %l\n  %r = xor i32 %s1, %f\n",
                oi("a", 5),
                oi("b", 0x0900_0000),
                on("h", 16, 0xF0F0),
            ),
            // umin 5; ctpop 8; ctlz(5) 29; fshl(5, 0x09000000, 8) = 0x509.
            (5 + 8 + 29) ^ 0x509,
        ),
        k(
            "abs_and_saturation",
            format!(
                "{}  %x = call i8 @llvm.abs.i8(i8 %a, i1 false)\n  %s = call i8 @air.add_sat.s.i8(i8 %x, i8 125)\n  %r = zext i8 %s to i32\n",
                on("a", 8, 0xFB)
            ),
            // abs(-5) = 5; 5 + 125 saturates at 127.
            127,
        ),
        k(
            "air_convert_saturates",
            format!(
                "{}{}  %i = call i32 @air.convert.s.i32.f.f32(float %a)\n  %u = call i8 @air.convert.u.i8.f.f32(float %b)\n  %uw = zext i8 %u to i32\n  %r = xor i32 %i, %uw\n",
                of("a", 3.0e9),
                of("b", -7.0)
            ),
            0x7FFF_FFFF,
        ),
        k(
            "air_float_functions",
            format!(
                "{}  %c = call float @air.clamp.f32(float %a, float 0.0, float 1.0)\n  %n = fneg float %a\n  %ro = call float @air.round.f32(float %n)\n  %fr = call float @air.fract.f32(float %a)\n  %s = fadd float %ro, %fr\n  %t = fadd float %s, %c\n  %r = bitcast float %t to i32\n",
                of("a", 2.5)
            ),
            // clamp 1, round(-2.5) = -3, fract 0.5: -1.5.
            (-1.5f32).to_bits(),
        ),
        k(
            "phi_of_a_widened_byte",
            format!(
                "{}  %c = icmp ugt i8 %a, 3\n  br i1 %c, label %t, label %e\nt:\n  %x = add i8 %a, 100\n  br label %j\ne:\n  br label %j\nj:\n  %p = phi i8 [ %x, %t ], [ 7, %e ]\n  %r = zext i8 %p to i32\n",
                on("a", 8, 200)
            ),
            44,
        ),
    ];
    // A switch on a narrow integer selects on its zero-extended container.
    v.push(k(
        "switch_on_a_widened_byte",
        format!(
            "{}  switch i8 %a, label %d [ i8 1, label %o\n i8 -56, label %n ]\no:\n  br label %d\nn:\n  br label %d\nd:\n  %r = phi i32 [ 7, %entry ], [ 11, %o ], [ 13, %n ]\n",
            on("a", 8, 200)
        ),
        13,
    ));
    // Without `shaderInt64` the selector is two words; a nonzero high word matches no (32-bit) case
    // literal.
    v.push(k(
        "switch_on_an_i40",
        format!(
            "{}  %w = zext i32 %a to i40\n  %s = shl i40 %w, 4\n  switch i40 %s, label %d [ i40 16, label %o ]\no:\n  br label %d\nd:\n  %r = phi i32 [ 7, %entry ], [ 11, %o ]\n",
            oi("a", 1)
        ),
        11,
    ));
    v.push(k(
        "switch_on_an_i64_high_word",
        format!(
            "{}  %w = zext i32 %a to i64\n  %s = shl i64 %w, 32\n  %t = or i64 %s, 16\n  switch i64 %t, label %d [ i64 16, label %o ]\no:\n  br label %d\nd:\n  %p = phi i32 [ 7, %entry ], [ 11, %o ]\n  %u = lshr i64 %t, 32\n  switch i64 %u, label %e [ i64 5, label %f ]\nf:\n  br label %e\ne:\n  %q = phi i32 [ 100, %d ], [ 200, %f ]\n  %r = add i32 %p, %q\n",
            oi("a", 5)
        ),
        207,
    ));
    // `mad_sat` on 64-bit integers saturates the exact 128-bit result.
    v.push(k(
        "mad_sat_i64_saturates_the_exact_result",
        format!(
            "{}  %x = zext i32 %a to i64\n  %y = shl i64 %x, 31\n  %s = call i64 @air.mad_sat.s.i64(i64 %y, i64 %y, i64 -5)\n  %u = call i64 @air.mad_sat.u.i64(i64 %x, i64 %x, i64 7)\n  %n = sub i64 0, %y\n  %m = call i64 @air.mad_sat.s.i64(i64 %n, i64 %y, i64 3)\n  %sh = lshr i64 %s, 32\n  %uh = lshr i64 %u, 32\n  %mh = lshr i64 %m, 32\n  %a0 = trunc i64 %s to i32\n  %a1 = trunc i64 %sh to i32\n  %a2 = trunc i64 %u to i32\n  %a3 = trunc i64 %uh to i32\n  %a4 = trunc i64 %mh to i32\n  %x1 = xor i32 %a0, %a1\n  %x2 = xor i32 %x1, %a2\n  %x3 = xor i32 %x2, %a3\n  %r = xor i32 %x3, %a4\n",
            oi("a", 0xFFFF_FFFF)
        ),
        0xFFFF_FFF6,
    ));
    v.push(k(
        "convert_round_and_repack",
        format!(
            "{}  %c1 = call i32 @air.convert.s.i32.f.f32(float %a)\n  %f1 = call float @air.round.f32(float %a)\n  %c2 = call i32 @air.convert.s.i32.f.f32(float %f1)\n  %v = bitcast i32 %c1 to <4 x i8>\n  %v2 = add <4 x i8> %v, <i8 1, i8 1, i8 1, i8 1>\n  %w = bitcast <4 x i8> %v2 to i32\n  %s = add i32 %c1, %c2\n  %r = add i32 %s, %w\n",
            of("a", 1.5)
        ),
        // trunc 1, round 2, <2, 1, 1, 1> = 0x01010102.
        3 + 0x0101_0102,
    ));
    v.push(Kernel {
        name: "mul_i40_masks",
        body: format!(
            "{}  %w = zext i32 %a to i40\n  %m = mul i40 %w, 4096\n  %s = lshr i40 %m, 8\n  %r = trunc i40 %s to i32\n",
            oi("a", 0x1234_5678)
        ),
        float: false,
        want: Some(0x2345_6780),
        needs: &[],
        note: None,
    });
    v.push(Kernel {
        name: "i64_carry",
        body: format!(
            "{}  %w = zext i32 %a to i64\n  %s = add i64 %w, 1\n  %h = lshr i64 %s, 32\n  %r = trunc i64 %h to i32\n",
            oi("a", 0xFFFF_FFFF)
        ),
        float: false,
        want: Some(1),
        needs: &[],
        note: None,
    });
    // Without `shaderInt64` 64-bit integers are two words; division is the exact word division
    // (`udivrem`), with no loop.
    v.push(k(
        "i64_division_as_words",
        format!(
            "{}  %x = zext i32 %a to i64\n  %y = shl i64 %x, 20\n  %q = udiv i64 %y, 1000003\n  %n = sub i64 0, %y\n  %s = srem i64 %n, 77\n  %ql = trunc i64 %q to i32\n  %qs = lshr i64 %q, 32\n  %qh = trunc i64 %qs to i32\n  %x1 = xor i32 %ql, %qh\n  %sl = trunc i64 %s to i32\n  %r = add i32 %x1, %sl\n",
            oi("a", 0x1234_5678)
        ),
        0x1316_B402,
    ));
    v.push(k(
        WIDE_DIVISIONS,
        format!(
            "{}{}  %x = zext i32 %a to i64\n  %y0 = zext i32 %b to i64\n  %y = shl i64 %y0, 24\n  %d = or i64 %x, 1\n  %q = udiv i64 %y, %d\n  %m = urem i64 %y, %d\n  %n = sub i64 0, %y\n  %s = sdiv i64 %n, %d\n  %t = srem i64 %n, 77\n  %v0 = insertelement <2 x i64> zeroinitializer, i64 %y, i32 0\n  %v = insertelement <2 x i64> %v0, i64 %n, i32 1\n  %w = udiv <2 x i64> %v, <i64 3, i64 1000003>\n  %e0 = extractelement <2 x i64> %w, i32 0\n  %e1 = extractelement <2 x i64> %w, i32 1\n  %s1 = add i64 %q, %m\n  %s2 = add i64 %s1, %s\n  %s3 = add i64 %s2, %t\n  %s4 = add i64 %s3, %e0\n  %s5 = add i64 %s4, %e1\n  %h = lshr i64 %s5, 32\n  %hl = trunc i64 %h to i32\n  %ll = trunc i64 %s5 to i32\n  %r = xor i32 %ll, %hl\n",
            oi("a", 0x1234_5678),
            oi("b", 0x0ABC_DEF1)
        ),
        0x925B_1770,
    ));
    v.push(k(
        "i64_phi_across_blocks",
        format!(
            "{}  %x = zext i32 %a to i64\n  %c = icmp ugt i32 %a, 5\n  br i1 %c, label %t, label %f\nt:\n  %u = shl i64 %x, 33\n  br label %d\nf:\n  %v = add i64 %x, 1\n  br label %d\nd:\n  %p = phi i64 [ %u, %t ], [ %v, %f ]\n  %h = lshr i64 %p, 32\n  %r = trunc i64 %h to i32\n",
            oi("a", 0x1234_5678)
        ),
        0x2468_ACF0,
    ));
    v.push(k(
        "fptosi_to_i64_as_words",
        format!(
            "{}  %i = fptosi float %f to i64\n  %h = lshr i64 %i, 32\n  %l = trunc i64 %i to i32\n  %hh = trunc i64 %h to i32\n  %r = xor i32 %l, %hh\n",
            of("f", -12_345_678.0)
        ),
        0x00BC_614D,
    ));
    // Chunked vectors of 64-bit integers are chunks of two-word lanes without `shaderInt64`:
    // lane-wise operations per chunk, constant and `i64` dynamic lane moves, a shuffle into a
    // five-lane vector (its last chunk one lane), a lane-wise intrinsic, a word bitcast, and a
    // truncation.
    v.push(k(
        "chunked_i64_lanes",
        format!(
            "{}{}  %x = zext i32 %a to i64\n  %h = shl i64 %x, 28\n  %v = insertelement <8 x i64> <i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8>, i64 %h, i32 5\n  %w = add <8 x i64> %v, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>\n  %m = mul <8 x i64> %w, <i64 3, i64 3, i64 3, i64 3, i64 3, i64 3, i64 3, i64 3>\n  %c = icmp ugt <8 x i64> %m, <i64 12884901900, i64 12884901900, i64 12884901900, i64 12884901900, i64 12884901900, i64 12884901900, i64 12884901900, i64 12884901900>\n  %s = select <8 x i1> %c, <8 x i64> %m, <8 x i64> %v\n  %i = zext i32 %b to i64\n  %d = extractelement <8 x i64> %s, i64 %i\n  %t = insertelement <8 x i64> %s, i64 %x, i64 %i\n  %q = shufflevector <8 x i64> %t, <8 x i64> %s, <5 x i32> <i32 5, i32 13, i32 0, i32 7, i32 2>\n  %u = call <5 x i64> @llvm.umax.v5i64(<5 x i64> %q, <5 x i64> <i64 100, i64 100, i64 100, i64 100, i64 100>)\n  %bw = bitcast <5 x i64> %u to <10 x i32>\n  %e0 = extractelement <10 x i32> %bw, i32 1\n  %e1 = extractelement <10 x i32> %bw, i32 2\n  %tr = trunc <5 x i64> %u to <5 x i32>\n  %e2 = extractelement <5 x i32> %tr, i32 4\n  %dl = trunc i64 %d to i32\n  %dh64 = lshr i64 %d, 32\n  %dh = trunc i64 %dh64 to i32\n  %x0 = xor i32 %e0, %e1\n  %x1 = xor i32 %x0, %e2\n  %x2 = xor i32 %x1, %dl\n  %r = xor i32 %x2, %dh\n",
            oi("a", 0x89AB_CDEF),
            oi("b", 5)
        ),
        0x19D0_36FB,
    ));
    v.push(k(
        "chunked_i64_converts_and_one_lane",
        format!(
            "{}  %x = zext i32 %a to i64\n  %n = sub i64 0, %x\n  %v = insertelement <6 x i64> <i64 -3, i64 7, i64 1099511627776, i64 -1099511627777, i64 5, i64 6>, i64 %n, i32 4\n  %f = sitofp <6 x i64> %v to <6 x float>\n  %g = extractelement <6 x float> %f, i32 4\n  %k = extractelement <6 x float> %f, i32 3\n  %gi = bitcast float %g to i32\n  %ki = bitcast float %k to i32\n  %one = insertelement <1 x i64> poison, i64 %x, i32 0\n  %two = add <1 x i64> %one, <i64 -1>\n  %e = extractelement <1 x i64> %two, i32 0\n  %el = trunc i64 %e to i32\n  %y = xor i32 %gi, %ki\n  %r = xor i32 %y, %el\n",
            oi("a", 0x89AB_CDEF)
        ),
        0x9522_6620,
    ));
    // A fast division is the device's quotient, noted.
    v.push(Kernel {
        name: "arcp_fdiv_notes_device_precision",
        body: format!("{}  %r = fdiv arcp float %a, 2.0\n", of("a", 7.0)),
        float: true,
        want: Some(3.5f32.to_bits()),
        needs: &[],
        note: Some(PrecisionNote::DeviceArithmetic),
    });
    v.push(Kernel {
        name: "fdiv_is_metals_precise_division",
        body: format!("{}  %r = fdiv float %a, 2.0\n", of("a", 7.0)),
        float: true,
        want: Some(3.5f32.to_bits()),
        needs: &[],
        note: None,
    });
    // Apple's math library defines precise `air.exp2`, `air.log2`, and `air.sqrt` as these `llvm.*`
    // calls, so they lower as those functions with their notes (precise `sqrt`, whose answer is the
    // one correctly rounded value, with none); `llvm.copysign` moves the sign bit exactly.
    v.push(Kernel {
        name: "llvm_sqrt_is_the_correctly_rounded_root",
        body: format!(
            "{}  %r = call float @llvm.sqrt.f32(float %a)\n",
            of("a", 2.25)
        ),
        float: true,
        want: Some(1.5f32.to_bits()),
        needs: &[],
        note: None,
    });
    for (name, f) in [
        ("llvm_exp2_notes_device_transcendentals", "exp2"),
        ("llvm_log2_notes_device_transcendentals", "log2"),
    ] {
        v.push(Kernel {
            name,
            body: format!(
                "{}  %r = call float @llvm.{f}.f32(float %a)\n",
                of("a", 3.0)
            ),
            float: true,
            want: None,
            needs: &[],
            note: Some(PrecisionNote::DeviceTranscendentals),
        });
    }
    v.push(Kernel {
        name: "llvm_copysign_takes_the_sign_bit",
        body: format!(
            "{}{}  %r = call float @llvm.copysign.f32(float %a, float %b)\n",
            of("a", 2.5),
            of("b", -0.0)
        ),
        float: true,
        want: Some((-2.5f32).to_bits()),
        needs: &[],
        note: None,
    });
    v.push(Kernel {
        name: "sin_notes_device_transcendentals",
        body: format!("{}  %r = call float @air.sin.f32(float %a)\n", of("a", 0.5)),
        float: true,
        want: None,
        needs: &[],
        note: Some(PrecisionNote::DeviceTranscendentals),
    });
    // `rgb565` is the device's product rounded by `RoundEven` (0.5 * 63 = 31.5 -> 32, exact in
    // f32), noted, and `rg11b10f` unpacks the half with the field's bits (0x3C0 -> red 1.0).
    v.push(Kernel {
        note: Some(PrecisionNote::DevicePackConversion),
        ..k(
            "rgb565_pack_rounds_the_float_product_to_even",
            format!(
                "{}  %v = insertelement <3 x float> <float 0.0, float 0.5, float 0.0>, float %a, i32 0\n  %p = call i16 @air.pack.unorm.rgb565.v3f32(<3 x float> %v)\n  %r = zext i16 %p to i32\n",
                of("a", 1.0)
            ),
            31 | (32 << 5),
        )
    });
    v.push(Kernel {
        name: "rg11b10f_unpacks_half_fields",
        body: format!(
            "{}  %v = call <3 x float> @air.unpack.unorm.rg11b10f.v3f32(i32 %w)\n  %x = extractelement <3 x float> %v, i32 0\n  %y = extractelement <3 x float> %v, i32 1\n  %r = fadd float %x, %y\n",
            oi("w", 0x3C0 | (0x180 << 11))
        ),
        float: true,
        want: Some((1.0f32 + 2f32.powi(-9)).to_bits()),
        needs: &[],
        note: None,
    });
    // A 64-bit integer just above a bfloat tie rounds up (through a round-to-odd f32).
    v.push(Kernel {
        name: "int64_to_bfloat_rounds_once",
        body: format!(
            "{}  %x = zext i32 %a to i64\n  %y = shl i64 %x, 32\n  %z = or i64 %y, 1\n  %b = uitofp i64 %z to bfloat\n  %r = fpext bfloat %b to float\n",
            oi("a", 0x0101_0000)
        ),
        float: true,
        want: Some(0x5B81_0000),
        needs: &[],
        note: None,
    });
    // Vectors of other than 2 to 4 lanes are chunked: lane-wise operations per chunk, lane moves
    // through chunk members, bitcasts through words.
    v.push(k(
        "chunked_add_and_constant_lane",
        format!(
            "{}  %v = insertelement <8 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>, i32 %a, i32 5\n  %w = add <8 x i32> %v, <i32 10, i32 20, i32 30, i32 40, i32 50, i32 60, i32 70, i32 80>\n  %x = extractelement <8 x i32> %w, i32 5\n  %y = extractelement <8 x i32> %w, i32 7\n  %r = add i32 %x, %y\n",
            oi("a", 3)
        ),
        3 + 60 + 88,
    ));
    v.push(Kernel {
        name: "chunked_dynamic_lanes_compare_and_select",
        body: format!(
            "{}{}  %v = insertelement <8 x float> <float 0x3FF0000000000000, float 0x4000000000000000, float 0x4008000000000000, float 0x4010000000000000, float 0x4014000000000000, float 0x4018000000000000, float 0x401C000000000000, float 0x4020000000000000>, float %a, i32 %i\n  %x = fmul <8 x float> %v, <float 0x4000000000000000, float 0x4000000000000000, float 0x4000000000000000, float 0x4000000000000000, float 0x4000000000000000, float 0x4000000000000000, float 0x4000000000000000, float 0x4000000000000000>\n  %c = fcmp ogt <8 x float> %x, <float 0x4014000000000000, float 0x4014000000000000, float 0x4014000000000000, float 0x4014000000000000, float 0x4014000000000000, float 0x4014000000000000, float 0x4014000000000000, float 0x4014000000000000>\n  %s = select <8 x i1> %c, <8 x float> %x, <8 x float> zeroinitializer\n  %j = add i32 %i, 1\n  %e = extractelement <8 x float> %s, i32 %i\n  %g = extractelement <8 x float> %s, i32 %j\n  %h = extractelement <8 x float> %s, i32 1\n  %t = fadd float %e, %g\n  %r = fadd float %t, %h\n",
            of("a", 3.0),
            oi("i", 6)
        ),
        float: true,
        want: Some(22.0f32.to_bits()),
        needs: &[],
        note: None,
    });
    v.push(k(
        "chunked_bytes_bitcast_both_ways",
        format!(
            "{}  %v = insertelement <4 x i32> zeroinitializer, i32 %a, i32 2\n  %b = bitcast <4 x i32> %v to <16 x i8>\n  %x = add <16 x i8> %b, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>\n  %y = extractelement <16 x i8> %x, i32 9\n  %z = bitcast <16 x i8> %x to <4 x i32>\n  %w = extractelement <4 x i32> %z, i32 2\n  %yw = zext i8 %y to i32\n  %r = xor i32 %yw, %w\n",
            oi("a", 0x0403_0201)
        ),
        0x0504_0302 ^ 3,
    ));
    v.push(k(
        "chunked_half_lanes_bitcast_through_words",
        format!(
            "{}  %v = insertelement <4 x float> zeroinitializer, float %f, i32 1\n  %h = bitcast <4 x float> %v to <8 x half>\n  %g = fadd <8 x half> %h, %h\n  %b = bitcast <8 x half> %g to <4 x i32>\n  %r = extractelement <4 x i32> %b, i32 1\n",
            of("f", f32::from_bits(0x3C00_4000))
        ),
        0x4000_4400,
    ));
    v.push(Kernel {
        name: "chunked_shuffles_widen_and_narrow",
        body: format!(
            "{}  %p = insertelement <4 x float> <float 0x3FF0000000000000, float 0x4000000000000000, float 0x4008000000000000, float 0x4010000000000000>, float %a, i32 0\n  %q = shufflevector <4 x float> %p, <4 x float> <float 0x4014000000000000, float 0x4018000000000000, float 0x401C000000000000, float 0x4020000000000000>, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>\n  %s = shufflevector <8 x float> %q, <8 x float> poison, <3 x i32> <i32 7, i32 0, i32 4>\n  %e = extractelement <3 x float> %s, i32 0\n  %g = extractelement <3 x float> %s, i32 1\n  %r = fadd float %e, %g\n",
            of("a", 1.5)
        ),
        float: true,
        want: Some(9.5f32.to_bits()),
        needs: &[],
        note: None,
    });
    v.push(Kernel {
        name: "one_lane_vector",
        body: format!(
            "{}  %v = insertelement <1 x float> poison, float %a, i32 0\n  %w = fmul <1 x float> %v, <float 3.0>\n  %r = extractelement <1 x float> %w, i32 0\n",
            of("a", 1.25)
        ),
        float: true,
        want: Some(3.75f32.to_bits()),
        needs: &[],
        note: None,
    });
    v.push(k(
        "five_lane_signed_division",
        format!(
            "{}  %v = insertelement <5 x i16> <i16 -10, i16 20, i16 30, i16 40, i16 -50>, i16 %a, i32 4\n  %w = sdiv <5 x i16> %v, <i16 3, i16 3, i16 3, i16 3, i16 3>\n  %x = extractelement <5 x i16> %w, i32 0\n  %y = extractelement <5 x i16> %w, i32 4\n  %xw = zext i16 %x to i32\n  %yw = zext i16 %y to i32\n  %ys = shl i32 %yw, 16\n  %r = or i32 %xw, %ys\n",
            on("a", 16, 7)
        ),
        0x0002_FFFD,
    ));
    v.push(Kernel {
        name: "chunked_math_convert_and_whole_select",
        body: format!(
            "{}{}  %v = insertelement <8 x float> <float 0x3FF0000000000000, float 0x4000000000000000, float 0x4008000000000000, float 0x4010000000000000, float 0x4014000000000000, float 0x4018000000000000, float 0x401C000000000000, float 0x4020000000000000>, float %a, i32 5\n  %m = call <8 x float> @air.fabs.v8f32(<8 x float> %v)\n  %n = fptosi <8 x float> %m to <8 x i32>\n  %lt = icmp slt i32 %i, 0\n  %s = select i1 %lt, <8 x i32> zeroinitializer, <8 x i32> %n\n  %z = freeze <8 x i32> %s\n  %f = sitofp <8 x i32> %z to <8 x float>\n  %r = extractelement <8 x float> %f, i32 5\n",
            of("a", -4.5),
            oi("i", 1)
        ),
        float: true,
        want: Some(4.0f32.to_bits()),
        needs: &[],
        note: None,
    });
    // `sincos` stores its cosine through the access family's store into the private cell the
    // kernel reads back; the interpreter does not model `Sin`/`Cos`.
    v.push(Kernel {
        name: "sincos_stores_the_cosine",
        body: format!(
            "  %c = alloca float, align 4\n{}  %s = call float @air.sincos.f32(float %a, ptr %c)\n  %cv = load float, ptr %c, align 4\n  %r = fadd float %s, %cv\n",
            of("a", 0.5)
        ),
        float: true,
        want: None,
        needs: &[],
        note: Some(PrecisionNote::DeviceTranscendentals),
    });
    v.push(Kernel {
        name: "sincos_half_vector_rounds_both",
        body: format!(
            "  %c = alloca <2 x half>, align 4\n{}  %h = fptrunc float %a to half\n  %v0 = insertelement <2 x half> zeroinitializer, half %h, i32 0\n  %v = insertelement <2 x half> %v0, half %h, i32 1\n  %s = call <2 x half> @air.sincos.v2f16(<2 x half> %v, ptr %c)\n  %cv = load <2 x half>, ptr %c, align 4\n  %t = fadd <2 x half> %s, %cv\n  %e = extractelement <2 x half> %t, i32 1\n  %r = fpext half %e to float\n",
            of("a", 0.25)
        ),
        float: true,
        want: None,
        needs: &[],
        note: None,
    });
    // `dot` of chunked vectors is each chunk's `OpDot` on every profile
    // (`dot_is_the_devices_op_dot_on_every_profile`), exact on these lanes; `sincos` of one
    // computes per chunk and stores its cosine as one chunked leaf.
    v.push(Kernel {
        name: "chunked_dot_sums_every_chunk",
        body: format!(
            "{}  %v = insertelement <8 x float> <float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0>, float %a, i32 7\n  %w = shufflevector <8 x float> %v, <8 x float> poison, <5 x i32> <i32 0, i32 1, i32 2, i32 3, i32 7>\n  %d8 = call float @air.dot.v8f32(<8 x float> %v, <8 x float> %v)\n  %d5 = call float @air.dot.v5f32(<5 x float> %w, <5 x float> %w)\n  %r = fadd float %d8, %d5\n",
            of("a", 0.5)
        ),
        float: true,
        // 1 + 4 + ... + 49 + 0.25 = 140.25, and 1 + 4 + 9 + 16 + 0.25 = 30.25.
        want: Some(170.5f32.to_bits()),
        needs: &[],
        note: None,
    });
    v.push(Kernel {
        name: "chunked_half_dot",
        body: format!(
            "{}  %h = fptrunc float %a to half\n  %v = insertelement <5 x half> <half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00>, half %h, i32 4\n  %d = call half @air.dot.v5f16(<5 x half> %v, <5 x half> %v)\n  %r = fpext half %d to float\n",
            of("a", 1.5)
        ),
        float: true,
        // 4 * 1 + 1.5^2 = 6.25, a half.
        want: Some(6.25f32.to_bits()),
        needs: &[],
        note: None,
    });
    v.push(Kernel {
        name: "chunked_sincos_stores_the_cosine",
        body: format!(
            "  %c = alloca <5 x float>, align 32\n{}  %v = insertelement <5 x float> zeroinitializer, float %a, i32 4\n  %s = call <5 x float> @air.sincos.v5f32(<5 x float> %v, ptr %c)\n  %cv = load <5 x float>, ptr %c, align 32\n  %t = fadd <5 x float> %s, %cv\n  %r = extractelement <5 x float> %t, i32 4\n",
            of("a", 0.5)
        ),
        float: true,
        want: None,
        needs: &[],
        note: Some(PrecisionNote::DeviceTranscendentals),
    });
    v.push(Kernel {
        name: "pow_notes_device_transcendentals",
        body: format!(
            "{}  %r = call float @air.pow.f32(float %a, float 3.0)\n",
            of("a", -2.0)
        ),
        float: true,
        want: None,
        needs: &[],
        note: Some(PrecisionNote::DeviceTranscendentals),
    });
    v.push(k(
        "mad_sat_saturates_the_exact_result",
        format!(
            "{}{}  %s = call i8 @air.mad_sat.s.i8(i8 %a, i8 %a, i8 -100)\n  %u = call i32 @air.mad_sat.u.i32(i32 %b, i32 %b, i32 7)\n  %sw = zext i8 %s to i32\n  %r = xor i32 %sw, %u\n",
            on("a", 8, 15),
            oi("b", 0x0001_0000)
        ),
        // 15 * 15 - 100 = 125; 2^32 + 7 saturates at 0xFFFFFFFF.
        125 ^ 0xFFFF_FFFF,
    ));
    v.push(Kernel {
        name: "extract_bits_on_i64",
        body: format!(
            "{}  %w = zext i32 %a to i64\n  %h = shl i64 %w, 28\n  %e = call i64 @air.extract_bits.s.i64(i64 %h, i32 28, i32 8)\n  %r = trunc i64 %e to i32\n",
            oi("a", 0x80)
        ),
        float: false,
        want: Some(0xFFFF_FF80),
        needs: &[],
        note: None,
    });
    // 0.25 * 255 = 63.75 -> 64; 0.5 * 255 = 127.5 -> 128 (ties to even).
    v.push(k(
        "pack_unorm4x8_rounds_to_even",
        format!(
            "{}  %v = insertelement <4 x float> <float 0.0, float 0.5, float 1.0, float 1.0>, float %a, i32 0\n  %r = call i32 @air.pack.unorm4x8.v4f32(<4 x float> %v)\n",
            of("a", 0.25)
        ),
        0xFFFF_8040,
    ));
    // 0x7FFF is 1.0; 0x8000 is -32768 / 32767, clamped to -1.0.
    v.push(Kernel {
        name: "unpack_snorm2x16_clamps",
        body: format!(
            "{}  %u = call <2 x float> @air.unpack.snorm2x16.v2f32(i32 %b)\n  %x = extractelement <2 x float> %u, i32 0\n  %y = extractelement <2 x float> %u, i32 1\n  %r = fadd float %x, %y\n",
            oi("b", 0x8000_7FFF)
        ),
        float: true,
        want: Some(0),
        needs: &[],
        // The division is exact.
        note: None,
    });
    // An 8-bit float to an integer is its value under the saturating rule (448 to `char` is 127,
    // -1.0 to `uint` is 0, 57344 and -2.0 to 64 bits, held as two words where the profile lacks
    // `shaderInt64`), and records `Fp8Value`; a widening to a float does not.
    for (name, call, bits, wide, want) in [
        (
            "fp8_e4m3fn_to_char_saturates",
            "i8 @air.convert.s.i8.f.f8e4m3fn",
            0x7Eu32,
            false,
            127u32,
        ),
        (
            "fp8_e4m3_to_uint_clamps_at_zero",
            "i32 @air.convert.u.i32.f.f8e4m3",
            0xB8,
            false,
            0,
        ),
        (
            "fp8_e5m2_to_ulong",
            "i64 @air.convert.u.i64.f.f8e5m2",
            0x7B,
            true,
            57344,
        ),
        (
            "fp8_e5m2_to_long",
            "i64 @air.convert.s.i64.f.f8e5m2",
            0xC0,
            true,
            0xFFFF_FFFE,
        ),
    ] {
        let width = call.split(' ').next().unwrap_or_default();
        let narrow = if wide {
            format!("  %r = trunc {width} %c to i32\n")
        } else if width == "i32" {
            "  %r = add i32 %c, 0\n".to_owned()
        } else {
            format!("  %r = sext {width} %c to i32\n")
        };
        v.push(Kernel {
            name,
            body: format!("{}  %c = call {call}(i8 %a)\n{narrow}", on("a", 8, bits)),
            float: false,
            want: Some(want),
            needs: &[],
            note: Some(PrecisionNote::Fp8Value),
        });
    }
    v.push(Kernel {
        name: "fp8_e5m2_to_float_is_exact",
        body: format!(
            "{}  %r = call float @air.convert.f.f32.f.f8e5m2(i8 %a)\n",
            on("a", 8, 0x3C)
        ),
        float: true,
        want: Some(1.0f32.to_bits()),
        needs: &[],
        note: None,
    });
    // Metal's `snorm` `rgb10a2` words (Apple M3 Max): code 65 unpacks to its product with
    // fl(1/511), one ULP below 65/511; just below 1.5/511 packs to 2 in each 10-bit field (the f32
    // product rounds onto the tie); a NaN packs to -1 in every field.
    v.push(Kernel {
        name: "unpack_snorm_rgb10a2_multiplies_by_the_reciprocal",
        body: format!(
            "{}  %u = call <4 x float> @air.unpack.snorm.rgb10a2.v4f32(i32 %b)\n  %r = extractelement <4 x float> %u, i32 0\n",
            oi("b", 0x4411_0441)
        ),
        float: true,
        want: Some(0x3E02_4120),
        needs: &[],
        note: None,
    });
    for (name, x, want) in [
        (
            "pack_snorm_rgb10a2_rounds_the_float_product",
            "0x3F680C0600000000",
            0x0020_0802u32,
        ),
        (
            "pack_snorm_rgb10a2_packs_nan_as_minus_one",
            "0x7FF8000000000000",
            0xE018_0601,
        ),
    ] {
        v.push(k(
            name,
            format!(
                "  %a = fadd float {x}, -0.0\n  %v0 = insertelement <4 x float> undef, float %a, i32 0\n  %v1 = insertelement <4 x float> %v0, float %a, i32 1\n  %v2 = insertelement <4 x float> %v1, float %a, i32 2\n  %v = insertelement <4 x float> %v2, float %a, i32 3\n  %r = call i32 @air.pack.snorm.rgb10a2.v4f32(<4 x float> %v)\n"
            ),
            want,
        ));
    }
    // The sRGB conversions are the MSL transfer functions by the device's `Pow` around the 8-bit
    // instructions, noted on every device (the interpreter models no `Pow`).
    v.push(Kernel {
        name: "srgb_unpack_and_pack_take_the_transfer_functions",
        body: format!(
            "{}  %u = call <4 x float> @air.unpack.unorm4x8.srgb.v4f32(i32 %b)\n  %r = call i32 @air.pack.unorm4x8.srgb.v4f32(<4 x float> %u)\n",
            oi("b", 0x8901_37FF)
        ),
        float: false,
        want: None,
        needs: &[],
        note: Some(PrecisionNote::DevicePackConversion),
    });
    v.push(Kernel {
        name: "double_needs_float64",
        body: format!(
            "{}  %d = fpext float %a to double\n  %s = fadd double %d, 0x3FF8000000000000\n  %r = fptrunc double %s to float\n",
            of("a", 2.0)
        ),
        float: true,
        want: Some(3.5f32.to_bits()),
        needs: &[Need::Float64],
        note: None,
    });
    // Half and bfloat are held widened in f32 on every profile, and each operation rounds to the
    // format: 2049 rounds to 2048 and 2048 + 1 rounds back to 2048 (ties to even), where f32
    // arithmetic would give 2050.
    v.push(Kernel {
        name: "half_rounds_each_operation",
        body: format!(
            "{}  %h = fptrunc float %a to half\n  %s = fadd half %h, 0xH3C00\n  %r = fpext half %s to float\n",
            of("a", 2049.0)
        ),
        float: true,
        want: Some(2048.0f32.to_bits()),
        needs: &[],
        note: None,
    });
    v.push(Kernel {
        name: "bfloat_rounds_each_operation",
        body: format!(
            "{}  %b = fptrunc float %a to bfloat\n  %s = fadd bfloat %b, 0xR3F80\n  %r = fpext bfloat %s to float\n",
            of("a", 256.0)
        ),
        float: true,
        want: Some(256.0f32.to_bits()),
        needs: &[],
        note: None,
    });
    v.push(k(
        "half_vector_bitcast",
        format!(
            "{}  %h = fptrunc float %a to half\n  %v = insertelement <2 x half> <half 0xH3C00, half 0xH4000>, half %h, i32 1\n  %r = bitcast <2 x half> %v to i32\n",
            of("a", 3.0)
        ),
        0x4200_3C00,
    ));
    // Three 16-bit lanes (48 bits) move lane by lane, with no 64-bit integer.
    v.push(k(
        "half3_bitcast_to_short3",
        format!(
            "{}  %h = fptrunc float %a to half\n  %v = insertelement <3 x half> <half 0xH3C00, half 0xH4000, half 0xH4400>, half %h, i32 1\n  %b = bitcast <3 x half> %v to <3 x i16>\n  %x = extractelement <3 x i16> %b, i32 1\n  %y = extractelement <3 x i16> %b, i32 2\n  %xw = zext i16 %x to i32\n  %yw = zext i16 %y to i32\n  %s = shl i32 %yw, 16\n  %r = or i32 %s, %xw\n",
            of("a", 3.0)
        ),
        0x4400_4200,
    ));
    v.push(Kernel {
        name: "short3_bitcast_to_half3",
        body: format!(
            "{}  %v = insertelement <3 x i16> <i16 15360, i16 0, i16 17408>, i16 %x, i32 1\n  %h = bitcast <3 x i16> %v to <3 x half>\n  %e = extractelement <3 x half> %h, i32 1\n  %f = extractelement <3 x half> %h, i32 2\n  %s = fadd half %e, %f\n  %r = fpext half %s to float\n",
            on("x", 16, 0x4200)
        ),
        float: true,
        want: Some(7.0f32.to_bits()),
        needs: &[],
        note: None,
    });
    // 2^24 + 2^16 + 1 is just above a bfloat tie; rounding through a plain f32 would land on
    // the tie and round down to 2^24.
    v.push(Kernel {
        name: "int_to_bfloat_rounds_once",
        body: format!(
            "{}  %b = sitofp i32 %a to bfloat\n  %r = fpext bfloat %b to float\n",
            oi("a", 0x0101_0001)
        ),
        float: true,
        want: Some(0x4B81_0000),
        needs: &[],
        note: None,
    });
    // 1 + 2^-11 + 2^-40 is just above a half tie; rounding through a plain f32 would land on
    // the tie and round down to 1.
    v.push(Kernel {
        name: "double_to_half_rounds_once",
        body: format!(
            "{}  %d = fpext float %a to double\n  %e = fadd double %d, 0x3D70000000000000\n  %h = fptrunc double %e to half\n  %r = fpext half %h to float\n",
            of("a", 1.0 + 2f32.powi(-11))
        ),
        float: true,
        want: Some((1.0f32 + 2f32.powi(-10)).to_bits()),
        needs: &[Need::Float64],
        note: None,
    });
    v
}

const DECLS: &str = "
declare i32 @llvm.umin.i32(i32, i32)
declare <4 x i1> @air.convert.u.v4i1.u.v4i8(<4 x i8>)
declare i1 @air.convert.u.i1.s.i32(i32)
declare i16 @llvm.ctpop.i16(i16)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i32 @llvm.fshl.i32(i32, i32, i32)
declare i8 @llvm.abs.i8(i8, i1)
declare i8 @air.add_sat.s.i8(i8, i8)
declare i32 @air.convert.s.i32.f.f32(float)
declare i8 @air.convert.u.i8.f.f32(float)
declare float @air.clamp.f32(float, float, float)
declare float @air.round.f32(float)
declare float @air.fract.f32(float)
declare float @air.sin.f32(float)
declare float @air.sincos.f32(float, ptr)
declare <8 x float> @air.fabs.v8f32(<8 x float>)
declare i16 @air.pack.unorm.rgb565.v3f32(<3 x float>)
declare <3 x float> @air.unpack.unorm.rg11b10f.v3f32(i32)
declare <2 x half> @air.sincos.v2f16(<2 x half>, ptr)
declare float @air.dot.v8f32(<8 x float>, <8 x float>)
declare float @air.dot.v5f32(<5 x float>, <5 x float>)
declare half @air.dot.v5f16(<5 x half>, <5 x half>)
declare <5 x float> @air.sincos.v5f32(<5 x float>, ptr)
declare float @air.pow.f32(float, float)
declare i8 @air.mad_sat.s.i8(i8, i8, i8)
declare i32 @air.mad_sat.u.i32(i32, i32, i32)
declare i64 @air.mad_sat.s.i64(i64, i64, i64)
declare i64 @air.mad_sat.u.i64(i64, i64, i64)
declare <5 x i64> @llvm.umax.v5i64(<5 x i64>, <5 x i64>)
declare i64 @air.extract_bits.s.i64(i64, i32, i32)
declare i32 @air.pack.unorm4x8.v4f32(<4 x float>)
declare <2 x float> @air.unpack.snorm2x16.v2f32(i32)
declare <4 x float> @air.unpack.unorm4x8.srgb.v4f32(i32)
declare i32 @air.pack.unorm4x8.srgb.v4f32(<4 x float>)
declare float @llvm.fmuladd.f32(float, float, float)
declare float @llvm.sqrt.f32(float)
declare float @llvm.exp2.f32(float)
declare float @llvm.log2.f32(float)
declare float @llvm.copysign.f32(float, float)
declare i8 @air.convert.s.i8.f.f8e4m3fn(i8)
declare i32 @air.convert.u.i32.f.f8e4m3(i8)
declare i64 @air.convert.u.i64.f.f8e5m2(i8)
declare i64 @air.convert.s.i64.f.f8e5m2(i8)
declare float @air.convert.f.f32.f.f8e5m2(i8)
declare <4 x float> @air.unpack.snorm.rgb10a2.v4f32(i32)
declare i32 @air.pack.snorm.rgb10a2.v4f32(<4 x float>)
";

/// The generated module with every kernel of `ks`.
fn module(ks: &[Kernel]) -> String {
    let mut s = String::from(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n",
    );
    for kk in ks {
        let ty = if kk.float { "float" } else { "i32" };
        s.push_str(&format!(
            "define void @{}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{\nentry:\n{}  store {ty} %r, ptr addrspace(1) %0, align 4\n  ret void\n}}\n\n",
            kk.name, kk.body
        ));
    }
    s.push_str(DECLS);
    let refs: Vec<String> = (0..ks.len()).map(|i| format!("!{}", 20 + i)).collect();
    s.push_str(&format!(
        "\n!llvm.module.flags = !{{!0}}\n!air.kernel = !{{{}}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n",
        refs.join(", ")
    ));
    s.push_str("!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n!10 = !{}\n!11 = !{!12}\n!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}\n!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n");
    for (i, kk) in ks.iter().enumerate() {
        s.push_str(&format!("!{} = !{{ptr @{}, !10, !11}}\n", 20 + i, kk.name));
    }
    s
}

/// Assemble `text`, or `None` (skipped visibly) without `llvm-as`.
fn assemble(tag: &str, text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new(&format!("m2v-arith-{tag}"));
    let dir = scratch.path();
    let ll = dir.join("arith.ll");
    std::fs::write(&ll, text).expect("write");
    let assembled = llvm_assemble(&ll);
    match assembled {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected the arith module: {e}"),
    }
}

/// Four scalar and one two-lane 64-bit division.
const WIDE_DIVISIONS: &str = "wide_divisions_share_their_routine";

fn has(p: &DeviceProfile, n: Need) -> bool {
    let l = &p.fields().leaf;
    match n {
        Need::Float64 => l.float64,
    }
}

/// The typed module of `entry` on `p`, run on the interpreter: the stored word.
fn interpret(bitcode: &[u8], entry: &str, p: &DeviceProfile) -> u32 {
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
fn arith_kernels_validate_and_compute_on_every_profile() {
    let ks = kernels();
    let Some(bitcode) = assemble("kernels", &module(&ks)) else {
        return;
    };
    // Every need is both met (and computed) and refused on some profile.
    let (mut met, mut refused) = (Vec::new(), Vec::new());
    let mut computed = 0;
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let missing = kk.needs.iter().copied().find(|n| !has(&p, *n));
            match (compile(&bitcode, kk.name, &p), missing) {
                (Ok(a), None) => {
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                    // The plan's estimate bounds the emitted words.
                    let plan = plan_bitcode(
                        &bitcode,
                        EntryRef::Name(kk.name),
                        &FunctionConstantValues::defaults(),
                        &p,
                        &LowerInputs::new(),
                        &mut meter(),
                    )
                    .unwrap_or_else(|e| panic!("{what}: plan: {e}"));
                    let words = a.module.words.len() as u64;
                    assert!(
                        plan.estimate.words >= words,
                        "{what}: estimate {} < emitted {words}",
                        plan.estimate.words
                    );
                    if let Some(n) = kk.note {
                        // A maths function off Metal's own functions records the device's
                        // precision.
                        let n = match n {
                            PrecisionNote::DeviceTranscendentals
                                if p.fields().float_functions != FloatFunctions::Metal =>
                            {
                                PrecisionNote::NativeTranscendentals
                            }
                            n => n,
                        };
                        assert!(a.precision.contains(&n), "{what}: {:?}", a.precision);
                    }
                    // Only an 8-bit float to integer conversion reads the unverified value.
                    assert_eq!(
                        a.precision.contains(&PrecisionNote::Fp8Value),
                        kk.note == Some(PrecisionNote::Fp8Value),
                        "{what}: {:?}",
                        a.precision
                    );
                    assert!(
                        a.strategies.entries.keys().any(|n| n.starts_with("arith.")),
                        "{what}: {:?}",
                        a.strategies.entries
                    );
                    if let Some(want) = kk.want {
                        let got = interpret(&bitcode, kk.name, &p);
                        assert_eq!(got, want, "{what}: {got:#x} != {want:#x}");
                        computed += 1;
                    }
                    met.extend(kk.needs.iter().copied());
                }
                (Err(Error::Unsupported(u)), Some(n)) => {
                    let (feature, needs) = match n {
                        Need::Float64 => (
                            Feature::Float64Arithmetic,
                            Needs::AnyOf(vec![vec![Capability::Feature(VkFeature::ShaderFloat64)]]),
                        ),
                    };
                    assert_eq!((u.feature, u.needs), (feature, needs), "{what}");
                    assert!(u.location.is_some(), "{what}: located");
                    refused.push(n);
                }
                (other, missing) => panic!("{what}: missing {missing:?}, got {other:?}"),
            }
        }
    }
    assert!(
        met.contains(&Need::Float64) && refused.contains(&Need::Float64),
        "Float64 met and refused"
    );
    assert!(
        computed >= PROFILES.len() * 10,
        "{computed} kernels computed"
    );
}

/// The number of `op` instructions in module `words`.
fn count_op(words: &[u32], op: crate::spv::Op) -> usize {
    let mut i = 5;
    let mut n = 0;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        n += usize::from(opcode == op as u32);
        i += len;
    }
    n
}

/// The function control of each `OpFunction` in module `words`, in order.
fn function_controls(words: &[u32]) -> Vec<u32> {
    let mut i = 5;
    let mut controls = Vec::new();
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if opcode == crate::spv::Op::Function as u32 {
            controls.push(words[i + 3]);
        }
        i += len;
    }
    controls
}

/// Without `shaderInt64` the wide division is one function per lane count, which each wide division
/// on that many lanes calls, so a module holds it once per lane count, not once per use; with
/// `shaderInt64` there is no routine. No function takes a control on these profiles; on the same
/// device under MoltenVK 1.4.1, whose SPIRV-Cross writes every function inlined at each call
/// (`DriverWorkaround::FunctionsInlinedAtEveryCall`), each routine is declared `DontInline` and the
/// entry is not.
#[test]
fn wide_divisions_share_one_routine_per_form() {
    let ks: Vec<Kernel> = kernels()
        .into_iter()
        .filter(|kk| kk.name == WIDE_DIVISIONS)
        .collect();
    let Some(bitcode) = assemble("divisions", &module(&ks)) else {
        return;
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/vk12-floor.json");
    let mut desc = DeviceDescription::from_khronos_json(
        &std::fs::read_to_string(&path).expect("profile"),
        None,
    )
    .expect("description");
    desc.driver_id = Some(crate::vk::DriverId::MOLTENVK);
    desc.driver_version = 10401;
    let converted = Device::new(desc, exact_maths())
        .expect("device")
        .profile()
        .clone();
    assert!(converted
        .fields()
        .workarounds
        .contains(&DriverWorkaround::FunctionsInlinedAtEveryCall));
    let cases = [
        (
            "vk12-floor",
            profile("vk12-floor", exact_maths()).expect("profile"),
            3,
            5,
            0,
        ),
        ("vk12-floor under MoltenVK 1.4.1", converted, 3, 5, 0x2),
        (
            "amd-rdna3-radv",
            profile("amd-rdna3-radv", exact_maths()).expect("profile"),
            1,
            0,
            0,
        ),
    ];
    for (stem, p, functions, calls, routine_control) in cases {
        let a = compile(&bitcode, WIDE_DIVISIONS, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let words = &a.module.words;
        assert_eq!(
            (
                count_op(words, crate::spv::Op::Function),
                count_op(words, crate::spv::Op::FunctionCall)
            ),
            (functions, calls),
            "{stem}: the entry and one routine per lane count (i64, <2 x i64>), one call per division"
        );
        let mut controls = function_controls(words);
        controls.sort_unstable();
        let mut want = vec![routine_control; functions - 1];
        want.insert(0, 0);
        assert_eq!(
            controls, want,
            "{stem}: each routine's control, and none on the entry"
        );
    }
}

/// `float` and `half` fused multiply-adds (`air.fma`, `llvm.fma`, `llvm.fmuladd`) take the
/// profile's route, inline: `OpFmaKHR` where `shaderFmaFloat32` is enabled (moltenvk-m3), else the
/// device's `Fma` (lavapipe, vk12-floor), noted `NativeFma`. A1 case 52343cc4's lane (`a = 1 +
/// 2^-23`, `b = 1 - 2^-24`, `c = -RN(a * b)`), on one lane and on a vector's first, gives the
/// product's rounding error, which an unfused `Fma` loses (the interpreter's `Fma` rounds once); a
/// `half` triple whose `float` sum is a `half` midpoint gives the once-rounded `half` as the
/// device's 16-bit `Fma`, which the driver computes as Metal's `half` `fma` (moltenvk-m3), and
/// rounds twice as the device's `Fma` on the operands held in `float`, its result rounded to
/// `half` (lavapipe, vk12-floor). No module notes `DeviceArithmetic`.
#[test]
fn fused_multiply_adds_take_the_profiles_route() {
    let (a, b) = (1.0f32 + f32::EPSILON, 1.0f32 - f32::EPSILON / 2.0);
    let c = -(a * b);
    let fused = a.mul_add(b, c);
    assert_ne!(fused, 0.0);
    let inputs = format!("{}{}{}", of("a", a), of("b", b), of("c", c));
    // `<2 x float>` lanes (a, 2), (b, 3), (c, 1): the fused residue and 7.
    let vector = "  %va0 = insertelement <2 x float> <float 0.0, float 2.0>, float %a, i32 0\n  %vb0 = insertelement <2 x float> <float 0.0, float 3.0>, float %b, i32 0\n  %vc0 = insertelement <2 x float> <float 0.0, float 1.0>, float %c, i32 0\n";
    let float_kernel = Kernel {
        name: "fma_float_rounds_once",
        body: format!(
            "{inputs}{vector}  %s = call float @air.fma.f32(float %a, float %b, float %c)\n  %m = call float @llvm.fmuladd.f32(float %a, float %b, float %c)\n  %v = call <2 x float> @air.fma.v2f32(<2 x float> %va0, <2 x float> %vb0, <2 x float> %vc0)\n  %v0 = extractelement <2 x float> %v, i32 0\n  %v1 = extractelement <2 x float> %v, i32 1\n  %si = bitcast float %s to i32\n  %mi = bitcast float %m to i32\n  %v0i = bitcast float %v0 to i32\n  %v1i = bitcast float %v1 to i32\n  %x = xor i32 %si, %mi\n  %y = xor i32 %si, %v0i\n  %z = or i32 %x, %y\n  %r = xor i32 %z, %v1i\n"
        ),
        float: false,
        // s, m and the vector's first lane are the same fused residue: their xors are 0,
        // leaving 7.0's bits.
        want: Some(7.0f32.to_bits()),
        needs: &[],
        note: None,
    };
    // 0x400a * 0x0bec + 0xbc9f: the exact sum rounds to 0xbc9f, the float sum to a half
    // midpoint that rounds away.
    let half_kernel = Kernel {
        name: "fma_half_rounds_once",
        body: format!(
            "{}{}{}  %h = bitcast i16 %t to half\n  %g = bitcast i16 %s to half\n  %u = bitcast i16 %v to half\n  %x = call half @air.fma.f16(half %h, half %g, half %u)\n  %r = fpext half %x to float\n",
            on("t", 16, 0x400A),
            on("s", 16, 0x0BEC),
            on("v", 16, 0xBC9F)
        ),
        float: true,
        want: Some((-1183.0f32 / 1024.0).to_bits()),
        needs: &[],
        note: None,
    };
    let ks = [float_kernel, half_kernel];
    let text = module(&ks).replace(
        DECLS,
        &format!(
            "{DECLS}declare float @air.fma.f32(float, float, float)\ndeclare <2 x float> @air.fma.v2f32(<2 x float>, <2 x float>, <2 x float>)\ndeclare half @air.fma.f16(half, half, half)\n"
        ),
    );
    let Some(bitcode) = assemble("fma-routes", &text) else {
        return;
    };
    // (profile, OpFmaKHR of the float kernel, whether the routes are the device's `Fma`).
    for (stem, khr, native) in [
        ("moltenvk-m3", 3, false),
        ("lavapipe", 0, true),
        ("vk12-floor", 0, true),
    ] {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            for n in [
                PrecisionNote::DeviceArithmetic,
                PrecisionNote::FastMathRewrites,
            ] {
                assert!(!a.precision.contains(&n), "{what}: {:?}", a.precision);
            }
            assert_eq!(
                a.precision.contains(&PrecisionNote::NativeFma),
                native,
                "{what}: {:?}",
                a.precision
            );
            let float = kk.name == "fma_float_rounds_once";
            assert_eq!(
                (
                    count_op(&words, crate::spv::Op::FmaKHR),
                    count_op(&words, crate::spv::Op::Function),
                    count_op(&words, crate::spv::Op::FunctionCall)
                ),
                (if float { khr } else { 0 }, 1, 0),
                "{what}: OpFmaKHR, functions, calls"
            );
            let got = interpret(&bitcode, kk.name, &p);
            // The device's `Fma` in `float` rounds the `half` midpoint to even, away from the
            // once-rounded `half`.
            let want = if native && !float {
                Some((-1182.0f32 / 1024.0).to_bits())
            } else {
                kk.want
            };
            assert_eq!(Some(got), want, "{what}: {got:#x}");
        }
    }
}

/// `<n x ty>` `%name` of opaque lanes `vals` (each `fadd <v>, -0.0`, a `half` lane truncated
/// from its exact `float`).
fn opaque_vector(name: &str, half: bool, vals: &[f32]) -> String {
    let ty = if half { "half" } else { "float" };
    let n = vals.len();
    let mut s = String::new();
    let mut prev = "poison".to_owned();
    for (i, v) in vals.iter().enumerate() {
        let lane = format!("{name}.{i}");
        s.push_str(&of(&lane, *v));
        let mut at = format!("%{lane}");
        if half {
            s.push_str(&format!("  %{lane}.h = fptrunc float %{lane} to half\n"));
            at = format!("%{lane}.h");
        }
        let next = if i + 1 == n {
            format!("%{name}")
        } else {
            format!("%{name}.v{i}")
        };
        s.push_str(&format!(
            "  {next} = insertelement <{n} x {ty}> {prev}, {ty} {at}, i32 {i}\n"
        ));
        prev = next;
    }
    s
}

/// `dot` on `float` and `half` is the device's `OpDot` on every profile, with and without fast
/// math, on two to four lanes; of more, each chunk's `OpDot` summed. Metal's is the products' fused
/// chain: a `float` one on a profile whose `float` functions are Metal's (moltenvk-m3) reaches
/// Metal's own `dot`, with no note; every other records `NativeDot` (a `half` one summed in its
/// `float` container), and a chunked one `DeviceArithmetic`. No fused multiply-add is emitted.
#[test]
fn dot_is_the_devices_op_dot_on_every_profile() {
    let vals = |n: usize, k: f32| (0..n).map(|i| k * (i as f32 + 1.5)).collect::<Vec<_>>();
    let mut ks = Vec::new();
    let float_names = [
        ["dot2_float", "dot3_float", "dot4_float"],
        ["dot2_float_fast", "dot3_float_fast", "dot4_float_fast"],
    ];
    for (fast, row) in float_names.iter().enumerate() {
        for (k, name) in row.iter().copied().enumerate() {
            let n = k + 2;
            let flag = if fast == 1 { "fast " } else { "" };
            ks.push(Kernel {
                name,
                body: format!(
                    "{}{}  %d = call {flag}float @air.dot.v{n}f32(<{n} x float> %x, <{n} x float> %y)\n  %r = bitcast float %d to i32\n",
                    opaque_vector("x", false, &vals(n, 0.75)),
                    opaque_vector("y", false, &vals(n, -1.25))
                ),
                float: false,
                want: None,
                needs: &[],
                note: None,
            });
        }
    }
    for (k, name) in ["dot2_half", "dot3_half", "dot4_half"]
        .into_iter()
        .enumerate()
    {
        let n = k + 2;
        ks.push(Kernel {
            name,
            body: format!(
                "{}{}  %d = call half @air.dot.v{n}f16(<{n} x half> %x, <{n} x half> %y)\n  %u = bitcast half %d to i16\n  %r = zext i16 %u to i32\n",
                opaque_vector("x", true, &vals(n, 0.75)),
                opaque_vector("y", true, &vals(n, -1.25))
            ),
            float: false,
            want: None,
            needs: &[],
            note: None,
        });
    }
    for (name, n, half) in [
        ("dot8_float_chunked", 8usize, false),
        ("dot16_half_chunked", 16, true),
    ] {
        let (ty, sym) = if half {
            ("half", "f16")
        } else {
            ("float", "f32")
        };
        let store = if half {
            "  %u = bitcast half %d to i16\n  %r = zext i16 %u to i32\n"
        } else {
            "  %r = bitcast float %d to i32\n"
        };
        ks.push(Kernel {
            name,
            body: format!(
                "{}{}  %d = call {ty} @air.dot.v{n}{sym}(<{n} x {ty}> %x, <{n} x {ty}> %y)\n{store}",
                opaque_vector("x", half, &vals(n, 0.75)),
                opaque_vector("y", half, &vals(n, -1.25))
            ),
            float: false,
            want: None,
            needs: &[],
            note: None,
        });
    }
    // `DECLS` declares the dots of 5 and 8 `float` lanes.
    let decls: String = [2, 3, 4]
        .iter()
        .map(|n| {
            format!(
                "declare float @air.dot.v{n}f32(<{n} x float>, <{n} x float>)\ndeclare half @air.dot.v{n}f16(<{n} x half>, <{n} x half>)\n"
            )
        })
        .chain(["declare half @air.dot.v16f16(<16 x half>, <16 x half>)\n".to_owned()])
        .collect();
    let text = module(&ks).replace(DECLS, &format!("{DECLS}{decls}"));
    let Some(bitcode) = assemble("dot-native", &text) else {
        return;
    };
    let fma = |words: &[u32]| {
        count_ops(words, |w| {
            w[0] & 0xFFFF == crate::spv::Op::ExtInst as u32
                && w.get(4) == Some(&(crate::spv::GlslStd450Op::Fma as u32))
        }) + count_op(words, crate::spv::Op::FmaKHR)
    };
    for stem in [
        "moltenvk-m3",
        "amd-rdna3-radv",
        "lavapipe",
        "vk12-floor",
        "lunarg-desktop-baseline-2023",
    ] {
        for policy in [exact_maths(), DevicePolicy::default()] {
            let p = profile(stem, policy).expect("profile");
            let metal = p.fields().float_functions == FloatFunctions::Metal;
            for kk in &ks {
                let what = format!("{} on {stem}", kk.name);
                let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let (_, words) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                assert!(count_op(&words, crate::spv::Op::Dot) > 0, "{what}: OpDot");
                assert_eq!(fma(&words), 0, "{what}: a fused chain");
                let chunked = kk.name.ends_with("_chunked");
                let float = kk.name.contains("_float");
                if chunked {
                    assert!(
                        a.precision.contains(&PrecisionNote::DeviceArithmetic),
                        "{what}: {:?}",
                        a.precision
                    );
                } else {
                    assert_eq!(
                        a.precision.contains(&PrecisionNote::NativeDot),
                        !(metal && float),
                        "{what}: {:?}",
                        a.precision
                    );
                }
            }
        }
    }
}

/// `air.fast_fmod` on `float` is Metal's `fma(-trunc(x / y), y, x)` (MSL Table 8.2), of the
/// device's quotient `OpFDiv(x, y)`, on every profile's fused multiply-add: `OpFmaKHR`
/// (moltenvk-m3) or the device's `Fma` (lavapipe, vk12-floor), inline, the entry the module's one
/// function. A1 case dd3b3b99's `fmod(1e10, 3)` is Metal's `-512` (its fast quotient one ulp above
/// the correctly rounded one); on the interpreter's correctly rounded quotient, `3333333248`, the
/// fused expression gives `256`. The module notes `DeviceArithmetic` (the quotient's rounding).
#[test]
fn fast_fmod_is_metals_composition() {
    let kernel = Kernel {
        name: "fast_fmod_composes",
        body: format!(
            "{}{}  %f = call fast float @air.fast_fmod.f32(float %x, float %y)\n  %r = fadd float %f, 0.0\n",
            of("x", 1e10),
            of("y", 3.0)
        ),
        float: true,
        want: Some(256.0f32.to_bits()),
        needs: &[],
        note: None,
    };
    let ks = [kernel];
    let text = module(&ks).replace(
        DECLS,
        &format!("{DECLS}declare float @air.fast_fmod.f32(float, float)\n"),
    );
    let Some(bitcode) = assemble("fast-fmod", &text) else {
        return;
    };
    for (stem, khr, functions) in [
        ("moltenvk-m3", 1, 1),
        ("lavapipe", 0, 1),
        ("vk12-floor", 0, 1),
    ] {
        let p = profile(stem, exact_maths()).expect("profile");
        let kk = &ks[0];
        let what = format!("{} on {stem}", kk.name);
        let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        assert!(
            a.precision.contains(&PrecisionNote::DeviceArithmetic),
            "{what}: {:?}",
            a.precision
        );
        assert_eq!(
            (
                count_op(&words, crate::spv::Op::FmaKHR),
                count_op(&words, crate::spv::Op::Function),
                count_op(&words, crate::spv::Op::FRem)
            ),
            (khr, functions, 0),
            "{what}: OpFmaKHR, functions, OpFRem"
        );
        let got = interpret(&bitcode, kk.name, &p);
        assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
    }
}

/// `fcmp one` is the negation of `OpFUnordEqual`, never `OpFOrdNotEqual`, on every profile:
/// SPIRV-Cross, which MoltenVK uses, renders `OpFOrdNotEqual` as MSL's `!=`, true for a NaN
/// operand, where `one` (and Metal, authored case df9e364a) answers false. The kernel's bits are
/// `one(NaN, 1)` (0), `one(1, 2)` (1), `one(2, 2)` (0), `ueq(NaN, 1)` (1), and the two lanes of
/// `one(<NaN, 1>, <1, 2>)` (0, 1).
#[test]
fn an_ordered_not_equal_is_the_negated_unordered_equal() {
    let compares = "  %c0 = fcmp one float %n, %one\n  %c1 = fcmp one float %one, %two\n  %c2 = fcmp one float %two, %two\n  %c3 = fcmp ueq float %n, %one\n  %va = insertelement <2 x float> <float 0.0, float 1.0>, float %n, i32 0\n  %vb = insertelement <2 x float> <float 0.0, float 2.0>, float %one, i32 0\n  %vc = fcmp one <2 x float> %va, %vb\n  %c4 = extractelement <2 x i1> %vc, i32 0\n  %c5 = extractelement <2 x i1> %vc, i32 1\n";
    let mut bits = String::from("  %b0 = zext i1 %c0 to i32\n");
    for i in 1..6 {
        bits.push_str(&format!(
            "  %z{i} = zext i1 %c{i} to i32\n  %s{i} = shl i32 %z{i}, {i}\n  %b{i} = or i32 %b{}, %s{i}\n",
            i - 1
        ));
    }
    let kernel = k(
        "ordered_not_equal",
        format!(
            "{}{}{}{compares}{bits}  %r = add i32 %b5, 0\n",
            of("n", f32::NAN),
            of("one", 1.0),
            of("two", 2.0)
        ),
        0b10_1010,
    );
    let ks = [kernel];
    let Some(bitcode) = assemble("ordered-not-equal", &module(&ks)) else {
        return;
    };
    for stem in ["moltenvk-m3", "lavapipe", "vk12-floor"] {
        let p = profile(stem, exact_maths()).expect("profile");
        let kk = &ks[0];
        let what = format!("{} on {stem}", kk.name);
        let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        assert_eq!(
            count_op(&words, crate::spv::Op::FOrdNotEqual),
            0,
            "{what}: OpFOrdNotEqual"
        );
        // One per `one` and `ueq`; the vector `one` may be split into its lanes.
        assert!(
            count_op(&words, crate::spv::Op::FUnordEqual) >= 4,
            "{what}: OpFUnordEqual"
        );
        let got = interpret(&bitcode, kk.name, &p);
        assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
    }
}

/// Metal divides `float` under `arcp` as `x` times its fast reciprocal of `y` (a probe cases
/// 9496749c and 3921d3e2, every flag set with `arcp`, every lane): the probe's lane 0, `0x42aa3a21
/// / 0x44c80075`, is `0x3d59e381` under `arcp` (`x` times `1 / y` rounded). The module takes the
/// device's `OpFDiv(x, y)` on every profile, with no `OpFmaKHR` and its note: here the
/// interpreter's correctly rounded quotient, `0x3d59e382`, as without a flag
/// ([`a_division_without_arcp_is_the_devices_quotient`]), and `6 / 3` is `2`.
#[test]
fn an_arcp_division_is_the_devices_quotient() {
    let xy = format!(
        "{}{}",
        of("x", f32::from_bits(0x42AA_3A21)),
        of("y", f32::from_bits(0x44C8_0075))
    );
    let kernel = |name: &'static str, body: String, want: u32| Kernel {
        name,
        body,
        float: true,
        want: Some(want),
        needs: &[],
        note: None,
    };
    let ks = [
        kernel(
            "arcp_divide",
            format!("{xy}  %r = fdiv arcp float %x, %y\n"),
            0x3D59_E382,
        ),
        kernel(
            "fast_divide",
            format!("{xy}  %r = fdiv fast float %x, %y\n"),
            0x3D59_E382,
        ),
        kernel(
            "arcp_six_by_three",
            format!(
                "{}{}  %r = fdiv arcp float %six, %three\n",
                of("six", 6.0),
                of("three", 3.0)
            ),
            0x4000_0000,
        ),
    ];
    let Some(bitcode) = assemble("fast-divide", &module(&ks)) else {
        return;
    };
    for stem in ["moltenvk-m3", "lavapipe", "vk12-floor"] {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_eq!(
                (
                    count_op(&words, crate::spv::Op::FmaKHR),
                    count_op(&words, crate::spv::Op::FDiv)
                ),
                (0, 1),
                "{what}: OpFmaKHR, OpFDiv"
            );
            assert!(
                a.precision.contains(&PrecisionNote::DeviceArithmetic),
                "{what}: {:?}",
                a.precision
            );
            let got = interpret(&bitcode, kk.name, &p);
            assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
        }
    }
}

/// Every committed profile's stem (`validation/profiles`), sorted.
fn every_profile() -> Vec<String> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles");
    let mut out: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.expect("profile directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| p.file_stem().expect("stem").to_string_lossy().into_owned())
        .collect();
    out.sort();
    out
}

/// Metal's fast division answers `0x7FC00000` for a NaN divisor of either sign, a NaN numerator,
/// `0 / 0` and `inf / inf`; the module leaves that NaN to the device's `OpFDiv` on every profile
/// (RADV answered `0xffc00000` for a negative NaN divisor, a probe cases 9496749c and 3921d3e2),
/// selecting none, so the module is the same on both sides of `MetalCanonicalNans`, and records
/// `PrecisionNote::NativeNanEncoding` where the device is not measured to answer Metal's. Its
/// `f32` denormals are the device's on every committed profile. The operands are integer bits, so
/// the division is the kernel's only float operation, and it records the device's arithmetic, its
/// division and its denormals, and never `DeviceNanEncoding`. The interpreter answers a NaN, its
/// bits the device's, and `6 / 3` exactly.
#[test]
fn a_fast_division_leaves_its_nans_to_the_device() {
    let bits = |name: &str, b: u32| {
        format!("  %{name}.i = call i32 @llvm.umin.i32(i32 {b}, i32 -1)\n  %{name} = bitcast i32 %{name}.i to float\n")
    };
    let kernel = |name: &'static str, x: u32, y: u32, want: Option<u32>| Kernel {
        name,
        body: format!(
            "{}{}  %r = fdiv arcp float %x, %y\n",
            bits("x", x),
            bits("y", y)
        ),
        float: true,
        want,
        needs: &[],
        note: None,
    };
    let ks = [
        kernel("fast_one_by_nan", 0x3F80_0000, 0x7FC0_0000, None),
        kernel("fast_inf_by_nan", 0x7F80_0000, 0x7FC0_0000, None),
        kernel("fast_one_by_negative_nan", 0x3F80_0000, 0xFFC0_0000, None),
        kernel("fast_nan_by_two", 0x7FC0_0001, 0x4000_0000, None),
        kernel("fast_zero_by_zero", 0, 0, None),
        kernel("fast_inf_by_inf", 0x7F80_0000, 0x7F80_0000, None),
        kernel(
            "fast_six_by_three",
            0x40C0_0000,
            0x4040_0000,
            Some(0x4000_0000),
        ),
    ];
    let Some(bitcode) = assemble("fast-divide-specials", &module(&ks)) else {
        return;
    };
    nans_are_the_device_s(&bitcode, &ks, false, &[], &|_, fact| !fact);
    for stem in every_profile() {
        let p = profile(&stem, exact_maths()).expect("profile");
        let device_nans =
            p.takes_driver_deviation(crate::error::DriverDeviation::MetalCanonicalNans);
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            for n in [
                PrecisionNote::DeviceArithmetic,
                PrecisionNote::NativeDivision,
                PrecisionNote::DeviceDenormals,
            ] {
                assert!(
                    a.precision.contains(&n),
                    "{what}: {n:?} not in {:?}",
                    a.precision
                );
            }
            assert_eq!(
                a.precision.contains(&PrecisionNote::NativeNanEncoding),
                !device_nans,
                "{what}: {:?}",
                a.precision
            );
            assert!(
                !a.precision.contains(&PrecisionNote::DeviceNanEncoding),
                "{what}: {:?}",
                a.precision
            );
            let got = interpret(&bitcode, kk.name, &p);
            match kk.want {
                Some(want) => assert_eq!(got, want, "{what}: {got:#x}"),
                None => assert!(stores_nan(got, false), "{what}: {got:#x}"),
            }
        }
    }
}

/// An `fdiv` without `arcp` on `float`, Metal's precise division, is the device's `OpFDiv` on
/// every profile (`arith.fdiv.native`), noted `NativeDivision`: here the interpreter's correctly
/// rounded quotient (lane 0x42aa3a21 / 0x44c80075 gives 0x3D59E382, Metal's word, where `arcp`
/// gives 0x3D59E381), and `0x7FC00000` for `0 / 0`. A `half` division under every flag set,
/// `arcp` included, is the `float` quotient rounded once to `half`.
#[test]
fn a_division_without_arcp_is_the_devices_quotient() {
    let xy = format!(
        "{}{}",
        of("x", f32::from_bits(0x42AA_3A21)),
        of("y", f32::from_bits(0x44C8_0075))
    );
    // 1 / 3 in `half`: 0x3555, whose `float` value is the stored word.
    let hxy = format!(
        "{}{}  %hx = bitcast i16 %hxb to half\n  %hy = bitcast i16 %hyb to half\n",
        on("hxb", 16, 0x3C00),
        on("hyb", 16, 0x4200)
    );
    let half_q = (f32::from_bits(0x3EAA_A000)).to_bits();
    let kernel = |name: &'static str, body: String, want: u32| Kernel {
        name,
        body,
        float: true,
        want: Some(want),
        needs: &[],
        note: None,
    };
    let ks = [
        kernel(
            "plain_divide",
            format!("{xy}  %r = fdiv float %x, %y\n"),
            0x3D59_E382,
        ),
        kernel(
            "divide_without_arcp",
            format!("{xy}  %r = fdiv nnan ninf nsz contract float %x, %y\n"),
            0x3D59_E382,
        ),
        kernel(
            "divide_zero_by_zero",
            format!("{}  %r = fdiv float %z, %z\n", of("z", 0.0)),
            0x7FC0_0000,
        ),
        kernel(
            "divide_half",
            format!("{hxy}  %q = fdiv half %hx, %hy\n  %r = fpext half %q to float\n"),
            half_q,
        ),
        kernel(
            "divide_half_under_arcp",
            format!("{hxy}  %q = fdiv fast half %hx, %hy\n  %r = fpext half %q to float\n"),
            half_q,
        ),
    ];
    let Some(bitcode) = assemble("precise-divide", &module(&ks)) else {
        return;
    };
    for stem in ["moltenvk-m3", "lavapipe", "vk12-floor"] {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert!(
                a.precision.contains(&PrecisionNote::NativeDivision),
                "{what}: {:?}",
                a.precision
            );
            assert!(
                a.strategies.entries.contains_key("arith.fdiv.native"),
                "{what}: {:?}",
                a.strategies.entries
            );
            let got = interpret(&bitcode, kk.name, &p);
            assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
        }
    }
}

/// `air.fast_sincos` on `float`, like `air.sincos`, is the device's `Sin` and `Cos` on every
/// profile, in a fast-math module and out of one: the module compares against no limit (Metal's
/// `fast::sincos` answers `+0` for both past `float(π/2) · 2^22`, `0x4AC90FDB`, where the device
/// answers its functions, `PrecisionNote::NativeTranscendentals` off Metal's own). (The
/// interpreter does not model `Sin`; the shape is the module's.)
#[test]
fn fast_sincos_is_the_devices_sine_and_cosine() {
    let body = |call: &str| {
        format!(
            "  %c = alloca float, align 4\n{}  %s = call float @{call}(float %a, ptr %c)\n  %cv = load float, ptr %c, align 4\n  %r = fadd float %s, %cv\n",
            of("a", 2.0)
        )
    };
    let kernel = |name: &'static str, call: &str| Kernel {
        name,
        body: body(call),
        float: true,
        want: None,
        needs: &[],
        note: None,
    };
    // Every operation with every fast-math flag: the module declares no
    // `SignedZeroInfNanPreserve`, so MoltenVK compiles it with fast math.
    let fast = Kernel {
        name: "fast_sincos_in_a_fast_module",
        body: format!(
            "  %c = alloca float, align 4\n  %a = fadd fast float 0x{:016X}, -0.0\n  %s = call fast float @air.fast_sincos.f32(float %a, ptr %c)\n  %cv = load float, ptr %c, align 4\n  %r = fadd fast float %s, %cv\n",
            f64::from(2.0f32).to_bits()
        ),
        float: true,
        want: None,
        needs: &[],
        note: None,
    };
    let ks = [
        kernel("fast_sincos_out_of_a_fast_module", "air.fast_sincos.f32"),
        kernel("sincos_keeps_the_device", "air.sincos.f32"),
        fast,
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!("{DECLS}declare float @air.fast_sincos.f32(float, ptr)\n"),
    );
    let Some(bitcode) = assemble("fast-sincos", &text) else {
        return;
    };
    for stem in ["moltenvk-m3", "lavapipe", "vk12-floor"] {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let limits = count_ops(&words, |w| {
                w[0] & 0xFFFF == crate::spv::Op::Constant as u32
                    && w.len() == 4
                    && w[3] == 0x4AC9_0FDB
            });
            assert_eq!(limits, 0, "{what}: a flush limit");
        }
    }
}

/// `float` `+ - *`, a comparison and `OpFmaKHR` record `DeviceDenormals` exactly where the
/// profile does not take `DriverDeviation::MetalFloatDenormals` (lavapipe), whose device keeps or
/// flushes them by its own controls; a remainder, the device's `OpFRem`, records it on every
/// profile; `sign`, the device's `FSign`, records it off Metal's own functions, and `half`
/// arithmetic on none; `fast::atan2`, whose quotient is the device's division, records it off the
/// deviation.
/// None records `DeviceNanEncoding`: the NaN is Metal's, the device's where the profile states it
/// and selected elsewhere. The operands come from bits (`llvm.umin` of constants, bitcast), not
/// from `float` arithmetic.
#[test]
fn device_float_environment_notes_follow_the_operations() {
    // 0x3fc03c00: a float near 1.5 whose low half is half 1.0.
    let w = "  %w = call i32 @llvm.umin.i32(i32 1069562880, i32 2147483647)\n  %x = bitcast i32 %w to float\n";
    let x = f32::from_bits(0x3FC0_3C00);
    let none = &[][..];
    let ks = [
        k("env_fadd", format!("{w}  %s = fadd float %x, %x\n  %r = bitcast float %s to i32\n"), (x + x).to_bits()),
        // The device's remainder of `x` by itself.
        k("env_frem", format!("{w}  %s = frem float %x, %x\n  %r = bitcast float %s to i32\n"), 0),
        k("env_sign", format!("{w}  %s = call float @air.sign.f32(float %x)\n  %r = bitcast float %s to i32\n"), 1.0f32.to_bits()),
        k("env_fcmp", format!("{w}  %c = fcmp olt float %x, 0.0\n  %r = zext i1 %c to i32\n"), 0),
        k("env_half", format!("{w}  %t = trunc i32 %w to i16\n  %h = bitcast i16 %t to half\n  %s = fadd half %h, %h\n  %u = bitcast half %s to i16\n  %r = zext i16 %u to i32\n"), 0x4000),
        k("env_fma", format!("{w}  %s = call float @air.fma.f32(float %x, float %x, float %x)\n  %r = bitcast float %s to i32\n"), x.mul_add(x, x).to_bits()),
        // `fast::atan2` selects Metal's NaN but returns the device's `Atan` of the device's
        // quotient. The value is not checked here.
        Kernel {
            want: None,
            ..k("env_fast_atan2", format!("{w}  %s = call float @air.fast_atan2.f32(float %x, float %x)\n  %r = bitcast float %s to i32\n"), 0)
        },
    ];
    // The notes each kernel records on moltenvk-m3 and on lavapipe.
    let dd = &[PrecisionNote::DeviceDenormals][..];
    let notes: [(&[PrecisionNote], &[PrecisionNote]); 7] = [
        (none, dd),
        (dd, dd),
        (none, dd),
        (none, dd),
        (none, none),
        (none, dd),
        (none, dd),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!("{DECLS}declare float @air.sign.f32(float)\ndeclare float @air.fma.f32(float, float, float)\ndeclare float @air.fast_atan2.f32(float, float)\n"),
    );
    let Some(bitcode) = assemble("float-environment", &text) else {
        return;
    };
    for (stem, column) in [("moltenvk-m3", 0), ("lavapipe", 1)] {
        let p = profile(stem, exact_maths()).expect("profile");
        for (kk, (m3, lp)) in ks.iter().zip(&notes) {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let want = if column == 0 { *m3 } else { *lp };
            for n in [
                PrecisionNote::DeviceDenormals,
                PrecisionNote::DeviceNanEncoding,
            ] {
                assert_eq!(
                    a.precision.contains(&n),
                    want.contains(&n),
                    "{what}: {n:?} in {:?}",
                    a.precision
                );
            }
            if kk.want.is_some() {
                let got = interpret(&bitcode, kk.name, &p);
                assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
            }
        }
    }
}

/// An `frem` on `float` without every fast-math flag is the device's `OpFRem`, noted
/// `DeviceArithmetic` and `DeviceDenormals`, where Metal computes `x - y * trunc(x / y)` of its
/// precise quotient, each step rounded: `frem(1, 0.1)` is `+0` on the M3 Max, and here the
/// interpreter's exact remainder, `0x3DCCCCCB`. With every flag it is Metal's `fast::fmod`
/// (`arith.fmod.fast`), with no `OpFRem`.
#[test]
fn a_float_remainder_is_the_devices_without_every_flag() {
    let one = "  %w = call i32 @llvm.umin.i32(i32 1065353216, i32 2147483647)\n  %x = bitcast i32 %w to float\n";
    let tenth = "  %v = call i32 @llvm.umin.i32(i32 1036831949, i32 2147483647)\n  %y = bitcast i32 %v to float\n";
    let ks = [
        k(
            "frem_precise",
            format!("{one}{tenth}  %s = frem float %x, %y\n  %r = bitcast float %s to i32\n"),
            0x3DCC_CCCB,
        ),
        Kernel {
            want: None,
            ..k(
                "frem_fast",
                format!(
                    "{one}{tenth}  %s = frem fast float %x, %y\n  %r = bitcast float %s to i32\n"
                ),
                0,
            )
        },
    ];
    let Some(bitcode) = assemble("frem", &module(&ks)) else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let precise = kk.name == "frem_precise";
            assert_eq!(
                count_op(&words, crate::spv::Op::FRem),
                usize::from(precise),
                "{what}: OpFRem"
            );
            if precise {
                for n in [
                    PrecisionNote::DeviceArithmetic,
                    PrecisionNote::DeviceDenormals,
                ] {
                    assert!(a.precision.contains(&n), "{what}: {n:?}");
                }
                let got = interpret(&bitcode, kk.name, &p);
                assert_eq!(got, 0x3DCC_CCCB, "{what}: {got:#x}");
            } else {
                assert!(
                    a.strategies.entries.contains_key("arith.fmod.fast"),
                    "{what}: {:?}",
                    a.strategies.entries
                );
            }
        }
    }
}

/// Everything the arithmetic family leaves unlowered is a typed, located `NotImplemented` refusal
/// of the arith family, never an internal error or wrong output.
#[test]
fn unlowered_arithmetic_refuses_as_not_implemented() {
    let cases = [
        k(
            "pack_to_a_packed_float",
            format!(
                "{}  %v = insertelement <3 x float> zeroinitializer, float %a, i32 0\n  %r = call i32 @air.pack.unorm.rg11b10f.v3f32(<3 x float> %v)\n",
                of("a", 1.0)
            ),
            0,
        ),
    ];
    let text = module(&cases).replace(
        DECLS,
        &format!("{DECLS}declare i32 @air.pack.unorm.rg11b10f.v3f32(<3 x float>)\n"),
    );
    let Some(bitcode) = assemble("refusals", &text) else {
        return;
    };
    let p = profile("moltenvk-m3", exact_maths()).expect("profile");
    for kk in &cases {
        match compile(&bitcode, kk.name, &p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    u.feature,
                    Feature::Lowering(LowerFamily::Arith),
                    "{}",
                    kk.name
                );
                assert_eq!(u.needs, Needs::NotImplemented, "{}", kk.name);
                assert!(u.location.is_some(), "{}: located", kk.name);
            }
            other => panic!(
                "{}: expected a NotImplemented refusal, got {other:?}",
                kk.name
            ),
        }
    }
}

/// The result ids of every `op` instruction in `words`, and the ids decorated `NoContraction`.
fn results_and_no_contraction(words: &[u32], op: crate::spv::Op) -> (Vec<u32>, Vec<u32>) {
    let (mut results, mut decorated) = (Vec::new(), Vec::new());
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if opcode == crate::spv::Op::Decorate as u32
            && words[i + 2] == crate::spv::Decoration::NoContraction as u32
        {
            decorated.push(words[i + 1]);
        }
        if opcode == op as u32 {
            results.push(words[i + 2]);
        }
        i += len;
    }
    (results, decorated)
}

/// The results of every `GLSL.std.450 Fma` whose addend is a `-0.0` `float` constant.
fn glsl_fma_results(words: &[u32]) -> Vec<u32> {
    const GLSL_FMA: u32 = 50;
    let (mut results, mut negative_zeros) = (Vec::new(), Vec::new());
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if opcode == crate::spv::Op::Constant as u32 && len == 4 && words[i + 3] == 0x8000_0000 {
            negative_zeros.push(words[i + 2]);
        }
        if opcode == crate::spv::Op::ExtInst as u32
            && words[i + 4] == GLSL_FMA
            && negative_zeros.contains(&words[i + 7])
        {
            results.push(words[i + 2]);
        }
        i += len;
    }
    results
}

/// An undecorated SPIR-V float operation lets the device fuse `a * b + c` (which rounds
/// differently) and regroup it with its neighbours. LLVM grants the two separately (`contract`,
/// `reassoc`), so only an operation carrying both is left undecorated, for the device's compiler to
/// group and fuse as it chooses; every other addition is decorated `NoContraction`, and every other
/// multiply is the product rounded on its own, decorated `NoContraction`: an `OpFMul` where the
/// profile's `float` functions are the device's, and `GLSL.std.450 Fma(a, b, -0.0)` where they are
/// Metal's, whose SPIR-V reaches Metal through SPIRV-Cross. The decoration matters where the
/// addition may be fused: Mesa splits a contractable `Fma` into a multiply and an add and folds the
/// `-0.0` away, and the multiply may then be fused with the `reassoc contract` add of
/// `mul_strict_add_fast` (NIR's `fuse` option on older Mesa, or a backend such as ACO) unless the
/// multiply is `NoContraction`. Nothing is fused by the translation: each pair is its multiply and
/// its addition in the AIR's order, and the interpreter, which rounds each operation on its own,
/// answers `(1 + 2^-12)^2 - (1 + 2^-11)` as 0 (2^-24 fused) on every profile.
#[test]
fn float_ops_without_contract_and_reassoc_are_no_contraction() {
    let inputs = format!(
        "{}{}",
        of("a", 1.0 + 2f32.powi(-12)),
        of("c", -(1.0 + 2f32.powi(-11)))
    );
    let kernel = |name: &'static str, mul: &str, add: &str| Kernel {
        name,
        body: format!("{inputs}  %p = fmul {mul}float %a, %a\n  %r = fadd {add}float %p, %c\n"),
        float: true,
        want: Some(0),
        needs: &[],
        note: None,
    };
    let ks = [
        kernel("mul_add_strict", "", ""),
        kernel("mul_add_contract", "contract ", "contract "),
        kernel("mul_add_reassoc", "reassoc ", "reassoc "),
        kernel(
            "mul_add_reassoc_contract",
            "reassoc contract ",
            "reassoc contract ",
        ),
        kernel("mul_add_fast", "fast ", "fast "),
        kernel("mul_strict_add_fast", "", "reassoc contract "),
    ];
    let Some(bitcode) = assemble("contract", &module(&ks)) else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (muls, decorated) = results_and_no_contraction(&words, crate::spv::Op::FMul);
            let (adds, _) = results_and_no_contraction(&words, crate::spv::Op::FAdd);
            let fmas = glsl_fma_results(&words);
            let (khr, _) = results_and_no_contraction(&words, crate::spv::Op::FmaKHR);
            assert!(khr.is_empty(), "{what}: fused {khr:?}");
            let mul_rewritable = matches!(kk.name, "mul_add_reassoc_contract" | "mul_add_fast");
            let add_rewritable = mul_rewritable || kk.name == "mul_strict_add_fast";
            // A multiply carrying both flags is the device's `OpFMul`, undecorated; any other is
            // the profile's product with `NoContraction`.
            let metal = p.fields().float_functions == FloatFunctions::Metal;
            let mul = match (muls.as_slice(), fmas.as_slice()) {
                ([m], []) if mul_rewritable || !metal => *m,
                ([], [m]) if !mul_rewritable && metal => *m,
                _ => panic!("{what}: one multiply in the profile's form: {muls:?} {fmas:?}"),
            };
            assert_eq!(
                decorated.contains(&mul),
                !mul_rewritable,
                "{what}: %{mul} NoContraction"
            );
            // The input fadds carry no flags; the last fadd is the kernel's.
            let add = *adds.last().expect("an fadd");
            assert_eq!(
                decorated.contains(&add),
                !add_rewritable,
                "{what}: %{add} NoContraction"
            );
            assert_eq!(interpret(&bitcode, kk.name, &p), 0, "{what}: unfused");
        }
    }
}

/// Float arithmetic the AIR lets a compiler rewrite with its neighbour notes `FastMathRewrites`:
/// two chained operations that both carry `reassoc` (through a negation), and a `contract`
/// multiply feeding a `contract` addition or subtraction (through a negation too), which the
/// device's compiler and Metal's may each fuse or not. Arithmetic whose neighbours carry no
/// matching flag does not: a lone `fast` operation on unflagged inputs, a `contract` multiply
/// feeding a `reassoc` addition, and strict arithmetic; nor does a `float` `llvm.fmuladd`, which is
/// the fused multiply-add Metal computes.
#[test]
fn arithmetic_the_air_lets_a_compiler_rewrite_notes_fast_math_rewrites() {
    let inputs = format!("{}{}{}", of("a", 1.5), of("b", 2.5), of("c", -3.0));
    let kernel = |name: &'static str, body: &str| Kernel {
        name,
        body: format!("{inputs}{body}"),
        float: true,
        want: None,
        needs: &[],
        note: None,
    };
    let noted = [
        kernel(
            "contract_mul_neg_add",
            "  %p = fmul contract float %a, %b\n  %n = fneg float %p\n  %r = fadd contract float %n, %c\n",
        ),
        kernel(
            "reassoc_sum",
            "  %s = fadd reassoc float %a, %b\n  %r = fadd reassoc float %s, %c\n",
        ),
        kernel(
            "reassoc_product_through_neg",
            "  %p = fmul reassoc float %a, %b\n  %n = fneg float %p\n  %r = fmul reassoc float %n, %c\n",
        ),
        kernel(
            "fast_mul_add",
            "  %p = fmul fast float %a, %b\n  %r = fadd fast float %p, %c\n",
        ),
        kernel(
            "contract_mul_sub",
            "  %p = fmul contract float %a, %b\n  %r = fsub contract float %c, %p\n",
        ),
    ];
    let quiet = [
        kernel(
            "fmuladd",
            "  %r = call float @llvm.fmuladd.f32(float %a, float %b, float %c)\n",
        ),
        kernel("lone_fast", "  %r = fmul fast float %a, %b\n"),
        kernel(
            "contract_mul_reassoc_add",
            "  %p = fmul contract float %a, %b\n  %r = fadd reassoc float %p, %c\n",
        ),
        kernel(
            "strict_mul_add",
            "  %p = fmul float %a, %b\n  %r = fadd float %p, %c\n",
        ),
    ];
    let noting = noted.len();
    let ks: Vec<Kernel> = noted.into_iter().chain(quiet).collect();
    let Some(bitcode) = assemble("rewrites", &module(&ks)) else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for (i, kk) in ks.iter().enumerate() {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_eq!(
                a.precision.contains(&PrecisionNote::FastMathRewrites),
                i < noting,
                "{what}: {:?}",
                a.precision
            );
        }
    }
}

/// Growth report: the same chain of 16 float operations on `float`, `half`, and `bfloat`. Each
/// widened `half` operation adds its rounding; each `bfloat` operation adds its integer rounding,
/// over the `float` chain. The measured words are printed (`--nocapture`) and bounded so a growth
/// regression fails.
#[test]
fn narrow_float_module_growth_is_bounded() {
    let chain = |ty: &str| -> String {
        let (enter, leave) = match ty {
            "float" => (
                "  %x0 = fadd float %a, -0.0\n".to_owned(),
                "  %r = fadd float %x16, -0.0\n".to_owned(),
            ),
            t => (
                format!("  %x0 = fptrunc float %a to {t}\n"),
                format!("  %r = fpext {t} %x16 to float\n"),
            ),
        };
        let mut s = format!("{}{enter}", of("a", 1.5));
        for i in 0..16 {
            let op = if i % 2 == 0 { "fadd" } else { "fmul" };
            s.push_str(&format!("  %x{} = {op} {ty} %x{i}, %x0\n", i + 1));
        }
        s.push_str(&leave);
        s
    };
    let kernel = |name: &'static str, ty: &str| Kernel {
        name,
        body: chain(ty),
        float: true,
        want: None,
        needs: &[],
        note: None,
    };
    let ks = [
        kernel("chain_float", "float"),
        kernel("chain_half", "half"),
        kernel("chain_bfloat", "bfloat"),
    ];
    let Some(bitcode) = assemble("growth", &module(&ks)) else {
        return;
    };
    for stem in PROFILES {
        let words_on = |policy: DevicePolicy, kk: &Kernel| {
            let p = profile(stem, policy).expect("profile");
            let a = compile(&bitcode, kk.name, &p)
                .unwrap_or_else(|e| panic!("{} on {stem}: {e:?}", kk.name));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{} on {stem}: L3: {e}", kk.name));
            a.module.words.len()
        };
        let words = ks.each_ref().map(|kk| words_on(exact_maths(), kk));
        eprintln!(
            "half and bfloat growth on {stem}: float {} words, half {} ({:+}), bfloat {} ({:+})",
            words[0],
            words[1],
            words[1] as i64 - words[0] as i64,
            words[2],
            words[2] as i64 - words[0] as i64
        );
        // Half: per rounding (the sixteen operations and the narrowing conversion) the device's
        // quantize alone on a profile without `shaderFloat16` (one instruction, four words), else
        // the conversion to 16 bits and its widening, each decorated `NoContraction` under the
        // float controls (at most fourteen words), plus the 16-bit type's and the modes' share.
        assert!(
            words[1] <= words[0] + 17 * 16 + 64,
            "{stem}: half {words:?}"
        );
        // Bfloat: a bounded integer rounding (at most 18 instructions) per operation.
        assert!(
            words[2] <= words[0] + 16 * 18 * 6 + 256,
            "{stem}: bfloat {words:?}"
        );
    }
}

/// A `half` result rounds through `OpQuantizeToF16` alone on a profile without `shaderFloat16`,
/// with no denormal arithmetic (no `RoundEven`) after any operation, and through the conversion
/// to 16 bits on one with it; the operation decides the rest, never a name. A profile with
/// `shaderFloat16` whose float controls do not make the conversion Metal's (moltenvk-m3) rounds
/// through the conversion of `x` itself and its widening, declaring no 16-bit float mode or
/// `RoundingModeRTE`. A profile whose float controls make that conversion Metal's
/// (amd-rdna3-radv) rounds through the conversion of `x` itself and its widening, each decorated
/// `NoContraction`, in an entry declaring `RoundingModeRTE`, `DenormPreserve` and
/// `SignedZeroInfNanPreserve` for 16 bits and the modes its independence ties to them
/// (`narrow::controlled_modes`). No other module declares a 16-bit float mode or
/// `RoundingModeRTE` for it. A quantize records `NativeHalfQuantize` on every profile.
///
/// The chain: one `fptrunc`, eight `fadd`, and eight `fmul` on `half`, strict and `fast`. Nothing
/// is fused: the `fast` chain's operations are the strict chain's, each in the AIR's order, so
/// both keep seventeen roundings. On a profile whose driver's 16-bit `+ - *`, kept apart, is
/// Metal's (moltenvk-m3), each `fadd` and `fmul` that rounds is that instruction, which reads the
/// previous one's 16-bit result itself and narrows `%x0`, and rounds nothing itself: the
/// conversion's rounding alone is left to the module, and the last result is widened once.
#[test]
fn half_rounding_follows_the_operation() {
    let chain = |fast: bool| -> String {
        let flag = if fast { "fast " } else { "" };
        let mut s = format!(
            "{}  %x0 = fptrunc float %a to half\n",
            of("a", 1.5).replace("fadd float", &format!("fadd {flag}float"))
        );
        for i in 0..16 {
            let op = if i % 2 == 0 { "fadd" } else { "fmul" };
            s.push_str(&format!("  %x{} = {op} {flag}half %x{i}, %x0\n", i + 1));
        }
        s.push_str("  %r = fpext half %x16 to float\n");
        s
    };
    let kernel = |name: &'static str, fast: bool| Kernel {
        name,
        body: chain(fast),
        float: true,
        want: None,
        needs: &[],
        note: None,
    };
    let ks = [kernel("strict_chain", false), kernel("fast_chain", true)];
    let Some(bitcode) = assemble("half-rounding", &module(&ks)) else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let quantize = count_ops(&words, |w| {
                w[0] & 0xFFFF == crate::spv::Op::QuantizeToF16 as u32
            });
            let fconvert = count_ops(&words, |w| w[0] & 0xFFFF == crate::spv::Op::FConvert as u32);
            // `GLSL.std.450 RoundEven` is instruction 2 of the set.
            let round_even = count_ops(&words, |w| {
                w[0] & 0xFFFF == crate::spv::Op::ExtInst as u32 && w.len() > 4 && w[4] == 2
            });
            let (q, c) = (17, 0);
            // With `shaderFloat16` each rounding is the device's conversion and its widening, with
            // no quantize and no denormal arithmetic (moltenvk-m3).
            let form = crate::lower::arith::narrow::quantize_form(&p);
            let converted = form == crate::spirv::builder::QuantizeForm::Converted;
            // Where the device's float controls make the conversion Metal's, each rounding is
            // that conversion and its widening too.
            let controlled = form == crate::spirv::builder::QuantizeForm::Controlled;
            // Where the device's 16-bit `+ - *` is Metal's, measured or under the 16-bit float
            // controls, the operations among the roundings: sixteen of each chain's seventeen. The
            // conversion's rounding stays the module's.
            let ha = (p.takes_driver_deviation(crate::error::DriverDeviation::MetalHalfArithmetic)
                && p.fields().leaf.float16)
                || controlled;
            assert!(
                !ha || converted || controlled,
                "{what}: the half arithmetic beside the conversion"
            );
            let kept = if ha { 16 } else { 0 };
            // Each 16-bit operation reads the previous one's 16-bit result itself, and narrows
            // `%x0`, the conversion's rounding, which stays a pair (the first operation reads it
            // twice); the last result is widened once.
            let operations = if kept > 0 { kept + 2 } else { 0 };
            // Without `shaderFloat16` each rounding is the device's quantize alone.
            let want = match converted || controlled {
                true => (0, c + 2 * (q - kept) + operations, 0),
                false => (q, c, 0),
            };
            assert_eq!(
                (quantize, fconvert, round_even),
                want,
                "{what}: OpQuantizeToF16, OpFConvert, RoundEven"
            );
            // Each rounding through the conversion converts the value itself, with no `x + -0.0`
            // decorated `NoContraction` between: only the strict chain's own
            // `%a = fadd float 1.5, -0.0`, decorated `NoContraction` without fast math, reads as
            // one where the `fptrunc` narrows it.
            let own_barrier = usize::from(kk.name == "strict_chain");
            assert_eq!(
                barred_roundings(&words),
                if converted || controlled {
                    (q - kept, q - kept - own_barrier)
                } else {
                    (0, 0)
                },
                "{what}: conversion roundings, and those of anything but an added -0.0"
            );
            let half_type = count_ops(&words, |w| {
                w[0] & 0xFFFF == crate::spv::Op::TypeFloat as u32 && w.len() > 2 && w[2] == 16
            });
            assert_eq!(
                half_type,
                usize::from(converted || controlled),
                "{what}: a 16-bit float type"
            );
            // The controlled form's modes: `RoundingModeRTE` at 16 bits and the widths tied to it.
            let rte: Vec<u32> = crate::lower::arith::narrow::controlled_modes(&p)
                .filter(|_| controlled)
                .unwrap_or_default()
                .iter()
                .filter_map(|m| match *m {
                    crate::spirv::ir::ExecMode::RoundingModeRte { width } => Some(width.into()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                (
                    declares_signed_zero_inf_nan(&words, 16),
                    rounds_even_at(&words)
                ),
                (controlled, rte),
                "{what}: a 16-bit float mode and RoundingModeRTE"
            );
        }
    }
}

/// Where the device's 16-bit `+ - *` is measured Metal's, a `half` sum is that instruction, which
/// rounds nothing of the module's: it requires `MetalHalfArithmetic`. A kernel there whose
/// `half`s (each read from its bits by a bitcast) only the device's 16-bit `fma` reads takes them
/// as 16-bit floats, with no widening, and requires nothing of that measurement.
#[test]
fn a_measured_half_sum_requires_the_arithmetic_fact_alone() {
    let halves = format!(
        "{}{}{}  %h = bitcast i16 %t to half\n  %g = bitcast i16 %s to half\n  %u = bitcast i16 %v to half\n",
        on("t", 16, 0x3C01),
        on("s", 16, 0x4100),
        on("v", 16, 0x3800)
    );
    let ks = [
        Kernel {
            name: "half_sum",
            body: format!("{halves}  %x = fadd half %h, %g\n  %r = fpext half %x to float\n"),
            float: true,
            want: None,
            needs: &[],
            note: None,
        },
        Kernel {
            name: "half_fma",
            body: format!(
                "{halves}  %x = call half @llvm.fma.f16(half %h, half %g, half %u)\n  %r = fpext half %x to float\n"
            ),
            float: true,
            want: None,
            needs: &[],
            note: None,
        },
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!("{DECLS}declare half @llvm.fma.f16(half, half, half)\n"),
    );
    let Some(bitcode) = assemble("half-conversion-fact", &text) else {
        return;
    };
    let mut converting = 0;
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        let converted = crate::lower::arith::narrow::quantize_form(&p)
            == crate::spirv::builder::QuantizeForm::Converted;
        let ha = p.takes_driver_deviation(crate::error::DriverDeviation::MetalHalfArithmetic)
            && p.fields().leaf.float16;
        converting += usize::from(converted);
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_eq!(
                a.requirements
                    .properties
                    .contains(&crate::device::requirements::VkProperty::MetalHalfArithmetic),
                ha && kk.name == "half_sum",
                "{what}: {:?}",
                a.requirements.properties
            );
        }
    }
    assert!(converting > 0, "no profile rounds through the conversion");
}

/// Without `shaderFloat16` every `half` rounding is the device's `OpQuantizeToF16` alone, with no
/// denormal arithmetic (no `RoundEven`) whatever the operation: a transcendental, `ldexp`, a
/// quotient, a square root, `fract`'s difference and the one rounding of a fused sum (`fmuladd`
/// and `mix`, the device's `Fma` and `FMix` in `float`), whose `f32` result below `2^-14` may be
/// off the `2^-24` grid (half `exp(-10)` is about `4.54e-5`) and is the device's there
/// (`NativeHalfQuantize`); an integer converted to `half` rounds by integer arithmetic with
/// neither. On a profile whose driver computes a 16-bit `Fma` as Metal's `half` `fma`
/// (moltenvk-m3), a fused sum rounds nothing itself: the device's `Fma` rounds it. On a profile
/// with `shaderFloat16` (moltenvk-m3, amd-rdna3-radv) no rounding holds either: each is the
/// conversion and its widening.
#[test]
fn a_half_rounding_without_float16_is_the_quantize_alone() {
    let halves = format!(
        "{}{}{}  %h = bitcast i16 %t to half\n  %g = bitcast i16 %s to half\n  %u = bitcast i16 %v to half\n",
        on("t", 16, 0x3C01),
        on("s", 16, 0x4100),
        on("v", 16, 0x3800)
    );
    // (name, the operation defining `%x`, `OpQuantizeToF16` count)
    let cases: [(&'static str, &str, usize); 9] = [
        ("half_exp", "call half @air.exp.f16(half %h)", 1),
        ("half_log", "call half @air.log.f16(half %h)", 1),
        (
            "half_ldexp",
            "call half @air.ldexp.f16(half %h, i32 -20)",
            1,
        ),
        ("half_fdiv", "fdiv half %h, %g", 1),
        ("half_sqrt", "call half @air.sqrt.f16(half %h)", 1),
        ("half_sitofp", "sitofp i32 %i to half", 0),
        ("half_fract", "call half @air.fract.f16(half %h)", 1),
        (
            "half_mix",
            "call half @air.mix.f16(half %h, half %g, half %u)",
            1,
        ),
        (
            "half_fmuladd",
            "call half @llvm.fmuladd.f16(half %h, half %g, half %u)",
            1,
        ),
    ];
    let ks: Vec<Kernel> = cases
        .iter()
        .map(|&(name, op, _)| Kernel {
            name,
            body: format!(
                "{halves}{}  %x = {op}\n  %r = fpext half %x to float\n",
                oi("i", 0x4040_0000)
            ),
            float: true,
            want: None,
            needs: &[],
            note: None,
        })
        .collect();
    let text = module(&ks).replace(
        DECLS,
        &format!(
            "{DECLS}declare half @air.exp.f16(half)\ndeclare half @air.log.f16(half)\ndeclare half @air.ldexp.f16(half, i32)\ndeclare half @air.sqrt.f16(half)\ndeclare half @air.fract.f16(half)\ndeclare half @air.mix.f16(half, half, half)\ndeclare half @llvm.fmuladd.f16(half, half, half)\n"
        ),
    );
    let Some(bitcode) = assemble("half-exact-below-normal", &text) else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for (kk, &(_, _, quantizes)) in ks.iter().zip(&cases) {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let quantize = count_ops(&words, |w| {
                w[0] & 0xFFFF == crate::spv::Op::QuantizeToF16 as u32
            });
            // `GLSL.std.450 RoundEven` is instruction 2 of the set.
            let round_even = count_ops(&words, |w| {
                w[0] & 0xFFFF == crate::spv::Op::ExtInst as u32 && w.len() > 4 && w[4] == 2
            });
            let fused = matches!(kk.name, "half_mix" | "half_fmuladd")
                && p.takes_driver_deviation(crate::error::DriverDeviation::MetalHalfFma);
            // A rounding through the conversion holds neither, under the float controls or not.
            let converts = crate::lower::arith::narrow::quantize_form(&p)
                != crate::spirv::builder::QuantizeForm::Device;
            let want = match fused || converts {
                true => (0, 0),
                false => (quantizes, 0),
            };
            assert_eq!(
                (quantize, round_even),
                want,
                "{what}: OpQuantizeToF16, RoundEven"
            );
        }
    }
}

/// The widths at which `words` (a module) declares `OpExecutionMode <entry> RoundingModeRTE`, in
/// declaration order.
fn rounds_even_at(words: &[u32]) -> Vec<u32> {
    let mut widths = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if opcode == crate::spv::Op::ExecutionMode as u32
            && len == 4
            && words[i + 2] == crate::spv::ExecutionMode::RoundingModeRTE as u32
        {
            widths.push(words[i + 3]);
        }
        i += len;
    }
    widths
}

/// The instructions of `words` (a module) for which `pick` holds of their words.
fn count_ops(words: &[u32], pick: impl Fn(&[u32]) -> bool) -> usize {
    let mut i = 5;
    let mut n = 0;
    while i < words.len() {
        let len = (words[i] >> 16) as usize;
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        n += usize::from(pick(&words[i..i + len]));
        i += len;
    }
    n
}

/// The `half` roundings through the conversion in `words` (an `OpFConvert` of an `f32` to 16 bits
/// whose result another `OpFConvert` widens back, `narrow::converted`), and how many of them
/// convert anything but an `OpFAdd` decorated `NoContraction` whose second operand is `-0.0` (a
/// constant, or a composite of them).
fn barred_roundings(words: &[u32]) -> (usize, usize) {
    use std::collections::{BTreeMap, BTreeSet};
    let mut widths = BTreeMap::new(); // float or vector type -> its lanes' width
    let mut minus_zero = BTreeSet::new(); // constants that are -0.0 in every lane
    let mut no_contraction = BTreeSet::new();
    let mut fadd = BTreeMap::new(); // result -> second operand
    let mut fconvert = Vec::new(); // (result type, result, operand)
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        let w = &words[i..i + len];
        match opcode {
            o if o == crate::spv::Op::TypeFloat as u32 => {
                widths.insert(w[1], w[2]);
            }
            o if o == crate::spv::Op::TypeVector as u32 => {
                if let Some(&width) = widths.get(&w[2]) {
                    widths.insert(w[1], width);
                }
            }
            o if o == crate::spv::Op::Constant as u32 && len == 4 && w[3] == 0x8000_0000 => {
                minus_zero.insert(w[2]);
            }
            o if o == crate::spv::Op::ConstantComposite as u32
                && w[3..].iter().all(|c| minus_zero.contains(c)) =>
            {
                minus_zero.insert(w[2]);
            }
            o if o == crate::spv::Op::Decorate as u32
                && w[2] == crate::spv::Decoration::NoContraction as u32 =>
            {
                no_contraction.insert(w[1]);
            }
            o if o == crate::spv::Op::FAdd as u32 => {
                fadd.insert(w[2], w[4]);
            }
            o if o == crate::spv::Op::FConvert as u32 => fconvert.push((w[1], w[2], w[3])),
            _ => {}
        }
        i += len;
    }
    let narrowed: BTreeMap<u32, u32> = fconvert
        .iter()
        .filter(|(ty, _, _)| widths.get(ty) == Some(&16))
        .map(|&(_, r, operand)| (r, operand))
        .collect();
    let roundings: Vec<u32> = fconvert
        .iter()
        .filter(|(ty, _, operand)| widths.get(ty) == Some(&32) && narrowed.contains_key(operand))
        .filter_map(|(_, _, n)| narrowed.get(n).copied())
        .collect();
    let unbarred = roundings
        .iter()
        .filter(|x| {
            !(no_contraction.contains(*x) && fadd.get(*x).is_some_and(|b| minus_zero.contains(b)))
        })
        .count();
    (roundings.len(), unbarred)
}

/// Whether `words` declares `OpExecutionMode <entry> SignedZeroInfNanPreserve <width>`.
pub(super) fn declares_signed_zero_inf_nan(words: &[u32], width: u32) -> bool {
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if opcode == crate::spv::Op::ExecutionMode as u32
            && len == 4
            && words[i + 2] == crate::spv::ExecutionMode::SignedZeroInfNanPreserve as u32
            && words[i + 3] == width
        {
            return true;
        }
        i += len;
    }
    false
}

/// A float operation without `nnan`/`ninf`/`nsz` makes the entry declare `SignedZeroInfNanPreserve
/// 32` where the profile supports it, and note `DeviceSignedZeroInfNan` where it does not; `fast`
/// operations need neither, but for a `fast` product's check at `±2^-126` where the profile checks
/// one, which answers Metal's zero of its sign. Half arithmetic is held in `f32`, so it needs the
/// 32-bit mode.
#[test]
fn float_ops_without_fast_flags_preserve_signed_zero_inf_nan() {
    let inputs = format!("{}{}", of("a", -0.0), of("b", 0.0));
    let kernel = |name: &'static str, body: String| Kernel {
        name,
        body: format!("{inputs}{body}"),
        float: true,
        want: None,
        needs: &[],
        note: None,
    };
    let ks = [
        kernel("strict_mul", "  %r = fmul float %a, %b\n".to_owned()),
        Kernel {
            // The opaque inputs are `fast` too, so nothing in the entry needs preservation.
            body: format!("{inputs}  %r = fmul fast float %a, %b\n")
                .replace("fadd float", "fadd fast float"),
            ..kernel("fast_mul", String::new())
        },
        kernel(
            "strict_half",
            "  %h = fptrunc float %a to half\n  %s = fmul half %h, %h\n  %r = fpext half %s to float\n"
                .to_owned(),
        ),
    ];
    let Some(bitcode) = assemble("fpenv", &module(&ks)) else {
        return;
    };
    let (mut declared, mut noted) = (0, 0);
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        let supported = p.fields().float_controls.f32.signed_zero_inf_nan_preserve;
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let note = a.precision.contains(&PrecisionNote::DeviceSignedZeroInfNan);
            let mode = declares_signed_zero_inf_nan(&words, 32);
            if kk.name == "fast_mul" {
                assert_eq!(
                    (mode, note),
                    (false, false),
                    "{what}: fast needs no preservation"
                );
            } else {
                assert_eq!((mode, note), (supported, !supported), "{what}");
                declared += usize::from(mode);
                noted += usize::from(note);
            }
        }
    }
    assert!(
        declared > 0 && noted > 0,
        "{declared} declared, {noted} noted"
    );
}

/// A `fast` entry's fused multiply-add is the device's instruction on every profile (`OpFmaKHR`,
/// the device's `Fma`, or a `half` one's 16-bit form), so no kernel declares
/// `SignedZeroInfNanPreserve`, at 32 bits or 64: their operands are bits, their calls `fast`.
#[test]
fn fused_multiply_adds_keep_no_zero_signs() {
    use crate::spirv::testing::committed_profiles_with;
    let ks = [
        Kernel {
            name: "fast_fma_float",
            body: format!(
                "{}{}{}  %r = call fast float @llvm.fma.f32(float %a, float %b, float %c)\n",
                ob("a", 0),
                ob("b", 100f32.to_bits()),
                ob("c", 0)
            ),
            float: true,
            want: None,
            needs: &[],
            note: None,
        },
        Kernel {
            name: "fast_fma_half",
            body: format!(
                "{}{}{}  %x = call fast half @llvm.fma.f16(half %h, half %g, half %u)\n  %xb = bitcast half %x to i16\n  %r = zext i16 %xb to i32\n",
                oh("h", 0),
                oh("g", 0x5640),
                oh("u", 0)
            ),
            float: false,
            want: None,
            needs: &[],
            note: None,
        },
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!(
            "{DECLS}declare float @llvm.fma.f32(float, float, float)\ndeclare half @llvm.fma.f16(half, half, half)\n"
        ),
    );
    let Some(bitcode) = assemble("fma-zero-signs", &text) else {
        return;
    };
    for (stem, p) in committed_profiles_with(DevicePolicy::default()).expect("profiles") {
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            for width in [32, 64] {
                assert!(
                    !declares_signed_zero_inf_nan(&words, width),
                    "{what}: mode {width}"
                );
            }
        }
    }
}

/// A float to integer `air.convert` converts NaN to 0, telling it by `OpIsNan` only where the entry
/// declares `SignedZeroInfNanPreserve` for `float`, and by its bits where it does not (a device may
/// assume there that no operand is a NaN, in `OpIsNan` and `NClamp` alike): a `fast` entry has no
/// `OpIsNan`. The NaN is a sum's. (AIR has no `double` conversion; `math::convert`'s unit tests
/// read a 64-bit container's NaN by its two words.)
#[test]
fn a_convert_tells_a_nan_by_its_bits_where_the_entry_may_assume_none() {
    let float = |name: &'static str, flags: &str| {
        k(
            name,
            format!(
                "{}  %f = bitcast i32 %a to float\n  %g = fadd {flags}float %f, %f\n  %i = call i32 @air.convert.s.i32.f.f32(float %g)\n  %u = call i8 @air.convert.u.i8.f.f32(float %g)\n  %uw = zext i8 %u to i32\n  %s = or i32 %i, %uw\n  %r = add i32 %s, 7\n",
                oi("a", 0x7FC0_0001)
            ),
            7,
        )
    };
    let ks = [
        float("strict_nan_convert", ""),
        float("fast_nan_convert", "fast "),
    ];
    let Some(bitcode) = assemble("nanconvert", &module(&ks)) else {
        return;
    };
    let mut bits = 0;
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let mode = declares_signed_zero_inf_nan(&words, 32);
            let is_nan = count_op(&words, crate::spv::Op::IsNan);
            assert_eq!(is_nan > 0, mode, "{what}: {is_nan} OpIsNan, mode {mode}");
            if kk.name != "strict_nan_convert" {
                assert!(!mode, "{what}: nothing needs preservation");
            }
            bits += usize::from(!mode);
            let got = interpret(&bitcode, kk.name, &p);
            assert_eq!(got, 7, "{what}: {got:#x}");
        }
    }
    assert!(bits >= PROFILES.len(), "{bits} by bits");
}

/// The profile of committed profile `stem` with its driver's `MetalFloatFunctions` deviation
/// taken or not as `metal` says, and nothing else changed.
fn with_float_functions(stem: &str, metal: bool) -> DeviceProfile {
    use crate::device::{Device, DeviceDescription};
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let deviation = crate::error::DriverDeviation::MetalFloatFunctions;
    Device::with_deviation(desc, exact_maths(), deviation, metal)
        .expect("device")
        .profile()
        .clone()
}

/// An `arcp` division is the device's `OpFDiv(x, y)` itself on every profile, which MoltenVK hands
/// Metal's compiler as the AIR's own fast `x / y` (Metal's compiler does not treat `x * (1.0 / y)`
/// the same way: case 24c263bb's `(a * 255) / a` was 255 on Metal and 254 as `x` times the
/// device's `OpFDiv(1, y)`). Where the device's `float` division is Metal's (its driver's
/// deviation `MetalFloatFunctions`) and Metal compiles the entry with fast math (it declares no
/// `SignedZeroInfNanPreserve`), the quotient is Metal's own and records no note; elsewhere it
/// records `DeviceArithmetic`. The interpreter's `OpFDiv` is correctly rounded: `RN(1e10 / 3)` is
/// `0x4f46aea1`.
#[test]
fn a_fast_division_is_the_device_s_quotient() {
    // The fast-math entries' opaque inputs are `fast` too, so nothing there needs preservation.
    let fast_inputs = |s: String| s.replace("fadd float", "fadd fast float");
    let (x, y) = (of("x", 1e10), of("y", 3.0));
    let vector = "  %vx = insertelement <4 x float> undef, float %x, i32 0\n  \
                  %vy = insertelement <4 x float> undef, float %y, i32 0\n  \
                  %vq = fdiv fast <4 x float> %vx, %vy\n  \
                  %r = extractelement <4 x float> %vq, i32 0\n";
    let scalar = "  %r = fdiv fast float %x, %y\n";
    let kernel = |name: &'static str, body: String| Kernel {
        name,
        body,
        float: true,
        want: None,
        needs: &[],
        note: None,
    };
    let ks = [
        kernel(
            "fast_quotient_fast_math",
            fast_inputs(format!("{x}{y}{scalar}")),
        ),
        kernel(
            "fast_quotient_fast_math_v4",
            fast_inputs(format!("{x}{y}{vector}")),
        ),
        kernel("fast_quotient_preserving", format!("{x}{y}{scalar}")),
        kernel("fast_quotient_preserving_v4", format!("{x}{y}{vector}")),
    ];
    // Whether each kernel's entry is compiled with fast math.
    let fast_math = [true, true, false, false];
    let Some(bitcode) = assemble("fast-quotient", &module(&ks)) else {
        return;
    };
    for (kk, fast_math) in ks.iter().zip(fast_math) {
        for metal in [true, false] {
            let p = with_float_functions("moltenvk-m3", metal);
            let what = format!("{} with the fact {metal}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_eq!(
                declares_signed_zero_inf_nan(&words, 32),
                !fast_math,
                "{what}: SignedZeroInfNanPreserve"
            );
            // Metal's own quotient in a fast entry records no note; the device's elsewhere does.
            let device = metal && fast_math;
            assert_eq!(
                a.precision.contains(&PrecisionNote::DeviceArithmetic),
                !device,
                "{what}: {:?}",
                a.precision
            );
            // The entry's one `OpFDiv`, and the module defines no routine.
            let form = (
                count_op(&words, crate::spv::Op::Function),
                count_op(&words, crate::spv::Op::FunctionCall),
                count_op(&words, crate::spv::Op::FDiv),
            );
            assert_eq!(form, (1, 0, 1), "{what}");
            let got = interpret(&bitcode, kk.name, &p);
            assert_eq!(got, 0x4F46_AEA1, "{what}: {got:#x}");
        }
    }
}

/// `%x` = an opaque `float` with bits `b` (any pattern, NaNs included): `llvm.umin` of a constant
/// and `-1`, bitcast.
fn ob(name: &str, b: u32) -> String {
    format!(
        "  %{name}.w = call i32 @llvm.umin.i32(i32 {}, i32 -1)\n  %{name} = bitcast i32 %{name}.w to float\n",
        b as i32
    )
}

/// Committed profile `stem` with its driver's `MetalCanonicalNans` deviation taken or not as
/// `fact` says.
fn profile_with_canonical_nans(stem: &str, fact: bool) -> DeviceProfile {
    profile_with_fact(
        stem,
        crate::error::DriverDeviation::MetalCanonicalNans,
        fact,
    )
}

/// Committed profile `stem` with its driver's deviation `deviation` taken or not as `fact` says.
fn profile_with_fact(
    stem: &str,
    deviation: crate::error::DriverDeviation,
    fact: bool,
) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::with_deviation(desc, exact_maths(), deviation, fact)
        .expect("device")
        .profile()
        .clone()
}

/// The bits tests' `OpUGreaterThan` in the entry point's own function of module `words` (one
/// entry point): a module-scope routine tests by the bits in entries of every float mode, and
/// compares integers so (a precise division's).
fn entry_bits_tests(words: &[u32]) -> usize {
    let (mut entry, mut inside, mut n) = (None, false, 0);
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if opcode == crate::spv::Op::EntryPoint as u32 {
            entry = Some(words[i + 2]);
        } else if opcode == crate::spv::Op::Function as u32 {
            inside = Some(words[i + 2]) == entry;
        } else if opcode == crate::spv::Op::FunctionEnd as u32 {
            inside = false;
        } else if inside && opcode == crate::spv::Op::UGreaterThan as u32 {
            n += 1;
        }
        i += len;
    }
    n
}

/// The NaN tests in module `words`: `OpIsNan` and the bits test's `OpUGreaterThan`.
fn nan_tests(words: &[u32]) -> (usize, usize) {
    (
        count_op(words, crate::spv::Op::IsNan),
        count_op(words, crate::spv::Op::UGreaterThan),
    )
}

/// Whether module `words` decorates any value `FPFastMathMode`.
fn decorates_fast_math(words: &[u32]) -> bool {
    count_ops(words, |inst| {
        inst[0] & 0xFFFF == crate::spv::Op::Decorate as u32
            && inst.get(2) == Some(&(crate::spv::Decoration::FPFastMathMode as u32))
    }) > 0
}

/// The kernels of the NaN-model tests: their declarations beside [`DECLS`].
const NAN_DECLS: &str = "declare float @air.fast_round.f32(float)\ndeclare float @air.fast_asin.f32(float)\ndeclare float @air.fast_tan.f32(float)\ndeclare float @air.asin.f32(float)\ndeclare float @air.sign.f32(float)\ndeclare float @air.fmedian3.f32(float, float, float)\n";

/// A kernel storing `%s`'s bits, `float` `%s` defined by `body`.
fn nan_kernel(name: &'static str, body: String, want: Option<u32>) -> Kernel {
    Kernel {
        want,
        ..k(name, format!("{body}  %r = bitcast float %s to i32\n"), 0)
    }
}

/// Whether `bits` are a NaN of the stored format: a `half`'s, zero-extended, where `half`.
fn stores_nan(bits: u32, half: bool) -> bool {
    match half {
        true => bits <= 0xFFFF && bits & 0x7C00 == 0x7C00 && bits & 0x3FF != 0,
        false => f32::from_bits(bits).is_nan(),
    }
}

/// Kernels `ks` of `bitcode` (each storing a NaN's bits, `half` ones zero-extended where `half`)
/// on `lavapipe` and `moltenvk-m3`, each with its driver's deviation `MetalCanonicalNans` taken
/// and not: the module selects no NaN, so it is the same word for word on both sides of the fact;
/// it records `PrecisionNote::NativeNanEncoding` exactly where `noted(kernel, fact)` or it reads a
/// `half` from its bits by the device's widening or encodes one by its narrowing
/// (`narrow::widens_bits`, `narrow::encodes_bits`: both profiles have `shaderFloat16`), and never
/// `DeviceNanEncoding`; and the interpreter's answer is the kernel's
/// `want`, or a NaN, its bits the device's, where it has none, but for the kernels `unread` names,
/// whose device instruction the interpreter does not evaluate.
fn nans_are_the_device_s(
    bitcode: &[u8],
    ks: &[Kernel],
    half: bool,
    unread: &[&str],
    noted: &dyn Fn(&str, bool) -> bool,
) {
    for stem in ["lavapipe", "moltenvk-m3"] {
        for kk in ks {
            let mut emitted = Vec::new();
            for fact in [false, true] {
                let p = profile_with_canonical_nans(stem, fact);
                let what = format!("{} on {stem}, MetalCanonicalNans {fact}", kk.name);
                let a = compile(bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let (_, words) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                let (_, m) = crate::pipeline::testing::typed_module(bitcode, kk.name, &p)
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let widens = crate::lower::arith::narrow::widens_bits(&m, &mut meter())
                    .and_then(|w| {
                        Ok(w || crate::lower::arith::narrow::encodes_bits(&m, &mut meter())?)
                    })
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                assert_eq!(
                    a.precision.contains(&PrecisionNote::NativeNanEncoding),
                    noted(kk.name, fact) || widens,
                    "{what}: {:?}",
                    a.precision
                );
                assert!(
                    !a.precision.contains(&PrecisionNote::DeviceNanEncoding),
                    "{what}: {:?}",
                    a.precision
                );
                if !unread.contains(&kk.name) {
                    let got = interpret(bitcode, kk.name, &p);
                    match kk.want {
                        Some(want) => assert_eq!(got, want, "{what}: {got:#x}"),
                        None => assert!(stores_nan(got, half), "{what}: {got:#x}"),
                    }
                }
                emitted.push(words);
            }
            assert!(
                emitted[0] == emitted[1],
                "{} on {stem}: the module changes with the fact",
                kk.name
            );
        }
    }
}

/// A NaN of `float` arithmetic is the device's on every profile: the module selects none, whatever
/// the profile's `MetalCanonicalNans` fact, and records `PrecisionNote::NativeNanEncoding` where
/// the device is not measured to answer Metal's: for `+`, whose canonical NaN the fact covers,
/// where the fact is absent, and for `round`, which returns a NaN operand's own bits on Metal,
/// whatever the fact. The interpreter keeps a NaN operand's payload, as such a device would.
#[test]
fn a_float_nan_is_the_device_s_on_every_profile() {
    let ks = [
        nan_kernel(
            "nan_fadd",
            format!("{}  %s = fadd float %x, 1.0\n", ob("x", 0xFFC0_1234)),
            None,
        ),
        nan_kernel(
            "nan_round",
            format!(
                "{}  %s = call float @air.round.f32(float %x)\n",
                ob("x", 0x7F80_0001)
            ),
            None,
        ),
    ];
    let Some(bitcode) = assemble("nan-fact", &module(&ks)) else {
        return;
    };
    nans_are_the_device_s(&bitcode, &ks, false, &[], &|name, fact| {
        name == "nan_round" || !fact
    });
    // The committed profiles state the fact for the measured device only.
    for (stem, fact) in [("moltenvk-m3", true), ("lavapipe", false)] {
        let p = profile(stem, exact_maths()).expect("profile");
        assert_eq!(
            p.fingerprint(),
            profile_with_canonical_nans(stem, fact).fingerprint(),
            "{stem}"
        );
    }
}

// ---- Metal's float denormals ----

/// Metal reads an `f32` denormal operand of `+ - *`, a comparison, `sqrt`, `floor`, `ceil` and
/// `ldexp` as the zero of its sign and flushes a denormal result. The module leaves the reading to
/// the device: where the profile takes its driver's measured deviation `MetalFloatDenormals` it
/// requires the fact for `+ - *`, the comparison, `sqrt` and `ldexp` (whose denormal operand and
/// result are the device's products, `math::ldexp_bits`), which then read them as Metal does;
/// elsewhere they are the device's, noted `DeviceDenormals` (`sqrt`, the device's maths function,
/// off Metal's own functions notes them beside its precision on either). `floor` and `ceil` are
/// the device's
/// whatever the fact, noted: no device measured reads their operand as Metal does. Flipping the
/// fact changes the requirement and the note, never the module. The interpreter keeps denormals as
/// lavapipe does. The operands are bits (`llvm.umin` of constants), so each kernel's float
/// operation is its only one.
#[test]
fn float_denormals_are_the_devices_and_noted_off_the_device_fact() {
    // The kernel's float `%r` from `body`, Metal's word, and the word of a device that keeps
    // denormals (the interpreter, where the module leaves them to the device).
    let float = |name: &'static str, body: String, metal: u32| Kernel {
        name,
        body,
        float: true,
        want: Some(metal),
        needs: &[],
        note: None,
    };
    let den = f32::from_bits(0x0040_0000);
    let (ks, devices): (Vec<Kernel>, Vec<u32>) = [
        (
            float(
                "den_fadd",
                format!(
                    "{}{}  %r = fadd float %x, %y\n",
                    ob("x", 0x0040_0000),
                    ob("y", 0x0040_0000)
                ),
                0,
            ),
            0x0080_0000,
        ),
        (
            float(
                "den_fmul_tiny",
                format!(
                    "{}{}  %r = fmul float %x, %y\n",
                    ob("x", 0x1F80_0000),
                    ob("y", 0x1F80_0000)
                ),
                0,
            ),
            0x0020_0000,
        ),
        (
            float(
                "den_fmul_tiny_negative",
                format!(
                    "{}{}  %r = fmul float %x, %y\n",
                    ob("x", 0x9F80_0000),
                    ob("y", 0x1F80_0000)
                ),
                0x8000_0000,
            ),
            0x8020_0000,
        ),
        (
            k(
                "den_fcmp",
                format!(
                    "{}{}  %c = fcmp ogt float %x, %y\n  %r = zext i1 %c to i32\n",
                    ob("x", 2),
                    ob("y", 1)
                ),
                0,
            ),
            1,
        ),
        (
            float(
                "den_floor",
                format!(
                    "{}  %r = call float @air.floor.f32(float %x)\n",
                    ob("x", 0x8000_0001)
                ),
                0x8000_0000,
            ),
            (-1.0f32).to_bits(),
        ),
        (
            float(
                "den_ceil",
                format!("{}  %r = call float @air.ceil.f32(float %x)\n", ob("x", 1)),
                0,
            ),
            1.0f32.to_bits(),
        ),
        (
            float(
                "den_sqrt",
                format!(
                    "{}  %r = call float @air.sqrt.f32(float %x)\n",
                    ob("x", 0x0040_0000)
                ),
                0,
            ),
            den.sqrt().to_bits(),
        ),
        (
            float(
                "den_ldexp",
                format!(
                    "{}  %r = call float @air.ldexp.f32(float %x, i32 1)\n",
                    ob("x", 0x0040_0000)
                ),
                0,
            ),
            0x0080_0000,
        ),
        (
            float(
                "den_ldexp_tiny",
                format!(
                    "{}  %r = call float @air.ldexp.f32(float %x, i32 -1)\n",
                    ob("x", 0x0080_0000)
                ),
                0,
            ),
            0x0040_0000,
        ),
    ]
    .into_iter()
    .unzip();
    let text = module(&ks).replace(
        DECLS,
        &format!("{DECLS}declare float @air.floor.f32(float)\ndeclare float @air.ceil.f32(float)\ndeclare float @air.sqrt.f32(float)\ndeclare float @air.ldexp.f32(float, i32)\n"),
    );
    let Some(bitcode) = assemble("denormal-fact", &text) else {
        return;
    };
    let deviation = crate::error::DriverDeviation::MetalFloatDenormals;
    for stem in ["moltenvk-m3", "lavapipe"] {
        let mut emitted: Vec<Vec<Vec<u32>>> = Vec::new();
        for fact in [true, false] {
            let p = profile_with_fact(stem, deviation, fact);
            let mut per = Vec::new();
            for (kk, device) in ks.iter().zip(&devices) {
                let what = format!("{} on {stem}, MetalFloatDenormals {fact}", kk.name);
                let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let (_, words) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                let never = matches!(kk.name, "den_floor" | "den_ceil");
                // The device's `Sqrt`, a maths function off Metal's own functions, records its
                // denormals with its precision.
                let function = kk.name == "den_sqrt"
                    && p.fields().float_functions != crate::device::profile::FloatFunctions::Metal;
                assert_eq!(
                    a.requirements
                        .properties
                        .contains(&crate::device::requirements::VkProperty::MetalFloatDenormals),
                    fact && !never,
                    "{what}"
                );
                assert_eq!(
                    a.precision.contains(&PrecisionNote::DeviceDenormals),
                    never || function || !fact,
                    "{what}: {:?}",
                    a.precision
                );
                let got = interpret(&bitcode, kk.name, &p);
                assert_eq!(got, *device, "{what}: {got:#x} != {device:#x}");
                per.push(words);
            }
            emitted.push(per);
        }
        for (i, kk) in ks.iter().enumerate() {
            assert_eq!(
                emitted[0][i], emitted[1][i],
                "{} on {stem}: the module with and without the fact",
                kk.name
            );
        }
    }
    // The committed profiles state the fact for the measured device only.
    for (stem, fact) in [("moltenvk-m3", true), ("lavapipe", false)] {
        let p = profile(stem, exact_maths()).expect("profile");
        assert_eq!(
            p.fingerprint(),
            profile_with_fact(stem, deviation, fact).fingerprint(),
            "{stem}"
        );
    }
}

/// `air.sincos` answers the NaN its `sin` and `cos` answer as calls of their own, since Metal's
/// `sincos` is its `sin` and `cos` word for word (cases 7b283404 and f6524207, every pair): precise
/// `sin` and `cos` without fast-math flags answer the canonical NaN, which the device answers where
/// the profile takes `MetalCanonicalNans`; elsewhere either result, read or not, is noted
/// `PrecisionNote::NativeNanEncoding`, and neither is selected.
#[test]
fn sincos_answers_the_device_s_nan() {
    let kernel = |name: &'static str, read: &str| {
        Kernel {
        name,
        body: format!(
            "{}  %c = alloca float, align 4\n  %t = call float @air.sincos.f32(float %x, ptr %c)\n  %cv = load float, ptr %c, align 4\n  %r = bitcast float {read} to i32\n",
            ob("x", 0xFFC0_1234)
        ),
        float: false,
        want: None,
        needs: &[],
        note: None,
    }
    };
    let ks = [
        kernel("sincos_nan_sine", "%t"),
        kernel("sincos_nan_cosine", "%cv"),
    ];
    let Some(bitcode) = assemble("sincos-nan", &module(&ks)) else {
        return;
    };
    let unread = ["sincos_nan_sine", "sincos_nan_cosine"];
    nans_are_the_device_s(&bitcode, &ks, false, &unread, &|_, fact| !fact);
}

/// A NaN test (Metal's saturating conversion's) is `OpIsNan` only in an entry that declares
/// `SignedZeroInfNanPreserve 32`; where every float operation carries `nnan`/`ninf`/`nsz` the entry
/// declares nothing, a device may fold `OpIsNan`, and the test is the bits instead, which still
/// answers Metal's NaN. No module
/// decorates a value `FPFastMathMode`, so nothing tells a device a tested value is no NaN; nor do
/// the NaN-model probes' fast-math fixtures.
#[test]
fn a_nan_test_no_float_mode_folds() {
    let x = ob("x", 0xFFC0_1234);
    let ks = [
        // A sum's NaN is the device's.
        nan_kernel(
            "strict_fadd",
            format!("{x}  %s = fadd float %x, 1.0\n"),
            None,
        ),
        nan_kernel(
            "fast_fadd",
            format!("{x}  %s = fadd fast float %x, 1.0\n"),
            None,
        ),
        nan_kernel(
            "fast_round",
            format!("{x}  %s = call fast float @air.fast_round.f32(float %x)\n"),
            None,
        ),
        // Metal's saturating conversion answers 0 for a NaN, which the module tests for; the
        // `fabs` before it, with or without flags, decides the entry's mode.
        k(
            "strict_convert",
            format!("{x}  %a = call float @air.fabs.f32(float %x)\n  %r = call i32 @air.convert.s.i32.f.f32(float %a)\n"),
            0,
        ),
        k(
            "fast_convert",
            format!("{x}  %a = call fast float @air.fast_fabs.f32(float %x)\n  %r = call i32 @air.convert.s.i32.f.f32(float %a)\n"),
            0,
        ),
        // The device's `asin` and `tan`, which the interpreter does not evaluate.
        nan_kernel(
            "fast_asin",
            format!("{x}  %s = call fast float @air.fast_asin.f32(float %x)\n"),
            None,
        ),
        nan_kernel(
            "fast_tan",
            format!("{x}  %s = call fast float @air.fast_tan.f32(float %x)\n"),
            None,
        ),
        nan_kernel(
            "strict_asin",
            format!("{x}  %s = call float @air.asin.f32(float %x)\n"),
            None,
        ),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!("{DECLS}{NAN_DECLS}declare float @air.fabs.f32(float)\ndeclare float @air.fast_fabs.f32(float)\n"),
    );
    let Some(bitcode) = assemble("nan-test", &text) else {
        return;
    };
    let (mut bits_tested, mut is_nan_tested) = (0, 0);
    for stem in PROFILES.into_iter().chain(["lavapipe"]) {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert!(!decorates_fast_math(&words), "{what}");
            let (is_nan, bits) = nan_tests(&words);
            let mode = declares_signed_zero_inf_nan(&words, 32);
            assert_eq!(
                mode,
                kk.name.starts_with("strict")
                    && p.fields().float_controls.f32.signed_zero_inf_nan_preserve,
                "{what}"
            );
            if !mode {
                assert_eq!(
                    is_nan, 0,
                    "{what}: OpIsNan without SignedZeroInfNanPreserve"
                );
            } else {
                assert_eq!(
                    entry_bits_tests(&words),
                    0,
                    "{what}: the bits test where OpIsNan is safe"
                );
            }
            bits_tested += usize::from(bits > 0);
            is_nan_tested += usize::from(is_nan > 0);
            if let Some(want) = kk.want {
                let got = interpret(&bitcode, kk.name, &p);
                assert_eq!(got, want, "{what}: {got:#x}");
            }
        }
    }
    assert!(
        bits_tested > 0 && is_nan_tested > 0,
        "{bits_tested} {is_nan_tested}"
    );
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine");
    for entry in ["nan_arith_fast", "nan_functions_fast", "nan_functions"] {
        let bitcode = match llvm_assemble(&dir.join(format!("{entry}.ll"))) {
            Ok(b) => b,
            Err(AssembleError::Missing(t)) => {
                eprintln!("SKIPPED: {t}");
                return;
            }
            Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected {entry}: {e}"),
        };
        for stem in PROFILES.into_iter().chain(["lavapipe"]) {
            let p = profile(stem, exact_maths()).expect("profile");
            let what = format!("{entry} on {stem}");
            let a = compile(&bitcode, entry, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert!(!decorates_fast_math(&words), "{what}");
        }
    }
}

/// `%name` = an opaque `half` with the bits `b`.
fn oh(name: &str, b: u16) -> String {
    format!(
        "  %{name}.w = call i32 @llvm.umin.i32(i32 {b}, i32 -1)\n  %{name}.t = trunc i32 %{name}.w to i16\n  %{name} = bitcast i16 %{name}.t to half\n"
    )
}

/// A kernel storing the bits of `half` `%s`, defined by `body`, zero-extended.
fn half_nan_kernel(name: &'static str, body: String, want: Option<u32>) -> Kernel {
    Kernel {
        want,
        ..k(
            name,
            format!("{body}  %b = bitcast half %s to i16\n  %r = zext i16 %b to i32\n"),
            0,
        )
    }
}

/// Class B: Metal's `half` `+`, `*`, and `air.fma.f16`, with or without fast-math flags, and its
/// narrowing of a `float` to `half` (`fptrunc`, and `air.convert` from class E's version) answer
/// the canonical `half` NaN `0x7E00`, as `half` `-` (class K's) does. The NaN is the device's on
/// every profile, selected nowhere, and noted `PrecisionNote::NativeNanEncoding` where the profile
/// does not take `MetalCanonicalNans`.
#[test]
fn half_nans_answer_the_device_s_nan() {
    let hx = oh("hx", 0xFE01);
    let x = ob("x", 0xFFC0_1234);
    let ks = [
        half_nan_kernel(
            "half_fadd",
            format!("{hx}  %s = fadd half %hx, 0xH3C00\n"),
            None,
        ),
        half_nan_kernel(
            "half_fmul",
            format!("{hx}  %s = fmul half %hx, 0xH4000\n"),
            None,
        ),
        half_nan_kernel(
            "half_fma",
            format!("{hx}  %s = call half @air.fma.f16(half %hx, half 0xH3C00, half 0xH0000)\n"),
            None,
        ),
        half_nan_kernel(
            "narrow",
            format!("{x}  %s = fptrunc float %x to half\n"),
            None,
        ),
        half_nan_kernel(
            "narrow_then_add",
            format!("{x}  %n = fptrunc float %x to half\n  %s = fadd half %n, 0xH3C00\n"),
            None,
        ),
        half_nan_kernel(
            "narrow_convert",
            format!(
                "{x}  %v = insertelement <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, float %x, i32 2\n  %w = call <4 x half> @air.convert.f.v4f16.f.v4f32(<4 x float> %v)\n  %s = extractelement <4 x half> %w, i32 2\n"
            ),
            None,
        ),
        half_nan_kernel(
            "half_fadd_fast",
            format!("{hx}  %s = fadd fast half %hx, 0xH3C00\n"),
            None,
        ),
        half_nan_kernel(
            "half_fsub",
            format!("{hx}  %s = fsub half %hx, 0xH3C00\n"),
            None,
        ),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!("{DECLS}declare half @air.fma.f16(half, half, half)\ndeclare <4 x half> @air.convert.f.v4f16.f.v4f32(<4 x float>)\n"),
    );
    let Some(bitcode) = assemble("nan-half", &text) else {
        return;
    };
    nans_are_the_device_s(&bitcode, &ks, true, &[], &|_, fact| !fact);
}

/// Class C: Metal widens a `half` NaN to the canonical `float` NaN `0x7FC00000`, sign and payload
/// dropped, by `fpext` and by its intrinsic form `air.convert`. A `half` is held widened, so
/// `fpext` emits no device operation: the NaN is the device's widening of the `half`'s bits where
/// the profile has `shaderFloat16` (both profiles here), and the module selects nothing and notes
/// `PrecisionNote::NativeNanEncoding` whatever the device fact states. The interpreter's widening
/// keeps the payload, as a device may.
#[test]
fn a_widened_half_nan_keeps_its_payload_on_every_profile() {
    let hx = oh("hx", 0xFE01);
    let ks = [
        nan_kernel(
            "widen",
            format!("{hx}  %s = fpext half %hx to float\n"),
            None,
        ),
        nan_kernel(
            "widen_vector",
            format!(
                "{hx}  %v = insertelement <2 x half> <half 0xH3C00, half 0xH3C00>, half %hx, i32 1\n  %w = fpext <2 x half> %v to <2 x float>\n  %s = extractelement <2 x float> %w, i32 1\n"
            ),
            None,
        ),
        nan_kernel(
            "widen_convert",
            format!(
                "{hx}  %v = insertelement <4 x half> <half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00>, half %hx, i32 1\n  %w = call <4 x float> @air.convert.f.v4f32.f.v4f16(<4 x half> %v)\n  %s = extractelement <4 x float> %w, i32 1\n"
            ),
            None,
        ),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!("{DECLS}declare <4 x float> @air.convert.f.v4f32.f.v4f16(<4 x half>)\n"),
    );
    let Some(bitcode) = assemble("nan-widen", &text) else {
        return;
    };
    nans_are_the_device_s(&bitcode, &ks, false, &[], &|_, _| true);
}

/// Class D: Metal's `half` `fneg` and `fabs` answer the canonical `half` NaN `0x7E00`, with or
/// without fast-math flags. A device computes both exactly on the held value's sign, so a NaN's
/// payload is kept on every profile: the module selects nothing and notes
/// `PrecisionNote::NativeNanEncoding` whatever the device fact states, as for an `fadd` whose only
/// use is the `fneg`. The interpreter keeps a `half` NaN's payload through the held value, as a
/// device does (the value stored unchanged is `0xFE01`, and notes nothing).
#[test]
fn half_sign_operations_keep_a_nan_s_payload_on_every_profile() {
    let hx = oh("hx", 0xFE01);
    let ks = [
        half_nan_kernel(
            "half_kept",
            format!("{hx}  %s = freeze half %hx\n"),
            Some(0xFE01),
        ),
        half_nan_kernel("half_fneg", format!("{hx}  %s = fneg half %hx\n"), None),
        half_nan_kernel(
            "half_fneg_fast",
            format!("{hx}  %s = fneg fast half %hx\n"),
            None,
        ),
        half_nan_kernel(
            "half_fabs",
            format!("{hx}  %s = call half @air.fabs.f16(half %hx)\n"),
            None,
        ),
        half_nan_kernel(
            "half_fabs_fast",
            format!("{hx}  %s = call fast half @air.fabs.f16(half %hx)\n"),
            None,
        ),
        half_nan_kernel(
            "half_fadd_then_fneg",
            format!("{hx}  %n = fadd half %hx, 0xH3C00\n  %s = fneg half %n\n"),
            None,
        ),
    ];
    let text = module(&ks).replace(DECLS, &format!("{DECLS}declare half @air.fabs.f16(half)\n"));
    let Some(bitcode) = assemble("nan-half-sign", &text) else {
        return;
    };
    nans_are_the_device_s(&bitcode, &ks, true, &[], &|name, _| name != "half_kept");
}

/// Class E: Metal's `float` `fneg` and `fabs` (`air.fabs`, `air.fast_fabs`, and `llvm.fabs`, which
/// takes `air.fabs`'s answer) flip or clear the sign bit alone, with and without fast-math flags: a
/// NaN keeps its payload, signaling or not, and a denormal stays. The lowering is the device's
/// `OpFNegate` and `FAbs` on every profile (RADV's `fneg` keeps a NaN's sign, NVIDIA's `fabs`
/// answers `0x7FFFFFFF`, RADV and ANV flush a denormal): no NaN is selected and nothing moves on
/// the bits, the NaN noted `NativeNanEncoding` and `fneg`'s denormal `DeviceDenormals`. The
/// interpreter keeps the payload and the denormal, as a device may.
#[test]
fn float_sign_operations_are_the_devices_on_every_profile() {
    let snan = ob("x", 0x7F80_0001);
    let neg_snan = ob("x", 0xFF80_0001);
    let ks = [
        nan_kernel(
            "float_kept",
            format!("{neg_snan}  %s = freeze float %x\n"),
            Some(0xFF80_0001),
        ),
        nan_kernel(
            "float_fneg",
            format!("{snan}  %s = fneg float %x\n"),
            Some(0xFF80_0001),
        ),
        nan_kernel(
            "float_fneg_fast",
            format!("{snan}  %s = fneg fast float %x\n"),
            Some(0xFF80_0001),
        ),
        nan_kernel(
            "float_fabs",
            format!("{neg_snan}  %s = call float @air.fabs.f32(float %x)\n"),
            Some(0x7F80_0001),
        ),
        nan_kernel(
            "float_fabs_fast",
            format!("{neg_snan}  %s = call fast float @air.fast_fabs.f32(float %x)\n"),
            Some(0x7F80_0001),
        ),
        nan_kernel(
            "float_llvm_fabs",
            format!(
                "{}  %s = call float @llvm.fabs.f32(float %x)\n",
                ob("x", 0xFFC0_1234)
            ),
            Some(0x7FC0_1234),
        ),
        nan_kernel(
            "float_fneg_denormal",
            format!("{}  %s = fneg float %x\n", ob("x", 1)),
            Some(0x8000_0001),
        ),
        nan_kernel(
            "float_fabs_denormal",
            format!(
                "{}  %s = call fast float @air.fast_fabs.f32(float %x)\n",
                ob("x", 0x8000_0001)
            ),
            Some(1),
        ),
        nan_kernel(
            "float_fneg_vector",
            format!("{snan}  %v = insertelement <2 x float> <float 1.0, float 1.0>, float %x, i32 1\n  %w = fneg <2 x float> %v\n  %s = extractelement <2 x float> %w, i32 1\n"),
            Some(0xFF80_0001),
        ),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!("{DECLS}declare float @air.fabs.f32(float)\ndeclare float @air.fast_fabs.f32(float)\ndeclare float @llvm.fabs.f32(float)\n"),
    );
    let Some(bitcode) = assemble("nan-float-sign", &text) else {
        return;
    };
    let fabs = |words: &[u32]| {
        count_ops(words, |w| {
            w[0] & 0xFFFF == crate::spv::Op::ExtInst as u32
                && w.get(4) == Some(&(crate::spv::GlslStd450Op::FAbs as u32))
        })
    };
    for stem in PROFILES.iter().copied().chain(["lavapipe"]) {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let neg = kk.name.contains("fneg");
            let abs = kk.name.contains("fabs");
            assert_eq!(count_op(&words, crate::spv::Op::FNegate) > 0, neg, "{what}");
            assert_eq!(fabs(&words) > 0, abs, "{what}");
            assert_eq!(nan_tests(&words), (0, 0), "{what}");
            if neg || abs {
                assert!(
                    a.precision.contains(&PrecisionNote::NativeNanEncoding),
                    "{what}: {:?}",
                    a.precision
                );
            }
            if neg {
                assert!(
                    a.precision.contains(&PrecisionNote::DeviceDenormals),
                    "{what}: {:?}",
                    a.precision
                );
            }
            let got = interpret(&bitcode, kk.name, &p);
            assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
        }
    }
}

/// Class F: Metal's `float` `fmin`, `fmax`, `min3`, `max3`, and `clamp`, precise and `fast_` alike,
/// and `llvm.minnum`/`llvm.maxnum` with every flag (Metal's fast-math `a < b ? a : b`), with none
/// and with some, answer an operand bit for bit (`nan_minmax`, `nan_minmax_fast`, and their
/// zeros-and-denormals cases, every lane; for the other flag states): a NaN loses, two NaNs and a
/// tie answer the second operand, and a denormal compares as zero but is returned itself. On every
/// profile the device's `NMin`, `NMax`, or `NClamp` decides it: where the profile takes
/// `DriverDeviation::MetalMinMax` measured to be Metal's (the interpreter models them so), and the
/// module requires `VkProperty::MetalMinMax`; elsewhere noted `PrecisionNote::NativeMinMax`. No
/// device note is recorded.
#[test]
fn float_min_max_are_the_device_s_on_every_profile() {
    let two = |a: u32, b: u32| format!("{}{}", ob("x", a), ob("y", b));
    let three = |a: u32, b: u32, c: u32| format!("{}{}{}", ob("x", a), ob("y", b), ob("z", c));
    let ks = [
        nan_kernel(
            "fmin_two_nans",
            format!(
                "{}  %s = call float @air.fmin.f32(float %x, float %y)\n",
                two(0xFFC0_1234, 0x7F80_0001)
            ),
            Some(0x7F80_0001),
        ),
        nan_kernel(
            "fmin_a_nan_loses",
            format!(
                "{}  %s = call float @air.fmin.f32(float %x, float %y)\n",
                two(0xFFC0_1234, 0x4000_0000)
            ),
            Some(0x4000_0000),
        ),
        nan_kernel(
            "fmax_tie_of_zeros",
            format!(
                "{}  %s = call float @air.fmax.f32(float %x, float %y)\n",
                two(0, 0x8000_0000)
            ),
            Some(0x8000_0000),
        ),
        nan_kernel(
            "fmin_denormal_compares_as_zero",
            format!(
                "{}  %s = call float @air.fmin.f32(float %x, float %y)\n",
                two(0, 1)
            ),
            Some(1),
        ),
        nan_kernel(
            "fast_fmin_tie_of_zeros",
            format!(
                "{}  %s = call fast float @air.fast_fmin.f32(float %x, float %y)\n",
                two(0x8000_0000, 0)
            ),
            Some(0),
        ),
        nan_kernel(
            "minnum_fast_a_nan_loses",
            format!(
                "{}  %s = call fast float @llvm.minnum.f32(float %x, float %y)\n",
                two(0x7FC0_0000, 0x3F80_0000)
            ),
            Some(0x3F80_0000),
        ),
        nan_kernel(
            "maxnum_fast_denormal_tie",
            format!(
                "{}  %s = call fast float @llvm.maxnum.f32(float %x, float %y)\n",
                two(0x8000_0000, 0x8000_0001)
            ),
            Some(0x8000_0001),
        ),
        nan_kernel(
            "minnum_unflagged_denormal_tie",
            format!(
                "{}  %s = call float @llvm.minnum.f32(float %x, float %y)\n",
                two(0, 1)
            ),
            Some(1),
        ),
        nan_kernel(
            "maxnum_nsz_arcp_denormal_tie",
            format!(
                "{}  %s = call nsz arcp float @llvm.maxnum.f32(float %x, float %y)\n",
                two(0x8000_0001, 0x8000_0000)
            ),
            Some(0x8000_0000),
        ),
        nan_kernel(
            "minnum_nnan_a_nan_loses",
            format!(
                "{}  %s = call nnan float @llvm.minnum.f32(float %x, float %y)\n",
                two(0x7FC0_0000, 0x3F80_0000)
            ),
            Some(0x3F80_0000),
        ),
        nan_kernel(
            "maxnum_unflagged_two_nans",
            format!(
                "{}  %s = call float @llvm.maxnum.f32(float %x, float %y)\n",
                two(0xFFC0_0001, 0x7F80_0001)
            ),
            Some(0x7F80_0001),
        ),
        nan_kernel(
            "clamp_of_a_nan",
            format!(
                "{}  %s = call float @air.clamp.f32(float %x, float %y, float %z)\n",
                three(0xFFC0_1234, 0x3F80_0000, 0x4000_0000)
            ),
            Some(0x3F80_0000),
        ),
        nan_kernel(
            "fast_min3_of_two_nans",
            format!(
                "{}  %s = call fast float @air.fast_fmin3.f32(float %x, float %y, float %z)\n",
                three(0xFFC0_1234, 0x7F80_0001, 0x8000_0000)
            ),
            Some(0x8000_0000),
        ),
        nan_kernel(
            "max3_keeps_a_nan_payload",
            format!(
                "{}  %s = call float @air.fmax3.f32(float %x, float %y, float %z)\n",
                three(0xFFC0_1234, 0x7F80_0001, 0x7FA0_0000)
            ),
            Some(0x7FA0_0000),
        ),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!(
            "{DECLS}declare float @air.fmin.f32(float, float)\ndeclare float @air.fmax.f32(float, float)\ndeclare float @air.fast_fmin.f32(float, float)\ndeclare float @llvm.minnum.f32(float, float)\ndeclare float @llvm.maxnum.f32(float, float)\ndeclare float @air.fast_fmin3.f32(float, float, float)\ndeclare float @air.fmax3.f32(float, float, float)\n"
        ),
    );
    let Some(bitcode) = assemble("nan-float-min-max", &text) else {
        return;
    };
    let device_pick = |words: &[u32]| {
        count_ops(words, |w| {
            w[0] & 0xFFFF == crate::spv::Op::ExtInst as u32
                && [
                    crate::spv::GlslStd450Op::NMin as u32,
                    crate::spv::GlslStd450Op::NMax as u32,
                    crate::spv::GlslStd450Op::NClamp as u32,
                ]
                .iter()
                .any(|op| w.get(4) == Some(op))
        })
    };
    for stem in PROFILES.iter().copied().chain(["lavapipe"]) {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let device = p.takes_driver_deviation(crate::error::DriverDeviation::MetalMinMax);
            assert!(device_pick(&words) > 0, "{what}");
            assert_eq!(
                a.precision.contains(&PrecisionNote::NativeMinMax),
                !device,
                "{what}: {:?}",
                a.precision
            );
            assert_eq!(
                a.requirements
                    .properties
                    .contains(&crate::device::requirements::VkProperty::MetalMinMax),
                device,
                "{what}"
            );
            for note in [
                PrecisionNote::DeviceDenormals,
                PrecisionNote::DeviceNanEncoding,
            ] {
                assert!(!a.precision.contains(&note), "{what}: {:?}", a.precision);
            }
            // The interpreter reads a denormal operand of `NMin` by its value; Metal's reading, a
            // zero, is the device's, measured by the driver facts and the gate lane
            // (`runtime_lane::driver_deviations`).
            if kk.name.contains("denormal") {
                continue;
            }
            let got = interpret(&bitcode, kk.name, &p);
            assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
        }
    }
}

/// Class H: Metal's `half` `fmod` answers `+0` for a zero remainder of a nonzero dividend and the
/// dividend itself for a zero one, with or without fast-math flags (`nan_fmod_zero`,
/// `nan_fmod_zero_fast`: `fmod(-2, 0.5)` is `+0`, `fmod(-0, 1)` is `-0`), on every profile; a
/// nonzero remainder keeps the dividend's sign, and a `float` zero remainder too (Metal's
/// `fmod(-2, 0.5)` on `float` is `-0`). The interpreter's `OpFRem` is C's `fmod`, the dividend's
/// sign, so the stored bits show the select.
#[test]
fn half_fmod_answers_metals_zero_remainder_on_every_profile() {
    let two = |a: u16, b: u16| format!("{}{}", oh("hx", a), oh("hy", b));
    let ks = [
        half_nan_kernel(
            "half_fmod_negative_dividend",
            format!(
                "{}  %s = call half @air.fmod.f16(half %hx, half %hy)\n",
                two(0xC000, 0x3800)
            ),
            Some(0),
        ),
        half_nan_kernel(
            "half_fmod_negative_dividend_fast",
            format!(
                "{}  %s = call fast half @air.fmod.f16(half %hx, half %hy)\n",
                two(0xFBFF, 0x7BFF)
            ),
            Some(0),
        ),
        half_nan_kernel(
            "half_fmod_of_negative_zero",
            format!(
                "{}  %s = call half @air.fmod.f16(half %hx, half %hy)\n",
                two(0x8000, 0x3C00)
            ),
            Some(0x8000),
        ),
        half_nan_kernel(
            "half_fmod_nonzero_remainder",
            format!(
                "{}  %s = call half @air.fmod.f16(half %hx, half %hy)\n",
                two(0xC200, 0x4000)
            ),
            Some(0xBC00),
        ),
        nan_kernel(
            "float_fmod_negative_dividend",
            format!(
                "{}{}  %s = call float @air.fmod.f32(float %x, float %y)\n",
                ob("x", 0xC000_0000),
                ob("y", 0x3F00_0000)
            ),
            Some(0x8000_0000),
        ),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!(
            "{DECLS}declare half @air.fmod.f16(half, half)\ndeclare float @air.fmod.f32(float, float)\n"
        ),
    );
    let Some(bitcode) = assemble("nan-half-fmod", &text) else {
        return;
    };
    for stem in PROFILES.iter().copied().chain(["lavapipe"]) {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let got = interpret(&bitcode, kk.name, &p);
            assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
        }
    }
}

/// Class I: Metal's `mix(x, y, t)` on `float` and `half` is `fma(t, y, fma(-t, x, x))`, with and
/// without fast-math flags (`nan_functions`: `mix(inf, 1, 0)` is the canonical NaN; `nan_half`:
/// `mix(-0.3, 0.7, 0.2)` is `0xAE68`, where the blend gives `0xAE66`). On every profile the module
/// takes the device's `FMix` and no fused multiply-add: a `float` one on a profile whose `float`
/// functions are Metal's (moltenvk-m3) reaches Metal's own `mix`, with no note; every other records
/// `NativeMix` (a `half` one the blend in its `float` container, rounded once).
#[test]
fn mix_is_the_devices_fmix_on_every_profile() {
    let three = |a: u32, b: u32, c: u32| format!("{}{}{}", ob("x", a), ob("y", b), ob("z", c));
    let halves = |a: u16, b: u16, c: u16| format!("{}{}{}", oh("hx", a), oh("hy", b), oh("hz", c));
    let ks = [
        nan_kernel(
            "mix_infinity_weighted_out",
            format!(
                "{}  %s = call float @air.mix.f32(float %x, float %y, float %z)\n",
                three(0x7F80_0000, 0x3F80_0000, 0)
            ),
            Some(0x7FC0_0000),
        ),
        nan_kernel(
            "mix_negative_zeros_fast",
            format!(
                "{}  %s = call fast float @air.mix.f32(float %x, float %y, float %z)\n",
                three(0x8000_0000, 0x8000_0000, 0)
            ),
            Some(0),
        ),
        nan_kernel(
            "mix_one_endpoint",
            format!(
                "{}  %s = call float @air.mix.f32(float %x, float %y, float %z)\n",
                three(0x501502F9, 0x3F80_0000, 0x3F80_0000)
            ),
            Some(0x3F80_0000),
        ),
        half_nan_kernel(
            "half_mix",
            format!(
                "{}  %s = call half @air.mix.f16(half %hx, half %hy, half %hz)\n",
                halves(0xB4CD, 0x399A, 0x3266)
            ),
            Some(0xAE68),
        ),
        half_nan_kernel(
            "half_mix_fast",
            format!(
                "{}  %s = call fast half @air.mix.f16(half %hx, half %hy, half %hz)\n",
                halves(0xB4CD, 0x399A, 0x3266)
            ),
            Some(0xAE68),
        ),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!(
            "{DECLS}declare float @air.mix.f32(float, float, float)\ndeclare half @air.mix.f16(half, half, half)\n"
        ),
    );
    let Some(bitcode) = assemble("mix-fused", &text) else {
        return;
    };
    let blend = |words: &[u32]| {
        count_ops(words, |w| {
            w[0] & 0xFFFF == crate::spv::Op::ExtInst as u32
                && w.get(4) == Some(&(crate::spv::GlslStd450Op::FMix as u32))
        })
    };
    let fma = |words: &[u32]| {
        count_ops(words, |w| {
            w[0] & 0xFFFF == crate::spv::Op::ExtInst as u32
                && w.get(4) == Some(&(crate::spv::GlslStd450Op::Fma as u32))
        }) + count_op(words, crate::spv::Op::FmaKHR)
    };
    for stem in PROFILES
        .iter()
        .copied()
        .chain(["lavapipe", "amd-rdna3-radv"])
    {
        let p = profile_with_canonical_nans(stem, false);
        let metal = p.fields().float_functions == FloatFunctions::Metal;
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert!(blend(&words) > 0, "{what}: FMix");
            assert_eq!(fma(&words), 0, "{what}: a fused form");
            let half = kk.name.starts_with("half_");
            assert_eq!(
                a.precision.contains(&PrecisionNote::NativeMix),
                !(metal && !half),
                "{what}: {:?}",
                a.precision
            );
        }
    }
}

/// Class K: Metal's `half` functions answer the canonical `half` NaN `0x7E00` with and without
/// fast-math flags (`nan_half`, `nan_half_fast`): `llvm.fmuladd`, `ldexp` (unlike `float`'s, which
/// returns the operand), and `sqrt`, which the device answers where the profile takes
/// `MetalCanonicalNans` and which are noted `PrecisionNote::NativeNanEncoding` elsewhere. `round`
/// and `fract` return a NaN operand itself on Metal, and `asin`'s domain error takes the operand's
/// sign: each the device's NaN on every profile, noted whatever the fact. None is selected.
/// `round`'s zero is C99's (`-0` for `round(-0.3)`, where Metal answers `+0`), and `fract` of an
/// infinity the device's `Fract`'s NaN (Metal's: `+0`, and `0x3BFF` with every flag).
#[test]
fn half_functions_answer_the_device_s_nan() {
    let hx = oh("hx", 0xFE01);
    let snan = oh("hx", 0x7C01);
    let inf = oh("hx", 0x7C00);
    let tenth = oh("hx", 0xB4CD);
    let ks = [
        half_nan_kernel(
            "half_fmuladd",
            format!(
                "{hx}  %s = call half @llvm.fmuladd.f16(half %hx, half 0xH3C00, half 0xH0000)\n"
            ),
            None,
        ),
        half_nan_kernel(
            "half_ldexp",
            format!("{hx}  %s = call half @air.ldexp.f16(half %hx, i32 2)\n"),
            None,
        ),
        half_nan_kernel(
            "half_sqrt_fast",
            format!("{hx}  %s = call fast half @air.sqrt.f16(half %hx)\n"),
            None,
        ),
        half_nan_kernel(
            "half_round_snan",
            format!("{snan}  %s = call half @air.round.f16(half %hx)\n"),
            None,
        ),
        half_nan_kernel(
            "half_round_to_zero",
            format!("{tenth}  %s = call fast half @air.round.f16(half %hx)\n"),
            Some(0x8000),
        ),
        half_nan_kernel(
            "half_fract_flagged",
            format!("{inf}  %s = call fast half @air.fract.f16(half %hx)\n"),
            None,
        ),
        half_nan_kernel(
            "half_fract",
            format!("{inf}  %s = call half @air.fract.f16(half %hx)\n"),
            None,
        ),
        half_nan_kernel(
            "half_asin",
            format!("{hx}  %s = call half @air.asin.f16(half %hx)\n"),
            None,
        ),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!(
            "{DECLS}declare half @llvm.fmuladd.f16(half, half, half)\ndeclare half @air.ldexp.f16(half, i32)\ndeclare half @air.sqrt.f16(half)\ndeclare half @air.round.f16(half)\ndeclare half @air.fract.f16(half)\ndeclare half @air.asin.f16(half)\n"
        ),
    );
    let Some(bitcode) = assemble("nan-half-functions", &text) else {
        return;
    };
    nans_are_the_device_s(&bitcode, &ks, true, &["half_asin"], &|name, fact| {
        name.starts_with("half_round")
            || name.starts_with("half_fract")
            || name == "half_asin"
            || !fact
    });
}
