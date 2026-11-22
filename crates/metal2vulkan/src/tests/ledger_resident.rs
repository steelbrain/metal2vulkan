//! What a translation holds stays within what its ledger counts.
//!
//! The ledger is an attempt's memory ceiling wherever no worker boundary measures one (the runtime
//! translates in its own process): it refuses past 450 MiB (`budget::LEDGER_BYTES`) so that the
//! attempt holds under 500 MiB. That holds only while the bytes the attempt holds at once never
//! pass what the ledger counts by more than a ninth of it (500 / 450), at every point of the
//! attempt. A structure held but not charged passes it in the stages that follow: the plan's
//! strategy table, one entry per value, was released from the ledger when selection returned,
//! though emission, the roll and serialization still hold the plan.
//!
//! So a kernel of that shape, a large function applied many times in a chain (the AIR inliner
//! writes each application out, every value has a strategy, and the roll writes the applications
//! as a loop), compiles under ledger caps of one to seven eighths of its uncapped ledger peak: each
//! is refused `LedgerBytes`, and the most bytes it held at once by then are at most the cap and a
//! ninth. Uncapped, the most bytes it held at once are at most its ledger peak and a ninth. The
//! allocator counts the bytes the translation asked for on its own thread
//! (`constant_blob_footprint::held_during`), the same number on any host.
//!
//! A kernel of one long block holds the same under the same caps. The SPIR-V builder cannot hold
//! the meter while the emitter grows it, so its growth is charged at the walk's poll points
//! (`structurize::emit::Poll`); charged only after each block, a long block's whole code was built
//! past the cap before the refusal, and the doublings of its instruction storage and of the
//! module's value table were reserved before they were charged.
//!
//! `llvm-as` assembles the kernel; without it the test skips visibly.

use crate::budget::{Budget, Meter, LEDGER_BYTES, SPIRV_WORDS, WORK_UNITS};
use crate::device::{DevicePolicy, DeviceProfile};
use crate::error::{Error, Limit};
use crate::library::FunctionConstantValues;
use crate::lower::select::LowerInputs;
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::testing::{compile_kept, llvm_assemble, profile, AssembleError, Scratch};
use crate::tests::constant_blob_footprint::held_during;

/// The `float` operations of the body.
const OPS: usize = 6_000;

/// The operations before the body's first selection; each later stretch is one longer than the
/// one before.
const SEGMENT: usize = 500;

/// The applications of `@body`, each reading the one before's result.
const CHAIN: usize = 16;

/// `@body(x, c, out)`: `x + c`, then `* 0.5` or `+ 0.25` as [`halves`] picks, `OPS - 1` times,
/// and after stretches of `SEGMENT`, `SEGMENT + 1`, ... operations a selection that stores `-1`
/// to word 0 when the value is below `-2^30`; `@chain` applies the body [`CHAIN`] times, each
/// application to the one before's result, with the constants 1, 2, ..., and stores the last
/// result to word 0.
fn module() -> String {
    let mut s = String::from(HEADER);
    s.push_str(
        "define internal fastcc float @body(float %x, float %c, ptr addrspace(1) %out) {\nb0:\n",
    );
    s.push_str("  %v0 = fadd float %x, %c\n");
    let (mut next, mut stretch) = (SEGMENT, SEGMENT);
    for i in 1..OPS {
        let (op, k) = if halves(i) {
            ("fmul", "0x3FE0000000000000")
        } else {
            ("fadd", "0x3FD0000000000000")
        };
        s.push_str(&format!("  %v{i} = {op} float %v{}, {k}\n", i - 1));
        if i == next && i + 1 < OPS {
            stretch += 1;
            next += stretch;
            s.push_str(&format!(
                "  %t{i} = fcmp olt float %v{i}, 0xC1D0000000000000\n  br i1 %t{i}, label %r{i}, label %j{i}\nr{i}:\n  store float -1.0, ptr addrspace(1) %out, align 4\n  br label %j{i}\nj{i}:\n"
            ));
        }
    }
    s.push_str(&format!("  ret float %v{}\n}}\n\n", OPS - 1));
    s.push_str(
        "define void @chain(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {\nentry:\n  %c0 = fadd float 0x3FF8000000000000, -0.0\n",
    );
    for j in 1..=CHAIN {
        s.push_str(&format!(
            "  %c{j} = call fastcc float @body(float %c{}, float {j}.0, ptr addrspace(1) %0)\n",
            j - 1
        ));
    }
    s.push_str(&format!(
        "  store float %c{CHAIN}, ptr addrspace(1) %0, align 4\n  ret void\n}}\n\n"
    ));
    s.push_str(&footer("chain"));
    s
}

/// The `bfloat` operations of [`long_block`].
const LONG: usize = 8_000;

/// `@long(out)`: one block that narrows word 0 to `bfloat` as `x`, applies `* x` or `+ x` to it
/// as [`halves`] picks, [`LONG`] times, and stores the result, widened, to word 0. Each `bfloat`
/// operation lowers to a bounded integer rounding of up to 18 instructions, so the block's code is
/// most of what the translation holds.
fn long_block() -> String {
    let mut s = String::from(HEADER);
    s.push_str(
        "define void @long(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {\nentry:\n  %a = load float, ptr addrspace(1) %0, align 4\n  %v0 = fptrunc float %a to bfloat\n",
    );
    for i in 1..=LONG {
        let op = if halves(i) { "fmul" } else { "fadd" };
        s.push_str(&format!("  %v{i} = {op} bfloat %v{}, %v0\n", i - 1));
    }
    s.push_str(&format!(
        "  %r = fpext bfloat %v{LONG} to float\n  store float %r, ptr addrspace(1) %0, align 4\n  ret void\n}}\n\n"
    ));
    s.push_str(&footer("long"));
    s
}

/// The module's target lines.
const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// The metadata of kernel `@name(device float *out [[buffer(0)]])`.
fn footer(name: &str) -> String {
    format!("!llvm.module.flags = !{{!0}}\n!air.kernel = !{{!20}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n!0 = !{{i32 7, !\"air.max_device_buffers\", i32 31}}\n!10 = !{{}}\n!11 = !{{!12}}\n!12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"out\"}}\n!17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n!20 = !{{ptr @{name}, !10, !11}}\n")
}

/// Whether operation `i` of the body halves the value (else it adds `0.25`): a bit of a
/// multiplicative hash of `i`, so no stretch of the body repeats another (only the applications
/// repeat).
fn halves(i: usize) -> bool {
    ((i as u64).wrapping_mul(2_654_435_761) >> 16) & 1 == 1
}

/// `text` assembled by `llvm-as`, or `None` (the test skips visibly) without it.
fn assembled(text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new("m2v-ledger-resident");
    let ll = scratch.path().join("kernel.ll");
    std::fs::write(&ll, text).expect("write");
    match llvm_assemble(&ll) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected the module: {e}"),
    }
}

/// Kernel `entry` compiled for `p` under a ledger cap of `cap` bytes: the result, the ledger
/// peak, and the most bytes the compile held at once.
fn compiled(bc: &[u8], entry: &str, p: &DeviceProfile, cap: u64) -> (Result<(), Error>, u64, u64) {
    let mut m = Meter::with_caps(&Budget::default(), WORK_UNITS, cap);
    let (r, held) = held_during(|| {
        compile_bitcode(
            bc,
            EntryRef::Name(entry),
            &FunctionConstantValues::defaults(),
            p,
            &LowerInputs::new(),
            &mut m,
            SPIRV_WORDS,
        )
    });
    (r.map(|_| ()), m.ledger_peak(), held)
}

/// The most bytes a ledger of `ledger` bytes stands for: it and a ninth (500 MiB for the 450 MiB
/// cap).
fn within(ledger: u64) -> u64 {
    ledger + ledger / 9
}

#[test]
fn what_a_translation_holds_stays_within_its_ledger() {
    let Some(bc) = assembled(&module()) else {
        return;
    };
    let p = profile("runtime-floor", DevicePolicy::RESIDENT_TABLES).expect("profile");
    // The shape: the applications are rolled into a loop (the body holds none).
    let (_, _, kept) = compile_kept(&bc, "chain", &p).expect("chain");
    assert_eq!(kept.loops().len(), 1, "the applications roll into one loop");
    holds_within_its_ledger(&bc, "chain", &p);
}

#[test]
fn a_long_block_s_code_is_charged_while_it_grows() {
    let Some(bc) = assembled(&long_block()) else {
        return;
    };
    let p = profile("runtime-floor", DevicePolicy::RESIDENT_TABLES).expect("profile");
    holds_within_its_ledger(&bc, "long", &p);
}

/// `entry` translates within the standard ledger, holding at most its ledger peak and a ninth at
/// once; under caps of one to seven eighths of that peak it is refused `LedgerBytes`, holding at
/// most the cap and a ninth by then.
fn holds_within_its_ledger(bc: &[u8], entry: &str, p: &DeviceProfile) {
    let (r, peak, held) = compiled(bc, entry, p, LEDGER_BYTES);
    r.expect("translates");
    eprintln!("uncapped: ledger peak {peak} bytes, {held} bytes held at once");
    assert!(
        held <= within(peak),
        "uncapped: {held} bytes held at once against a ledger peak of {peak} bytes"
    );
    for eighths in 1..8 {
        let cap = peak * eighths / 8;
        let (r, _, held) = compiled(bc, entry, p, cap);
        eprintln!("cap {cap} bytes ({eighths}/8): {held} bytes held at once");
        assert_eq!(
            r,
            Err(Error::LimitExceeded(Limit::LedgerBytes)),
            "cap {cap} bytes ({eighths}/8)"
        );
        assert!(
            held <= within(cap),
            "cap {cap} bytes ({eighths}/8): {held} bytes held at once by the refusal"
        );
    }
}
