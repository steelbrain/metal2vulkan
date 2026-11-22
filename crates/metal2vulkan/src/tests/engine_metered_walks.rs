//! Input-scaled walks of the memory plan and the access family, metered or removed, and
//! canonicalization's SCCP over a PHI of many incoming values.
//!
//! Each kernel is generated at a size where the code previously walked a table per use or re-walked
//! a pointer's roots per access, unmetered; the tests check the work units and ledger bytes the
//! meter charges now, which are deterministic, and that the outputs pass L3.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::budget::{Budget, Meter, LEDGER_BYTES, SPIRV_WORDS, WORK_UNITS};
use crate::device::profile::DeviceProfile;
use crate::device::DevicePolicy;
use crate::error::Error;
use crate::library::FunctionConstantValues;
use crate::lower::select::LowerInputs;
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::testing::{link_and_validate, llvm_assemble, profile, AssembleError};

const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

fn p(stem: &str) -> DeviceProfile {
    profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile")
}

fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let text = format!(
        "target datalayout = \"{DATALAYOUT}\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n{body}"
    );
    let dir = std::env::temp_dir().join(format!(
        "m2v-walks-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, &text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}\n{text}"),
    }
}

/// The work units and ledger peak of compiling entry `entry` of `bc` for `stem`, with L3 when it
/// translates.
fn charged(bc: &[u8], entry: &str, stem: &str) -> Result<(u64, u64), Error> {
    charged_on(bc, entry, &p(stem), stem)
}

/// [`charged`] on profile `prof`, named `stem` in messages.
fn charged_on(
    bc: &[u8],
    entry: &str,
    prof: &DeviceProfile,
    stem: &str,
) -> Result<(u64, u64), Error> {
    let mut m = Meter::with_caps(&Budget::default(), WORK_UNITS, LEDGER_BYTES);
    let a = compile_bitcode(
        bc,
        EntryRef::Name(entry),
        &FunctionConstantValues::defaults(),
        prof,
        &LowerInputs::new(),
        &mut m,
        SPIRV_WORDS,
    )?;
    link_and_validate(&a, prof).unwrap_or_else(|e| panic!("{entry} on {stem}: L3: {e}"));
    Ok((m.work(), m.ledger_peak()))
}

/// The metadata of a kernel `@NAME(out, t)`.
const KERNEL_MD: &str = r#"!air.kernel = !{!0}
!0 = !{ptr @NAME, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// A kernel with `arena` allocas in the private arena (each stored as a word and read back as a
/// byte, so its accesses are punned) and `typed` allocas that are typed objects (each stored and
/// loaded as a word), every value summed into `out[t]`. Each alloca is used where it is defined,
/// so the code previously searched the arena's entries, or the function's typed objects, per use:
/// quadratic in the allocas.
fn homes_kernel(arena: u32, typed: u32) -> String {
    let mut s = String::from("define void @homes(ptr addrspace(1) %out, i32 %t) {\n");
    s += "  %s0 = add i32 %t, 0\n";
    let mut k = 0;
    for i in 0..arena {
        s += &format!(
            "  %a{i} = alloca i32, align 4\n  store i32 %t, ptr %a{i}, align 4\n  %b{i} = load i8, ptr %a{i}, align 1\n  %e{i} = zext i8 %b{i} to i32\n  %s{} = add i32 %s{k}, %e{i}\n",
            k + 1
        );
        k += 1;
    }
    for j in 0..typed {
        s += &format!(
            "  %c{j} = alloca i32, align 4\n  store i32 %s{k}, ptr %c{j}, align 4\n  %d{j} = load i32, ptr %c{j}, align 4\n  %s{} = add i32 %s{k}, %d{j}\n",
            k + 1
        );
        k += 1;
    }
    s += &format!(
        "  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t\n  store i32 %s{k}, ptr addrspace(1) %o, align 4\n  ret void\n}}\n\n{}",
        KERNEL_MD.replace("NAME", "homes")
    );
    s
}

/// A use of an arena alloca finds its arena entry by a search, and a use of a typed object its
/// position by one, so a function with many of each selects and emits in work linear in its allocas
/// (`ArenaLayout::entry`, `local::use_home`, `emit_typed`). 4096 of each translate and pass L3, and
/// twice as many charge under 2.2 times the work units, on `moltenvk-m3` with a thread's 192 KiB of
/// Function storage, as the M3 Max holds (past the 16 KiB every device holds). The scans previously
/// were not metered, so the units are the same there; at 16384 of each the previous code took
/// 32.0 s in a test build where this code took 6.8 s, L3 included (one run each).
#[test]
fn many_arena_and_typed_allocas_are_found_without_a_scan() {
    let mut prof = p("moltenvk-m3");
    prof.edit(|f| {
        f.function_storage = crate::device::profile::FunctionStorage::Assumed(192 * 1024)
    });
    let work = |n: u32| {
        let bc = assemble(&format!("homes_{n}"), &homes_kernel(n, n))?;
        Some(
            charged_on(&bc, "homes", &prof, "moltenvk-m3")
                .expect("translates")
                .0,
        )
    };
    let (Some(small), Some(large)) = (work(1 << 12), work(1 << 13)) else {
        return;
    };
    assert!(
        large * 10 < small * 22,
        "{large} units for 2^13, {small} for 2^12"
    );
}

/// A kernel `@roots(out, tab)` whose table `tab` holds `count` buffers (Metal slots
/// `1..=count`): it loads the element at the dynamic index `out[0]`, `%el`, a pointer whose root
/// set is every element, then runs `body`, which leaves its sum in `%sum`, stored to `out[1]`.
fn table_kernel(count: u32, body: &str) -> String {
    format!(
        "define void @roots(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr noundef %1) {{\nentry:\n  %k = load i32, ptr addrspace(1) %0, align 4\n  %x = zext i32 %k to i64\n  %tp = getelementptr inbounds ptr addrspace(1), ptr %1, i64 %x\n  %el = load ptr addrspace(1), ptr %tp, align 8\n{body}  %o = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1\n  store i32 %sum, ptr addrspace(1) %o, align 4\n  ret void\n}}\n\n\
         !air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @roots, !10, !11}}\n!10 = !{{}}\n!11 = !{{!12, !13}}\n\
         !12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
         !13 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 {count}, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"tab\"}}\n\
         !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// `steps` pointers, each one word past the last from `%el` on, and a read through the last.
fn chain_body(steps: u32) -> String {
    let mut body =
        String::from("  %p0 = getelementptr inbounds i32, ptr addrspace(1) %el, i64 1\n");
    for i in 1..steps {
        body += &format!(
            "  %p{i} = getelementptr inbounds i32, ptr addrspace(1) %p{}, i64 1\n",
            i - 1
        );
    }
    body += &format!(
        "  %sum = load i32, ptr addrspace(1) %p{}, align 4\n",
        steps - 1
    );
    body
}

/// The largest buffer table a Metal function binds: Metal's buffer argument table has 31 entries
/// per function (every GPU family), and `out` takes slot 0.
const LARGEST_TABLE: u32 = 30;

/// What a buffer pointer's root set holds is read once per set, before selection, so a chain of
/// `steps` pointers over one set of `roots` buffers costs work in `steps + roots`, not in `steps *
/// roots` (`root_sets`, `access::statics`, provenance's marks). A table is at most
/// [`LARGEST_TABLE`] buffers, so the set's size is fixed and the chain grows: what a table of 30
/// buffers costs beyond a table of one stays the same from 2^11 steps to 2^12, where a read of the
/// set per pointer would charge 29 more units per pointer, doubling the difference. All four
/// translate and pass L3. Previously those walks were not metered, so the units do not tell the two
/// apart there; the times were measured. With the read of a set charged again for every pointer
/// that names it, this test fails.
#[test]
fn a_chain_of_pointers_over_many_roots_reads_their_set_once() {
    let work = |steps: u32, roots: u32| {
        let bc = assemble(
            &format!("chain_{steps}_{roots}"),
            &table_kernel(roots, &chain_body(steps)),
        )?;
        Some(charged(&bc, "roots", "lavapipe").expect("translates").0)
    };
    let extra = |steps: u32| Some(work(steps, LARGEST_TABLE)? - work(steps, 1)?);
    let (Some(short), Some(long)) = (extra(1 << 11), extra(1 << 12)) else {
        return;
    };
    assert!(
        long * 10 < short * 15 + 640,
        "a table of 30 buffers costs {long} more units than one over 2^12 steps, {short} over 2^11"
    );
}

/// `accesses` reads through `%el`, each at its own offset, summed.
fn reads_body(accesses: u32) -> String {
    let mut body = String::from("  %s0 = add i32 %k, 0\n");
    for i in 0..accesses {
        body += &format!(
            "  %p{i} = getelementptr inbounds i32, ptr addrspace(1) %el, i64 {i}\n  %v{i} = load i32, ptr addrspace(1) %p{i}, align 4\n  %s{} = add i32 %s{i}, %v{i}\n",
            i + 1
        );
    }
    body += &format!("  %sum = add i32 %s{accesses}, 0\n");
    body
}

/// Past the per-stage storage-buffer limit, `finish_slots` moves the storage views to their
/// fallback slots one at a time, keeping the descriptor census as it goes and the order sorted
/// once, so the slots are placed in work linear in their number. A table is at most
/// [`LARGEST_TABLE`] buffers (Metal's 31 buffer entries), so the moves are bounded by Metal's
/// table: tables of 15 and 30 buffers beside `out` on `lavapipe` held to 30 storage buffers per
/// stage (`roadmap-2022`'s) translate, pass L3, and
/// charge under 2.2 times the work units for twice the buffers. Previously
/// each move recounted every slot and placed every push field, unmetered, so the units are the same
/// there; the measured times (at 2^12 and 2^13 buffers, a table no Metal function binds) were
/// measured.
#[test]
fn slots_past_the_storage_limit_move_in_one_pass() {
    let prof = crate::pipeline::testing::profile_with_storage_buffers(
        "lavapipe",
        DevicePolicy::RESIDENT_TABLES,
        30,
    )
    .expect("profile");
    let work = |count: u32| {
        let bc = assemble(
            &format!("slots_{count}"),
            &table_kernel(count, &reads_body(1)),
        )?;
        Some(
            charged_on(&bc, "roots", &prof, "lavapipe")
                .expect("translates")
                .0,
        )
    };
    let (Some(small), Some(large)) = (work(LARGEST_TABLE / 2), work(LARGEST_TABLE)) else {
        return;
    };
    assert!(
        large * 10 < small * 22,
        "{large} units for 30 buffers, {small} for 15"
    );
}

/// The `(literal, label)` pairs of every `OpSwitch` in module `words`.
fn switch_pairs(words: &[u32]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut at = 5;
    while let Some(first) = words.get(at) {
        let count = (first >> 16) as usize;
        if count == 0 {
            break;
        }
        // `OpSwitch selector default (literal label)*`, one-word literals for a 32-bit selector.
        if first & 0xffff == 251 {
            out.push((count - 3) / 2);
        }
        at += count;
    }
    out
}

/// A kernel `@cases(out, t)` that switches on `out[0]` over `cases` case values, each its own
/// block storing its value to `out[1]` (the default stores `t`): a source switch of one literal
/// per arm, which the structurizer emits as one `OpSwitch` pair per case.
fn switch_kernel(cases: u32) -> String {
    let mut s = String::from(
        "define void @cases(ptr addrspace(1) %out, i32 %t) {\nentry:\n  %k = load i32, ptr addrspace(1) %out, align 4\n  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1\n  switch i32 %k, label %d [",
    );
    for i in 0..cases {
        s += &format!(" i32 {i}, label %c{i}");
    }
    s += " ]\n";
    for i in 0..cases {
        s += &format!("c{i}:\n  store i32 {i}, ptr addrspace(1) %o, align 4\n  br label %join\n");
    }
    s += &format!(
        "d:\n  store i32 %t, ptr addrspace(1) %o, align 4\n  br label %join\njoin:\n  ret void\n}}\n\n{}",
        KERNEL_MD.replace("NAME", "cases")
    );
    s
}

/// The pairs of each `OpSwitch` of `@cases` with `cases` case values on `stem`, after L3.
fn source_switches(cases: u32, stem: &str) -> Option<Vec<usize>> {
    let bc = assemble(&format!("cases_{cases}"), &switch_kernel(cases))?;
    let prof = p(stem);
    let mut m = Meter::with_caps(&Budget::default(), WORK_UNITS, LEDGER_BYTES);
    let a = compile_bitcode(
        &bc,
        EntryRef::Name("cases"),
        &FunctionConstantValues::defaults(),
        &prof,
        &LowerInputs::new(),
        &mut m,
        SPIRV_WORDS,
    )
    .unwrap_or_else(|e| panic!("{cases} cases on {stem}: {e}"));
    let (_, words) =
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{cases} cases on {stem}: L3: {e}"));
    Some(switch_pairs(&words))
}

/// A switch of 16,384 case values is one more `(literal, label)` pair than one `OpSwitch` may carry
/// (SPIR-V's universal limit, which `spirv-val` enforces). Previously such a module failed L3, as
/// did a switch-at-use over a table of 16,384 buffers (a probe; no Metal function binds that
/// table); the switch is now split, every `OpSwitch` within the limit. At 16,383 case values the
/// switch is one `OpSwitch` of 16,383 pairs, as before.
#[test]
fn a_switch_past_the_pair_limit_is_split_and_passes_l3() {
    let Some(at_limit) = source_switches(16_383, "moltenvk-m3") else {
        return;
    };
    assert_eq!(at_limit.iter().max(), Some(&16_383), "{at_limit:?}");
    let Some(past) = source_switches(16_384, "moltenvk-m3") else {
        return;
    };
    assert!(
        past.iter().all(|n| *n <= 16_383) && past.len() > 1,
        "an OpSwitch past the limit, or no split: {past:?}"
    );
}

/// A kernel `@phis(out, t)` whose pointer phi `%p` merges `incoming` pointers into `out` (one
/// per arm of a switch on `out[0]`), then reads through `%p` in each of `uses` later blocks, so
/// every read crosses a block and reads `%p`'s representation; the sum goes to `out[0]`.
fn phi_kernel(incoming: u32, uses: u32) -> String {
    let mut s = String::from(
        "define void @phis(ptr addrspace(1) %out, i32 %t) {\nentry:\n  %k = load i32, ptr addrspace(1) %out, align 4\n  switch i32 %k, label %b0 [",
    );
    for i in 1..incoming {
        s += &format!(" i32 {i}, label %b{i}");
    }
    s += " ]\n";
    for i in 0..incoming {
        s += &format!(
            "b{i}:\n  %g{i} = getelementptr inbounds i32, ptr addrspace(1) %out, i64 {i}\n  br label %join\n"
        );
    }
    s += "join:\n  %p = phi ptr addrspace(1) ";
    let arms: Vec<String> = (0..incoming).map(|i| format!("[ %g{i}, %b{i} ]")).collect();
    s += &arms.join(", ");
    s += "\n  %s0 = add i32 %t, 0\n  br label %u0\n";
    for j in 0..uses {
        s += &format!(
            "u{j}:\n  %q{j} = getelementptr inbounds i32, ptr addrspace(1) %p, i64 {j}\n  %v{j} = load i32, ptr addrspace(1) %q{j}, align 4\n  %s{} = add i32 %s{j}, %v{j}\n  br label %u{}\n",
            j + 1,
            j + 1
        );
    }
    s += &format!(
        "u{uses}:\n  store i32 %s{uses}, ptr addrspace(1) %out, align 4\n  ret void\n}}\n\n{}",
        KERNEL_MD.replace("NAME", "phis")
    );
    s
}

/// A pointer phi's incoming values are read once per phi before selection, not on every read of its
/// representation (`access::lowered_pointer` and the families' representations), so a phi of
/// `incoming` values read in `uses` blocks costs work in `incoming + uses`. Twice the incoming
/// values and twice the uses translate, pass L3, and charge under 2.2 times the units, where a walk
/// of the incoming values per read would charge about four times as many. Previously the walk per
/// read was not metered, so the units do not tell the two apart there; the times were measured.
#[test]
fn a_pointer_phi_is_read_once_however_many_uses_read_it() {
    let work = |n: u32| {
        let bc = assemble(&format!("phis_{n}"), &phi_kernel(n, n))?;
        Some(charged(&bc, "phis", "lavapipe").expect("translates").0)
    };
    let (Some(small), Some(large)) = (work(1 << 10), work(1 << 11)) else {
        return;
    };
    assert!(
        large * 10 < small * 22,
        "{large} units for 2^11 incoming values and uses, {small} for 2^10"
    );
}

/// Canonicalization's SCCP meets each of a PHI's incoming values once as its edge becomes
/// executable and once as the value changes, each meet a work unit, instead of meeting all of them
/// at each of the PHI's visits (one visit per incoming edge: quadratic in the incoming values,
/// unmetered; 12.0 s at 2^14 in release). A phi of twice the incoming values, read in 16 blocks,
/// translates, passes L3, and charges under 2.2 times the units. This catches a regression that
/// keeps the meets metered. An unmetered one keeps the units linear and only the time grows; the
/// unit test `air::canon::sccp::tests::a_phi_costs_work_linear_in_its_entries` counts the meets
/// themselves, metered or not, and fails on it.
#[test]
fn a_phi_of_many_incoming_values_is_solved_in_linear_work() {
    let work = |n: u32| {
        let bc = assemble(&format!("sccp_{n}"), &phi_kernel(n, 16))?;
        Some(charged(&bc, "phis", "lavapipe").expect("translates").0)
    };
    let (Some(small), Some(large)) = (work(1 << 11), work(1 << 12)) else {
        return;
    };
    assert!(
        large * 10 < small * 22,
        "{large} units for 2^12 incoming values, {small} for 2^11"
    );
}

/// A kernel `@hands(ts, out, gid)` whose texture `%t0` is element `gid % count` of the texture
/// array `ts` of `count` textures (a handle of `count` candidates), selected against the last
/// selection in each of `uses` blocks (every select reads `%t0` across a block), then read once
/// through the last selection.
fn handle_kernel(count: u32, uses: u32) -> String {
    let mut s = format!(
        "define void @hands(ptr readonly byval([{count} x ptr addrspace(1)]) %ts, ptr addrspace(1) %out, i32 %gid) {{\nentry:\n  %i = urem i32 %gid, {count}\n  %x = zext i32 %i to i64\n  %e = getelementptr inbounds [{count} x ptr addrspace(1)], ptr %ts, i64 0, i64 %x\n  %t0 = load ptr addrspace(1), ptr %e, align 8\n  %odd = and i32 %gid, 1\n  %c = icmp eq i32 %odd, 0\n  br label %u0\n"
    );
    for j in 0..uses {
        s += &format!(
            "u{j}:\n  %t{} = select i1 %c, ptr addrspace(1) %t0, ptr addrspace(1) %t{j}\n  br label %u{}\n",
            j + 1,
            j + 1
        );
    }
    s += &format!(
        "u{uses}:\n  %s = call ptr addrspace(2) @air.get_read_sampler()\n  %xy = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0\n  %r = call {{ <4 x float>, i8 }} @air.read_texture_2d.v4f32(ptr addrspace(1) %t{uses}, ptr addrspace(2) %s, <2 x i32> %xy, <2 x i32> zeroinitializer, i32 0, i32 0)\n  %v = extractvalue {{ <4 x float>, i8 }} %r, 0\n  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %gid\n  store <4 x float> %v, ptr addrspace(1) %p, align 16\n  ret void\n}}\n\n\
         declare ptr addrspace(2) @air.get_read_sampler()\n\
         declare {{ <4 x float>, i8 }} @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)\n\n\
         !air.kernel = !{{!0}}\n!0 = !{{ptr @hands, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4, !5}}\n\
         !3 = !{{i32 0, !\"air.texture\", !\"air.location_index\", i32 0, i32 {count}, !\"air.sample\", !\"air.arg_type_name\", !\"array<texture2d<float, sample>, {count}>\", !\"air.arg_name\", !\"ts\"}}\n\
         !4 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 16, !\"air.arg_type_align_size\", i32 16, !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"out\"}}\n\
         !5 = !{{i32 2, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}}\n"
    );
    s
}

/// A texture handle's domain is read once per root set (and, for an array element, per offset
/// range) before selection, not on every read of its representation or operand (`access::handle`),
/// so a handle of `count` candidates selected in `uses` blocks costs work in `count + uses`; only
/// the read that holds the candidates walks them again, metered. A texture array is at most 128
/// textures (Metal's texture argument table): 64 candidates in 2^10 blocks and twice both
/// translate, pass L3, and charge under 2.2 times the units, where a walk of the candidates per
/// read would charge about four times as many. Previously those walks were not metered, so the
/// units do not tell the two apart there; the times were measured.
#[test]
fn a_handle_is_read_once_however_many_uses_read_it() {
    let work = |count: u32, uses: u32| {
        let bc = assemble(
            &format!("hands_{count}_{uses}"),
            &handle_kernel(count, uses),
        )?;
        Some(charged(&bc, "hands", "lavapipe").expect("translates").0)
    };
    let (Some(small), Some(large)) = (work(64, 1 << 10), work(128, 1 << 11)) else {
        return;
    };
    assert!(
        large * 10 < small * 22,
        "{large} units for 128 candidates and 2^11 uses, {small} for 64 and 2^10"
    );
}
