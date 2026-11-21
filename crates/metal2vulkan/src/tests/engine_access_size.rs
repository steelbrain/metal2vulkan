//! What one device- or constant-memory access of each AIR shape emits under the identity domains,
//! the runtime's, on a generated module.
//!
//! Each shape is two kernels: one that makes the access once and one that makes it twice, at
//! another offset, everything else equal. The second's entry weight (`ModuleStats::entry_weight`)
//! less the first's is what one access costs, its address arithmetic (a `getelementptr` and its
//! index's add) and the add that keeps its value live included; its device-address loads and
//! stores less the first's are the SPIR-V accesses it makes. The shapes: through a pointer loaded
//! from memory (a wide pointer) a `float`, `int`, `float2` and `float4` load and a `float4` store;
//! a pointer loaded at a run-time offset of a `constant` buffer of alignment 1 (reached from the
//! word below each access) and a `float` loaded through it; a `float4` load and store at a 4-byte
//! offset of a `uint` buffer (alignment 4 proven); and a `float4` load from a `float4` buffer
//! (alignment 16 proven).
//!
//! On every committed profile `Device::from_queries` derives under the identity domains, each
//! access of a leaf of whole words is one device-address load or store (`memory::leaf`'s whole
//! words; the pointer chase's two), and each shape's instructions stay within its bound. The
//! table of the first profile is printed.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::fmt::Write as _;

use super::engine_word_phases::{device_alignments, identity_profiles};
use crate::pipeline::testing::{compile_kept, link_and_validate, llvm_assemble, AssembleError};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// One access shape: its name, the instructions of its `j`th access (`%x{j}` its loaded value,
/// where it loads, of type `ty`), how the values are combined, and the most instructions one access
/// may cost.
struct Shape {
    name: &'static str,
    access: fn(u32) -> String,
    /// The loaded value's type and the instruction combining two of them (`None` for a store).
    value: Option<(&'static str, &'static str)>,
    /// The device-address loads and stores one access makes.
    accesses: u32,
    bound: u64,
}

/// The `i64` index `%te + j` as `%i{j}`.
fn index(j: u32) -> String {
    format!("  %i{j} = add i64 %te, {j}\n")
}

fn wide_load(ty: &str, align: u32, j: u32) -> String {
    format!(
        "{}  %g{j} = getelementptr {ty}, ptr addrspace(1) %p, i64 %i{j}\n  %x{j} = load {ty}, ptr addrspace(1) %g{j}, align {align}\n",
        index(j)
    )
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "wide float load",
        access: |j| wide_load("float", 4, j),
        value: Some(("float", "fadd")),
        accesses: 1,
        bound: 8,
    },
    Shape {
        name: "wide int load",
        access: |j| wide_load("i32", 4, j),
        value: Some(("i32", "add")),
        accesses: 1,
        bound: 8,
    },
    Shape {
        name: "wide float2 load",
        access: |j| wide_load("<2 x float>", 8, j),
        value: Some(("<2 x float>", "fadd")),
        accesses: 1,
        bound: 8,
    },
    Shape {
        name: "wide float4 load",
        access: |j| wide_load("<4 x float>", 16, j),
        value: Some(("<4 x float>", "fadd")),
        accesses: 1,
        bound: 8,
    },
    Shape {
        name: "wide float4 store",
        access: |j| {
            format!(
                "{}  %g{j} = getelementptr <4 x float>, ptr addrspace(1) %p, i64 %i{j}\n  store <4 x float> %v, ptr addrspace(1) %g{j}, align 16\n",
                index(j)
            )
        },
        value: None,
        accesses: 1,
        bound: 8,
    },
    Shape {
        name: "pointer chase from an align-1 constant buffer, then a float",
        access: |j| {
            format!(
                "{}  %o{j} = shl i64 %i{j}, 3\n  %c{j} = getelementptr i8, ptr addrspace(2) %2, i64 %o{j}\n  %q{j} = load ptr addrspace(1), ptr addrspace(2) %c{j}, align 8\n  %h{j} = getelementptr float, ptr addrspace(1) %q{j}, i64 %te\n  %x{j} = load float, ptr addrspace(1) %h{j}, align 4\n",
                index(j)
            )
        },
        value: Some(("float", "fadd")),
        accesses: 2,
        bound: 24,
    },
    Shape {
        name: "float4 load at a uint offset",
        access: |j| {
            format!(
                "{}  %g{j} = getelementptr i32, ptr addrspace(1) %0, i64 %i{j}\n  %x{j} = load <4 x float>, ptr addrspace(1) %g{j}, align 16\n",
                index(j)
            )
        },
        value: Some(("<4 x float>", "fadd")),
        accesses: 1,
        bound: 16,
    },
    Shape {
        name: "float4 store at a uint offset",
        access: |j| {
            format!(
                "{}  %g{j} = getelementptr i32, ptr addrspace(1) %0, i64 %i{j}\n  store <4 x float> %v, ptr addrspace(1) %g{j}, align 16\n",
                index(j)
            )
        },
        value: None,
        accesses: 1,
        bound: 16,
    },
    Shape {
        name: "float4 load at a float4 offset",
        access: |j| {
            format!(
                "{}  %g{j} = getelementptr <4 x float>, ptr addrspace(1) %1, i64 %i{j}\n  %x{j} = load <4 x float>, ptr addrspace(1) %g{j}, align 16\n",
                index(j)
            )
        },
        value: Some(("<4 x float>", "fadd")),
        accesses: 1,
        bound: 16,
    },
];

/// The kernel of `shape` making its access `n` times: its name and its text.
fn kernel(s: usize, shape: &Shape, n: u32) -> (String, String) {
    let name = format!("shape{s}_{n}");
    let mut body = String::from(
        "  %te = zext i32 %3 to i64\n  %p = load ptr addrspace(1), ptr addrspace(2) %2, align 8\n  %tf = uitofp i32 %3 to float\n  %w = insertelement <4 x float> poison, float %tf, i64 0\n  %v = shufflevector <4 x float> %w, <4 x float> poison, <4 x i32> zeroinitializer\n",
    );
    for j in 0..n {
        body.push_str(&(shape.access)(j));
    }
    if let Some((ty, op)) = shape.value {
        let mut acc = String::from("%x0");
        for j in 1..n {
            let _ = writeln!(body, "  %a{j} = {op} {ty} {acc}, %x{j}");
            acc = format!("%a{j}");
        }
        let _ = writeln!(
            body,
            "  %r = getelementptr <4 x float>, ptr addrspace(1) %1, i64 %te\n  store {ty} {acc}, ptr addrspace(1) %r, align 16"
        );
    }
    let text = format!(
        "define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, ptr addrspace(2) noundef %2, i32 noundef %3) {{\nentry:\n{body}  ret void\n}}\n\n"
    );
    (name, text)
}

/// Every shape's two kernels in one module, with their metadata: `uint *words` (0), `float4
/// *vecs` (1), `constant uchar *table` (2), the thread's position (3).
fn module() -> (String, Vec<(String, String)>) {
    let mut text = String::from(HEADER);
    let mut names = Vec::new();
    for (s, shape) in SHAPES.iter().enumerate() {
        let (one, a) = kernel(s, shape, 1);
        let (two, b) = kernel(s, shape, 2);
        text.push_str(&a);
        text.push_str(&b);
        names.push((one, two));
    }
    let kernels: Vec<String> = (0..names.len() * 2)
        .map(|k| format!("!{}", 100 + k))
        .collect();
    let _ = writeln!(
        text,
        "!air.kernel = !{{{}}}\n!air.version = !{{!1}}\n!air.language_version = !{{!2}}\n!1 = !{{i32 2, i32 8, i32 0}}\n!2 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n!3 = !{{}}\n!4 = !{{!5, !6, !7, !8}}\n\
         !5 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"words\"}}\n\
         !6 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 16, !\"air.arg_type_align_size\", i32 16, !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"vecs\"}}\n\
         !7 = !{{i32 2, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 1, !\"air.arg_type_align_size\", i32 1, !\"air.arg_type_name\", !\"uchar\", !\"air.arg_name\", !\"table\"}}\n\
         !8 = !{{i32 3, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}}",
        kernels.join(", ")
    );
    for (k, (one, two)) in names.iter().enumerate() {
        let _ = writeln!(text, "!{} = !{{ptr @{one}, !3, !4}}", 100 + 2 * k);
        let _ = writeln!(text, "!{} = !{{ptr @{two}, !3, !4}}", 101 + 2 * k);
    }
    (text, names)
}

/// The module assembled in a scratch directory of its own, removed once `llvm-as` is done.
fn assemble(text: &str) -> Option<Vec<u8>> {
    let dir = std::env::temp_dir().join(format!("m2v-access-size-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(e) => panic!("access size: {e:?}\n{text}"),
    }
}

/// Each shape's instructions and device-address accesses for one access, on profile `p`.
fn measure(
    bc: &[u8],
    names: &[(String, String)],
    stem: &str,
    p: &crate::device::DeviceProfile,
) -> Vec<(u64, u32)> {
    names
        .iter()
        .map(|(one, two)| {
            let [a, b] = [one, two].map(|n| {
                let (a, _, _) =
                    compile_kept(bc, n, p).unwrap_or_else(|e| panic!("{stem} {n}: {e}"));
                link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem} {n}: L3: {e}"));
                let accesses = u32::try_from(device_alignments(&a.module.words).len());
                (a.module.stats.entry_weight, accesses.expect("a count"))
            });
            (b.0.saturating_sub(a.0), b.1.saturating_sub(a.1))
        })
        .collect()
}

/// Every access of a leaf of whole words through a device address is one SPIR-V load or store,
/// and each shape's instructions stay within its bound, on every identity profile.
#[test]
fn a_device_access_of_whole_words_is_one_access() {
    let (text, names) = module();
    let Some(bc) = assemble(&text) else {
        return;
    };
    let profiles = identity_profiles();
    assert!(!profiles.is_empty());
    for (k, (stem, p)) in profiles.iter().enumerate() {
        let measured = measure(&bc, &names, stem, p);
        if k == 0 {
            eprintln!("one access on {stem}: instructions, device-address accesses");
            for (shape, (weight, accesses)) in SHAPES.iter().zip(&measured) {
                eprintln!("  {:<62} {weight:>4} {accesses:>2}", shape.name);
            }
        }
        for (shape, (weight, accesses)) in SHAPES.iter().zip(&measured) {
            assert_eq!(*accesses, shape.accesses, "{stem}: {}", shape.name);
            assert!(
                *weight <= shape.bound,
                "{stem}: {} costs {weight} instructions, over {}",
                shape.name,
                shape.bound
            );
        }
    }
}
