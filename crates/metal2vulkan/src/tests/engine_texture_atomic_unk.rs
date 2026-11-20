//! The texture atomics the decoder refused previously end to end: family C, over the owned fixture
//! `unk_atomic_ops` and small kernels, on every committed profile that compiles them (and
//! `moltenvk-m3` and `lavapipe` without robustness). Every module links and passes L3 (`G-val`).
//! The CPU interpreter does not model image texel pointers, so the tests check the emitted
//! instructions:
//!
//! - every operation (`fetch_{add,sub,and,or,xor,min,max}`, `exchange`, `load`, `store`,
//!   `compare_exchange_weak`) on 1D, 1D array, 2D array, 3D, buffer, and 2D (16-bit coordinate)
//!   textures is one SPIR-V atomic on an `OpImageTexelPointer` of an `R32ui` / `R32i` image, the
//!   minimum and maximum signed by the texel, a store an exchange whose old texel is unused (no
//!   `OpAtomicStore`, whose SPIRV-Cross MSL does not compile);
//! - a compare-exchange reads lane 0 of `expected`, compares it with the old texel, tests the
//!   texel against the image's size on every profile (Metal's out-of-bounds exchange fails and
//!   leaves `expected`), and writes lane 0 back;
//! - a result is Metal's `(old, 0, 0, 1)` and a compare-exchange writes lane 0 of `expected`
//!   only (measured), so results and `expected` read past lane 0 lower;
//! - a 2D, 2D array or 3D atomic in the legacy form, without the offset position (the AIR
//!   macOS 26.6's Metal on x86-64 hands its device), compiles as the same atomic with a zero
//!   offset.
//!
//! Previously every kernel but the 2D `fetch_max` was refused `UnknownIntrinsic`, so these tests
//! fail without it. A profile without a format's atomic feature refuses with the typed device gap,
//! which the tests accept and count.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, Needs};
use crate::library::FunctionArtifact;
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, AssembleError,
};
use crate::spirv::testing::committed_profiles;

const OP_TYPE_INT: u16 = 21;
const OP_TYPE_IMAGE: u16 = 25;
const OP_CONSTANT: u16 = 43;
const OP_COMPOSITE_CONSTRUCT: u16 = 80;
const OP_IMAGE_TEXEL_POINTER: u16 = 60;
const OP_IMAGE_QUERY_SIZE: u16 = 104;
const OP_I_EQUAL: u16 = 170;
const OP_ATOMIC_LOAD: u16 = 227;
const OP_ATOMIC_STORE: u16 = 228;
const OP_ATOMIC_EXCHANGE: u16 = 229;
const OP_ATOMIC_COMPARE_EXCHANGE: u16 = 230;
const OP_ATOMIC_I_ADD: u16 = 234;
const OP_ATOMIC_I_SUB: u16 = 235;
const OP_ATOMIC_S_MIN: u16 = 236;
const OP_ATOMIC_U_MIN: u16 = 237;
const OP_ATOMIC_S_MAX: u16 = 238;
const OP_ATOMIC_U_MAX: u16 = 239;
const OP_ATOMIC_AND: u16 = 240;
const OP_ATOMIC_OR: u16 = 241;
const OP_ATOMIC_XOR: u16 = 242;
const FORMAT_R32I: u32 = 24;
const FORMAT_R32UI: u32 = 33;

const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

fn fixture(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
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
    let text = format!(
        "target datalayout = \"{DATALAYOUT}\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n{body}"
    );
    let dir = std::env::temp_dir().join(format!(
        "m2v-texture-atomic-unk-{}-{}-{name}",
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

/// `stem` with the robustness features removed.
fn unrobust(stem: &str) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    for f in [
        VkFeature::RobustImageAccess,
        VkFeature::RobustImageAccess2,
        VkFeature::RobustBufferAccess,
        VkFeature::RobustBufferAccess2,
    ] {
        desc.features.remove(&f);
    }
    Device::from_queries(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// Every committed profile, and `moltenvk-m3` and `lavapipe` without robustness.
fn profiles() -> Vec<(String, DeviceProfile)> {
    let mut out = committed_profiles().expect("profiles");
    for stem in ["moltenvk-m3", "lavapipe"] {
        out.push((format!("{stem} unrobust"), unrobust(stem)));
    }
    out
}

/// The artifact of `entry` on `p`, linked and past L3; `None` when `p` lacks a capability it
/// needs (a typed device gap). Any other refusal fails the test.
fn artifact(
    bc: &[u8],
    entry: &str,
    stem: &str,
    p: &DeviceProfile,
) -> Option<std::sync::Arc<FunctionArtifact>> {
    match compile(bc, entry, p) {
        Ok(a) => {
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: {entry}: L3: {e}"));
            Some(a)
        }
        Err(Error::Unsupported(u)) if matches!(u.needs, Needs::AnyOf(_)) => None,
        Err(e) => panic!("{stem}: {entry}: {e}"),
    }
}

/// The instructions of a SPIR-V module: opcode and its words.
fn insts(words: &[u32]) -> Vec<(u16, &[u32])> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = ((words[i] >> 16) as usize).max(1);
        out.push((
            (words[i] & 0xFFFF) as u16,
            &words[i..(i + n).min(words.len())],
        ));
        i += n;
    }
    out
}

fn count(a: &FunctionArtifact, op: u16) -> usize {
    insts(&a.module.words)
        .into_iter()
        .filter(|(o, _)| *o == op)
        .count()
}

/// The `Image Format` operands of the module's `OpTypeImage`s, sorted.
fn image_formats(a: &FunctionArtifact) -> Vec<u32> {
    let mut out: Vec<u32> = insts(&a.module.words)
        .into_iter()
        .filter(|(o, _)| *o == OP_TYPE_IMAGE)
        .filter_map(|(_, w)| w.get(8).copied())
        .collect();
    out.sort();
    out
}

fn uses(a: &FunctionArtifact, name: &str) -> u32 {
    a.strategies.entries.get(name).map_or(0, |u| u.uses)
}

/// The uses of `texture.atomic_<op>`, guarded or not.
fn op_uses(a: &FunctionArtifact, op: &str) -> u32 {
    uses(a, &format!("texture.atomic_{op}")) + uses(a, &format!("texture.atomic_{op}_guarded"))
}

#[test]
fn every_texture_atomic_operation_is_one_image_atomic() {
    let Some(bc) = fixture("unk_atomic_ops") else {
        return;
    };
    let (mut compiled, mut gaps) = (0, 0);
    for (stem, p) in profiles() {
        let Some(a) = artifact(&bc, "unk_atomic_ops", &stem, &p) else {
            gaps += 1;
            continue;
        };
        compiled += 1;
        for (op, n) in [
            (OP_ATOMIC_I_ADD, 5),
            (OP_ATOMIC_I_SUB, 2),
            (OP_ATOMIC_U_MIN, 2),
            (OP_ATOMIC_S_MIN, 2),
            (OP_ATOMIC_U_MAX, 3),
            (OP_ATOMIC_S_MAX, 1),
            (OP_ATOMIC_AND, 2),
            (OP_ATOMIC_OR, 3),
            (OP_ATOMIC_XOR, 2),
            // Three exchanges and two stores.
            (OP_ATOMIC_EXCHANGE, 5),
            (OP_ATOMIC_COMPARE_EXCHANGE, 4),
            (OP_ATOMIC_LOAD, 7),
            (OP_ATOMIC_STORE, 0),
            (OP_IMAGE_TEXEL_POINTER, 38),
        ] {
            assert_eq!(count(&a, op), n, "{stem}: opcode {op}");
        }
        // Three `uint` and three `int` textures, each `R32` of its type.
        assert_eq!(
            image_formats(&a),
            [
                FORMAT_R32I,
                FORMAT_R32I,
                FORMAT_R32I,
                FORMAT_R32UI,
                FORMAT_R32UI,
                FORMAT_R32UI
            ],
            "{stem}"
        );
        for (op, n) in [
            ("add", 5),
            ("sub", 2),
            ("and", 2),
            ("or", 3),
            ("xor", 2),
            ("min", 4),
            ("max", 4),
            ("exchange", 3),
            ("load", 7),
            ("store", 2),
            ("compare_exchange", 4),
        ] {
            assert_eq!(op_uses(&a, op), n, "{stem}: texture.atomic_{op}");
        }
    }
    assert!(compiled >= 4, "compiled {compiled}, gaps {gaps}");
}

/// A kernel compare-exchanging texel `gid` of a `texture1d<uint, read_write>` (slot 0) with
/// `expected` an `alloca` holding (1, 2, 3, 4) and the desired texel 9, then running `tail` (which
/// may read `%e` and store to `uint *out`, buffer 0).
fn exchange_kernel(tail: &str) -> String {
    let name = "air.atomic_compare_exchange_weak_explicit_texture_1d.u.v4i32";
    format!(
        r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, i32 %gid) {{
  %e = alloca <4 x i32>, align 16
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %e, align 16
  %ok = call i1 @{name}(ptr addrspace(1) %t, i32 %gid, ptr %e, <4 x i32> <i32 9, i32 9, i32 9, i32 9>, i32 0, i32 0, i32 3)
  %okw = zext i1 %ok to i32
  store i32 %okw, ptr addrspace(1) %out, align 4
  {tail}
  ret void
}}

declare i1 @{name}(ptr addrspace(1), i32, ptr, <4 x i32>, i32, i32, i32)

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture1d<uint, read_write>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
"#
    )
}

const SECOND: &str = "%p1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1";

/// A compare-exchange whose `expected` is read only at lane 0 (a 32-bit load, or a vector load read
/// at lane 0) is `OpAtomicCompareExchange` with the old texel compared with the comparator, behind
/// the image-size test on every profile, robust or not.
#[test]
fn compare_exchange_reads_and_writes_lane_zero_of_expected() {
    let word = format!(
        "{SECOND}\n  %x = load i32, ptr %e, align 16\n  store i32 %x, ptr addrspace(1) %p1, align 4"
    );
    let vector = format!(
        "{SECOND}\n  %v = load <4 x i32>, ptr %e, align 16\n  %x = extractelement <4 x i32> %v, i64 0\n  store i32 %x, ptr addrspace(1) %p1, align 4"
    );
    let mut compiled = 0;
    for (what, tail) in [
        ("word", word),
        ("vector", vector),
        ("unread", String::new()),
    ] {
        let Some(bc) = assemble(what, &exchange_kernel(&tail)) else {
            return;
        };
        for (stem, p) in profiles() {
            let Some(a) = artifact(&bc, "k", &stem, &p) else {
                continue;
            };
            compiled += 1;
            let w = format!("{stem} {what}");
            assert_eq!(count(&a, OP_ATOMIC_COMPARE_EXCHANGE), 1, "{w}");
            assert_eq!(count(&a, OP_IMAGE_TEXEL_POINTER), 1, "{w}");
            assert!(count(&a, OP_I_EQUAL) >= 1, "{w}");
            assert!(
                count(&a, OP_IMAGE_QUERY_SIZE) >= 1,
                "{w}: no in-bounds test"
            );
            assert_eq!(op_uses(&a, "compare_exchange"), 1, "{w}");
            assert_eq!(image_formats(&a), [FORMAT_R32UI], "{w}");
        }
    }
    assert!(compiled >= 12, "{compiled}");
}

/// The `OpCompositeConstruct`s of four `u32` lanes whose last three are the constants 0, 0, 1:
/// a texture atomic's result, Metal's `(old, 0, 0, 1)`.
fn metal_lanes(a: &FunctionArtifact) -> usize {
    let all = insts(&a.module.words);
    let u32_types: Vec<u32> = all
        .iter()
        .filter(|(o, w)| *o == OP_TYPE_INT && w.get(2) == Some(&32) && w.get(3) == Some(&0))
        .filter_map(|(_, w)| w.get(1).copied())
        .collect();
    let constant = |value: u32| -> Vec<u32> {
        all.iter()
            .filter(|(o, w)| {
                *o == OP_CONSTANT
                    && w.get(1).is_some_and(|t| u32_types.contains(t))
                    && w.get(3) == Some(&value)
            })
            .filter_map(|(_, w)| w.get(2).copied())
            .collect()
    };
    let (zero, one) = (constant(0), constant(1));
    all.iter()
        .filter(|(o, w)| {
            *o == OP_COMPOSITE_CONSTRUCT
                && w.len() == 7
                && zero.contains(&w[4])
                && zero.contains(&w[5])
                && one.contains(&w[6])
        })
        .count()
}

/// Metal's texture atomics return `(old, 0, 0, 1)` and a compare-exchange writes lane 0 of
/// `expected` only, inside the texture and outside (authored cases `be2717e3`, `4d76953a`), so a
/// result read past lane 0 and an `expected` read there lower: a compare-exchange whose
/// `expected` is read at lane 1, or at lane 2 through a `getelementptr`, and the lane probe's
/// kernel, every result and `expected` read whole, its seven texel results each
/// `(old, 0, 0, 1)`, on every profile that compiles them.
#[test]
fn every_lane_of_results_and_expected_lowers() {
    let lane1 = format!(
        "{SECOND}\n  %v = load <4 x i32>, ptr %e, align 16\n  %x = extractelement <4 x i32> %v, i64 1\n  store i32 %x, ptr addrspace(1) %p1, align 4"
    );
    let gep = format!(
        "{SECOND}\n  %q = getelementptr inbounds i32, ptr %e, i64 2\n  %x = load i32, ptr %q, align 8\n  store i32 %x, ptr addrspace(1) %p1, align 4"
    );
    let mut compiled = 0;
    for (what, tail) in [("lane 1", lane1), ("lane 2 through a pointer", gep)] {
        let Some(bc) = assemble(what, &exchange_kernel(&tail)) else {
            return;
        };
        for (stem, p) in profiles() {
            let Some(a) = artifact(&bc, "k", &stem, &p) else {
                continue;
            };
            compiled += 1;
            assert_eq!(count(&a, OP_ATOMIC_COMPARE_EXCHANGE), 1, "{stem} {what}");
        }
    }
    let Some(bc) = fixture("unk_atomic_lanes") else {
        return;
    };
    for (stem, p) in profiles() {
        let Some(a) = artifact(&bc, "unk_atomic_lanes", &stem, &p) else {
            continue;
        };
        compiled += 1;
        assert_eq!(count(&a, OP_IMAGE_TEXEL_POINTER), 10, "{stem}");
        assert_eq!(count(&a, OP_ATOMIC_COMPARE_EXCHANGE), 2, "{stem}");
        assert_eq!(metal_lanes(&a), 7, "{stem}");
    }
    assert!(compiled >= 18, "{compiled}");
}

/// The arguments of an argument or parameter list, split at the commas outside `()` and `<>`.
fn split_args(list: &str) -> Vec<&str> {
    let (mut out, mut depth, mut start) = (Vec::new(), 0i32, 0);
    for (i, c) in list.char_indices() {
        match c {
            '(' | '<' => depth += 1,
            ')' | '>' => depth -= 1,
            ',' if depth == 0 => {
                out.push(list[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(list[start..].trim());
    out
}

/// `line` in the legacy atomic form when it declares or calls a 2D, 2D array or 3D texture
/// atomic: the offset position dropped, with whether the line is the declaration. The dropped
/// argument is checked to be the zero offset vector.
fn legacy_atomic_line(line: &str) -> Option<(String, bool)> {
    let at = line.find("@air.atomic_")?;
    let name_len = line[at..].find('(')?;
    let name = &line[at + "@air.atomic_".len()..at + name_len];
    let (op, rest) = name.split_once("_explicit_texture_")?;
    let shape = rest.split('.').next()?;
    if !matches!(shape, "2d" | "2d_array" | "3d") {
        return None;
    }
    // The offset follows the texture, a store's value, the coordinate and the layer.
    let offset = 2 + usize::from(op == "store") + usize::from(shape == "2d_array");
    let open = at + name_len;
    let mut depth = 0;
    let close = line[open..].char_indices().find_map(|(i, c)| {
        depth += match c {
            '(' => 1,
            ')' => -1,
            _ => 0,
        };
        (depth == 0).then_some(open + i)
    })?;
    let mut args = split_args(&line[open + 1..close]);
    let dropped = args.remove(offset);
    let declaration = line.starts_with("declare ");
    assert!(
        dropped.starts_with("<2 x i") || dropped.starts_with("<3 x i"),
        "{name}: position {offset} is {dropped:?}, not an offset"
    );
    assert!(
        declaration || dropped.ends_with("zeroinitializer"),
        "{name}: the offset {dropped:?} is not zero"
    );
    let rewritten = format!(
        "{}({}){}",
        &line[..open],
        args.join(", "),
        &line[close + 1..]
    );
    Some((rewritten, declaration))
}

/// `text` with every 2D, 2D array and 3D texture atomic in the legacy form
/// ([`legacy_atomic_line`]), and the number of declarations and calls rewritten.
fn legacy_atomics(text: &str) -> (String, usize, usize) {
    let (mut decls, mut calls) = (0, 0);
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        match legacy_atomic_line(line) {
            Some((rewritten, declaration)) => {
                if declaration {
                    decls += 1;
                } else {
                    calls += 1;
                }
                out.push_str(&rewritten);
            }
            None => out.push_str(line),
        }
        out.push('\n');
    }
    (out, decls, calls)
}

/// A 2D, 2D array or 3D texture atomic in the legacy form, without the offset position (the AIR
/// macOS 26.6's Metal on x86-64 hands its device), compiles as the same atomic with a zero
/// offset: the owned atomic fixtures (every operation, `uint` and `int`, 16-bit coordinates,
/// compare-exchange, store, `ulong` `min` and `max`) with every offset dropped compile on the same
/// profiles, with the same strategies and the same image atomics. Without the legacy form such a
/// declaration matches no layout of its name and is refused `UnknownIntrinsic`.
#[test]
fn legacy_texture_atomics_without_the_offset_compile_alike() {
    use metal2vulkan_devtools::llvm_assemble_text;
    let mut compiled = 0;
    for (stem, entry) in [
        ("unk_atomic_ops", "unk_atomic_ops"),
        ("unk_atomic_lanes", "unk_atomic_lanes"),
        ("unk_ulong_atomics", "unk_ulong_atomics"),
        ("trh_texture_atomic", "trh_texture_atomic"),
    ] {
        let Some(current) = fixture(stem) else {
            return;
        };
        let text = std::fs::read_to_string(engine_dir().join(format!("{stem}.ll"))).expect("read");
        let (legacy_text, decls, calls) = legacy_atomics(&text);
        assert!(
            decls > 0 && calls >= decls,
            "{stem}: {decls} declarations, {calls} calls"
        );
        let legacy = match llvm_assemble_text(&legacy_text) {
            Ok(b) => b,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected legacy {stem}: {why}"),
        };
        for (profile, p) in profiles() {
            let what = format!("{profile}: legacy {stem}");
            let a = artifact(&current, entry, &profile, &p);
            let b = artifact(&legacy, entry, &profile, &p);
            let (Some(a), Some(b)) = (a.as_ref(), b.as_ref()) else {
                assert_eq!(a.is_some(), b.is_some(), "{what}");
                continue;
            };
            compiled += 1;
            let uses = |x: &FunctionArtifact| {
                x.strategies
                    .entries
                    .iter()
                    .map(|(k, u)| (k.clone(), u.uses))
                    .collect::<Vec<_>>()
            };
            assert_eq!(uses(a), uses(b), "{what}: strategies");
            for op in [
                OP_IMAGE_TEXEL_POINTER,
                OP_ATOMIC_LOAD,
                OP_ATOMIC_EXCHANGE,
                OP_ATOMIC_COMPARE_EXCHANGE,
                OP_ATOMIC_I_ADD,
                OP_ATOMIC_I_SUB,
                OP_ATOMIC_S_MIN,
                OP_ATOMIC_U_MIN,
                OP_ATOMIC_S_MAX,
                OP_ATOMIC_U_MAX,
                OP_ATOMIC_AND,
                OP_ATOMIC_OR,
                OP_ATOMIC_XOR,
            ] {
                assert_eq!(count(a, op), count(b, op), "{what}: opcode {op}");
            }
            assert_eq!(image_formats(a), image_formats(b), "{what}");
        }
    }
    assert!(compiled >= 12, "{compiled}");
}
