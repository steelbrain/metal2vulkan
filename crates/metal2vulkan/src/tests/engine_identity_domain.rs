//! The identity domains' contained domain is the bound buffer, as Metal's is: an access before the
//! bind offset reads the buffer's bytes there.
//!
//! Owned synthetic kernels, each storing one word it loads from `data` (buffer 0) to `out[0]`:
//! `run_words` loads `data[k]`, `k` the `int` in `constant` buffer 2, and `const_words`
//! loads `data[-1]`, over a `uint` buffer, which the identity domains reach from its own address
//! (the direct form); `run_bytes` and `const_bytes` load the `uint` at byte `k` and at byte -4
//! of a `uchar` buffer, which they reach from the word below each access (the word-below form).
//!
//! On the CPU interpreter, with `data` a 32-byte buffer of words `0xD0 + i`:
//! - under the identity domains, in both forms, a run-time -4 at bind offset 16 reads word 3 (byte
//!   12), and at bind offset 0, before the buffer's start, reads zero; a constant -4 at bind
//!   offset 16 reads word 3;
//! - on the resident-table domains the same kernels at bind offset 16 read word 3, as `docs/API.md`
//!   says a storage view of a whole window does (from the buffer's start).
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;

use super::engine_word_phases::{bounds_word, identity_profiles, slot_word};
use crate::device::DevicePolicy;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{compile_kept, profile, AssembleError};
use crate::spirv::interp::{run, Buffer, Config, Resources};

/// Where the interpreter holds `data`, `out`, `k` and the zero range.
const DATA: u64 = 0x1_0000_0000;
const OUT: u64 = 0x2_0000_1000;
const K: u64 = 0x3_0000_2000;
const ZERO: u64 = 0x7_0000_0000;
/// `data`'s words.
const WORDS: u32 = 8;

fn word(i: u32) -> u32 {
    0xD0 + i
}

/// The four kernels.
fn kernels() -> String {
    let kernel = |name: &str, elem: &str, align: u32, load: &str, run: bool| {
        let k = match run {
            true => "  %k = load i32, ptr addrspace(2) %2, align 4\n  %e = sext i32 %k to i64\n",
            false => "",
        };
        format!(
            "define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, ptr addrspace(2) noundef %2) {{\n\
             entry:\n{k}  %p = getelementptr inbounds {elem}, ptr addrspace(1) %0, i64 {load}\n  \
             %v = load i32, ptr addrspace(1) %p, align {align}\n  \
             store i32 %v, ptr addrspace(1) %1, align 4\n  ret void\n}}\n\n"
        )
    };
    let args = |data: &str, align: u32| {
        format!(
            "!{{!{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 {align}, !\"air.arg_type_align_size\", i32 {align}, !\"air.arg_type_name\", !\"{data}\", !\"air.arg_name\", !\"data\"}}, \
             !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}, \
             !{{i32 2, !\"air.buffer\", !\"air.buffer_size\", i32 4, !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"int\", !\"air.arg_name\", !\"k\"}}}}"
        )
    };
    let mut text = String::from(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
         target triple = \"air64_v28-apple-macosx26.0.0\"\n\n",
    );
    text += &kernel("run_words", "i32", 4, "%e", true);
    text += &kernel("const_words", "i32", 4, "-1", false);
    text += &kernel("run_bytes", "i8", 1, "%e", true);
    text += &kernel("const_bytes", "i8", 1, "-4", false);
    text +=
        "!air.kernel = !{!1, !2, !3, !4}\n!air.version = !{!20}\n!air.language_version = !{!21}\n";
    text += &format!("!1 = !{{ptr @run_words, !{{}}, {}}}\n", args("uint", 4));
    text += &format!("!2 = !{{ptr @const_words, !{{}}, {}}}\n", args("uint", 4));
    text += &format!("!3 = !{{ptr @run_bytes, !{{}}, {}}}\n", args("uchar", 1));
    text += &format!("!4 = !{{ptr @const_bytes, !{{}}, {}}}\n", args("uchar", 1));
    text += "!20 = !{i32 2, i32 8, i32 0}\n!21 = !{!\"Metal\", i32 4, i32 0, i32 0}\n";
    text
}

fn assemble() -> Option<Vec<u8>> {
    match metal2vulkan_devtools::llvm_assemble_text(&kernels()) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the kernels: {why}"),
    }
}

/// `data`'s 32 bytes.
fn data() -> Vec<u8> {
    (0..WORDS).flat_map(|i| word(i).to_le_bytes()).collect()
}

fn spec() -> BTreeMap<crate::pipeline::sym::SpecKey, u64> {
    use crate::pipeline::sym::{Axis, SpecKey};
    BTreeMap::from([
        (SpecKey::LocalSize(Axis::X), 1),
        (SpecKey::LocalSize(Axis::Y), 1),
        (SpecKey::LocalSize(Axis::Z), 1),
        (SpecKey::FlatLocalSize, 1),
    ])
}

/// What kernel `entry` stores under the identity domains on lavapipe, `data` bound at `bind`,
/// `k` the `int` it reads.
fn identity_run(bc: &[u8], entry: &str, bind: u64, k: i32) -> u32 {
    let (stem, p) = identity_profiles()
        .into_iter()
        .find(|(stem, _)| stem == "lavapipe")
        .expect("lavapipe");
    let (_, plan, module) = compile_kept(bc, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        let PushLoc::Push { offset } = f.loc else {
            panic!("{stem} {entry}: {f:?} spilled");
        };
        let o = offset as usize;
        match f.field {
            PushField::SlotGuestBase(s) => {
                let at = [DATA + bind, OUT, K][s.index as usize];
                push[o..o + 8].copy_from_slice(&slot_word(at).to_le_bytes());
            }
            PushField::SlotBounds(s) if s.index == 0 => {
                push[o..o + 8].copy_from_slice(&bounds_word(4 * WORDS as usize, bind))
            }
            PushField::ZeroAddress => push[o..o + 8].copy_from_slice(&ZERO.to_le_bytes()),
            _ => panic!("{stem} {entry}: unexpected push field {f:?}"),
        }
    }
    let buffer = |address: u64, bytes: Vec<u8>| Buffer {
        slot: None,
        address: Some(address),
        bytes,
    };
    let res = Resources {
        buffers: vec![
            buffer(DATA, data()),
            buffer(OUT, vec![0; 4]),
            buffer(K, k.to_le_bytes().to_vec()),
            buffer(ZERO, vec![0; 16]),
        ],
        push,
        spec: spec(),
    };
    let got =
        run(&module, &Config::default(), res).unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
    u32::from_le_bytes(got.buffers[1][..4].try_into().expect("a word"))
}

/// What kernel `entry` stores on the resident-table domains on lavapipe, `data` bound at `bind`
/// through a descriptor of the whole buffer, `k` the `int` it reads.
fn default_run(bc: &[u8], entry: &str, bind: u64, k: i32) -> u32 {
    let p = profile("lavapipe", DevicePolicy::RESIDENT_TABLES).unwrap_or_else(|e| panic!("{e}"));
    let (_, plan, module) = compile_kept(bc, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        let PushLoc::Push { offset } = f.loc else {
            panic!("{entry}: {f:?} spilled");
        };
        let o = offset as usize;
        match f.field {
            // The residual: the range starts at the buffer's start.
            PushField::SlotBindOffset(s) => {
                let residual = if s.index == 0 { bind as u32 } else { 0 };
                push[o..o + 4].copy_from_slice(&residual.to_le_bytes());
            }
            _ => panic!("{entry}: unexpected push field {f:?}"),
        }
    }
    let buffer = |index: u32, bytes: Vec<u8>| Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        }),
        address: None,
        bytes,
    };
    // A view's range is whole 16-byte rows, as `engine_access` binds its buffers.
    let mut k_row = k.to_le_bytes().to_vec();
    k_row.resize(16, 0);
    let res = Resources {
        buffers: vec![buffer(0, data()), buffer(1, vec![0; 16]), buffer(2, k_row)],
        push,
        spec: spec(),
    };
    let got = run(&module, &Config::default(), res).unwrap_or_else(|e| panic!("{entry}: {e}"));
    u32::from_le_bytes(got.buffers[1][..4].try_into().expect("a word"))
}

#[test]
fn an_access_before_the_bind_offset_reads_the_buffer_under_the_identity_domains() {
    let Some(bc) = assemble() else {
        return;
    };
    // Each form's run-time kernel and the `k` that reads 4 bytes before the bind offset.
    for (run, constant, k) in [
        ("run_words", "const_words", -1),
        ("run_bytes", "const_bytes", -4),
    ] {
        assert_eq!(
            identity_run(&bc, run, 16, k),
            word(3),
            "{run} at bind offset 16"
        );
        assert_eq!(identity_run(&bc, run, 0, k), 0, "{run} at bind offset 0");
        assert_eq!(
            identity_run(&bc, constant, 16, 0),
            word(3),
            "{constant} at bind offset 16"
        );
    }
}

#[test]
fn an_access_before_the_bind_offset_reads_the_buffer_on_the_default_domains() {
    let Some(bc) = assemble() else {
        return;
    };
    for (entry, k) in [
        ("run_words", -1),
        ("const_words", 0),
        ("run_bytes", -4),
        ("const_bytes", 0),
    ] {
        assert_eq!(default_run(&bc, entry, 16, k), word(3), "{entry}");
    }
}
