//! A render pipeline's functions spill past their shares of the device's push range under the
//! identity domains, as a kernel spills past its window.
//!
//! Owned synthetic functions: the vertex function `vmany` sums word `vid` of each of its `n`
//! `device` buffers into word `vid` of buffer `n`, and the fragment function `fmany` returns the
//! sum of word `x` of its two buffers, `x` its position's. Every access is at a run-time offset, so
//! each slot pushes its address and its bounds, and each function the zero range's address. `n` is
//! 24 on a device whose push window is 256 bytes and 9 on one of 128 ([`inputs`]): the most whose
//! addresses the vertex function's window holds beside its block's address and the zero range's.
//!
//! - On every committed profile `Device::from_queries` derives under the identity domains, the
//!   vertex function's fields pass its window, `[0, W - 8)` (`W` the device's window): it pushes its
//!   parameter block's address first, then every slot's address, and only the check's fields
//!   (`ZeroAddress`, the bounds) from the first that does not fit go to the block. Its module
//!   passes L3.
//! - The pair compiles on each such profile: the vertex function's fields end at `W - 8`, and the
//!   fragment function, from there, pushes its block's address in the range's last 8 bytes and
//!   reads the rest, its slots' addresses too, from the block. Both modules pass L3.
//! - On the CPU interpreter, at 256 bytes (`lavapipe`) and at 128 (`runtime-floor`), four vertices
//!   with each buffer four words long sum Metal's words where every slot's pushed buffer is 16
//!   bytes long; with the odd slots' pushed as 12 bytes long, in the window and in the block alike,
//!   vertex 3 reads zero from each of those slots. No access traps.
//!
//! `llvm-as` assembles the functions; without it the tests skip visibly.

use std::collections::BTreeMap;

use super::engine_word_phases::{bounds_word, identity_profiles, slot_word};
use crate::budget::Budget;
use crate::device::{Device, DevicePolicy, HandleDomain, PointerDomain};
use crate::library::Library;
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::interface::{PushLoc, PushPlacement};
use crate::pipeline::sym::{PushField, SlotClass, Stage};
use crate::pipeline::testing::{compile_kept, description, link_and_validate, AssembleError};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::testing::{check_words_valid, spirv_dis};

/// The vertex function's input buffers on a device whose push window is `bytes`: 24 at 256 and 9
/// at 128, so its window holds its block's address and every slot's address, and the check's
/// fields spill. Its output is buffer `inputs(bytes)`.
fn inputs(bytes: u32) -> u32 {
    match bytes {
        256 => 24,
        128 => 9,
        other => panic!("no fixture for a push window of {other} bytes"),
    }
}
/// Vertices drawn, and words in each buffer.
const VERTICES: u32 = 4;
/// Where the interpreter holds buffer `i` (`BUFFERS + i * STRIDE`), the parameter block and the
/// zero range.
const BUFFERS: u64 = 0x1_0000_0000;
const STRIDE: u64 = 0x1000;
const BLOCK: u64 = 0x5_0000_0000;
const ZERO: u64 = 0x6_0000_0000;

fn functions(inputs: u32) -> String {
    let n = inputs as usize;
    let mut params: Vec<String> = (0..n)
        .map(|i| format!("ptr addrspace(1) noundef \"air-buffer-no-alias\" %b{i}"))
        .collect();
    params.push("ptr addrspace(1) noundef \"air-buffer-no-alias\" %out".to_owned());
    params.push("i32 noundef %vid".to_owned());
    let mut body = String::from("  %e = zext i32 %vid to i64\n  %s0 = add i32 0, 0\n");
    for i in 0..n {
        body += &format!(
            "  %p{i} = getelementptr inbounds i32, ptr addrspace(1) %b{i}, i64 %e\n  \
             %v{i} = load i32, ptr addrspace(1) %p{i}, align 4\n  \
             %s{} = add i32 %s{i}, %v{i}\n",
            i + 1
        );
    }
    let buffer = |at: usize, access: &str, name: &str| {
        format!(
            "!{{i32 {at}, !\"air.buffer\", !\"air.location_index\", i32 {at}, i32 1, !\"{access}\", \
             !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, \
             !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", \
             !\"air.arg_name\", !\"{name}\"}}"
        )
    };
    let mut meta = Vec::new();
    let mut inputs = Vec::new();
    for i in 0..n {
        meta.push(format!(
            "!{} = {}",
            100 + i,
            buffer(i, "air.read", &format!("b{i}"))
        ));
        inputs.push(format!("!{}", 100 + i));
    }
    meta.push(format!(
        "!{} = {}",
        100 + n,
        buffer(n, "air.read_write", "out")
    ));
    inputs.push(format!("!{}", 100 + n));
    meta.push(format!(
        "!{} = !{{i32 {}, !\"air.vertex_id\", !\"air.arg_type_name\", !\"uint\", \
         !\"air.arg_name\", !\"vid\"}}",
        101 + n,
        n + 1
    ));
    inputs.push(format!("!{}", 101 + n));
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"
target triple = \"air64_v28-apple-macosx26.0.0\"

define <4 x float> @vmany({params}) {{
entry:
{body}  %q = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %e
  store i32 %s{n}, ptr addrspace(1) %q, align 4
  ret <4 x float> <float 0.0, float 0.0, float 0.0, float 1.0>
}}

define <4 x float> @fmany(<4 x float> %pos, ptr addrspace(1) noundef %c0, ptr addrspace(1) noundef %c1) {{
entry:
  %px = extractelement <4 x float> %pos, i64 0
  %x = fptoui float %px to i32
  %e = zext i32 %x to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %c0, i64 %e
  %a = load i32, ptr addrspace(1) %p, align 4
  %r = getelementptr inbounds i32, ptr addrspace(1) %c1, i64 %e
  %b = load i32, ptr addrspace(1) %r, align 4
  %s = add i32 %a, %b
  %f = uitofp i32 %s to float
  %c = insertelement <4 x float> <float 0.0, float 0.0, float 0.0, float 1.0>, float %f, i64 0
  ret <4 x float> %c
}}

!air.vertex = !{{!1}}
!air.fragment = !{{!5}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}
!1 = !{{ptr @vmany, !2, !4}}
!2 = !{{!3}}
!3 = !{{!\"air.position\", !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"position\"}}
!4 = !{{{inputs}}}
!5 = !{{ptr @fmany, !6, !8}}
!6 = !{{!7}}
!7 = !{{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", !\"float4\"}}
!8 = !{{!9, !10, !11}}
!9 = !{{i32 0, !\"air.position\", !\"air.center\", !\"air.no_perspective\", !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"pos\"}}
!10 = {c0}
!11 = {c1}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!\"Metal\", i32 4, i32 0, i32 0}}
{meta}
",
        params = params.join(", "),
        inputs = inputs.join(", "),
        c0 = buffer(1, "air.read", "c0"),
        c1 = buffer(2, "air.read", "c1"),
        meta = meta.join("\n"),
    )
}

fn assemble(inputs: u32) -> Option<Vec<u8>> {
    match metal2vulkan_devtools::llvm_assemble_text(&functions(inputs)) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the functions: {why}"),
    }
}

fn identity() -> DevicePolicy {
    DevicePolicy {
        pointer_domain: PointerDomain::Identity,
        handle_domain: HandleDomain::Identity,
        ..DevicePolicy::RESIDENT_TABLES
    }
}

/// The placement pushes its block's address first at `start`, and, where `addresses`, every slot's
/// address in the window and only the check's fields, from the first that does not fit, in the
/// block.
fn assert_spills(what: &str, p: &PushPlacement, start: u32, addresses: bool) {
    let first = p.fields.first().expect("fields");
    assert_eq!(
        (first.field, first.loc),
        (
            PushField::ArgumentTableAddress,
            PushLoc::Push { offset: start }
        ),
        "{what}: the block's address first"
    );
    assert!(p.argument_table_size > 0, "{what}: no block");
    if !addresses {
        return;
    }
    let spilled: Vec<PushField> = (p.fields.iter())
        .filter(|f| matches!(f.loc, PushLoc::ArgumentTable { .. }))
        .map(|f| f.field)
        .collect();
    assert!(
        spilled
            .iter()
            .all(|f| matches!(f, PushField::SlotBounds(_) | PushField::ZeroAddress)),
        "{what}: a field other than the check's spilled: {spilled:?}"
    );
    assert!(
        (p.fields.iter())
            .filter(|f| matches!(f.field, PushField::SlotGuestBase(_)))
            .all(|f| matches!(f.loc, PushLoc::Push { .. })),
        "{what}: a slot's address spilled"
    );
}

#[test]
fn a_vertex_function_past_its_share_spills_the_checks_fields_into_its_block() {
    for (stem, p) in identity_profiles() {
        let window = p.push_window();
        let Some(bc) = assemble(inputs(window)) else {
            return;
        };
        let (a, plan, _) = compile_kept(&bc, "vmany", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_spills(&stem, &plan.push, 0, true);
        let end = plan.push.push_range.map_or(0, |(o, s)| o + s);
        assert!(
            end <= window - 8,
            "{stem}: the vertex window of {window} ends at {end}"
        );
        println!("{stem}: push window {window}, the vertex function's fields end at {end}");
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let text = spirv_dis(&a.module.words);
        if text.starts_with('<') {
            eprintln!("SKIPPED: {text}");
            continue;
        }
        let lines = spilled_lines(&plan.push);
        assert_eq!(
            block_loads(&text),
            (lines, lines, lines),
            "{stem}: each 16-byte line of the spilled fields loaded once, in the function's \
             first block, as a uvec4 aligned 16 through a NonWritable block:\n{}",
            pointer_lines(&text)
        );
    }
}

/// The loads through a parameter block's address in disassembly `text`: a `ulong` loaded from the
/// push block (selected against null by the floor's tail's test where the function has the tail,
/// `lower::push::fold_never_pushed`), plus a constant (none for the block's first line, where the
/// sum may fold), made a
/// pointer to a one-member block (`OpConvertUToPtr`), whose member is loaded. Their number, how
/// many of them lie in a function's first block, and how many read a 16-byte line: a `uvec4`,
/// `Aligned 16`, through a block whose member is `NonWritable`.
fn block_loads(text: &str) -> (usize, usize, usize) {
    use std::collections::{BTreeMap, BTreeSet};
    let (mut push, mut chains, mut bases) = (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
    let (mut constants, mut blocks) = (BTreeSet::new(), BTreeSet::new());
    let (mut pointees, mut read_only, mut members) =
        (BTreeMap::new(), BTreeSet::new(), BTreeMap::new());
    let (mut loads, mut first, mut lines) = (0, 0, 0);
    let (mut opened, mut entry) = (false, false);
    for line in text.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words[..] {
            [_, "=", "OpFunction", ..] => opened = true,
            [_, "=", "OpLabel"] => {
                entry = opened;
                opened = false;
            }
            [id, "=", "OpTypePointer", "PhysicalStorageBuffer", pointee] => {
                pointees.insert(id, pointee);
            }
            ["OpMemberDecorate", block, "0", "NonWritable"] => {
                read_only.insert(block);
            }
            [id, "=", "OpVariable", _, "PushConstant"] => {
                push.insert(id);
            }
            [id, "=", "OpConstant", "%ulong", _] => {
                constants.insert(id);
            }
            [id, "=", "OpAccessChain" | "OpInBoundsAccessChain", _, var, ..]
                if push.contains(var) =>
            {
                chains.insert(id);
            }
            [id, "=", "OpLoad", "%ulong", chain, ..] if chains.contains(chain) => {
                bases.insert(id);
            }
            [id, "=", "OpSelect", "%ulong", _, _, base] if bases.contains(base) => {
                bases.insert(id);
            }
            [id, "=", "OpIAdd", "%ulong", a, b]
                if (bases.contains(a) && constants.contains(b))
                    || (constants.contains(a) && bases.contains(b)) =>
            {
                blocks.insert(id);
            }
            [id, "=", "OpConvertUToPtr", ty, at] if bases.contains(at) || blocks.contains(at) => {
                members.insert(id, pointees.get(ty).copied());
            }
            [id, "=", "OpAccessChain" | "OpInBoundsAccessChain", _, block, _]
                if members.contains_key(block) =>
            {
                let pointee = members[block];
                members.insert(id, pointee);
            }
            [_, "=", "OpLoad", ty, pointer, ref rest @ ..] if members.contains_key(pointer) => {
                loads += 1;
                first += usize::from(entry);
                let non_writable = members[pointer].is_some_and(|b| read_only.contains(b));
                lines += usize::from(ty == "%v4uint" && rest == ["Aligned", "16"] && non_writable);
            }
            _ => {}
        }
    }
    (loads, first, lines)
}

/// The 16-byte lines of the parameter block that placement `p`'s spilled fields lie in.
fn spilled_lines(p: &PushPlacement) -> usize {
    let mut lines = std::collections::BTreeSet::new();
    for f in &p.fields {
        if let PushLoc::ArgumentTable { offset } = f.loc {
            lines.extend(offset / 16..=(offset + f.size.max(1) - 1) / 16);
        }
    }
    lines.len()
}

/// The lines of disassembly `text` that read the push block or make a device pointer, for a
/// failure's message.
fn pointer_lines(text: &str) -> String {
    (text.lines())
        .filter(|l| {
            ["PushConstant", "OpConvertUToPtr", "OpIAdd %ulong"]
                .iter()
                .any(|k| l.contains(k))
        })
        .take(60)
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn a_render_pair_past_the_floor_spills_both_functions() {
    for (stem, profile) in identity_profiles() {
        let window = profile.push_window();
        let Some(bc) = assemble(inputs(window)) else {
            return;
        };
        let library = Library::from_air(&bc).expect("the library");
        let function = |name| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(function("vmany"));
        d.fragment = Some(function("fmany"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let desc = description(&stem).unwrap_or_else(|e| panic!("{e}"));
        let device = Device::from_queries(desc, identity()).unwrap_or_else(|e| panic!("{e}"));
        let p = (device.compile_render_pipeline(&d, &Budget::standard()))
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        let [vertex, fragment] = &p.interface.push[..] else {
            panic!("{stem}: two placements");
        };
        assert_eq!(
            (vertex.stage, fragment.stage),
            (Stage::Vertex, Stage::Fragment)
        );
        assert_spills(&format!("{stem} vertex"), vertex, 0, true);
        // The vertex function fills its window to `window - 8`: the fragment function's block
        // address takes the range's last 8 bytes, and its slots' addresses go to its block too.
        let base = vertex.push_range.map_or(0, |(o, s)| o + s);
        assert_eq!(base, window - 8, "{stem}");
        assert_spills(&format!("{stem} fragment"), fragment, base, false);
        let end = fragment.push_range.map_or(0, |(o, s)| o + s);
        assert!(end <= window, "{stem}: the range ends at {end}");
        println!(
            "{stem}: push window {window}, the vertex function ends at {base}, the pair at {end}"
        );
        for stage in &p.stages {
            check_words_valid(&stage.spirv, &p.requirements, device.profile())
                .unwrap_or_else(|e| panic!("{stem}: {:?}: {e}", stage.stage));
        }
    }
}

/// Buffer `i`'s words: `(i + 1) * 100 + v` at word `v`.
fn word(i: u32, v: u32) -> u32 {
    (i + 1) * 100 + v
}

/// One draw of `VERTICES` vertices of `vmany` on committed profile `stem`, each input slot's
/// pushed buffer `length(i)` bytes long: the output buffer's words.
fn draw(stem: &str, length: impl Fn(u32) -> usize) -> Vec<u32> {
    let (stem, p) = identity_profiles()
        .into_iter()
        .find(|(name, _)| name == stem)
        .unwrap_or_else(|| panic!("{stem}"));
    let n = inputs(p.push_window());
    let Some(bc) = assemble(n) else {
        return Vec::new();
    };
    let (_, plan, module) = compile_kept(&bc, "vmany", &p).unwrap_or_else(|e| panic!("{e}"));
    let bytes = 4 * VERTICES as usize;
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(o, s)| (o + s) as usize)];
    let mut block = vec![0u8; plan.push.argument_table_size as usize];
    for f in &plan.push.fields {
        let (to, o) = match f.loc {
            PushLoc::Push { offset } => (&mut push, offset as usize),
            PushLoc::ArgumentTable { offset } => (&mut block, offset as usize),
        };
        match f.field {
            PushField::ArgumentTableAddress => to[o..o + 8].copy_from_slice(&BLOCK.to_le_bytes()),
            PushField::SlotGuestBase(s) if s.class == SlotClass::Buffer => {
                let at = BUFFERS + u64::from(s.index) * STRIDE;
                to[o..o + 8].copy_from_slice(&slot_word(at).to_le_bytes());
            }
            PushField::SlotBounds(s) if s.index < n => {
                to[o..o + 8].copy_from_slice(&bounds_word(length(s.index), 0))
            }
            PushField::SlotBounds(_) => to[o..o + 8].copy_from_slice(&bounds_word(bytes, 0)),
            PushField::ZeroAddress => to[o..o + 8].copy_from_slice(&ZERO.to_le_bytes()),
            _ => panic!("{stem}: unexpected push field {f:?}"),
        }
    }
    let buffer = |address: u64, bytes: Vec<u8>| Buffer {
        slot: None,
        address: Some(address),
        bytes,
    };
    let mut buffers: Vec<Buffer> = (0..n)
        .map(|i| {
            let words = (0..VERTICES)
                .flat_map(|v| word(i, v).to_le_bytes())
                .collect();
            buffer(BUFFERS + u64::from(i) * STRIDE, words)
        })
        .collect();
    buffers.push(buffer(BUFFERS + u64::from(n) * STRIDE, vec![0; bytes]));
    buffers.push(buffer(BLOCK, block));
    buffers.push(buffer(ZERO, vec![0; 16]));
    let res = Resources {
        buffers,
        push,
        spec: BTreeMap::new(),
    };
    let cfg = Config {
        workgroups: [VERTICES, 1, 1],
        ..Config::default()
    };
    let got = run(&module, &cfg, res).unwrap_or_else(|e| panic!("{stem}: {e}"));
    assert_eq!(
        got.buffers[n as usize + 2],
        vec![0; 16],
        "{stem}: the zero range"
    );
    (got.buffers[n as usize].chunks(4))
        .map(|w| u32::from_le_bytes(w.try_into().expect("a word")))
        .collect()
}

#[test]
fn spilled_bounds_check_their_slot_as_pushed_ones_do() {
    for (stem, window) in [("lavapipe", 256), ("runtime-floor", 128)] {
        let n = inputs(window);
        let all = |v: u32| (0..n).map(|i| word(i, v)).sum::<u32>();
        let whole = draw(stem, |_| 4 * VERTICES as usize);
        if whole.is_empty() {
            return;
        }
        assert_eq!(whole, (0..VERTICES).map(all).collect::<Vec<_>>(), "{stem}");
        // The odd slots one word short: vertex 3 reads zero from each of them.
        let short = draw(stem, |i| 4 * (VERTICES as usize - (i as usize % 2)));
        let even = |v: u32| (0..n).step_by(2).map(|i| word(i, v)).sum::<u32>();
        let mut want: Vec<u32> = (0..VERTICES - 1).map(all).collect();
        want.push(even(VERTICES - 1));
        assert_eq!(short, want, "{stem}");
        println!("{stem}: push window {window}, {n} inputs, each vertex's sum");
    }
}
