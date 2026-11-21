//! The samplers a pipeline state numbers for a function, decided from the specialized entry's AIR
//! alone: `StageInterface::numbered_samplers`.
//!
//! Metal's pipeline creation gives each of these a plain sampler resource ID (the runtime's
//! `resource_ids.rs` numbers them as Metal does). Measured on an Apple M4 (macOS 27.0.1) and an
//! M3 Max (macOS 26.7), plain Metal, in two sets of probes (the M3's 108 rows of the first the
//! same):
//!
//! - **Constexpr samplers**, each once by state (two of equal state are one), in the reverse of
//!   the order the module's `!air.sampler_states` lists them (Apple's compiler lists them in the
//!   order of their first use): a kernel that samples `s` then `t` numbers `t`, then `s`, however
//!   it declares them. A sample, a gather, a comparison and a level query all use one, through a
//!   texture argument or an argument buffer's alike. A sampler argument, or an argument buffer's
//!   sampler, numbers nothing. The entry reaches only the constexpr samplers its specialized body
//!   uses (`Entry::static_samplers`, after pruning), so a function constant that removes a sample
//!   removes its sampler.
//! - **`Read`**, after them, where the body reads a texel of a texture (`TexOp::Read`): of every
//!   texture type measured (2D, 2D array, 3D, cube, depth, multisampled, texture buffer, `half`
//!   texels), through an argument, a `sample`-access argument, or an argument buffer, in a kernel,
//!   a vertex, a fragment or a tile function, or in a visible function it links. A texture
//!   argument the body never reads, a size query, a fence and an atomic number nothing; a
//!   function constant that removes the read removes it.
//! - **`Write`**, last, where the body writes a texel (`TexOp::Write`): 2D textures and texture
//!   buffers, in a kernel, a vertex or a fragment function, or a visible function it links. A
//!   write from an imageblock slice (`TexOp::WriteImageblockSlice`) numbers nothing.
//!
//! **Linked visible functions** (an M4 under macOS 27.0.1; sampler states differing only in their
//! reduction are distinct, sid13). The constexpr samplers of a visible function the stage links
//! (`MTLLinkedFunctions.functions` or `.privateFunctions`, called directly or through a
//! visible-function table; a render stage's own linked functions alike) are numbered after the
//! function's own constexpr samplers and before `Read`, each linked function's in the reverse of
//! its first use, as a module's own are (`Entry::linked_sampler_groups`, one group per function
//! linking imported). The stage's function and the state hold them, not the linked function's
//! library (sid13). A function of `functions` the stage never names is numbered too (sid14 and
//! sid15), and so is every constexpr sampler of one the stage calls where pruning removed the call
//! or the sample: Metal numbers a function a table can hold whole, called or not
//! (`LinkedSamplerGroup::states`, a group of its own for a function linking did not import). One of
//! `privateFunctions` is numbered as far as the stage reaches it, as the stage's own: not where the
//! stage never names it, as linking imports none, nor where a function constant removed its only
//! call. Where two or more linked functions bring constexpr samplers, Metal numbers them in an
//! order the pipeline's inputs do not decide: it followed neither the call, prototype, library or
//! array order, the names, nor the functions' hashes, and differed between two M4 minis for the
//! same program (sid15 to sid18). So the module numbers them in linking's order and records
//! [`PrecisionNote::LinkedSamplerNumbering`](crate::pipeline::PrecisionNote::LinkedSamplerNumbering)
//! (the runtime announces it once), as it does where a linked function may bring samplers the
//! groups lack (`Entry::unnumbered_linked_samplers`: an unimported function a table can hold
//! could not be planned), and where a dynamic library's symbol, a binary function, or a
//! function-pointer group's function brings them, whose numbering was not measured.

use crate::air;
use crate::air::intrinsics::{AirIntrinsic, TexOp};
use crate::air::ir::{Callee, InstKind};
use crate::pipeline::interface::NumberedSampler;

/// The samplers a pipeline state numbers for entry `e`, in Metal's order (see the module), and
/// whether Metal's numbering of them may differ from that order: two or more linked functions'
/// constexpr samplers, a dynamic library's, or (`unmeasured`) a binary function's or a
/// function-pointer group's, or a linked function's the entry cannot name
/// (`Entry::unnumbered_linked_samplers`).
pub(super) fn samplers(e: &air::Entry, unmeasured: bool) -> (Vec<NumberedSampler>, bool) {
    let mut numbered: Vec<NumberedSampler> = Vec::new();
    let mut number = |states: &mut dyn Iterator<Item = &air::abi::SamplerState>| {
        for state in states {
            let sampler = NumberedSampler::Constexpr(crate::lower::sampler::state_words(state));
            if !numbered.contains(&sampler) {
                numbered.push(sampler);
            }
        }
    };
    // The module's own, then each linked function's group, each reversed.
    let linked: usize = e.linked_sampler_groups.iter().map(|g| g.kept).sum();
    let (own, _) = e
        .static_samplers
        .split_at(e.static_samplers.len().saturating_sub(linked));
    number(&mut own.iter().rev().map(|(_, state)| state));
    for group in &e.linked_sampler_groups {
        number(&mut group.states.iter().rev());
    }
    let differs = e.linked_sampler_groups.len() > 1
        || e.linked_sampler_groups.iter().any(|g| g.library)
        || (unmeasured && !e.linked_sampler_groups.is_empty())
        || e.unnumbered_linked_samplers;
    let (mut reads, mut writes) = (false, false);
    let bodies = e
        .module
        .parts()
        .funcs
        .iter()
        .filter_map(|(_, f)| f.body.as_ref());
    for inst in bodies.flat_map(|body| body.insts.values()) {
        if let InstKind::Call(call) = &inst.kind {
            if let Callee::Air {
                op: AirIntrinsic::Texture(t),
                ..
            } = &call.callee
            {
                reads |= t.op == TexOp::Read;
                writes |= t.op == TexOp::Write;
            }
        }
    }
    numbered.extend(reads.then_some(NumberedSampler::Read));
    numbered.extend(writes.then_some(NumberedSampler::Write));
    (numbered, differs)
}

#[cfg(test)]
mod tests {
    use crate::air::abi::SamplerState;
    use crate::budget::SPIRV_WORDS;
    use crate::device::DevicePolicy;
    use crate::library::{FunctionConstantValue, FunctionConstantValues};
    use crate::lower::select::LowerInputs;
    use crate::metal::DataType;
    use crate::pipeline::compile::{compile_bitcode, EntryRef};
    use crate::pipeline::interface::NumberedSampler;
    use crate::pipeline::testing::{llvm_assemble, meter, profile, AssembleError};

    /// The words of a constexpr sampler clamped to its edge on every axis, of minification and
    /// magnification `filter` (0 nearest, 1 linear) and mip filter `mip` (0 none, 2 linear).
    const fn words(filter: u64, mip: u64) -> u64 {
        0x007b_ff00_0008_0000 | 1 | 1 << 3 | 1 << 6 | filter << 9 | filter << 11 | mip << 13
    }

    /// The module's constexpr samplers: `@v`'s state is `@s`'s.
    const SAMPLERS: [(&str, u64); 4] = [
        ("s", words(0, 0)),
        ("t", words(1, 0)),
        ("u", words(1, 2)),
        ("v", words(0, 0)),
    ];

    /// What a pipeline state numbers for the sampler `name` of [`SAMPLERS`].
    fn constexpr(name: &str) -> NumberedSampler {
        let (_, w) = SAMPLERS.iter().find(|(n, _)| *n == name).expect("sampler");
        let state = SamplerState::from_words([*w, 0]).expect("state");
        NumberedSampler::Constexpr(crate::lower::sampler::state_words(&state))
    }

    /// One access of the test kernel: through `%t` (a `sample` texture, slot 0), `%w` (a
    /// `read_write` texture, slot 1) or the sampler argument `%s` (slot 0).
    #[derive(Clone, Copy)]
    enum Op {
        /// `t.sample(x, ...)` through the constexpr sampler `@x`, or `%s` for `"%s"`.
        Sample(&'static str),
        /// `t.calculate_clamped_lod(x, ...)` through the constexpr sampler `@x`.
        Lod(&'static str),
        /// `w.read(...)`.
        Read,
        /// `w.write(...)` of what the kernel has summed so far.
        Write,
        /// `t.get_width()`.
        Width,
        /// `if (on) w.read(...)`, `on` the bool function constant at index 0. First, if at all.
        GatedRead,
        /// `if (on) t.sample(x, ...)`. First, if at all.
        GatedSample(&'static str),
    }

    const SAMPLE: &str = "call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2)";
    const SAMPLE_TAIL: &str = "<2 x float> zeroinitializer, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)";
    const READ: &str =
        "call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %w, ptr addrspace(2)";
    const READ_TAIL: &str = "<2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 3)";

    /// The sampler operand of `name`: the argument `%s`, or the constexpr sampler's global.
    fn operand(name: &str) -> String {
        if name.starts_with('%') {
            name.to_string()
        } else {
            format!("@{name}")
        }
    }

    /// Kernel `k` doing `ops` in turn and storing their sum to `out` (buffer 0), its module
    /// listing the constexpr samplers `states` in `!air.sampler_states` in that order.
    fn text(ops: &[Op], states: &[&str]) -> String {
        let mut body = String::new();
        let mut acc = "zeroinitializer".to_string();
        for (i, op) in ops.iter().enumerate() {
            let x = match *op {
                Op::Sample(s) => format!(
                    "  %r{i} = {SAMPLE} {}, {SAMPLE_TAIL}\n  %x{i} = extractvalue {{ <4 x float>, i8 }} %r{i}, 0\n",
                    operand(s)
                ),
                Op::Lod(s) => format!(
                    "  %l{i} = call float @air.calculate_clamped_lod_texture_2d(ptr addrspace(1) %t, ptr addrspace(2) {}, <2 x float> zeroinitializer, i32 0)\n  %x{i} = insertelement <4 x float> zeroinitializer, float %l{i}, i64 0\n",
                    operand(s)
                ),
                Op::Read => format!(
                    "  %q{i} = call ptr addrspace(2) @air.get_read_sampler()\n  %r{i} = {READ} %q{i}, {READ_TAIL}\n  %x{i} = extractvalue {{ <4 x float>, i8 }} %r{i}, 0\n"
                ),
                Op::Write => {
                    body.push_str(&format!(
                        "  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %w, <2 x i32> zeroinitializer, <4 x float> {acc}, i32 0, i32 3)\n"
                    ));
                    continue;
                }
                Op::Width => format!(
                    "  %n{i} = call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 0)\n  %u{i} = uitofp i32 %n{i} to float\n  %x{i} = insertelement <4 x float> zeroinitializer, float %u{i}, i64 0\n"
                ),
                Op::GatedRead | Op::GatedSample(_) => {
                    assert_eq!(i, 0, "a gated access comes first");
                    let access = match *op {
                        Op::GatedSample(s) => format!("%g = {SAMPLE} {}, {SAMPLE_TAIL}", operand(s)),
                        _ => format!(
                            "%q = call ptr addrspace(2) @air.get_read_sampler()\n  %g = {READ} %q, {READ_TAIL}"
                        ),
                    };
                    format!(
                        "  %f = load i8, ptr addrspace(2) @k.MTL_FC_INIT_0_b, align 1\n  %on = trunc i8 %f to i1\n  br i1 %on, label %gated, label %join\n\ngated:\n  {access}\n  %y = extractvalue {{ <4 x float>, i8 }} %g, 0\n  br label %join\n\njoin:\n  %x{i} = phi <4 x float> [ %y, %gated ], [ zeroinitializer, %entry ]\n"
                    )
                }
            };
            body.push_str(&x);
            body.push_str(&format!("  %a{i} = fadd <4 x float> {acc}, %x{i}\n"));
            acc = format!("%a{i}");
        }
        let globals: String = SAMPLERS
            .iter()
            .map(|(n, w)| {
                format!(
                    "@{n} = internal addrspace(2) constant [2 x i64] [i64 {w}, i64 0], align 8\n"
                )
            })
            .collect();
        let (list, nodes): (Vec<String>, String) = states
            .iter()
            .enumerate()
            .map(|(i, n)| {
                (
                    format!("!{}", 30 + i),
                    format!(
                        "!{} = !{{!\"air.sampler_state\", ptr addrspace(2) @{n}}}\n",
                        30 + i
                    ),
                )
            })
            .unzip();
        let states_md = if states.is_empty() {
            String::new()
        } else {
            format!("!air.sampler_states = !{{{}}}\n", list.join(", "))
        };
        format!(
            r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@k.MTL_FC_INIT_0_b = internal addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
{globals}
define void @k(ptr addrspace(1) %t, ptr addrspace(1) %w, ptr addrspace(2) %s, ptr addrspace(1) %out) {{
entry:
{body}  store <4 x float> {acc}, ptr addrspace(1) %out, align 16
  ret void
}}

declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare float @air.calculate_clamped_lod_texture_2d(ptr addrspace(1), ptr addrspace(2), <2 x float>, i32)
declare ptr addrspace(2) @air.get_read_sampler()
declare {{ <4 x float>, i8 }} @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)

!air.kernel = !{{!0}}
!air.function_constants = !{{!9}}
{states_md}!air.version = !{{!20}}
!air.language_version = !{{!21}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"w"}}
!5 = !{{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
!6 = !{{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!9 = !{{ptr addrspace(2) @k.MTL_FC_INIT_0_b, !"bool", !"on", i32 0}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!"Metal", i32 4, i32 0, i32 0}}
{nodes}"#
        )
    }

    /// What a pipeline state numbers for kernel `k` of `ops` (with `states` listed), compiled
    /// on lavapipe with the function constant `on`; `None` without `llvm-as`.
    fn numbered(name: &str, ops: &[Op], states: &[&str], on: bool) -> Option<Vec<NumberedSampler>> {
        let text = text(ops, states);
        let dir = std::env::temp_dir().join(format!(
            "m2v-numbered-{}-{:?}-{}",
            std::process::id(),
            std::thread::current().id(),
            name.replace(' ', "-")
        ));
        std::fs::create_dir_all(&dir).expect("scratch");
        let ll = dir.join("k.ll");
        std::fs::write(&ll, &text).expect("write");
        let bc = llvm_assemble(&ll);
        let _ = std::fs::remove_dir_all(&dir);
        let bc = match bc {
            Ok(b) => b,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return None;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}\n{text}"),
        };
        let values = FunctionConstantValues::defaults().with_index(
            0,
            FunctionConstantValue::new(DataType::Bool, [u8::from(on)]),
        );
        let p = profile("lavapipe", DevicePolicy::RESIDENT_TABLES).expect("profile");
        let a = compile_bitcode(
            &bc,
            EntryRef::Name("k"),
            &values,
            &p,
            &LowerInputs::new(),
            &mut meter(),
            SPIRV_WORDS,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}\n{text}"));
        Some(a.interface.numbered_samplers.clone())
    }

    /// The constexpr samplers come in the reverse of the order the module lists them, whatever
    /// order the body uses them in, each state once; then `Read`, then `Write`.
    #[test]
    fn constexpr_samplers_reverse_the_module_s_list_then_read_then_write() {
        use NumberedSampler as N;
        let cases: [(&str, &[Op], &[&str], Vec<NumberedSampler>); 7] = [
            (
                "listed stu",
                &[Op::Sample("s"), Op::Sample("t"), Op::Sample("u")],
                &["s", "t", "u"],
                vec![constexpr("u"), constexpr("t"), constexpr("s")],
            ),
            (
                "listed ust",
                &[Op::Sample("s"), Op::Sample("t"), Op::Sample("u")],
                &["u", "s", "t"],
                vec![constexpr("t"), constexpr("s"), constexpr("u")],
            ),
            (
                "equal states",
                &[Op::Sample("s"), Op::Sample("t"), Op::Sample("v")],
                &["s", "t", "v"],
                vec![constexpr("v"), constexpr("t")],
            ),
            (
                "read after a sample",
                &[Op::Sample("s"), Op::Read],
                &["s"],
                vec![constexpr("s"), N::Read],
            ),
            (
                "write, read, sample",
                &[Op::Write, Op::Read, Op::Sample("t")],
                &["t"],
                vec![constexpr("t"), N::Read, N::Write],
            ),
            ("write alone", &[Op::Write], &[], vec![N::Write]),
            ("level query", &[Op::Lod("u")], &["u"], vec![constexpr("u")]),
        ];
        for (name, ops, states, expected) in cases {
            let Some(got) = numbered(name, ops, states, false) else {
                return;
            };
            assert_eq!(got, expected, "{name}");
        }
    }

    /// A size query, a sample through a sampler argument, and a listed constexpr sampler the body
    /// never uses number nothing.
    #[test]
    fn queries_sampler_arguments_and_unused_samplers_number_nothing() {
        let cases: [(&str, &[Op], &[&str], Vec<NumberedSampler>); 3] = [
            ("width", &[Op::Width], &[], vec![]),
            (
                "sampler argument",
                &[Op::Sample("%s"), Op::Width],
                &[],
                vec![],
            ),
            (
                "listed unused",
                &[Op::Sample("s")],
                &["s", "t"],
                vec![constexpr("s")],
            ),
        ];
        for (name, ops, states, expected) in cases {
            let Some(got) = numbered(name, ops, states, false) else {
                return;
            };
            assert_eq!(got, expected, "{name}");
        }
    }

    /// A function constant that removes a read or a sample removes what it numbers.
    #[test]
    fn a_function_constant_removes_what_its_access_numbers() {
        for (on, read, sample) in [
            (true, vec![NumberedSampler::Read], vec![constexpr("t")]),
            (false, vec![], vec![]),
        ] {
            let Some(got) = numbered("gated read", &[Op::GatedRead], &[], on) else {
                return;
            };
            assert_eq!(got, read, "gated read, on = {on}");
            let Some(got) = numbered("gated sample", &[Op::GatedSample("t")], &["t"], on) else {
                return;
            };
            assert_eq!(got, sample, "gated sample, on = {on}");
        }
    }
}
