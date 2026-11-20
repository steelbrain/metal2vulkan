//! A name must not decide a translation: the engine's guard of AGENTS.md's rule "structure and
//! semantics over names".
//!
//! Decide emit and lowering from IR structure (types, storage classes, access chains, AIR metadata
//! ABI) and never from a function, type, or variable identifier. The only names the engine may
//! dispatch on are the AIR/LLVM contract's: the `air.` and `llvm.` families. A name-keyed branch
//! that green-lights one shader while failing an identically shaped one is a silent defect: the
//! shader that was tested keeps working, so nothing fails until a consumer ships a differently
//! spelled shader.
//!
//! The check is metamorphic. Rewrite every identifier outside the contract (global symbols, named
//! struct, class, and union types, named local values and labels) to a generated placeholder,
//! assemble both texts (`llvm-as`), and compile every entry point of both libraries on two
//! committed profiles: one that reaches buffers by device address and one that binds them
//! (`vk12-floor`, Profile F). Structure and semantics are untouched, so each entry point must
//! compile to the same final SPIR-V words or be refused with the same typed outcome. The engine
//! writes no debug names and names every SPIR-V entry point `main`, so the words are compared
//! whole. A kernel is compiled into a compute pipeline; any other stage is linked alone and
//! relocated, as `src/tests/golden_hashes.rs` does.
//!
//! The authored inputs below spell every user identifier as a near-miss of a name the engine does
//! dispatch on, so a branch that matched loosely (`contains("texture_buffer")`,
//! `starts_with("air")`, a bare `_cube_array`) is caught here rather than by a consumer. The sweep
//! covers every committed opaque-pointer fixture (`validation/fixtures/public` and its
//! `engine/` directory); real AIR leaves most values numbered, so each fixture perturbs a few
//! names, but the sweep grows with the fixtures.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::Error;
use crate::library::{FunctionStage, Library};
use crate::pipeline::testing::{
    engine_dir, fixtures_dir, link_standalone, llvm_assemble, AssembleError,
};
use crate::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, LoweringInputs, StageSource};
use crate::Budget;

// ---- The renamer ----

/// Name prefixes that are the contract, and therefore are not perturbed:
///
/// - `air.` / `llvm.`: the AIR and LLVM intrinsic families AGENTS.md names.
/// - `__tensorops_impl_`: MetalPerformancePrimitives' `extern "C"` functions, which the Metal
///   front end declares in section `air.externally_defined` and Metal defines when it creates the
///   pipeline (`air::tensor_ops`). The engine reads them by the declaration's section and the
///   symbol's grammar, as it reads an `air.` intrinsic.
/// - `___metal_`: Metal's runtime routines (`___metal_frexp_*`, `___metal_ilogb_*`), which Apple's
///   compiler calls and Metal's backend defines when it creates the pipeline. Their declarations
///   carry no section or metadata, so the symbol is the ABI: an identifier in the implementation's
///   reserved namespace (three leading underscores), read by its grammar and only where the
///   declaration's signature is the routine's, as an `air.` intrinsic is.
///
/// The engine dispatches on no other name, so every other identifier is renamed, including
/// Metal's spellings that its metadata already states (`MTL_FC_INIT_*` function-constant globals,
/// `.MTL_VISIBLE_FN_REF` stubs, `__air_sampler_state`, `_GLOBAL__sub_I_*` initializers).
const CONTRACT_NAME_PREFIXES: &[&str] = &["air.", "llvm.", "__tensorops_impl_", "___metal_"];

/// Placeholder prefix for renamed globals. Distinctive enough that an input cannot already contain
/// it, which [`assert_name_independent`] checks rather than assumes.
const RENAMED_SYMBOL: &str = "renamed.symbol.";
/// Placeholder prefix for renamed named types.
const RENAMED_TYPE: &str = "renamed.type.";
/// Placeholder prefix for renamed named local values and labels.
const RENAMED_LOCAL: &str = "renamed.local.";

fn owned_by_contract(name: &str) -> bool {
    CONTRACT_NAME_PREFIXES
        .iter()
        .any(|prefix| name.starts_with(prefix))
}

/// The identifier token starting at `start` (just past its `@` or `%` sigil): either a quoted name
/// or a run of LLVM's unquoted identifier characters. Returns the name and the index just past the
/// whole token.
fn identifier_at(source: &str, start: usize) -> Option<(String, usize)> {
    let bytes = source.as_bytes();
    if bytes.get(start) == Some(&b'"') {
        let mut index = start + 1;
        while index < bytes.len() {
            match bytes[index] {
                b'\\' => index += 2,
                b'"' => return Some((source.get(start + 1..index)?.to_string(), index + 1)),
                _ => index += 1,
            }
        }
        return None;
    }
    let mut index = start;
    while index < bytes.len()
        && matches!(bytes[index], b'-' | b'$' | b'.' | b'_' | b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z')
    {
        index += 1;
    }
    (index > start).then(|| (source[start..index].to_string(), index))
}

/// Whether a `%`-sigil identifier names a type rather than a local value. LLVM prints named struct
/// types quoted whenever the C++ name needs it and unquoted as `struct.X` / `class.X` / `union.X`
/// otherwise; both forms are distinguishable from a local value's name.
fn looks_like_a_type(name: &str, was_quoted: bool) -> bool {
    was_quoted
        || name.starts_with("struct.")
        || name.starts_with("class.")
        || name.starts_with("union.")
}

/// Assign, once per distinct source name, the placeholder that replaces every occurrence of it.
fn placeholder_for(assigned: &mut Vec<(String, String)>, name: &str, kind: &str) -> String {
    if let Some((_, to)) = assigned.iter().find(|(from, _)| from == name) {
        return to.clone();
    }
    let to = format!("{kind}{}", assigned.len());
    assigned.push((name.to_string(), to.clone()));
    to
}

/// Rewrite every identifier the AIR/LLVM contract does not own. Returns the rewritten module and
/// how many distinct identifiers were replaced, so a caller can refuse to draw a conclusion from a
/// module the rewrite did not actually perturb.
///
/// Numbered identifiers (`%0`, `%12`) are left alone: LLVM derives an unnamed function's implicit
/// value and label numbering from their order, so naming one shifts every later number and would
/// change the module rather than only its spelling.
fn rename_names_the_contract_does_not_own(source: &str) -> (String, usize) {
    let bytes = source.as_bytes();
    let mut out = String::with_capacity(source.len());
    let mut assigned: Vec<(String, String)> = Vec::new();
    let mut index = 0usize;
    let mut at_line_start = true;
    while index < bytes.len() {
        let from = index;
        // A block label is DEFINED bare (`entry:`) and referenced with a sigil (`%entry`). Renaming
        // only the references leaves the module unparseable, so the definition is matched here:
        // the one place in LLVM's grammar where an identifier starts a line and ends in a colon.
        if at_line_start {
            if let Some((name, end)) = identifier_at(source, index) {
                let numbered = name.starts_with(|character: char| character.is_ascii_digit());
                if bytes.get(end) == Some(&b':') && !numbered && !owned_by_contract(&name) {
                    out.push_str(&placeholder_for(&mut assigned, &name, RENAMED_LOCAL));
                    index = end;
                    at_line_start = false;
                    continue;
                }
            }
        }
        match bytes[index] {
            // A comment runs to end of line and can mention any identifier without referring to it.
            b';' => {
                let end = source[index..]
                    .find('\n')
                    .map_or(bytes.len(), |offset| index + offset);
                out.push_str(&source[index..end]);
                index = end;
            }
            // A string literal that is not an identifier: metadata (`!"air.buffer"`), an attribute
            // (`"air-buffer-no-alias"`), a section, a source filename. Its contents are data.
            b'"' => {
                let end = identifier_at(source, index).map_or(index + 1, |(_, end)| end);
                out.push_str(&source[index..end]);
                index = end;
            }
            sigil @ (b'@' | b'%') => {
                let was_quoted = bytes.get(index + 1) == Some(&b'"');
                let Some((name, end)) = identifier_at(source, index + 1) else {
                    out.push(sigil as char);
                    index += 1;
                    continue;
                };
                let numbered = name.starts_with(|character: char| character.is_ascii_digit());
                if numbered || owned_by_contract(&name) {
                    out.push_str(&source[index..end]);
                    index = end;
                    continue;
                }
                let kind = if sigil == b'@' {
                    RENAMED_SYMBOL
                } else if looks_like_a_type(&name, was_quoted) {
                    RENAMED_TYPE
                } else {
                    RENAMED_LOCAL
                };
                out.push(sigil as char);
                out.push_str(&placeholder_for(&mut assigned, &name, kind));
                index = end;
            }
            _ => {
                let character = source[index..].chars().next().expect("in-bounds character");
                out.push(character);
                index += character.len_utf8();
            }
        }
        let consumed = &source[from..index];
        at_line_start = match consumed.rfind('\n') {
            Some(position) => consumed[position + 1..].chars().all(char::is_whitespace),
            None => at_line_start && consumed.chars().all(char::is_whitespace),
        };
    }
    (out, assigned.len())
}

// ---- Compiling ----

/// The two committed profiles every input compiles on: a device that reaches buffers by device
/// address, and the Vulkan 1.2 floor, which binds them (Profile F).
const PROFILES: [&str; 2] = ["nvidia-rtx5080-laptop", "vk12-floor"];

fn devices() -> Vec<(&'static str, Device)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    PROFILES
        .iter()
        .map(|&stem| {
            let path = dir.join(format!("{stem}.json"));
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let description = DeviceDescription::from_khronos_json(&text, None)
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            let device = Device::from_queries(description, DevicePolicy::default())
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            (stem, device)
        })
        .collect()
}

/// What compiling one entry point on one profile gives.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Outcome {
    /// The final SPIR-V words of the entry point's stage module.
    Words(Vec<u32>),
    /// The typed outcome of a refusal, without its diagnostics ([`refusal`]).
    Refused(String),
}

impl Outcome {
    fn of(result: Result<Vec<u32>, Error>) -> Outcome {
        match result {
            Ok(words) => Outcome::Words(words),
            Err(error) => Outcome::Refused(refusal(&error)),
        }
    }
}

/// A refusal's typed outcome, as `src/tests/golden_hashes.rs` records it: the feature and what it needs,
/// the input error's kind, the missing runtime input, the limit, the descriptor error, or the
/// internal error's kind. A refusal's `location` and `symbol` are diagnostics that name what they
/// point at, so they are left out; what was decided is compared.
fn refusal(error: &Error) -> String {
    match error {
        Error::Unsupported(u) => format!("Unsupported {:?} {:?}", u.feature, u.needs),
        Error::NeedsInput(input) => format!("NeedsInput {input:?}"),
        Error::InvalidInput(input) => format!("InvalidInput {:?}", input.kind),
        Error::InvalidDescriptor(descriptor) => format!("InvalidDescriptor {descriptor:?}"),
        Error::LimitExceeded(limit) => format!("LimitExceeded {limit:?}"),
        Error::Internal(internal) => format!("Internal {:?}", internal.kind),
        other => format!("{other:?}"),
    }
}

/// The library `bitcode` decodes to, then each entry point's stage and outcome on each device, in
/// the library's function order. `Err` is the decode's refusal.
fn compile_all(
    bitcode: &[u8],
    devices: &[(&str, Device)],
) -> Result<Vec<(FunctionStage, Vec<Outcome>)>, String> {
    let library = Library::from_air(bitcode).map_err(|error| refusal(&error))?;
    let budget = Budget::default();
    let mut out = Vec::new();
    for (index, info) in (0u32..).zip(library.functions()) {
        if info.stage == FunctionStage::Visible {
            continue;
        }
        let outcomes = devices
            .iter()
            .map(|(_, device)| {
                Outcome::of((|| {
                    let function = FunctionDescriptor::new(library.function_at(index)?);
                    let compiled =
                        device.compile_function(&function, &LoweringInputs::new(), &budget)?;
                    if info.stage == FunctionStage::Kernel {
                        let descriptor =
                            ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
                        let pipeline = device.compile_compute_pipeline(&descriptor, &budget)?;
                        Ok(pipeline.stage.spirv.to_vec())
                    } else {
                        Ok(link_standalone(compiled.artifact(), device.profile())?.1)
                    }
                })())
            })
            .collect();
        out.push((info.stage, outcomes));
    }
    Ok(out)
}

/// `llvm-as` of `text`, through a scratch file that is removed at once.
fn assemble(text: &str) -> Result<Vec<u8>, AssembleError> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "m2v-name-independence-{}-{}.ll",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let assembled = llvm_assemble(&path);
    std::fs::remove_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assembled
}

/// What [`assert_name_independent`] found.
enum Checked {
    /// Both texts compiled the same way: the original's entry points and outcomes, and how many
    /// distinct identifiers the rewrite replaced.
    Compared {
        entries: Vec<(FunctionStage, Vec<Outcome>)>,
        renamed: usize,
    },
    /// The original does not decode; its renamed twin is refused the same way.
    Undecodable(String),
    /// This `llvm-as` rejects the original (an older LLVM than the fixture's), so nothing was
    /// compared.
    Unassembled(String),
}

/// Compile `source` and its renamed twin on `devices`, and require every entry point to compile to
/// the same words or be refused the same way. `None` when `llvm-as` is not installed (printed).
fn assert_name_independent(
    label: &str,
    source: &str,
    devices: &[(&str, Device)],
) -> Option<Checked> {
    for prefix in [RENAMED_SYMBOL, RENAMED_TYPE, RENAMED_LOCAL] {
        assert!(
            !source.contains(prefix),
            "{label} already contains the placeholder prefix {prefix}, so renaming could collide \
             with a name the module already uses"
        );
    }
    let (renamed_source, renamed) = rename_names_the_contract_does_not_own(source);
    let original = match assemble(source) {
        Ok(bitcode) => compile_all(&bitcode, devices),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return None;
        }
        Err(AssembleError::Rejected(why)) => return Some(Checked::Unassembled(why)),
    };
    let perturbed = match assemble(&renamed_source) {
        Ok(bitcode) => compile_all(&bitcode, devices),
        Err(error) => panic!(
            "{label} assembles, but its renamed twin does not, so the renamer broke it: \
             {error:?}\n{renamed_source}"
        ),
    };
    match (&original, &perturbed) {
        (Ok(a), Ok(b)) => {
            assert_eq!(
                a.len(),
                b.len(),
                "{label}: renaming {renamed} identifiers changed how many entry points it has"
            );
            for (index, ((stage_a, outcomes_a), (stage_b, outcomes_b))) in
                a.iter().zip(b).enumerate()
            {
                assert_eq!(
                    stage_a, stage_b,
                    "{label}: entry point {index}'s stage changed after {renamed} identifiers \
                     outside the AIR/LLVM contract were renamed"
                );
                for (((stem, _), x), y) in devices.iter().zip(outcomes_a).zip(outcomes_b) {
                    assert!(
                        x == y,
                        "{label}: entry point {index} on {stem} compiled differently after \
                         {renamed} identifiers outside the AIR/LLVM contract were renamed; some \
                         decision reads a name instead of the structure.\n  original: {}\n  \
                         renamed:  {}",
                        brief(x),
                        brief(y)
                    );
                }
            }
        }
        (Err(a), Err(b)) => assert_eq!(
            a, b,
            "{label}: the original and its renamed twin are refused differently at decode"
        ),
        _ => panic!(
            "{label}: renaming {renamed} identifiers changed whether the library decodes:\n  \
             original: {original:?}\n  renamed:  {:?}",
            perturbed.as_ref().map(|_| "decodes")
        ),
    }
    Some(match original {
        Ok(entries) => Checked::Compared { entries, renamed },
        Err(why) => Checked::Undecodable(why),
    })
}

fn brief(outcome: &Outcome) -> String {
    match outcome {
        Outcome::Words(words) => format!("{} words", words.len()),
        Outcome::Refused(why) => format!("refused: {why}"),
    }
}

/// The words of the entry points that compiled on any profile.
fn compiled_modules(entries: &[(FunctionStage, Vec<Outcome>)]) -> Vec<&[u32]> {
    entries
        .iter()
        .flat_map(|(_, outcomes)| outcomes)
        .filter_map(|outcome| match outcome {
            Outcome::Words(words) => Some(words.as_slice()),
            Outcome::Refused(_) => None,
        })
        .collect()
}

// ---- Authored inputs ----

/// The module header of the authored inputs: Metal 4's target and the AIR version metadata a
/// compiled `.air` carries.
const HEADER: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"
"#;

/// The module-level metadata every authored input ends with (numbered from 90 up, clear of the
/// inputs' own).
const TRAILER: &str = r#"!air.compile_options = !{!90, !91, !92}
!air.version = !{!93}
!air.language_version = !{!94}
!90 = !{!"air.compile.denorms_disable"}
!91 = !{!"air.compile.fast_math_disable"}
!92 = !{!"air.compile.framebuffer_fetch_enable"}
!93 = !{i32 2, i32 8, i32 0}
!94 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

fn authored(body: &str) -> String {
    format!("{HEADER}\n{body}\n{TRAILER}")
}

/// A kernel that copies one word through two user helpers, with every user-chosen identifier
/// spelled as a near-miss of something the engine does dispatch on.
///
/// None of these is a contract spelling. The AIR and LLVM families are dot-separated
/// (`air.get_width_texture_2d`, `llvm.memcpy.p0.p0.i64`); `air_get_width_texture_2d` and
/// `llvm_fabs_f32` are ordinary user names that a `starts_with("air")` or a substring match would
/// swallow, and `__metal_ilogb_float` is Metal's runtime routine `___metal_ilogb_float` (with its
/// signature) short of one underscore. The helpers are *called*, so a test that took them for
/// intrinsics would refuse or mislower the calls instead of inlining them. `texture_buffer` and
/// `_cube_array` are the
/// substrings a texture shape is spelled with, in an AIR metadata type string, never in the name
/// of a local.
const LOOK_ALIKE_CONTRACT_NAMES: &str = r#"define void @air_get_width_texture_2d(ptr addrspace(1) noundef %llvm_memcpy_p0_p0_i64, ptr addrspace(1) noundef %texture_buffer) {
_cube_array.entry:
  %_cube_array = call i32 @llvm_fabs_f32(ptr addrspace(1) %llvm_memcpy_p0_p0_i64)
  %_1d_array = call i32 @air_get_num_samples_texture_2d_ms(i32 %_cube_array)
  %_2d_ms = bitcast i32 %_1d_array to float
  %_3d = call i32 @__metal_ilogb_float(float %_2d_ms, i32 0)
  store i32 %_3d, ptr addrspace(1) %texture_buffer, align 4
  ret void
}

define internal i32 @llvm_fabs_f32(ptr addrspace(1) %source) {
  %value = load i32, ptr addrspace(1) %source, align 4
  ret i32 %value
}

define internal i32 @air_get_num_samples_texture_2d_ms(i32 %value) {
  %sum = add i32 %value, %value
  ret i32 %sum
}

define internal i32 @__metal_ilogb_float(float %value, i32 %mode) {
  %bits = bitcast float %value to i32
  %sum = add i32 %bits, %mode
  ret i32 %sum
}

!air.kernel = !{!0}
!0 = !{ptr @air_get_width_texture_2d, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"input"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"output"}
"#;

/// A texture-handle array whose LLVM struct type name describes the wrong texture entirely.
///
/// The second handle is reached through a struct spelled
/// `metal::texture_buffer<float, access::write>`, while the AIR metadata declares
/// `array<texture2d<float, sample>, 2>`. The metadata is the ABI and has to win: the emitted image
/// must be a sampled 2D image, not a buffer image.
const MISLEADING_HANDLE_TYPE_NAME: &str = r#"%"struct.metal::texture_buffer<float, access::write>" = type { ptr addrspace(1) }

define void @width_of_second_texture(ptr readonly captures(none) %textures, ptr addrspace(1) noundef writeonly captures(none) %out) local_unnamed_addr #0 {
entry:
  %element = getelementptr %"struct.metal::texture_buffer<float, access::write>", ptr %textures, i64 1, i32 0
  %handle = load ptr addrspace(1), ptr %element, align 8
  %width = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) readonly captures(none) %handle, i32 0) #1
  store i32 %width, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.get_width_texture_2d(ptr addrspace(1) readonly captures(none), i32) local_unnamed_addr #1

attributes #0 = { convergent nounwind }
attributes #1 = { convergent nounwind memory(none) }

!air.kernel = !{!0}
!0 = !{ptr @width_of_second_texture, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 2, !"air.sample", !"air.arg_type_name", !"array<texture2d<float, sample>, 2>", !"air.arg_name", !"textures"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;

/// A kernel whose Metal entry point is called `main`, the name the engine gives every SPIR-V entry
/// point: the one input where an incoming name and an emitted name collide.
const ENTRY_ALREADY_NAMED_MAIN: &str = r#"define void @main(ptr addrspace(1) noundef %input, ptr addrspace(1) noundef %output) {
  %value = load i32, ptr addrspace(1) %input, align 4
  store i32 %value, ptr addrspace(1) %output, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"input"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"output"}
"#;

/// Compile an authored input and its renamed twin; every entry point must compile on every
/// profile. Returns the original's modules.
fn authored_modules(label: &str, body: &str) -> Option<Vec<Vec<u32>>> {
    let devices = devices();
    let (entries, renamed) = match assert_name_independent(label, &authored(body), &devices)? {
        Checked::Compared { entries, renamed } => (entries, renamed),
        Checked::Undecodable(why) => panic!("{label} does not decode: {why}"),
        Checked::Unassembled(why) => panic!("llvm-as rejects {label}: {why}"),
    };
    assert!(
        renamed > 0,
        "{label} has no identifier outside the AIR/LLVM contract, so compiling it twice proves \
         nothing about name independence"
    );
    for (index, (_, outcomes)) in entries.iter().enumerate() {
        for ((stem, _), outcome) in devices.iter().zip(outcomes) {
            assert!(
                matches!(outcome, Outcome::Words(_)),
                "{label}: entry point {index} on {stem}: {}",
                brief(outcome)
            );
        }
    }
    Some(
        compiled_modules(&entries)
            .into_iter()
            .map(<[u32]>::to_vec)
            .collect(),
    )
}

#[test]
fn identifiers_that_look_like_contract_names_are_still_just_identifiers() {
    authored_modules("the look-alike-named kernel", LOOK_ALIKE_CONTRACT_NAMES);
}

/// The `Dim` operand of each `OpTypeImage` in `words` (SPIR-V: opcode 25, the dimension at word 3).
fn image_dims(words: &[u32]) -> Vec<u32> {
    const HEADER_WORDS: usize = 5;
    const OP_TYPE_IMAGE: u32 = 25;
    let mut dims = Vec::new();
    let mut index = HEADER_WORDS;
    while index < words.len() {
        let length = (words[index] >> 16) as usize;
        assert!(length > 0, "an instruction of zero words at {index}");
        if words[index] & 0xffff == OP_TYPE_IMAGE {
            dims.push(words[index + 3]);
        }
        index += length;
    }
    dims
}

#[test]
fn the_texture_shape_comes_from_the_metadata_not_the_handle_type_name() {
    const DIM_2D: u32 = 1;
    let Some(modules) = authored_modules(
        "the misleadingly typed texture handle",
        MISLEADING_HANDLE_TYPE_NAME,
    ) else {
        return;
    };
    for words in modules {
        assert_eq!(
            image_dims(&words),
            [DIM_2D],
            "the handle's struct is named texture_buffer and the metadata declares texture2d; \
             the metadata is the ABI, so the one image type must be 2D"
        );
    }
}

#[test]
fn an_entry_point_already_named_main_compiles_like_any_other() {
    authored_modules(
        "the kernel whose entry is named main",
        ENTRY_ALREADY_NAMED_MAIN,
    );
}

// ---- The fixture sweep ----

/// The committed opaque-pointer fixtures: the `.ll` files of `validation/fixtures/public` and its
/// `engine/` directory, without the typed-pointer `.air.ll` ones (`xcrun metal-as`).
fn fixtures() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for dir in [fixtures_dir(), engine_dir()] {
        for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
            let path = entry.expect("fixture directory entry").path();
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
            if name.is_some_and(|n| n.ends_with(".ll") && !n.ends_with(".air.ll")) {
                paths.push(path);
            }
        }
    }
    paths.sort();
    paths
}

/// The same contract over every committed fixture: each entry point compiles to the same words, or
/// is refused the same way, on both profiles once its names are rewritten.
#[test]
fn every_public_fixture_compiles_the_same_under_renamed_identifiers() {
    let devices = devices();
    let (mut perturbed, mut compiled, mut unassembled) = (0usize, 0usize, 0usize);
    for path in fixtures() {
        let source = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
        let label = path
            .strip_prefix(fixtures_dir())
            .unwrap_or(&path)
            .display()
            .to_string();
        match assert_name_independent(&label, &source, &devices) {
            None => return,
            Some(Checked::Compared { entries, renamed }) if renamed > 0 => {
                perturbed += 1;
                compiled += usize::from(!compiled_modules(&entries).is_empty());
            }
            Some(Checked::Compared { .. } | Checked::Undecodable(_)) => {}
            Some(Checked::Unassembled(why)) => {
                eprintln!("{label}: this llvm-as rejects it, so it is not compared: {why}");
                unassembled += 1;
            }
        }
    }
    eprintln!(
        "{compiled} fixtures with a renamable identifier compiled an entry point ({perturbed} had \
         a renamable identifier; this llvm-as rejected {unassembled})"
    );
    // 401 of 432 at this guard's introduction; the floor leaves room for fixtures to come and go.
    assert!(
        compiled >= 300,
        "only {compiled} fixtures (of {perturbed} with a renamable identifier) compiled an entry \
         point, so the sweep checked far fewer than it did"
    );
}

mod renamer_tests {
    use super::*;

    #[test]
    fn contract_families_survive_and_everything_else_does_not() {
        let source = concat!(
            "%\"struct.metal::texture2d\" = type { i32 }\n",
            "@fc.MTL_FC_INIT_3_b = internal constant i8 undef\n",
            "define void @kernel_main(ptr %in) {\n",
            "  %v = call i32 @air.get_width_texture_2d(ptr %in, i32 0)\n",
            "  %w = call i32 @llvm.smax.i32(i32 %v, i32 0)\n",
            "  %x = call i32 @air_get_width(i32 %w)\n",
            "  %y = call i32 @___metal_ilogb_float(float 1.0, i32 0)\n",
            "  %z = call i32 @__metal_ilogb_float(float 1.0, i32 0)\n",
            "  ret void\n",
            "}\n",
        );
        let (renamed, count) = rename_names_the_contract_does_not_own(source);
        assert!(renamed.contains("@air.get_width_texture_2d"));
        assert!(renamed.contains("@llvm.smax.i32"));
        assert!(renamed.contains("@___metal_ilogb_float"));
        assert!(!renamed.contains("@__metal_ilogb_float"));
        assert!(!renamed.contains("@fc.MTL_FC_INIT_3_b"));
        assert!(!renamed.contains("@kernel_main"));
        assert!(!renamed.contains("@air_get_width"));
        assert!(!renamed.contains("%\"struct.metal::texture2d\""));
        // The struct type, the function constant, kernel_main, air_get_width, __metal_ilogb_float,
        // and the locals in, v, w, x, y, z.
        assert_eq!(count, 11, "{renamed}");
    }

    #[test]
    fn numbered_values_keep_their_numbers() {
        let source = "define void @f(i32 %0) {\n  %2 = add i32 %0, 1\n  br label %3\n}\n";
        let (renamed, count) = rename_names_the_contract_does_not_own(source);
        assert!(renamed.contains("%0"), "{renamed}");
        assert!(renamed.contains("%2"), "{renamed}");
        assert!(renamed.contains("%3"), "{renamed}");
        assert_eq!(count, 1, "only @f is renamable: {renamed}");
    }

    #[test]
    fn a_label_is_renamed_where_it_is_defined_as_well_as_where_it_is_used() {
        let source = "define void @f() {\nentry:\n  br label %body\nbody:\n  ret void\n}\n";
        let (renamed, count) = rename_names_the_contract_does_not_own(source);
        assert_eq!(
            renamed,
            "define void @renamed.symbol.0() {\nrenamed.local.1:\n  br label %renamed.local.2\n\
             renamed.local.2:\n  ret void\n}\n"
        );
        assert_eq!(count, 3, "@f, entry, and body");
    }

    #[test]
    fn quoted_data_is_not_an_identifier() {
        let source = "; @commented_out\n@g = global i8 0, section \"air.fc_initializer\"\n!0 = !{!\"air.buffer\", !\"input\"}\n";
        let (renamed, _) = rename_names_the_contract_does_not_own(source);
        assert!(renamed.contains("; @commented_out"), "{renamed}");
        assert!(
            renamed.contains("section \"air.fc_initializer\""),
            "{renamed}"
        );
        assert!(renamed.contains("!\"air.buffer\", !\"input\""), "{renamed}");
    }
}
