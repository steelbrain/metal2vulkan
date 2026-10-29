//! Modules built with Metal debug info translate to the words they give without it.
//!
//! Xcode's `MTL_ENABLE_DEBUG_INFO` compiles with `-gline-tables-only`, and at `INCLUDE_SOURCE`
//! (its default for Debug builds) with `-frecord-sources` too: a compile unit, a subprogram per
//! function, a location per instruction, two module flags and, until the linker moves them out of
//! the module, the command line, the working directory and the sources. `FILL` is the kernel
//! Apple's compiler writes with those flags (paths shortened); with its debug info removed it is,
//! up to metadata numbering, the same compiler's output without them. Both spellings compile to
//! the same words on every committed profile, and so do the linked-function fixtures with line
//! tables added to their modules. `llvm-as` assembles them; a missing tool skips visibly.

use crate::budget::SPIRV_WORDS;
use crate::library::FunctionConstantValues;
use crate::lower::select::LowerInputs;
use crate::pipeline::compile::{compile_bitcode_linked, EntryRef, LinkedSource};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, meter, AssembleError, Scratch,
};
use crate::spirv::testing::committed_profiles;

/// `kernel void fill(device float *out [[buffer(0)]], uint id [[thread_position_in_grid]])
/// { out[id] = float(id) * 0.5f; }` compiled with `-gline-tables-only -frecord-sources`.
/// `<d>`…`</d>` marks what the debug info adds.
const FILL: &str = r#"
source_filename = "fill.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v29-apple-macosx27.0.0"

define void @fill(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, i32 noundef %1) local_unnamed_addr #0<d> !dbg !35</d> {
  %3 = tail call fast float @air.convert.f.f32.u.i32(i32 %1) #2<d>, !dbg !37</d>
  %4 = fmul fast float %3, 5.000000e-01<d>, !dbg !38</d>
  %5 = zext i32 %1 to i64<d>, !dbg !39</d>
  %6 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %5<d>, !dbg !39</d>
  store float %4, ptr addrspace(1) %6, align 4<d>, !dbg !40</d>, !tbaa !41, !alias.scope !45
  ret void<d>, !dbg !48</d>
}

declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: write) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, <d>!1, !2, </d>!3, !4, !5, !6, !7, !8, !9, !10}
<d>!llvm.dbg.cu = !{!11}
</d>!air.kernel = !{!19}
<d>!llvm_utils.workingdir = !{!24}
!llvm_utils.sources = !{!25}
</d>!air.compile_options = !{!27, !28, !29}
!llvm.ident = !{!30}
<d>!llvm.commandline = !{!31}
</d>!air.version = !{!32}
!air.language_version = !{!33}
!air.source_file_name = !{!34}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 0]}
<d>!1 = !{i32 7, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
</d>!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{i32 7, !"air.max_device_buffers", i32 31}
!6 = !{i32 7, !"air.max_constant_buffers", i32 31}
!7 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!8 = !{i32 7, !"air.max_textures", i32 128}
!9 = !{i32 7, !"air.max_read_write_textures", i32 8}
!10 = !{i32 7, !"air.max_samplers", i32 16}
<d>!11 = distinct !DICompileUnit(language: DW_LANG_Metal, file: !12, producer: "metal", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, imports: !13, splitDebugInlining: false, nameTableKind: None, sysroot: "/sdk", sdk: "MacOSX27.0.sdk")
!12 = !DIFile(filename: "fill.metal", directory: "/src")
!13 = !{!14, !17}
!14 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !15, file: !16, line: 1)
!15 = !DIModule(scope: null, name: "metal_types", includePath: "/metal/include")
!16 = !DIFile(filename: "<built-in>", directory: "/src")
!17 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !18, file: !12, line: 1)
!18 = !DIModule(scope: null, name: "metal_stdlib", includePath: "/metal/include")
</d>!19 = !{ptr @fill, !20, !21}
!20 = !{}
!21 = !{!22, !23}
!22 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!23 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"id"}
<d>!24 = !{!"/src"}
!25 = !{!26}
!26 = !{!"/src/fill.metal", !"kernel void fill(device float *out [[buffer(0)]], uint id [[thread_position_in_grid]]) {\0A    out[id] = float(id) * 0.5f;\0A}\0A"}
</d>!27 = !{!"air.compile.denorms_disable"}
!28 = !{!"air.compile.fast_math_enable"}
!29 = !{!"air.compile.framebuffer_fetch_enable"}
!30 = !{!"metal"}
<d>!31 = !{!"metal -c -gline-tables-only -frecord-sources=yes -o fill.air fill.metal"}
</d>!32 = !{i32 2, i32 9, i32 0}
!33 = !{!"Metal", i32 4, i32 1, i32 0}
!34 = !{!"/src/fill.metal"}
<d>!35 = distinct !DISubprogram(name: "fill", scope: !12, file: !12, line: 3, type: !36, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !11, retainedNodes: !20)
!36 = !DISubroutineType(types: !20)
!37 = !DILocation(line: 4, column: 21, scope: !35)
!38 = !DILocation(line: 4, column: 25, scope: !35)
!39 = !DILocation(line: 4, column: 5, scope: !35)
!40 = !DILocation(line: 4, column: 13, scope: !35)
</d>!41 = !{!42, !42, i64 0}
!42 = !{!"float", !43, i64 0}
!43 = !{!"omnipotent char", !44, i64 0}
!44 = !{!"Simple C++ TBAA"}
!45 = !{!46}
!46 = distinct !{!46, !47, !"air-alias-scope-arg(0)"}
!47 = distinct !{!47, !"air-alias-scopes(fill)"}
<d>!48 = !DILocation(line: 5, column: 1, scope: !35)
</d>"#;

/// `FILL` with its debug info or without it.
fn fill(debug: bool) -> String {
    let mut out = String::new();
    let mut rest = FILL;
    while let Some((before, after)) = rest.split_once("<d>") {
        let (inside, after) = after.split_once("</d>").expect("an unclosed <d>");
        out.push_str(before);
        if debug {
            out.push_str(inside);
        }
        rest = after;
    }
    out.push_str(rest);
    out
}

/// `text` assembled, or `None` (skipped) without `llvm-as`.
fn assemble(scratch: &Scratch, name: &str, text: &str) -> Option<Vec<u8>> {
    let ll = scratch.path().join(format!("{name}.ll"));
    std::fs::write(&ll, text).expect("write");
    match llvm_assemble(&ll) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(e) => panic!("llvm-as rejected {name}: {e:?}"),
    }
}

#[test]
fn a_kernel_with_line_tables_and_recorded_sources_compiles_to_the_same_words() {
    let scratch = Scratch::new("m2v-debug-info");
    let (Some(debug), Some(plain)) = (
        assemble(&scratch, "debug", &fill(true)),
        assemble(&scratch, "plain", &fill(false)),
    ) else {
        return;
    };
    assert_ne!(debug, plain, "the debug info reaches the bitcode");
    let profiles = committed_profiles().expect("profiles");
    assert!(!profiles.is_empty());
    for (stem, p) in &profiles {
        let words = |bitcode: &[u8]| {
            let a = compile(bitcode, "fill", p).unwrap_or_else(|e| panic!("{stem}: {e}"));
            link_and_validate(&a, p)
                .unwrap_or_else(|e| panic!("{stem}: {e}"))
                .1
        };
        assert_eq!(words(&debug), words(&plain), "{stem}");
    }
}

/// `text` with the line tables `-gline-tables-only` writes: a compile unit, the two debug-info
/// module flags, a subprogram per definition and a location per instruction.
fn with_line_tables(text: &str) -> String {
    let flags = "!llvm.module.flags = !{";
    assert!(text.contains(flags), "a module-flags list");
    let mut out = String::new();
    let mut nodes = String::from(concat!(
        "!llvm.dbg.cu = !{!900}\n",
        "!900 = distinct !DICompileUnit(language: DW_LANG_Metal, file: !901, isOptimized: true, ",
        "emissionKind: LineTablesOnly)\n",
        "!901 = !DIFile(filename: \"m.metal\", directory: \"/src\")\n",
        "!902 = !{i32 7, !\"Dwarf Version\", i32 4}\n",
        "!903 = !{i32 2, !\"Debug Info Version\", i32 3}\n",
        "!904 = !DISubroutineType(types: !{})\n",
    ));
    let mut next = 905;
    let mut scope = None;
    for line in text.lines() {
        let line = line.replacen(flags, "!llvm.module.flags = !{!902, !903, ", 1);
        if let Some(head) = line.strip_suffix(" {").filter(|l| l.starts_with("define ")) {
            let name = head
                .split_once('@')
                .and_then(|(_, rest)| rest.split_once('('))
                .expect("a defined name")
                .0;
            nodes += &format!(
                "!{next} = distinct !DISubprogram(name: \"{name}\", scope: !901, file: !901, \
                 line: 1, type: !904, scopeLine: 1, spFlags: DISPFlagDefinition | \
                 DISPFlagOptimized, unit: !900)\n"
            );
            out += &format!("{head} !dbg !{next} {{\n");
            scope = Some(next);
            next += 1;
        } else if line == "}" {
            scope = None;
            out += "}\n";
        } else if let Some(sp) = scope.filter(|_| line.starts_with("  ")) {
            nodes += &format!("!{next} = !DILocation(line: {next}, scope: !{sp})\n");
            out += &format!("{line}, !dbg !{next}\n");
            next += 1;
        } else {
            out += &line;
            out += "\n";
        }
    }
    out + &nodes
}

/// The linked-function fixtures (`engine/linked_visible_caller`, whose kernel calls
/// `linked_scale` and `linked_blend` of `engine/linked_visible_functions` and its own
/// `linked_offset`, all `[[visible]]`) with line tables in both modules link and compile to the
/// words they give without them.
#[test]
fn linked_visible_functions_with_line_tables_compile_to_the_same_words() {
    let scratch = Scratch::new("m2v-debug-info-linked");
    let read = |stem: &str| {
        std::fs::read_to_string(engine_dir().join(format!("{stem}.ll"))).expect("read")
    };
    let (caller, functions) = (
        read("linked_visible_caller"),
        read("linked_visible_functions"),
    );
    let (Some(caller), Some(functions), Some(debug_caller), Some(debug_functions)) = (
        assemble(&scratch, "caller", &caller),
        assemble(&scratch, "functions", &functions),
        assemble(&scratch, "debug_caller", &with_line_tables(&caller)),
        assemble(&scratch, "debug_functions", &with_line_tables(&functions)),
    ) else {
        return;
    };
    assert_ne!(debug_caller, caller, "the debug info reaches the bitcode");
    let none = FunctionConstantValues::defaults();
    for (stem, p) in &committed_profiles().expect("profiles") {
        let words = |caller: &[u8], functions: &[u8]| {
            let linked = [
                LinkedSource::new("linked_scale", functions, &none),
                LinkedSource::new("linked_blend", functions, &none),
                LinkedSource::new("linked_offset", caller, &none),
            ];
            let a = compile_bitcode_linked(
                caller,
                EntryRef::Name("linked_visible_caller"),
                (&none, &linked),
                p,
                &LowerInputs::new(),
                &mut meter(),
                SPIRV_WORDS,
            )
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
            link_and_validate(&a, p)
                .unwrap_or_else(|e| panic!("{stem}: {e}"))
                .1
        };
        assert_eq!(
            words(&debug_caller, &debug_functions),
            words(&caller, &functions),
            "{stem}"
        );
    }
}
