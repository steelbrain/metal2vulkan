use super::*;
use crate::air::{entry_count_from_ll, entry_text, module_entries};

/// A module of three kernels (the synthetic multi-entry module):
/// - `a` and `b` both call `@shared_loop` (a loop) and sample with the static sampler `@sampler`;
/// - `c` calls neither, and reads the function table `@table`, whose initializer alone names
///   `@via_table`;
/// - `@init`, the module's static initializer, calls `@air.init_call`;
/// - `!air.pair`'s `!31` names one helper of `a` and one of `b`, and its `!32` names the helper
///   `a` and `b` share;
/// - `@llvm.compiler.used` keeps `@sampler` alive.
pub(crate) const THREE_KERNELS: &str = r#"; ModuleID = 'three'
source_filename = "three"
target datalayout = "e-p:64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.pair = type { i32, i32 }

@sampler = internal addrspace(2) constant [2 x i64] [i64 34901797601036946, i64 0], align 8
@table = internal addrspace(2) constant [1 x ptr] [ptr @via_table], align 8
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @init, ptr null }]
@llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(2) @sampler to ptr)], section "llvm.metadata"

; Function Attrs: nounwind
define void @a(ptr addrspace(1) %0) local_unnamed_addr #0 {
  %2 = tail call i32 @shared_loop(i32 4)
  %3 = tail call i32 @only_a(ptr addrspace(2) @sampler)
  store i32 %2, ptr addrspace(1) %0, align 4, !tbaa !40
  ret void
}

define void @b(ptr addrspace(1) %0) local_unnamed_addr #0 {
  %2 = tail call i32 @shared_loop(i32 8)
  %3 = tail call i32 @only_b(ptr addrspace(2) @sampler)
  store i32 %2, ptr addrspace(1) %0, align 4
  ret void
}

define void @c(ptr addrspace(1) %0) local_unnamed_addr #0 {
  %2 = load ptr, ptr addrspace(2) @table, align 8
  %3 = tail call i32 @air.c_only(i32 1)
  store i32 %3, ptr addrspace(1) %0, align 4
  ret void
}

define internal i32 @shared_loop(i32 %0) #0 {
  br label %2

2:                                                ; preds = %2, %1
  %3 = phi i32 [ 0, %1 ], [ %4, %2 ]
  %4 = add i32 %3, 1
  %5 = icmp ult i32 %4, %0
  br i1 %5, label %2, label %6

6:                                                ; preds = %2
  ret i32 %4
}

define internal i32 @only_a(ptr addrspace(2) %0) #0 {
  %2 = tail call i32 @air.a_only(ptr addrspace(2) %0)
  ret i32 %2
}

define internal i32 @only_b(ptr addrspace(2) %0) #0 {
  ret i32 7
}

define internal i32 @via_table(i32 %0) #0 {
  %2 = tail call i32 @air.table_only(i32 %0)
  ret i32 %2
}

define internal void @init() #0 section "air.static_init" {
  %1 = tail call i32 @air.init_call(i32 0)
  ret void
}

declare i32 @air.a_only(ptr addrspace(2)) #1
declare i32 @air.c_only(i32) #1
declare i32 @air.table_only(i32) #1
declare i32 @air.init_call(i32) #1
declare i32 @air.unused(i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0}
!air.kernel = !{!10, !11, !12}
!air.sampler_states = !{!30}
!air.pair = !{!31, !32}
!air.compile_options = !{!33}

!0 = !{i32 7, !"air.max_samplers", i32 16}
!10 = !{ptr @a, !13, !14}
!11 = !{ptr @b, !13, !15}
!12 = !{ptr @c, !13, !16}
!13 = !{}
!14 = !{!17}
!15 = !{!18}
!16 = !{!19}
!17 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.arg_name", !"a_out"}
!18 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.arg_name", !"b_out"}
!19 = !{i32 0, !"air.render_target", !"air.arg_name", !"c_out"}
!30 = !{!"air.sampler_state", ptr addrspace(2) @sampler}
!31 = !{ptr @only_a, ptr @only_b}
!32 = !{ptr @shared_loop}
!33 = !{!"air.compile.fast_math_enable"}
!40 = !{!"int", !41}
!41 = !{!"tbaa root"}
"#;

fn projection(entry: &str) -> String {
    entry_projection(THREE_KERNELS, entry).unwrap()
}

fn defines(text: &str, symbol: &str) -> bool {
    text.lines().any(|line| {
        ((line.starts_with("define ") || line.starts_with("declare "))
            && line.contains(&format!("@{symbol}(")))
            || line.starts_with(&format!("@{symbol} = "))
    })
}

#[test]
fn each_entry_keeps_what_it_reaches_and_nothing_another_entry_reaches() {
    let entries = module_entries(THREE_KERNELS).unwrap();
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<Vec<_>>(),
        ["a", "b", "c"]
    );
    assert_eq!(entry_count_from_ll(THREE_KERNELS), 3);
    let reaches = |entry: &str, symbols: &[&str], not: &[&str]| {
        let text = projection(entry);
        for symbol in symbols {
            assert!(defines(&text, symbol), "{entry} keeps @{symbol}:\n{text}");
        }
        for symbol in not {
            assert!(!defines(&text, symbol), "{entry} drops @{symbol}:\n{text}");
        }
        text
    };
    let always = ["llvm.global_ctors", "init", "air.init_call"];
    let a = reaches(
        "a",
        &[
            &always[..],
            &["a", "shared_loop", "only_a", "air.a_only", "sampler"],
        ]
        .concat(),
        &[
            "b",
            "c",
            "only_b",
            "table",
            "via_table",
            "air.c_only",
            "air.unused",
        ],
    );
    let b = reaches(
        "b",
        &[&always[..], &["b", "shared_loop", "only_b", "sampler"]].concat(),
        &["a", "c", "only_a", "air.a_only", "table", "via_table"],
    );
    // `@via_table` is reached only through `@table`'s initializer.
    let c = reaches(
        "c",
        &[
            &always[..],
            &["c", "table", "via_table", "air.table_only", "air.c_only"],
        ]
        .concat(),
        &["a", "b", "shared_loop", "only_a", "only_b", "sampler"],
    );
    for text in [&a, &b, &c] {
        // Liveness annotations are no entry's, and comments are dropped.
        assert!(!text.contains("llvm.compiler.used"), "{text}");
        assert!(!text.contains("; Function Attrs"), "{text}");
        // What no entry owns is in every projection.
        for line in [
            "source_filename = \"three\"",
            "%struct.pair = type { i32, i32 }",
            "attributes #1 = { nounwind willreturn memory(none) }",
            "!llvm.module.flags = !{!0}",
            "!air.compile_options = !{!33}",
            "!33 = !{!\"air.compile.fast_math_enable\"}",
        ] {
            assert!(text.lines().any(|kept| kept == line), "{line}:\n{text}");
        }
    }
    // Each declares exactly its own entry.
    assert!(a.contains("!air.kernel = !{!10}\n"), "{a}");
    assert!(b.contains("!air.kernel = !{!11}\n"), "{b}");
    assert!(c.contains("!air.kernel = !{!12}\n"), "{c}");
    // Another entry's argument metadata is not in the projection.
    assert!(a.contains("!\"a_out\"") && !a.contains("!\"b_out\""), "{a}");
    assert!(
        !c.contains("air.buffer") && !c.contains("air.texture"),
        "{c}"
    );
    // The function's own attachments are kept.
    assert!(
        a.contains("!40 = !{!\"int\", !41}") && a.contains("!41 = "),
        "{a}"
    );
    assert!(!b.contains("!40 = "), "{b}");
}

/// A named-metadata operand is kept for exactly the entries that reach every symbol it names:
/// the static sampler's for `a` and `b`, the shared helper's for `a` and `b`, and the operand
/// naming one helper of each of `a` and `b` for neither.
#[test]
fn named_metadata_is_kept_for_the_entries_that_reach_every_symbol_it_names() {
    assert!(projection("a").contains("!air.sampler_states = !{!30}\n"));
    assert!(projection("b").contains("!air.sampler_states = !{!30}\n"));
    let c = projection("c");
    assert!(
        !c.contains("air.sampler_states") && !c.contains("!30 = "),
        "{c}"
    );
    for entry in ["a", "b"] {
        let text = projection(entry);
        assert!(text.contains("!air.pair = !{!32}\n"), "{text}");
        assert!(!text.contains("!31 = "), "{text}");
    }
    assert!(!c.contains("air.pair"), "{c}");
}

#[test]
fn a_projection_is_its_own_projection() {
    for entry in ["a", "b", "c"] {
        let text = projection(entry);
        assert_eq!(entry_count_from_ll(&text), 1);
        assert_eq!(entry_projection(&text, entry).unwrap(), text);
        assert_eq!(entry_text(&text, entry).unwrap(), text);
    }
    // A module with one entry is its own text.
    assert!(matches!(
        entry_text(&projection("a"), "a").unwrap(),
        std::borrow::Cow::Borrowed(_)
    ));
}

/// Top-level assembly is whitespace-insensitive: an indented module projects onto the same items.
#[test]
fn indentation_does_not_move_what_the_projection_places() {
    let indented: String = THREE_KERNELS
        .lines()
        .map(|line| format!("    {line}\n"))
        .collect();
    for entry in ["a", "b", "c"] {
        let projected = entry_projection(&indented, entry).unwrap();
        let unindented: Vec<&str> = projected.lines().map(str::trim_start).collect();
        let expected = projection(entry);
        assert_eq!(
            unindented,
            expected.lines().map(str::trim_start).collect::<Vec<_>>()
        );
    }
}

/// Every function the runtime starts is every entry's, listed or not: a lone `air.static_init`
/// function with no `@llvm.global_ctors`, which the product runs (`air::canon::static_init`), and
/// an `air.fc_initializer` function, which the oracle's fuel plan starts.
#[test]
fn a_function_the_runtime_starts_is_every_entrys_without_a_list() {
    let ctors = THREE_KERNELS
        .lines()
        .find(|line| line.starts_with("@llvm.global_ctors = "))
        .unwrap();
    let lone = THREE_KERNELS.replace(&format!("{ctors}\n"), "").replace(
        "attributes #0 = { nounwind }",
        "define internal void @fc_init() #0 section \"air.fc_initializer\" {\n  \
         %1 = tail call i32 @air.fc_call(i32 0)\n  ret void\n}\n\
         declare i32 @air.fc_call(i32) #1\n\
         attributes #0 = { nounwind }",
    );
    assert!(!lone.contains("llvm.global_ctors"));
    for entry in ["a", "b", "c"] {
        let text = entry_projection(&lone, entry).unwrap();
        for symbol in ["init", "air.init_call", "fc_init", "air.fc_call"] {
            assert!(defines(&text, symbol), "{entry} keeps @{symbol}:\n{text}");
        }
    }
}

/// The projection places every line by rule; anything else is an error naming it.
#[test]
fn what_the_projection_cannot_place_is_an_error_naming_it() {
    let unknown = THREE_KERNELS.replace(
        "attributes #0 = { nounwind }",
        "uselistorder ptr @a, { 1, 0 }\nattributes #0 = { nounwind }",
    );
    let error = entry_projection(&unknown, "a").unwrap_err();
    assert!(
        error.contains("uselistorder") && error.contains("no top-level item"),
        "{error}"
    );
    let undefined = THREE_KERNELS.replace("@air.a_only(ptr addrspace(2) %0)", "@nowhere(i32 0)");
    let error = entry_projection(&undefined, "a").unwrap_err();
    assert!(error.contains("@nowhere"), "{error}");
    let unclosed = THREE_KERNELS.replace(
        "  %1 = tail call i32 @air.init_call(i32 0)\n  ret void\n}\n",
        "  %1 = tail call i32 @air.init_call(i32 0)\n  ret void\n",
    );
    let error = entry_projection(&unclosed, "a").unwrap_err();
    assert!(error.contains("never closes"), "{error}");
    let error = entry_projection(THREE_KERNELS, "only_a").unwrap_err();
    assert!(error.contains("no entry only_a"), "{error}");
    let missing_node = THREE_KERNELS.replace("!41 = !{!\"tbaa root\"}\n", "");
    let error = entry_projection(&missing_node, "a").unwrap_err();
    assert!(error.contains("!41"), "{error}");
}

/// Every committed public module with several entries projects onto each of them, and the
/// projection declares exactly that entry.
#[test]
fn every_committed_multi_entry_module_projects_onto_each_entry() {
    let root = metal2vulkan_devtools::fixtures_dir();
    let mut entries = 0;
    let mut modules = 0;
    for directory in ["", "engine"] {
        for file in std::fs::read_dir(root.join(directory)).unwrap() {
            let path = file.unwrap().path();
            let name = path.file_name().unwrap().to_str().unwrap();
            if !name.ends_with(".ll") || name.ends_with(".air.ll") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            if entry_count_from_ll(&text) < 2 {
                continue;
            }
            modules += 1;
            let declared = module_entries(&text).unwrap();
            assert_eq!(declared.len(), entry_count_from_ll(&text), "{name}");
            for entry in declared {
                let projected = entry_projection(&text, &entry.name)
                    .unwrap_or_else(|error| panic!("{name} {}: {error}", entry.name));
                assert_eq!(
                    module_entries(&projected).unwrap(),
                    std::slice::from_ref(&entry),
                    "{name} {}",
                    entry.name
                );
                entries += 1;
            }
        }
    }
    assert!(
        modules >= 28 && entries >= 196,
        "{modules} modules, {entries} entries"
    );
}
