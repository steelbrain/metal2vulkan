//! A state whose reflection is refused still answers what it may store through each slot.
//!
//! The shape here is the one a consumer's workload failed on: a kernel whose arguments include
//! one whose *type* the reflection cannot describe (a struct pointee the AIR gives no
//! `air.struct_type_info` and whose `air.arg_type_name` is not a name the MSL data-type grammar
//! covers). Deriving the type description refuses the whole entry's bindings, so `reflection`
//! answers nothing about any slot, including the read-only ones; the writability question derives
//! no type description and answers each slot.

use super::{SlotWritability, SlotWrite, SlotWriteGap};
use crate::budget::Budget;
use crate::error::{Error, Feature};
use crate::library::Library;
use crate::metal::{BindingAccess, BindingType};
use crate::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};

/// A kernel whose four bindings are, in declaration order:
///
/// - `buffer(0)`, `constant` and `air.read`: a struct pointee with no `air.struct_type_info`, so
///   its `air.arg_type_name` (`Uniforms`) is the only thing stating its type, and the data-type
///   grammar does not name it. The reflection of the entry is refused here.
/// - `buffer(1)`, `device` and `air.read_write`: a plain `float` pointee.
/// - `texture(0)` declared `access::read`.
/// - `texture(1)` declared `access::write`.
///
/// The two read-only bindings are the ones a consumer must be able to tell apart from the
/// writable ones: a read-only vertex or index buffer may sit at `buffer(0)`.
fn module() -> String {
    r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.Uniforms = type { float, float }

define void @shade(ptr addrspace(2) noundef %0, ptr addrspace(1) noundef %1, ptr addrspace(1) noundef %2, ptr addrspace(1) noundef %3, <3 x i32> noundef %4) {
  %6 = load float, ptr addrspace(2) %0, align 4
  store float %6, ptr addrspace(1) %1, align 4
  ret void
}

!air.kernel = !{!0}
!air.version = !{!20}
!air.language_version = !{!21}

!0 = !{ptr @shade, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Uniforms", !"air.arg_name", !"uniforms"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float>", !"air.arg_name", !"source"}
!6 = !{i32 3, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float>", !"air.arg_name", !"destination"}
!7 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tid"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#
    .to_owned()
}

/// The kernel of [`module`] as a compute pipeline descriptor; `None` where `llvm-as` is missing.
fn kernel() -> Option<ComputePipelineDescriptor> {
    let bitcode = match metal2vulkan_devtools::llvm_assemble_text(&module()) {
        Ok(bitcode) => bitcode,
        Err(metal2vulkan_devtools::AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return None;
        }
        Err(metal2vulkan_devtools::AssembleError::Rejected(why)) => {
            panic!("llvm-as rejected the module: {why}")
        }
    };
    let library = Library::from_air(&bitcode).expect("the module decodes");
    let f = library.function("shade").expect("the kernel");
    Some(ComputePipelineDescriptor::new(StageSource::Function(
        FunctionDescriptor::new(f),
    )))
}

/// The reflection of the kernel is refused, naming the argument whose type it cannot describe, so
/// it answers nothing about any slot: the shape a consumer's session died on, kept as the reason
/// the writability question exists apart from it.
#[test]
fn a_reflection_is_refused_for_an_argument_type_it_cannot_describe() {
    let Some(kernel) = kernel() else { return };
    match kernel.reflection(&Budget::standard()) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::Reflection);
            assert_eq!(
                u.symbol.as_ref().map(ToString::to_string).as_deref(),
                Some("Uniforms"),
                "the refusal names the argument's type"
            );
        }
        other => panic!("the reflection was not refused: {other:?}"),
    }
}

/// The same kernel answers what it may store through each of its slots: the `constant` buffer and
/// the `access::read` texture cannot be written, the `device` buffer and the `access::write`
/// texture may be. Every binding is answered, the one whose type refused the reflection included,
/// and no answer is `Unknown`.
#[test]
fn a_state_whose_reflection_is_refused_still_answers_its_slot_writes() {
    let Some(kernel) = kernel() else { return };
    let writes = kernel
        .slot_writes(&Budget::standard())
        .expect("the slot writes are answered where the reflection is refused");
    let slots = writes.kernel_slots();
    let seen: Vec<_> = slots
        .iter()
        .map(|s| (s.name.as_str(), s.binding_type, s.index, s.writes))
        .collect();
    assert_eq!(
        seen,
        [
            (
                "uniforms",
                Some(BindingType::Buffer),
                Some(0),
                SlotWritability::Never
            ),
            (
                "out",
                Some(BindingType::Buffer),
                Some(1),
                SlotWritability::Maybe
            ),
            (
                "source",
                Some(BindingType::Texture),
                Some(0),
                SlotWritability::Never
            ),
            (
                "destination",
                Some(BindingType::Texture),
                Some(1),
                SlotWritability::Maybe
            ),
        ]
    );
    // The builtin binds no slot.
    assert_eq!(slots.len(), 4);
    assert!(slots.iter().all(|s| s.writes.gap().is_none()));
    assert!(slots.iter().all(|s| s.used));
}

/// The per-slot question a consumer asks: the read-only slots answer `Never`, which is what lets
/// a read-only buffer sit at one. A blanket "every slot may be written" answer fails here.
#[test]
fn the_slot_question_separates_the_read_only_slots_from_the_writable_ones() {
    let Some(kernel) = kernel() else { return };
    let writes = kernel.slot_writes(&Budget::standard()).expect("answered");
    let at = |binding_type, index| SlotWrite::may_write(writes.kernel_slots(), binding_type, index);
    assert_eq!(at(BindingType::Buffer, 0), SlotWritability::Never);
    assert_eq!(at(BindingType::Buffer, 1), SlotWritability::Maybe);
    assert_eq!(at(BindingType::Texture, 0), SlotWritability::Never);
    assert_eq!(at(BindingType::Texture, 1), SlotWritability::Maybe);
    // A slot no argument binds is one nothing stores through.
    assert_eq!(at(BindingType::Buffer, 7), SlotWritability::Never);
    assert_eq!(at(BindingType::Sampler, 0), SlotWritability::Never);
}

/// An argument of undetermined writability is one answer among determined ones, and it never
/// reads as writable nor as not writable: the slots it may cover answer `Unknown` with its gap,
/// and the other slots keep their own answers.
#[test]
fn an_undetermined_slot_stays_undetermined_and_leaves_the_others_alone() {
    let answer = |name: &str, binding_type, index, count, access, writes| SlotWrite {
        name: name.to_owned(),
        parameter: 0,
        binding_type,
        index,
        count,
        used: true,
        access,
        writes,
    };
    let slots = [
        answer(
            "read_only",
            Some(BindingType::Buffer),
            Some(0),
            Some(1),
            Some(BindingAccess::ReadOnly),
            SlotWritability::Never,
        ),
        // A slot only a function constant states: the declared access is known (read-write), but
        // which buffer slot it applies to is not.
        answer(
            "computed",
            Some(BindingType::Buffer),
            None,
            None,
            Some(BindingAccess::ReadWrite),
            SlotWritability::Unknown(SlotWriteGap::UnresolvedIndex),
        ),
        answer(
            "sampled",
            Some(BindingType::Texture),
            Some(3),
            Some(1),
            Some(BindingAccess::ReadOnly),
            SlotWritability::Never,
        ),
    ];
    let at = |binding_type, index| SlotWrite::may_write(&slots, binding_type, index);
    // Every buffer slot is undetermined, the one a determined answer calls read-only included:
    // the argument of unknown slot may be that very slot.
    for index in [0, 1, 30] {
        assert_eq!(
            at(BindingType::Buffer, index),
            SlotWritability::Unknown(SlotWriteGap::UnresolvedIndex),
            "buffer {index}"
        );
    }
    // The uncertainty stays with the buffer slots: a texture slot keeps its own answer.
    assert_eq!(at(BindingType::Texture, 3), SlotWritability::Never);
    assert_eq!(at(BindingType::Texture, 0), SlotWritability::Never);
}

/// A texture `array<texture2d<float>, N>` binds `N` consecutive slots from its index, and only
/// those: a writable array leaves the slots past its span alone.
#[test]
fn an_argument_array_covers_its_own_span_alone() {
    let slots = [SlotWrite {
        name: "images".to_owned(),
        parameter: 0,
        binding_type: Some(BindingType::Texture),
        index: Some(2),
        count: Some(3),
        used: true,
        access: Some(BindingAccess::ReadWrite),
        writes: SlotWritability::Maybe,
    }];
    let at = |index| SlotWrite::may_write(&slots, BindingType::Texture, index);
    assert_eq!(at(1), SlotWritability::Never);
    for index in 2..5 {
        assert_eq!(at(index), SlotWritability::Maybe, "texture {index}");
    }
    assert_eq!(at(5), SlotWritability::Never);
}

/// Over every kernel of the public reflection fixtures: the answer is never refused, and wherever
/// the pipeline's reflection is not refused the two agree, slot for slot, on what may be stored
/// through it. The reflection derives a type description and this does not, so the fixtures whose
/// argument shapes refuse a reflection are the point: they still answer here.
///
/// It is the broad form of the claim the whole module exists for -- a state that compiles answers
/// -- and the cross-check that the two derivations cannot drift apart, since `Binding::access` is
/// held to Metal's recorded answers by `library_reflection_matches_metal` and the pipeline cases.
#[test]
fn every_fixture_kernel_answers_and_agrees_with_its_reflection() {
    let stems = [
        "argument_buffers",
        "buffers",
        "encoder_arrays",
        "encoder_reads",
        "encoder_types",
        "encoders",
        "leftovers",
        "specialized_reads",
        "tables",
        "tables_in_arguments",
        "texture_accesses",
        "textures",
        "tile",
    ];
    let mut answered = 0;
    let mut compared = 0;
    let mut refused_reflections = 0;
    // The recorded fixtures, then the synthetic kernel whose argument type refuses a reflection:
    // the fixtures state the agreement over shapes Metal's answers are recorded for, and the
    // synthetic one makes the refused leg run, since no recorded fixture kernel refuses.
    for stem in stems.iter().copied().chain(["(synthetic)"]) {
        let bitcode = if stem == "(synthetic)" {
            match metal2vulkan_devtools::llvm_assemble_text(&module()) {
                Ok(b) => b,
                Err(metal2vulkan_devtools::AssembleError::Missing(why)) => {
                    eprintln!("SKIPPED: {why}");
                    return;
                }
                Err(metal2vulkan_devtools::AssembleError::Rejected(why)) => {
                    panic!("llvm-as rejected the synthetic module: {why}")
                }
            }
        } else {
            let path =
                crate::pipeline::testing::fixtures_dir().join(format!("reflection/{stem}.ll"));
            match crate::pipeline::testing::llvm_assemble(&path) {
                Ok(b) => b,
                Err(crate::pipeline::testing::AssembleError::Missing(why)) => {
                    eprintln!("SKIPPED: {why}");
                    return;
                }
                Err(crate::pipeline::testing::AssembleError::Rejected(why)) => {
                    panic!("llvm-as rejected {stem}.ll: {why}")
                }
            }
        };
        let library = Library::from_air(&bitcode).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        let names: Vec<String> = library
            .functions()
            .filter(|info| info.stage == crate::library::FunctionStage::Kernel)
            .map(|info| info.name.to_owned())
            .collect();
        for name in names {
            let f = library
                .function(&name)
                .unwrap_or_else(|e| panic!("{stem} {name}: {e:?}"));
            let at = format!("{stem} {name}");
            let kernel =
                ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
            let writes = match kernel.slot_writes(&Budget::standard()) {
                Ok(writes) => writes,
                // A function no pipeline of compiles: the reflection refuses it too. What must
                // never happen is a refusal over a reflection gap, which is what this asserts.
                Err(error) => {
                    assert!(
                        !matches!(&error, Error::Unsupported(u) if u.feature == Feature::Reflection),
                        "{at}: refused as a reflection gap: {error:?}"
                    );
                    assert!(
                        kernel.reflection(&Budget::standard()).is_err(),
                        "{at}: refused where the reflection answers: {error:?}"
                    );
                    continue;
                }
            };
            answered += 1;
            let slots = writes.kernel_slots();
            assert!(
                slots.iter().all(|s| s.writes.gap().is_none()),
                "{at}: an undetermined slot among {slots:?}"
            );
            let reflection = match kernel.reflection(&Budget::standard()) {
                Ok(r) => r,
                Err(Error::Unsupported(u)) if u.feature == Feature::Reflection => {
                    refused_reflections += 1;
                    continue;
                }
                Err(e) => panic!("{at}: {e:?}"),
            };
            // A kernel without a stage input descriptor reflects its own arguments alone, so the
            // two lists line up one for one.
            let bindings = reflection.bindings();
            assert_eq!(bindings.len(), slots.len(), "{at}");
            for (b, s) in bindings.iter().zip(slots) {
                assert_eq!(
                    (
                        b.name(),
                        Some(b.binding_type()),
                        Some(b.index()),
                        b.is_used()
                    ),
                    (s.name.as_str(), s.binding_type, s.index, s.used),
                    "{at}: the two answers name different bindings"
                );
                // The declared access is the same value, not merely a compatible verdict.
                assert_eq!(
                    s.access,
                    Some(b.access()),
                    "{at}: {} disagrees with its reflection's access",
                    b.name()
                );
                let by_access = match b.access() {
                    BindingAccess::ReadOnly => SlotWritability::Never,
                    BindingAccess::WriteOnly | BindingAccess::ReadWrite => SlotWritability::Maybe,
                };
                assert_eq!(
                    s.writes,
                    by_access,
                    "{at}: {} reads a verdict its access does not give",
                    b.name()
                );
                compared += 1;
            }
        }
    }
    assert!(answered >= 20, "{answered} kernels answered");
    assert!(compared >= 40, "{compared} bindings compared");
    assert!(
        refused_reflections > 0,
        "no subject's reflection was refused, so this proves nothing about the gap it covers: \
         the sweep needs a kernel whose argument shapes Metal's answer is unmeasured for"
    );
}
