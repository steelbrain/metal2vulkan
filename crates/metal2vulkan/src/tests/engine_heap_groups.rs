//! Heap groups: the loads of one argument-buffer member share a heap group, so a read of the
//! member types its queries (never the MSL type name); a group only queried is declared `Float`
//! with `RuntimeFact::HeapNumericType`. Under the identity domains such a group, a 2D texture's or
//! a cube array's alike, is `Unsupported{IdentityDomains}`, not implemented, naming that fact. No
//! device is opened.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use super::engine_word_phases::identity_profiles;
use crate::device::DevicePolicy;
use crate::error::{Error, Feature, Needs};
use crate::library::FunctionArtifact;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{HandleClass, ImageHandleClass, ResourceKind, SampledType};
use crate::pipeline::testing::{compile, link_and_validate, llvm_assemble, profile, AssembleError};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

fn assembled(name: &str, r: Result<Vec<u8>, AssembleError>) -> Option<Vec<u8>> {
    match r {
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
    let dir = std::env::temp_dir().join(format!(
        "m2v-heap-binder-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, format!("{HEADER}{body}")).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    assembled(name, r)
}

/// Two kernels over one argument-buffer member that only the MSL type name calls an integer
/// texture: `heap_query` only queries it, `heap_query_read` also reads it as `uint`.
const QUERY_KERNELS: &str = r#"%struct.T = type { ptr addrspace(1) }
%struct.R = type { %struct.T }
define void @heap_query(ptr addrspace(2) %r, ptr addrspace(1) %out, i32 %gid) {
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 0)
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %gid
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
}
define void @heap_query_read(ptr addrspace(2) %r, ptr addrspace(1) %out, i32 %gid) {
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 0)
  %t2 = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %c = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  %v = call { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) %t2, ptr addrspace(2) %s, <2 x i32> %c, <2 x i32> zeroinitializer, i32 0, i32 0)
  %x4 = extractvalue { <4 x i32>, i8 } %v, 0
  %x = extractelement <4 x i32> %x4, i64 0
  %sum = add i32 %w, %x
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %gid
  store i32 %sum, ptr addrspace(1) %o, align 4
  ret void
}
declare ptr addrspace(2) @air.get_read_sampler()
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
declare { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
!air.kernel = !{!0, !11}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @heap_query, !1, !2}
!1 = !{}
!2 = !{!3, !7, !8}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<uint, read>", !"t", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<uint, read>", !"air.arg_name", !"t"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!8 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
!11 = !{ptr @heap_query_read, !1, !2}
"#;

/// The heap classes a function declares.
fn heap_classes_of(a: &FunctionArtifact) -> Vec<ImageHandleClass> {
    a.interface
        .descriptors
        .iter()
        .filter_map(|d| match d.slot.kind {
            ResourceKind::Heap(HandleClass::Image(c)) => Some(c),
            _ => None,
        })
        .collect()
}

/// A heap group that is only queried has no texel class: its array is declared `Float`, and
/// `RuntimeFact::HeapNumericType` says so. A load of the same member that is read types every
/// group loaded from it: the member has one declared type.
#[test]
fn a_member_read_anywhere_types_its_queries() {
    let Some(bc) = assemble("heap_query", QUERY_KERNELS) else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    let a = compile(&bc, "heap_query", &p).expect("heap_query");
    link_and_validate(&a, &p).expect("L3");
    let classes = heap_classes_of(&a);
    // The float class, and its unsigned integer twin, which a member of a form a heap reads is
    // queried through.
    assert_eq!(classes.len(), 2);
    assert_eq!(classes[0].sampled_type, SampledType::Float);
    assert_eq!(
        classes[1],
        ImageHandleClass {
            sampled_type: SampledType::Uint,
            ..classes[0]
        }
    );
    assert!(a
        .facts
        .contains(&RuntimeFact::HeapNumericType { class: classes[0] }));

    let a = compile(&bc, "heap_query_read", &p).expect("heap_query_read");
    link_and_validate(&a, &p).expect("L3");
    assert_eq!(
        a.strategies
            .entries
            .get("access.handle.lookup")
            .map_or(0, |u| u.uses),
        2
    );
    // Both loads of the member share the unsigned integer class; the read and the query read a
    // member of another numeric class through its shape's class of that type.
    let mut classes = heap_classes_of(&a);
    classes.sort_by_key(|c| c.sampled_type);
    assert_eq!(
        classes,
        [SampledType::Float, SampledType::Sint, SampledType::Uint].map(|sampled_type| {
            ImageHandleClass {
                sampled_type,
                ..classes[0]
            }
        }),
        "both loads of the member share one class"
    );
    assert!(!a
        .facts
        .iter()
        .any(|f| matches!(f, RuntimeFact::HeapNumericType { .. })));
}

/// Two kernels that only query an argument buffer's texture: `abq_2d` a `texture2d<float>`'s
/// width, `abq_cube_array` a `texturecube_array<float>`'s array size.
const QUERY_ONLY_KERNELS: &str = r#"%struct.Q = type { ptr addrspace(1) }
define void @abq_2d(ptr addrspace(2) %q, ptr addrspace(1) %out, i32 %gid) {
  %pt = getelementptr inbounds %struct.Q, ptr addrspace(2) %q, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 0)
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %gid
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
}
define void @abq_cube_array(ptr addrspace(2) %q, ptr addrspace(1) %out, i32 %gid) {
  %pt = getelementptr inbounds %struct.Q, ptr addrspace(2) %q, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %n = call i32 @air.get_array_size_texture_cube_array(ptr addrspace(1) %t)
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %gid
  store i32 %n, ptr addrspace(1) %o, align 4
  ret void
}
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
declare i32 @air.get_array_size_texture_cube_array(ptr addrspace(1))
!air.kernel = !{!0, !9}
!air.version = !{!7}
!air.language_version = !{!8}
!0 = !{ptr @abq_2d, !1, !2}
!1 = !{}
!2 = !{!3, !5, !6}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Q", !"air.arg_name", !"q"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"t", !"air.indirect_argument", !12}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!7 = !{i32 2, i32 8, i32 0}
!8 = !{!"Metal", i32 4, i32 0, i32 0}
!9 = !{ptr @abq_cube_array, !1, !10}
!10 = !{!11, !5, !6}
!11 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !13, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Q", !"air.arg_name", !"q"}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!13 = !{i32 0, i32 8, i32 0, !"texturecube_array<float, sample>", !"t", !"air.indirect_argument", !14}
!14 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texturecube_array<float, sample>", !"air.arg_name", !"t"}
"#;

/// Under the identity domains, an argument buffer's texture that a kernel only queries is
/// refused, whatever its shape: its group has no texel class, and the identity domains source no
/// `RuntimeFact::HeapNumericType`. Each kernel of [`QUERY_ONLY_KERNELS`] is
/// `Unsupported{IdentityDomains}`, not implemented, naming that fact of its member's shape, on
/// every profile the identity domains derive.
#[test]
fn a_query_only_argument_buffer_texture_is_refused_naming_its_numeric_type() {
    let Some(bc) = assemble("query_only", QUERY_ONLY_KERNELS) else {
        return;
    };
    let profiles = identity_profiles();
    assert!(
        !profiles.is_empty(),
        "no profile the identity domains derive"
    );
    let form = "state of a bound texture or sampler beyond its resource ID (HeapNumericType {";
    for (stem, p) in profiles {
        for (entry, shape) in [
            ("abq_2d", "dim: D2, arrayed: false"),
            ("abq_cube_array", "dim: Cube, arrayed: true"),
        ] {
            let Err(Error::Unsupported(u)) = compile(&bc, entry, &p) else {
                panic!("{stem}: {entry}: not refused");
            };
            assert_eq!(
                (u.feature, u.needs),
                (Feature::IdentityDomains, Needs::NotImplemented),
                "{stem}: {entry}"
            );
            let symbol = u.symbol.map(|s| s.to_string()).unwrap_or_default();
            assert!(
                symbol.starts_with(form) && symbol.contains(shape),
                "{stem}: {entry}: {symbol}"
            );
        }
    }
}
