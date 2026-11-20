//! The families once refused are typed and located, and `air.is_uniform` stands for its asserted
//! `true` only where nothing observes its result.
//!
//! Every module is written here, assembled with `llvm-as` (a missing tool skips visibly), and
//! compiled by the engine for every committed profile. Each family is
//! `Unsupported{needs: NotImplemented}` with its own feature on every profile until it lands,
//! located in the input: a call or instruction at its (function, block, instruction), an entry
//! parameter or the entry's interface at its ABI metadata. A refused call names its callee.

use crate::air::error::{CanonError, InputErrorKind};
use crate::device::profile::DeviceProfile;
use crate::device::DevicePolicy;
use crate::error::{DiagName, Error, Feature, Location, LowerFamily, Needs, Unsupported};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, profile, AssembleError, Scratch,
};
use crate::spirv::testing::{committed_profiles, threadgroup_profiles};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// The version metadata every module below ends with.
const VERSIONS: &str = "!air.version = !{!90}\n!air.language_version = !{!91}\n\
                        !90 = !{i32 2, i32 8, i32 0}\n!91 = !{!\"Metal\", i32 4, i32 0, i32 0}\n";

/// A device buffer `out` at parameter 0.
const OUT: &str = "!{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, \
                   !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", \
                   i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", \
                   !\"air.arg_name\", !\"out\"}";

/// `text` (after [`HEADER`]) assembled from a scratch file, or `None` when `llvm-as` is missing.
fn assemble(tag: &str, text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new(&format!("m2v-engine-refused-{tag}"));
    let dir = scratch.path();
    let path = dir.join("module.ll");
    std::fs::write(&path, format!("{HEADER}{text}{VERSIONS}")).expect("write");
    let bc = llvm_assemble(&path);
    match bc {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {tag}: {why}"),
    }
}

/// A kernel `k(device uint *out [[buffer(0)]], uint tid [[thread_position_in_grid]])` whose body
/// is `body` (ending in `ret void`), with `decls` after it.
fn kernel(body: &str, decls: &str) -> String {
    format!(
        "define void @k(ptr addrspace(1) %0, i32 %1) {{\n{body}}}\n\n{decls}\n\
         !air.kernel = !{{!0}}\n!0 = !{{ptr @k, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4}}\n!3 = {OUT}\n\
         !4 = !{{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", \
         !\"air.arg_name\", !\"tid\"}}\n"
    )
}

/// The refusal of `entry` of `bitcode`: the same on every committed profile, since none of these
/// refusals depends on the device.
fn refusal(bitcode: &[u8], entry: &str) -> Unsupported {
    refusal_on(
        bitcode,
        entry,
        committed_profiles().expect("committed profiles"),
    )
}

/// [`refusal`] on `profiles`.
fn refusal_on(bitcode: &[u8], entry: &str, profiles: Vec<(String, DeviceProfile)>) -> Unsupported {
    let mut first: Option<Unsupported> = None;
    for (stem, p) in profiles {
        let u = match compile(bitcode, entry, &p) {
            Err(Error::Unsupported(u)) => u,
            Err(other) => panic!("{entry} on {stem}: {other}"),
            Ok(_) => panic!("{entry} on {stem}: compiled"),
        };
        match &first {
            Some(f) => assert_eq!(&u, f, "{entry} on {stem}"),
            None => first = Some(u),
        }
    }
    first.expect("at least one committed profile")
}

/// `u` refuses `feature`, not implemented yet.
fn assert_refused(u: &Unsupported, feature: Feature) {
    assert_eq!(u.feature, feature, "{u}");
    assert_eq!(u.needs, Needs::NotImplemented, "{u}");
}

/// `u` is located at an instruction of function `func`.
fn assert_at_instruction(u: &Unsupported, func: &str) {
    let site = u.location.as_ref().and_then(Location::in_function);
    match site {
        Some((f, Some(_), Some(_))) => assert_eq!(f, &DiagName::new(func), "{u}"),
        _ => panic!("not at an instruction of {func}: {u}"),
    }
}

/// `u` is located at the ABI metadata of entry `entry`, at argument `arg` when it has one.
fn assert_at_metadata(u: &Unsupported, entry: &str, arg: Option<u32>) {
    assert_eq!(
        u.location,
        Some(Location::metadata(Some(DiagName::new(entry)), arg, None)),
        "{u}"
    );
}

/// Ray tracing: an acceleration structure used other than as an intersector's structure (its
/// address taken) is ray tracing on every profile; an intersector itself lowers where the device
/// traces rays (`src/tests/engine_ray_tracing.rs`). Another undecoded intrinsic stays
/// `UnknownIntrinsic`, not implemented.
#[test]
fn escaped_acceleration_structures_are_ray_tracing() {
    let future = kernel(
        "  %d = call float @air.some_future_op(i32 %1)\n  store float %d, ptr addrspace(1) %0, \
         align 4\n  ret void\n",
        "declare float @air.some_future_op(i32)\n",
    );
    let Some(bc) = assemble("future", &future) else {
        return;
    };
    let u = refusal(&bc, "k");
    assert_eq!(u.feature, Feature::UnknownIntrinsic, "{u}");
    assert_eq!(u.needs, Needs::NotImplemented, "{u}");
    assert_eq!(u.symbol, Some(DiagName::new("air.some_future_op")), "{u}");
    assert_at_instruction(&u, "k");

    let accel = "define void @k(ptr addrspace(1) %0, ptr addrspace(1) %1) {\n  %i = ptrtoint \
                 ptr addrspace(1) %1 to i64\n  store i64 %i, ptr addrspace(1) %0, align 8\n  ret \
                 void\n}\n\n!air.kernel = !{!0}\n!0 = !{ptr @k, !1, !2}\n!1 = !{}\n!2 = !{!3, \
                 !4}\n!4 = !{i32 1, !\"air.instance_acceleration_structure\", \
                 !\"air.location_index\", i32 1, i32 1, !\"air.read\", !\"air.arg_type_name\", \
                 !\"acceleration_structure<instancing>\", !\"air.arg_name\", !\"scene\"}\n";
    let Some(bc) = assemble("accel", &format!("{accel}!3 = {OUT}\n")) else {
        return;
    };
    // A device that traces rays binds the structure and refuses the escape where it stands; any
    // other refuses the structure's argument, naming what it lacks.
    for (stem, p) in committed_profiles().expect("committed profiles") {
        let u = match compile(&bc, "k", &p) {
            Err(Error::Unsupported(u)) => u,
            Err(other) => panic!("{stem}: {other}"),
            Ok(_) => panic!("{stem}: compiled"),
        };
        assert_eq!(u.feature, Feature::RayTracing, "{stem}: {u}");
        match &u.needs {
            Needs::NotImplemented => assert_at_instruction(&u, "k"),
            Needs::AnyOf(_) => assert_at_metadata(&u, "k", Some(1)),
            _ => panic!("{stem}: {u}"),
        }
    }
}

/// A ray tracing name that does not decode stays ray tracing, by its family's first name
/// component: an intersector whose tags the grammar rejects, and an `intersection_query` operation,
/// which is not decoded yet. Each is refused where it is called, naming it.
#[test]
fn undecodable_ray_tracing_names_are_ray_tracing() {
    for (tag, callee) in [
        (
            "intersect-tag",
            "air.intersect.triangle_data.some_future_tag",
        ),
        (
            "query",
            "air.next_intersection_query.instancing.triangle_data",
        ),
    ] {
        let text = kernel(
            &format!(
                "  %d = call i32 @{callee}(i32 %1)\n  store i32 %d, ptr addrspace(1) %0, align \
                 4\n  ret void\n"
            ),
            &format!("declare i32 @{callee}(i32)\n"),
        );
        let Some(bc) = assemble(tag, &text) else {
            return;
        };
        let u = refusal(&bc, "k");
        assert_refused(&u, Feature::RayTracing);
        assert_eq!(u.symbol, Some(DiagName::new(callee)), "{u}");
        assert_at_instruction(&u, "k");
    }
}

/// Function tables and function pointers: an indirect call is a dynamic call.
#[test]
fn indirect_calls_are_dynamic_calls() {
    let text = kernel(
        "  %f = load ptr, ptr addrspace(1) %0, align 8\n  call void %f()\n  ret void\n",
        "",
    );
    let Some(bc) = assemble("indirect", &text) else {
        return;
    };
    let u = refusal(&bc, "k");
    assert_refused(&u, Feature::DynamicCall);
    assert_eq!(u.symbol, None, "{u}");
    assert_at_instruction(&u, "k");
}

/// A call to a function the module does not define (the shape of the tensor-ops library calls)
/// needs linking, naming the callee.
#[test]
fn calls_to_undefined_functions_are_linking() {
    let callee = "__tensorops_impl_matmul2d_op_cooperative_tensor_init";
    let text = kernel(
        &format!("  call void @{callee}(ptr addrspace(1) %0)\n  ret void\n"),
        &format!("declare void @{callee}(ptr addrspace(1))\n"),
    );
    let Some(bc) = assemble("linking", &text) else {
        return;
    };
    let u = refusal(&bc, "k");
    assert_refused(&u, Feature::Linking);
    assert_eq!(u.symbol, Some(DiagName::new(callee)), "{u}");
    assert_at_instruction(&u, "k");
}

/// A call whose function type is not its callee's (12 arguments to a 10-parameter function, as
/// two corpus fragment shaders pass) decodes as a call through the callee's address, not a direct
/// call; it is still classified by the function it calls. Nothing binds a call's arguments to a
/// defined function's parameters at another type (an LLVM call's arguments match its own type; a
/// call through an incompatible type is undefined behavior in C and C++), so it is invalid input
/// at the call on every profile, not a direct call to implement.
#[test]
fn a_mistyped_call_to_a_defined_function_is_invalid_input() {
    let text = kernel(
        "  call void @h(ptr addrspace(1) %0, i32 %1, i32 0)
  ret void
",
        "define internal void @h(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}
",
    );
    let Some(bc) = assemble("mistyped-defined", &text) else {
        return;
    };
    for (stem, p) in committed_profiles().expect("committed profiles") {
        match compile(&bc, "k", &p) {
            Err(Error::InvalidInput(e)) => {
                assert_eq!(
                    e.kind,
                    InputErrorKind::Canon(CanonError::CallTypeMismatch),
                    "{stem}"
                );
                match e.at.in_function() {
                    Some((f, Some(_), Some(_))) => assert_eq!(f, &DiagName::new("k"), "{stem}"),
                    _ => panic!("{stem}: not at an instruction of k: {e}"),
                }
            }
            Err(other) => panic!("{stem}: {other}"),
            Ok(_) => panic!("{stem}: compiled"),
        }
    }
}

/// The same mistyped call to a function the module only declares needs linking, naming it.
#[test]
fn a_mistyped_call_to_an_undefined_function_is_linking() {
    let text = kernel(
        "  call void @ext(ptr addrspace(1) %0, i32 %1, i32 0)
  ret void
",
        "declare void @ext(ptr addrspace(1), i32)
",
    );
    let Some(bc) = assemble("mistyped-undefined", &text) else {
        return;
    };
    let u = refusal(&bc, "k");
    assert_refused(&u, Feature::Linking);
    assert_eq!(u.symbol, Some(DiagName::new("ext")), "{u}");
    assert_at_instruction(&u, "k");
}

/// An imageblock query in a kernel without an explicit imageblock parameter has no imageblock to
/// answer from: refused at the instruction.
#[test]
fn imageblocks_are_refused() {
    let text = kernel(
        "  %w = call i16 @air.get_imageblock_width()\n  %r = zext i16 %w to i32\n  store i32 %r, \
         ptr addrspace(1) %0, align 4\n  ret void\n",
        "declare i16 @air.get_imageblock_width()\n",
    );
    let Some(bc) = assemble("imageblock", &text) else {
        return;
    };
    let u = refusal(&bc, "k");
    assert_refused(&u, Feature::Imageblocks);
    assert_at_instruction(&u, "k");
}

/// A fragment function's threadgroup memory (a `[[threadgroup(0)]]` buffer, which the corpus's
/// splat rasterizers set bits in) is its tile's memory, kept across the render pass: it compiles on
/// every committed profile with `fragmentStoresAndAtomics`, binding the pass's tile memory buffer
/// to the fragment stage, depending on its argument's placement, and valid under `spirv-val`. A
/// profile without the feature refuses it by name.
#[test]
fn fragment_threadgroup_memory_is_its_tiles() {
    use crate::pipeline::facts::RuntimeFact;
    use crate::pipeline::sym::{ResourceKind, Stage, SyntheticBinding};
    let text = "define <4 x float> @f(ptr addrspace(3) %0) {\n  %r = call i32 \
                @air.atomic.local.or.u.i32(ptr addrspace(3) %0, i32 1, i32 0, i32 1, i1 true)\n  \
                ret <4 x float> zeroinitializer\n}\n\ndeclare i32 \
                @air.atomic.local.or.u.i32(ptr addrspace(3), i32, i32, i32, i1)\n\n\
                !air.fragment = !{!0}\n!0 = !{ptr @f, !1, !3}\n!1 = !{!2}\n\
                !2 = !{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", \
                !\"float4\", !\"air.arg_name\", !\"color\"}\n!3 = !{!4}\n\
                !4 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, \
                !\"air.read_write\", !\"air.address_space\", i32 3, !\"air.arg_type_size\", i32 4, \
                !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", \
                !\"air.arg_name\", !\"bits\"}\n";
    let Some(bc) = assemble("fragment-threadgroup", text) else {
        return;
    };
    for (stem, p) in committed_profiles().expect("committed profiles") {
        if !p.fields().leaf.fragment_stores_and_atomics {
            match compile(&bc, "f", &p) {
                Err(Error::Unsupported(u)) => {
                    assert_eq!(u.feature, Feature::StageStoresAndAtomics, "{stem}: {u}")
                }
                other => panic!("{stem}: {other:?}"),
            }
            continue;
        }
        let a = compile(&bc, "f", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            a.interface
                .descriptors
                .iter()
                .any(|d| d.slot.stage == Stage::Fragment
                    && d.slot.kind == ResourceKind::Synthetic(SyntheticBinding::TileMemory)),
            "{stem}: no tile memory binding"
        );
        assert!(
            a.facts
                .contains(&RuntimeFact::TileThreadgroupMemory { index: 0 }),
            "{stem}: {:?}",
            a.facts
        );
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
    }
}

/// A threadgroup pointer made from an integer literal (`inttoptr (i64 1024 to ptr addrspace(3))`,
/// the operand of a corpus kernel's `atomic_fetch_sub`) names a byte of the threadgroup memory
/// Apple's compiler lays out after AIR, which AIR does not carry: not representable, at the
/// instruction. A plain load through one is refused the same way.
#[test]
fn a_threadgroup_address_made_from_an_integer_is_not_representable() {
    let address = "inttoptr (i64 1024 to ptr addrspace(3))";
    let atomic = kernel(
        &format!(
            "  %r = call i32 @air.atomic.local.sub.u.i32(ptr addrspace(3) {address}, i32 1, i32 0, \
             i32 1, i1 true)\n  store i32 %r, ptr addrspace(1) %0, align 4\n  ret void\n"
        ),
        "declare i32 @air.atomic.local.sub.u.i32(ptr addrspace(3), i32, i32, i32, i1)\n",
    );
    let load = kernel(
        &format!(
            "  %v = load i32, ptr addrspace(3) {address}, align 4\n  store i32 %v, ptr \
             addrspace(1) %0, align 4\n  ret void\n"
        ),
        "",
    );
    for (tag, text) in [("tg-int-atomic", atomic), ("tg-int-load", load)] {
        let Some(bc) = assemble(tag, &text) else {
            return;
        };
        // On every profile that takes threadgroup memory.
        let u = refusal_on(&bc, "k", threadgroup_profiles().expect("profiles"));
        assert_eq!(u.feature, Feature::DereferenceLoadedPointer, "{tag}: {u}");
        assert_eq!(u.needs, Needs::NotRepresentable, "{tag}: {u}");
        assert_at_instruction(&u, "k");
    }
}

/// A vertex function `v` returning `float4 [[position]]` with parameters `params` (metadata
/// nodes `!4` onward, listed in `!2`) and stage attributes `attrs` (a trailing entry operand).
fn vertex(params: &[&str], attrs: &str, body: &str, sig: &str) -> String {
    let list = (0..params.len())
        .map(|i| format!("!{}", 4 + i))
        .collect::<Vec<_>>()
        .join(", ");
    let nodes = params
        .iter()
        .enumerate()
        .map(|(i, p)| format!("!{} = {p}\n", 4 + i))
        .collect::<String>();
    format!(
        "define <4 x float> @v({sig}) {{\n{body}}}\n\n!air.vertex = !{{!0}}\n\
         !0 = !{{ptr @v, !1, !2{attrs}}}\n!1 = !{{!3}}\n!2 = !{{{list}}}\n\
         !3 = !{{!\"air.position\", !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", \
         !\"position\"}}\n{nodes}"
    )
}

/// A post-tessellation vertex function lowers; a call it makes to a declaration that is not its
/// control-point function is still linking, naming the callee, on every profile.
#[test]
fn a_post_tessellation_call_to_another_declaration_is_linking() {
    let patch = "!9 = !{!\"air.patch\", !\"triangle\", !\"air.patch_control_point\", i32 3}\n";
    let body = "  call void @v.MTL_CONTROL_POINT_FN()\n  ret <4 x float> <float 0.0, float 0.0, \
                float 0.0, float 1.0>\n";
    let decls = "declare void @v.MTL_CONTROL_POINT_FN()\n";
    let text = format!("{}{patch}{decls}", vertex(&[], ", !9", body, ""));
    let Some(bc) = assemble("tessellation-call", &text) else {
        return;
    };
    let u = refusal(&bc, "v");
    assert_refused(&u, Feature::Linking);
    assert_eq!(
        u.symbol,
        Some(DiagName::new("v.MTL_CONTROL_POINT_FN")),
        "{u}"
    );
}

/// Kernel `uniform_k(device uint *out [[buffer(0)]], uint t [[thread_index_in_threadgroup]])`
/// whose body `body` computes `%s` (a `uint`), stored to `out[t]`.
fn uniform_kernel(body: &str) -> String {
    format!(
        "define void @uniform_k(ptr addrspace(1) %0, i32 %1) {{\nentry:\n{body}  %e = zext i32 %1 \
         to i64\n  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %e\n  store i32 %s, \
         ptr addrspace(1) %p, align 4\n  ret void\n}}\n\n\
         declare i1 @air.is_uniform.i32(i32)\ndeclare void @llvm.assume(i1 noundef)\n\n\
         !air.kernel = !{{!0}}\n!0 = !{{ptr @uniform_k, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4}}\n\
         !3 = {OUT}\n!4 = !{{i32 1, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", \
         !\"uint\", !\"air.arg_name\", !\"t\"}}\n"
    )
}

/// The H1b probe. A uniformity assertion whose result a `select` reads (and a phi that a `select`
/// reads) is a runtime uniformity test, not an assertion: lowering it to `true` would write 1 on
/// every lane. It is refused as `Lowering(Simd)` at the `air.is_uniform` call on every profile.
#[test]
fn an_observed_uniformity_result_is_refused() {
    let direct = "  %u = tail call i1 @air.is_uniform.i32(i32 %1)\n  %s = select i1 %u, i32 1, \
                  i32 2\n";
    let through_phi = "  %c = icmp ult i32 %1, 8\n  br i1 %c, label %a, label %b\na:\n  %ua = \
                       tail call i1 @air.is_uniform.i32(i32 %1)\n  br label %j\nb:\n  br label \
                       %j\nj:\n  %u = phi i1 [ %ua, %a ], [ true, %b ]\n  %s = select i1 %u, i32 \
                       1, i32 2\n";
    for (tag, body) in [("observed", direct), ("observed-phi", through_phi)] {
        let Some(bc) = assemble(tag, &uniform_kernel(body)) else {
            return;
        };
        let u = refusal(&bc, "uniform_k");
        assert_eq!(
            u.feature,
            Feature::Lowering(LowerFamily::Simd),
            "{tag}: {u}"
        );
        assert_eq!(u.needs, Needs::NotImplemented, "{tag}: {u}");
        assert_at_instruction(&u, "uniform_k");
    }
}

/// A uniformity assertion whose result feeds only `llvm.assume` (directly, or through phis that
/// feed only assumptions) is the asserted `true`: it compiles, passes L3, and selects
/// `simd.assumed_uniform` once per call.
#[test]
fn an_assumed_uniformity_result_is_true() {
    let direct = "  %u = tail call i1 @air.is_uniform.i32(i32 %1)\n  tail call void \
                  @llvm.assume(i1 %u)\n  %s = add i32 %1, 7\n";
    let through_phis = "  %c = icmp ult i32 %1, 8\n  br i1 %c, label %a, label %b\na:\n  %ua = \
                        tail call i1 @air.is_uniform.i32(i32 %1)\n  br label %j\nb:\n  %ub = tail \
                        call i1 @air.is_uniform.i32(i32 0)\n  br label %j\nj:\n  %u = phi i1 [ \
                        %ua, %a ], [ %ub, %b ]\n  br label %k\nk:\n  %w = phi i1 [ %u, %j ]\n  \
                        tail call void @llvm.assume(i1 %w)\n  %s = add i32 %1, 7\n";
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    for (tag, body, calls) in [("assumed", direct, 1u32), ("assumed-phi", through_phis, 2)] {
        let Some(bc) = assemble(tag, &uniform_kernel(body)) else {
            return;
        };
        let a = compile(&bc, "uniform_k", &p).unwrap_or_else(|e| panic!("{tag}: {e}"));
        assert_eq!(
            a.strategies
                .entries
                .get("simd.assumed_uniform")
                .map(|e| e.uses),
            Some(calls),
            "{tag}: {:?}",
            a.strategies.entries
        );
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{tag}: L3: {e}"));
    }
}
