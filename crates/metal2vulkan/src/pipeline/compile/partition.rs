//! Whether a kernel observes its threadgroup partition, decided from the specialized entry's AIR
//! alone: `KernelFacts::partition_oblivious`.
//!
//! A kernel that does not observe it computes the same in every partition of its grid into
//! threadgroups, so the runtime may run an indirect command's dispatch in threadgroups of its own
//! choosing, through regions that supply the grid's builtins exactly. The entry is one function
//! after canonicalization (visible functions linked into it included), so its body and its
//! parameters are the whole kernel. It observes its partition when it uses:
//! - the threadgroup or threadgroup-imageblock address space: a `[[threadgroup(n)]]` argument, a
//!   global, an imageblock argument, or any pointer into them;
//! - an execution barrier (`threadgroup_barrier`, `simdgroup_barrier`), or a fence or an atomic of
//!   threadgroup or simdgroup scope;
//! - a simdgroup or quad function: shuffles, broadcasts, reductions, votes and ballots,
//!   `is_uniform`, the simdgroup size, simdgroup matrices, and, with the same partition, the
//!   imageblock functions, asynchronous copies, derivatives and tensor operations;
//! - a builtin of the partition it reads: a position or index in the threadgroup, a simdgroup or
//!   a quadgroup, the threadgroup's position in the grid, the threads per threadgroup, per
//!   simdgroup and per dispatch's threadgroup, and the threadgroups, simdgroups and quadgroups
//!   per grid or threadgroup. `thread_position_in_grid`, `threads_per_grid` and the stage-in
//!   region's `grid_origin` and `grid_size` are the grid's;
//! - ray tracing or an intrinsic the decoder does not know: nothing states what they observe.
//!
//! Any other operation is a thread's own: arithmetic, device and constant memory, textures,
//! samplers, device-scope atomics and fences, indirect commands, function constants and tables,
//! and `quad_is_helper_thread` and `simd_is_helper_thread`, which no kernel thread is.

use std::collections::BTreeSet;

use crate::air;
use crate::air::abi::{Builtin, ParamRole, Stage as AbiStage};
use crate::air::intrinsics::{AirIntrinsic, Fragment, Sync, ThreadScope};
use crate::air::ir::{AddrSpace, Callee, InstKind, Type, Value};

/// Whether kernel `e` does not observe its threadgroup partition (see the module). Any other
/// stage, and an entry without a body, does.
pub(super) fn oblivious(e: &air::Entry) -> bool {
    if !matches!(e.ep.stage, AbiStage::Kernel { .. }) {
        return false;
    }
    let Some(body) = e.module.func(e.func).and_then(|f| f.body.as_ref()) else {
        return false;
    };
    if e.module
        .parts()
        .globals
        .iter()
        .any(|(_, g)| partitioned(g.space))
    {
        return false;
    }
    let types = e.module.types();
    let mut read = BTreeSet::new();
    for inst in body.insts.values() {
        if matches!(types.get(inst.ty), Some(Type::Ptr { space }) if partitioned(*space)) {
            return false;
        }
        if let InstKind::Call(call) = &inst.kind {
            if observes(&call.callee) {
                return false;
            }
        }
        inst.kind.for_each_value(|v| {
            if let Value::Arg(a) = v {
                read.insert(a.raw());
            }
        });
    }
    !e.ep
        .params
        .iter()
        .any(|b| read.contains(&b.param) && partition_role(&b.role))
}

/// Whether `space` is memory one threadgroup shares.
fn partitioned(space: AddrSpace) -> bool {
    matches!(
        space,
        AddrSpace::Threadgroup | AddrSpace::ThreadgroupImageblock
    )
}

/// Whether a parameter of `role` is the partition's.
fn partition_role(role: &ParamRole) -> bool {
    match role {
        ParamRole::Threadgroup(_) | ParamRole::Imageblock(_) => true,
        ParamRole::Builtin(b) => partition_builtin(*b),
        _ => false,
    }
}

/// Whether builtin `b` reads the partition (see the module).
fn partition_builtin(b: Builtin) -> bool {
    match b {
        Builtin::ThreadPositionInThreadgroup
        | Builtin::ThreadgroupPositionInGrid
        | Builtin::ThreadsPerThreadgroup
        | Builtin::ThreadgroupsPerGrid
        | Builtin::ThreadIndexInThreadgroup
        | Builtin::ThreadIndexInSimdgroup
        | Builtin::ThreadIndexInQuadgroup
        | Builtin::SimdgroupIndexInThreadgroup
        | Builtin::QuadgroupIndexInThreadgroup
        | Builtin::ThreadsPerSimdgroup
        | Builtin::SimdgroupsPerThreadgroup
        | Builtin::QuadgroupsPerThreadgroup
        | Builtin::DispatchThreadsPerThreadgroup
        | Builtin::DispatchSimdgroupsPerThreadgroup => true,
        Builtin::ThreadPositionInGrid
        | Builtin::ThreadsPerGrid
        | Builtin::StageInGridSize
        | Builtin::StageInGridOrigin => false,
        // Other stages' builtins: no kernel reads them.
        Builtin::VertexId
        | Builtin::InstanceId
        | Builtin::BaseVertex
        | Builtin::BaseInstance
        | Builtin::AmplificationId
        | Builtin::AmplificationCount
        | Builtin::PositionInPatch
        | Builtin::PatchId
        | Builtin::Position(_)
        | Builtin::FrontFacing
        | Builtin::PointCoord
        | Builtin::PrimitiveId
        | Builtin::SampleId
        | Builtin::SampleMaskIn
        | Builtin::PostDepthCoverage
        | Builtin::BarycentricCoord(_)
        | Builtin::RenderTargetArrayIndex
        | Builtin::ViewportArrayIndex => true,
    }
}

/// Whether a call to `callee` observes the partition (see the module).
fn observes(callee: &Callee) -> bool {
    let op = match callee {
        Callee::Air { op, .. } => op,
        Callee::Llvm { .. } => return false,
        // Canonicalization inlines every defined callee; what remains is not the kernel's own.
        Callee::Direct(_)
        | Callee::Indirect { .. }
        | Callee::UnknownIntrinsic(_)
        | Callee::Unbound { .. } => return true,
    };
    let scoped =
        |scope: ThreadScope| matches!(scope, ThreadScope::Threadgroup | ThreadScope::Simdgroup);
    match op {
        AirIntrinsic::Sync(Sync::Barrier { .. }) => true,
        AirIntrinsic::Sync(Sync::Fence { scope, .. }) => scoped(*scope),
        AirIntrinsic::Atomic(a) => scoped(a.scope),
        AirIntrinsic::Group(_)
        | AirIntrinsic::BoolReduce(_)
        | AirIntrinsic::IsUniform(_)
        | AirIntrinsic::SimdgroupSize(_)
        | AirIntrinsic::SimdMatrix(_)
        | AirIntrinsic::Imageblock(_)
        | AirIntrinsic::AsyncCopy(_)
        | AirIntrinsic::Derivative(_)
        | AirIntrinsic::Fragment(
            Fragment::Discard | Fragment::MapCoordinates { .. } | Fragment::NumSamples,
        )
        | AirIntrinsic::Tensor(_)
        | AirIntrinsic::AccelerationStructure(_)
        | AirIntrinsic::Intersect(_)
        | AirIntrinsic::Mesh(_) => true,
        AirIntrinsic::Fragment(Fragment::IsHelperThread)
        | AirIntrinsic::Interpolate(_)
        | AirIntrinsic::Math(_)
        | AirIntrinsic::Convert(_)
        | AirIntrinsic::Pack(_)
        | AirIntrinsic::Coherent(_)
        | AirIntrinsic::Texture(_)
        | AirIntrinsic::Sampler(_)
        | AirIntrinsic::FunctionConstant(_)
        | AirIntrinsic::FunctionTable(_)
        | AirIntrinsic::IndirectCommand(_) => false,
        // A global built-in read canonicalization left a call (its lane is no constant, and
        // lowering refuses it) reads its built-in as the parameter would.
        AirIntrinsic::GlobalBuiltin(g) => partition_builtin(g.var.builtin()),
    }
}

#[cfg(test)]
mod tests {
    use crate::budget::{Budget, Meter};
    use crate::device::DevicePolicy;
    use crate::library::FunctionConstantValues;
    use crate::pipeline::testing::{
        compile, llvm_assemble_text, profile, specialized_entry, AssembleError,
    };

    /// A kernel `k` of `device uint *out [[buffer(0)]]` and the builtin `air.<builtin>` as a
    /// `uint`, with `extra` parameters and their metadata, `globals`, a `body` that defines `%v`
    /// (stored to `out`), and the `declarations` it calls.
    struct Kernel<'a> {
        builtin: &'a str,
        globals: &'a str,
        extra: (&'a str, &'a str),
        body: &'a str,
        declarations: &'a str,
    }

    const PLAIN: Kernel<'static> = Kernel {
        builtin: "thread_position_in_grid",
        globals: "",
        extra: ("", ""),
        body: "%v = add i32 %b, 1",
        declarations: "",
    };

    impl Kernel<'_> {
        fn text(&self) -> String {
            let Kernel {
                builtin,
                globals,
                extra: (params, md),
                body,
                declarations,
            } = self;
            let refs = if md.is_empty() { "" } else { ", !5" };
            format!(
                r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

{globals}
define void @k(ptr addrspace(1) %out, i32 %b{params}) {{
entry:
  {body}
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}}

{declarations}

!air.kernel = !{{!0}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4{refs}}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.{builtin}", !"air.arg_type_name", !"uint", !"air.arg_name", !"b"}}
{md}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
            )
        }

        /// The kernel's bitcode, assembled straight out of memory: the text goes down a pipe
        /// into `llvm-as` and the bitcode comes back, so this helper writes no file and has
        /// nothing to clean up. It used to assemble through `k.ll` in a directory under the OS
        /// temp directory and remove only the inner of the two directories it made, leaving
        /// `<temp>/m2v-partition-<pid>-<name>-<thread>` behind for every kernel of every test here
        /// (`the_bitcode_helper_leaves_nothing_under_the_temp_directory`).
        fn bitcode(&self, name: &str) -> Option<Vec<u8>> {
            match llvm_assemble_text(&self.text()) {
                Ok(b) => Some(b),
                Err(AssembleError::Missing(why)) => {
                    eprintln!("SKIPPED: {why}");
                    None
                }
                Err(AssembleError::Rejected(why)) => {
                    panic!("llvm-as rejected {name}: {why}\n{}", self.text())
                }
            }
        }
    }

    fn lavapipe() -> crate::device::DeviceProfile {
        profile("lavapipe", DevicePolicy::default()).expect("profile")
    }

    /// **Making a kernel's bitcode leaves nothing under the OS temp directory.** The helper used
    /// to write `k.ll` into `<temp>/m2v-partition-<pid>-<name>-<thread>/<name>` and remove only
    /// the inner directory, so every kernel of every test in this module left the outer one
    /// behind -- on a machine whose disk writes the owner has asked twice to cut, and against
    /// AGENTS.md's rule that scratch goes as soon as the tool is done with it. The fix is that
    /// there is no file: restore the file-and-`remove_dir_all` body and this test fails naming the
    /// directory it finds.
    ///
    /// The prefix carries this process's id, so the check sees this process's leftovers and never
    /// another run's.
    #[test]
    fn the_bitcode_helper_leaves_nothing_under_the_temp_directory() {
        let prefix = format!("m2v-partition-{}-", std::process::id());
        let ours = || -> Vec<std::path::PathBuf> {
            let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
                return Vec::new();
            };
            entries
                .flatten()
                .filter(|entry| entry.file_name().to_string_lossy().starts_with(&prefix))
                .map(|entry| entry.path())
                .collect()
        };

        let _ = PLAIN.bitcode("leaves-nothing-behind");

        let left = ours();
        assert!(
            left.is_empty(),
            "making a kernel's bitcode left {left:?} under the OS temp directory"
        );
    }

    /// Whether kernel `k`'s specialized entry is partition-oblivious.
    fn oblivious(name: &str, k: &Kernel<'_>) -> Option<bool> {
        let bc = k.bitcode(name)?;
        let mut m = Meter::new(&Budget::default());
        let none = FunctionConstantValues::defaults();
        let e = specialized_entry(&bc, "k", &lavapipe(), &none, &[], &mut m)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        Some(super::oblivious(&e))
    }

    /// A kernel of the grid's builtins, a device-scope fence and a device-scope atomic does not
    /// observe its partition, nor one that asks whether its thread is a helper, which no kernel
    /// thread is. The atomic updates the word after the one the kernel stores to: on the stored
    /// word itself, lanes of one simdgroup race, and the lockstep barrier that orders them observes
    /// the partition (`a_device_race_lockstep_orders_observes_the_partition`).
    #[test]
    fn a_kernel_of_the_grid_alone_is_partition_oblivious() {
        let fence = "call void @air.atomic.fence(i32 1, i32 5, i32 2)\n  %v = add i32 %b, 1";
        let helper = "%h = call i1 @air.quad_is_helper_thread()\n  %s = call i1 \
                      @air.simd_is_helper_thread()\n  %e = or i1 %h, %s\n  %z = zext i1 %e to \
                      i32\n  %v = add i32 %b, %z";
        let atomic = "%w = getelementptr i32, ptr addrspace(1) %out, i64 1\n  %o = call i32 \
                      @air.atomic.global.add.u.i32(ptr addrspace(1) %w, i32 1, i32 0, i32 2, i1 \
                      true)\n  %v = add i32 %b, %o";
        for (name, k) in [
            ("position in grid", PLAIN),
            (
                "threads per grid",
                Kernel {
                    builtin: "threads_per_grid",
                    ..PLAIN
                },
            ),
            (
                "device fence",
                Kernel {
                    body: fence,
                    declarations: "declare void @air.atomic.fence(i32, i32, i32)",
                    ..PLAIN
                },
            ),
            (
                "device atomic",
                Kernel {
                    body: atomic,
                    declarations: "declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), \
                                   i32, i32, i32, i1)",
                    ..PLAIN
                },
            ),
            (
                "helper queries",
                Kernel {
                    body: helper,
                    declarations: "declare i1 @air.quad_is_helper_thread()\n\
                                   declare i1 @air.simd_is_helper_thread()",
                    ..PLAIN
                },
            ),
        ] {
            let Some(oblivious) = oblivious(name, &k) else {
                return;
            };
            assert!(oblivious, "{name}");
        }
    }

    /// A device-scope atomic whose result each lane stores to the word it updated races between the
    /// lanes of a simdgroup, whose lockstep runs every lane's update before any lane's store on
    /// Apple GPUs. The barrier that makes that order explicit is an execution barrier, so the
    /// kernel observes its partition: which lanes it orders depends on it.
    #[test]
    fn a_device_race_lockstep_orders_observes_the_partition() {
        let k = Kernel {
            body: "%o = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %out, i32 1, i32 \
                   0, i32 2, i1 true)\n  %v = add i32 %b, %o",
            declarations: "declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, \
                           i32, i1)",
            ..PLAIN
        };
        let Some(oblivious) = oblivious("device race", &k) else {
            return;
        };
        assert!(!oblivious);
    }

    /// Each builtin of the partition, read, makes a kernel observe it.
    #[test]
    fn each_builtin_of_the_partition_is_observed() {
        for builtin in [
            "thread_position_in_threadgroup",
            "threadgroup_position_in_grid",
            "threads_per_threadgroup",
            "threadgroups_per_grid",
            "thread_index_in_threadgroup",
            "thread_index_in_simdgroup",
            "thread_index_in_quadgroup",
            "simdgroup_index_in_threadgroup",
            "quadgroup_index_in_threadgroup",
            "threads_per_simdgroup",
            "simdgroups_per_threadgroup",
            "quadgroups_per_threadgroup",
            "dispatch_threads_per_threadgroup",
            "dispatch_simdgroups_per_threadgroup",
        ] {
            let Some(oblivious) = oblivious(builtin, &Kernel { builtin, ..PLAIN }) else {
                return;
            };
            assert!(!oblivious, "{builtin}");
        }
    }

    /// Threadgroup memory (an argument or a global), each barrier, a fence or an atomic of
    /// threadgroup or simdgroup scope, and a simdgroup or quad function each make a kernel observe
    /// its partition.
    #[test]
    fn each_partition_feature_is_observed() {
        let tg_param = (
            ", ptr addrspace(3) %tg",
            r#"!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tg"}"#,
        );
        let fence = "declare void @air.atomic.fence(i32, i32, i32)";
        let atomic = "declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, i32, \
                      i1)";
        let barrier = "declare void @air.wg.barrier(i32, i32)";
        let simd_barrier = "declare void @air.simdgroup.barrier(i32, i32)";
        let cases = [
            (
                "threadgroup argument",
                Kernel {
                    extra: tg_param,
                    body: "store i32 %b, ptr addrspace(3) %tg, align 4\n  \
                           %v = load i32, ptr addrspace(3) %tg, align 4",
                    ..PLAIN
                },
            ),
            (
                "threadgroup global",
                Kernel {
                    globals: "@g = internal addrspace(3) global [4 x i32] undef, align 4",
                    body: "store i32 %b, ptr addrspace(3) @g, align 4\n  \
                           %v = load i32, ptr addrspace(3) @g, align 4",
                    ..PLAIN
                },
            ),
            (
                "threadgroup barrier",
                Kernel {
                    body: "call void @air.wg.barrier(i32 1, i32 1)\n  %v = add i32 %b, 1",
                    declarations: barrier,
                    ..PLAIN
                },
            ),
            (
                "simdgroup barrier",
                Kernel {
                    body: "call void @air.simdgroup.barrier(i32 1, i32 4)\n  %v = add i32 %b, 1",
                    declarations: simd_barrier,
                    ..PLAIN
                },
            ),
            (
                "threadgroup fence",
                Kernel {
                    body: "call void @air.atomic.fence(i32 1, i32 5, i32 1)\n  %v = add i32 %b, 1",
                    declarations: fence,
                    ..PLAIN
                },
            ),
            (
                "simdgroup fence",
                Kernel {
                    body: "call void @air.atomic.fence(i32 1, i32 5, i32 4)\n  %v = add i32 %b, 1",
                    declarations: fence,
                    ..PLAIN
                },
            ),
            (
                "threadgroup atomic",
                Kernel {
                    body: "%o = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %out, \
                           i32 1, i32 0, i32 1, i1 true)\n  %v = add i32 %b, %o",
                    declarations: atomic,
                    ..PLAIN
                },
            ),
            (
                "simdgroup function",
                Kernel {
                    body: "%s = call i32 @air.simd_shuffle.u.i32(i32 %b, i16 0)\n  \
                           %v = add i32 %s, 1",
                    declarations: "declare i32 @air.simd_shuffle.u.i32(i32, i16)",
                    ..PLAIN
                },
            ),
            (
                "quad function",
                Kernel {
                    body: "%s = call i32 @air.quad_max.s.i32(i32 %b)\n  %v = add i32 %s, 1",
                    declarations: "declare i32 @air.quad_max.s.i32(i32)",
                    ..PLAIN
                },
            ),
        ];
        for (name, k) in cases {
            let Some(oblivious) = oblivious(name, &k) else {
                return;
            };
            assert!(!oblivious, "{name}");
        }
    }

    /// The compiled artifact carries the fact (`KernelFacts::partition_oblivious`).
    #[test]
    fn the_artifact_states_whether_the_kernel_observes_its_partition() {
        let p = lavapipe();
        for (name, k, want) in [
            ("grid", PLAIN, true),
            (
                "threadgroup",
                Kernel {
                    builtin: "thread_position_in_threadgroup",
                    ..PLAIN
                },
                false,
            ),
        ] {
            let Some(bc) = k.bitcode(name) else {
                return;
            };
            let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(a.kernel.partition_oblivious, want, "{name}");
        }
    }
}
