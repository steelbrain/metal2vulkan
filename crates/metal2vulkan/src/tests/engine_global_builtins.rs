//! Global built-in variables: a kernel that reads its built-ins through Metal 3.1's program-scope
//! variables (`air.get_*` calls) compiles to the words of the same kernel reading them as built-in
//! parameters, on every committed profile, and runs as it does. A fragment function's global
//! `thread_index_in_quadgroup`, whose render pipeline Metal refuses, is refused as its own feature.

use std::collections::BTreeMap;

use crate::air::intrinsics::GlobalVar;
use crate::error::{Error, Feature, LowerFamily, Needs};
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, typed_module, AssembleError, Scratch,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::testing::committed_profiles;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// One built-in a kernel reads: its global variable and its element width (16 or 32).
#[derive(Clone, Copy)]
struct Read {
    var: GlobalVar,
    bits: u8,
}

impl Read {
    fn name(self) -> &'static str {
        GlobalVar::ALL
            .iter()
            .find(|(v, _, _)| *v == self.var)
            .map(|(_, n, _)| *n)
            .expect("listed")
    }

    /// The parameter metadata key: `air.<the Metal attribute>`.
    fn key(self) -> String {
        let metal = match self.var {
            GlobalVar::ThreadPositionInGrid => "thread_position_in_grid",
            GlobalVar::ThreadPositionInThreadgroup => "thread_position_in_threadgroup",
            GlobalVar::ThreadgroupPositionInGrid => "threadgroup_position_in_grid",
            GlobalVar::ThreadsPerThreadgroup => "threads_per_threadgroup",
            GlobalVar::ThreadsPerGrid => "threads_per_grid",
            GlobalVar::ThreadgroupsPerGrid => "threadgroups_per_grid",
            GlobalVar::DispatchThreadsPerThreadgroup => "dispatch_threads_per_threadgroup",
            GlobalVar::ThreadIndexInThreadgroup => "thread_index_in_threadgroup",
            GlobalVar::ThreadIndexInSimdgroup => "thread_index_in_simdgroup",
            GlobalVar::SimdgroupIndexInThreadgroup => "simdgroup_index_in_threadgroup",
            GlobalVar::SimdgroupsPerThreadgroup => "simdgroups_per_threadgroup",
            GlobalVar::DispatchSimdgroupsPerThreadgroup => "dispatch_simdgroups_per_threadgroup",
            GlobalVar::ThreadIndexInQuadgroup => "thread_index_in_quadgroup",
            GlobalVar::QuadgroupIndexInThreadgroup => "quadgroup_index_in_threadgroup",
            GlobalVar::QuadgroupsPerThreadgroup => "quadgroups_per_threadgroup",
        };
        format!("air.{metal}")
    }

    fn scalar(self) -> String {
        format!("i{}", self.bits)
    }

    fn ty(self) -> String {
        if self.var.is_vector() {
            format!("<3 x i{}>", self.bits)
        } else {
            self.scalar()
        }
    }

    fn type_name(self) -> &'static str {
        match (self.var.is_vector(), self.bits) {
            (true, 16) => "ushort3",
            (true, _) => "uint3",
            (false, 16) => "ushort",
            (false, _) => "uint",
        }
    }
}

/// The values a kernel stores: each read's lanes (0 to 2 for a vector built-in), widened to
/// `i32`, at `out[lin * n + j]`, `lin` its `thread_index_in_threadgroup`, read first.
/// `global`: through `air.get_*` calls; otherwise through built-in parameters, one per built-in
/// and width in the order the global form first reads them (the order canonicalization appends
/// them in), a vector's lane by `extractelement` and a scalar the parameter itself, as
/// canonicalization makes the calls.
fn kernel(reads: &[Read], global: bool) -> String {
    let lin = Read {
        var: GlobalVar::ThreadIndexInThreadgroup,
        bits: 32,
    };
    let mut all = vec![lin];
    all.extend_from_slice(reads);
    // The parameters: each built-in and width once, in first-read order.
    let mut order: Vec<(GlobalVar, u8)> = Vec::new();
    for r in &all {
        if !order.contains(&(r.var, r.bits)) {
            order.push((r.var, r.bits));
        }
    }
    let param = |r: &Read| -> String {
        let at = order
            .iter()
            .position(|o| *o == (r.var, r.bits))
            .expect("listed");
        format!("%{}", at + 1)
    };
    let mut body = String::new();
    let mut values = Vec::new();
    let mut decls = String::new();
    let mut tmp = 0;
    let mut next = |prefix: &str| {
        tmp += 1;
        format!("%{prefix}{tmp}")
    };
    let mut lin_value = String::new();
    for (i, r) in all.iter().enumerate() {
        let (sc, ty, p) = (r.scalar(), r.ty(), param(r));
        if global {
            let decl = if r.var.is_vector() {
                format!("declare {sc} @air.{}.{sc}({sc})\n", r.name())
            } else {
                format!("declare {sc} @air.{}.{sc}()\n", r.name())
            };
            if !decls.contains(&decl) {
                decls.push_str(&decl);
            }
        }
        let lanes: &[u32] = if r.var.is_vector() { &[0, 1, 2] } else { &[0] };
        for &l in lanes {
            let v = if global && r.var.is_vector() {
                let v = next("v");
                body.push_str(&format!(
                    "  {v} = tail call {sc} @air.{}.{sc}({sc} {l})\n",
                    r.name()
                ));
                v
            } else if global {
                let v = next("v");
                body.push_str(&format!(
                    "  {v} = tail call {sc} @air.{}.{sc}()\n",
                    r.name()
                ));
                v
            } else if r.var.is_vector() {
                let v = next("v");
                body.push_str(&format!("  {v} = extractelement {ty} {p}, {sc} {l}\n"));
                v
            } else {
                p.clone()
            };
            let w = if r.bits == 32 {
                v
            } else {
                let w = next("w");
                body.push_str(&format!("  {w} = zext {sc} {v} to i32\n"));
                w
            };
            if i == 0 {
                lin_value = w;
            } else {
                values.push(w);
            }
        }
    }
    let n = values.len();
    let base = next("b");
    body.push_str(&format!("  {base} = mul i32 {lin_value}, {n}\n"));
    for (j, v) in values.iter().enumerate() {
        let (ix, ex, ptr) = (next("i"), next("e"), next("p"));
        body.push_str(&format!(
            "  {ix} = add i32 {base}, {j}\n  {ex} = zext i32 {ix} to i64\n  {ptr} = getelementptr inbounds i32, ptr addrspace(1) %0, i64 {ex}\n  store i32 {v}, ptr addrspace(1) {ptr}, align 4\n"
        ));
    }
    let mut params = vec!["ptr addrspace(1) noundef \"air-buffer-no-alias\" %0".to_string()];
    let mut md = String::from(
        "!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}\n",
    );
    let mut arg_md = vec!["!12".to_string()];
    if !global {
        for (i, &(var, bits)) in order.iter().enumerate() {
            let r = Read { var, bits };
            let id = 20 + i;
            params.push(format!("{} noundef %{}", r.ty(), i + 1));
            md.push_str(&format!(
                "!{id} = !{{i32 {}, !\"{}\", !\"air.arg_type_name\", !\"{}\", !\"air.arg_name\", !\"b{i}\"}}\n",
                i + 1,
                r.key(),
                r.type_name()
            ));
            arg_md.push(format!("!{id}"));
        }
    }
    format!(
        "{HEADER}define void @k({}) {{\nentry:\n{body}  ret void\n}}\n\n{decls}\n\
         !air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @k, !10, !11}}\n!10 = !{{}}\n!11 = !{{{}}}\n{md}\
         !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n",
        params.join(", "),
        arg_md.join(", ")
    )
}

/// `text` assembled from a scratch file, or `None` (a skip) without `llvm-as`.
fn assemble(tag: &str, text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new(&format!("m2v-engine-global-builtins-{tag}"));
    let dir = scratch.path();
    let path = dir.join("kernel.ll");
    std::fs::write(&path, text).expect("write");
    let bc = llvm_assemble(&path);
    match bc {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {tag}: {why}\n{text}"),
    }
}

/// The global form and the parameter form of a kernel reading `reads`.
fn both(tag: &str, reads: &[Read]) -> Option<(Vec<u8>, Vec<u8>)> {
    let g = assemble(&format!("{tag}-global"), &kernel(reads, true))?;
    let p = assemble(&format!("{tag}-param"), &kernel(reads, false))?;
    Some((g, p))
}

fn every_read() -> Vec<Read> {
    let mut reads: Vec<Read> = GlobalVar::ALL
        .iter()
        .map(|&(var, _, _)| Read { var, bits: 32 })
        .collect();
    reads.push(Read {
        var: GlobalVar::ThreadsPerThreadgroup,
        bits: 16,
    });
    reads.push(Read {
        var: GlobalVar::ThreadPositionInGrid,
        bits: 16,
    });
    reads
}

/// A kernel reading every global built-in (each vector built-in at all three lanes, and `ushort3`
/// reads beside the `uint3` ones) compiles on every committed profile to the words of the same
/// kernel reading built-in parameters, or is refused as it is.
#[test]
fn global_builtins_compile_to_their_parameter_forms_words() {
    let Some((global, param)) = both("every", &every_read()) else {
        return;
    };
    let mut compiled = Vec::new();
    for (stem, p) in committed_profiles().expect("profiles") {
        let g = compile(&global, "k", &p).and_then(|a| {
            link_and_validate(&a, &p)
                .map(|(_, w)| w)
                .map_err(|e| panic!("{stem}: L3: {e}"))
        });
        let q = compile(&param, "k", &p).and_then(|a| {
            link_and_validate(&a, &p)
                .map(|(_, w)| w)
                .map_err(|e| panic!("{stem}: L3: {e}"))
        });
        match (g, q) {
            (Ok(g), Ok(q)) => {
                assert!(g == q, "{stem}: the forms' words differ");
                compiled.push(stem);
            }
            (Err(g), Err(q)) => assert_eq!(g.to_string(), q.to_string(), "{stem}"),
            (g, q) => panic!("{stem}: global {:?}, parameters {:?}", g.err(), q.err()),
        }
    }
    assert!(
        compiled.iter().any(|s| s == "moltenvk-m3"),
        "compiled on {compiled:?}"
    );
}

/// Every word of buffer 0 (`words` of them) after one workgroup of local size `local` ran the
/// kernel `bc` on `moltenvk-m3` with 32-wide subgroups, identical in both scheduling orders.
fn run_kernel(bc: &[u8], local: [u32; 3], words: usize) -> Vec<u32> {
    let p = crate::pipeline::testing::profile(
        "moltenvk-m3",
        crate::device::DevicePolicy::RESIDENT_TABLES,
    )
    .expect("profile");
    let (plan, m) = typed_module(bc, "k", &p).expect("plan");
    let mut spec = BTreeMap::new();
    spec.extend(crate::pipeline::threadgroup_constants(local));
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        assert!(matches!(f.loc, PushLoc::Push { .. }), "{f:?}");
        assert!(
            matches!(f.field, PushField::SlotBindOffset(_)),
            "unexpected push field {f:?}"
        );
    }
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0; words * 4],
        }],
        push,
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [1, 1, 1],
            subgroup: SubgroupSize::W32,
            order,
            ..Config::default()
        };
        let out = run(&m, &cfg, res.clone()).expect("run");
        outs.push(
            out.buffers[0]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<u32>>(),
        );
    }
    assert_eq!(outs[0], outs[1], "order independent");
    outs.swap_remove(0)
}

/// The threadgroup's built-ins read through global variables (its size as `uint3` and `ushort3`,
/// the thread's position, its simdgroup and quad lanes and indices) hold what the parameters hold,
/// in threadgroups of several shapes: `threads_per_threadgroup.y` of a 8x4 threadgroup reads 4 in
/// every thread.
#[test]
fn global_builtins_run_as_their_parameters() {
    let reads: Vec<Read> = [
        (GlobalVar::ThreadsPerThreadgroup, 32),
        (GlobalVar::ThreadPositionInThreadgroup, 32),
        (GlobalVar::ThreadIndexInSimdgroup, 32),
        (GlobalVar::SimdgroupIndexInThreadgroup, 32),
        (GlobalVar::SimdgroupsPerThreadgroup, 32),
        (GlobalVar::ThreadIndexInQuadgroup, 32),
        (GlobalVar::QuadgroupIndexInThreadgroup, 32),
        (GlobalVar::QuadgroupsPerThreadgroup, 32),
        (GlobalVar::ThreadsPerThreadgroup, 16),
    ]
    .into_iter()
    .map(|(var, bits)| Read { var, bits })
    .collect();
    // Three lanes per vector read, one per scalar one.
    let per_thread: usize = reads
        .iter()
        .map(|r| if r.var.is_vector() { 3 } else { 1 })
        .sum();
    let Some((global, param)) = both("run", &reads) else {
        return;
    };
    for local in [[8, 4, 1], [32, 1, 1], [16, 2, 2], [5, 3, 1]] {
        let n = local.iter().product::<u32>() as usize;
        let g = run_kernel(&global, local, n * per_thread);
        let q = run_kernel(&param, local, n * per_thread);
        assert_eq!(g, q, "{local:?}");
        for t in 0..n {
            let at = t * per_thread;
            assert_eq!(&g[at..at + 3], &local[..], "{local:?} thread {t}");
            assert_eq!(
                &g[at + per_thread - 3..at + per_thread],
                &local[..],
                "{local:?} thread {t}: ushort3"
            );
        }
    }
}

/// A global built-in read at a lane that is no constant below 3 is refused
/// `NotImplemented(Lowering(Dispatch))` at the call.
#[test]
fn a_global_builtin_read_at_an_unknown_lane_is_refused() {
    let text = format!(
        "{HEADER}define void @k(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{\nentry:\n  %l = load i32, ptr addrspace(1) %0, align 4\n  %v = tail call i32 @air.get_local_size.i32(i32 %l)\n  store i32 %v, ptr addrspace(1) %0, align 4\n  ret void\n}}\n\n\
         declare i32 @air.get_local_size.i32(i32)\n\n\
         !air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @k, !10, !11}}\n!10 = !{{}}\n!11 = !{{!12}}\n\
         !12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
         !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    );
    let Some(bc) = assemble("dynamic", &text) else {
        return;
    };
    let p = crate::pipeline::testing::profile(
        "moltenvk-m3",
        crate::device::DevicePolicy::RESIDENT_TABLES,
    )
    .expect("profile");
    match compile(&bc, "k", &p) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::Lowering(LowerFamily::Dispatch), "{u:?}")
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
}

/// The owned fixture `gsm_global_builtins_fragment_quad`, a fragment function reading a global
/// `thread_index_in_quadgroup` (whose render pipeline the M4's and the M3 Max's Metal refuse), is
/// refused `GlobalBuiltinInFragment` `NotImplemented`, naming its call, on every committed profile;
/// `gsm_global_builtins_fragment`, one reading a global `threads_per_simdgroup` (which Metal runs),
/// compiles on each.
#[test]
fn a_fragment_s_global_quad_lane_is_refused_as_its_own_feature() {
    let fixture = |stem: &str| match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}: {why}"),
    };
    let (Some(quad), Some(width)) = (
        fixture("gsm_global_builtins_fragment_quad"),
        fixture("gsm_global_builtins_fragment"),
    ) else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        match compile(&quad, "global_quad_lane_in_a_fragment", &p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    (u.feature, u.needs),
                    (Feature::GlobalBuiltinInFragment, Needs::NotImplemented),
                    "{stem}"
                );
                assert_eq!(
                    u.symbol.map(|s| s.to_string()).as_deref(),
                    Some("air.get_thread_index_in_quadgroup.i32"),
                    "{stem}"
                );
            }
            other => panic!("{stem}: {:?}", other.map(|_| ())),
        }
        if let Err(e) = compile(&width, "global_builtins_in_a_fragment", &p) {
            panic!("{stem}: {e}");
        }
    }
}
