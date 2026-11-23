//! The sampler tail.
//!
//! - The read sampler (`air.get_read_sampler`) has no value: a texture read in another block than
//!   its definition compiles and passes L3 (the `engine/sampling_tail` fixture's kernel and
//!   fragment entries, the shape Metal's front end emits), and the kernel's reads compute their
//!   Metal result on the CPU interpreter in both blocks. Every other use of it, in another block
//!   too, stays a typed refusal.
//! - A dynamic texel offset reaches SPIR-V as a signed `int` vector, on a gather and on a sample,
//!   so SPIRV-Cross passes MSL's `sample` and `gather` the `int2` they take; the modules convert
//!   and compile as MoltenVK's MSL.
//! - A cube's offset never reaches the builder: the decoder admits no offset operand on a cube
//!   sample, so such a call is refused as input.
//! - A bicubic constexpr sampler's sample is Metal's measured filter emulated by fetches on every
//!   profile (the filter itself is checked in `src/tests/engine_bicubic.rs`).
//!
//! `llvm-as`, `spirv-cross`, and the Metal compiler are external; without them the tests skip
//! visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::device::profile::DeviceProfile;
use crate::device::DevicePolicy;
use crate::error::{Error, Feature, LowerFamily, Needs};
use crate::library::FunctionArtifact;
use crate::lower::select::Plan;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, metal_assemble, moltenvk_msl_compile,
    profile, typed_module, AssembleError,
};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

fn skip_missing<T>(r: Result<T, AssembleError>, what: &str) -> Option<T> {
    match r {
        Ok(x) => Some(x),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("{what}: {why}"),
    }
}

/// The bitcode of fixture `engine/<name>` (`.ll` through `llvm-as`, `.air.ll` through
/// `metal-as`).
fn fixture(name: &str) -> Option<Vec<u8>> {
    let path = engine_dir().join(name);
    let r = if name.ends_with(".air.ll") {
        metal_assemble(&path)
    } else {
        llvm_assemble(&path)
    };
    skip_missing(r, name)
}

/// Bitcode of the module `body` (after the AIR header), or `None` without `llvm-as`.
fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-sampler-tail-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, format!("{HEADER}{body}")).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    skip_missing(r, name)
}

fn m3() -> DeviceProfile {
    profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile")
}

fn uses(a: &FunctionArtifact, name: &str) -> u32 {
    a.strategies.entries.get(name).map_or(0, |u| u.uses)
}

/// The compiled, linked, relocated, L3-validated words of `entry` on `p`, and its artifact.
fn final_words(bc: &[u8], entry: &str, p: &DeviceProfile) -> (Vec<u32>, Arc<FunctionArtifact>) {
    let a = compile(bc, entry, p).unwrap_or_else(|e| panic!("{entry}: compile: {e}"));
    let words = link_and_validate(&a, p)
        .unwrap_or_else(|e| panic!("{entry}: L3: {e}"))
        .1;
    (words, a)
}

/// The fixture's reads use the entry block's read sampler in a later block. Both entries, in both
/// companions, compile and pass L3 on every committed profile, with one valueless read-sampler
/// strategy and two reads.
#[test]
fn reads_use_the_read_sampler_in_any_block() {
    for file in ["sampling_tail.ll", "sampling_tail.air.ll"] {
        let Some(bc) = fixture(file) else {
            continue;
        };
        for (stem, p) in committed_profiles().expect("profiles") {
            for entry in ["read_across_blocks", "fragment_read_across_blocks"] {
                let a = compile(&bc, entry, &p)
                    .unwrap_or_else(|e| panic!("{file} {entry} {stem}: compile: {e}"));
                link_and_validate(&a, &p)
                    .unwrap_or_else(|e| panic!("{file} {entry} {stem}: L3: {e}"));
                assert_eq!(uses(&a, "sampler.read_sampler"), 1, "{file} {entry} {stem}");
                // Clamped where the profile lacks robust image access.
                let reads = uses(&a, "texture.read") + uses(&a, "texture.read_clamped");
                assert_eq!(reads, 2, "{file} {entry} {stem}");
            }
        }
    }
}

/// A kernel whose read sampler `%rs`, defined in the entry block, is used by `use_` in block
/// `then` (taken for `gid < 4`); `done` may merge it (`merge`). `%t` is `texture2d<float,
/// sample>` slot 0, `%s` is `[[sampler(0)]]`, `%out` is `float4 *` at buffer 0.
fn other_block_kernel(use_: &str, merge: &str) -> String {
    format!(
        r#"define void @k(ptr addrspace(1) %t, ptr addrspace(2) %s, ptr addrspace(1) %out, i32 %gid) {{
entry:
  %rs = call ptr addrspace(2) @air.get_read_sampler()
  %c = icmp ult i32 %gid, 4
  br i1 %c, label %then, label %done

then:
  %g = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  {use_}
  br label %done

done:
  {merge}
  ret void
}}

declare ptr addrspace(2) @air.get_read_sampler()
declare {{ <4 x float>, i8 }} @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!6 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
"#
    )
}

const READ_THEN_STORE: &str = "%r = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %g, <2 x i32> zeroinitializer, i32 0, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  store <4 x float> %v, ptr addrspace(1) %out, align 16";

/// A read in another block compiles; a sample through the read sampler, a select of it, and a phi
/// of it, each in another block, are refused `Lowering(Sampler)` where they are selected (the read
/// sampler samples nothing and has no handle index).
#[test]
fn other_uses_of_the_read_sampler_stay_refused_in_any_block() {
    let p = m3();
    let Some(bc) = assemble("read", &other_block_kernel(READ_THEN_STORE, "")) else {
        return;
    };
    let (_, a) = final_words(&bc, "k", &p);
    assert_eq!(
        uses(&a, "texture.read") + uses(&a, "texture.read_clamped"),
        1
    );
    let sample = "%r = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x float> zeroinitializer, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  store <4 x float> %v, ptr addrspace(1) %out, align 16";
    let select = "%odd = trunc i32 %gid to i1
  %u = select i1 %odd, ptr addrspace(2) %rs, ptr addrspace(2) %s
  %r = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %u, <2 x i32> %g, <2 x i32> zeroinitializer, i32 0, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  store <4 x float> %v, ptr addrspace(1) %out, align 16";
    let phi = "%m = phi ptr addrspace(2) [ %rs, %entry ], [ %s, %then ]
  %r = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %m, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  store <4 x float> %v, ptr addrspace(1) %out, align 16";
    for (name, use_, merge) in [
        ("sample", sample, ""),
        ("select", select, ""),
        ("phi", "", phi),
    ] {
        let Some(bc) = assemble(name, &other_block_kernel(use_, merge)) else {
            return;
        };
        match compile(&bc, "k", &p) {
            Err(Error::Unsupported(u)) => assert_eq!(
                (u.feature, u.needs),
                (
                    Feature::Lowering(LowerFamily::Sampler),
                    Needs::NotImplemented
                ),
                "{name}"
            ),
            other => panic!("{name}: {:?}", other.map(|_| ())),
        }
    }
}

fn fl(x: f32) -> u32 {
    x.to_bits()
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Resources of a dispatch of one-thread threadgroups: `[[buffer(i)]]` holding `init[i]`, and the
/// execution status.
fn resources(plan: &Plan, init: &[Vec<u8>]) -> Resources {
    let mut spec = BTreeMap::new();
    spec.extend(crate::pipeline::threadgroup_constants([1, 1, 1]));
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let buffers: Vec<Buffer> = init
        .iter()
        .enumerate()
        .map(|(i, b)| Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: i as u32,
            }),
            address: None,
            bytes: b.clone(),
        })
        .collect();
    Resources {
        buffers,
        push,
        spec,
    }
}

/// `read_across_blocks` on the CPU interpreter: over a 4x2 grid with `n = 2`, `out[4y + x]` is
/// texel `(x, y)`, plus texel `(x + 1, y)` where `x < n` (the read in the branch), in both
/// scheduling orders, on every committed profile.
#[test]
fn reads_across_blocks_compute_their_texels_on_the_interpreter() {
    let Some(bc) = fixture("sampling_tail.ll") else {
        return;
    };
    let texel = |x: u32, y: u32| [fl(x as f32), fl(y as f32), fl((10 * x + y) as f32), fl(1.0)];
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let (plan, m): (Plan, TypedModule) =
            typed_module(&bc, "read_across_blocks", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let mut t = Image::new(
            ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Texture,
                index: 0,
            },
            [4, 2, 1],
            1,
            1,
            1,
        );
        for y in 0..2 {
            for x in 0..4 {
                let i = t.index(0, 0, 0, [x, y, 0]).expect("texel");
                t.texels[i] = texel(x, y);
            }
        }
        let res = resources(&plan, &[vec![0; 8 * 16], 2u32.to_le_bytes().to_vec()]);
        let mut outs = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let cfg = Config {
                workgroups: [4, 2, 1],
                order,
                ..Config::default()
            };
            let (o, _) = run_with_images(&m, &cfg, res.clone(), vec![t.clone()])
                .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
            outs.push(o.buffers);
        }
        assert_eq!(outs[0], outs[1], "{stem}: scheduling orders disagree");
        let out = words(&outs[0][0]);
        for y in 0..2u32 {
            for x in 0..4u32 {
                let a = texel(x, y).map(f32::from_bits);
                let b = if x < 2 {
                    texel(x + 1, y).map(f32::from_bits)
                } else {
                    [0.0; 4]
                };
                let want: Vec<u32> = (0..4).map(|k| fl(a[k] + b[k])).collect();
                let at = 4 * (4 * y + x) as usize;
                assert_eq!(out[at..at + 4], want[..], "{stem} ({x}, {y})");
            }
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// The instructions of a SPIR-V module: (opcode, words).
fn insts(words: &[u32]) -> Vec<(u16, &[u32])> {
    let mut out = Vec::new();
    let mut at = 5;
    while let Some(w) = words.get(at) {
        let n = ((w >> 16) as usize).max(1);
        let Some(inst) = words.get(at..at + n) else {
            break;
        };
        out.push(((w & 0xffff) as u16, inst));
        at += n;
    }
    out
}

const OP_TYPE_INT: u16 = 21;
const OP_TYPE_VECTOR: u16 = 23;
const OP_IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
const OP_IMAGE_GATHER: u16 = 96;

/// The `Offset` operand of every instruction `op` whose image-operand mask is word `mask_at`: the
/// operands follow the mask in bit order (`Bias`, `Lod` one id each, `Grad` two, `ConstOffset`
/// one, then `Offset`).
fn offset_operands(words: &[u32], op: u16, mask_at: usize) -> Vec<u32> {
    const OFFSET: u32 = 0x10;
    let mut out = Vec::new();
    for (o, inst) in insts(words) {
        if o != op {
            continue;
        }
        let mask = inst[mask_at];
        if mask & OFFSET == 0 {
            continue;
        }
        let before: usize = [(0x1, 1), (0x2, 1), (0x4, 2), (0x8, 1)]
            .iter()
            .filter(|(bit, _)| mask & bit != 0)
            .map(|(_, n)| n)
            .sum();
        out.push(inst[mask_at + 1 + before]);
    }
    out
}

/// Opcodes whose words are `[opcode, result type, result, ...]` and that can define an offset
/// value: `OpConstant`, `OpConstantComposite`, `OpLoad`, `OpVectorShuffle`,
/// `OpCompositeConstruct`, `OpCopyObject`, `OpBitcast`, `OpIAdd`, `OpExtInst`, and
/// `OpBitFieldSExtract`.
const VALUE_OPS: [u16; 10] = [43, 44, 61, 79, 80, 83, 124, 128, 12, 202];

/// The (width, signedness) of the integer components of the type of value `id`.
fn int_components(words: &[u32], id: u32) -> Option<(u32, u32)> {
    let all = insts(words);
    let ty = all
        .iter()
        .find(|(o, i)| VALUE_OPS.contains(o) && i.get(2) == Some(&id))
        .map(|(_, i)| i[1])?;
    // Type declarations are `[opcode, result, ...]`.
    let def = |t: u32| {
        all.iter()
            .find(|(o, i)| matches!(*o, OP_TYPE_INT | OP_TYPE_VECTOR) && i.get(1) == Some(&t))
    };
    let mut t = def(ty)?;
    if t.0 == OP_TYPE_VECTOR {
        t = def(t.1[2])?;
    }
    (t.0 == OP_TYPE_INT).then(|| (t.1[2], t.1[3]))
}

/// A dynamic offset reaches SPIR-V as a signed 32-bit `int` vector, on a gather
/// (`gather_dynamic_offset`) and on a sample with `maintenance8`
/// (`sampling_completion`'s `sample_dynamic_offset`), and both modules convert to MSL as MoltenVK
/// does and compile (an unsigned offset would reach MSL as a `uint2`, which `gather` and `sample`
/// reject).
#[test]
fn dynamic_offsets_are_signed_and_compile_as_moltenvk_msl() {
    let p = m3();
    for (file, entry, op, mask_at) in [
        (
            "sampling_tail.ll",
            "gather_dynamic_offset",
            OP_IMAGE_GATHER,
            6,
        ),
        (
            "sampling_completion.ll",
            "sample_dynamic_offset",
            OP_IMAGE_SAMPLE_EXPLICIT_LOD,
            5,
        ),
    ] {
        let Some(bc) = fixture(file) else {
            return;
        };
        let (words, _) = final_words(&bc, entry, &p);
        let offsets = offset_operands(&words, op, mask_at);
        assert_eq!(offsets.len(), 1, "{entry}");
        assert_eq!(
            int_components(&words, offsets[0]),
            Some((32, 1)),
            "{entry}: the Offset operand is a signed 32-bit int vector"
        );
        skip_missing(moltenvk_msl_compile(&words), entry);
    }
}

/// A cube sample has no offset operand in AIR (MSL's cube `sample` takes none), so a call that
/// passes an offset flag and an offset to it is refused as input before lowering: it never reaches
/// the builder as an internal error.
#[test]
fn a_cube_sample_with_an_offset_is_refused_as_input() {
    let body = r#"define void @k(ptr addrspace(1) %t, ptr addrspace(2) %s, ptr addrspace(1) %out) {
  %r = call { <4 x float>, i8 } @air.sample_texture_cube.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <3 x float> <float 1.0, float 0.0, float 0.0>, i1 true, <3 x i32> <i32 1, i32 0, i32 0>, i1 true, float 0.0, float 0.0, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  store <4 x float> %v, ptr addrspace(1) %out, align 16
  ret void
}

declare { <4 x float>, i8 } @air.sample_texture_cube.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, <3 x i32>, i1, float, float, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texturecube<float, sample>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
"#;
    let Some(bc) = assemble("cube_offset", body) else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        match compile(&bc, "k", &p) {
            Err(Error::InvalidInput(_)) => {}
            other => panic!("{stem}: {:?}", other.map(|_| ())),
        }
    }
}

/// The fixture's bicubic constexpr sampler is emulated by fetches, which every committed profile
/// compiles and validates.
#[test]
fn bicubic_samplers_are_emulated_on_every_profile() {
    let Some(bc) = fixture("sampling_tail.ll") else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = compile(&bc, "sample_bicubic", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(
            a.strategies.entries.get("sampler.bicubic").map(|u| u.uses),
            Some(1),
            "{stem}"
        );
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
    }
}
