//! The texture tail: status words carried by copies, the residency their bit 0 reports, the texture
//! atomic, and the refusals that stay typed.
//!
//! Every compiled module passes `spirv-val` under its profile (`G-val`); texels are checked on the
//! CPU interpreter's image model where it models the instructions (reads and writes; the texel
//! pointer of an image atomic is not modelled, so the atomic is checked structurally).
//!
//! `llvm-as` assembles the synthetic kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::{SparseTileMap, TileMapping};
use crate::device::profile::DeviceProfile;
use crate::device::DevicePolicy;
use crate::error::{Error, Feature, LowerFamily, Unsupported};
use crate::library::FunctionArtifact;
use crate::lower::select::Plan;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, profile, typed_module, AssembleError,
};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;

const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let text = format!(
        "target datalayout = \"{DATALAYOUT}\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n{body}"
    );
    let dir = std::env::temp_dir().join(format!(
        "m2v-texture-tail-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, &text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}\n{text}"),
    }
}

fn ok(
    a: Result<std::sync::Arc<FunctionArtifact>, Error>,
    p: &DeviceProfile,
    what: &str,
) -> std::sync::Arc<FunctionArtifact> {
    let a = a.unwrap_or_else(|e| panic!("{what}: compile: {e}"));
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
    a
}

fn uses(a: &FunctionArtifact, name: &str) -> u32 {
    a.strategies.entries.get(name).map_or(0, |u| u.uses)
}

fn texture_not_implemented() -> Unsupported {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::Texture))
}

/// `r` is the typed `Lowering(Texture)` refusal.
fn refused_texture(r: Result<std::sync::Arc<FunctionArtifact>, Error>, what: &str) {
    let want = texture_not_implemented();
    match r {
        Err(Error::Unsupported(u)) => {
            assert_eq!((u.feature, u.needs), (want.feature, want.needs), "{what}")
        }
        other => panic!("{what}: {:?}", other.map(|_| ())),
    }
}

// ---- The interpreter harness (as `src/tests/engine_texture_interp.rs`) ----

/// Resources of a dispatch of 1x1x1 threadgroups, `[[buffer(i)]]` holding `init[i]`.
fn resources(plan: &Plan, init: &[Vec<u8>]) -> Resources {
    let mut spec = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(axis), 1);
    }
    spec.insert(SpecKey::FlatLocalSize, 1);
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            // A bound value read as data, which the interpreter writes from `spec`.
            (PushField::Specialized(_), PushLoc::Push { .. }) => {}
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

fn texture(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Texture,
        index,
    }
}

/// Run over `grid` threadgroups of one thread in both scheduling orders; both must agree.
fn run_both(
    m: &TypedModule,
    res: &Resources,
    images: &[Image],
    grid: [u32; 3],
) -> (Vec<Vec<u8>>, Vec<Image>) {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: grid,
            order,
            ..Config::default()
        };
        let (o, imgs) = run_with_images(m, &cfg, res.clone(), images.to_vec()).expect("run");
        outs.push((o.buffers, imgs));
    }
    assert_eq!(outs[0], outs[1], "scheduling orders disagree");
    outs.swap_remove(0)
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

// ---- Status words carried by copies ----

/// A kernel over two `texture2d<float, read>` slots 0 and 1 and `float4 *out` (buffer 0): each
/// thread reads texel `gid` of slot 0 on even `gid.x` and of slot 1 on odd `gid.x` in two blocks,
/// merges the `{T, i8}` results as `merge` spells (`%r` is the merged value), runs `tail`, and
/// stores `%v` to `out[gid.x + 2 * gid.y]`.
fn merged_reads(merge: &str, tail: &str) -> String {
    format!(
        r#"define void @k(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, <2 x i32> %gid) {{
entry:
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %odd = and i32 %x, 1
  %c = icmp eq i32 %odd, 0
  br i1 %c, label %even, label %other

even:
  %s0 = call ptr addrspace(2) @air.get_read_sampler()
  %r0 = call {{ <4 x float>, i8 }} @air.read_texture_2d.v4f32(ptr addrspace(1) %a, ptr addrspace(2) %s0, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  br label %join

other:
  %s1 = call ptr addrspace(2) @air.get_read_sampler()
  %r1 = call {{ <4 x float>, i8 }} @air.read_texture_2d.v4f32(ptr addrspace(1) %b, ptr addrspace(2) %s1, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  br label %join

join:
  {merge}
  {tail}
  %y2 = shl i32 %y, 1
  %i = add i32 %x, %y2
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %i
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}}

declare ptr addrspace(2) @air.get_read_sampler()
declare {{ <4 x float>, i8 }} @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"a"}}
!4 = !{{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"b"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!6 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}}
"#
    )
}

const PHI: &str = "%r = phi { <4 x float>, i8 } [ %r0, %even ], [ %r1, %other ]";

/// `%r` as values that carry the merged read `%q`'s status word on through member insertions:
/// `%q` with its texel written back into member 0 (as Apple's GL code generator writes a swizzled
/// read), a `{T, i8}` rebuilt from `%q`'s two members, `%q` nested in a larger aggregate and taken
/// out again, and a select of `%q` and its rewritten copy.
fn carried_reads() -> Vec<(&'static str, String)> {
    let q = "%q = phi { <4 x float>, i8 } [ %r0, %even ], [ %r1, %other ]";
    let t = "%q0 = extractvalue { <4 x float>, i8 } %q, 0";
    vec![
        (
            "texel reinserted",
            format!("{q}\n  {t}\n  %r = insertvalue {{ <4 x float>, i8 }} %q, <4 x float> %q0, 0"),
        ),
        (
            "rebuilt",
            format!(
                "{q}\n  {t}\n  %q1 = extractvalue {{ <4 x float>, i8 }} %q, 1\n  %a0 = insertvalue {{ <4 x float>, i8 }} undef, <4 x float> %q0, 0\n  %r = insertvalue {{ <4 x float>, i8 }} %a0, i8 %q1, 1"
            ),
        ),
        (
            "nested",
            format!(
                "{q}\n  %nest = insertvalue {{ {{ <4 x float>, i8 }}, i32 }} undef, {{ <4 x float>, i8 }} %q, 0\n  %r = extractvalue {{ {{ <4 x float>, i8 }}, i32 }} %nest, 0"
            ),
        ),
        (
            "select of reinserted",
            format!(
                "{q}\n  {t}\n  %w = insertvalue {{ <4 x float>, i8 }} %q, <4 x float> %q0, 0\n  %r = select i1 %c, {{ <4 x float>, i8 }} %w, {{ <4 x float>, i8 }} %q"
            ),
        ),
    ]
}

/// The merges whose texel only is taken: a phi, a frozen phi, a select of the frozen phi and a
/// constant, the phi with its word overwritten, and each of [`carried_reads`].
fn texel_only_merges() -> Vec<(&'static str, String)> {
    let take = "%v = extractvalue { <4 x float>, i8 } %r2, 0";
    let carried = carried_reads().into_iter().map(|(n, head)| {
        (
            n,
            format!("{head}\n  %r2 = freeze {{ <4 x float>, i8 }} %r"),
        )
    });
    vec![
        (
            "word overwritten",
            format!("{PHI}\n  %r2 = insertvalue {{ <4 x float>, i8 }} %r, i8 0, 1"),
        ),
        (
            "phi",
            "%r2 = phi { <4 x float>, i8 } [ %r0, %even ], [ %r1, %other ]".to_owned(),
        ),
        ("freeze", format!("{PHI}\n  %r2 = freeze {{ <4 x float>, i8 }} %r")),
        (
            "select",
            format!(
                "{PHI}\n  %f = freeze {{ <4 x float>, i8 }} %r\n  %r2 = select i1 true, {{ <4 x float>, i8 }} %f, {{ <4 x float>, i8 }} zeroinitializer"
            ),
        ),
    ]
    .into_iter()
    .chain(carried)
    .map(|(n, m)| (n, format!("{m}\n  {take}")))
    .collect()
}

/// A `{T, i8}` read result carried through phis, freezes, selects, and member insertions and
/// extractions before its texel is taken does not observe the status word; each form compiles on
/// every profile, and its texels are the Metal ones on the interpreter (even columns read slot 0,
/// odd columns slot 1).
#[test]
fn status_words_carried_by_copies_are_not_observed() {
    let mut ran = 0;
    for (name, merge) in texel_only_merges() {
        let Some(bc) = assemble(name, &merged_reads(&merge, "")) else {
            return;
        };
        for (stem, p) in committed_profiles().expect("profiles") {
            let what = format!("{stem} {name}");
            let a = ok(compile(&bc, "k", &p), &p, &what);
            assert_eq!(
                uses(&a, "texture.read") + uses(&a, "texture.read_clamped"),
                2,
                "{what}"
            );
            let (plan, m) = typed_module(&bc, "k", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let mut ta = Image::new(texture(0), [2, 2, 1], 1, 1, 1);
            let mut tb = Image::new(texture(1), [2, 2, 1], 1, 1, 1);
            for (i, (a, b)) in ta.texels.iter_mut().zip(tb.texels.iter_mut()).enumerate() {
                let x = i as f32;
                *a = [fl(x), fl(1.0), fl(2.0), fl(3.0)];
                *b = [fl(-x), fl(10.0), fl(20.0), fl(30.0)];
            }
            let res = resources(&plan, &[vec![0; 64]]);
            let (bufs, _) = run_both(&m, &res, &[ta, tb], [2, 2, 1]);
            let out = words(&bufs[0]);
            for i in 0..4usize {
                let x = i as f32;
                let want = if i % 2 == 0 {
                    [fl(x), fl(1.0), fl(2.0), fl(3.0)]
                } else {
                    [fl(-x), fl(10.0), fl(20.0), fl(30.0)]
                };
                assert_eq!(out[4 * i..4 * i + 4], want, "{what} texel {i}");
            }
            ran += 1;
        }
    }
    assert!(ran > 0);
}

/// A status word whose effects depend on a bit other than bit 0 is the typed refusal on every
/// device: MSL reads no other bit, so its value is not established. The forms: the whole word
/// converted; bit 1 tested; the word shifted down; bits 0 and 1 kept; bit 7 moved down by an
/// arithmetic shift; the sign extension's bit 8 (the word's bit 7); the whole word compared with 0;
/// a division; bit 1 of a product and of a negation (their carries reach it from bit 0 and bit 1);
/// a select on bit 0 that chooses the word shifted down; the whole word taken after each of
/// [`carried_reads`]; and the read stored whole.
#[test]
fn status_words_with_another_bit_observed_are_refused() {
    let take = "%st = extractvalue { <4 x float>, i8 } %r, 1";
    let word = |tail: &str| {
        format!(
            "{PHI}\n  {take}\n  {tail}\n  %v = insertelement <4 x float> zeroinitializer, float %mf, i32 0"
        )
    };
    let observing = [
        (
            "frozen extract",
            format!(
                "{PHI}\n  %f = freeze {{ <4 x float>, i8 }} %r\n  %st = extractvalue {{ <4 x float>, i8 }} %f, 1\n  %mf = uitofp i8 %st to float\n  %v = insertelement <4 x float> zeroinitializer, float %mf, i32 0"
            ),
        ),
        (
            "bit 1",
            word("%m = and i8 %st, 2\n  %mf = uitofp i8 %m to float"),
        ),
        (
            "shifted",
            word("%h = lshr i8 %st, 1\n  %m = and i8 %h, 1\n  %mf = uitofp i8 %m to float"),
        ),
        (
            "and 3",
            word("%m = and i8 %st, 3\n  %mf = uitofp i8 %m to float"),
        ),
        (
            "ashr 7",
            word("%h = ashr i8 %st, 7\n  %m = and i8 %h, 1\n  %mf = uitofp i8 %m to float"),
        ),
        (
            "sext lshr 8",
            word(
                "%w = sext i8 %st to i32\n  %h = lshr i32 %w, 8\n  %m = and i32 %h, 1\n  %mf = uitofp i32 %m to float",
            ),
        ),
        (
            "icmp whole",
            word("%e = icmp eq i8 %st, 0\n  %mf = select i1 %e, float 2.0, float 3.0"),
        ),
        (
            "udiv",
            word("%d = udiv i8 %st, 3\n  %m = and i8 %d, 1\n  %mf = uitofp i8 %m to float"),
        ),
        (
            "mul bit 1",
            word("%pr = mul i8 %st, 3\n  %m = and i8 %pr, 2\n  %mf = uitofp i8 %m to float"),
        ),
        (
            "neg bit 1",
            word("%ng = sub i8 0, %st\n  %m = and i8 %ng, 2\n  %mf = uitofp i8 %m to float"),
        ),
        (
            "select of shifted",
            word(
                "%lo = trunc i8 %st to i1\n  %h = lshr i8 %st, 1\n  %sv = select i1 %lo, i8 %h, i8 0\n  %m = and i8 %sv, 1\n  %mf = uitofp i8 %m to float",
            ),
        ),
    ];
    // The whole word, taken after a carrier, and a carrier stored whole.
    let carried = carried_reads().into_iter().map(|(n, head)| {
        (
            n,
            format!(
                "{head}\n  {take}\n  %mf = uitofp i8 %st to float\n  %v = insertelement <4 x float> zeroinitializer, float %mf, i32 0"
            ),
        )
    });
    let stored = (
        "stored",
        format!(
            "{PHI}\n  store {{ <4 x float>, i8 }} %r, ptr addrspace(1) %out, align 16\n  %v = extractvalue {{ <4 x float>, i8 }} %r, 0"
        ),
    );
    let observing: Vec<_> = observing
        .into_iter()
        .chain(carried)
        .chain([stored])
        .collect();
    for stem in ["moltenvk-m3", "lavapipe"] {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        for (name, merge) in &observing {
            let Some(bc) = assemble(name, &merged_reads(merge, "")) else {
                return;
            };
            refused_texture(compile(&bc, "k", &p), &format!("{stem} {name}"));
        }
    }
}

/// The spellings of `resident()` on the merged read `%r`, each storing the texel with lane 3
/// replaced by a value that tells residency: the value for a resident texel, and for one that is
/// not. MSL's front end: `and 1` then `xor 1`, or `icmp eq 0`; a `trunc` to `i1`; the optimizer's
/// packing of the bit into a wider word (`zext`, `shl`, `and`, `xor`). And forms whose bit 0
/// depends on the word's bit 0 alone: bit 0 shifted to bit 7 and back; bit 0 of a negation and of
/// a product; the word as either value of a select on another condition (the other value keeps
/// the word's bit 0). The first spelling also after each of [`carried_reads`].
fn resident_tails() -> Vec<(String, String, [f32; 2])> {
    let take = "%t = extractvalue { <4 x float>, i8 } %r, 0\n  %st = extractvalue { <4 x float>, i8 } %r, 1";
    let put = "%v = insertelement <4 x float> %t, float %mf, i32 3";
    let tails = [
        (
            "and xor",
            "%m = and i8 %st, 1\n  %n = xor i8 %m, 1\n  %mf = uitofp i8 %n to float",
            [1.0, 0.0],
        ),
        (
            "icmp",
            "%m = and i8 %st, 1\n  %e = icmp eq i8 %m, 0\n  %mf = select i1 %e, float 2.0, float 3.0",
            [2.0, 3.0],
        ),
        (
            "trunc",
            "%lo = trunc i8 %st to i1\n  %mf = select i1 %lo, float 5.0, float 4.0",
            [4.0, 5.0],
        ),
        (
            "packed",
            "%z = zext i8 %st to i32\n  %s = shl i32 %z, 3\n  %m = and i32 %s, 8\n  %q = xor i32 %m, 8\n  %mf = uitofp i32 %q to float",
            [8.0, 0.0],
        ),
        (
            "shl lshr",
            "%hi = shl i8 %st, 7\n  %m = lshr i8 %hi, 7\n  %n = xor i8 %m, 1\n  %mf = uitofp i8 %n to float",
            [1.0, 0.0],
        ),
        (
            "neg",
            "%ng = sub i8 0, %st\n  %m = and i8 %ng, 1\n  %n = xor i8 %m, 1\n  %mf = uitofp i8 %n to float",
            [1.0, 0.0],
        ),
        (
            "mul",
            "%pr = mul i8 %st, 3\n  %m = and i8 %pr, 1\n  %n = xor i8 %m, 1\n  %mf = uitofp i8 %n to float",
            [1.0, 0.0],
        ),
        (
            "select value",
            "%w = or i8 %st, 4\n  %sv = select i1 %c, i8 %st, i8 %w\n  %m = and i8 %sv, 1\n  %n = xor i8 %m, 1\n  %mf = uitofp i8 %n to float",
            [1.0, 0.0],
        ),
    ];
    let (first, first_tail, first_want) = tails[0];
    let mut out: Vec<_> = tails
        .into_iter()
        .map(|(n, tail, want)| {
            (
                n.to_owned(),
                format!("{PHI}\n  {take}\n  {tail}\n  {put}"),
                want,
            )
        })
        .collect();
    for (carrier, head) in carried_reads() {
        out.push((
            format!("{carrier} {first}"),
            format!("{head}\n  {take}\n  {first_tail}\n  {put}"),
            first_want,
        ));
    }
    out
}

/// `OpImageSparseFetch` and `OpImageSparseRead`.
const OP_IMAGE_SPARSE_FETCH: u16 = 313;
const OP_IMAGE_SPARSE_READ: u16 = 320;

fn texture_slot(index: u32) -> crate::pipeline::sym::SlotRef {
    crate::pipeline::sym::SlotRef {
        stage: Stage::Kernel,
        class: crate::pipeline::sym::SlotClass::Texture,
        index,
    }
}

/// Bit 0 of a status word, observed after a copy as `resident()` spells it, is lowered on every
/// profile, for a sampled texture (a fetch) and a read_write one (a storage read). Each slot states
/// `RuntimeFact::SparseTexture` with the tile-map arm on every profile, and the artifact notes no
/// `SparseResidency`: a read's word and value are Metal's. On the interpreter a texture that is not
/// sparse (the default specialization) is resident at every texel; a sparse texture
/// (`SpecKey::SparseTexture` 1) whose texel is unmapped reads zero and reports it not resident
/// (slot 0's texel 0 and slot 1's texel 3), from the slot's tile map, over an image whose unmapped
/// texels read garbage: a block of a sparse image that holds no mapped tile reads the device's
/// value, which the read replaces with Metal's.
#[test]
fn residency_of_status_words_is_lowered() {
    use crate::pipeline::artifact::PrecisionNote;
    use crate::pipeline::facts::RuntimeFact;
    let (mut plain, mut sparse) = ([0; 2], [0; 2]);
    for (k, access) in ["read", "read_write"].into_iter().enumerate() {
        for (name, merge, [yes, no]) in resident_tails() {
            let text = merged_reads(&merge, "").replace(
                "!\"air.read\", !\"air.arg_type_name\", !\"texture2d<float, read>\"",
                &format!(
                    "!\"air.{access}\", !\"air.arg_type_name\", !\"texture2d<float, {access}>\""
                ),
            );
            let Some(bc) = assemble(&name, &text) else {
                return;
            };
            for (stem, p) in committed_profiles().expect("profiles") {
                let what = format!("{stem} {access} {name}");
                let (plan, m) = match typed_module(&bc, "k", &p) {
                    Ok(x) => x,
                    // A storage texture without a format the profile reads.
                    Err(Error::NeedsInput(_)) if access == "read_write" => continue,
                    Err(e) => panic!("{what}: {e}"),
                };
                let a = ok(compile(&bc, "k", &p), &p, &what);
                assert_eq!(
                    uses(&a, "texture.read.resident") + uses(&a, "texture.read_clamped.resident"),
                    2,
                    "{what}"
                );
                for index in [0, 1] {
                    let fact = RuntimeFact::SparseTexture {
                        slot: texture_slot(index),
                        sparse_arm: false,
                        tile_map: true,
                    };
                    assert!(plan.facts.contains(&fact), "{what}: {fact:?}");
                }
                assert!(
                    !plan.precision.contains(&PrecisionNote::SparseResidency),
                    "{what}"
                );
                for op in [OP_IMAGE_SPARSE_FETCH, OP_IMAGE_SPARSE_READ] {
                    assert!(!opcodes(&a.module.words).contains(&op), "{what}");
                }
                let images = || {
                    let mut ta = Image::new(texture(0), [2, 2, 1], 1, 1, 1);
                    let mut tb = Image::new(texture(1), [2, 2, 1], 1, 1, 1);
                    for (i, (a, b)) in ta.texels.iter_mut().zip(tb.texels.iter_mut()).enumerate() {
                        let x = i as f32;
                        *a = [fl(x), fl(1.0), fl(2.0), fl(3.0)];
                        *b = [fl(-x), fl(10.0), fl(20.0), fl(30.0)];
                    }
                    [ta, tb]
                };
                let want = |i: usize, resident: bool| {
                    let x = i as f32;
                    match (resident, i.is_multiple_of(2)) {
                        (false, _) => [0, 0, 0, fl(no)],
                        (true, true) => [fl(x), fl(1.0), fl(2.0), fl(yes)],
                        (true, false) => [fl(-x), fl(10.0), fl(20.0), fl(yes)],
                    }
                };
                // Not sparse: the default specialization, every texel resident.
                let res = resources(&plan, &[vec![0; 64]]);
                let (bufs, _) = run_both(&m, &res, &images(), [2, 2, 1]);
                let out = words(&bufs[0]);
                for i in 0..4usize {
                    assert_eq!(out[4 * i..4 * i + 4], want(i, true), "{what} texel {i}");
                }
                plain[k] += 1;
                let mut res = res;
                for index in [0, 1] {
                    res.spec
                        .insert(SpecKey::SparseTexture(texture_slot(index)), 1);
                }
                let images = {
                    // The tile-map arm: images whose unmapped texels read the device's value of a
                    // block that holds no mapped tile, garbage the module must not return, and a
                    // map of one-texel tiles per slot.
                    let mut imgs = images();
                    imgs[0].texels[0] = [0xDEAD_BEEF; 4];
                    imgs[1].texels[3] = [0xDEAD_BEEF; 4];
                    for (index, unmapped) in [(0u32, [0, 0, 0]), (1, [1, 1, 0])] {
                        let mut map = SparseTileMap::new(
                            crate::metal::TextureType::Type2D,
                            [2, 2, 1],
                            1,
                            1,
                            [1; 3],
                            1,
                        )
                        .expect("tile map");
                        map.update(TileMapping::Map, 0, 0, [0; 3], [2, 2, 1])
                            .expect("map");
                        map.update(TileMapping::Unmap, 0, 0, unmapped, [1; 3])
                            .expect("unmap");
                        res.buffers.push(Buffer {
                            slot: Some(ResourceSlot {
                                stage: Stage::Kernel,
                                kind: ResourceKind::TileMap,
                                index,
                            }),
                            address: None,
                            bytes: map.to_bytes(),
                        });
                    }
                    imgs
                };
                let (bufs, _) = run_both(&m, &res, &images, [2, 2, 1]);
                let out = words(&bufs[0]);
                for i in 0..4usize {
                    let resident = i == 1 || i == 2;
                    assert_eq!(
                        out[4 * i..4 * i + 4],
                        want(i, resident),
                        "{what} sparse texel {i}"
                    );
                }
                sparse[k] += 1;
            }
        }
    }
    assert!(
        plain.iter().chain(&sparse).all(|n| *n > 0),
        "plain {plain:?}, sparse {sparse:?}"
    );
}

/// A kernel over one `texture2d<float, {access}>` slot 0 and `float4 *out` (buffer 0): each
/// thread reads texel `gid` and stores it to `out[gid.x + 4 * gid.y]`, its lane 3 replaced by
/// `resident()` (1 or 0).
const OUTSIDE_READ: &str = r#"
define void @k(ptr addrspace(1) %a, ptr addrspace(1) %out, <2 x i32> %gid) {
entry:
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %s = call ptr addrspace(2) @air.get_read_sampler()
  %r = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %a, ptr addrspace(2) %s, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  %t = extractvalue { <4 x float>, i8 } %r, 0
  %st = extractvalue { <4 x float>, i8 } %r, 1
  %m = and i8 %st, 1
  %n = xor i8 %m, 1
  %mf = uitofp i8 %n to float
  %v = insertelement <4 x float> %t, float %mf, i32 3
  %y4 = shl i32 %y, 2
  %i = add i32 %x, %y4
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %i
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.{access}", !"air.arg_type_name", !"texture2d<float, {access}>", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;

/// Profile `stem` with image robustness removed (`robustImageAccess` and `robustImageAccess2`),
/// so every read is checked in the shader.
fn without_robust_images(stem: &str) -> DeviceProfile {
    use crate::device::requirements::VkFeature;
    use crate::device::{Device, DeviceDescription};
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    desc.features.remove(&VkFeature::RobustImageAccess);
    desc.features.remove(&VkFeature::RobustImageAccess2);
    Device::new(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// A read outside a sparse texture is resident, as Metal answers for every texture (its readings of
/// 760fce8c and fe3320d0), whichever texel the device or the shader's clamp would report. On every
/// committed profile, as stated and without image robustness, a sampled read and a storage read of
/// a sparse 2x2 texture whose column x = 1 is unmapped run over a 4x3 grid on the interpreter,
/// through the slot's tile map (on every profile): in bounds, a mapped texel reads its value
/// resident and an unmapped one zero not resident, whatever the image holds there; outside it, the
/// read is zero and resident. Under the shader's check the clamped texel of a read past the width
/// is unmapped, and the map's clamped decoding reads that column's tile. Without the word's
/// in-bounds test, every leg answers not resident there.
#[test]
fn a_read_outside_a_sparse_texture_is_resident() {
    // Runs under the device's robustness, and under the shader's check.
    let mut legs = [0; 2];
    for access in ["read", "sample"] {
        let text = OUTSIDE_READ.replace("{access}", access);
        let Some(bc) = assemble(&format!("outside_{access}"), &text) else {
            return;
        };
        for (stem, stated) in committed_profiles().expect("profiles") {
            for (leg, p) in [
                ("stated", stated.clone()),
                ("without image robustness", without_robust_images(&stem)),
            ] {
                let what = format!("{stem} {leg} {access}");
                let (plan, m) =
                    typed_module(&bc, "k", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
                let a = ok(compile(&bc, "k", &p), &p, &what);
                let checked = uses(&a, "texture.read_clamped.resident") == 1;
                assert_eq!(
                    u32::from(checked) + uses(&a, "texture.read.resident"),
                    1,
                    "{what}"
                );
                let mut img = Image::new(texture(0), [2, 2, 1], 1, 1, 1);
                for (i, t) in img.texels.iter_mut().enumerate() {
                    *t = [fl(i as f32 + 1.0), fl(10.0), fl(20.0), fl(30.0)];
                }
                let mut res = resources(&plan, &[vec![0; 16 * 12]]);
                // An image whose unmapped texels read garbage (a sparse image's block that holds
                // no mapped tile reads the device's value), and the map of its one-texel tiles
                // with column x = 1 unmapped.
                img.texels[1] = [0xDEAD_BEEF; 4];
                img.texels[3] = [0xDEAD_BEEF; 4];
                let mut map = SparseTileMap::new(
                    crate::metal::TextureType::Type2D,
                    [2, 2, 1],
                    1,
                    1,
                    [1; 3],
                    1,
                )
                .expect("tile map");
                map.update(TileMapping::Map, 0, 0, [0; 3], [1, 2, 1])
                    .expect("map");
                res.buffers.push(Buffer {
                    slot: Some(ResourceSlot {
                        stage: Stage::Kernel,
                        kind: ResourceKind::TileMap,
                        index: 0,
                    }),
                    address: None,
                    bytes: map.to_bytes(),
                });
                if !checked {
                    img.robust = Some([0; 4]);
                }
                res.spec.insert(SpecKey::SparseTexture(texture_slot(0)), 1);
                let (bufs, _) = run_both(&m, &res, &[img], [4, 3, 1]);
                let out = words(&bufs[0]);
                for y in 0..3u32 {
                    for x in 0..4u32 {
                        let i = (x + 4 * y) as usize;
                        let want = match (x < 2 && y < 2, x == 1) {
                            (false, _) => [0, 0, 0, fl(1.0)],
                            (true, true) => [0, 0, 0, fl(0.0)],
                            (true, false) => {
                                [fl((x + 2 * y) as f32 + 1.0), fl(10.0), fl(20.0), fl(1.0)]
                            }
                        };
                        assert_eq!(out[4 * i..4 * i + 4], want, "{what} texel ({x}, {y})");
                    }
                }
                legs[usize::from(checked)] += 1;
            }
        }
    }
    assert!(legs.iter().all(|n| *n > 0), "legs {legs:?}");
}

/// A fragment function returning an implicit-LOD sample of `texture2d<float, sample>` slot 0
/// through `[[sampler(0)]]` at the position, its lane 3 replaced by `resident()` (1 or 0).
const RESIDENT_FRAGMENT: &str = r#"
define <4 x float> @f(<4 x float> %pos, ptr addrspace(1) %t, ptr addrspace(2) %s) {
  %uv = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %r = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %tx = extractvalue { <4 x float>, i8 } %r, 0
  %st = extractvalue { <4 x float>, i8 } %r, 1
  %m = and i8 %st, 1
  %n = xor i8 %m, 1
  %mf = uitofp i8 %n to float
  %v = insertelement <4 x float> %tx, float %mf, i64 3
  ret <4 x float> %v
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.fragment = !{!0}
!0 = !{ptr @f, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5, !6}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!6 = !{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
"#;

/// `OpImageSampleImplicitLod` and `OpImageSparseSampleImplicitLod`.
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
const OP_IMAGE_SPARSE_SAMPLE_IMPLICIT_LOD: u16 = 305;

/// A fragment's implicit-LOD sample through a runtime sampler whose residency the function observes
/// compiles and validates on every profile, with the slot's `SparseTexture` fact. Where the profile
/// enables `shaderResourceResidency`, the slot's sparse arm (`resident_access`) selects between
/// `OpImageSparseSampleImplicitLod` and the plain `OpImageSampleImplicitLod` on the specialization
/// constant (uniform, so both keep implicit derivatives), and the stage notes `SparseResidency`
/// (the device's footprint); elsewhere the plain sample alone, its word 0, and no note: the binder
/// refuses a sparse texture there.
#[test]
fn a_fragments_implicit_lod_sample_observes_residency() {
    use crate::device::requirements::VkFeature;
    use crate::pipeline::artifact::PrecisionNote;
    use crate::pipeline::facts::RuntimeFact;
    use crate::pipeline::sym::{SlotClass, SlotRef};
    let Some(bc) = assemble("resident_fragment", RESIDENT_FRAGMENT) else {
        return;
    };
    let slot = SlotRef {
        stage: Stage::Fragment,
        class: SlotClass::Texture,
        index: 0,
    };
    let mut seen = [0; 2];
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = ok(compile(&bc, "f", &p), &p, stem.as_str());
        assert_eq!(uses(&a, "sampler.sample_implicit.resident"), 1, "{stem}");
        let (plan, _) = typed_module(&bc, "f", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let residency = p
            .fields()
            .enabled
            .features
            .contains(&VkFeature::ShaderResourceResidency);
        // A runtime sampler's footprint is the device filter's: no tile map answers it.
        let fact = RuntimeFact::SparseTexture {
            slot,
            sparse_arm: residency,
            tile_map: false,
        };
        assert!(plan.facts.contains(&fact), "{stem}: {fact:?}");
        assert_eq!(
            plan.precision.contains(&PrecisionNote::SparseResidency),
            residency,
            "{stem}"
        );
        let ops = opcodes(&a.module.words);
        assert!(ops.contains(&OP_IMAGE_SAMPLE_IMPLICIT_LOD), "{stem}");
        assert_eq!(
            ops.contains(&OP_IMAGE_SPARSE_SAMPLE_IMPLICIT_LOD),
            residency,
            "{stem}"
        );
        seen[usize::from(residency)] += 1;
    }
    assert!(seen[0] > 0 && seen[1] > 0, "{seen:?}");
}

// ---- Texture atomics ----

const OP_IMAGE_TEXEL_POINTER: u16 = 60;
const OP_ATOMIC_SMAX: u16 = 238;
const OP_ATOMIC_UMAX: u16 = 239;
const OP_TYPE_IMAGE: u16 = 25;
const FORMAT_R32I: u32 = 24;
const FORMAT_R32UI: u32 = 33;

/// The opcodes of a SPIR-V module, in order.
fn opcodes(words: &[u32]) -> Vec<u16> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = (words[i] >> 16) as usize;
        out.push((words[i] & 0xFFFF) as u16);
        i += n.max(1);
    }
    out
}

/// The `Image Format` operands of a module's `OpTypeImage`s.
fn image_formats(words: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = (words[i] >> 16) as usize;
        if (words[i] & 0xFFFF) as u16 == OP_TYPE_IMAGE {
            out.extend(words.get(i + 8).copied());
        }
        i += n.max(1);
    }
    out
}

/// A kernel updating `texture2d<T, read_write>` slot 0 (`T` = `uint` or `int`, sign token `s`)
/// at `gid` with `atomic_fetch_max(gid, uint4(gid.x + 7))` (`.i16` coordinates when `short`),
/// passing `offset` and `order` at the positions Apple's compiler fills with zero, then running
/// `tail` on the old texel `%old` (which may store to `uint *out`, buffer 0).
fn atomic_kernel(signed: bool, short: bool, offset: &str, order: &str, tail: &str) -> String {
    let (t, s) = if signed { ("int", "s") } else { ("uint", "u") };
    let (c, coord, name) = if short {
        (
            "<2 x i16>",
            "%c = trunc <2 x i32> %gid to <2 x i16>",
            format!("air.atomic_fetch_max_explicit_texture_2d.i16.{s}.v4i32"),
        )
    } else {
        (
            "<2 x i32>",
            "%c = add <2 x i32> %gid, zeroinitializer",
            format!("air.atomic_fetch_max_explicit_texture_2d.{s}.v4i32"),
        )
    };
    format!(
        r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, <2 x i32> %gid) {{
  %x = extractelement <2 x i32> %gid, i64 0
  %x7 = add i32 %x, 7
  %v0 = insertelement <4 x i32> poison, i32 %x7, i64 0
  %v = shufflevector <4 x i32> %v0, <4 x i32> poison, <4 x i32> zeroinitializer
  {coord}
  %old = call <4 x i32> @{name}(ptr addrspace(1) %t, {c} %c, {c} {offset}, <4 x i32> %v, i32 {order}, i32 3)
  {tail}
  ret void
}}

declare <4 x i32> @{name}(ptr addrspace(1), {c}, {c}, <4 x i32>, i32, i32)

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<{t}, read_write>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}}
"#
    )
}

const LANE0: &str =
    "%o = extractelement <4 x i32> %old, i64 0\n  store i32 %o, ptr addrspace(1) %out, align 4";

/// `atomic_fetch_max` is `OpAtomicUMax` (uint) or `OpAtomicSMax` (int) on the texel pointer of an
/// `R32ui` / `R32i` storage image, with and without `.i16` coordinates, its result unread or read
/// at lane 0; a texel outside the image is guarded exactly without `robustImageAccess`, and
/// the slot's interface reads and writes it.
#[test]
fn texture_atomic_max_is_an_image_atomic() {
    let (mut guarded_n, mut plain_n) = (0, 0);
    for (signed, short, tail) in [
        (false, false, LANE0),
        (false, true, LANE0),
        (true, false, LANE0),
        (true, true, ""),
    ] {
        let text = atomic_kernel(signed, short, "zeroinitializer", "0", tail);
        let Some(bc) = assemble("atomic", &text) else {
            return;
        };
        for (stem, p) in committed_profiles().expect("profiles") {
            let what = format!("{stem} signed={signed} short={short}");
            let format = if signed {
                crate::vk::Format::R32_SINT
            } else {
                crate::vk::Format::R32_UINT
            };
            let usage = crate::device::requirements::FormatUsage::StorageImageAtomic;
            if !p.supports_format(format, usage) {
                // A profile whose format table lacks the storage atomic: the typed need.
                let want = Unsupported::needs(
                    Feature::Format,
                    crate::error::Capability::Format { format, usage },
                );
                match compile(&bc, "k", &p) {
                    Err(Error::Unsupported(u)) => assert_eq!(u.needs, want.needs, "{what}"),
                    other => panic!("{what}: {:?}", other.map(|_| ())),
                }
                continue;
            }
            let a = ok(compile(&bc, "k", &p), &p, &what);
            let guarded = !p.fields().images.robust_image_access;
            if guarded {
                guarded_n += 1;
            } else {
                plain_n += 1;
            }
            assert_eq!(
                uses(&a, "texture.atomic_max_guarded"),
                u32::from(guarded),
                "{what}"
            );
            assert_eq!(
                uses(&a, "texture.atomic_max"),
                u32::from(!guarded),
                "{what}"
            );
            let ops = opcodes(&a.module.words);
            assert!(ops.contains(&OP_IMAGE_TEXEL_POINTER), "{what}");
            let (want, other) = if signed {
                (OP_ATOMIC_SMAX, OP_ATOMIC_UMAX)
            } else {
                (OP_ATOMIC_UMAX, OP_ATOMIC_SMAX)
            };
            assert!(ops.contains(&want) && !ops.contains(&other), "{what}");
            let format = if signed { FORMAT_R32I } else { FORMAT_R32UI };
            assert_eq!(image_formats(&a.module.words), vec![format], "{what}");
            let slot = crate::pipeline::sym::SlotRef {
                stage: Stage::Kernel,
                class: crate::pipeline::sym::SlotClass::Texture,
                index: 0,
            };
            match a.interface.slots.get(&slot) {
                Some(crate::pipeline::interface::SlotInterface::Texture(t)) => {
                    assert!(
                        t.access.read && t.access.write && !t.access.sample,
                        "{what}"
                    )
                }
                other => panic!("{what}: slot 0 interface {other:?}"),
            }
        }
    }
    assert!(
        guarded_n > 0 && plain_n > 0,
        "guarded {guarded_n}, plain {plain_n}"
    );
}

/// A result read beyond lane 0 lowers, since Metal's lanes are measured (`(old, 0, 0, 1)`); a
/// nonzero value at the offset position and a memory order other than relaxed stay the typed
/// refusal; a storage choice other than the `R32` format of the texel's type is not representable.
#[test]
fn texture_atomic_lanes_lower_and_unestablished_operands_are_refused() {
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let wide =
        "%o = extractelement <4 x i32> %old, i64 1\n  store i32 %o, ptr addrspace(1) %out, align 4";
    let whole = "%o = bitcast <4 x i32> %old to <2 x i64>\n  %w = extractelement <2 x i64> %o, i64 0\n  %n = trunc i64 %w to i32\n  store i32 %n, ptr addrspace(1) %out, align 4";
    for (name, text) in [
        (
            "lane 1",
            atomic_kernel(false, false, "zeroinitializer", "0", wide),
        ),
        (
            "whole",
            atomic_kernel(false, false, "zeroinitializer", "0", whole),
        ),
    ] {
        let Some(bc) = assemble(name, &text) else {
            return;
        };
        ok(compile(&bc, "k", &p), &p, name);
    }
    for (name, text) in [
        (
            "offset",
            atomic_kernel(false, false, "<i32 1, i32 0>", "0", LANE0),
        ),
        (
            "order",
            atomic_kernel(false, false, "zeroinitializer", "5", LANE0),
        ),
    ] {
        let Some(bc) = assemble(name, &text) else {
            return;
        };
        refused_texture(compile(&bc, "k", &p), name);
    }
    let Some(bc) = assemble(
        "choice",
        &atomic_kernel(false, false, "zeroinitializer", "0", LANE0),
    ) else {
        return;
    };
    use crate::pipeline::lower_inputs::{StorageImageChoice, StorageImageClass};
    let slot = crate::pipeline::sym::SlotRef {
        stage: Stage::Kernel,
        class: crate::pipeline::sym::SlotClass::Texture,
        index: 0,
    };
    let with = |f: crate::vk::Format| {
        let mut inputs = crate::lower::select::LowerInputs::new();
        inputs.storage_images.insert(
            slot,
            StorageImageChoice::new(f, StorageImageClass::Identity),
        );
        crate::pipeline::compile::compile_bitcode(
            &bc,
            crate::pipeline::compile::EntryRef::Name("k"),
            &crate::library::FunctionConstantValues::defaults(),
            &p,
            &inputs,
            &mut crate::pipeline::testing::meter(),
            crate::budget::SPIRV_WORDS,
        )
    };
    ok(with(crate::vk::Format::R32_UINT), &p, "R32_UINT choice");
    let representable = Unsupported::not_representable(Feature::Format);
    match with(crate::vk::Format::R8G8B8A8_UINT) {
        Err(Error::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (representable.feature, representable.needs)
        ),
        other => panic!("rgba8 choice: {:?}", other.map(|_| ())),
    }
}

// ---- Cube-array writes and size queries at a level ----

/// The module of `entry` on `p`, or `None` when `p` needs a runtime input for it (a storage
/// texture format without a formatless store).
fn module(bc: &[u8], entry: &str, stem: &str, p: &DeviceProfile) -> Option<(Plan, TypedModule)> {
    match typed_module(bc, entry, p) {
        Ok(m) => Some(m),
        Err(Error::NeedsInput(_)) => None,
        Err(e) => panic!("{stem}: {entry}: {e}"),
    }
}

/// `texturecube_array<float, access::write>` slot 0: every thread writes `(1, 2, 3, 4)` at `gid`
/// of face 3 of cube 1, then stores the width (buffer 0).
const CUBE_ARRAY_WRITE: &str = r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, <2 x i32> %gid) {
  call void @air.write_texture_cube_array.v4f32(ptr addrspace(1) %t, <2 x i32> %gid, i32 3, i32 1, <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, i32 0, i32 2)
  %w = call i32 @air.get_width_texture_cube_array(ptr addrspace(1) %t, i32 0)
  store i32 %w, ptr addrspace(1) %out, align 4
  ret void
}

declare void @air.write_texture_cube_array.v4f32(ptr addrspace(1), <2 x i32>, i32, i32, <4 x float>, i32, i32)
declare i32 @air.get_width_texture_cube_array(ptr addrspace(1), i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texturecube_array<float, write>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;

/// A cube-array write names the face before the array index (MSL `write(color, coord, face,
/// array)`), and writes layer `face + 6 * cube` of the binder's 2D-array view; its width query
/// reads the view's width.
#[test]
fn cube_array_writes_address_face_then_cube() {
    let Some(bc) = assemble("cube_array_write", CUBE_ARRAY_WRITE) else {
        return;
    };
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let Some((plan, m)) = module(&bc, "k", &stem, &p) else {
            continue;
        };
        ok(compile(&bc, "k", &p), &p, &stem);
        let img = Image::new(texture(0), [2, 2, 1], 12, 1, 1);
        let res = resources(&plan, &[vec![0; 4]]);
        let (bufs, imgs) = run_both(&m, &res, &[img], [2, 2, 1]);
        assert_eq!(words(&bufs[0]), [2], "{stem}");
        let img = &imgs[0];
        for layer in 0..12 {
            for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let i = img.index(0, layer, 0, [x, y, 0]).expect("texel");
                let want = if layer == 9 {
                    [fl(1.0), fl(2.0), fl(3.0), fl(4.0)]
                } else {
                    [0; 4]
                };
                assert_eq!(img.texels[i], want, "{stem} layer {layer} ({x}, {y})");
            }
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// `texture2d<float>` slot 0: stores its width and height at the level `buffer(0)` holds, and
/// its width at level 0, to buffer 1.
const SIZE_AT_LEVEL: &str = r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %lod, ptr addrspace(1) %out) {
  %l = load i32, ptr addrspace(1) %lod, align 4
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 %l)
  %h = call i32 @air.get_height_texture_2d(ptr addrspace(1) %t, i32 %l)
  %w0 = call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 0)
  store i32 %w, ptr addrspace(1) %out, align 4
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %h, ptr addrspace(1) %o1, align 4
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 2
  store i32 %w0, ptr addrspace(1) %o2, align 4
  ret void
}

declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
declare i32 @air.get_height_texture_2d(ptr addrspace(1), i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"lod"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;

/// `texture3d<float, read_write>` slot 0: stores its width, height and depth at the level
/// `buffer(0)` holds to buffer 1.
const SIZE_AT_LEVEL_3D: &str = r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %lod, ptr addrspace(1) %out) {
  %l = load i32, ptr addrspace(1) %lod, align 4
  %w = call i32 @air.get_width_texture_3d(ptr addrspace(1) %t, i32 %l)
  %h = call i32 @air.get_height_texture_3d(ptr addrspace(1) %t, i32 %l)
  %d = call i32 @air.get_depth_texture_3d(ptr addrspace(1) %t, i32 %l)
  store i32 %w, ptr addrspace(1) %out, align 4
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %h, ptr addrspace(1) %o1, align 4
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 2
  store i32 %d, ptr addrspace(1) %o2, align 4
  ret void
}

declare i32 @air.get_width_texture_3d(ptr addrspace(1), i32)
declare i32 @air.get_height_texture_3d(ptr addrspace(1), i32)
declare i32 @air.get_depth_texture_3d(ptr addrspace(1), i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture3d<float, read_write>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"lod"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;

/// Metal's extent at level `l` of a texture whose first level's extent is `e`:
/// `max(1, e >> (l & 255))`, a shift of 32 or more answering 1.
fn metal_extent(e: u32, l: u32) -> u32 {
    e.checked_shr(l & 255).unwrap_or(0).max(1)
}

/// A size query at a level answers Metal's extent there on every profile, past the level count
/// too (`emit::level_extent`; measured on an Apple M3 Max under macOS 27.0.1): a 3-level 8x4
/// sampled texture's width and height, and a 16x8x4 `read_write` 3D texture's width, height and
/// depth, which needs no level views, at levels 0 to 6, 31 to 33, 255 to 258, 2^31 + 1 and
/// 2^32 - 1. The interpreter traps a query of a level the image lacks, so none is made. The
/// constant level 0 is a plain query.
#[test]
fn size_queries_answer_metal_s_extent_at_every_level() {
    let Some(sampled) = assemble("size_at_level", SIZE_AT_LEVEL) else {
        return;
    };
    let Some(storage) = assemble("size_at_level_3d", SIZE_AT_LEVEL_3D) else {
        return;
    };
    let levels = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        31,
        32,
        33,
        255,
        256,
        257,
        258,
        0x8000_0001,
        u32::MAX,
    ];
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        for (what, bc, extent, image_levels, query_uses) in [
            ("sampled", &sampled, [8, 4, 1], 3, (2, 1)),
            ("storage", &storage, [16, 8, 4], 1, (3, 0)),
        ] {
            let what = format!("{what} on {stem}");
            let a = ok(compile(bc, "k", &p), &p, &what);
            assert_eq!(
                (uses(&a, "texture.query_level"), uses(&a, "texture.query")),
                query_uses,
                "{what}"
            );
            let (plan, m) = typed_module(bc, "k", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let img = Image::new(texture(0), extent, 1, image_levels, 1);
            for l in levels {
                let res = resources(&plan, &[l.to_le_bytes().to_vec(), vec![0; 12]]);
                let cfg = Config {
                    workgroups: [1, 1, 1],
                    ..Config::default()
                };
                let (o, _) = run_with_images(&m, &cfg, res, vec![img.clone()])
                    .unwrap_or_else(|e| panic!("{what}, level {l}: {e:?}"));
                let [w, h, d] = extent;
                let want = match image_levels {
                    3 => [metal_extent(w, l), metal_extent(h, l), w],
                    _ => [metal_extent(w, l), metal_extent(h, l), metal_extent(d, l)],
                };
                assert_eq!(words(&o.buffers[1]), want, "{what}, level {l}");
            }
            ran += 1;
        }
    }
    assert!(ran > 0);
}

// ---- A function's writes to a sparse texture ----

/// A kernel over one `texture2d<float, read_write>` slot 0 and `float4 *out` (buffer 0): each
/// thread writes `(100 + i, 1, 2, 3)` at texel `gid` (`i = gid.x + 2 * gid.y`), fences the
/// texture, reads the texel back by `sparse_read`, and stores it to `out[i]`, its lane 3 replaced
/// by `resident()` (1 or 0).
const WRITE_THEN_READ: &str = r#"
define void @k(ptr addrspace(1) %a, ptr addrspace(1) %out, <2 x i32> %gid) {
entry:
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %y2 = shl i32 %y, 1
  %i = add i32 %x, %y2
  %f = uitofp i32 %i to float
  %w = fadd float %f, 1.000000e+02
  %v = insertelement <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, float %w, i32 0
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %a, <2 x i32> %gid, <4 x float> %v, i32 0, i32 3)
  call void @air.fence_texture_2d(ptr addrspace(1) %a)
  %s = call ptr addrspace(2) @air.get_read_sampler()
  %r = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %a, ptr addrspace(2) %s, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 3)
  %t = extractvalue { <4 x float>, i8 } %r, 0
  %st = extractvalue { <4 x float>, i8 } %r, 1
  %m = and i8 %st, 1
  %n = xor i8 %m, 1
  %mf = uitofp i8 %n to float
  %o = insertelement <4 x float> %t, float %mf, i32 3
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %i
  store <4 x float> %o, ptr addrspace(1) %p, align 16
  ret void
}

declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
declare void @air.fence_texture_2d(ptr addrspace(1))
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;

/// Metal drops every write a function makes to a sparse texture, its mapped tiles' included, and
/// the writing thread's read after a fence returns the texel as it was. On every committed profile
/// that compiles the kernel, it states `SparseTextureWrites` and the slot's residency arm (the tile
/// map on every profile), and on the interpreter over a 2x2 texture: not sparse, each thread reads
/// its own write back and the texture holds it; sparse with texel 3 unmapped (its image's texel
/// garbage, as a sparse image's block that holds no mapped tile reads), each mapped texel reads its
/// old value resident, texel 3 reads zero not resident, and the texture is unchanged.
#[test]
fn a_functions_writes_to_a_sparse_texture_are_dropped() {
    use crate::pipeline::facts::RuntimeFact;
    let Some(bc) = assemble("write_then_read", WRITE_THEN_READ) else {
        return;
    };
    let mut seen = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let (plan, m) = match typed_module(&bc, "k", &p) {
            Ok(x) => x,
            // A storage texture without a format the profile reads.
            Err(Error::NeedsInput(_)) => continue,
            Err(e) => panic!("{stem}: {e}"),
        };
        ok(compile(&bc, "k", &p), &p, &stem);
        for fact in [
            RuntimeFact::SparseTextureWrites {
                slot: texture_slot(0),
                atomic: false,
            },
            RuntimeFact::SparseTexture {
                slot: texture_slot(0),
                sparse_arm: false,
                tile_map: true,
            },
        ] {
            assert!(plan.facts.contains(&fact), "{stem}: {fact:?}");
        }
        let old = |i: usize| [fl(i as f32), fl(1.0), fl(2.0), fl(3.0)];
        let image = || {
            let mut t = Image::new(texture(0), [2, 2, 1], 1, 1, 1);
            for (i, texel) in t.texels.iter_mut().enumerate() {
                *texel = old(i);
            }
            t
        };
        // Not sparse: the default specialization; every write lands.
        let res = resources(&plan, &[vec![0; 64]]);
        let (bufs, imgs) = run_both(&m, &res, &[image()], [2, 2, 1]);
        let out = words(&bufs[0]);
        for i in 0..4usize {
            let written = [fl(100.0 + i as f32), fl(1.0), fl(2.0), fl(3.0)];
            let read = [written[0], written[1], written[2], fl(1.0)];
            assert_eq!(out[4 * i..4 * i + 4], read, "{stem} not sparse, texel {i}");
            assert_eq!(imgs[0].texels[i], written, "{stem} not sparse, texel {i}");
        }
        // Sparse, texel (1, 1) unmapped.
        let mut res = res;
        res.spec.insert(SpecKey::SparseTexture(texture_slot(0)), 1);
        let mut t = image();
        {
            // The tile-map arm: the image's unmapped texel the device's value, which the module
            // must not read, and a map of one-texel tiles.
            t.texels[3] = [0xDEAD_BEEF; 4];
            let mut map = SparseTileMap::new(
                crate::metal::TextureType::Type2D,
                [2, 2, 1],
                1,
                1,
                [1; 3],
                1,
            )
            .expect("tile map");
            map.update(TileMapping::Map, 0, 0, [0; 3], [2, 2, 1])
                .expect("map");
            map.update(TileMapping::Unmap, 0, 0, [1, 1, 0], [1; 3])
                .expect("unmap");
            res.buffers.push(Buffer {
                slot: Some(ResourceSlot {
                    stage: Stage::Kernel,
                    kind: ResourceKind::TileMap,
                    index: 0,
                }),
                address: None,
                bytes: map.to_bytes(),
            });
        }
        let before = t.texels.clone();
        let (bufs, imgs) = run_both(&m, &res, &[t], [2, 2, 1]);
        let out = words(&bufs[0]);
        for i in 0..4usize {
            let read = if i == 3 {
                [0, 0, 0, 0]
            } else {
                let o = old(i);
                [o[0], o[1], o[2], fl(1.0)]
            };
            assert_eq!(out[4 * i..4 * i + 4], read, "{stem} sparse, texel {i}");
        }
        assert_eq!(
            imgs[0].texels, before,
            "{stem}: a write to a sparse texture landed"
        );
        seen += 1;
    }
    assert!(seen > 0);
}
