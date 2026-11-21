//! Metal readings that the engine's first MoltenVK run disagreed with, checked on the CPU
//! interpreter where it models the instruction and structurally where it does not.
//!
//! - Texel offsets: Metal applies the two's-complement value of each offset component's low four
//!   bits (authored case `a2_dynamic_texel_offset`, 128 of 128 words on an Apple M3 Max). A
//!   dynamic `Offset` operand is `OpBitFieldSExtract(v, 0, 4)`, and a pixel sampler's exact fetch
//!   adds the same value, which the interpreter runs.
//! - Integer textures are never filtered: a linear constexpr sampler reads one as nearest (the
//!   authored case `subsample_quarter_luma`), so a linear pixel sampler fetches an integer texel
//!   exactly, and the layout's immutable sampler is `NEAREST` (a linear one on an integer view is
//!   VUID-vkCmdDispatch-magFilter-04553).
//! - A loop-exit tail runs after its loop with every invocation that left through it: in the
//!   authored case `a3_multi_level_break_tail` AGX runs the tail of a two-level break once per
//!   outer iteration, which the structurizer's natural loops give on the
//!   interpreter's 32-lane subgroup.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::DevicePolicy;
use crate::lower::select::Plan;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::layout::BindingSource;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, profile, typed_module, AssembleError,
};
use crate::spirv::interp::{
    run as interp_run, run_with_images, Buffer, Config, Image, Order, Resources,
};
use crate::spirv::ir::{GlobalKind, TypedModule};
use crate::vk;

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

/// Bitcode of the module `body` (after the AIR header), or `None` without `llvm-as`.
fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-metal-readings-{}-{}-{name}",
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

/// The offset Metal applies for component `x`: its low four bits as a two's-complement value.
fn applied(x: i32) -> i32 {
    (x << 28) >> 28
}

/// The dynamic offsets of the authored case `a2_dynamic_texel_offset`, one `(x, y)` per lane.
const OFFSETS: [(i32, i32); 16] = [
    (8, 0),
    (0, 8),
    (-9, 0),
    (0, -9),
    (8, 7),
    (-8, -9),
    (15, -16),
    (16, 0),
    (0, -16),
    (-17, 17),
    (24, -24),
    (31, -32),
    (100, -100),
    (-128, 127),
    (255, -256),
    (i32::MAX, i32::MIN),
];

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

const OP_CONSTANT: u16 = 43;
const OP_EXT_INST: u16 = 12;
const OP_IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
const OP_IMAGE_FETCH: u16 = 95;
const OP_IMAGE_GATHER: u16 = 96;
const OP_BIT_FIELD_S_EXTRACT: u16 = 202;
const GLSL_SCLAMP: u32 = 45;

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

/// The value of the scalar `OpConstant` `id`.
fn constant(words: &[u32], id: u32) -> Option<u32> {
    insts(words)
        .into_iter()
        .find(|(o, i)| *o == OP_CONSTANT && i.get(2) == Some(&id))
        .and_then(|(_, i)| i.get(3).copied())
}

/// Every dynamic offset of the authored case's fixture, on its sample and on its gather, is
/// `OpBitFieldSExtract(v, 0, 4)` of the loaded offset, and no `SClamp` reaches it (the clamp
/// answered 7 where Metal applies -8).
#[test]
fn dynamic_offsets_are_read_as_their_low_four_bits() {
    let path = engine_dir().join("a2_dynamic_texel_offset.ll");
    let Some(bc) = skip_missing(llvm_assemble(&path), "a2_dynamic_texel_offset") else {
        return;
    };
    let p = m3();
    let a = compile(&bc, "a2_dynamic_texel_offset", &p).expect("compile");
    let (_, words) = link_and_validate(&a, &p).expect("L3");
    let all = insts(&words);
    let mut offsets = offset_operands(&words, OP_IMAGE_SAMPLE_EXPLICIT_LOD, 5);
    offsets.extend(offset_operands(&words, OP_IMAGE_GATHER, 6));
    assert_eq!(offsets.len(), 2, "the sample's and the gather's");
    for id in offsets {
        let (_, def) = all
            .iter()
            .find(|(_, i)| i.get(2) == Some(&id))
            .expect("the offset's definition");
        assert_eq!(
            def[0] & 0xffff,
            u32::from(OP_BIT_FIELD_S_EXTRACT),
            "{def:?}"
        );
        assert_eq!(
            (constant(&words, def[4]), constant(&words, def[5])),
            (Some(0), Some(4)),
            "bits 0..4"
        );
    }
    assert!(!all
        .iter()
        .any(|(o, i)| *o == OP_EXT_INST && i.get(4) == Some(&GLSL_SCLAMP)));
}

/// The combined image-sampler binding a module declares for texture slot 0.
fn combined_slot(m: &TypedModule) -> ResourceSlot {
    m.globals()
        .iter()
        .find_map(|g| match g.kind {
            GlobalKind::Resource(
                s @ ResourceSlot {
                    kind: ResourceKind::CombinedTextureSampler(_),
                    ..
                },
            ) => Some(s),
            _ => None,
        })
        .expect("a combined binding")
}

/// Resources of a dispatch of one-thread threadgroups: `[[buffer(i)]]` holding `init[i]`, and the
/// execution status.
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

/// `m` over `groups` one-thread threadgroups with `image` bound, in both scheduling orders, which
/// must agree: the final buffers.
fn run(m: &TypedModule, res: &Resources, image: &Image, groups: u32) -> Vec<Vec<u8>> {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [groups, 1, 1],
            order,
            ..Config::default()
        };
        let (o, _) = run_with_images(m, &cfg, res.clone(), vec![image.clone()])
            .unwrap_or_else(|e| panic!("interpreter: {e:?}"));
        outs.push(o.buffers);
    }
    assert_eq!(outs[0], outs[1], "the scheduling orders disagree");
    outs.swap_remove(0)
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The words of a nearest, clamp-to-edge, pixel-coordinate constexpr sampler (`coord::pixel`,
/// `filter::nearest`); with bits 9 and 11 set it is `filter::linear`.
const PIXEL_NEAREST: u64 = 34_901_797_601_050_697;
const LINEAR: u64 = (1 << 9) | (1 << 11);

/// A kernel sampling `texture2d<float, sample>` slot 0 through the nearest pixel sampler at the
/// centre of texel (32, 32) with lane `tid`'s `int2` offset from buffer 0, storing the sample's
/// `x` to `float` buffer 1.
fn offset_kernel() -> String {
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {PIXEL_NEAREST}, i64 0], align 8

define void @k(ptr addrspace(1) %t, ptr addrspace(1) %offs, ptr addrspace(1) %out, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <2 x i32>, ptr addrspace(1) %offs, i64 %i
  %o = load <2 x i32>, ptr addrspace(1) %p, align 8
  %r = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> <float 32.5, float 32.5>, i1 true, <2 x i32> %o, i1 true, float 0.0, float 0.0, i32 0)
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  %x = extractelement <4 x float> %v, i64 0
  %q = getelementptr inbounds float, ptr addrspace(1) %out, i64 %i
  store float %x, ptr addrspace(1) %q, align 4
  ret void
}}

declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{{!0}}
!air.sampler_states = !{{!7}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"int2", !"air.arg_name", !"offs"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}}
!6 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
!7 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
"#
    )
}

/// A nearest pixel sampler's exact fetch adds the offset Metal applies. Over a 64x64 image whose
/// texel `(x, y)` holds `x + 64y`, the authored case's sixteen offsets land on `(32 + applied(ox),
/// 32 + applied(oy))`, on the interpreter, in both scheduling orders.
#[test]
fn a_pixel_fetch_adds_the_offset_metal_applies() {
    let Some(bc) = assemble("offset", &offset_kernel()) else {
        return;
    };
    let p = m3();
    let a = compile(&bc, "k", &p).expect("compile");
    link_and_validate(&a, &p).expect("L3");
    assert_eq!(
        a.strategies
            .entries
            .get("sampler.pixel_fetch")
            .map(|u| u.uses),
        Some(1)
    );
    let (plan, m) = typed_module(&bc, "k", &p).expect("typed module");
    let mut image = Image::new(combined_slot(&m), [64, 64, 1], 1, 1, 1);
    for y in 0..64 {
        for x in 0..64 {
            let at = image.index(0, 0, 0, [x, y, 0]).expect("texel");
            image.texels[at] = [((x + 64 * y) as f32).to_bits(), 0, 0, 1f32.to_bits()];
        }
    }
    let offs: Vec<u8> = OFFSETS
        .iter()
        .flat_map(|(x, y)| [x.to_le_bytes(), y.to_le_bytes()])
        .flatten()
        .collect();
    let res = resources(&plan, &[offs, vec![0xab; 4 * OFFSETS.len()]]);
    let out = words(&run(&m, &res, &image, OFFSETS.len() as u32)[1]);
    for (lane, (ox, oy)) in OFFSETS.iter().enumerate() {
        let (x, y) = (32 + applied(*ox), 32 + applied(*oy));
        assert_eq!(
            f32::from_bits(out[lane]),
            (x + 64 * y) as f32,
            "lane {lane}: offset ({ox}, {oy})"
        );
    }
}

/// A kernel subsampling `texture2d<ushort, sample>` slot 0 through the pixel sampler `words` at
/// `(2 gid.x, 2 gid.y)`, storing the `ushort4` to buffer 0 at `gid.x + 2 gid.y`.
fn integer_kernel(words: u64) -> String {
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8

define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, <2 x i32> %gid) {{
  %gx = extractelement <2 x i32> %gid, i64 0
  %gy = extractelement <2 x i32> %gid, i64 1
  %x2 = shl i32 %gx, 1
  %y2 = shl i32 %gy, 1
  %fx = uitofp i32 %x2 to float
  %fy = uitofp i32 %y2 to float
  %c0 = insertelement <2 x float> undef, float %fx, i64 0
  %c = insertelement <2 x float> %c0, float %fy, i64 1
  %r = call {{ <4 x i16>, i8 }} @air.sample_texture_2d.u.v4i16(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %c, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %v = extractvalue {{ <4 x i16>, i8 }} %r, 0
  %row = shl i32 %gy, 1
  %lin = add i32 %row, %gx
  %i = zext i32 %lin to i64
  %q = getelementptr inbounds <4 x i16>, ptr addrspace(1) %out, i64 %i
  store <4 x i16> %v, ptr addrspace(1) %q, align 8
  ret void
}}

declare {{ <4 x i16>, i8 }} @air.sample_texture_2d.u.v4i16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{{!0}}
!air.sampler_states = !{{!6}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<ushort, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ushort4", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}}
!6 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
"#
    )
}

/// An integer texture is never filtered. Through a linear pixel sampler (the authored case
/// `subsample_quarter_luma`'s, whose Metal words are the nearest texels) an integer texture is
/// fetched exactly like through a nearest one, with no scaling note, and the layout's immutable
/// sampler is `NEAREST`, so no Vulkan sampler asks the integer view for linear filtering. On the
/// interpreter every thread reads texel `(2x, 2y)` of a 4x4 image of distinct texels, the
/// corner a linear filter would blend with three neighbours.
#[test]
fn an_integer_texture_samples_nearest_through_a_linear_sampler() {
    let p = m3();
    for (name, words) in [
        ("nearest", PIXEL_NEAREST),
        ("linear", PIXEL_NEAREST | LINEAR),
    ] {
        let Some(bc) = assemble(name, &integer_kernel(words)) else {
            return;
        };
        let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{name}: {e}"));
        let (_, spirv) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name}: L3: {e}"));
        assert_eq!(
            a.strategies
                .entries
                .get("sampler.pixel_fetch")
                .map(|u| u.uses),
            Some(1),
            "{name}"
        );
        let ops: Vec<u16> = insts(&spirv).into_iter().map(|(o, _)| o).collect();
        assert!(ops.contains(&OP_IMAGE_FETCH), "{name}");
        assert!(!ops.contains(&OP_IMAGE_SAMPLE_EXPLICIT_LOD), "{name}");
        assert!(a.precision.is_empty(), "{name}: {:?}", a.precision);
        let l = link(&[&a], &LinkInputs::compute(), &p).expect("link");
        let b = l.layout.sets[0]
            .bindings
            .iter()
            .find(|b| matches!(b.source, BindingSource::CombinedTextureSampler { .. }))
            .expect("combined binding");
        let baked = b.immutable_sampler.expect("immutable sampler");
        assert_eq!(
            (baked.min_filter, baked.mag_filter, baked.mipmap_mode),
            (
                vk::Filter::NEAREST,
                vk::Filter::NEAREST,
                vk::SamplerMipmapMode::NEAREST
            ),
            "{name}"
        );

        let (plan, m) = typed_module(&bc, "k", &p).expect("typed module");
        let mut image = Image::new(combined_slot(&m), [4, 4, 1], 1, 1, 1);
        let texel = |x: u32, y: u32| [10 * x + y, 100 + x, 200 + y, 300 + x + y];
        for y in 0..4 {
            for x in 0..4 {
                let at = image.index(0, 0, 0, [x, y, 0]).expect("texel");
                image.texels[at] = texel(x, y);
            }
        }
        let mut res = resources(&plan, &[vec![0xab; 8 * 4]]);
        res.spec.insert(SpecKey::LocalSize(Axis::X), 2);
        res.spec.insert(SpecKey::LocalSize(Axis::Y), 2);
        res.spec.insert(SpecKey::FlatLocalSize, 4);
        let out = run(&m, &res, &image, 1);
        for y in 0..2u32 {
            for x in 0..2u32 {
                let at = 8 * (x + 2 * y) as usize;
                let got: Vec<u32> = out[0][at..at + 8]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u32::from(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                assert_eq!(got, texel(2 * x, 2 * y), "{name}: thread ({x}, {y})");
            }
        }
    }
}

/// The Metal words of the authored case `a3_multi_level_break_tail` (032467bc, ORACLE_ABI v34,
/// Apple M3 Max, macOS 26.7): `out[2t]` is the `simd_or` mask of the lanes that ran lane `t`'s
/// tail with it, `out[2t + 1]` its stop `i << 8 | j` (`0xFFFF` and mask 0 when it never stops).
const MULTI_LEVEL_BREAK_METAL: [u32; 64] = [
    0x08428421, 0x00000000, 0x4214210a, 0x00000103, 0x00000000, 0x0000ffff, 0x4214210a, 0x00000100,
    0x10a10850, 0x00000203, 0x08428421, 0x00000001, 0x10a10850, 0x00000200, 0x00000000, 0x0000ffff,
    0x4214210a, 0x00000101, 0x00000000, 0x0000ffff, 0x08428421, 0x00000002, 0x10a10850, 0x00000201,
    0x00000000, 0x0000ffff, 0x4214210a, 0x00000102, 0x00000000, 0x0000ffff, 0x08428421, 0x00000003,
    0x10a10850, 0x00000202, 0x08428421, 0x00000000, 0x4214210a, 0x00000103, 0x00000000, 0x0000ffff,
    0x4214210a, 0x00000100, 0x10a10850, 0x00000203, 0x08428421, 0x00000001, 0x10a10850, 0x00000200,
    0x00000000, 0x0000ffff, 0x4214210a, 0x00000101, 0x00000000, 0x0000ffff, 0x08428421, 0x00000002,
    0x10a10850, 0x00000201, 0x00000000, 0x0000ffff, 0x4214210a, 0x00000102, 0x00000000, 0x0000ffff,
];

/// The tail of a two-level break runs after the inner loop, once per outer iteration, with every
/// lane that left the inner loop through it in that iteration, as AGX runs it. The fixture's 32
/// lanes stop at flat iteration `7t mod 17` of a 3 x 4 nest; on the interpreter's one 32-lane
/// subgroup, under maximal reconvergence, the engine's module writes Metal's 64 words. Each
/// mask is the set of lanes stopping in the same outer iteration (the in-loop tail answered only
/// the lanes stopping at the same `(i, j)`).
#[test]
fn a_multi_level_break_tail_runs_once_per_outer_iteration() {
    const LANES: u32 = 32;
    const TRIPS: [u32; 2] = [3, 4];
    let path = engine_dir().join("a3_multi_level_break_tail.ll");
    let Some(bc) = skip_missing(llvm_assemble(&path), "a3_multi_level_break_tail") else {
        return;
    };
    let p = m3();
    let (plan, m) = typed_module(&bc, "a3_multi_level_break_tail", &p).expect("typed module");
    let stop: Vec<u32> = (0..LANES).map(|t| 7 * t % 17).collect();
    let bytes = |w: &[u32]| -> Vec<u8> { w.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mut res = resources(
        &plan,
        &[bytes(&stop), bytes(&TRIPS), vec![0xab; 8 * LANES as usize]],
    );
    res.spec.insert(SpecKey::LocalSize(Axis::X), LANES.into());
    res.spec.insert(SpecKey::FlatLocalSize, LANES.into());
    let out = interp_run(&m, &Config::default(), res)
        .unwrap_or_else(|e| panic!("interpreter: {e:?}"))
        .buffers;
    assert_eq!(words(&out[2]), MULTI_LEVEL_BREAK_METAL);

    // The reading: lanes share a mask exactly when they stop in the same outer iteration.
    let inner = TRIPS[1];
    let flat = TRIPS[0] * inner;
    for t in 0..LANES as usize {
        let s = stop[t];
        let mask = (0..LANES)
            .filter(|&u| {
                s < flat && stop[u as usize] < flat && stop[u as usize] / inner == s / inner
            })
            .fold(0u32, |m, u| m | (1 << u));
        let at = if s < flat {
            ((s / inner) << 8) | (s % inner)
        } else {
            0xffff
        };
        assert_eq!(
            MULTI_LEVEL_BREAK_METAL[2 * t..2 * t + 2],
            [mask, at],
            "lane {t}"
        );
    }
}
