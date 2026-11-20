//! Every `half` bit pattern through the moves a kernel makes of a `half` without arithmetic, on
//! the CPU interpreter: where Metal moves the 16 bits, the module must hand back the same 16 bits.
//!
//! One kernel of 16,384 threads, thread `g` holding the `half4` of bits `4g` to `4g + 3` (loaded
//! from a buffer, so they are decoded values, not constants), moves it:
//!
//! - straight to a device buffer (move 0);
//! - through a `select` on a runtime word to a device buffer (move 1: the store cannot reuse the
//!   bits the load decoded, and runs the encode of the decoded value);
//! - through the select and a threadgroup `half4` array, read back by another thread after a
//!   barrier (move 2);
//! - into an `RGBA16Float` storage texture, as loaded and through the select, through the class the
//!   binder chooses for the profile (`storage_image_choice`): the module hands the device the
//!   widened half, whose store is the device's conversion (a NaN's encoding is the device's there,
//!   recorded as `PrecisionNote::DeviceNanEncoding`), so the texel must be the exact widening of
//!   each value that is not a NaN, and a NaN for each NaN.
//!
//! Each runs on every committed profile under the default policy and under
//! `DevicePolicy::exact_answers`. A module may leave a NaN's bits to the device, and then says so:
//! a `half` read from its bits by the device's widening (every profile with `shaderFloat16`) or a
//! NaN's bits left to the device's arithmetic and conversions, each `PrecisionNote::NativeNanEncoding`
//! (under either policy), or the native conversion's `PrecisionNote::NativeHalfConversion`. There
//! a NaN pattern need only come back a NaN (a move through the select or the threadgroup array; a
//! move straight to a buffer reuses the loaded bits). No other pattern may change, nor any where
//! the module records neither note.
//!
//! The profiles and policies compile to a few distinct modules: each runs once, all at once on
//! threads of their own.
//!
//! `llvm-as` assembles the kernel; without it the test skips visibly.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::device::profile::DeviceProfile;
use crate::device::DevicePolicy;
use crate::lower::select::{LowerInputs, Plan};
use crate::metal::format::storage_image_choice;
use crate::metal::PixelFormat;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{
    Axis, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage,
};
use crate::pipeline::testing::{compile_with, llvm_assemble, typed_module_with, AssembleError};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles_with;

/// The patterns, the threads (four patterns each), the threads per threadgroup.
const N: u32 = 1 << 16;
const THREADS: u32 = N / 4;
const LOCAL: u32 = 256;

/// The texture's side, two texels a thread: the loaded `half4` in the top half, the selected one in
/// the bottom half.
const WIDTH: u32 = 256;
const HEIGHT: u32 = 2 * THREADS / WIDTH;

const KERNEL: &str = r#"@h = internal addrspace(3) global [256 x <4 x half>] undef, align 8

define void @half_moves(ptr addrspace(1) %src, ptr addrspace(1) %out, ptr addrspace(1) %flags, ptr addrspace(1) %img, i32 %g, i32 %t) {
  %ps = getelementptr inbounds <4 x half>, ptr addrspace(1) %src, i32 %g
  %a = load <4 x half>, ptr addrspace(1) %ps, align 8
  %f = load i32, ptr addrspace(1) %flags, align 4
  %c = icmp eq i32 %f, 0
  %x = select i1 %c, <4 x half> %a, <4 x half> <half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00>
  %o0 = getelementptr inbounds <4 x half>, ptr addrspace(1) %out, i32 %g
  store <4 x half> %a, ptr addrspace(1) %o0, align 8
  %g1 = add i32 %g, 16384
  %o1 = getelementptr inbounds <4 x half>, ptr addrspace(1) %out, i32 %g1
  store <4 x half> %x, ptr addrspace(1) %o1, align 8
  %pt = getelementptr inbounds [256 x <4 x half>], ptr addrspace(3) @h, i32 0, i32 %t
  store <4 x half> %x, ptr addrspace(3) %pt, align 8
  tail call void @air.wg.barrier(i32 2, i32 1)
  %r = sub i32 255, %t
  %pr = getelementptr inbounds [256 x <4 x half>], ptr addrspace(3) @h, i32 0, i32 %r
  %y = load <4 x half>, ptr addrspace(3) %pr, align 8
  %base = sub i32 %g, %t
  %gr = add i32 %base, %r
  %g2 = add i32 %gr, 32768
  %o2 = getelementptr inbounds <4 x half>, ptr addrspace(1) %out, i32 %g2
  store <4 x half> %y, ptr addrspace(1) %o2, align 8
  %cx = and i32 %g, 255
  %cy = lshr i32 %g, 8
  %cy2 = add i32 %cy, 64
  %cx16 = trunc i32 %cx to i16
  %cy16 = trunc i32 %cy to i16
  %cy216 = trunc i32 %cy2 to i16
  %c0 = insertelement <2 x i16> undef, i16 %cx16, i64 0
  %c1 = insertelement <2 x i16> %c0, i16 %cy16, i64 1
  %c2 = insertelement <2 x i16> %c0, i16 %cy216, i64 1
  call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %img, <2 x i16> %c1, <4 x half> %a, i16 0, i32 2)
  call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %img, <2 x i16> %c2, <4 x half> %x, i16 0, i32 2)
  ret void
}

declare void @air.wg.barrier(i32, i32)
declare void @air.write_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, <4 x half>, i16, i32)

!air.kernel = !{!0}
!0 = !{ptr @half_moves, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7, !8}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"flags"}
!6 = !{i32 3, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"img"}
!7 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"g"}
!8 = !{i32 5, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

fn assemble() -> Option<Vec<u8>> {
    let dir = std::env::temp_dir().join(format!("m2v-half-moves-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, KERNEL).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected half_moves: {why}"),
    }
}

fn slot(kind: ResourceKind, index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind,
        index,
    }
}

/// The resources of one run: the patterns, the output (three blocks of `N` halves), the flag word
/// 0, and the local size.
fn resources(plan: &Plan) -> Resources {
    let mut spec = BTreeMap::new();
    spec.insert(SpecKey::LocalSize(Axis::X), u64::from(LOCAL));
    spec.insert(SpecKey::LocalSize(Axis::Y), 1);
    spec.insert(SpecKey::LocalSize(Axis::Z), 1);
    spec.insert(SpecKey::FlatLocalSize, u64::from(LOCAL));
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (
                PushField::SlotBindOffset(_)
                | PushField::DispatchRegion
                | PushField::Specialized(_),
                PushLoc::Push { .. },
            ) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let src: Vec<u8> = (0..N).flat_map(|h| (h as u16).to_le_bytes()).collect();
    let buffer = |index: u32, bytes: Vec<u8>| Buffer {
        slot: Some(slot(ResourceKind::Buffer, index)),
        address: None,
        bytes,
    };
    Resources {
        buffers: vec![
            buffer(0, src),
            buffer(1, vec![0; 3 * N as usize * 2]),
            buffer(2, vec![0; 4]),
        ],
        push,
        spec,
    }
}

fn is_nan(h: u16) -> bool {
    h & 0x7C00 == 0x7C00 && h & 0x3FF != 0
}

/// The `f32` bits of half `h` widened exactly (a NaN's payload in the top of the mantissa).
fn widened(h: u16) -> u32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from(h >> 10) & 0x1F;
    let man = u32::from(h) & 0x3FF;
    match exp {
        0 => sign | (f32::from(man as u16) * 2f32.powi(-24)).to_bits(),
        0x1F => sign | 0x7F80_0000 | (man << 13),
        e => sign | ((e + 112) << 23) | (man << 13),
    }
}

/// What one run found against the raw copy.
struct Found {
    /// Patterns moved to a different word, by move.
    changed: [Vec<(u16, u16)>; 3],
    /// Patterns whose texel channel is not the expected one (as loaded, through the select).
    texels: Vec<(u16, u32)>,
}

/// One distinct module of the kernel: what runs, and the notes its translation records.
struct Compiled {
    plan: Plan,
    module: TypedModule,
    notes: Vec<String>,
}

/// The kernel compiled for `p`, and its SPIR-V words (which name the module); `None` where the
/// profile refuses the kernel.
fn compile_once(
    stem: &str,
    p: &DeviceProfile,
    bc: &[u8],
    what: &str,
) -> Option<(Arc<[u32]>, Compiled)> {
    let choice = storage_image_choice(PixelFormat::RGBA16Float, p).ok()?;
    let mut inputs = LowerInputs::new();
    inputs.storage_images.insert(
        SlotRef {
            stage: Stage::Kernel,
            class: SlotClass::Texture,
            index: 0,
        },
        choice,
    );
    let (plan, module) = match typed_module_with(bc, "half_moves", p, &inputs) {
        Ok(x) => x,
        Err(crate::error::Error::NeedsInput(_) | crate::error::Error::Unsupported(_)) => {
            return None
        }
        Err(e) => panic!("{what} {stem}: {e}"),
    };
    let artifact = compile_with(bc, "half_moves", p, &inputs).expect("compile");
    let notes: Vec<String> = artifact
        .precision
        .iter()
        .map(|n| format!("{n:?}"))
        .collect();
    let words: Arc<[u32]> = Arc::from(artifact.module.words.as_slice());
    Some((
        words,
        Compiled {
            plan,
            module,
            notes,
        },
    ))
}

/// Run `c` over every pattern.
fn run_once(c: &Compiled, what: &str) -> Found {
    let cfg = Config {
        workgroups: [THREADS / LOCAL, 1, 1],
        order: Order::Forward,
        step_limit: 1 << 30,
        ..Config::default()
    };
    let mut img = Image::new(slot(ResourceKind::Texture, 0), [WIDTH, HEIGHT, 1], 1, 1, 1);
    img.texels.fill([0xDEAD_0001; 4]);
    let (o, imgs) = run_with_images(&c.module, &cfg, resources(&c.plan), vec![img])
        .unwrap_or_else(|e| panic!("{what}: run: {e:?}"));
    let out: Vec<u16> = o.buffers[1]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect();
    let mut changed: [Vec<(u16, u16)>; 3] = Default::default();
    for (k, list) in changed.iter_mut().enumerate() {
        for h in 0..N {
            let got = out[k * N as usize + h as usize];
            if got != h as u16 {
                list.push((h as u16, got));
            }
        }
    }
    let mut texels = Vec::new();
    for h in 0..N {
        let (g, lane) = (h / 4, (h % 4) as usize);
        for y in [g / WIDTH, THREADS / WIDTH + g / WIDTH] {
            let at = imgs[0].index(0, 0, 0, [g % WIDTH, y, 0]).expect("texel");
            let t = imgs[0].texels[at][lane];
            let h = h as u16;
            let ok = if is_nan(h) {
                f32::from_bits(t).is_nan()
            } else {
                t == widened(h)
            };
            if !ok {
                texels.push((h, t));
            }
        }
    }
    Found { changed, texels }
}

/// Every `half` pattern comes back as its bits from each move (buffer, select, threadgroup array)
/// and as its exact widening from the texel on every profile under every policy, but that a NaN pattern may come back
/// another NaN where the module records the note that leaves a NaN's bits to the device (the module
/// docs).
#[test]
fn every_half_pattern_moves_as_its_bits() {
    let Some(bc) = assemble() else { return };
    // Each (policy, profile) and the module it compiles to; each distinct module runs once.
    let mut runs: Vec<(&str, String, Arc<[u32]>)> = Vec::new();
    let mut modules: BTreeMap<Arc<[u32]>, Compiled> = BTreeMap::new();
    {
        let (policy_name, policy) = ("default", DevicePolicy::default());
        for (stem, p) in committed_profiles_with(policy).expect("profiles") {
            let Some((words, compiled)) = compile_once(&stem, &p, &bc, policy_name) else {
                continue;
            };
            modules.entry(Arc::clone(&words)).or_insert(compiled);
            runs.push((policy_name, stem, words));
        }
    }
    let found: BTreeMap<Arc<[u32]>, Found> = std::thread::scope(|s| {
        let handles: Vec<_> = modules
            .iter()
            .enumerate()
            .map(|(i, (words, c))| (words, s.spawn(move || run_once(c, &format!("module {i}")))))
            .collect();
        handles
            .into_iter()
            .map(|(words, h)| (Arc::clone(words), h.join().expect("run")))
            .collect()
    });
    eprintln!("{} runs, {} distinct modules", runs.len(), modules.len());
    let mut report = Vec::new();
    let mut device_nans = Vec::new();
    for (policy_name, stem, words) in &runs {
        let (c, found) = (&modules[words], &found[words]);
        let notes = &c.notes;
        let noted = notes
            .iter()
            .any(|n| n == "NativeHalfConversion" || n == "NativeNanEncoding");
        let nan_only = |list: &[(u16, u16)]| list.iter().all(|&(h, got)| is_nan(h) && is_nan(got));
        let mut moved_nans = 0;
        for (k, list) in found.changed.iter().enumerate() {
            if list.is_empty() {
                continue;
            }
            if noted && k > 0 && nan_only(list) {
                moved_nans += list.len();
                continue;
            }
            report.push(format!(
                "{policy_name} {stem} move {k}: {} changed, NaNs alone {}, first {:x?}, notes \
                 {notes:?}",
                list.len(),
                nan_only(list),
                &list[..list.len().min(4)]
            ));
        }
        if !found.texels.is_empty() {
            report.push(format!(
                "{policy_name} {stem}: {} texel channels, first {:x?}, notes {notes:?}",
                found.texels.len(),
                &found.texels[..found.texels.len().min(4)]
            ));
        }
        if moved_nans > 0 {
            device_nans.push(format!("{policy_name} {stem}: {moved_nans} ({notes:?})"));
        }
    }
    eprintln!("NaN patterns the device's, as noted: {device_nans:#?}");
    assert!(report.is_empty(), "{report:#?}");
}
