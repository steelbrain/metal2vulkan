//! A stage's private memory against the device's Function storage.
//!
//! Owned synthetic functions: the kernel `words` keeps a private array of `N` 32-bit words,
//! written and read at indices only the thread id decides (so it stays an array in memory), as do
//! the vertex function `vwords` (by its vertex ID) and the fragment function `fwords` (by its
//! position); `lookup` reads a constant table at such an index (held per thread as a `Private`
//! variable); `fixed` keeps an array of `N` words it reaches only at constant indices; `plain`
//! keeps no private memory.
//!
//! - Every device's Function storage is the floor, 16 KiB, which holds with no note: an array of
//!   16 KiB, the constant table, a kernel without private memory (`amd-rdna3-radv`). Past it a
//!   kernel is refused needing it, and a vertex or fragment function runs, recording
//!   `PrecisionNote::FunctionStorageAssumed`, up to the 192 KiB Metal was measured to hold; 64
//!   bytes more is refused needing that.
//! - An array reached only at constant indices counts nothing: six times the floor compiles with
//!   no note on the RTX 5080's profile and RDNA 3's.
//! - An array whose every index the constants bound counts only what they reach: the kernel
//!   `reached` declares 256 KiB and reaches 128 bytes of it, which both profiles hold with no
//!   note; with one index unbounded it keeps the 256 KiB and is refused on both. So do the owned
//!   MSL fixtures `engine/private_reach_tiles`'s kernels on RDNA 3's profile: one whose tile sizes,
//!   function constants, reach 128 bytes of its 256 KiB, and one whose loop a buffer word bounds.
//!
//! `llvm-as` assembles the functions; a missing tool skips visibly.

use crate::budget::SPIRV_WORDS;
use crate::device::profile::{HandleDomain, PointerDomain};
use crate::device::DevicePolicy;
use crate::device::{DeviceProfile, VkLimit};
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::library::{FunctionConstantValue, FunctionConstantValues};
use crate::lower::select::LowerInputs;
use crate::metal::DataType;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::compile::{compile_bitcode_linked, EntryRef, LinkedSource};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::PushField;
use crate::pipeline::testing::{
    compile, compile_kept, engine_dir, link_and_validate, llvm_assemble, meter, profile,
    AssembleError,
};

/// The functions, `words`, `vwords` and `fwords` with a private array of `n` words.
fn module(n: u32) -> Option<Vec<u8>> {
    let text = format!(
        r#"; Owned synthetic functions: private memory of {n} words, a constant table, none.
source_filename = "private_memory.metal"

@table = internal addrspace(2) constant [8 x i32] [i32 3, i32 1, i32 4, i32 1, i32 5, i32 9, i32 2, i32 6], align 4

define void @words(ptr addrspace(1) %out, i32 %gid) {{
entry:
  %a = alloca [{n} x i32], align 4
  %w = urem i32 %gid, {n}
  %p = getelementptr inbounds [{n} x i32], ptr %a, i32 0, i32 %w
  store i32 %gid, ptr %p, align 4
  %r = mul i32 %gid, 7
  %i = urem i32 %r, {n}
  %q = getelementptr inbounds [{n} x i32], ptr %a, i32 0, i32 %i
  %x = load i32, ptr %q, align 4
  %g = zext i32 %gid to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g
  store i32 %x, ptr addrspace(1) %o, align 4
  ret void
}}

define void @lookup(ptr addrspace(1) %out, i32 %gid) {{
entry:
  %i = and i32 %gid, 7
  %q = getelementptr inbounds [8 x i32], ptr addrspace(2) @table, i32 0, i32 %i
  %x = load i32, ptr addrspace(2) %q, align 4
  %g = zext i32 %gid to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g
  store i32 %x, ptr addrspace(1) %o, align 4
  ret void
}}

define void @fixed(ptr addrspace(1) %out, i32 %gid) {{
entry:
  %a = alloca [{n} x i32], align 4
  %p0 = getelementptr inbounds [{n} x i32], ptr %a, i32 0, i32 0
  store i32 %gid, ptr %p0, align 4
  %pl = getelementptr inbounds [{n} x i32], ptr %a, i32 0, i32 {last}
  %g3 = mul i32 %gid, 3
  store i32 %g3, ptr %pl, align 4
  %x = load i32, ptr %p0, align 4
  %y = load i32, ptr %pl, align 4
  %s = add i32 %x, %y
  %g = zext i32 %gid to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g
  store i32 %s, ptr addrspace(1) %o, align 4
  ret void
}}

define void @plain(ptr addrspace(1) %out, i32 %gid) {{
entry:
  %g = zext i32 %gid to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g
  store i32 %gid, ptr addrspace(1) %o, align 4
  ret void
}}

define <4 x float> @vwords(i32 %vid) {{
entry:
  %a = alloca [{n} x i32], align 4
  %w = urem i32 %vid, {n}
  %p = getelementptr inbounds [{n} x i32], ptr %a, i32 0, i32 %w
  store i32 %vid, ptr %p, align 4
  %r = mul i32 %vid, 7
  %i = urem i32 %r, {n}
  %q = getelementptr inbounds [{n} x i32], ptr %a, i32 0, i32 %i
  %x = load i32, ptr %q, align 4
  %f = uitofp i32 %x to float
  %c = insertelement <4 x float> <float 0.0, float 0.0, float 0.0, float 1.0>, float %f, i64 0
  ret <4 x float> %c
}}

define <4 x float> @fwords(<4 x float> %pos) {{
entry:
  %a = alloca [{n} x i32], align 4
  %px = extractelement <4 x float> %pos, i64 0
  %gid = fptoui float %px to i32
  %w = urem i32 %gid, {n}
  %p = getelementptr inbounds [{n} x i32], ptr %a, i32 0, i32 %w
  store i32 %gid, ptr %p, align 4
  %r = mul i32 %gid, 7
  %i = urem i32 %r, {n}
  %q = getelementptr inbounds [{n} x i32], ptr %a, i32 0, i32 %i
  %x = load i32, ptr %q, align 4
  %f = uitofp i32 %x to float
  %c = insertelement <4 x float> <float 0.0, float 0.0, float 0.0, float 1.0>, float %f, i64 0
  ret <4 x float> %c
}}

!air.kernel = !{{!0, !5, !6, !7}}
!air.vertex = !{{!8}}
!air.fragment = !{{!13}}
!0 = !{{ptr @words, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!5 = !{{ptr @lookup, !1, !2}}
!6 = !{{ptr @plain, !1, !2}}
!7 = !{{ptr @fixed, !1, !2}}
!8 = !{{ptr @vwords, !9, !11}}
!9 = !{{!10}}
!10 = !{{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}}
!11 = !{{!12}}
!12 = !{{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}}
!13 = !{{ptr @fwords, !14, !16}}
!14 = !{{!15}}
!15 = !{{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}}
!16 = !{{!17}}
!17 = !{{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}}
"#,
        last = n - 1
    );
    match metal2vulkan_devtools::llvm_assemble_text(&text) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the functions: {why}"),
    }
}

/// `Unsupported{PrivateMemory}` needing `required` bytes a thread, of which the device holds
/// `available`.
fn needs(required: u64, available: u64) -> Result<(), Error> {
    Err(Error::Unsupported(Unsupported::needs_any_of(
        Feature::PrivateMemory,
        vec![vec![Capability::Limit {
            limit: VkLimit::FunctionStorageBytesPerThread,
            required,
            available,
        }]],
    )))
}

/// Function `entry` of [`module`]`(n)` on `stem`: its precision notes, or what refused it.
fn notes(stem: &str, n: u32, entry: &str) -> Option<Result<Vec<PrecisionNote>, Error>> {
    notes_on(
        &profile(stem, crate::device::DevicePolicy::RESIDENT_TABLES).expect("profile"),
        stem,
        n,
        entry,
    )
}

/// Function `entry` of [`module`]`(n)` on profile `p`, named `name`: its precision notes, or what
/// refused it.
fn notes_on(
    p: &DeviceProfile,
    name: &str,
    n: u32,
    entry: &str,
) -> Option<Result<Vec<PrecisionNote>, Error>> {
    let bitcode = module(n)?;
    Some(compile(&bitcode, entry, p).map(|a| {
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{name} {entry} {n}: L3: {e}"));
        a.precision.clone()
    }))
}

/// On a device not measured the floor holds with no note: 4096 words, the constant table and a
/// kernel without private memory record nothing, in a kernel, a vertex and a fragment function
/// alike. A kernel of 4112 words does not fit.
#[test]
fn private_memory_within_the_floor_records_nothing() {
    let stem = "amd-rdna3-radv";
    let assumed = PrecisionNote::FunctionStorageAssumed;
    for entry in ["words", "vwords", "fwords", "lookup", "plain"] {
        let Some(fits) = notes(stem, 4096, entry) else {
            return;
        };
        let fits = fits.unwrap_or_else(|e| panic!("{entry} 4096 words: {e}"));
        assert!(!fits.contains(&assumed), "{entry}: {fits:?}");
    }
    let past = notes(stem, 4112, "words").expect("assembled");
    assert_eq!(past.map(|_| ()), needs(4112 * 4, 16 * 1024));
}

/// On a device not measured a vertex or fragment function runs past the floor, recording that it
/// rests on no measurement: 17 KiB (4352 words), 100 KiB (25600) and Metal's 192 KiB (49152) a
/// thread; 49168 words are refused needing them, of the 192 KiB they may keep. A kernel of 17 KiB
/// is refused needing it, of the floor's 16 KiB.
#[test]
fn render_stages_past_the_floor_of_a_device_not_measured_run_noted_up_to_metal_s() {
    let stem = "amd-rdna3-radv";
    let assumed = PrecisionNote::FunctionStorageAssumed;
    for entry in ["vwords", "fwords"] {
        for n in [4352, 25600, 49152] {
            let Some(noted) = notes(stem, n, entry) else {
                return;
            };
            let noted = noted.unwrap_or_else(|e| panic!("{entry} {n} words: {e}"));
            assert!(noted.contains(&assumed), "{entry} {n} words: {noted:?}");
        }
        let past = notes(stem, 49168, entry).expect("assembled");
        assert_eq!(
            past.map(|_| ()),
            needs(49168 * 4, 192 * 1024),
            "{entry} 49168 words"
        );
    }
    let kernel = notes(stem, 4352, "words").expect("assembled");
    assert_eq!(kernel.map(|_| ()), needs(4352 * 4, 16 * 1024));
}

/// Under the identity domains a kernel whose private memory passes the floor keeps it in the
/// runtime's device memory, one slot an invocation (`memory::plan::ArenaLayout::device`): the
/// `words` kernel of 7176 words (28,704 bytes), 9216 (36 KiB) and 51,200 (200 KiB) compiles on the
/// 780M's, the Iris Xe's and an RTX 50's profiles with no note, pushes
/// `PushField::PrivateArenaAddress`, states `RuntimeFact::PrivateArena` of its arena's bytes, a
/// multiple of 16, holds no private bytes, and passes L3. Under the resident-table domains, whose
/// binder supplies no such memory, it is refused as before, needing its bytes of the floor.
#[test]
fn a_kernel_past_the_floor_keeps_its_private_memory_in_the_runtimes_memory_under_identity() {
    let identity = DevicePolicy {
        pointer_domain: PointerDomain::Identity,
        handle_domain: HandleDomain::Identity,
        ..DevicePolicy::RESIDENT_TABLES
    };
    for stem in ["amd-780m-radv", "intel-irisxe-anv", "nvidia-rtx50-linux"] {
        let p = profile(stem, identity).expect("profile");
        for n in [7176u32, 9216, 51_200] {
            let Some(bitcode) = module(n) else {
                return;
            };
            let what = format!("{stem}: words {n}");
            let (a, plan, _) =
                compile_kept(&bitcode, "words", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert!(plan.families.access.local.private_device, "{what}");
            assert_eq!(plan.private_bytes, 0, "{what}");
            assert!(
                plan.facts
                    .contains(&RuntimeFact::PrivateArena { slot_bytes: 4 * n }),
                "{what}: {:?}",
                plan.facts
            );
            assert!(
                (plan.push.fields.iter()).any(|f| f.field == PushField::PrivateArenaAddress),
                "{what}"
            );
            assert!(
                !a.precision.contains(&PrecisionNote::FunctionStorageAssumed),
                "{what}: {:?}",
                a.precision
            );
            let default = notes(stem, n, "words").expect("assembled");
            assert_eq!(
                default.map(|_| ()),
                needs(4 * u64::from(n), 16 * 1024),
                "{what}"
            );
        }
    }
}

/// An array reached only at constant indices counts nothing: a driver holds its parts in registers.
/// Six times the floor's 16 KiB compiles with no note on the RTX 5080's profile and on RDNA 3's;
/// the same array indexed at run time is refused.
#[test]
fn an_array_reached_only_at_constant_indices_counts_nothing() {
    let n = 3 * 8192;
    let Some(fits) = notes("nvidia-rtx5080-laptop", n, "fixed") else {
        return;
    };
    let fits = fits.unwrap_or_else(|e| panic!("{n} words at constant indices: {e}"));
    assert!(
        !fits.contains(&PrecisionNote::FunctionStorageAssumed),
        "{fits:?}"
    );
    let floor = notes("amd-rdna3-radv", n, "fixed").expect("assembled");
    let floor = floor.unwrap_or_else(|e| panic!("on the floor: {e}"));
    assert!(
        !floor.contains(&PrecisionNote::FunctionStorageAssumed),
        "{floor:?}"
    );
    let indexed = notes("nvidia-rtx5080-laptop", n, "words").expect("assembled");
    assert_eq!(indexed.map(|_| ()), needs(u64::from(n) * 4, 16 * 1024));
}

/// The kernel `reached` keeps `float a[64][64][16]`, 256 KiB a thread, as a kernel written for its
/// largest tile does: for `i` below 2 and `j` below 4 it writes `a[i][j][3] = gid + j`, then
/// stores `a[read][3][3]` to its output word, `read` being the LLVM value `read` names (of `%gid`).
fn reached(read: &str) -> Option<Vec<u8>> {
    let text = format!(
        r#"; Owned synthetic kernel: a private array its counters reach in part.
source_filename = "private_reach.metal"

define void @reached(ptr addrspace(1) %out, i32 %gid) {{
entry:
  %a = alloca [64 x [64 x [16 x float]]], align 4
  br label %outer
outer:
  %i = phi i32 [ 0, %entry ], [ %i1, %latch ]
  br label %inner
inner:
  %j = phi i32 [ 0, %outer ], [ %j1, %inner ]
  %s = add i32 %gid, %j
  %v = uitofp i32 %s to float
  %p = getelementptr inbounds [64 x [64 x [16 x float]]], ptr %a, i32 0, i32 %i, i32 %j, i32 3
  store float %v, ptr %p, align 4
  %j1 = add nuw i32 %j, 1
  %jd = icmp eq i32 %j1, 4
  br i1 %jd, label %latch, label %inner
latch:
  %i1 = add nuw i32 %i, 1
  %id = icmp eq i32 %i1, 2
  br i1 %id, label %done, label %outer
done:
  %k = {read}
  %q = getelementptr inbounds [64 x [64 x [16 x float]]], ptr %a, i32 0, i32 %k, i32 3, i32 3
  %x = load float, ptr %q, align 4
  %g = zext i32 %gid to i64
  %o = getelementptr inbounds float, ptr addrspace(1) %out, i64 %g
  store float %x, ptr addrspace(1) %o, align 4
  ret void
}}

!air.kernel = !{{!0}}
!0 = !{{ptr @reached, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
"#
    );
    match metal2vulkan_devtools::llvm_assemble_text(&text) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the kernel: {why}"),
    }
}

/// An array whose every index the constants bound keeps only what its indices reach: `float
/// a[64][64][16]` (256 KiB) reached at `a[0..2][0..4][3]` and `a[gid & 1][3][3]` is allocated as
/// `float a[2][4][4]`, 128 bytes a thread, which the floor's 16 KiB holds with no note on the RTX
/// 5080's profile and RDNA 3's. Read at `a[gid][3][3]`, an index with no bound, it keeps its 256
/// KiB and is refused on both, needing them.
#[test]
fn an_array_its_bounded_indices_reach_in_part_keeps_only_that_part() {
    let assumed = PrecisionNote::FunctionStorageAssumed;
    let Some(bounded) = reached("and i32 %gid, 1") else {
        return;
    };
    let unbounded = reached("add i32 %gid, 0").expect("assembled");
    for stem in ["nvidia-rtx5080-laptop", "amd-rdna3-radv"] {
        let p = profile(stem, crate::device::DevicePolicy::RESIDENT_TABLES).expect("profile");
        let a = compile(&bounded, "reached", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert_eq!(a.interface.private_bytes, 2 * 4 * 4 * 4, "{stem}");
        assert!(!a.precision.contains(&assumed), "{stem}: {:?}", a.precision);
        let declared = compile(&unbounded, "reached", &p).map(|_| ());
        assert_eq!(declared, needs(64 * 64 * 16 * 4, 16 * 1024), "{stem}");
    }
}

/// The owned fixture `engine/{stem}`, assembled, or `None` (skipped) without `llvm-as`.
fn fixture(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}: {why}"),
    }
}

/// The owned fixtures `engine/private_reach_tiles` (kernels written for their largest tile, each
/// thread keeping `float acc[32][32][4][4][4]`, 256 KiB, and calling the visible function
/// `postfix`) and `engine/private_reach_functions` (`postfix`), compiled from MSL by the Metal
/// compiler. With the tile sizes 2 and 4 (function constants 0 and 1) `private_reach_tiles`
/// reaches `acc[0..2][0..4][0..4][0][0]`: it holds 128 bytes a thread, which the floor's 16 KiB
/// holds with no note on RDNA 3's profile, and validates. `private_reach_rows`, whose outer loop runs as often as
/// a word of its input says, keeps the 256 KiB and is refused needing them.
#[test]
fn a_tile_kernel_holds_only_the_corner_its_tile_sizes_reach() {
    let (Some(tiles), Some(functions)) = (
        fixture("private_reach_tiles"),
        fixture("private_reach_functions"),
    ) else {
        return;
    };
    let uint = |v: u32| FunctionConstantValue::new(DataType::UInt, v.to_le_bytes());
    let sizes = FunctionConstantValues::defaults()
        .with_index(0, uint(2))
        .with_index(1, uint(4));
    let none = FunctionConstantValues::defaults();
    let linked = [LinkedSource::new("postfix", &functions, &none)];
    let p = profile(
        "amd-rdna3-radv",
        crate::device::DevicePolicy::RESIDENT_TABLES,
    )
    .expect("profile");
    let kernel = |name| {
        compile_bitcode_linked(
            &tiles,
            EntryRef::Name(name),
            (&sizes, &linked),
            &p,
            &LowerInputs::new(),
            &mut meter(),
            SPIRV_WORDS,
        )
    };
    let a = kernel("private_reach_tiles").unwrap_or_else(|e| panic!("{e}"));
    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("L3: {e}"));
    assert_eq!(a.interface.private_bytes, 2 * 4 * 4 * 4);
    assert!(
        !a.precision.contains(&PrecisionNote::FunctionStorageAssumed),
        "{:?}",
        a.precision
    );
    let rows = kernel("private_reach_rows").map(|_| ());
    assert_eq!(rows, needs(32 * 32 * 4 * 4 * 4 * 4, 16 * 1024));
}
