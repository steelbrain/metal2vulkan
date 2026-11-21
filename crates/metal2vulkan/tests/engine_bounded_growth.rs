//! What a compile costs must grow with the size of a function, not with its square: the new
//! engine's end-to-end guard.
//!
//! 500 MiB and 20 seconds per translation attempt are hard budgets (AGENTS.md), and neither gets
//! broken by a leak or a slow loop body. Both get broken by a quadratic term that a small graph
//! cannot show: every individual compile is correct, and then a large shader dies at the worker
//! boundary. An analysis can be quadratic in two ways, and the old engine shipped one of each:
//!
//! * **The representation**: a table per block that holds something per block (its structurizer's
//!   dominator sets).
//! * **The repetition**: something proportional to the whole function rebuilt once per construct
//!   or per block. Each copy is linear and short-lived, so the footprint never looks wrong; the
//!   wall clock does.
//!
//! So this measures the shape of the growth rather than a byte count, end to end (decode, compile,
//! and the compute pipeline), and it measures it twice: against the high-water mark, which sees the
//! first kind, and against the total bytes handed out, which sees the second. Both ratios are
//! properties of the code, not of the machine: the allocator below counts bytes the compile asked
//! for, which is the same number on any host. The engine's meters bound the work and ledger each
//! pass charges (`src/tests/engine_metered_walks.rs`); this measures what no charge covers, every
//! allocation of the whole attempt.
//!
//! The engine shipped the second kind too, and this guard found it: the emitter copied the
//! structurizer's variables, which grow with the function, into each block it emitted. At this
//! guard's size that measured 3.18x of total allocation per doubling, against 1.97x once the
//! variables were shared; at a tenth of the size it measured 2.24x, under the bound, which is why
//! the size is what it is.
//!
//! The single `#[test]` is deliberate. The counters are process-wide, so a second test running
//! concurrently in this binary would be measured into the first one's.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use metal2vulkan::device::{Device, DeviceDescription, DevicePolicy};
use metal2vulkan::library::Library;
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, FunctionDescriptor, LoweringInputs, StageSource,
};
use metal2vulkan::Budget;

/// Live bytes, the high-water mark since it was last reset, and every byte ever handed out.
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static HANDED_OUT: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
            HANDED_OUT.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let grown = unsafe { System.realloc(pointer, layout, new_size) };
        if !grown.is_null() {
            let live = LIVE
                .fetch_add(new_size, Ordering::Relaxed)
                .saturating_add(new_size)
                .saturating_sub(layout.size());
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            PEAK.fetch_max(live, Ordering::Relaxed);
            HANDED_OUT.fetch_add(new_size.saturating_sub(layout.size()), Ordering::Relaxed);
        }
        grown
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// What the allocator saw while `work` ran.
struct Cost {
    /// Peak bytes live at any point, above what was already live when `work` started.
    peak: usize,
    /// Bytes of fresh memory handed out over the whole run, whether or not they were held at once.
    handed_out: usize,
}

fn cost_of<T>(work: impl FnOnce() -> T) -> (T, Cost) {
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let handed_out_before = HANDED_OUT.load(Ordering::Relaxed);
    let value = work();
    let cost = Cost {
        peak: PEAK.load(Ordering::Relaxed).saturating_sub(before),
        handed_out: HANDED_OUT
            .load(Ordering::Relaxed)
            .saturating_sub(handed_out_before),
    };
    (value, cost)
}

/// A kernel of `groups` chained irreducible pairs: `%a` and `%b` each branch into the other, so
/// neither heads a natural loop and no shape tree can nest them.
///
/// Irreducible on purpose: it is the shape that reaches the structurizer's irreducible-region
/// handling, where a per-block table of blocks would be quadratic, and it keeps every block alive
/// through the planner. Each group loads a distinct buffer word, so the conditions cannot be folded
/// away, and each group's two operations are drawn from four by a hash of its index, so no run of
/// groups long enough to be written once as a routine repeats: the graph survives intact.
fn irreducible_chain(groups: usize) -> String {
    let mut out = String::from(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @k(ptr addrspace(1) noundef %out) {
entry:
  br label %g0
"#,
    );
    const OPS: [&str; 4] = ["add", "sub", "xor", "or"];
    for group in 0..groups {
        let word = group % 1024;
        // splitmix64's finalizer of the group's index: two operations from its low four bits.
        let mut z = (group as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let (op_a, op_b) = (OPS[(z & 3) as usize], OPS[((z >> 2) & 3) as usize]);
        out.push_str(&format!(
            "g{group}:
  %p{group} = getelementptr inbounds i32, ptr addrspace(1) %out, i64 {word}
  %v{group} = load i32, ptr addrspace(1) %p{group}, align 4
  %c{group} = icmp sgt i32 %v{group}, 0
  br i1 %c{group}, label %a{group}, label %b{group}
a{group}:
  %av{group} = {op_a} i32 %v{group}, 1
  %ac{group} = icmp sgt i32 %av{group}, 3
  br i1 %ac{group}, label %b{group}, label %g{next}
b{group}:
  %bv{group} = {op_b} i32 %v{group}, 2
  %bc{group} = icmp sgt i32 %bv{group}, 5
  br i1 %bc{group}, label %a{group}, label %g{next}
",
            next = group + 1
        ));
    }
    out.push_str(&format!(
        r#"g{groups}:
  ret void
}}

!air.kernel = !{{!0}}
!air.version = !{{!4}}
!air.language_version = !{{!5}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 2, i32 8, i32 0}}
!5 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    ));
    out
}

/// `llvm-as` of `text`. `None` when `llvm-as` is not installed (the reason is printed).
fn assemble(text: &str) -> Option<Vec<u8>> {
    common::llvm_as(text, "the generated kernel")
}

/// Decode `bitcode`, compile its kernel, and make the compute pipeline: one end-to-end attempt.
/// Returns the pipeline's word count.
fn compile(bitcode: &[u8], device: &Device) -> usize {
    let budget = Budget::default();
    let library = Library::from_air(bitcode).expect("the generated kernel decodes");
    let function = FunctionDescriptor::new(library.function_at(0).expect("the kernel"));
    let compiled = device
        .compile_function(&function, &LoweringInputs::new(), &budget)
        .expect("the generated kernel compiles");
    let descriptor = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
    device
        .compile_compute_pipeline(&descriptor, &budget)
        .expect("the generated kernel's pipeline compiles")
        .stage
        .spirv
        .len()
}

/// Groups in the smaller of the two functions. Three blocks per group plus the entry and the exit,
/// so 1000 groups is a 3002-block function and 2000 a 6002-block one: large enough that a quadratic
/// term with a small constant still separates from the linear one, small enough to compile in a
/// debug build in about a second.
const SMALL: usize = 1000;

/// The most the peak may grow for a doubled function before the growth stops being linear: linear
/// growth doubles, a quadratic term quadruples, and this sits between them.
const MAX_PEAK_GROWTH: f64 = 2.6;

/// The most the total allocation may grow for a doubled function.
///
/// A bounded peak is not a bounded compile. An analysis that allocates a table proportional to the
/// whole function, runs, and frees it holds a linear peak however many times it repeats, and
/// repeating it once per construct is how the 20-second ceiling gets broken. Total bytes handed
/// out sees that where a high-water mark cannot: it is the work, not the footprint.
const MAX_WORK_GROWTH: f64 = 2.6;

#[test]
fn compile_cost_grows_linearly_with_the_block_count() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/runtime-floor.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let description = DeviceDescription::from_khronos_json(&text, None).expect("runtime-floor");
    let device =
        Device::from_queries(description, DevicePolicy::RESIDENT_TABLES).expect("runtime-floor");
    let Some(small) = assemble(&irreducible_chain(SMALL)) else {
        return;
    };
    let Some(large) = assemble(&irreducible_chain(SMALL * 2)) else {
        return;
    };
    // Compile once before measuring: the first compile in the process pays for lazily built
    // one-off tables, and that constant would otherwise land on the smaller function.
    compile(&small, &device);

    let (small_words, small_cost) = cost_of(|| compile(&small, &device));
    let (large_words, large_cost) = cost_of(|| compile(&large, &device));
    assert!(
        large_words > small_words * 3 / 2,
        "the doubled kernel's module ({large_words} words) is not about twice the smaller one's \
         ({small_words}), so the generated graph was folded away"
    );
    assert!(
        small_cost.peak > 0 && large_cost.peak > 0,
        "the allocator measured nothing: {} and {} bytes",
        small_cost.peak,
        large_cost.peak
    );
    let peak_growth = large_cost.peak as f64 / small_cost.peak as f64;
    let work_growth = large_cost.handed_out as f64 / small_cost.handed_out as f64;
    eprintln!(
        "peak {} -> {} bytes ({peak_growth:.2}x); handed out {} -> {} bytes ({work_growth:.2}x); \
         {small_words} -> {large_words} words",
        small_cost.peak, large_cost.peak, small_cost.handed_out, large_cost.handed_out
    );
    assert!(
        peak_growth <= MAX_PEAK_GROWTH,
        "doubling the function multiplied peak compile memory by {peak_growth:.2} ({} bytes at \
         {SMALL} groups, {} bytes at {}); some analysis is quadratic in the block count, which is \
         how the 500 MiB per-translation budget gets broken",
        small_cost.peak,
        large_cost.peak,
        SMALL * 2
    );
    assert!(
        work_growth <= MAX_WORK_GROWTH,
        "doubling the function multiplied total compile allocation by {work_growth:.2} ({} bytes \
         at {SMALL} groups, {} bytes at {}); something the size of the whole function is rebuilt \
         once per block or per construct, which is how the 20-second per-translation ceiling gets \
         broken",
        small_cost.handed_out,
        large_cost.handed_out,
        SMALL * 2
    );
}
