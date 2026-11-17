//! A constant blob is held once.
//!
//! A closed constant past the function's typed room goes to the constant blob, whose bytes the
//! memory plan charges to the ledger. Selection and the artifact share those bytes; previously
//! selection copied them, uncharged, while the plan's were still live. A struct DAG constant `d`
//! levels deep is `2^(d+2)` bytes for `d + 1` types, so that copy alone took a 26-level kernel
//! past 500 MiB resident while the ledger saw half of it.
//!
//! The allocator below counts the bytes the translation asked for, so the ledger's share of them
//! is a property of the code, not of the host. It is the global allocator of the crate's whole
//! test build, so it counts only inside `held_during`, and every other test allocates at the cost
//! of one thread-local read. Its counters are the measuring thread's own: the translation runs on
//! its caller's thread, so a measurement holds whether its test runs alone in its process (as
//! nextest runs every test) or beside others on libtest's threads (`cargo test`), whose
//! allocations it does not see. Other tests count allocations with it ([`allocations_during`]).
//!
//! `llvm-as` assembles the kernel; without it the test skips visibly.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use crate::budget::{Budget, Meter, LEDGER_BYTES, SPIRV_WORDS, WORK_UNITS};
use crate::cache::PipelineArtifact;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, Feature, Limit};
use crate::library::{FunctionConstantValues, Library};
use crate::lower::select::LowerInputs;
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::testing::{llvm_assemble, profile, AssembleError, Scratch};
use crate::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};

/// One thread's counts, while `held_during` measures on it.
#[derive(Clone, Copy)]
struct Counts {
    /// Whether the allocator counts this thread's allocations.
    counting: bool,
    /// The bytes live since `held_during` armed it, and their high-water mark.
    live: isize,
    peak: isize,
    /// The allocations and reallocations made since `held_during` armed it.
    allocations: u64,
}

thread_local! {
    /// The calling thread's counts: a constant with no destructor, so the allocator reads it
    /// without allocating, at any point of the thread's life.
    static COUNTS: Cell<Counts> = const {
        Cell::new(Counts {
            counting: false,
            live: 0,
            peak: 0,
            allocations: 0,
        })
    };
}

/// `grown` bytes allocated and `freed` released on the calling thread, while the allocator counts
/// there: an allocation or reallocation where `grown` is not 0.
fn count(grown: usize, freed: usize) {
    let _ = COUNTS.try_with(|counts| {
        let mut now = counts.get();
        if now.counting {
            now.allocations += u64::from(grown != 0);
            now.live += grown as isize - freed as isize;
            now.peak = now.peak.max(now.live);
            counts.set(now);
        }
    });
}

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            count(layout.size(), 0);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            count(layout.size(), 0);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        count(0, layout.size());
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let grown = unsafe { System.realloc(pointer, layout, new_size) };
        if !grown.is_null() {
            count(new_size, layout.size());
        }
        grown
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// `work`'s result, and the most bytes it held at once on the calling thread beyond what was live
/// when it started; and the allocations and reallocations it made there.
fn measured<T>(work: impl FnOnce() -> T) -> (T, Counts) {
    let armed = Counts {
        counting: true,
        live: 0,
        peak: 0,
        allocations: 0,
    };
    COUNTS.with(|counts| counts.set(armed));
    let value = work();
    let counts = COUNTS.with(|counts| {
        let mut now = counts.get();
        now.counting = false;
        counts.set(now);
        now
    });
    (value, counts)
}

/// `work`'s result, and the most bytes it held at once beyond what was live when it started (the
/// calling thread's, on which the translation runs).
pub(crate) fn held_during<T>(work: impl FnOnce() -> T) -> (T, u64) {
    let (value, counts) = measured(work);
    (value, counts.peak.max(0) as u64)
}

/// `work`'s result, and the allocations and reallocations it made while it ran (the calling
/// thread's).
pub(super) fn allocations_during<T>(work: impl FnOnce() -> T) -> (T, u64) {
    let (value, counts) = measured(work);
    (value, counts.allocations)
}

const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

/// Where a kernel's escaping constant is, if it has one ([`blob_kernel`]).
#[derive(Clone, Copy, Debug)]
enum Escaping {
    /// None: both words are read from the DAG.
    No,
    /// Word 0 is read from the escaping constant before word 1 is read from the DAG.
    ReadFirst,
    /// Word 1 is read from the DAG before word 0 is read from the escaping constant.
    ReadLast,
}

/// A kernel reading words 0 and 1 of `@g`, a closed constant struct DAG `depth` levels deep
/// (`%e1 = { i32, i32 }`, `%e{k} = { %e{k-1}, %e{k-1} }`), all zero, into `out[2t]` and
/// `out[2t + 1]`. Past 14 levels its unshared type size is past the function's typed room, so
/// `@g` is in the constant blob, `2^(depth+2)` bytes. With an escaping constant, word 0 is instead
/// byte 0 of `@a = { i32 5, ptr @s }`, declared first: an `i8` load holds it untyped, so it is in
/// the blob too, where its pointer has no bytes (`ConstantGlobalEscaping`). Pruning numbers the
/// globals in the order its walk reaches them, so which of `@a` and `@g` is placed first follows
/// the order the body reads them.
fn blob_kernel(depth: u32, escaping: Escaping) -> String {
    let mut types = String::from("%e1 = type { i32, i32 }\n");
    for k in 2..=depth {
        types += &format!("%e{k} = type {{ %e{p}, %e{p} }}\n", p = k - 1);
    }
    let first = match escaping {
        Escaping::No => "",
        Escaping::ReadFirst | Escaping::ReadLast => {
            "@s = internal addrspace(2) constant i32 1, align 4\n@a = internal addrspace(2) constant { i32, ptr addrspace(2) } { i32 5, ptr addrspace(2) @s }, align 8\n"
        }
    };
    let word0 = match escaping {
        Escaping::No => "  %v0 = load i32, ptr addrspace(2) @g, align 4",
        Escaping::ReadFirst | Escaping::ReadLast => {
            "  %b = load i8, ptr addrspace(2) @a, align 8\n  %v0 = zext i8 %b to i32"
        }
    };
    let word1 = "  %q = getelementptr inbounds i8, ptr addrspace(2) @g, i32 4\n  %v1 = load i32, ptr addrspace(2) %q, align 4";
    let words = match escaping {
        Escaping::No | Escaping::ReadFirst => format!("{word0}\n{word1}"),
        Escaping::ReadLast => format!("{word1}\n{word0}"),
    };
    format!(
        r#"target datalayout = "{DATALAYOUT}"
target triple = "air64_v28-apple-macosx26.0.0"

{types}{first}@g = internal addrspace(2) constant %e{depth} zeroinitializer, align 4

define void @blob(ptr addrspace(1) %out, i32 %t) {{
{words}
  %i0 = shl i32 %t, 1
  %o0 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i0
  store i32 %v0, ptr addrspace(1) %o0, align 4
  %i1 = add i32 %i0, 1
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i1
  store i32 %v1, ptr addrspace(1) %o1, align 4
  ret void
}}

!air.kernel = !{{!0}}
!0 = !{{ptr @blob, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}}
"#
    )
}

fn assemble(depth: u32, escaping: Escaping) -> Option<Vec<u8>> {
    let scratch = Scratch::new(&format!("m2v-blob-footprint-{depth}-{escaping:?}"));
    let dir = scratch.path();
    let ll = dir.join("k.ll");
    std::fs::write(&ll, blob_kernel(depth, escaping)).expect("write");
    let r = llvm_assemble(&ll);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => {
            panic!("llvm-as rejected the {depth}-level kernel: {why}")
        }
    }
}

/// Compile `bc` for `lavapipe` with the default caps: the result, the ledger peak, and the most
/// bytes the compile held at once.
fn compile(bc: &[u8]) -> (Result<(), Error>, Option<u64>, u64, u64) {
    let p = profile("lavapipe", DevicePolicy::default()).expect("profile");
    let mut m = Meter::with_caps(&Budget::default(), WORK_UNITS, LEDGER_BYTES);
    let (compiled, held) = held_during(|| {
        compile_bitcode(
            bc,
            EntryRef::Name("blob"),
            &FunctionConstantValues::defaults(),
            &p,
            &LowerInputs::new(),
            &mut m,
            SPIRV_WORDS,
        )
    });
    let blob = compiled
        .as_ref()
        .ok()
        .and_then(|a| a.blob.as_ref())
        .map(|b| b.len() as u64);
    (compiled.map(|_| ()), blob, m.ledger_peak(), held)
}

/// A writer that keeps only the count of the bytes written to it.
struct Counted(u64);

impl std::io::Write for Counted {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 += bytes.len() as u64;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The compute pipeline of `bc`'s kernel on `lavapipe`, written to a counting writer: the bytes
/// written, and the most bytes the writing held at once beside the pipeline.
fn written(bc: &[u8]) -> (u64, u64, u64) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/lavapipe.json");
    let text = std::fs::read_to_string(&path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let device = Device::new(desc, DevicePolicy::default()).expect("device");
    let f = Library::from_air(bc)
        .expect("library")
        .function("blob")
        .expect("function");
    let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    let pipeline = device
        .compile_compute_pipeline(&d, &Budget::standard())
        .expect("pipeline");
    let blob = pipeline
        .stage
        .constant_blob
        .as_ref()
        .map_or(0, |b| b.len() as u64);
    let mut out = Counted(0);
    let (written, held) = held_during(|| pipeline.write_to(&mut out));
    written.expect("write");
    assert_eq!(
        out.0,
        pipeline.to_bytes().len() as u64,
        "the artifact's bytes"
    );
    (blob, out.0, held)
}

/// At 26 levels the blob is `2^28` bytes (256 MiB). The kernel translates for `lavapipe` with the
/// default caps, the artifact's blob is that many bytes, and the ledger peak is at least seven
/// tenths of the bytes the translation held at once (validation's `LEDGER_FLOOR`): the blob is held
/// once, and charged. Copying it in selection, as before, leaves the ledger near half. At 27 levels
/// the blob alone passes the ledger cap (`LedgerBytes`), with no blob allocated.
///
/// With an escaping constant (`blob_kernel`'s `escaping`), placed before the DAG or after it, both
/// depths are refused as `ConstantGlobalEscaping`, as when each blob-form constant was written as
/// it was placed: placing checks what writing every constant would refuse before it charges any
/// constant's bytes, so the refusal comes before the DAG's bytes are charged or allocated.
/// Checking only when writing, after every constant is placed, made 27 levels `LedgerBytes` and
/// 26 levels allocate the whole blob first; checking each constant as it was placed, after the
/// bytes of those placed before it, charged the whole DAG first when it was placed first.
///
/// A pipeline holding a 20-level kernel's blob (4 MiB) writes its artifact
/// (`PipelineArtifact::write_to`, the validation worker's report) holding under a quarter of the
/// blob beside the pipeline: the encoding goes out a chunk at a time, where `to_bytes` holds the
/// body and the frame, twice the blob.
#[test]
fn a_constant_blob_is_held_once_and_charged() {
    if let Some(bc) = assemble(20, Escaping::No) {
        let (blob, bytes, held) = written(&bc);
        eprintln!("20 levels: wrote {bytes} bytes holding {held} bytes for a {blob}-byte blob");
        assert_eq!(blob, 4 << 20);
        assert!(bytes > blob, "{bytes} bytes for a {blob}-byte blob");
        assert!(
            held < blob / 4,
            "{held} bytes held writing a {blob}-byte blob"
        );
    }
    for depth in [26u32, 27] {
        for escaping in [Escaping::No, Escaping::ReadFirst, Escaping::ReadLast] {
            footprint(depth, escaping);
        }
    }
}

/// [`a_constant_blob_is_held_once_and_charged`]'s case at `depth` levels.
fn footprint(depth: u32, escaping: Escaping) {
    let Some(bc) = assemble(depth, escaping) else {
        return;
    };
    let (compiled, bytes, ledger, held) = compile(&bc);
    let blob = 4u64 << depth;
    let what = format!("{depth} levels, escaping {escaping:?}");
    eprintln!("{what}: ledger peak {ledger} bytes, {held} bytes held at once, blob {blob}");
    match (depth, escaping) {
        (26, Escaping::No) => {
            compiled.expect("translates");
            assert_eq!(bytes, Some(blob), "{what}");
            assert!(
                ledger * 10 >= held * 7,
                "{what}: ledger peak {ledger} bytes against {held} bytes held at once"
            );
            assert!(
                held < blob + blob / 4,
                "{what}: {held} bytes held at once for a {blob}-byte blob"
            );
        }
        (_, Escaping::No) => {
            assert_eq!(
                compiled,
                Err(Error::LimitExceeded(Limit::LedgerBytes)),
                "{what}"
            );
            assert!(held < blob / 4, "{what}: {held} bytes held at once");
        }
        (_, Escaping::ReadFirst | Escaping::ReadLast) => {
            match compiled {
                Err(Error::Unsupported(u)) => {
                    assert_eq!(u.feature, Feature::ConstantGlobalEscaping, "{what}")
                }
                other => panic!("{what}: expected the escaping refusal, got {other:?}"),
            }
            assert!(ledger < 1 << 20, "{what}: ledger peak {ledger} bytes");
            assert!(held < 1 << 20, "{what}: {held} bytes held at once");
        }
    }
}
