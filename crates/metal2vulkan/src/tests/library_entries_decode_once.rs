//! What the entries of **one** library cost to compile, counted rather than timed.
//!
//! # Why counts and not times
//!
//! Decoding a module reads its bitcode and nothing else: which entry of it is being compiled, and
//! with which constants, linked functions and call-stack depth, reaches no part of the decode. So
//! a library of six entries ought to decode its module once and not six times, and whether it
//! does is a count the build settles exactly, where a wall clock on a shared host settles
//! nothing. [`StageEntries::decodes`] counts the decodes read of bitcode and the asks that copied
//! one already read, and [`Meter::work`] counts deterministic work units -- the same input on the
//! same profile spends the same, whatever the load. Both are read here; no wall clock is.
//!
//! # What they hold
//!
//! - **One library, one decode.** Six entries of one module compiled through one
//!   [`StageEntries`], as the runtime's device gives one object to every creation of its
//!   pipelines, read **one** decode between them
//!   ([`the_entries_of_one_library_decode_its_module_once`]). An object each, which is what a
//!   consumer that shares nothing does, reads six.
//! - **A plain answer and the entry share it too.** A render creation asks whether its vertex
//!   stage returns a value before it compiles it, and both the question and the compile need the
//!   module's decode: through one object they read it once
//!   ([`a_plain_answer_and_the_stages_entry_decode_the_module_once`]).
//! - **A partition check and the compile share one order.** A compute creation asks whether its
//!   kernel observes the threadgroup partition before it compiles it, and both read the kernel's
//!   specialized entry ordered for the device: through one object the entry is ordered once, and
//!   the compile reads the check's
//!   ([`a_partition_check_and_the_kernels_compile_order_its_entry_once`]).
//! - **Not across libraries.** Two libraries of different bytes read a decode each, however the
//!   object is shared: the module's own AIR hash keys the decode, so nothing of one library is
//!   taken for another ([`two_libraries_decode_their_modules_apart`]).
//! - **The same module, word for word.** Every entry compiled from a shared decode is the module
//!   its own unshared compile makes, word for word, and its artifact is byte for byte the same.
//! - **The same charge, unit for unit.** Every entry compiled from a shared decode is charged the
//!   work units its own unshared compile is charged, the decode it copied included: the object
//!   spares the decode's time and never its budget, so a refusal at the work cap is the same
//!   through either.
//! - **The hash is the library's, not a second one.** A compile records its function's AIR hash
//!   from the library's table rather than hashing the bitcode again, which is exact because a
//!   module's hash is the hash of the bytes every entry of it shares, and a metallib function's
//!   `HASH` is verified against its bitcode when the container is parsed
//!   ([`a_librarys_stated_air_hash_is_the_hash_of_the_bitcode_a_compile_reads`],
//!   [`a_metallib_whose_stated_hash_is_not_its_bitcodes_is_refused`]).
//! - **Bounded.** Filled past its bounds, it keeps the most recently used entries and no more,
//!   and what it holds stays within its byte bounds
//!   ([`an_object_filled_past_its_bounds_keeps_only_the_most_recent`]).
//! - **What holding it costs.** The headline part reports what one decode charges the ledger and
//!   holds four of the largest committed module's to the object's byte bound, so a bound that
//!   could not hold a library's one decode would fail here.

use std::path::{Path, PathBuf};

use crate::air::container;
use crate::budget::{Budget, Meter};
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::library::{Library, LibraryType};
use crate::lower::select::LowerInputs;
use crate::pipeline::compile::{
    compile_stage, compile_stage_from, kernel_observes_partition, InputsFrom, StageEntries,
};
use crate::pipeline::testing::{fixtures_dir, llvm_assemble, AssembleError};
use crate::pipeline::FunctionDescriptor;

/// The multi-entry fixture: one module of six kernels, which is the shape a library of several
/// entries has (`reflection/encoder_types`, the fixture a translation split was recorded over).
const MANY: &str = "reflection/encoder_types";

/// A second fixture, of other bytes: what must not take the first's decode.
const OTHER: &str = "reflection/textures";

fn device() -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    desc.features.insert(VkFeature::RobustBufferAccess);
    Device::new(desc, DevicePolicy::default()).expect("device")
}

fn fixture(name: &str) -> Option<Vec<u8>> {
    let path: PathBuf = fixtures_dir().join(format!("{name}.ll"));
    match llvm_assemble(&path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}.ll: {why}"),
    }
}

/// Every stage of `library`, in table order: a library's entries as a consumer asks for them one
/// after another (`newFunctionWithName:`, then a pipeline a function).
fn stages(library: &Library) -> Vec<FunctionDescriptor> {
    use crate::library::FunctionStage::{Fragment, Kernel, Mesh, Object, Vertex};
    let names: Vec<String> = library
        .functions()
        .filter(|i| matches!(i.stage, Kernel | Vertex | Fragment | Mesh | Object))
        .map(|i| i.name.to_owned())
        .collect();
    names
        .iter()
        .map(|name| FunctionDescriptor::new(library.function(name).expect("function")))
        .collect()
}

/// `fd` compiled on its own, sharing nothing, with the work units it cost: what every shared
/// compile below is held to, word for word.
fn alone(fd: &FunctionDescriptor, device: &Device) -> (Vec<u32>, Vec<u8>, u64) {
    let inputs = LowerInputs::new();
    let mut meter = Meter::new(&Budget::standard());
    let compiled = compile_stage(
        &fd.function,
        &fd.constants,
        (&fd.linked, &fd.libraries),
        device.profile(),
        InputsFrom::Given(&inputs),
        &mut meter,
    )
    .expect("the entry compiles");
    (
        compiled.artifact().module.words.clone(),
        compiled.to_bytes(),
        meter.work(),
    )
}

/// `fd` compiled reading `entries`, held to `expected` word for word and byte for byte, with the
/// work units it cost.
fn from_entries(
    fd: &FunctionDescriptor,
    device: &Device,
    entries: &StageEntries,
    expected: &(Vec<u32>, Vec<u8>, u64),
) -> u64 {
    let inputs = LowerInputs::new();
    let mut meter = Meter::new(&Budget::standard());
    let compiled = compile_stage_from(
        fd,
        device.profile(),
        InputsFrom::Given(&inputs),
        entries,
        &mut meter,
    )
    .expect("the entry compiles");
    assert_eq!(
        &compiled.artifact().module.words,
        &expected.0,
        "an entry compiled from a shared decode must be the module its own compile makes, word \
         for word ({})",
        fd.function.info().name
    );
    assert_eq!(
        compiled.to_bytes(),
        expected.1,
        "an entry compiled from a shared decode must be the artifact its own compile makes, byte \
         for byte ({})",
        fd.function.info().name
    );
    meter.work()
}

/// Six entries of one module, compiled one after another through one object as the runtime's
/// device compiles the stages of its pipelines, read **one** decode between them; an object each
/// reads six. Counted in both directions, so neither reading is vacuous, and every entry is held
/// to its own unshared compile word for word and byte for byte, so the sharing changes no output.
#[test]
fn the_entries_of_one_library_decode_its_module_once() {
    let Some(bitcode) = fixture(MANY) else { return };
    let device = device();
    let library = Library::from_air(&bitcode).expect("library");
    let stages = stages(&library);
    let n = u64::try_from(stages.len()).expect("the entries");
    assert!(
        n >= 6,
        "{MANY} must hold the six entries this counts over, not {n}"
    );
    let alone: Vec<(Vec<u32>, Vec<u8>, u64)> = stages.iter().map(|fd| alone(fd, &device)).collect();

    // One object every entry of the library reads, as the device's pipeline memo gives one object
    // to every creation.
    let shared = StageEntries::new();
    let shared_work: Vec<u64> = stages
        .iter()
        .zip(&alone)
        .map(|(fd, expected)| from_entries(fd, &device, &shared, expected))
        .collect();
    assert_eq!(
        shared.decodes(),
        (1, n - 1),
        "{n} entries of one module must read its decode once and copy it the other {} times",
        n - 1
    );
    assert_eq!(
        shared.counts(),
        (n, 0),
        "each entry is its own specialized entry: {n} made and none found"
    );

    // An object each, which is what a consumer that shares nothing between its functions does.
    let mut apart_work = Vec::new();
    for (fd, expected) in stages.iter().zip(&alone) {
        let entries = StageEntries::new();
        apart_work.push(from_entries(fd, &device, &entries, expected));
        assert_eq!(
            entries.decodes(),
            (1, 0),
            "an object of its own reads the decode for its one entry and copies nothing ({})",
            fd.function.info().name
        );
    }

    // A copied decode is charged what reading it charged: every entry costs the work units of
    // its own unshared compile, through a shared decode or one of its own.
    for (i, fd) in stages.iter().enumerate() {
        let (shared, apart, own) = (shared_work[i], apart_work[i], alone[i].2);
        assert_eq!(
            (shared, apart),
            (own, own),
            "entry {} must cost the work units of its own compile, {own}, reading a shared \
             decode ({shared}) and reading its own ({apart})",
            fd.function.info().name
        );
    }
    eprintln!("work units (shared, apart): {shared_work:?} {apart_work:?}");

    // What holding the decode costs: what decoding this module charges the compile's ledger,
    // which is what the object's byte bound counts. Four of them must fit that bound, or the
    // bound would evict the only decode a library's entries share.
    let mut meter = Meter::new(&Budget::standard());
    let before = meter.ledger();
    let decoded = crate::air::decode(&bitcode, &mut meter).expect("the module decodes");
    let charged = meter.ledger() - before;
    drop(decoded);
    eprintln!(
        "one decode of {MANY} ({} bytes of AIR) charges {charged} ledger bytes",
        bitcode.len()
    );
    let bound = crate::pipeline::compile::KEPT_DECODE_BYTES;
    assert!(
        charged * 4 <= bound,
        "four decodes of {MANY} charge {} bytes, past the {bound} the object keeps",
        charged * 4
    );

    // The same for the largest module the committed fixtures hold, which is what the bound has to
    // hold four of for a library of its size.
    for name in ["engine/hfb_filter_bank", "engine/tns_mm_flags"] {
        let Some(bitcode) = fixture(name) else { return };
        let mut meter = Meter::new(&Budget::standard());
        let before = meter.ledger();
        let decoded = crate::air::decode(&bitcode, &mut meter).expect("the module decodes");
        let charged = meter.ledger() - before;
        drop(decoded);
        eprintln!(
            "one decode of {name} ({} bytes of AIR) charges {charged} ledger bytes",
            bitcode.len()
        );
        assert!(
            charged * 4 <= bound,
            "four decodes of {name} charge {} bytes, past the {bound} the object keeps",
            charged * 4
        );
    }
}

/// Two libraries of different bytes read a decode each, however the object is shared: the
/// module's own AIR hash keys the decode, so a fix that collapsed decodes across unrelated input
/// would fail here.
#[test]
fn two_libraries_decode_their_modules_apart() {
    let (Some(many), Some(other)) = (fixture(MANY), fixture(OTHER)) else {
        return;
    };
    let device = device();
    let first = Library::from_air(&many).expect("library");
    let second = Library::from_air(&other).expect("library");
    let one = stages(&first).swap_remove(0);
    let two = stages(&second).swap_remove(0);
    assert_ne!(
        one.function.info().air_hash,
        two.function.info().air_hash,
        "the two fixtures must be different modules"
    );
    let expected = [alone(&one, &device), alone(&two, &device)];

    let shared = StageEntries::new();
    for (fd, expected) in [(&one, &expected[0]), (&two, &expected[1])] {
        from_entries(fd, &device, &shared, expected);
    }
    assert_eq!(
        shared.decodes(),
        (2, 0),
        "two libraries of different bytes must read a decode each and copy neither"
    );

    // Each of them asked for again: now each copies its own decode, and neither has taken the
    // other's.
    for fd in [&one, &two] {
        let mut inputs = LowerInputs::new();
        // A depth the first ask did not use, so the entry is made again and needs the decode.
        inputs.max_call_stack_depth = Some(3);
        let mut meter = Meter::new(&Budget::standard());
        compile_stage_from(
            fd,
            device.profile(),
            InputsFrom::Given(&inputs),
            &shared,
            &mut meter,
        )
        .expect("the entry compiles");
    }
    assert_eq!(
        shared.decodes(),
        (2, 2),
        "a second ask over each library must copy that library's own decode"
    );
}

/// A render creation asks a plain question of its vertex stage before it compiles it
/// (`RenderPipelineDescriptor::vertex_returns_value`), and both the question and the compile need
/// the module's decode. Through one object they read it once; the question's answer and the
/// stage's entry are still each made once, so what the sharing removed is a decode and nothing
/// else.
#[test]
fn a_plain_answer_and_the_stages_entry_decode_the_module_once() {
    let Some(bitcode) = fixture(MANY) else { return };
    let device = device();
    let library = Library::from_air(&bitcode).expect("library");
    let fd = stages(&library).swap_remove(0);
    let expected = alone(&fd, &device);

    let entries = StageEntries::new();
    let mut meter = Meter::new(&Budget::standard());
    crate::pipeline::compile::stage_returns_value(&fd, &entries, &mut meter)
        .expect("the plain answer");
    assert_eq!(
        entries.decodes(),
        (1, 0),
        "the plain answer reads the module's decode"
    );
    from_entries(&fd, &device, &entries, &expected);
    assert_eq!(
        entries.decodes(),
        (1, 1),
        "the stage's compile copies the decode the plain answer read, rather than reading it again"
    );
    assert_eq!(
        entries.counts(),
        (2, 0),
        "the answer and the entry are each made once: nothing else was shared"
    );
}

/// Each kernel of one library, checked for whether it observes its threadgroup partition and then
/// compiled, as a compute creation does, through one object: the check orders the kernel's
/// specialized entry for the device and the compile finds that order rather than ordering again.
/// Each compile is held to the kernel's own unshared compile word for word, byte for byte and
/// unit for unit, so the kept order changes no output and spares no charge.
#[test]
fn a_partition_check_and_the_kernels_compile_order_its_entry_once() {
    use crate::library::FunctionStage::Kernel;
    let Some(bitcode) = fixture(MANY) else { return };
    let device = device();
    let library = Library::from_air(&bitcode).expect("library");
    let kernels: Vec<FunctionDescriptor> = stages(&library)
        .into_iter()
        .filter(|fd| matches!(fd.function.info().stage, Kernel))
        .collect();
    assert!(
        kernels.len() >= 2,
        "{MANY} must hold the kernels this orders, not {}",
        kernels.len()
    );
    for fd in &kernels {
        let name = fd.function.info().name;
        let expected = alone(fd, &device);
        let entries = StageEntries::new();
        let inputs = LowerInputs::new();
        let mut meter = Meter::new(&Budget::standard());
        kernel_observes_partition(
            fd,
            device.profile(),
            InputsFrom::Given(&inputs),
            &entries,
            &mut meter,
        )
        .expect("the partition check");
        assert_eq!(
            entries.orders(),
            (1, 0),
            "the partition check orders the kernel's entry ({name})"
        );
        let work = from_entries(fd, &device, &entries, &expected);
        assert_eq!(
            entries.orders(),
            (1, 1),
            "the compile finds the entry the check ordered rather than ordering it again ({name})"
        );
        assert_eq!(
            entries.counts(),
            (1, 1),
            "the check made the specialized entry and the compile found it ({name})"
        );
        assert_eq!(
            work, expected.2,
            "the compile is charged the work units of its own unshared compile, the order it found \
             included ({name})"
        );
    }
}

/// The AIR hash a library states of a function is the SHA-256 of the bitcode a compile of it
/// reads: for a module, the whole module every entry of it shares. A compile records that hash
/// rather than hashing the module again, so this is what makes the two the same value.
#[test]
fn a_librarys_stated_air_hash_is_the_hash_of_the_bitcode_a_compile_reads() {
    let mut seen = 0;
    for name in [
        MANY,
        OTHER,
        "reflection/encoder_arrays",
        "reflection/argument_buffers",
    ] {
        let Some(bitcode) = fixture(name) else { return };
        let library = Library::from_air(&bitcode).expect("library");
        let whole = container::sha256(&bitcode);
        assert!(!library.is_empty(), "{name} declares no function");
        for info in library.functions() {
            assert_eq!(
                info.air_hash, whole,
                "{name}: the hash stated of {} must be the module's own",
                info.name
            );
            seen += 1;
        }
    }
    assert!(seen >= 6, "only {seen} functions were checked");
}

/// A metallib whose `HASH` is not the SHA-256 of the function's bitcode does not parse, so the
/// hash a parsed metallib states of a function is that function's bitcode hash -- which is what a
/// compile takes in place of hashing the bitcode itself. Checked in both directions: the same
/// container with the bitcode left alone parses and states that hash.
#[test]
fn a_metallib_whose_stated_hash_is_not_its_bitcodes_is_refused() {
    use metal2vulkan_devtools::metallib::{metallib, wrap, MtlbFunction};
    let Some(bitcode) = fixture(OTHER) else {
        return;
    };
    let wrapped = wrap(&bitcode);
    let bytes = metallib(&[MtlbFunction {
        name: "reflection_textures",
        type_code: 2,
        bitcode: wrapped.clone(),
        constants: Vec::new(),
        public: Vec::new(),
        tags: Vec::new(),
    }]);
    let library = Library::from_metallib(&bytes).expect("the metallib parses");
    assert_eq!(library.library_type(), LibraryType::Executable);
    let stated = library.functions().next().expect("one function").air_hash;
    assert_eq!(
        stated,
        container::sha256(&wrapped),
        "the hash a parsed metallib states of its function is its bitcode's"
    );

    // The same container with one byte of the function's bitcode flipped, the `HASH` tag left as
    // written: the bytes and the stated hash now disagree.
    let at = bytes
        .windows(wrapped.len())
        .position(|w| w == wrapped.as_slice())
        .expect("the wrapped bitcode lies in the container");
    let mut changed = bytes.clone();
    let last = at + wrapped.len() - 1;
    changed[last] ^= 1;
    assert_ne!(changed, bytes, "a byte of the bitcode must have moved");
    let refused = Library::from_metallib(&changed);
    assert!(
        refused.is_err(),
        "a metallib whose HASH is not its bitcode's must be refused, not parsed"
    );
}

/// Filled past its bounds, the object keeps the most recently used and no more: one entry of
/// [`MANY`] asked at forty call-stack depths, each depth its own specialized entry, leaves at most
/// `KEPT_ENTRIES` entries and one decode within the byte bounds ([`StageEntries::held`]); of the
/// depths, the latest is found again, while the first, gone past the bound, is made again.
#[test]
fn an_object_filled_past_its_bounds_keeps_only_the_most_recent() {
    use crate::pipeline::compile::{KEPT_DECODE_BYTES, KEPT_ENTRIES, KEPT_ENTRY_BYTES};
    let Some(bitcode) = fixture(MANY) else { return };
    let device = device();
    let library = Library::from_air(&bitcode).expect("library");
    let fd = stages(&library).into_iter().next().expect("an entry");
    let entries = StageEntries::new();
    let ask = |depth: u32| {
        let mut inputs = LowerInputs::new();
        inputs.max_call_stack_depth = Some(depth);
        let mut meter = Meter::new(&Budget::standard());
        let compiled = compile_stage_from(
            &fd,
            device.profile(),
            InputsFrom::Given(&inputs),
            &entries,
            &mut meter,
        );
        compiled.expect("the entry compiles at every depth");
    };
    for depth in 1..=40 {
        ask(depth);
    }
    let (held, decodes, bytes) = entries.held();
    let kept = u64::try_from(KEPT_ENTRIES).expect("the bound");
    assert!(
        (1..=kept).contains(&held),
        "40 entries asked must leave between 1 and {kept} held, not {held}"
    );
    assert_eq!(decodes, 1, "one module, one decode held");
    assert!(
        bytes <= KEPT_ENTRY_BYTES + KEPT_DECODE_BYTES,
        "the held bytes, {bytes}, must stay within the bounds"
    );
    assert_eq!(
        entries.counts(),
        (40, 0),
        "forty depths, forty entries made"
    );
    ask(40);
    assert_eq!(entries.counts(), (40, 1), "the latest depth is found again");
    ask(1);
    assert_eq!(
        entries.counts(),
        (41, 1),
        "the first depth, gone past the bound, is made again"
    );
    assert!(entries.held().0 <= kept, "asking again keeps the bound");
}
