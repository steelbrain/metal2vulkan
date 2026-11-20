//! `moltenvk_msl_args` on hand-built modules: the stage's SPIRV-Cross options and the Metal math
//! mode and invariance MoltenVK derives; the assemblers' bitcode, and so the meter's charges,
//! follow a text and not the path it is read from.

use super::*;
use crate::spv::{BuiltIn, Decoration, ExecutionMode, ExecutionModel, FPFastMathMode, Op};

fn inst(op: Op, operands: &[u32]) -> Vec<u32> {
    let mut words = vec![((operands.len() as u32 + 1) << 16) | op as u32];
    words.extend_from_slice(operands);
    words
}

fn string(text: &str) -> Vec<u32> {
    let mut bytes = text.as_bytes().to_vec();
    bytes.push(0);
    bytes.resize(bytes.len().div_ceil(4) * 4, 0);
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// A module of `body` after the header (only what the argument derivation reads).
fn module(body: &[Vec<u32>]) -> Vec<u32> {
    let mut words = vec![0x0723_0203, 0x0001_0500, 0, 100, 0];
    words.extend(body.iter().flatten());
    words
}

const MAIN: u32 = 1;

fn entry(model: ExecutionModel) -> Vec<u32> {
    let mut operands = vec![model as u32, MAIN];
    operands.extend(string("main"));
    inst(Op::EntryPoint, &operands)
}

fn args(body: &[Vec<u32>]) -> MoltenVkMslArgs {
    moltenvk_msl_args(&module(body)).expect("MoltenVK arguments")
}

fn has(args: &[String], run: &[&str]) -> bool {
    args.windows(run.len())
        .any(|w| w.iter().map(String::as_str).eq(run.iter().copied()))
}

fn math(a: &MoltenVkMslArgs) -> (&str, &str) {
    let flag = |prefix: &str| {
        a.metal
            .iter()
            .find_map(|arg| arg.strip_prefix(prefix))
            .expect("math flag")
    };
    (
        flag("-fmetal-math-mode="),
        flag("-fmetal-math-fp32-functions="),
    )
}

#[test]
fn a_compute_module_without_float_controls_compiles_under_fast_math() {
    let a = args(&[entry(ExecutionModel::GLCompute)]);
    for run in [
        &["--msl", "--msl-version", "40000"][..],
        &["--msl-argument-buffers", "--msl-argument-buffer-tier", "1"],
        &["--msl-texture-buffer-native", "--msl-texture-1d-as-2d"],
        &["--entry", "main", "--stage", "comp"],
        &["--msl-fixed-subgroup-size", "32"],
    ] {
        assert!(has(&a.spirv_cross, run), "{run:?} in {:?}", a.spirv_cross);
    }
    assert!(!has(&a.spirv_cross, &["--flip-vert-y"]));
    assert_eq!(math(&a), ("fast", "fast"));
    assert!(has(&a.metal, &["-sdk", "macosx", "metal", "-std=metal4.0"]));
    assert_eq!(a.metal.last().map(String::as_str), Some("-c"));
    assert!(!a.metal.iter().any(|arg| arg == "-fpreserve-invariance"));
}

#[test]
fn float_controls_select_the_math_mode_moltenvk_derives() {
    let mode = |extra: Vec<Vec<u32>>| {
        let mut body = vec![entry(ExecutionModel::GLCompute)];
        body.extend(extra);
        let a = args(&body);
        let (mode, functions) = math(&a);
        (mode.to_owned(), functions.to_owned())
    };
    let pair = |m: &str, f: &str| (m.to_owned(), f.to_owned());
    // SignedZeroInfNanPreserve drops NSZ, NotInf, and NotNaN: not even relaxed.
    assert_eq!(
        mode(vec![inst(
            Op::ExecutionMode,
            &[MAIN, ExecutionMode::SignedZeroInfNanPreserve as u32, 32],
        )]),
        pair("safe", "precise")
    );
    // ContractionOff drops AllowContract, one of the relaxed four.
    assert_eq!(
        mode(vec![inst(
            Op::ExecutionMode,
            &[MAIN, ExecutionMode::ContractionOff as u32],
        )]),
        pair("safe", "precise")
    );
    // A decoration that keeps the relaxed four but drops NotNaN.
    let relaxed = (FPFastMathMode::NSZ
        | FPFastMathMode::ALLOW_RECIP
        | FPFastMathMode::ALLOW_REASSOC
        | FPFastMathMode::ALLOW_CONTRACT)
        .bits();
    assert_eq!(
        mode(vec![inst(
            Op::Decorate,
            &[7, Decoration::FPFastMathMode as u32, relaxed | 2],
        )]),
        pair("relaxed", "precise")
    );
    // FPFastMathDefault through its constant.
    assert_eq!(
        mode(vec![
            inst(
                Op::ExecutionModeId,
                &[MAIN, ExecutionMode::FPFastMathDefault as u32, 3, 9],
            ),
            inst(Op::Constant, &[4, 9, relaxed]),
        ]),
        pair("relaxed", "precise")
    );
    // Another entry point's execution modes do not count.
    assert_eq!(
        mode(vec![inst(
            Op::ExecutionMode,
            &[MAIN + 1, ExecutionMode::SignedZeroInfNanPreserve as u32, 32],
        )]),
        pair("fast", "fast")
    );
    // A default whose flags are no constant is not guessed.
    let mut body = vec![entry(ExecutionModel::GLCompute)];
    body.push(inst(
        Op::ExecutionModeId,
        &[MAIN, ExecutionMode::FPFastMathDefault as u32, 3, 9],
    ));
    assert!(moltenvk_msl_args(&module(&body)).is_err());
}

#[test]
fn graphics_stages_take_their_pipeline_options() {
    let v = args(&[entry(ExecutionModel::Vertex)]);
    assert!(has(&v.spirv_cross, &["--stage", "vert"]));
    assert!(has(&v.spirv_cross, &["--flip-vert-y"]));
    assert!(has(&v.spirv_cross, &["--msl-agx-manual-cube-grad-fixup"]));
    assert!(!has(&v.spirv_cross, &["--msl-fixed-subgroup-size"]));
    let f = args(&[entry(ExecutionModel::Fragment)]);
    assert!(has(&f.spirv_cross, &["--stage", "frag"]));
    assert!(has(&f.spirv_cross, &["--msl-fixed-subgroup-size", "32"]));
    assert!(has(&f.spirv_cross, &["--msl-check-discarded-frag-stores"]));
    assert!(has(&f.spirv_cross, &["--msl-pad-fragment-output"]));
    assert!(!has(&f.spirv_cross, &["--flip-vert-y"]));
}

#[test]
fn an_invariant_position_preserves_invariance() {
    let position = |target: &[u32]| {
        let op = if target.len() == 1 {
            Op::Decorate
        } else {
            Op::MemberDecorate
        };
        let mut operands = target.to_vec();
        operands.extend([Decoration::BuiltIn as u32, BuiltIn::Position as u32]);
        inst(op, &operands)
    };
    let invariant = |target: &[u32]| {
        let op = if target.len() == 1 {
            Op::Decorate
        } else {
            Op::MemberDecorate
        };
        let mut operands = target.to_vec();
        operands.push(Decoration::Invariant as u32);
        inst(op, &operands)
    };
    let preserves = |body: Vec<Vec<u32>>| {
        let mut all = vec![entry(ExecutionModel::Vertex)];
        all.extend(body);
        args(&all)
            .metal
            .iter()
            .any(|a| a == "-fpreserve-invariance")
    };
    assert!(preserves(vec![position(&[5]), invariant(&[5])]));
    assert!(preserves(vec![position(&[6, 0]), invariant(&[6, 0])]));
    assert!(!preserves(vec![position(&[6, 0]), invariant(&[6, 1])]));
    assert!(!preserves(vec![position(&[5]), invariant(&[8])]));
    assert!(!preserves(vec![position(&[5])]));
}

#[test]
fn a_module_without_exactly_one_known_stage_has_no_arguments() {
    assert!(moltenvk_msl_args(&module(&[])).is_err());
    assert!(moltenvk_msl_args(&module(&[
        entry(ExecutionModel::GLCompute),
        entry(ExecutionModel::Fragment),
    ]))
    .is_err());
    assert!(moltenvk_msl_args(&module(&[entry(ExecutionModel::TessellationControl)])).is_err());
    assert!(moltenvk_msl_args(&[0x0723_0203]).is_err());
}

/// A kernel with no `source_filename` line: an assembler names its module after its input.
const UNNAMED_KERNEL: &str = r#"target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define void @k(ptr addrspace(1) %out, <3 x i32> %gid) {
entry:
  %x = extractelement <3 x i32> %gid, i64 0
  %i = zext i32 %x to i64
  %slot = getelementptr i32, ptr addrspace(1) %out, i64 %i
  store i32 %x, ptr addrspace(1) %slot, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3"}
"#;

/// `text` written to `row.ll` in two directories whose paths differ in length, each in its own
/// scratch (removed on drop).
fn two_places(text: &str) -> ([PathBuf; 2], [Scratch; 2]) {
    let (near, far) = (Scratch::new("m2v-near"), Scratch::new("m2v-far"));
    let deep = far
        .path()
        .join("a-directory-whose-name-makes-the-path-longer");
    std::fs::create_dir_all(&deep).unwrap();
    let paths = [near.path().join("row.ll"), deep.join("row.ll")];
    for path in &paths {
        std::fs::write(path, text).unwrap();
    }
    (paths, [near, far])
}

/// One text assembles to one bitcode wherever it lies, so one row is charged the same work units
/// and ledger bytes on every run. Read from a path, `llvm-as` named an unnamed module after that
/// path, the bitcode carried it, and the bitstream reader meters a record by its operands: a census
/// row assembled in scratch directories named for process ids and counters was charged by their
/// lengths. Missing `llvm-as` skips visibly.
#[test]
fn a_text_is_charged_the_same_wherever_it_is_assembled() {
    let ([near, far], _scratch) = two_places(UNNAMED_KERNEL);
    let assembled = |path: &Path| match llvm_assemble(path) {
        Ok(bitcode) => Some(bitcode),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the kernel: {why}"),
    };
    let (Some(a), Some(b)) = (assembled(&near), assembled(&far)) else {
        return;
    };
    let p = profile("runtime-floor", DevicePolicy::default()).expect("runtime-floor");
    let charged = |bitcode: &[u8]| {
        let mut m = meter();
        let artifact = compile_bitcode(
            bitcode,
            EntryRef::Name("k"),
            &FunctionConstantValues::defaults(),
            &p,
            &LowerInputs::new(),
            &mut m,
            SPIRV_WORDS,
        )
        .expect("the kernel compiles");
        (m.work(), m.ledger_peak(), artifact)
    };
    let (near_cost, far_cost) = (charged(&a), charged(&b));
    assert_eq!(
        (near_cost.0, near_cost.1),
        (far_cost.0, far_cost.1),
        "one text was charged different work units or ledger bytes in two directories"
    );
    assert!(
        near_cost.2 == far_cost.2,
        "one text compiled to two artifacts"
    );
    assert!(
        a == b,
        "one text assembled to two bitcodes in two directories"
    );
}

/// For `metal-as`, which the typed-pointer fixtures go through: a companion's text without its
/// `source_filename` line assembles to one bitcode in two directories. Skips visibly off macOS or
/// without the Metal toolchain.
#[test]
fn a_typed_pointer_text_assembles_the_same_wherever_it_lies() {
    let companion = std::fs::read_dir(fixtures_dir())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.to_str().is_some_and(|p| p.ends_with(".air.ll")))
        .min()
        .expect("a typed-pointer companion fixture");
    let text: String = std::fs::read_to_string(&companion)
        .unwrap()
        .lines()
        .filter(|line| !line.starts_with("source_filename"))
        .map(|line| format!("{line}\n"))
        .collect();
    let ([near, far], _scratch) = two_places(&text);
    let (a, b) = match (metal_assemble(&near), metal_assemble(&far)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(AssembleError::Missing(why)), _) | (_, Err(AssembleError::Missing(why))) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        (Err(e), _) | (_, Err(e)) => panic!("metal-as rejected {}: {e:?}", companion.display()),
    };
    assert!(a == b, "one typed-pointer text assembled to two bitcodes");
}
