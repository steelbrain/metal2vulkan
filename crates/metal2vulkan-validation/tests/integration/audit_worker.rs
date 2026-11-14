//! The bitcode-reader audit lane across a real process boundary: `corpus-triage` serves each
//! attempt as a killable worker, and a row audited through it reports what the lane reports in
//! process, with what the parent measured; a program that does not serve the worker stops the
//! attempt, which is the row's result.

use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};
use metal2vulkan_validation::bitcode_audit::{self, Outcome};
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> Option<String> {
    match llvm_assemble(&engine_dir().join(name)) {
        Ok(bytes) => Some(base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            bytes,
        )),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

/// `corpus-triage`, executed once first: the OS may scan a freshly linked binary on its first exec
/// (seconds on macOS), which the 20-s worker deadline would count.
fn triage() -> PathBuf {
    let program = bin_exe!("corpus-triage");
    let _ = std::process::Command::new(&program)
        .arg("--help")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    program
}

#[test]
fn a_row_read_in_the_worker_reports_what_the_reader_reports_in_process() {
    let (Some(blob), Ok(llvm_dis)) = (fixture("store_const.ll"), bitcode_audit::resolve_llvm_dis())
    else {
        return;
    };
    let program = &triage();
    for blob in [blob.as_str(), "bm90IGJpdGNvZGU="] {
        let expected = bitcode_audit::audit_row(Some(blob), &llvm_dis);
        let audited = bitcode_audit::audit_row_in_worker(program, Some(blob), &llvm_dis);
        assert_eq!(audited.outcome, expected.outcome);
        assert_eq!(audited.census, expected.census);
        assert_eq!(
            (audited.ledger_peak, audited.work),
            (expected.ledger_peak, expected.work)
        );
        let measured = audited.worker.expect("a worker ran");
        assert!(measured.allocator_peak_bytes.is_some_and(|peak| peak > 0));
    }
    // "not bitcode" is refused by both sides.
    assert!(matches!(
        bitcode_audit::audit_row_in_worker(program, Some("bm90IGJpdGNvZGU="), &llvm_dis).outcome,
        Outcome::Refused {
            class: "malformed",
            llvm_accepts: false,
            ..
        }
    ));
    // No blob, and a blob that is not base64, never reach a worker.
    let missing = bitcode_audit::audit_row_in_worker(program, None, &llvm_dis);
    assert_eq!(missing.outcome, Outcome::NoBlob);
    assert!(missing.worker.is_none());
    let corrupt = bitcode_audit::audit_row_in_worker(program, Some("not*base64"), &llvm_dis);
    assert!(matches!(corrupt.outcome, Outcome::InvalidBlob(_)));
    assert!(corrupt.worker.is_none());
}

/// A program that does not serve the audit worker stops each attempt; the stop is the row's
/// result, measured, never a match or an analysis.
#[cfg(unix)]
#[test]
fn an_attempt_the_boundary_stops_is_the_rows_result() {
    let program = Path::new("true");
    let read = bitcode_audit::audit_row_in_worker(
        program,
        Some("bm90IGJpdGNvZGU="),
        Path::new("/nonexistent/llvm-dis"),
    );
    match &read.outcome {
        Outcome::Stopped { key, detail } => {
            assert_eq!(key, "worker_exit", "{detail}");
            assert_eq!(read.outcome.shape(), "worker-stopped worker_exit");
        }
        other => panic!("{other:?}"),
    }
    assert!(read.worker.is_some() && read.census.is_none());
}
