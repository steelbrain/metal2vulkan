//! Every tool that runs engine cases serves the device worker: started with
//! `--engine-device-worker`, it reads the parent's input from stdin, applies the parent's
//! profiles-layer restriction, and reports a typed result with its build identity and
//! framebuffer-fetch policy on stdout, so the parent's device half never runs in the parent.

use metal2vulkan_validation::observation::TRANSLATOR_FINGERPRINT;
use metal2vulkan_validation::worker::device::{VALIDATION_FINGERPRINT, WORKER_ARG};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

/// The binary that executes engine cases (`candidate::engine::execute_prepared`).
fn serving() -> PathBuf {
    bin_exe!("corpus-vulkan")
}

/// A public authored case (`single-thread-store-constant`), whose AIR the public fixtures give.
const CASE: &str = r#"{"acceleration_structures":[],"air_sha256":"dc4e4be315732ae7ea24fe171945c4c25b42ca65e10f83530eed6cc95f80b50b","argument_buffer_buffers":[],"argument_buffer_intersection_function_tables":[],"argument_buffer_textures":[],"authored_by":"metal2vulkan project","buffers":[{"binding":0,"initial_bytes_b64":"q6urqw==","role":"output"}],"case_id":"1e54315eb0ffcac633d4992c577f3e6ac384ed1bdf2321887c1f37242c00c887","compare":{"kind":"exact"},"device_buffer_arrays":[],"dispatch":{"grid":[1,1,1],"threads_per_threadgroup":[1,1,1]},"entry":"store_const","execution_safety":"loop_free","function_constants":[],"intersection_function_tables":[],"kernel_stage_inputs":[],"name":"single-thread-store-constant","output":{"binding":0,"kind":"buffer","length":4,"offset":0},"public_source":"crates/metal2vulkan-validation/fixtures/public/kernel_store_const.ll","rationale":"A single invocation writes the integer constant 42 into one explicitly poisoned output word.","render_targets":[],"samplers":[],"stage":"kernel","texture_arrays":[],"textures":[],"threadgroup_memory":[],"vertex_inputs":[],"visible_function_references":[],"visible_function_tables":[]}"#;

/// The worker input for [`CASE`] restricted to `restricted_to`, with `VK_EXT_device_generated_commands` enabled where supported or not (`generated_commands`),
/// claiming a checked-input digest the case does not have.
fn input(restricted_to: Option<&str>, generated_commands: bool) -> String {
    format!(
        r#"{{"root":"/nonexistent/device-worker-corpus","case":{CASE},"device":"Cpu","input_sha256":"{}","wait_timeout_ms":10000,"restricted_to":{},"generated_commands":{generated_commands}}}"#,
        "0".repeat(64),
        serde_json::to_string(&restricted_to).unwrap()
    )
}

/// `program` started as a device worker on `input`.
fn serve(program: &Path, input: &str) -> Output {
    let mut child = Command::new(program)
        .arg(WORKER_ARG)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

/// The worker's own check disagrees with its parent's input digest, and it reports that as the
/// run's typed failure, with its build identity and the generated-commands path its parent asked
/// for, before it opens a device: unrestricted, and restricted to a committed profile.
#[test]
fn every_case_running_tool_serves_the_device_worker() {
    let path = serving();
    let program = path.display();
    for (restricted_to, generated_commands) in
        [(None, true), (Some("gate-lavapipe"), true), (None, false)]
    {
        let output = serve(&path, &input(restricted_to, generated_commands));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{program}: {stderr}");
        let report: serde_json::Value = serde_json::from_slice(&output.stdout)
            .unwrap_or_else(|error| panic!("{program}: {error}: {stderr}"));
        let result = &report["result"];
        assert_eq!(
            result["product_fingerprint"], TRANSLATOR_FINGERPRINT,
            "{program}"
        );
        assert_eq!(
            result["validation_fingerprint"], VALIDATION_FINGERPRINT,
            "{program}"
        );
        assert_eq!(
            result["generated_commands"], generated_commands,
            "{program}"
        );
        let failed = result["result"]["Err"]["Failed"]
            .as_str()
            .unwrap_or_else(|| panic!("{program}: {report}"));
        assert!(
            failed.contains(&format!("its parent as {}", "0".repeat(64))),
            "{program}: {failed}"
        );
    }
}

/// The worker restricts its contexts to the profile its parent's are restricted to before it
/// checks the case or opens a device: a profile it cannot apply stops it there (exit 1, named on
/// stderr), with no report.
#[test]
fn a_worker_applies_the_parents_restriction_first() {
    let path = serving();
    let program = path.display();
    let output = serve(&path, &input(Some("no-such-profile"), true));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{program}: {stderr}");
    assert!(
        stderr.starts_with("engine device worker: unknown profile \"no-such-profile\""),
        "{program}: {stderr}"
    );
    assert!(output.stdout.is_empty(), "{program}");
}

/// An input the worker cannot decode is the worker's own failure (exit 1, named on stderr), not a
/// tool's usage error.
#[test]
fn a_worker_refuses_an_input_it_cannot_decode() {
    let path = serving();
    let program = path.display();
    let output = serve(&path, "{}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{program}: {stderr}");
    assert!(
        stderr.starts_with("engine device worker: decode the device worker's input"),
        "{program}: {stderr}"
    );
    assert!(output.stdout.is_empty(), "{program}");
}
