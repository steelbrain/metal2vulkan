//! Every tool that runs engine cases serves the device worker: started with
//! `--engine-device-worker`, it reads the parent's input from stdin, applies the parent's
//! profiles-layer restriction, and reports a typed result with its build identity and
//! framebuffer-fetch policy on stdout, so the parent's device half never runs in the parent.

use metal2vulkan_validation::observation::TRANSLATOR_FINGERPRINT;
use metal2vulkan_validation::worker::device::{VALIDATION_FINGERPRINT, WORKER_ARG};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

/// Every binary that executes engine cases (`candidate::engine::execute_prepared`).
fn serving() -> [PathBuf; 2] {
    [bin_exe!("corpus-vulkan"), bin_exe!("corpus-moltenvk")]
}

/// A public authored case (`primitive-acceleration-structure-intersection`), whose AIR the
/// public fixtures give.
const CASE: &str = r#"{"acceleration_structures":[{"binding":5,"kind":"primitive","primitive_triangles_b64":"AACAvwAAgL8AAAAAAACAPwAAgL8AAAAAAAAAAAAAgD8AAAAA"}],"air_sha256":"3dec51430c72d7901180dcdc22624a135640bf74af06c3a59b100b048a4107c6","argument_buffer_buffers":[],"argument_buffer_intersection_function_tables":[],"argument_buffer_textures":[],"authored_by":"codex:gpt-5.6-sol","buffers":[{"binding":0,"initial_bytes_b64":"q6urq6urq6urq6urq6urq6urq6urq6ur","role":"output"}],"case_id":"21de8b823123dfcac765adf9786fbb1f0741b4543d307064df6832313eb6b575","compare":{"kind":"exact"},"device_buffer_arrays":[],"dispatch":{"grid":[1,1,1],"threads_per_threadgroup":[1,1,1]},"entry":"primitive_as_intersect","execution_safety":"loop_free","function_constants":[],"intersection_function_tables":[],"kernel_stage_inputs":[],"name":"primitive-acceleration-structure-intersection","output":{"binding":0,"kind":"buffer","length":24,"offset":0},"rationale":"A downward ray hits one authored counter-clockwise triangle at unit distance; all observable triangle-intersection result fields are compared exactly.","render_targets":[],"samplers":[],"stage":"kernel","texture_arrays":[],"textures":[],"threadgroup_memory":[],"vertex_inputs":[],"visible_function_references":[],"visible_function_tables":[]}"#;

/// The worker input for [`CASE`] restricted to `restricted_to`, in the stated robustness leg,
/// with `VK_EXT_device_generated_commands` enabled where supported or not (`generated_commands`),
/// claiming a checked-input digest the case does not have.
fn input(restricted_to: Option<&str>, generated_commands: bool) -> String {
    format!(
        r#"{{"root":"/nonexistent/device-worker-corpus","case":{CASE},"device":"Cpu","backend":"vulkan","input_sha256":"{}","wait_timeout_ms":10000,"restricted_to":{},"leg":"stated","generated_commands":{generated_commands}}}"#,
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
    for path in serving() {
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
}

/// The worker restricts its contexts to the profile its parent's are restricted to before it
/// checks the case or opens a device: a profile it cannot apply stops it there (exit 1, named on
/// stderr), with no report.
#[test]
fn a_worker_applies_the_parents_restriction_first() {
    for path in serving() {
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
}

/// An input the worker cannot decode is the worker's own failure (exit 1, named on stderr), not a
/// tool's usage error.
#[test]
fn a_worker_refuses_an_input_it_cannot_decode() {
    for path in serving() {
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
}
