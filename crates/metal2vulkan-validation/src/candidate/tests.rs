use super::*;

// Generated-shader differential tests on a Vulkan device: the dev-tools crate's generators
// (`metal2vulkan_devtools::fuzz`) on the engine on either device, its modules as compiled. The
// product's own tests run the same generators on the CPU interpreter in the ordinary gate
// (`src/tests/fuzz_interp.rs`).
#[cfg(all(any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
mod access_fuzz;
#[cfg(all(any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
mod gpu_fuzz;
#[cfg(all(any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
mod inline_fuzz;
#[cfg(all(any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
mod irreducible_fuzz;
#[cfg(all(any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
mod pointer_select_fuzz;
#[cfg(all(any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
mod structured_fuzz;
#[cfg(all(any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
mod workgroup_fuzz;

// These execute on a Vulkan device: a real GPU under `--cfg m2v_gpu` or a CPU ICD under
// `--cfg m2v_cpu_icd`. The ordinary gate opens no device.
// L4 driver acceptance: creates engine pipelines, dispatches nothing.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod acceptance_pipelines;
// The engine compute candidate: dispatches through the runtime under the validation layer, cyclic
// ones included, on either device.
#[cfg(all(any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
pub(crate) mod engine_compute;
// The same candidate with textures, texture buffers, and samplers.
#[cfg(all(any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
mod engine_textures;
// The engine render candidate: owned vertex/fragment draws through the runtime, the looping
// fragment included, on either device.
#[cfg(all(any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
pub(crate) mod engine_render;
// The loop kernels whose exits read values their own trips defined, and the loop a thread leaves
// early, on either device, with the stores of an undefined value their modules ran.
#[cfg(all(any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
mod loop_exits;
mod pipeline_contracts;
// The widest-pin rule's emulated form against its pinned form on a GPU: exact integer answers.
#[cfg(all(m2v_gpu, target_os = "linux"))]
mod widest_pin;

/// The compile worker the device tests compile through: this test binary
/// (`CompileWorker::lib_tests`), which installs the counting allocator and serves the worker through
/// its fixture test, so each test pipeline compiles under the 20-s and 500-MiB boundary in a
/// killable child, never in the process that holds the Vulkan device. It is the binary under
/// test, built with the same flags, and every selection that runs these tests holds it: a
/// `--lib`-only build or archive has no package binary.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(crate) fn compile_worker() -> &'static crate::worker::compile::CompileWorker {
    static WORKER: std::sync::OnceLock<crate::worker::compile::CompileWorker> =
        std::sync::OnceLock::new();
    WORKER.get_or_init(|| crate::worker::compile::CompileWorker::lib_tests().unwrap())
}
