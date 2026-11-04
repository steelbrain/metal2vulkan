use super::*;

// Generated-shader differential tests on a Vulkan device: the dev-tools crate's generators
// (`metal2vulkan_devtools::fuzz`) on the engine on either device, its modules as compiled. The
// product's own tests run the same generators on the CPU interpreter in the ordinary gate
// (`src/tests/fuzz_interp.rs`).
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod access_fuzz;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod gpu_fuzz;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod inline_fuzz;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod irreducible_fuzz;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod pointer_select_fuzz;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod structured_fuzz;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod workgroup_fuzz;

// These execute on a Vulkan device: a real GPU under `--cfg m2v_gpu` or a CPU ICD under
// `--cfg m2v_cpu_icd`. The ordinary gate opens no device.
// L4 driver acceptance: creates engine pipelines, dispatches nothing.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod acceptance_pipelines;
// The engine compute candidate: binder-planned dispatches under the validation layer, cyclic
// ones included, on either device.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(crate) mod engine_compute;
// The same candidate with textures, texture buffers, and samplers.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod engine_textures;
// The engine render candidate: owned vertex/fragment draws, the looping fragment included, on
// either device.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(crate) mod engine_render;
mod pipeline_contracts;

/// The compile worker the device tests compile through: `corpus-pipeline`, the validation binary
/// in this test binary's target profile directory, which serves the worker under the counting allocator, so each
/// test pipeline compiles under the 20-s and 500-MiB boundary in a killable child, never in the
/// process that holds the Vulkan device. It is executed once first: the OS may scan a freshly
/// linked binary on its first exec (seconds on macOS), which the worker's deadline would count.
/// Cargo builds the package's binaries for any test run that includes its integration tests; a
/// `--lib`-only run that never built them fails here, naming the binary.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(crate) fn compile_worker() -> &'static crate::worker::compile::CompileWorker {
    static WORKER: std::sync::OnceLock<crate::worker::compile::CompileWorker> =
        std::sync::OnceLock::new();
    WORKER.get_or_init(|| {
        let binary = format!("corpus-pipeline{}", std::env::consts::EXE_SUFFIX);
        let test = std::env::current_exe().expect("the test binary's path");
        // The package's binaries are in `target/<profile>/`, above the test binary wherever
        // Cargo's layout puts it: `<profile>/deps/<test>`, or, in its newer build-directory
        // layout, `<profile>/build/<package>/<hash>/out/<test>`.
        let program = test
            .ancestors()
            .skip(1)
            .map(|directory| directory.join(&binary))
            .find(|path| path.is_file())
            .unwrap_or_else(|| {
                panic!(
                    "the compile worker {binary} is not built above {}: run the package's tests \
                     without --lib, or build its binaries with the same RUSTFLAGS \
                     (cargo build -p metal2vulkan-validation --bins)",
                    test.display()
                )
            });
        let _ = std::process::Command::new(&program)
            .arg("--help")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        crate::worker::compile::CompileWorker::program(program)
    })
}
