//! The executor plumbing of the generated-shader differential tests on a Vulkan device.
//!
//! The generators and their Rust reference models live in the dev-tools crate
//! (`metal2vulkan_devtools::fuzz`); the `*_fuzz.rs` test modules next to this one run them on
//! [`device::Engine`] (`--cfg m2v_cpu_icd` or `--cfg m2v_gpu`): the engine on the test's
//! Vulkan device through the binder, the candidate executor's own path
//! (`platform::execute_engine`: the validation layer on, the status read before the output).
//! The modules carry no execution bound, so cyclic programs run on a GPU as compiled: this is the
//! GPU lane (Q-fuzz). A program that does not end is not cancelled: the finite device wait ends
//! only the test's wait (the test fails), the hang tripwire only refuses later device opens, and
//! only the OS or driver GPU timeout stops the shader. The product's own tests run the same
//! generators on the CPU interpreter (`src/tests/fuzz_interp.rs`).

/// The device wait of the fuzz lanes on a CPU ICD: llvmpipe compiles a compute shader when it is
/// first dispatched, on its queue thread, so there the dispatch's wait includes the JIT of the
/// whole module. The largest generated programs (a depth-5 `structured` nest, 650 KB of AIR, 167k
/// SPIR-V words) took 13.7 s end to end on lavapipe with `LP_NUM_THREADS=1`, past the executor's 10
/// s default, which a GPU keeps (it compiles at pipeline creation, held to the 5 s creation
/// ceiling).
#[cfg(m2v_cpu_icd)]
const CPU_ICD_WAIT: std::time::Duration = std::time::Duration::from_secs(120);

/// Set this test process's device wait for the test's device: [`CPU_ICD_WAIT`] on a CPU ICD, the
/// executor's default on a GPU.
pub(super) fn device_wait() {
    #[cfg(m2v_cpu_icd)]
    crate::gpu::wait::set_wait_timeout(CPU_ICD_WAIT).expect("the CPU ICD's device wait");
}

/// The engine on the test's device, which is opened once here for what it supports.
pub(super) fn engine() -> device::Engine {
    device::Engine::open().expect("open the test's device")
}

/// The engine on the Vulkan device the test binary was built for.
pub(super) mod device {
    use super::super::*;
    use crate::candidate::engine::{ComputeCase, ComputeOutput};
    use crate::check::admission::ExecutorError;
    use metal2vulkan::bind::Dispatch;
    use metal2vulkan::library::FunctionConstantValues;
    use metal2vulkan_devtools::fuzz::{
        bytes_words, word_bytes, Execute, Input, Kernel, Run, Target,
    };
    use metal2vulkan_devtools::{llvm_assemble, AssembleError};
    use std::collections::BTreeMap;

    /// The engine on the test's device.
    pub(in crate::candidate::tests) struct Engine {
        /// Whether the opened device reaches buffers by address (Profile A).
        addressed: bool,
    }

    impl Engine {
        /// Open the test's device once for what the generators may build for it.
        pub(in crate::candidate::tests) fn open() -> Result<Engine, ExecutorError> {
            let device = platform::opened_worker_device(crate::gpu::TEST_DEVICE)?;
            Ok(Engine {
                addressed: device.device().profile().addressed_needs().is_empty(),
            })
        }
    }

    impl Execute for Engine {
        fn target(&self) -> Target {
            Target {
                addressed: self.addressed,
                ..Target::default()
            }
        }

        /// Assemble `kernel`, map it onto the candidate's [`ComputeCase`], and run it through
        /// the binder on the test's device.
        fn execute(&mut self, kernel: &Kernel<'_>) -> Result<Run, String> {
            let bitcode = assemble(kernel.entry, kernel.air)?;
            let case = compute_case(kernel)?;
            let run = run(&format!("fuzz-{}", kernel.entry), &case, &bitcode)
                .map_err(|error| error.to_string())?;
            if run.output.len() != kernel.output_init.len() {
                return Err(format!(
                    "read back {} bytes, wanted {}",
                    run.output.len(),
                    kernel.output_init.len()
                ));
            }
            Ok(Run {
                words: bytes_words(&run.output),
            })
        }
    }

    /// Bitcode of the generated text `air` (`llvm-as`, in a scratch directory).
    pub(in crate::candidate::tests) fn assemble(entry: &str, air: &str) -> Result<Vec<u8>, String> {
        let scratch = crate::ScratchDir::new("fuzz-device")?;
        let path = scratch.path().join(format!("{entry}.ll"));
        std::fs::write(&path, air).map_err(|error| format!("write {}: {error}", path.display()))?;
        llvm_assemble(&path).map_err(|error| match error {
            AssembleError::Missing(why) => format!("llvm-as: {why}"),
            AssembleError::Rejected(why) => format!("llvm-as rejected the generated text: {why}"),
        })
    }

    /// One engine dispatch of `case` (`bitcode`'s entry) on the test's device, compiled in
    /// the device tests' compile worker for the bind shape the dispatch plans, labelled `label` for
    /// the hang protocol.
    pub(in crate::candidate::tests) fn run(
        label: &str,
        case: &ComputeCase,
        bitcode: &[u8],
    ) -> Result<crate::candidate::engine::ComputeRun, ExecutorError> {
        super::device_wait();
        platform::execute_engine(
            crate::gpu::TEST_DEVICE,
            label,
            case,
            |device, shape| {
                crate::candidate::engine::compile_case(
                    super::super::compile_worker(),
                    device,
                    &crate::acceptance::retained(bitcode),
                    &[],
                    case,
                    shape,
                )
            },
            |_, token, _, _| Err(format!("no template has pipeline token {token}").into()),
            &[],
        )
        .result
    }

    /// `kernel` as one candidate dispatch: buffer 0 the output, each input at its index, a
    /// pointer table ([`Input::Addresses`]) an owner whose fields point at argument-buffer
    /// buffers holding the listed inputs' bytes (exact: inputs are read only, so a copy and the
    /// bound buffer never differ), and one threadgroup of the kernel's threads.
    pub(in crate::candidate::tests) fn compute_case(
        kernel: &Kernel<'_>,
    ) -> Result<ComputeCase, String> {
        let mut buffers = BTreeMap::from([(0, kernel.output_init.clone())]);
        let mut argument_buffer_buffers = BTreeMap::new();
        for (at, input) in kernel.inputs.iter().enumerate() {
            let index = u32::try_from(at + 1).map_err(|error| error.to_string())?;
            match input {
                Input::Words(words) => {
                    buffers.insert(index, word_bytes(words));
                }
                Input::Addresses(listed) => {
                    buffers.insert(index, vec![0; 8 * listed.len()]);
                    for (field, &buffer) in listed.iter().enumerate() {
                        let pointee = buffer.checked_sub(1).and_then(|at| kernel.inputs.get(at));
                        let Some(Input::Words(words)) = pointee else {
                            return Err(format!(
                                "pointer table {index} lists buffer {buffer}, not a word input"
                            ));
                        };
                        let field = u32::try_from(8 * field).map_err(|error| error.to_string())?;
                        argument_buffer_buffers.insert((index, field), word_bytes(words));
                    }
                }
            }
        }
        Ok(ComputeCase {
            out_of_bounds: Vec::new(),
            creation_bound: crate::budgets::PIPELINE_CREATION_CEILING,
            entry: kernel.entry.to_owned(),
            constants: FunctionConstantValues::defaults(),
            buffers,
            argument_buffer_buffers,
            argument_buffer_textures: BTreeMap::new(),
            argument_buffer_samplers: BTreeMap::new(),
            objects: BTreeMap::new(),
            command_buffers: BTreeMap::new(),
            textures: BTreeMap::new(),
            samplers: BTreeMap::new(),
            threadgroup_lengths: BTreeMap::new(),
            dispatch: Dispatch::Threads {
                threads: [kernel.threads, 1, 1],
                threads_per_group: [kernel.threads, 1, 1],
            },
            output: ComputeOutput::Buffer {
                binding: 0,
                offset: 0,
                length: kernel.output_init.len() as u64,
            },
        })
    }
}
