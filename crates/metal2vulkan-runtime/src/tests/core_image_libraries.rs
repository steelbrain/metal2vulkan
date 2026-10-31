//! A Core Image kernel library's functions as the runtime's library answers them: extern
//! functions, named by `externFunctionNames` and not by `functionNames`, made by
//! `newExternFunctionWithName:` and `newFunctionWithName:` alike with `functionType` 4, and no
//! stage a compute state compiles. What Metal answers was measured with kernels written from
//! scratch (a probe, an M3 Max under macOS 27.0.1); `metal2vulkan-devtools` writes the container
//! in `metallib -cikernel`'s layout. An executable library answers no extern names, and its kernel
//! is made by either call.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`). The
//! executable library's kernel is the owned fixture
//! `validation/fixtures/public/engine/kernel_stage_in.ll`'s `stage_in_fetch_wide` (`llvm-as`).

use super::blit_holds::{device, Scratch};
use crate::PipelineOptions;
use metal2vulkan::metal::FunctionType;
use metal2vulkan_devtools::metallib::{core_image_metallib, metallib, wrap, MtlbFunction};
use std::path::Path;

fn function<'a>(name: &'a str, type_code: u8, bitcode: &[u8]) -> MtlbFunction<'a> {
    MtlbFunction {
        name,
        type_code,
        bitcode: bitcode.to_vec(),
        constants: Vec::new(),
        public: Vec::new(),
        tags: Vec::new(),
    }
}

#[test]
fn a_core_image_library_answers_its_extern_functions() {
    let scratch = Scratch::new("core-image-libraries");
    let device = device(&scratch.0);
    let module = wrap(b"BC\xc0\xde\x35\x14\x00\x00\x05\x00\x00\x00");
    let bytes = core_image_metallib(&[function("tint", 5, &module), function("shift", 5, &module)]);
    let library = device.new_library_with_data(&bytes).expect("library");
    assert!(library.function_names().is_empty());
    // The reverse of the table's order, as Metal lists them.
    assert_eq!(library.extern_function_names(), ["shift", "tint"]);
    for name in ["tint", "shift"] {
        for f in [
            library.new_extern_function_with_name(name),
            library.new_function_with_name(name),
        ] {
            let f = f.expect("function");
            assert_eq!(
                f.function_type().expect("type"),
                FunctionType::Extern,
                "{name}"
            );
            assert!(
                matches!(
                    device
                        .new_compute_pipeline_state_with_function(&f, &PipelineOptions::default()),
                    Err(crate::Error::Compile(_))
                ),
                "{name}"
            );
        }
    }
    assert!(library.new_extern_function_with_name("nope").is_err());

    // An executable library: no extern names, and its kernel made by either call.
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/kernel_stage_in.ll");
    let kernel = wrap(&metal2vulkan_devtools::llvm_assemble(&ll).unwrap());
    let executable = device
        .new_library_with_data(&metallib(&[function("stage_in_fetch_wide", 2, &kernel)]))
        .expect("executable");
    assert!(executable.extern_function_names().is_empty());
    let f = executable
        .new_extern_function_with_name("stage_in_fetch_wide")
        .expect("kernel");
    assert_eq!(f.function_type().expect("type"), FunctionType::Kernel);
}
