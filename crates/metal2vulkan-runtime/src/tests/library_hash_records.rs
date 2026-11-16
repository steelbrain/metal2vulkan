//! A library of a metallib of 1 MiB or more, whose content hash the translator computes beside its
//! parse (`metal2vulkan::library`, `HASHED_BESIDE`), is the library the inline path makes of the
//! same bytes, and names itself in its states' pipeline records alike: the same metallib held as a
//! universal file's AIR slice takes the inline path (its content is the slice, which only the parse
//! finds), and a state of the same kernel from each hands the record hook the same bytes.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`). The kernel is
//! the owned fixture `validation/fixtures/public/examples/runtime_examples.ll`'s `scale_and_offset`
//! (`llvm-as`), beside functions of bitcode no test decodes, which take the metallib past 1 MiB.

use super::blit_holds::{options, Scratch};
use crate::{Device, PipelineOptions, PipelineRecord};
use metal2vulkan_devtools::metallib::{metallib, universal, wrap, MtlbFunction, UniversalHeader};
use std::sync::{Arc, Mutex};

/// The kernel the states are made of.
const KERNEL: &str = "scale_and_offset";

/// The AIR slice's CPU type in a universal file.
const AIR_SLICE: u32 = 0x0100_0017;

/// The least metallib the translator hashes beside its parse (`HASHED_BESIDE`).
const BESIDE: usize = 1 << 20;

/// A metallib of [`KERNEL`] and 48 functions of 32 KiB of bitcode no test decodes.
fn large_metallib() -> Vec<u8> {
    let ll = metal2vulkan_devtools::fixtures_dir().join("examples/runtime_examples.ll");
    let kernel = wrap(&metal2vulkan_devtools::llvm_assemble(&ll).unwrap());
    let names: Vec<String> = (0..48).map(|i| format!("padding_{i:02}")).collect();
    let function = |name, bitcode| MtlbFunction {
        name,
        type_code: 2,
        bitcode,
        constants: Vec::new(),
        public: Vec::new(),
        tags: Vec::new(),
    };
    let mut functions = vec![function(KERNEL, kernel)];
    functions.extend(names.iter().enumerate().map(|(i, name)| {
        let mut stream = b"BC\xc0\xde".to_vec();
        stream.extend((0..32 << 10).map(|at: usize| (at ^ i) as u8));
        function(name, wrap(&stream))
    }));
    let bytes = metallib(&functions);
    assert!(bytes.len() >= BESIDE, "{} bytes", bytes.len());
    bytes
}

#[test]
fn a_large_library_and_its_records_are_the_inline_path_s() {
    let scratch = Scratch::new("library-hash-records");
    let records: Arc<Mutex<Vec<PipelineRecord>>> = Arc::default();
    let mut options = options(&scratch.0);
    let kept = Arc::clone(&records);
    options.pipeline_records = Some(Arc::new(move |record| kept.lock().unwrap().push(record)));
    let device = Device::system_default_with(&options).unwrap();
    let bytes = large_metallib();
    let beside = device.new_library_with_data(&bytes).expect("the metallib");
    let wrapped = universal(UniversalHeader::MachO, &[(AIR_SLICE, &bytes)]);
    let inline = device
        .new_library_with_data(&wrapped)
        .expect("the universal file");
    assert_eq!(
        beside.compile_library().content_hash(),
        inline.compile_library().content_hash()
    );
    assert!(
        beside.compile_library() == inline.compile_library(),
        "the whole library is the inline path's"
    );
    let states: Vec<_> = [&beside, &inline]
        .into_iter()
        .map(|library| {
            let function = library.new_function_with_name(KERNEL).expect("the kernel");
            device
                .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
                .expect("its state")
        })
        .collect();
    let records = records.lock().unwrap();
    assert_eq!(records.len(), 2, "one record a creation");
    assert_eq!(
        records[0].to_bytes(),
        records[1].to_bytes(),
        "the large library's record is the inline path's, byte for byte"
    );
    drop(states);
}
