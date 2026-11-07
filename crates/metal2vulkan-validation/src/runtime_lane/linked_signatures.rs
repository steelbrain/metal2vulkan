//! A visible function called through a declaration of another type than its definition's,
//! through the runtime on the device the cfg selects, under the validation layer, against what
//! the probe `linked_function_signatures.m` (the conformance crate's `probes/`) measured on
//! Metal: each case of `metal2vulkan_devtools::linked_signatures` links its function into its
//! kernel of the owned fixture `linked_signature_caller`, the arguments passed by position, and
//! stores Metal's word 0 where Metal defined one (the stitched `stitched_pair(a, b)` linked for
//! `stitched_pair(a, b, c)` stores 5, `float int_param(float, int, float)` linked for
//! `(float, float, float)` stores 2 + the bits of 3.0, 1.07794e9) and the translator's own where
//! Metal's is undefined; the creation of a state whose shape no adapter passes is refused as not
//! implemented. The cases Metal measured in `privateFunctions` too (`stitched_extra`, `extra`)
//! link that way as well.
//!
//! The functions are `linked_signature_functions`'s; the stitched library is
//! `metal2vulkan_devtools::stitched`'s of `STITCHED_PAIR`, made from the library's bytes
//! (`newLibraryWithData:`).

use super::host_memory::{on_device, run};
use super::read_buffer;
use metal2vulkan::error::{Error as CompileError, Feature};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, LinkedFunctions, StageSource};
use metal2vulkan_devtools::linked_signatures::{Answer, Case, CASES, STITCHED_PAIR};
use metal2vulkan_devtools::stitched;
use metal2vulkan_devtools::{engine_dir, llvm_assemble, llvm_assemble_text};
use metal2vulkan_runtime::{Device, Error, Library, PipelineOptions, ResourceOptions, Size};

/// The caller's library, the functions' library and the stitched library.
fn libraries(device: &Device) -> [Library; 3] {
    let fixture = |stem: &str| llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
    let module = llvm_assemble_text(&stitched::module(&STITCHED_PAIR)).unwrap();
    let named: [(&str, &[u8]); 1] = [(STITCHED_PAIR.name, &module)];
    [
        device
            .new_library_with_air(&fixture("linked_signature_caller"))
            .unwrap(),
        device
            .new_library_with_air(&fixture("linked_signature_functions"))
            .unwrap(),
        device
            .new_library_with_data(&stitched::library(&named))
            .unwrap(),
    ]
}

/// Words 0 and 1 the case's kernel stores linking `linked`, one thread on `in = {2, 3, 4, 5}`, or
/// the state's refusal.
fn words(
    device: &Device,
    caller: &Library,
    case: &Case,
    linked: LinkedFunctions,
) -> Result<[u32; 2], Error> {
    let kernel = caller
        .new_function_with_name(&format!("k_{}", case.name))
        .unwrap();
    let state = device.new_compute_pipeline_state(
        &ComputePipelineDescriptor::new(StageSource::Function(
            kernel.function_descriptor().with_linked(linked),
        )),
        &PipelineOptions::default(),
    )?;
    let input: Vec<u8> = [2.0f32, 3.0, 4.0, 5.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let input = device
        .new_buffer_with_bytes(&input, ResourceOptions::shared())
        .unwrap();
    let out = device
        .new_buffer_with_bytes(&[0xff; 8], ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder.set_buffer(Some(&input), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(1, 1, 1), Size::new(1, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
    let bytes = read_buffer(&out, 0, 8).unwrap();
    let word = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    Ok([word(0), word(1)])
}

/// Each case stores Metal's word where Metal defined one and the translator's where Metal's is
/// undefined, then the sentinel, linked in `functions` (and in `privateFunctions` for the cases
/// Metal measured so); a shape no adapter passes refuses the state's creation as not
/// implemented, where Metal makes it.
#[test]
fn a_reference_of_another_type_computes_by_position() {
    on_device(|device| {
        let [caller, functions, stitched] = libraries(device);
        for case in &CASES {
            let library = if case.stitched { &stitched } else { &functions };
            let f = library.new_function_with_name(case.function).unwrap();
            let mut lists = vec![LinkedFunctions::new().with_function(f.linked_function())];
            if matches!(case.name, "stitched_extra" | "extra") {
                lists.push(LinkedFunctions::new().with_private_function(f.linked_function()));
            }
            for linked in lists {
                let got = words(device, &caller, case, linked);
                match case.answer {
                    Answer::Metal(w) | Answer::Undefined(w) => {
                        let got = got.map_err(|e| format!("{e:?}"));
                        assert_eq!(got, Ok([w, 0x5e47]), "{}", case.name);
                    }
                    Answer::Refused => match got {
                        Err(Error::Compile(CompileError::Unsupported(u))) => {
                            assert_eq!(u.feature, Feature::Linking, "{}: {u}", case.name);
                        }
                        other => panic!("{}: not refused: {other:?}", case.name),
                    },
                }
            }
        }
    });
}
