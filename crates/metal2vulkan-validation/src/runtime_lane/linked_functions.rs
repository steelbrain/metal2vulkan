//! Linked functions under a specialized name, and linked modules that declare function constants,
//! through the runtime on the device the cfg selects, under the validation layer, against what the
//! probe `linked_function_specialized_name.m` (the conformance crate's `probes/`) measured on an
//! Apple M3 Max under macOS 27.0.1.
//!
//! The libraries are the owned fixtures `linked_specialized_caller` (kernels calling `custom_fn`,
//! which no library defines, and `base_fn`, `guarded_fn` and `plain_fn` by their own names) and
//! `linked_specialized_functions` (those visible functions, in a module that declares the float
//! constant `scale` at index 0: `base_fn(x) = x * scale + 1`, `derived_fn(x) = x * twice` with
//! `twice = scale * 2` computed in the static initializer, `guarded_fn(x) =
//! is_function_constant_defined(scale) ? x * scale : x - 1`, `plain_fn(x) = x + 2`), the probe's
//! callee. Each kernel stores its callee's value at `tid + 5` to word `tid`, so word 0 is the
//! probe's word.

use super::at_plan;
use super::host_memory::on_device_with;
use super::host_memory::run;
use super::read_buffer;
use crate::air_library::AirLibrary;
use metal2vulkan::error::RuntimeInput;
use metal2vulkan::error::{DescriptorError, DiagName, Error as CompileError, LinkProblem};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::DataType;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, LinkedFunctions, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ComputePipelineState, Device, Error, Function, FunctionDescriptor, Library, PipelineOptions,
    ResourceOptions, Size,
};

/// Threads of each dispatch.
const N: u32 = 8;

/// The caller's library and the functions' library.
fn libraries(device: &Device) -> (Library, Library) {
    let fixture = |stem: &str| llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
    (
        device
            .new_library_with_air(&fixture("linked_specialized_caller"))
            .unwrap(),
        device
            .new_library_with_air(&fixture("linked_specialized_functions"))
            .unwrap(),
    )
}

/// `scale` set to `v`.
fn scale(v: f32) -> FunctionConstantValues {
    FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(DataType::Float, v.to_le_bytes()),
    )
}

/// `newFunctionWithDescriptor:` of `name` with the specialized name and constant values given.
fn made(
    library: &Library,
    name: &str,
    specialized: Option<&str>,
    values: Option<FunctionConstantValues>,
) -> Function {
    let mut d = FunctionDescriptor::new(name);
    d.specialized_name = specialized.map(str::to_owned);
    d.constant_values = values;
    library.new_function_with_descriptor(&d).unwrap()
}

/// The state of `kernel` of `caller` linking `functions` in `MTLLinkedFunctions.functions`.
fn state(
    caller: &Library,
    kernel: &str,
    functions: &[&Function],
) -> Result<ComputePipelineState, Error> {
    let linked = functions.iter().fold(LinkedFunctions::new(), |l, f| {
        l.with_function(f.linked_function())
    });
    let kernel = caller.new_function_with_name(kernel).unwrap();
    caller.device().new_compute_pipeline_state(
        &ComputePipelineDescriptor::new(StageSource::Function(
            kernel.function_descriptor().with_linked(linked),
        )),
        &PipelineOptions::default(),
    )
}

/// The words `state` stores over `N` threads.
fn words(device: &Device, state: &ComputePipelineState) -> Vec<f32> {
    let out = device
        .new_buffer(u64::from(N) * 4, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder
        .dispatch_threads(Size::new(N.into(), 1, 1), Size::new(N.into(), 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
    read_buffer(&out, 0, out.length())
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect()
}

/// `f(t + 5)` for each thread `t`.
fn expected(f: impl Fn(f32) -> f32) -> Vec<f32> {
    (0..N).map(|t| f(t as f32 + 5.0)).collect()
}

fn needs(symbol: &str) -> Error {
    Error::Compile(CompileError::NeedsInput(RuntimeInput::LinkedFunction {
        symbol: DiagName::new(symbol),
    }))
}

/// A function made with the specialized name `custom_fn` answers it and resolves the kernel's
/// `custom_fn`, storing the probe's 16 at word 0; made without one or with another it resolves
/// nothing, and the state needs `custom_fn`; linked beside a plain one it stores 16 as well. The
/// probe's `specialized`, `plain`, `other` and `both`.
#[test]
fn a_function_made_with_a_specialized_name_links_under_it() {
    on_device_with(at_plan, |device| {
        let (caller, functions) = libraries(device);
        let specialized = made(&functions, "base_fn", Some("custom_fn"), Some(scale(3.0)));
        let plain = made(&functions, "base_fn", None, Some(scale(3.0)));
        let other = made(&functions, "base_fn", Some("another_fn"), Some(scale(3.0)));
        assert_eq!(
            [specialized.name(), plain.name(), other.name()],
            ["custom_fn", "base_fn", "another_fn"]
        );
        let kernel = "linked_specialized_caller";
        let one = state(&caller, kernel, &[&specialized]).unwrap();
        let got = words(device, &one);
        assert_eq!(got[0], 16.0);
        assert_eq!(got, expected(|x| x * 3.0 + 1.0));
        assert_eq!(
            state(&caller, kernel, &[&plain]).map(|_| ()),
            Err(needs("custom_fn"))
        );
        assert_eq!(
            state(&caller, kernel, &[&other]).map(|_| ()),
            Err(needs("custom_fn"))
        );
        let both = state(&caller, kernel, &[&specialized, &plain]).unwrap();
        assert_eq!(words(device, &both)[0], 16.0);
    });
}

/// A linked function computes with its own constant values, its module's static initializer
/// run over them, and reads one it was given no value for as undefined; made by
/// `newFunctionWithName:`, it refuses the state where it reaches a constant, guarded or not, and
/// links where it reaches none. The probe's `derived` (30), `guarded-valued` (15), `guarded` (4),
/// `constant-free` (7), `byname-valued` (16), `byname`, `guarded-byname`, `byname-empty`,
/// `descriptor-nil` and `constant-free-byname` (7). Metal's undefined read stored nothing; the
/// runtime's reads zero, so `base_fn` without a value stores 1.
#[test]
fn a_linked_function_s_constants_answer_as_metal_s_do() {
    on_device_with(at_plan, |device| {
        let (caller, functions) = libraries(device);
        let custom = |name: &str, values| made(&functions, name, Some("custom_fn"), values);
        let custom_words = |f: &Function| {
            words(
                device,
                &state(&caller, "linked_specialized_caller", &[f]).unwrap(),
            )
        };
        assert_eq!(
            custom_words(&custom("derived_fn", Some(scale(3.0)))),
            expected(|x| x * 6.0)
        );
        assert_eq!(
            custom_words(&custom("guarded_fn", Some(scale(3.0)))),
            expected(|x| x * 3.0)
        );
        assert_eq!(
            custom_words(&custom("guarded_fn", None)),
            expected(|x| x - 1.0)
        );
        assert_eq!(
            custom_words(&custom("plain_fn", None)),
            expected(|x| x + 2.0)
        );
        assert_eq!(custom_words(&custom("base_fn", None)), expected(|_| 1.0));

        let by_name = |name: &str| functions.new_function_with_name(name).unwrap();
        let unresolved = |name: &str| {
            Err(Error::Compile(CompileError::InvalidDescriptor(
                DescriptorError::LinkedFunction {
                    name: DiagName::new(name),
                    problem: LinkProblem::UnresolvedConstant { index: 0 },
                },
            )))
        };
        let base = "linked_specialized_base";
        assert_eq!(
            state(&caller, base, &[&by_name("base_fn")]).map(|_| ()),
            unresolved("base_fn")
        );
        assert_eq!(
            state(
                &caller,
                "linked_specialized_guarded",
                &[&by_name("guarded_fn")]
            )
            .map(|_| ()),
            unresolved("guarded_fn")
        );
        let free = state(&caller, "linked_specialized_free", &[&by_name("plain_fn")]).unwrap();
        assert_eq!(words(device, &free)[0], 7.0);
        let valued = functions
            .new_function_with_name_constant_values("base_fn", &scale(3.0))
            .unwrap();
        assert_eq!(
            words(device, &state(&caller, base, &[&valued]).unwrap())[0],
            16.0
        );
        let empty = functions
            .new_function_with_name_constant_values("base_fn", &FunctionConstantValues::defaults())
            .unwrap();
        assert_eq!(
            words(device, &state(&caller, base, &[&empty]).unwrap()),
            expected(|_| 1.0)
        );
        let descriptor = made(&functions, "base_fn", None, None);
        assert_eq!(
            words(device, &state(&caller, base, &[&descriptor]).unwrap()),
            expected(|_| 1.0)
        );
    });
}
