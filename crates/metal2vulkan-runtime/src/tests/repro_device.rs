//! A FALLBACK repro bundle of a refused creation (`repro.rs`), on a device: a compute state of
//! the owned fixture `ob3_stage_input`, whose kernel reads `[[stage_in]]`, made with no
//! stage-input descriptor, is refused (`Compile(NeedsInput(StageInputDescriptor))`). With a
//! directory (the test seam, `TestDirectory`), the refusal writes one bundle whose command and
//! `device.json` replay it: the device it builds is the runtime's compile device (its profile's
//! fingerprint), and compiling the bundle's request on it answers the same refusal. A second
//! identical refusal writes none, and so does a lookup of a name the library lacks (Metal's own
//! answer); without a directory, none is written. The refusal is the same each time.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::repro::TestDirectory;
use crate::{DebugMessage, Device, DeviceKind, DeviceOptions, Error, MessageSink, PipelineOptions};
use metal2vulkan::device::Device as CompileDevice;
use metal2vulkan::error::{DescriptorError, Error as CompileError, RuntimeInput};
use metal2vulkan::library::Library as CompileLibrary;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};
use metal2vulkan::repro::ReproDevice;
use metal2vulkan::Budget;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// A private directory, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "m2v-repro-device-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Scratch(path)
    }

    /// The directories under it, sorted.
    fn bundles(&self) -> Vec<PathBuf> {
        let mut bundles: Vec<PathBuf> = std::fs::read_dir(&self.0)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        bundles.sort();
        bundles
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A device of the cfg's kind, the texts of its `metal2vulkan-repro` lines into the answer's list.
fn device() -> (Device, Arc<Mutex<Vec<String>>>) {
    let heard: Arc<Mutex<Vec<String>>> = Arc::default();
    let hearing = Arc::clone(&heard);
    let sink = MessageSink::new(move |message: &DebugMessage| {
        if message.id_name == "metal2vulkan-repro" {
            hearing.lock().unwrap().push(message.text.clone());
        }
    });
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        messages: Some(sink),
        ..DeviceOptions::default()
    };
    (Device::system_default_with(&options).unwrap(), heard)
}

#[test]
fn a_refused_creation_writes_one_bundle_that_replays_it() {
    assert!(
        std::env::var_os("METAL2VULKAN_REPRO_DIR").is_none(),
        "the test runs without the variable"
    );
    let scratch = Scratch::new();
    let (device, heard) = device();
    let air = std::fs::read(metal2vulkan_devtools::engine_dir().join("ob3_stage_input.air"))
        .expect("the committed AIR");
    let library =
        device.new_library_with_compile_library(CompileLibrary::from_air(&air).expect("library"));
    let function = library
        .new_function_with_name("ob3_stage_input")
        .expect("the kernel");
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    let create = || {
        device
            .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
            .expect_err("a kernel reading [[stage_in]] with no stage-input descriptor is refused")
    };

    // Without a directory: the refusal, and no bundle.
    let refusal = create();
    let Error::Compile(compile_refusal) = &refusal else {
        panic!("{refusal:?}");
    };
    assert!(
        matches!(
            compile_refusal,
            CompileError::NeedsInput(RuntimeInput::StageInputDescriptor { .. })
        ),
        "{refusal:?}"
    );
    assert!(heard.lock().unwrap().is_empty());

    // With one: the same refusal, and one bundle.
    let _seam = TestDirectory::set(&scratch.0);
    assert_eq!(create(), refusal);
    let bundles = scratch.bundles();
    let [bundle] = &bundles[..] else {
        panic!("one bundle: {bundles:?}");
    };
    let name = bundle.file_name().unwrap().to_string_lossy().into_owned();
    assert!(name.starts_with("metal2vulkan-repro-"), "{name}");
    let mut names: Vec<String> = std::fs::read_dir(bundle)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "device.json",
            "failure.txt",
            "input.air",
            "repro-command.txt",
            "repro.sh"
        ]
    );
    assert_eq!(std::fs::read(bundle.join("input.air")).unwrap(), air);
    assert_eq!(
        std::fs::read_to_string(bundle.join("repro-command.txt")).unwrap(),
        "metal2vulkan input.air out.spv --device device.json --function ob3_stage_input\n"
    );
    let failure = std::fs::read_to_string(bundle.join("failure.txt")).unwrap();
    assert!(
        failure.contains("call: new_compute_pipeline_state"),
        "{failure}"
    );
    assert!(failure.contains("StageInputDescriptor"), "{failure}");
    assert!(!failure.contains("not replayed"), "{failure}");
    // The line names the bundle and the refusal.
    let lines = heard.lock().unwrap().clone();
    let [line] = &lines[..] else {
        panic!("one line: {lines:?}");
    };
    assert!(line.contains(&name), "{line}");
    assert!(
        line.contains("new_compute_pipeline_state refused"),
        "{line}"
    );

    // The bundle replays the refusal: its device is the runtime's compile device, and the request
    // its command names compiles to the same refusal there.
    let json = std::fs::read_to_string(bundle.join("device.json")).unwrap();
    let replayed = ReproDevice::from_json(&json).expect("device.json");
    let rebuilt = CompileDevice::new(replayed.description, replayed.policy).expect("the device");
    assert_eq!(
        rebuilt.profile_fingerprint(),
        device.compile_device().profile_fingerprint()
    );
    let input = std::fs::read(bundle.join("input.air")).unwrap();
    let parsed = CompileLibrary::parse(&input).expect("input.air");
    let kernel = FunctionDescriptor::new(parsed.function("ob3_stage_input").unwrap());
    let request = ComputePipelineDescriptor::new(StageSource::Function(kernel));
    let answer = rebuilt
        .compile_compute_pipeline(&request, &Budget::standard())
        .expect_err("the replay is refused");
    assert_eq!(&answer, compile_refusal);

    // A second identical refusal writes no bundle and says nothing.
    assert_eq!(create(), refusal);
    assert_eq!(scratch.bundles().len(), 1);
    assert_eq!(heard.lock().unwrap().len(), 1);

    // A lookup of a name the library lacks is Metal's own answer: no bundle.
    let absent = library.new_function_with_name("absent").err().unwrap();
    assert!(
        matches!(
            absent,
            Error::Compile(CompileError::InvalidDescriptor(
                DescriptorError::EntryNotFound { .. }
            ))
        ),
        "{absent:?}"
    );
    assert_eq!(scratch.bundles().len(), 1);
}
