//! A Core Image kernel library through the public API.
//!
//! `metal -fcikernel` compiles Core Image kernels (`!air.ci`) into one module, and
//! `metallib -cikernel` writes a metallib of file type 1 that lists each kernel as an extern
//! function (`TYPE` 5) over that module; `metal2vulkan-devtools` writes the container here in that
//! layout. What Metal answers was measured with kernels written from scratch (a probe, an M3 Max
//! under macOS 27.0.1): `newLibraryWithData:error:` takes the library as
//! `MTLLibraryTypeExecutable`, `newFunctionWithName:` makes each kernel with `functionType` 4, and
//! a compute pipeline made from one fails ("Function tint was not found in the module"), as one
//! made from a name the library lacks does. No GPU is used.

use metal2vulkan_devtools::metallib::{core_image_metallib, wrap, MtlbFunction};

use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{DescriptorError, DiagName, Error};
use crate::library::{FunctionStage, Library, LibraryType};
use crate::metal::FunctionType;
use crate::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};

fn device() -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// A library of the extern functions `tint` and `shift`, over one module.
fn core_image_library() -> Vec<u8> {
    // Any module: nothing here reads the kernels' bitcode.
    let module = wrap(b"BC\xc0\xde\x35\x14\x00\x00\x05\x00\x00\x00");
    let kernel = |name| MtlbFunction {
        name,
        type_code: 5,
        bitcode: module.clone(),
        constants: Vec::new(),
        public: Vec::new(),
        tags: Vec::new(),
    };
    core_image_metallib(&[kernel("tint"), kernel("shift")])
}

/// The library is executable and its functions are extern functions over its one module, each
/// `functionType` 4; neither compiles as a stage.
#[test]
fn a_core_image_library_holds_extern_functions() {
    let lib = Library::parse(&core_image_library()).expect("library");
    assert_eq!(lib.library_type(), LibraryType::Executable);
    let stages: Vec<(&str, FunctionStage)> = lib.functions().map(|f| (f.name, f.stage)).collect();
    assert_eq!(
        stages,
        [
            ("tint", FunctionStage::Extern),
            ("shift", FunctionStage::Extern)
        ]
    );
    let hashes: Vec<_> = lib.functions().map(|f| f.air_hash).collect();
    assert_eq!(hashes[0], hashes[1], "one module, hashed once");
    for name in ["tint", "shift"] {
        let f = lib.function(name).expect("function");
        let properties = f.properties(&Budget::standard()).expect("properties");
        assert_eq!(properties.function_type(), FunctionType::Extern, "{name}");
        let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
        assert_eq!(
            device()
                .compile_compute_pipeline(&d, &Budget::standard())
                .map(|_| ()),
            Err(Error::InvalidDescriptor(DescriptorError::EntryNotFound {
                requested: DiagName::new(name),
                available: 0,
            })),
            "{name}"
        );
    }
}
