# metal2vulkan

[![License: GPL-2.0-or-later OR LGPL-3.0-or-later](https://img.shields.io/badge/License-GPL%202.0%20or%20later%20OR%20LGPL%203.0%20or%20later-blue.svg)](LICENSE)

metal2vulkan runs Metal shaders and Metal programs on Vulkan. It reads **Metal AIR** (a metallib,
or AIR bitcode) and compiles it into Vulkan pipelines, and it offers the Metal API itself on top
of a Vulkan device.

> **Beta.** The API, the command line, the pipeline JSON and the emitted SPIR-V may change during
> the `0.x` series. Version your cached outputs.

## The crates

| Crate | Directory | What it is |
|---|---|---|
| `metal2vulkan` | [`crates/metal2vulkan`](crates/metal2vulkan) | The translator. A Metal-shaped API (device, library, pipeline descriptors, binding plans) that compiles compute, render, tessellation, tile and mesh pipelines for a described Vulkan device: SPIR-V per stage, one pipeline layout, fixed-function state, and per-draw descriptor writes and push constants. The `cli` feature builds the `metal2vulkan` command. |
| `metal2vulkan-runtime` | [`crates/metal2vulkan-runtime`](crates/metal2vulkan-runtime) | The Metal API on Vulkan: devices, command queues and buffers, encoders, buffers, textures, heaps, pipeline states and events, run on a Vulkan device through `ash`. It uses only the translator's public API. |
| `metal2vulkan-air` | [`crates/metal2vulkan-air`](crates/metal2vulkan-air) | The AIR front end: the metallib container, the bitcode reader, a typed IR, AIR's metadata ABI, intrinsics and canonicalization. For tools that read AIR without translating it. |
| `metal2vulkan-validation` | [`crates/metal2vulkan-validation`](crates/metal2vulkan-validation) | Not published. Authored cases run on Metal and on Vulkan and compared, and bounded audits over a private corpus. |
| `metal2vulkan-devtools` | [`crates/metal2vulkan-devtools`](crates/metal2vulkan-devtools) | Not published. The external-tool runner and fixture assemblers the tests share. |

The three published crates share one version and are released together; `metal2vulkan` pins
`metal2vulkan-air` exactly.

What the translator promises:

- **Typed refusals.** Input the device cannot run exactly fails with an error that names the Metal
  feature and what would make it work. It never returns a module that computes something else.
- **Recorded differences.** Where a pipeline gives a near-equivalent of Metal's answer, its
  artifact records a precision note, and the runtime announces it once with a stable code.
- **Fast by default.** Off Apple's GPUs, choices that trade exactness for speed take the device's
  native form; each has an opt-in exact switch ([docs/FAST-PATHS.md](docs/FAST-PATHS.md)).
- **Bounded work.** Every compile runs under a deadline, a cancellation token and deterministic
  work and memory caps.

## Install

0.2.0 is not on crates.io yet (crates.io has 0.1.0, an unrelated earlier design). Depend on a git
revision:

```toml
[dependencies]
metal2vulkan = { git = "https://github.com/steelbrain/metal2vulkan", rev = "<sha>" }
# For the Metal API on Vulkan:
metal2vulkan-runtime = { git = "https://github.com/steelbrain/metal2vulkan", rev = "<sha>" }
```

Features of `metal2vulkan`, none needed by the library:

- `serde`: serialization of the API's plain-data values.
- `ash`: conversions to and from `ash::vk`, and `DeviceDescription::query`, which reads a physical
  device.
- `cli`: the `metal2vulkan` command (enables `serde`).

Requirements: Rust 1.98 or newer. The libraries run no external tool. The runtime opens the Vulkan
loader at run time and needs a Vulkan 1.2 device.

## Command line

```sh
cargo install --git https://github.com/steelbrain/metal2vulkan metal2vulkan --features cli
```

The command compiles one pipeline of a metallib, an AIR bitcode module or an LLVM text module
(`.ll`) for the device a Khronos profiles JSON describes, and writes its SPIR-V:

```sh
# A kernel: one module.
metal2vulkan library.metallib kernel.spv --profile device.json --function my_kernel

# A render pipeline: one module per stage (pipeline.vert.spv, pipeline.frag.spv).
metal2vulkan shaders.air pipeline.spv --profile device.json \
  --function vertex_main --fragment fragment_main --vertex-descriptor vertex.json \
  --color-format 0:BGRA8Unorm --depth-format Depth32Float

# The pipeline's layout, interface, dependencies and stages as JSON.
metal2vulkan kernel.ll kernel.spv --profile device.json --emit-meta kernel.json
```

- `--profile` is required: the crate ships no device description. Any Khronos profiles JSON works;
  [`crates/metal2vulkan-validation/profiles/`](crates/metal2vulkan-validation/profiles) has
  examples, from a bare Vulkan 1.2 device to desktop GPUs and MoltenVK.
- Without `--function` it compiles the library's only kernel or vertex function. A kernel with
  `--color-format` compiles as a tile pipeline. A post-tessellation vertex function adds
  `.tesc.spv` and `.tese.spv` modules on a device with `tessellationShader`; on one without, it is
  the `.vert.spv` module itself, which draws the tessellator's triangles.
- A compile that needs a pipeline input fails naming it and the option that supplies it:
  `--constant ID:TYPE:VALUE` for a function constant, `--stage-input FILE` or
  `--vertex-descriptor FILE` for a stage-input or vertex descriptor (as JSON).
- A `.ll` input is assembled with `llvm-as` (LLVM 21 or newer). Each module written is checked with
  `spirv-val` when it is found, under the environments the library states for the device;
  `--no-spirv-val` skips it. Both tools come from `PATH`, or from `METAL2VULKAN_LLVM_AS` and
  `METAL2VULKAN_SPIRV_VAL`.
- Success prints a line with `PASS` and exits 0. Failure prints `metal2vulkan: FALLBACK (<why>)`,
  writes nothing, and exits 2; a failure after the inputs are read also writes a repro bundle
  under `$TMPDIR/metal2vulkan-repros` (or `METAL2VULKAN_REPRO_DIR`).
- `metal2vulkan --help` lists every option.

## Library

The flow follows Metal's object lifecycle: describe the device, parse a library, compile a
pipeline, then plan and instantiate each dispatch or draw.

```rust,ignore
use metal2vulkan::device::{Device, DeviceDescription, DevicePolicy};
use metal2vulkan::library::Library;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};
use metal2vulkan::Budget;

let description = DeviceDescription::from_khronos_json(&profile_json, None)?;
let device = Device::new(description, DevicePolicy::default())?;
let library = Library::parse(&metallib_bytes)?;
let kernel = FunctionDescriptor::new(library.function("my_kernel")?);
let descriptor = ComputePipelineDescriptor::new(StageSource::Function(kernel));
let pipeline = device.compile_compute_pipeline(&descriptor, &Budget::standard())?;
// pipeline.stage.spirv, pipeline.layout, pipeline.interface: create the Vulkan objects from these,
// then pipeline.plan(...) and plan.instantiate_dispatch(...) per dispatch.
```

[docs/API.md](docs/API.md) walks the whole flow on a runnable example,
[`examples/api_preview.rs`](crates/metal2vulkan/examples/api_preview.rs):

```sh
cargo run -p metal2vulkan --example api_preview   # needs llvm-as on PATH
```

For the runtime, start at [its README](crates/metal2vulkan-runtime/README.md).

## Documentation

| Document | Read it when |
|---|---|
| [docs/API.md](docs/API.md) | You use the translator's API: devices, libraries, pipelines, binding, caching, errors; and the runtime's degrade notices |
| [Runtime README](crates/metal2vulkan-runtime/README.md) | You use the Metal API on Vulkan |
| [docs/HOWTO.md](docs/HOWTO.md) | You integrate: tools, caching, failures, isolation, the runtime's shader cache |
| [docs/RUNTIME-PORTING.md](docs/RUNTIME-PORTING.md) | You port Metal calls (Objective-C or metal-rs) to the runtime |
| [docs/FAST-PATHS.md](docs/FAST-PATHS.md) | You need every speed-for-accuracy default and its exact switch |
| [docs/METAL-COVERAGE.md](docs/METAL-COVERAGE.md) | You ask whether a Metal API item is covered |
| [docs/REFUSALS.md](docs/REFUSALS.md) | You ask what Metal accepts that the code refuses |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | You change the translator |
| [docs/VALIDATION.md](docs/VALIDATION.md) | You check behavior against Metal |
| [CONTRIBUTING.md](CONTRIBUTING.md) | You build and test the repository |

The rustdoc of each crate (`cargo doc --workspace --all-features --open`) documents every public
item; each runtime item names its Metal counterpart.

## Shaders in this repository

The repository ships no third-party shaders. Tests use synthetic fixtures written for it. Passing
`spirv-val` shows a module is valid, not that it computes what Metal computes; that claim needs the
authored validation cases ([docs/VALIDATION.md](docs/VALIDATION.md)).

## License

metal2vulkan is dual-licensed: you can redistribute it and/or modify it under the terms of either
of these licenses, at your choice ([LICENSE](LICENSE)):

- the [GNU General Public License, version 2](LICENSE-GPL-2.0), or (at your option) any later
  version;
- the [GNU Lesser General Public License, version 3](LICENSE-LGPL-3.0), or (at your option) any
  later version. It incorporates the terms of the
  [GNU General Public License, version 3](LICENSE-GPL-3.0).

As an SPDX expression: `GPL-2.0-or-later OR LGPL-3.0-or-later`. Each published crate carries the
notice and the three texts.

Metal is a trademark of Apple Inc. metal2vulkan is an independent project and is not affiliated
with, sponsored by, or endorsed by Apple Inc.
uncommitted work
