metal2vulkan: the API (reference)

Metal AIR in; Vulkan SPIR-V, pipeline layouts, and per-draw binding instructions out, through an
API shaped like Metal's object lifecycle. The consumer never parses AIR, never parses or patches
SPIR-V, never infers binding numbers from Metal indices, and never classifies errors by message
text.

This reference shows exactly the API's frozen surface. A walkthrough with a runnable example is
`docs/API.md` in the repository (`examples/api_preview.rs`).

# Stability

- **Public in every build.** The API needs no feature.
- **Names and shapes are frozen.** The surface is the items named at the roots of
  [`error`], [`device`], [`library`], [`reflection`], [`pipeline`], [`bind`], [`cache`],
  [`metal`] (with its mapping modules `format`, `sampler`, `state`, and `vertex`), [`vk`], and
  `ash_interop` (feature `ash`), plus [`Budget`] and [`CancelToken`] here. `tests/api_surface.rs` pins every
  name, its kind, its construction class, and `Send + Sync`, and fails on any change until the
  table changes with it. Method signatures are not pinned by that table; the tests drive
  the consumer flow through them.
- **Changes are additive.** Every public struct and enum is `#[non_exhaustive]` or opaque (a
  private field), so fields and variants can be added in a minor version: construct values with
  `new`, `Default`, or their constants and assign fields, and match enums with a wildcard arm.
  The exceptions are exhaustive by design, because their fields are their identity or are fixed
  by Vulkan: [`pipeline::SlotRef`], [`pipeline::ResourceSlot`], [`pipeline::AirUserKey`],
  [`pipeline::VaryingKey`], [`device::ApiVersion`], [`device::OpBits`], [`bind::StatusBits`],
  [`bind::ExecutionStatus`], [`vk::ComponentMapping`], and [`vk::SpecializationMapEntry`].
- **Semver.** The crate is `0.x`: a breaking change bumps the minor version.
- **Everything else is internal.** An item this reference does not show is not API: every
  submodule of `library`, `pipeline`, `bind`, and `device` (name a type by its root path,
  `pipeline::SlotRef`, not `pipeline::sym::SlotRef`), and the translator core (`air`, `budget`,
  `graph`, `lower`, `memory`, `spirv`, `spv`, `structurize`). They are crate-private, and no
  feature or cfg makes them reachable: the repository's own validation and dev tools use this
  public API too.

# The flow

1. **A device.** Describe the Vulkan device as created ([`device::DeviceDescription`]) and pick a
   [`device::DevicePolicy`]; [`device::Device::new`] derives the profile every compile decides by.
2. **A library.** [`library::Library::parse`] reads a metallib or AIR bitcode;
   [`library::Library::function`] finds a function, and [`library::Function::reflection`]
   answers what its arguments bind, in Metal's shape ([`reflection`]).
3. **A pipeline.** Describe it the way Metal does ([`pipeline::ComputePipelineDescriptor`],
   [`pipeline::RenderPipelineDescriptor`]) and compile it with
   [`device::Device::compile_compute_pipeline`] or [`device::Device::compile_render_pipeline`]:
   SPIR-V per stage, one pipeline layout, vertex input and fixed-function state, and the
   [`pipeline::PipelineInterface`].
4. **A plan.** Describe what is bound, keyed by Metal slot ([`bind::BindState`]), and plan the
   pipeline for its shape ([`bind::BindingPlan`]): the descriptor writes, vertex buffers, and
   resource uses. Memoize it per shape.
5. **Commands.** Instantiate the plan per dispatch or draw ([`bind::DispatchCommands`],
   [`bind::DrawCommands`]): descriptor ranges, push-constant bytes, table uploads, and the Vulkan
   dispatches or draw to record. The binder never calls Vulkan.
6. **Caching.** Keys are known before compiling ([`cache::PipelineKey`],
   [`library::FunctionCompileKey`]), and artifacts encode and decode through
   [`cache::PipelineArtifact`]. The crate keeps no cache.

Failures are typed ([`error::Error`], [`error::BindError`]): a refusal
([`error::Unsupported`]) names the Metal feature and what would make it translatable, and a
pipeline input the compile needs ([`error::RuntimeInput`]) is asked for, never guessed. Where an
output accepts a documented difference from Metal, its stage says so
([`pipeline::PrecisionNote`]).

Every compile call takes a [`Budget`]: a deadline and a [`CancelToken`]. Missing the deadline
or being cancelled is transient; deterministic caps are permanent for the input.

# Features

The library needs none of them.

- `ash`: `From` conversions between [`vk`] values and `ash::vk`, and
  `DeviceDescription::query`, which reads a physical device into a description (`ash_interop`).
- `serde`: `Serialize` and `Deserialize` on the plain-data values: device descriptions and
  policies, bind states and shapes, plans and commands, pipeline interfaces and layouts, and the
  Metal and Vulkan enums. Not on artifacts, which have their own encoding
  ([`cache::PipelineArtifact`]), nor on descriptors that hold a library.
- `cli`: the `metal2vulkan` command-line tool (`cargo install metal2vulkan --features cli`), which
  compiles one pipeline with this API and writes its SPIR-V. It enables `serde` and adds nothing
  else to the library.

The Metal-like runtime is its own crate, `metal2vulkan-rt`, built on this API alone.
