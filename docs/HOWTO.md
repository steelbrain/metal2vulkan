# Integrating metal2vulkan

The checklist around the library: the tools each piece needs, caching, failures, isolating a
compile, host memory a driver refuses, and checking an integration. The API itself is
[API.md](API.md); the runtime's is [its README](../crates/metal2vulkan-runtime/README.md).

## 1. Tools

The libraries read AIR bitcode (a metallib, a bitcode wrapper or a raw module) and run no external
tool. The command line (feature `cli`) runs two, both optional:

- `llvm-as` (LLVM 21 or newer) to assemble an LLVM text input (`.ll`). Not needed for a metallib
  or bitcode.
- `spirv-val` (SPIRV-Tools) to check each module it writes. When it is missing the tool says so and
  still writes the modules, which the library has verified itself.

Put them on `PATH`, or name each one:

```sh
export METAL2VULKAN_LLVM_AS=/path/to/llvm-as
export METAL2VULKAN_SPIRV_VAL=/path/to/spirv-val
```

The runtime opens the Vulkan loader when it makes a device (or takes the consumer's
`vkGetInstanceProcAddr` when it adopts one). Nothing links against Vulkan at build time.

## 2. Compile

From the command line, see the [README](../README.md#command-line). From Rust, [API.md](API.md)
walks the flow in the order a consumer meets it:

1. [A device](API.md#1-a-device): describe the Vulkan device as created and choose a policy.
2. [A library](API.md#2-a-library): parse a metallib or AIR bitcode.
3. [A compute pipeline](API.md#3-a-compute-pipeline) or
   [a render pipeline](API.md#4-a-render-pipeline), under a `Budget`.
4. [The interface and the plan](API.md#5-the-interface-and-the-plan): the layout, what each slot
   binds, what a dispatch or draw needs.
5. [At bind time](API.md#6-at-bind-time): descriptor writes and push-constant bytes from what is
   bound.

Every `air.simd_*` operation keeps Metal's 32-lane simdgroup; there is no option to use the
driver's subgroup width instead.

A `half` computed in `float` rounds through the device's conversion to 16 bits and back, Metal's
`half(float)` under the float controls the entry declares, where the driver keeps that pair (ANV,
RADV). Elsewhere it rounds through the device's own conversion, or its `OpQuantizeToF16` alone
without `shaderFloat16` (which may flush a value below the smallest normal `half`), and the
pipeline records `NativeHalfConversion` or `NativeHalfQuantize`; not `NativeHalfConversion` on
MoltenVK on the Apple M3 Max's macOS 27 identity (`DriverDeviation::MetalHalfConversion`), where
every such `half` is a conversion of a `float` the function holds or of a texel, measured to be
Metal's. NVIDIA's driver (615.71.09) folds the pair even under the float controls, so its
rounding there is the device's conversion, or none; its pipelines still declare the float controls
and compute a `half` `+`, `-`, `*`, `/` and `sqrt` with the device's 16-bit instructions, as ANV's
and RADV's do. RADV's compiler fuses a `float` addition, subtraction, multiply or fused
multiply-add into the conversion of its result and rounds the exact result once. On a device
`Device::from_queries` made (`metal2vulkan-runtime`), which reads no driver identity, every device with
the float controls takes the pair, NVIDIA's included, and every pipeline that rounds through it
records `NativeHalfConversion`, as a driver's compiler may fold the pair or fuse the operation so.
A `half` read from memory or from its bits is the device's widening on every device (an
`OpFConvert` of the bits read as a 16-bit float with `shaderFloat16`, `GLSL.std.450
UnpackHalf2x16` without it), and a `half` stored or reinterpreted as bits, unless they are the
bits it was loaded from, the device's narrowing (an `OpFConvert` to 16 bits, `GLSL.std.450
PackHalf2x16` without `shaderFloat16`); the pipeline records `NativeNanEncoding` (a NaN's bits;
not on MoltenVK on the Apple M3 Max's macOS 27 identity, `DriverDeviation::MetalNanMoves`, where
the `half` moves through the registers alone or into a texture write's texel) and, where that conversion is neither under the
float controls nor measured to be Metal's, `NativeHalfConversion` (a denormal the device may
flush).

To read AIR without translating it (list a metallib's functions, decode a module into typed IR,
read its entry points' bindings, canonicalize an entry under function-constant values), depend on
[`metal2vulkan-air`](../crates/metal2vulkan-air). Use the same version as `metal2vulkan`, which
pins it exactly; the translator's input errors are that crate's types.

## 3. Cache

The translator keeps no cache; it gives you keys and encodings to keep one
([API.md section 7](API.md#7-caching)):

- A key is a SHA-256 of the inputs, `cache::SEMANTICS_VERSION` and the device's profile
  fingerprint, known before compiling.
- A decode checks the format and semantics versions, the key, the profile fingerprint and a
  checksum. Every `ArtifactError` is a cache miss: compile again and replace the entry.
- The versions move independently: the crate's (the API), `SEMANTICS_VERSION` (the output; a bump
  misses every cached artifact), `ARTIFACT_FORMAT_VERSION` (the encoding) and `RUNTIME_VERSION`
  (what the runtime does with an unchanged artifact).

### The runtime's pipeline cache

The runtime keeps its lowered functions and its pipeline cache in the directory
`DeviceOptions::pipeline_cache_dir` names, from one device to the next; with none, every boot
lowers every function and compiles every pipeline cold ([its README](../crates/metal2vulkan-runtime/README.md), "The cache
directory").

## 4. Handle failures

Every failure is a typed value; match variants, never messages
([API.md section 8](API.md#8-refusals-requested-inputs-and-precision)).

- A refusal names what would change the answer: the feature the device lacks, the pipeline input
  the compile needs, or a limit.
- A deadline or a cancellation (`Error::Budget`, `Error::Cancelled`) is transient: retry later.
  Every other error is the answer for that input and device and may be cached as one
  (`error.retry()`).
- Never use a module from a compile that returned an error.
- The command line prints `metal2vulkan: FALLBACK (<why>)`, exits 2, and writes a repro bundle
  (the input, the profile, the descriptor files, `repro.sh`, `failure.txt`) under
  `$TMPDIR/metal2vulkan-repros`, or `METAL2VULKAN_REPRO_DIR`. Its directory is named by its content
  (`metal2vulkan-repro-<16 hex digits>`), so the same failure writes it once; `repro.sh` replays the
  failure from the bundle alone, with `metal2vulkan` on `PATH`.

## 5. Bound a compile

Every compile runs in-process under the caller's `Budget`: a wall deadline (`Budget::standard()`
is 20 seconds), a `CancelToken` any thread may trigger, a deterministic work-unit cap and a byte
ledger charged before each input-sized allocation. A panic inside a compile becomes
`Error::Internal`. That is enough for a consumer that trusts its process.

When an abort, a stack overflow or a failed allocation in the compiler must not take the consumer
down, or a hard memory ceiling must hold, compile in a child process:

1. Spawn one worker per compile in its own process group, with the input in a file and the result
   written to a file. Send the library bytes, the entry and descriptor, and the device description
   and policy (with feature `serde` they serialize; equal description and policy give the same
   profile fingerprint and keys).
2. In the child, install a counting global allocator that refuses allocations above a budget, build
   the `Device`, and compile with the same `Budget`.
3. In the parent, poll the deadline, cancellation and the process group's resident memory. On a
   breach, kill the whole group and remove its scratch files. A result that arrives late is a
   timeout.
4. Load the result with `PipelineArtifact::from_bytes` against the parent's own `Device`, and check
   that its key is the one the parent computed (`Device::compute_pipeline_key`,
   `render_pipeline_key`).
5. Classify as in-process: a typed error is the answer; a deadline, cancellation or killed child is
   transient.

The validation crate's worker (`crates/metal2vulkan-validation/src/worker/`) implements this,
under 20 seconds and 500 MiB per attempt.

## 6. Host memory a driver refuses

The runtime imports a no-copy buffer's memory as a block, and the consumer places the buffer in
it: host memory by its address (`Device::import_host_memory`) or, on Linux, a dma-buf
(`Device::import_dma_buf`) or several (`Device::import_dma_bufs`). Which to try first, and what to
do where they refuse, is the consumer's; every refusal comes before any allocation is attempted,
with nothing imported ([its README](../crates/metal2vulkan-runtime/README.md), "Imported blocks").

## 7. Memory another API writes

A block the consumer exports (an opaque fd or a dma-buf on a Unix host, an NT handle on Windows)
is the memory another API writes in place, and an exported surface's planes are textures over its
own exported memory; the contract for each is in
[the runtime's README](../crates/metal2vulkan-runtime/README.md), "Exported blocks" and "Exported
surfaces".

## 8. Check the integration

`spirv-val` checks structural validity, not agreement with Metal:

```sh
spirv-val --target-env vulkan1.2 output.spv
```

`device.profile().spirv_validation()` names the environments and relaxations a module compiled
for the device is valid under; the command line runs each.

For a claim about behavior, follow [VALIDATION.md](VALIDATION.md): a synthetic regression test,
the golden output hashes where output must not move, and an authored case run on Metal and Vulkan
where it should.

A host note: RADV under Ubuntu 26.04's stock kernel 7.0.0-38 with linux-firmware 20260319
sometimes loses a GPU write into a newly made buffer under heavy multi-process GPU memory churn
(about 1 in 10^4 to 10^5, with no error); kernel 7.2.6 with linux-firmware 20260916 loses none.

A host note: on an RTX 3090 behind a Thunderbolt 3 bridge (NVIDIA 620.70.06, Linux 7.3), several
processes making devices with extra compute and video decode queues at once now and then meet an
error the kernel logs on one of their channels: Xid 69 (a 3D class error at
`SET_REPORT_SEMAPHORE_D`, offset `0x1b0c`, whose data word is a method header, so the GPU read the
stream one word short), Xid 32 (an invalid pushbuffer entry) or Xid 13 (a compute class exception).
Those processes' devices are lost, and a submission of another process's may never complete.

A host note: on a laptop whose discrete GPU the system suspends when idle (NVIDIA's runtime D3:
the PCI device's `power/runtime_status` reads `suspended`), the first driver call that needs the
GPU blocks the thread that makes it while the GPU resumes. On one RTX 5080 Laptop GPU (driver
615.71.09, Linux 7.2), each call made after 25 seconds idle with the GPU suspended: an empty
command buffer's `vkQueueSubmit` held its thread 982 ms, a host write to device-local host-visible
memory 986 ms, a device-local `vkAllocateMemory` 988 ms, and a command pool's reset and the
recording of 8192 dispatches 987 ms. A program that must not stall can keep the GPU from
suspending with GPU work of its own, at its own power cost, or make such calls from a thread it can
afford to block.

A host note: drop the runtime's devices before the process exits; what a device kept to the exit
meets on NVIDIA's driver is in [the runtime's README](../crates/metal2vulkan-runtime/README.md).
