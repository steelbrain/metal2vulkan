# metal2vulkan-rt

The Metal API on Vulkan, with one path for every device. It replaces `metal2vulkan-runtime` and
takes its name when it does; until then it is unpublished.

Its public API so far: the instance and the device, memory blocks and the buffers in them, the
device's counters, and `floor`, the check of a Vulkan device against the floor below.

## Instance and device

- `Instance::new` loads the Vulkan loader and creates the instance at the floor's version, with
  the floor's instance extensions and each extension the consumer names that the loader offers:
  the window system's, and any other it uses. `Instance::extensions` lists those enabled.
  `Instance::physical_devices` lists the physical devices in the loader's order.
- `Device::new` checks a physical device against the floor and makes one `VkDevice`. Every
  command queue, every guest process and the compositor share it. It takes the floor's queue
  family, the first that does graphics and compute: one queue for the runtime, and a second for
  presenting where the family has one. Where it has one queue, presenting shares the runtime's
  queue and its lock. `DeviceOptions` carries what the consumer adds: its `BlockMappings` hook,
  and the device extensions it uses beyond the floor's. Each `DeviceExtension` names one and,
  with `DeviceExtension::with_features`, its feature structure: the members the consumer needs,
  and a test of the device's answer. The device enables it where it lists the extension and its
  answer passes, and passes over it otherwise; `Device::extensions` lists those enabled, the
  floor's first. The consumer names an extension's dependencies too.
- `Device::raw` gives the instance, physical device, device and queue family handles, and
  `Device::with_present_queue` runs the consumer's presents on the presenting queue under its lock.
  The runtime keeps owning every handle.
- `Device::has_unified_memory` is true where Vulkan's device type is integrated or CPU.

### Teardown

A device is torn down when its last handle, and the last handle of every object made from it, is
dropped. Blocks the consumer holds do not keep it alive. Teardown joins the block-maker thread
and waits until the GPU is idle; then, on the dropping thread, it frees every block, calling
`BlockMappings::unmap` first for each block the consumer was handed, and destroys the `VkDevice`.
A physical device and a device each hold their instance. No thread of the runtime outlives its
device and nothing relies on a static destructor or an `atexit` handler, so the consumer drops
every handle before the process exits: a driver may unmap device memory inside `exit()`.

### Counters

`Device::counters` returns a handle that stays readable after the device is gone. `metrics()` are
cumulative counts (submissions, passes, barriers, bytes copied, pipelines created, blocks made);
`gauges()` are live counts per kind of resource, every one zero after teardown. They are relaxed
atomics, always on.

## Memory blocks and buffers

Memory comes in blocks the consumer takes and places buffers in, as a placement heap's buffers sit
in its memory.

- `Device::take_block(class, len)` gives a block of `MemoryClass::Private` (device-local, no host
  mapping) or `MemoryClass::Shared` (host-visible, host-coherent and host-cached, mapped for the
  block's life) memory, as a `BlockInfo`: its number, class, length, device address and, for
  Shared, its host address. One `VkBuffer` spans each block, with every buffer usage and a device
  address. Nothing in a block is exportable.
- Up to `BLOCK_LEN` (256 MiB) a block is pooled. The block-maker thread keeps one block of each
  class ready, made off every caller's path, so a take waits only when none is ready. Past
  `BLOCK_LEN` a block is dedicated: the length rounded up to 64 KiB, made in the call, up to the
  device's `maxMemoryAllocationSize` and `maxBufferSize` (`Limit::BlockLen` past them).
- Every block reads zero when it is handed out, new or taken again, as Metal's new buffers do. A
  Shared block is zeroed through its mapping, which also makes every page resident before anyone
  holds it; a Private block by a GPU fill.
- `Device::release_block` gives a block back. It takes effect once no buffer of the block remains.
  A pooled block is then zeroed and kept, with the same number, address and mapping, while its
  class has fewer than two idle blocks; the block-maker thread frees the rest. A dedicated block is
  always freed. Block numbers are never reused during the device's life.
- `Device::new_buffer(block, offset, length, zeroed)` makes a `Buffer`: a view of the range, with
  no Vulkan object of its own. Its `gpu_address` is the block's address plus the offset, and its
  `contents` the block's mapping plus the offset, or `None` in a Private block. `zeroed` zeroes the
  range before the call returns, as `newBufferWithLength:options:` does: through the mapping for
  Shared, by a GPU fill submitted and waited for in Private, which writes exactly the range. A
  buffer placed in a heap passes false, since Metal does not zero heap memory. A length of 0 and a
  range past the block are refused, as Metal answers nil for both.

The consumer's ranges follow Metal's retention: it reuses a range only once the work that named
the buffer there before has completed.

### The consumer's mappings

A consumer that maps blocks itself, into a guest for instance, passes `DeviceOptions::block_mappings`.
Before the runtime frees a block it handed out, it calls `BlockMappings::unmap(block)`, which
returns once every mapping the consumer made of it is gone and none of its threads touches the
block. The memory stays mapped and valid until then. It runs on the block-maker thread, or at
teardown on the dropping thread after the runtime's threads are joined, and it never calls into the
runtime. Blocks the consumer never took are freed without it.

## One path for every device

Every difference between devices comes from a Vulkan query: a feature, a property, a format, a
limit or a memory type. The crate holds:

- no vendor, device or driver identity;
- no environment gate;
- no runtime policy switch;
- no check that runs beside the program's work.

The test `tests::one_path` reads the crate's sources and fails on the first three.

## Floor

The runtime requires Vulkan 1.3 and every extension, feature, limit, queue family and memory type
below, of the instance and of every device. `floor::check` decides for one physical device: it
returns what the device offers above the floor (`floor::Capabilities`), or `floor::BelowFloor`, a
typed refusal that names every item the device lacks (`floor::Missing`: the version, an
extension, a feature, a limit, the queue family or a memory type). `floor::memory_types` makes the
last choice on the created device.

### Extensions

Presenting runs on the runtime's own device; nothing is exported or imported.

| Extension | Where | Used for |
|---|---|---|
| `VK_KHR_surface` | instance | presenting |
| `VK_KHR_swapchain` | device | presenting |
| `VK_KHR_workgroup_memory_explicit_layout` | device | threadgroup memory laid out as Metal lays it |

### Features

| Feature | Structure | Used for |
|---|---|---|
| `bufferDeviceAddress` | `VkPhysicalDeviceVulkan12Features` | a buffer's `gpuAddress` is its device address, used directly by shaders |
| `shaderInt64` | `VkPhysicalDeviceFeatures` | 64-bit addresses in shaders |
| `runtimeDescriptorArray` | `VkPhysicalDeviceVulkan12Features` | one descriptor array per image class: sampled textures, read-write textures, samplers |
| `descriptorBindingPartiallyBound` | `VkPhysicalDeviceVulkan12Features` | array slots that hold nothing |
| `descriptorBindingUpdateUnusedWhilePending` | `VkPhysicalDeviceVulkan12Features` | writing a new slot while work that uses the array runs |
| `descriptorBindingSampledImageUpdateAfterBind` | `VkPhysicalDeviceVulkan12Features` | the sampled-texture and sampler arrays |
| `descriptorBindingStorageImageUpdateAfterBind` | `VkPhysicalDeviceVulkan12Features` | the read-write texture array |
| `shaderSampledImageArrayDynamicIndexing` | `VkPhysicalDeviceFeatures` | indexing the sampled-texture and sampler arrays by a `gpuResourceID` |
| `shaderStorageImageArrayDynamicIndexing` | `VkPhysicalDeviceFeatures` | indexing the read-write texture array by a `gpuResourceID` |
| `shaderSampledImageArrayNonUniformIndexing` | `VkPhysicalDeviceVulkan12Features` | a `gpuResourceID` that differs between threads |
| `shaderStorageImageArrayNonUniformIndexing` | `VkPhysicalDeviceVulkan12Features` | a `gpuResourceID` that differs between threads |
| `shaderStorageImageWriteWithoutFormat` | `VkPhysicalDeviceFeatures` | writes to array elements of any pixel format |
| `shaderInt16` | `VkPhysicalDeviceFeatures` | `short` and `ushort` in shaders |
| `shaderInt8` | `VkPhysicalDeviceVulkan12Features` | `char` and `uchar` in shaders |
| `storageBuffer16BitAccess` | `VkPhysicalDeviceVulkan11Features` | 16-bit values in buffers |
| `storageBuffer8BitAccess` | `VkPhysicalDeviceVulkan12Features` | 8-bit values in buffers |
| `scalarBlockLayout` | `VkPhysicalDeviceVulkan12Features` | buffer and push-constant blocks laid out as Metal lays them, where std430 cannot |
| `uniformBufferStandardLayout` | `VkPhysicalDeviceVulkan12Features` | uniform blocks in std430 layout |
| `workgroupMemoryExplicitLayout` | `VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR` | threadgroup blocks with an explicit layout |
| `workgroupMemoryExplicitLayoutScalarBlockLayout` | `VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR` | threadgroup blocks in scalar layout |
| `workgroupMemoryExplicitLayout8BitAccess` | `VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR` | 8-bit values in threadgroup blocks |
| `workgroupMemoryExplicitLayout16BitAccess` | `VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR` | 16-bit values in threadgroup blocks |
| `vulkanMemoryModel` | `VkPhysicalDeviceVulkan12Features` | the translator's atomics and barriers |
| `vulkanMemoryModelDeviceScope` | `VkPhysicalDeviceVulkan12Features` | atomics and barriers of device scope |
| `timelineSemaphore` | `VkPhysicalDeviceVulkan12Features` | submission and completion |
| `synchronization2` | `VkPhysicalDeviceVulkan13Features` | barriers, layout changes and submission |
| `dynamicRendering` | `VkPhysicalDeviceVulkan13Features` | render passes |

### Limits

The descriptor limits are what the runtime's fixed arrays need: 65,536 sampled textures, 4,096
read-write textures and 4,000 samplers. 4,000 is Vulkan's minimum `maxSamplerAllocationCount`, so
every slot can hold a live sampler on every device. The compute limits let a threadgroup take
Metal's 1,024 threads in any shape up to 1,024 by 1,024 by 64. A Boolean limit's minimum is 1,
`VK_TRUE`; `timestampValidBits` is the queue family's.

| Limit | Minimum | Used for |
|---|---|---|
| `maxPerStageDescriptorUpdateAfterBindSampledImages` | `65536` | the sampled-texture array |
| `maxDescriptorSetUpdateAfterBindSampledImages` | `65536` | the sampled-texture array |
| `maxPerStageDescriptorUpdateAfterBindStorageImages` | `4096` | the read-write texture array |
| `maxDescriptorSetUpdateAfterBindStorageImages` | `4096` | the read-write texture array |
| `maxPerStageDescriptorUpdateAfterBindSamplers` | `4000` | the sampler array |
| `maxDescriptorSetUpdateAfterBindSamplers` | `4000` | the sampler array |
| `maxSamplerAllocationCount` | `4000` | a live sampler in every slot |
| `maxPerStageUpdateAfterBindResources` | `69632` | both texture arrays in one stage; a fragment stage needs its `maxColorAttachments` more |
| `maxUpdateAfterBindDescriptorsInAllPools` | `73632` | all three arrays |
| `maxPushConstantsSize` | `128` | per-command values; anything larger goes in a block whose address is pushed |
| `maxComputeWorkGroupInvocations` | `1024` | Metal's most threads per threadgroup |
| `maxComputeWorkGroupSize[0]` | `1024` | a threadgroup's width |
| `maxComputeWorkGroupSize[1]` | `1024` | a threadgroup's height |
| `maxComputeWorkGroupSize[2]` | `64` | a threadgroup's depth |
| `timestampComputeAndGraphics` | `1` | GPU times of command buffers |
| `timestampValidBits` | `36` | GPU times of command buffers |

### Queue family and memory types

The runtime submits to the first queue family with `VK_QUEUE_GRAPHICS_BIT` and
`VK_QUEUE_COMPUTE_BIT` (`floor::Capabilities::queue_family`). It allocates buffers from three memory
types, each the first with its flags that a buffer of `floor::BUFFER_USAGE` can use: every buffer
usage Vulkan 1.3 defines, and device addresses. Vulkan guarantees such a buffer a device-local
type, so of the three only Shared and the ring can be missing on a conforming device. Vulkan answers which memory types a buffer can use
only for a created device, so `floor::check` finds a type with each set of flags, and
`floor::memory_types` chooses on the device (`floor::MemoryTypes`).

| Memory type | Flags | Used for |
|---|---|---|
| Private | `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` | Private storage, which only the GPU reads and writes |
| Shared | `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`, `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`, `VK_MEMORY_PROPERTY_HOST_CACHED_BIT` | Shared storage, which the host maps and reads |
| ring | `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`, `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`, `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` | the parameter ring, which the host writes with no flush and the GPU reads |

### Chosen by a query

Four features choose between two forms by the device's answer, and no other feature does.
`floor::Capabilities` holds each answer, and the features to enable: the floor's, and each
optional one the device has.

| Optional feature | Structure | Used for |
|---|---|---|
| `shaderFloat16` | `VkPhysicalDeviceVulkan12Features` | `half` arithmetic in 16 bits |
| `shaderStorageImageReadWithoutFormat` | `VkPhysicalDeviceFeatures` | reads of read-write textures of any format; where it is false, each format's `VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT` answers |
| `shaderBufferInt64Atomics` | `VkPhysicalDeviceVulkan12Features` | 64-bit atomics on buffers |
| `shaderSharedInt64Atomics` | `VkPhysicalDeviceVulkan12Features` | 64-bit atomics on threadgroup memory |
| `shaderFloat64` | `VkPhysicalDeviceFeatures` | 64-bit floats |

The test `tests::floor` holds these tables equal to the code, checks that each missing item is
named, and, under `--cfg m2v_gpu` or `--cfg m2v_cpu_icd`, checks every device the machine's
Vulkan loader lists.
