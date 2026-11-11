#!/usr/bin/env python3
"""Regenerate `crates/metal2vulkan-validation/profiles/*.json` from pinned public sources.

Every committed device profile is a Khronos profiles-schema document with exactly one profile,
named `VP_M2V_<file stem>`, plus a top-level `provenance` object. Sources are pinned by URL and
SHA-256; nothing is invented. A field the source does not report is absent from the output, and a
source field this converter cannot represent is listed under `provenance.unmapped` instead of
being guessed.

Source kinds:

* `gpuinfo`: a raw vulkan.gpuinfo.org report (`api/v2/getreport.php`), converted to the profiles
  schema with the pinned Vulkan-Headers `vk.xml` for member types, enum names, and flag bits.
* `khronos`: one profile and the capability blocks it references, extracted verbatim from a pinned
  KhronosGroup/Vulkan-Profiles file. A profile that states no formats of its own defers to its
  core version's Required Format Support tables, so it gets the same format blocks as `derived`
  (Vulkan-Profiles does not encode those tables); one that states formats is kept as is.
* `derived`: the Vulkan 1.2 floor, built from the LunarG 1.2 minimum-requirements
  profile, with features reduced to the unconditional `VK_VERSION_1_0..1_2` requirements of
  `vk.xml`, plus the Vulkan 1.2 specification's Required Format Support tables
  (`required_formats.py`) and the weakest permitted value of each property the
  product reads, and VK_LAYER_KHRONOS_profiles sets, that the base profile leaves without a
  requirement (`UNREQUIRED_PROPERTIES`).
* `vulkaninfo`: a local `vulkaninfo --json` capture. The capture opens a device, so it is never
  taken by this script; pass it with `--capture NAME=PATH`. `vulkaninfo --json` never writes
  `VkFormatProperties3`, so a source that declares `formats` also takes that device's
  `corpus-vulkan --capture-formats` supplement (`--formats NAME=PATH`): the
  conversion checks that both files describe one device and that each format's low 32 bits are
  the capture's `VkFormatProperties`, then writes every format's `VkFormatProperties3`. Without a
  capture the committed conversion is kept.

Whatever the kind, the last step withholds the members that identify the reporting host or
driver build (`HOST_IDENTITY`: UUIDs, LUIDs, PCI address, DRM nodes) and names them in
`provenance.withheld`. A document states no deviation from the specifications: the product
takes a driver's measured deviations from its release
(`crates/metal2vulkan/src/device/driver_facts.rs`).

Downloads are cached under `.cache/device-profiles/` (gitignored) and verified before use.
"""
import argparse
import copy
import hashlib
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import required_formats

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "crates" / "metal2vulkan-validation" / "profiles"
CACHE_DIR = REPO / ".cache" / "device-profiles"
SCHEMA = "https://schema.khronos.org/vulkan/profiles-0.8-latest.json#"
CONVERTER = "scripts/regen-device-profiles/regen_device_profiles.py"
# Date recorded in generated profile history. Bump it when regenerated content changes, so that
# rerunning the script over unchanged sources is byte-identical.
GENERATED = "2026-09-29"

RAW = "https://raw.githubusercontent.com/KhronosGroup/"
REGISTRY = {
    "url": RAW + "Vulkan-Headers/v1.4.364/registry/vk.xml",
    "sha256": "4cfe3c137f3a95c15275b1cfda0aee407dbf131de2005a3552aa41c49f895c89",
}
PROFILES_TAG = RAW + "Vulkan-Profiles/vulkan-sdk-1.4.357.0/profiles/"
MINIMUM_REQUIREMENTS = {
    "url": PROFILES_TAG + "LunarG/VP_LUNARG_minimum_requirements.json",
    "sha256": "0feaf3db72f398717f1de96d138896b6756934ef8d178d934147de0cd6691e72",
    "profile": "VP_LUNARG_minimum_requirements_1_2",
}
GPUINFO_REPORT = "https://vulkan.gpuinfo.org/api/v2/getreport.php?id={}"
GPUINFO_DISPLAY = "https://vulkan.gpuinfo.org/displayreport.php?id={}"

# Properties the product reads at Vulkan 1.2 that the base profile leaves unstated because the
# specification sets no requirement on them, at the weakest value it permits: the value the
# product's reader takes for an absent member. A derived profile states them so that
# VK_LAYER_KHRONOS_profiles, which keeps the device's own value for a member a profile omits,
# reports the profile as the product reads it. The float-controls independence
# members are not among them: vk.xml gives them `limittype="exact"`, so the layer never sets them
# and keeps the device's value whatever a profile states.
UNREQUIRED_PROPERTIES_BLOCK = "m2v_unrequired_properties"
UNREQUIRED_PROPERTIES = {
    "VkPhysicalDeviceProperties": {"limits": {"strictLines": False}},
    "VkPhysicalDeviceVulkan11Properties": {"subgroupQuadOperationsInAllStages": False},
}

# The block that completes a requirements document (`khronos`) for VK_LAYER_KHRONOS_profiles.
# A Vulkan-Profiles requirements profile states only what it requires. The product's
# reader takes a member it omits as its version's guarantee, or as none, and a feature its version
# requires as present; the layer, restricting a device to it, reports the device's own value for an
# omitted member of a structure the context queries, and an omitted feature as absent. So a run
# restricted to the profile would find the device's values, on every device. The block states
# those members at the values the product reads, so the layer reports the profile as the product
# reads it, and the product's reading does not change. A source opts in with `layer_presentation`;
# a requirements document without the block is one validation refuses to restrict a device to.
LAYER_PRESENTATION_BLOCK = "m2v_layer_presentation"

# The members the product reads (src/device/khronos.rs) of the core device-property structures, by
# vk.xml structure, with the minor version that made each core, at the value its reader takes for
# an absent member: the version's guarantee (`required_limits`: Vulkan's Required Limits as
# VP_LUNARG_minimum_requirements_1_2 transcribes them, and maxBufferSize from Vulkan 1.3), else
# none (0, false, an empty list). A context at a profile's version queries each structure of that
# version or earlier, so a completed requirements document states each one it omits. Not the
# device's identity (`noauto`) nor denormBehaviorIndependence and roundingModeIndependence
# (`exact`): the layer keeps the device's value of each, and validation bounds them.
# VkPhysicalDeviceVulkan14Properties is not modelled: completing a Vulkan 1.4 document fails.
FOUR_SAMPLES = ["VK_SAMPLE_COUNT_1_BIT", "VK_SAMPLE_COUNT_4_BIT"]
FLOAT_CONTROLS = [f"{kind}Float{bits}" for kind in (
    "shaderSignedZeroInfNanPreserve", "shaderDenormPreserve", "shaderDenormFlushToZero",
    "shaderRoundingModeRTE", "shaderRoundingModeRTZ") for bits in (16, 32, 64)]
CORE_PROPERTIES = {
    "VkPhysicalDeviceProperties": (0, {"limits": {
        "maxImageDimension1D": 4096,
        "maxImageDimension2D": 4096,
        "maxImageDimension3D": 256,
        "maxImageDimensionCube": 4096,
        "maxImageArrayLayers": 256,
        "maxTexelBufferElements": 65536,
        "maxUniformBufferRange": 16384,
        "maxStorageBufferRange": 134217728,
        "maxPushConstantsSize": 128,
        "maxBoundDescriptorSets": 4,
        "maxPerStageDescriptorSamplers": 16,
        "maxPerStageDescriptorUniformBuffers": 12,
        "maxPerStageDescriptorStorageBuffers": 4,
        "maxPerStageDescriptorSampledImages": 16,
        "maxPerStageDescriptorStorageImages": 4,
        "maxPerStageDescriptorInputAttachments": 4,
        "maxPerStageResources": 128,
        "maxDescriptorSetSamplers": 96,
        "maxDescriptorSetUniformBuffers": 72,
        "maxDescriptorSetStorageBuffers": 24,
        "maxDescriptorSetSampledImages": 96,
        "maxDescriptorSetStorageImages": 24,
        "maxDescriptorSetInputAttachments": 4,
        "maxVertexInputAttributes": 16,
        "maxVertexInputBindings": 16,
        "maxVertexInputAttributeOffset": 2047,
        "maxVertexInputBindingStride": 2048,
        "maxVertexOutputComponents": 64,
        "maxTessellationGenerationLevel": 0,
        "maxTessellationEvaluationOutputComponents": 0,
        # 0 without `geometryShader`, which no core version requires.
        "maxGeometryInputComponents": 0,
        "maxGeometryOutputComponents": 0,
        "maxGeometryTotalOutputComponents": 0,
        "maxFragmentInputComponents": 64,
        "maxFragmentOutputAttachments": 4,
        "maxFragmentDualSrcAttachments": 0,
        "maxFragmentCombinedOutputResources": 4,
        "maxComputeSharedMemorySize": 16384,
        "maxComputeWorkGroupCount": [65535, 65535, 65535],
        "maxComputeWorkGroupInvocations": 128,
        "maxComputeWorkGroupSize": [128, 128, 64],
        "maxSamplerLodBias": 2.0,
        "maxSamplerAnisotropy": 1.0,
        "minTexelBufferOffsetAlignment": 256,
        "minUniformBufferOffsetAlignment": 256,
        "minStorageBufferOffsetAlignment": 256,
        "minTexelOffset": -8,
        "maxTexelOffset": 7,
        "minTexelGatherOffset": -8,
        "maxTexelGatherOffset": 7,
        "minInterpolationOffset": 0.0,
        "maxInterpolationOffset": 0.0,
        "framebufferColorSampleCounts": FOUR_SAMPLES,
        "framebufferDepthSampleCounts": FOUR_SAMPLES,
        "framebufferStencilSampleCounts": FOUR_SAMPLES,
        "framebufferNoAttachmentsSampleCounts": FOUR_SAMPLES,
        "maxColorAttachments": 4,
        "storageImageSampleCounts": ["VK_SAMPLE_COUNT_1_BIT"],
        "maxClipDistances": 0,
        "maxCullDistances": 0,
        "maxCombinedClipAndCullDistances": 0,
        "maxFramebufferWidth": 4096,
        "maxFramebufferHeight": 4096,
        "maxFramebufferLayers": 256,
        # 2^32 - 1 where the profile states fullDrawIndexUint32 (FEATURE_EXACT_PROPERTIES).
        "maxDrawIndexedIndexValue": 16777215,
        "maxTessellationPatchSize": 0,
        # Without `largePoints` a device's points are 1.0 (`pointSizeRange` (1.0, 1.0)), the
        # bound the reader takes for an absent range.
        "pointSizeRange": [1.0, 1.0],
        # Without `wideLines` a device's lines are 1.0 (`lineWidthRange` (1.0, 1.0),
        # `lineWidthGranularity` 0.0); subPixelPrecisionBits' guarantee is 4; strictLines none.
        "subPixelPrecisionBits": 4,
        "lineWidthRange": [1.0, 1.0],
        "lineWidthGranularity": 0.0,
        "strictLines": False,
    }}),
    "VkPhysicalDeviceVulkan11Properties": (1, {
        "subgroupSize": 1,
        "subgroupSupportedStages": ["VK_SHADER_STAGE_COMPUTE_BIT"],
        "subgroupSupportedOperations": ["VK_SUBGROUP_FEATURE_BASIC_BIT"],
        "subgroupQuadOperationsInAllStages": False,
        "maxMultiviewViewCount": 0,
        "maxMultiviewInstanceIndex": 0,
        "maxMemoryAllocationSize": 1073741824,
    }),
    "VkPhysicalDeviceVulkan12Properties": (2, {
        **{member: False for member in FLOAT_CONTROLS},
        "maxPerStageDescriptorUpdateAfterBindSamplers": 0,
        "maxPerStageDescriptorUpdateAfterBindSampledImages": 0,
        "maxPerStageDescriptorUpdateAfterBindStorageImages": 0,
        "maxPerStageUpdateAfterBindResources": 0,
        "maxDescriptorSetUpdateAfterBindSamplers": 0,
        "maxDescriptorSetUpdateAfterBindStorageBuffers": 0,
        "maxDescriptorSetUpdateAfterBindSampledImages": 0,
        "maxDescriptorSetUpdateAfterBindStorageImages": 0,
        "robustBufferAccessUpdateAfterBind": False,
        "filterMinmaxSingleComponentFormats": False,
        "framebufferIntegerColorSampleCounts": ["VK_SAMPLE_COUNT_1_BIT"],
        # VkPhysicalDeviceDepthStencilResolveProperties: "VK_RESOLVE_MODE_SAMPLE_ZERO_BIT must be
        # included in the set" of each.
        "supportedDepthResolveModes": ["VK_RESOLVE_MODE_SAMPLE_ZERO_BIT"],
        "supportedStencilResolveModes": ["VK_RESOLVE_MODE_SAMPLE_ZERO_BIT"],
        "independentResolveNone": False,
        "independentResolve": False,
    }),
    "VkPhysicalDeviceVulkan13Properties": (3, {
        "minSubgroupSize": 0,
        "maxSubgroupSize": 0,
        "maxComputeWorkgroupSubgroups": 0,
        "requiredSubgroupSizeStages": [],
        "maxBufferSize": 1073741824,
    }),
}

# The members the product reads (src/device/khronos.rs `PROPERTY_STRUCTS`) of the extension
# device-property structures beyond core Vulkan 1.2, by vk.xml name, at the value its reader takes
# for an absent member. A completed requirements document that lists the extension states each one
# it omits.
EXTENSION_PROPERTIES = {
    "VkPhysicalDeviceSubgroupSizeControlProperties": {
        "minSubgroupSize": 0,
        "maxSubgroupSize": 0,
        "maxComputeWorkgroupSubgroups": 0,
        "requiredSubgroupSizeStages": [],
    },
    "VkPhysicalDeviceVertexAttributeDivisorPropertiesEXT": {"maxVertexAttribDivisor": 0},
    "VkPhysicalDeviceMaintenance5Properties": {"nonStrictWideLinesUseParallelogram": False},
    "VkPhysicalDeviceSampleLocationsPropertiesEXT": {
        "sampleLocationSampleCounts": [],
        "variableSampleLocations": False,
    },
}

# Core device-property members whose value a feature the profile states makes exact, by vk.xml
# structure and member, with the feature and that value. Where the profile states the feature
# true and omits the member, its presentation states this value, as the product's reader takes it
# (src/device/khronos.rs `Merged::describe`), in place of CORE_PROPERTIES' guarantee.
# "If this feature is supported, maxDrawIndexedIndexValue must be 2^32-1" (fullDrawIndexUint32).
FEATURE_EXACT_PROPERTIES = {
    ("VkPhysicalDeviceProperties", "maxDrawIndexedIndexValue"):
        ("fullDrawIndexUint32", 4294967295),
}

# Members a derived profile's base (VP_LUNARG_minimum_requirements_1_2) states at a value the
# Vulkan 1.2 specification contradicts, by vk.xml structure and member, with the base's value and
# the specification's. The derivation states the specification's value in the base's place and
# names each correction in its provenance; a base value other than the one listed fails the
# regeneration, so an upstream fix is noticed.
# - VkPhysicalDeviceDepthStencilResolveProperties: "VK_RESOLVE_MODE_SAMPLE_ZERO_BIT must be
#   included in the set" of supported depth and of supported stencil resolve modes.
# - The Required Limits table: maxDrawIndexedIndexValue is at least 2^24 - 1, where the base
#   states 2^24.
SPEC_CORRECTIONS = {
    ("VkPhysicalDeviceVulkan12Properties", "supportedDepthResolveModes"):
        ([], ["VK_RESOLVE_MODE_SAMPLE_ZERO_BIT"]),
    ("VkPhysicalDeviceVulkan12Properties", "supportedStencilResolveModes"):
        ([], ["VK_RESOLVE_MODE_SAMPLE_ZERO_BIT"]),
    ("VkPhysicalDeviceProperties", "maxDrawIndexedIndexValue"): (16777216, 16777215),
}

# The other extension device-property structures the product reads (src/device/khronos.rs
# `PROPERTY_STRUCTS`), by vk.xml name: completing a requirements document that lists the extension
# of one and omits a member fails, until EXTENSION_PROPERTIES states the values the reader takes.
# `VkPhysicalDeviceMaintenance10PropertiesKHR` is not among them: vk.xml gives its member
# `limittype="exact"`, which the layer never sets, and validation bounds it.
UNMODELLED_PROPERTY_STRUCTS = {
    "VkPhysicalDeviceVertexAttributeDivisorProperties",
    "VkPhysicalDeviceMaintenance4Properties",
    "VkPhysicalDevicePushDescriptorProperties",
    "VkPhysicalDevicePortabilitySubsetPropertiesKHR",
    "VkPhysicalDeviceAccelerationStructurePropertiesKHR",
    "VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT",
}

# Capability blocks holding the Vulkan 1.2 Required Format Support tables: every unconditional
# entry, and the entries a feature the profile requires makes unconditional (one block per
# feature, named by the prefix plus the feature).
REQUIRED_FORMATS_BLOCK = "m2v_vulkan12_required_formats"

# Device-property members no committed profile carries, whatever its source kind:
# they identify the host or the driver build a report came from instead of describing what the
# device can do. Nothing in translation reads them, and a profile simulated on another machine
# (VK_LAYER_KHRONOS_profiles) would report them as that machine's. Two kinds:
# - opaque identifiers: every `VK_UUID_SIZE` or `VK_LUID_SIZE` byte array that a device property
#   struct of the pinned vk.xml declares (`Registry.identifier_members`; regeneration fails when
#   the registry declares one this set lacks);
# - where the device sits on the host: its PCI address and its DRM node numbers.
HOST_IDENTITY = {
    "deviceLUID", "deviceUUID", "driverUUID", "optimalTilingLayoutUUID", "pipelineCacheUUID",
    "shaderBinaryUUID", "shaderModuleIdentifierAlgorithmUUID",
    "pciBus", "pciDevice", "pciDomain", "pciFunction",
    "primaryMajor", "primaryMinor", "renderMajor", "renderMinor",
}

# Each color attachment format's sample counts, which no profiles-schema structure holds
# (`VkImageFormatProperties`, Vulkan-Profiles issue 341): a `vulkaninfo` profile's top-level
# `SAMPLE_COUNTS_KEY` object, keyed by profile name, maps each format whose optimal tiling has
# `COLOR_ATTACHMENT` to the `VkSampleCountFlagBits` of a 2D, optimal-tiling image of it with
# `COLOR_ATTACHMENT` usage, from its `corpus-vulkan --capture-formats` supplement.
SAMPLE_COUNTS_KEY = "m2v_color_attachment_sample_counts"
SAMPLE_COUNTS_MEMBER = "colorAttachmentSampleCounts"

SOURCES = [
    {
        "name": "vk12-floor",
        "kind": "derived",
        "label": "Vulkan 1.2 required features and minimum limits only",
    },
    {
        "name": "lunarg-desktop-baseline-2023",
        "kind": "khronos",
        "url": PROFILES_TAG + "LunarG/VP_LUNARG_desktop_baseline.json",
        "sha256": "6a609a0ecfac0c57bb050c37dc300147bec4e24a75196aa79dfdb7edd6d36ab1",
        "profile": "VP_LUNARG_desktop_baseline_2023",
        "layer_presentation": True,
    },
    {
        "name": "roadmap-2022",
        "kind": "khronos",
        "url": PROFILES_TAG + "Khronos/VP_KHR_roadmap.json",
        "sha256": "41d8e5b2421bcd78e36d346de18933f5e8ac8e13a82bbcb197bed3fb7fec066d",
        "profile": "VP_KHR_roadmap_2022",
        "layer_presentation": True,
    },
    {
        "name": "nvidia-rtx50-linux",
        "kind": "gpuinfo",
        "report": 51770,
        "sha256": "9cfb004ba47300329269d78f6e4bc2ea39b1a35d48c93ef5fb361f071247827d",
    },
    {
        "name": "nvidia-pascal-linux",
        "kind": "gpuinfo",
        "report": 52074,
        "sha256": "36cdb075156f5255ec0ece4e91eaf8fe2ae8c1dfa1f7013aca196a27f387d73b",
    },
    {
        "name": "amd-rdna3-radv",
        "kind": "gpuinfo",
        "report": 51898,
        "sha256": "d930ad92ca5da4f19b1c44ccf71a3fa34998e1e3f6834b06da1fb0661abbdd3a",
    },
    {
        "name": "amd-rdna2-apu-radv",
        "kind": "gpuinfo",
        "report": 51886,
        "sha256": "27b8aa8d78d26dace85f0c9360a77dd1f4603f58ad47c9f280f89d532b7787f0",
    },
    {
        "name": "intel-arl-anv",
        "kind": "gpuinfo",
        "report": 52029,
        "sha256": "962998684be1ebdd2163b7810cf8a3aaea6b8ec01585282cb84a79e3f82d80f4",
    },
    {
        "name": "intel-irisxe-anv",
        "kind": "gpuinfo",
        "report": 51836,
        "sha256": "fb74c05355f2b8c75766949550bf5b970a7e8486f14edb11ad75766f1f9cf9fd",
    },
    {
        "name": "moltenvk-m3",
        "kind": "vulkaninfo",
        "sha256": "7e5c9fee5dc7d9fec91af17d782176781f2f02726394774de8e97e004105b9ed",
        "profile": "VP_VULKANINFO_Apple_M3_Max_0_2_2210_portability_subset",
        "command": "vulkaninfo --json",
        "host": ("Apple M3 Max, macOS 27.0.1 (26A434), Homebrew molten-vk 1.4.2, "
                 "vulkan-tools 1.4.363.0"),
        "formats": {
            "command": "corpus-vulkan --capture-formats OUT --device-type gpu",
            "sha256": "5c51e94668daac9f1310de8bac90bdc2b2b22a053efd8c0508277b93eac0bdc4",
        },
    },
    {
        "name": "lavapipe",
        "kind": "gpuinfo",
        "report": 51983,
        "sha256": "be9cbc3518501d272eb2d8e20383674095a8a0c125be424cd328e7dea761f29c",
    },
    {
        # The Linux gate's lavapipe: the committed `lavapipe` above is an x86
        # 256-bit llvmpipe with 8-lane subgroups; the gate image's is arm64, 128-bit, 4 lanes.
        "name": "gate-lavapipe",
        "kind": "vulkaninfo",
        "sha256": "66e849c60423e37387a34ffdee8b10cc0068f82e273451ac91dfebe2ef9c0fa9",
        "profile": "VP_VULKANINFO_llvmpipe_(LLVM_21_1_8,_128_bits)_26_0_8",
        "command": "vulkaninfo --json",
        "host": ("scripts/linux-gate --icd lavapipe, image m2v-linux-gate:c07668da48ca42af-arm64 "
                 "(Ubuntu 26.04, arm64): mesa-vulkan-drivers 26.0.8-1ubuntu0.3, libvulkan1 and "
                 "vulkan-tools 1.4.341.0"),
        "formats": {
            "command": "corpus-vulkan --capture-formats OUT --device-type cpu",
            "sha256": "9c109ad771f374d605e1bd5668d41254017fb6f5763fb42817a80bf5ad61c782",
        },
    },
    {
        "name": "gate-lavapipe-x86_64",
        "kind": "vulkaninfo",
        "sha256": "30066e1ab691eff34251fe41e575014f0da21f6092d4eb4f7fa7b10e8aa47a92",
        "profile": "VP_VULKANINFO_llvmpipe_(LLVM_21_1_8,_256_bits)_26_0_8",
        "command": "vulkaninfo --json",
        "host": ("scripts/linux-gate --icd lavapipe, image m2v-linux-gate:c07668da48ca42af-amd64 "
                 "(Ubuntu 26.04, x86_64): mesa-vulkan-drivers 26.0.8-1ubuntu0.3, libvulkan1 and "
                 "vulkan-tools 1.4.341.0"),
        "formats": {
            "command": "corpus-vulkan --capture-formats OUT --device-type cpu",
            "sha256": "a6becd36af52862861931fd072856fb3572be7bb23d9cb8d670831fb0b306c5d",
        },
    },
    {
        "name": "amd-780m-radv",
        "kind": "vulkaninfo",
        "sha256": "f109b309ae26d97339bb5b26f4ed5d6aa60c16dd8affeec6c5719bcda4d90601",
        "profile": "VP_VULKANINFO_AMD_Radeon_780M_Graphics_(RADV_PHOENIX)_26_2_6",
        "command": "vulkaninfo --json",
        "host": ("AMD Radeon 780M (Phoenix), CachyOS: mesa and vulkan-radeon 26.2.6, "
                 "vulkan-icd-loader and vulkan-tools 1.4.363.0"),
        "formats": {
            "command": "corpus-vulkan --capture-formats OUT --device-type gpu",
            "sha256": "987f79dcaea1d8ac4e1f7565377e81d656e15af810dee851a4b14d8515969387",
        },
    },
    {
        # The Radeon 780M of Hawk Point, the GPU of `amd-780m-radv` under another deviceID
        # (0x1900), whose measured driver facts it shares; no venue names it. Its capture, under
        # the same RADV build, differs from `amd-780m-radv`'s only in the deviceID.
        "name": "amd-780m-hawkpoint-radv",
        "kind": "vulkaninfo",
        "sha256": "2a79388cb4ef4f6dbe99fd31217b6eb6b200c1d92c3e12304e99efb15c46cd3c",
        "profile": "VP_VULKANINFO_AMD_Radeon_780M_Graphics_(RADV_PHOENIX)_26_2_4",
        "command": "vulkaninfo --json",
        "host": ("AMD Radeon 780M (Hawk Point, Ryzen 7 8845HS), CachyOS: mesa and vulkan-radeon "
                 "26.2.4, vulkan-icd-loader and vulkan-tools 1.4.363.0"),
        "formats": {
            "command": "corpus-vulkan --capture-formats OUT --device-type gpu",
            "sha256": "d999da3206596ae9b15e9c523b098bdda379b4c2a8a214076e7fc8804e095321",
        },
    },
    {
        "name": "intel-irisxe-rplp-anv",
        "kind": "vulkaninfo",
        "sha256": "6d34a34db319ee2f6b45f0fadac382cb5d3b3e7a5f5816513a71cd4080d34c8d",
        "profile": "VP_VULKANINFO_Intel(R)_Iris(R)_Xe_Graphics_(RPL-P)_26_2_4",
        "command": "vulkaninfo --json",
        "host": ("Intel Iris Xe Graphics (Raptor Lake-P), CachyOS: mesa and vulkan-intel "
                 "26.2.4, vulkan-icd-loader and vulkan-tools 1.4.357.0"),
        "formats": {
            "command": "corpus-vulkan --capture-formats OUT --device-type gpu",
            "sha256": "6e532de1596b32bb703505c7ca6670358f9200386dd2a2d72f4ad0229164e8ff",
        },
    },
    {
        # The second device the intel-irisxe-rplp-anv venue allows: an i9-13900's UHD Graphics 770
        # (Raptor Lake-S), under the same ANV build and loader as the venue's Iris Xe (RPL-P); its
        # capture derives the venue's translation key.
        "name": "intel-uhd770-rpls-anv",
        "kind": "vulkaninfo",
        "sha256": "c4088e885bf46fc414e55a56fdb0f5b946ae6a34af61e2667891cab6089f8ca2",
        "profile": "VP_VULKANINFO_Intel(R)_Graphics_(RPL-S)_26_2_4",
        "command": "vulkaninfo --json",
        "host": ("Intel UHD Graphics 770 (Raptor Lake-S, i9-13900), CachyOS: mesa and "
                 "vulkan-intel 26.2.4, vulkan-icd-loader and vulkan-tools 1.4.357.0"),
        "formats": {
            "command": "corpus-vulkan --capture-formats OUT --device-type gpu",
            "sha256": "e1ed48d542902316c8bf2f4f0cbde7a3387b3aecac2a9c4dfc763acecd97a8d2",
        },
    },
    {
        "name": "nvidia-rtx5080-laptop",
        "kind": "vulkaninfo",
        "sha256": "94b71e5bd1f1bc61e06921f173bb12459f525ff6c49a1bf26dd933f914253469",
        "profile": "VP_VULKANINFO_NVIDIA_GeForce_RTX_5080_Laptop_GPU_615_71_9_0",
        "command": "vulkaninfo --json",
        "host": ("NVIDIA GeForce RTX 5080 Laptop GPU, CachyOS: nvidia-utils 615.71.09 "
                 "(nvidia-open kernel module), vulkan-icd-loader and vulkan-tools 1.4.357.0"),
        "formats": {
            "command": "corpus-vulkan --capture-formats OUT --device-type gpu",
            "sha256": "86bc23122d165c02808b913be956569b1a35ef976e150aea410a5248cad4c003",
        },
    },
    {
        "name": "swiftshader",
        "kind": "gpuinfo",
        "report": 38278,
        "sha256": "122ca1fd87e5fc588a4acfd48eb2428519867c6104ea48e54520da746405fcb5",
    },
]




def profile_name(stem):
    return "VP_M2V_" + stem.replace("-", "_")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, pinned, offline):
    """Returns the pinned bytes of `url`, from the cache or the network."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / pinned
    if cached.exists():
        data = cached.read_bytes()
    elif offline:
        raise SystemExit(f"offline and not cached: {url}")
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "metal2vulkan-profiles"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
    actual = sha256(data)
    if actual != pinned:
        raise SystemExit(f"{url}: sha256 {actual} does not match pin {pinned}")
    cached.write_bytes(data)
    return data


def supported(element):
    api = element.get("api")
    return api is None or "vulkan" in api.split(",")


class Member:
    def __init__(self, element):
        self.name = element.find("name").text
        self.type = element.find("type").text
        parts = [element.text or ""]
        for child in element:
            if child.tag != "comment":
                parts.append(child.text or "")
            parts.append(child.tail or "")
        tail = "".join(parts)
        self.pointer = "*" in tail
        # Array bounds are literal numbers or API constants (`<enum>` children, whose text
        # `itertext` includes between the brackets).
        self.dims = [dim.strip() for dim in re.findall(r"\[([^\]]*)\]", tail)]
        if any(not dim for dim in self.dims):
            raise ValueError(f"member {self.name}: empty array bound")


class Registry:
    """The subset of `vk.xml` the conversion needs: structs, enums, bitmasks, extensions."""

    def __init__(self, data):
        root = ET.fromstring(data)
        self.constants = {}
        self.structs = {}
        self.aliases = {}
        self.categories = {}
        self.bitmask_bits = {}
        self.enum_values = {}
        for element in root.find("types"):
            if element.tag != "type" or not supported(element):
                continue
            name = element.get("name") or element.findtext("name")
            category = element.get("category")
            if element.get("alias"):
                self.aliases[name] = element.get("alias")
                self.categories[name] = category
                continue
            self.categories[name] = category
            if category in ("struct", "union"):
                self.structs[name] = {
                    "members": [Member(m) for m in element.findall("member") if supported(m)],
                    "extends": set(filter(None, (element.get("structextends") or "").split(","))),
                }
            elif category == "bitmask":
                bits = element.get("requires") or element.get("bitvalues")
                if bits:
                    self.bitmask_bits[name] = bits
        for enums in root.findall("enums"):
            name = enums.get("name")
            if name == "API Constants":
                for constant in enums.findall("enum"):
                    if constant.get("value"):
                        self.constants[constant.get("name")] = constant.get("value")
                continue
            values = self.enum_values.setdefault(name, {})
            for enum in enums.findall("enum"):
                if supported(enum):
                    self._add_enum(values, enum, None)
        self.extension_types = {}
        for feature in root.findall("feature"):
            if supported(feature):
                for require in feature.findall("require"):
                    for enum in require.findall("enum"):
                        if enum.get("extends") and supported(enum):
                            self._add_enum(self.enum_values.setdefault(enum.get("extends"), {}), enum, None)
        for extension in root.find("extensions"):
            number = int(extension.get("number"))
            types = []
            for require in extension.findall("require"):
                if not supported(require):
                    continue
                types.extend(t.get("name") for t in require.findall("type"))
                for enum in require.findall("enum"):
                    if enum.get("extends") and supported(enum):
                        values = self.enum_values.setdefault(enum.get("extends"), {})
                        self._add_enum(values, enum, number)
            self.extension_types[extension.get("name")] = types
        self.required_by_version = self._required_by_version(root)
        self.required_features = {
            member for version in ("VK_VERSION_1_0", "VK_VERSION_1_1", "VK_VERSION_1_2")
            for _, member in self.required_by_version.get(version, [])}

    @staticmethod
    def _add_enum(values, enum, extension_number):
        if enum.get("alias"):
            return
        name = enum.get("name")
        if enum.get("bitpos") is not None:
            value = 1 << int(enum.get("bitpos"))
        elif enum.get("offset") is not None:
            number = int(enum.get("extnumber") or extension_number)
            value = 1000000000 + (number - 1) * 1000 + int(enum.get("offset"))
            if enum.get("dir") == "-":
                value = -value
        elif enum.get("value") is not None:
            text = enum.get("value")
            try:
                value = int(text, 0)
            except ValueError:
                return
        else:
            return
        # The first non-alias name for a value wins, which is the core spelling.
        values.setdefault(value, name)

    @staticmethod
    def _required_by_version(root):
        """Each core version's feature requirements without a condition, as (structure, member)
        pairs, by version name (`VK_VERSION_1_2`)."""
        out = {}
        for feature in root.findall("feature"):
            if not supported(feature):
                continue
            pairs = out.setdefault(feature.get("name"), [])
            for require in feature.findall("require"):
                if require.get("depends"):
                    continue
                pairs.extend((f.get("struct"), f.get("name")) for f in require.findall("feature"))
        return out

    def identifier_members(self):
        """Names of the opaque identifiers among device properties: every `VK_UUID_SIZE` or
        `VK_LUID_SIZE` byte array of `VkPhysicalDeviceProperties` or of a struct extending
        `VkPhysicalDeviceProperties2`."""
        names = set()
        for name, struct in self.structs.items():
            if (name != "VkPhysicalDeviceProperties"
                    and "VkPhysicalDeviceProperties2" not in struct["extends"]):
                continue
            for member in struct["members"]:
                if member.dims in (["VK_UUID_SIZE"], ["VK_LUID_SIZE"]):
                    names.add(member.name)
        return names

    def resolve(self, name):
        seen = set()
        while name in self.aliases:
            if name in seen:
                raise ValueError(f"alias cycle at {name}")
            seen.add(name)
            name = self.aliases[name]
        return name

    def struct(self, name):
        return self.structs[self.resolve(name)]

    def member(self, struct, name):
        for member in self.struct(struct)["members"]:
            if member.name == name:
                return member
        return None

    def extension_structs(self, extension, chain):
        """Structs an extension adds to `chain` (e.g. `VkPhysicalDeviceFeatures2`), as spelled by
        the extension."""
        result = []
        for name in self.extension_types.get(extension, []):
            resolved = self.resolve(name)
            if resolved in self.structs and chain in self.structs[resolved]["extends"]:
                result.append(name)
        return result


class Unmapped(Exception):
    pass


def parse_int(raw):
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str):
        return int(raw, 0)
    raise Unmapped(f"not an integer: {raw!r}")


INTEGER_TYPES = {
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t", "int32_t", "int64_t",
    "size_t", "VkDeviceSize", "VkDeviceAddress", "VkSampleMask",
}


class Converter:
    def __init__(self, registry):
        self.registry = registry

    def enum_name(self, type_name, raw):
        value = parse_int(raw)
        values = self.registry.enum_values.get(self.registry.resolve(type_name), {})
        if value not in values:
            raise Unmapped(f"{type_name} value {value}")
        return values[value]

    def flag_names(self, type_name, raw):
        bits_type = self.registry.bitmask_bits.get(self.registry.resolve(type_name))
        value = parse_int(raw)
        if bits_type is None:
            if value == 0:
                return []
            raise Unmapped(f"{type_name} has no bits but value {value}")
        values = self.registry.enum_values.get(bits_type, {})
        names = []
        for bit in range(64):
            mask = 1 << bit
            if value & mask:
                if mask not in values:
                    raise Unmapped(f"{type_name} bit {bit}")
                names.append(values[mask])
        return names

    def scalar(self, type_name, raw):
        category = self.registry.categories.get(type_name)
        if type_name == "VkBool32":
            if raw in ("true", "false"):
                return raw == "true"
            value = parse_int(raw)
            if value not in (0, 1):
                raise Unmapped(f"VkBool32 value {value}")
            return value == 1
        if type_name in INTEGER_TYPES:
            return parse_int(raw)
        if type_name in ("float", "double"):
            return float(raw)
        if category == "enum":
            return self.enum_name(type_name, raw)
        if category == "bitmask":
            return self.flag_names(type_name, raw)
        if category == "struct":
            return self.struct_value(type_name, raw)
        raise Unmapped(f"unsupported member type {type_name}")

    def value(self, member, raw):
        if member.type == "char" and member.dims:
            if not isinstance(raw, str):
                raise Unmapped(f"{member.name}: expected a string")
            return raw
        if not member.dims:
            return self.scalar(member.type, raw)
        if not isinstance(raw, list):
            raise Unmapped(f"{member.name}: expected an array")
        return [self.scalar(member.type, item) for item in raw]

    def struct_value(self, type_name, raw):
        if self.registry.resolve(type_name) == "VkConformanceVersion" and isinstance(raw, str):
            parts = raw.split(".")
            if len(parts) != 4:
                raise Unmapped(f"conformance version {raw!r}")
            raw = dict(zip(["major", "minor", "subminor", "patch"], parts))
        if isinstance(raw, list):
            # gpuinfo spells small all-scalar structs such as VkExtent2D as positional arrays.
            members = self.registry.struct(type_name)["members"]
            if len(raw) != len(members) or any(m.dims or m.pointer for m in members):
                raise Unmapped(f"{type_name}: positional value {raw!r}")
            raw = {member.name: item for member, item in zip(members, raw)}
        if not isinstance(raw, dict):
            raise Unmapped(f"{type_name}: expected an object")
        result, unmapped = self.fill(type_name, raw, type_name)
        if unmapped:
            raise Unmapped(f"{type_name}: {', '.join(unmapped)}")
        return result

    def fill(self, struct, raw, where):
        """Converts the members of `raw` that `struct` declares, in declaration order. Returns the
        object and the list of source fields that could not be represented."""
        result = {}
        unmapped = []
        known = set()
        for member in self.registry.struct(struct)["members"]:
            known.add(member.name)
            if member.name in ("sType", "pNext") or member.pointer or member.name not in raw:
                continue
            try:
                result[member.name] = self.value(member, raw[member.name])
            except (Unmapped, ValueError, TypeError) as error:
                unmapped.append(f"{where}.{member.name} ({error})")
        for name in raw:
            if name not in known:
                unmapped.append(f"{where}.{name} (not a member)")
            elif self.registry.member(struct, name).pointer:
                unmapped.append(f"{where}.{name} (pointer member)")
        return result, unmapped


def group_extended(registry, rows, chain, output, unmapped, converter):
    """Folds gpuinfo's flat `{extension, name, value}` rows into the extension's structs."""
    by_extension = {}
    for row in rows:
        by_extension.setdefault(row["extension"], []).append(row)
    for extension in sorted(by_extension):
        structs = registry.extension_structs(extension, chain)
        values = {}
        for row in by_extension[extension]:
            raw = row["supported"] if "supported" in row else row["value"]
            owner = next((s for s in structs if registry.member(s, row["name"])), None)
            if owner is None:
                unmapped.append(f"{extension}.{row['name']} (no {chain} struct declares it)")
                continue
            values.setdefault(owner, {})[row["name"]] = raw
        for owner in structs:
            if owner in values:
                result, missing = converter.fill(owner, values[owner], owner)
                unmapped.extend(missing)
                if result:
                    output[owner] = result


def convert_gpuinfo(registry, report, source):
    converter = Converter(registry)
    unmapped = []
    properties = report["properties"]
    features = {}
    if "features" in report:
        result, missing = converter.fill("VkPhysicalDeviceFeatures", report["features"], "features")
        features["VkPhysicalDeviceFeatures"] = result
        unmapped.extend(missing)
    for version in ("11", "12", "13", "14"):
        core = report.get(f"core{version}") or {}
        if core.get("features"):
            name = f"VkPhysicalDeviceVulkan{version}Features"
            result, missing = converter.fill(name, core["features"], f"core{version}.features")
            features[name] = result
            unmapped.extend(missing)
    extended = report.get("extended") or {}
    group_extended(registry, extended.get("devicefeatures2") or [], "VkPhysicalDeviceFeatures2",
                   features, unmapped, converter)

    props = {}
    core10 = {k: properties[k] for k in properties if registry.member("VkPhysicalDeviceProperties", k)}
    result, missing = converter.fill("VkPhysicalDeviceProperties", core10, "properties")
    props["VkPhysicalDeviceProperties"] = result
    unmapped.extend(missing)
    ignored = {"apiVersionText", "deviceTypeText", "driverVersionText", "headerversion",
               "hasFormatFeatureFlags2", "subgroupProperties", "productModel", "productManufacturer"}
    for key in properties:
        if key not in core10 and key not in ignored:
            unmapped.append(f"properties.{key} (not a member)")
    if properties.get("subgroupProperties"):
        result, missing = converter.fill("VkPhysicalDeviceSubgroupProperties",
                                         properties["subgroupProperties"], "subgroupProperties")
        props["VkPhysicalDeviceSubgroupProperties"] = result
        unmapped.extend(missing)
    for version in ("11", "12", "13", "14"):
        core = report.get(f"core{version}") or {}
        if core.get("properties"):
            name = f"VkPhysicalDeviceVulkan{version}Properties"
            result, missing = converter.fill(name, core["properties"], f"core{version}.properties")
            props[name] = result
            unmapped.extend(missing)
    group_extended(registry, extended.get("deviceproperties2") or [], "VkPhysicalDeviceProperties2",
                   props, unmapped, converter)

    extensions = {}
    for extension in report.get("extensions") or []:
        name = extension["extensionName"]
        if name not in registry.extension_types:
            unmapped.append(f"extensions.{name} (not in the registry)")
            continue
        extensions[name] = parse_int(extension["specVersion"])

    formats = {}
    flags2 = bool(properties.get("hasFormatFeatureFlags2"))
    for format_id, entry in report.get("formats") or []:
        if not entry.get("supported"):
            continue
        try:
            name = converter.enum_name("VkFormat", format_id)
            legacy = {}
            extended_props = {}
            for field in ("linearTilingFeatures", "optimalTilingFeatures", "bufferFeatures"):
                value = parse_int(entry[field])
                legacy_bits = registry.enum_values["VkFormatFeatureFlagBits"]
                legacy[field] = [legacy_bits[1 << b] for b in range(32)
                                 if value & (1 << b) and (1 << b) in legacy_bits]
                if flags2:
                    extended_props[field] = converter.flag_names("VkFormatFeatureFlags2", value)
                elif any(value & (1 << b) and (1 << b) not in legacy_bits for b in range(64)):
                    raise Unmapped(f"{field} has bits outside VkFormatFeatureFlagBits")
            formats[name] = {"VkFormatProperties": legacy}
            if flags2:
                formats[name]["VkFormatProperties3"] = extended_props
        except Unmapped as error:
            unmapped.append(f"formats[{format_id}] ({error})")

    queues = []
    for queue in report.get("queues") or []:
        raw = {k: v for k, v in queue.items() if k != "supportsPresent"}
        result, missing = converter.fill("VkQueueFamilyProperties", raw, "queues")
        unmapped.extend(missing)
        queues.append({"VkQueueFamilyProperties": result})

    device = {
        "extensions": dict(sorted(extensions.items())),
        "features": dict(sorted(features.items())),
        "properties": dict(sorted(props.items())),
        "formats": dict(sorted(formats.items())),
        "queueFamiliesProperties": queues,
    }
    environment = report.get("environment") or {}
    core12 = (report.get("core12") or {}).get("properties") or {}
    api_version = properties["apiVersionText"]
    report_id = source["report"]
    osname = f"{environment.get('name', '')} {environment.get('version', '')}".strip()
    osname = osname.replace("'", "")
    label = (f"{properties['deviceName']} driver {properties['driverVersionText']} "
             f"on {osname}")
    provenance = {
        "source": "gpuinfo",
        "report_id": report_id,
        "url": GPUINFO_DISPLAY.format(report_id),
        "raw_url": GPUINFO_REPORT.format(report_id),
        "sha256": source["sha256"],
        "device_name": properties["deviceName"],
        "vendor_id": properties.get("vendorID"),
        "device_id": properties.get("deviceID"),
        "driver_name": core12.get("driverName"),
        "driver_info": core12.get("driverInfo"),
        "driver_version": properties["driverVersionText"],
        "api_version": api_version,
        "os": osname,
        "report_version": environment.get("reportversion"),
        "converter": CONVERTER,
        "registry": REGISTRY,
        "unmapped": sorted(unmapped),
    }
    return document(source["name"], api_version, label,
                    f"Converted from vulkan.gpuinfo.org report {report_id} "
                    f"({GPUINFO_DISPLAY.format(report_id)}).",
                    {"device": device}, ["device"], [], provenance)


def document(stem, api_version, label, description, capabilities, required, optionals,
             provenance):
    api = re.match(r"^(\d+)\.(\d+)\.(\d+)", api_version)
    if not api:
        raise SystemExit(f"{stem}: bad api version {api_version}")
    profile = {
        "version": 1,
        "api-version": api.group(0),
        "label": label,
        "description": description,
        "contributors": {},
        "history": [{
            "revision": 1,
            "date": GENERATED,
            "author": "metal2vulkan",
            "comment": f"Generated by {CONVERTER}; see provenance.",
        }],
        "capabilities": required,
    }
    if optionals:
        profile["optionals"] = optionals
    return {
        "$schema": SCHEMA,
        "provenance": provenance,
        "capabilities": capabilities,
        "profiles": {profile_name(stem): profile},
    }


def referenced(entries):
    names = []
    for entry in entries:
        for name in entry if isinstance(entry, list) else [entry]:
            if name not in names:
                names.append(name)
    return names


def required_features(capabilities, entries):
    """The feature members a profile's capability list requires: those a required block sets, and
    those every alternative of a one-of entry sets (one alternative alone guarantees nothing)."""
    def enabled(name):
        return {member for members in capabilities[name].get("features", {}).values()
                for member, value in members.items() if value is True}

    result = set()
    for entry in entries:
        if isinstance(entry, list):
            result |= set.intersection(*(enabled(name) for name in entry)) if entry else set()
        else:
            result |= enabled(entry)
    return result


def required_format_blocks(capabilities, required, provenance, formats):
    """Adds the Required Format Support blocks a requirements profile implies: every
    unconditional entry, and one block per feature the profile requires that makes conditional
    entries unconditional. The provenance names the pinned source and lists every conditional
    requirement, including the choices between formats or tables a profile cannot state."""
    features = required_features(capabilities, required)

    def block(table):
        return {"formats": {name: {"VkFormatProperties": props}
                            for name, props in table.items()}}

    capabilities[REQUIRED_FORMATS_BLOCK] = block(formats["unconditional"])
    required.append(REQUIRED_FORMATS_BLOCK)
    by_feature = {}
    for requirement in formats["conditional"]:
        feature = requirement.get("if_feature")
        if feature in features:
            [table] = requirement["any_of"]
            by_feature.setdefault(feature, {}).update(copy.deepcopy(table))
    for feature, table in sorted(by_feature.items()):
        name = f"{REQUIRED_FORMATS_BLOCK}_if_{feature}"
        capabilities[name] = block(dict(sorted(table.items())))
        required.append(name)
    provenance["required_formats"] = {
        "url": required_formats.SPEC["url"],
        "sha256": required_formats.SPEC["sha256"],
        "tag": required_formats.SPEC["tag"],
        "attributes": required_formats.VERSION_1_2,
        "tables": formats["tables"],
        "rule": (f"the Required Format Support tables of the Vulkan 1.2 specification, read by "
                 f"scripts/regen-device-profiles/required_formats.py: {REQUIRED_FORMATS_BLOCK} "
                 f"holds every unconditional entry and {REQUIRED_FORMATS_BLOCK}_if_<feature> "
                 f"the entries a feature this profile requires makes unconditional; "
                 f"`conditional` lists every conditional requirement (at least one table of "
                 f"`any_of` holds in full, on a device supporting `if_feature` when present)"),
        "conditional": formats["conditional"],
        "not_represented": formats["not_represented"],
    }


def stated_members(capabilities, section):
    """The member names `capabilities` state in `section` (`features` or `properties`), of any
    structure, as the product's reader reads them (flattened, `limits` included)."""
    names = set()
    for block in capabilities.values():
        for members in block.get(section, {}).values():
            for member, value in members.items():
                if member == "limits" and isinstance(value, dict):
                    names.update(value)
                else:
                    names.add(member)
    return names


def presented_formats(capabilities, stem):
    """Each format several of `capabilities` (in the profile's order) state, at the union of their
    flags, where it differs from the last block's entry. `VK_LAYER_KHRONOS_profiles` keeps one
    block's entry of a format, the last that states it, while the product's reader takes the
    union of every block's flags (`roadmap-2022`'s Required Format Support blocks
    state 23 formats twice, and the layer reported the conditional block's storage bit alone)."""
    entries = {}
    for block in capabilities.values():
        for name, props in block.get("formats", {}).items():
            entries.setdefault(name, []).append(props)
    formats = {}
    for name, stated in sorted(entries.items()):
        if len(stated) < 2:
            continue
        union = {}
        for props in stated:
            for struct, fields in props.items():
                target = union.setdefault(struct, {})
                for field, flags in fields.items():
                    if not isinstance(flags, list):
                        raise SystemExit(
                            f"{stem}: {name}'s {struct}.{field} is stated by several blocks and "
                            "is not a list of flags: its union is not modelled")
                    have = target.setdefault(field, [])
                    have.extend(flag for flag in flags if flag not in have)

        def sets(props):
            return {(struct, field): set(flags) for struct, fields in props.items()
                    for field, flags in fields.items() if flags}

        if sets(union) != sets(stated[-1]):
            formats[name] = union
    return formats


def layer_presentation(registry, capabilities, api_version, stem):
    """The `LAYER_PRESENTATION_BLOCK` of a requirements document: each feature its core version
    requires without a condition, each member the product reads of the core device-property
    structures of its version (`CORE_PROPERTIES`), and each member the product reads of the
    device-property structures of the extensions it lists (`EXTENSION_PROPERTIES`), that
    `capabilities` omit, at the value the product reads; and each format several blocks state, at
    the union the product reads ([`presented_formats`]). The block is the profile's last, so the
    layer keeps its entry of each such format. Returns the block and the stated members' names."""
    major, minor = (int(part) for part in api_version.split(".")[:2])
    modelled = max(core for core, _ in CORE_PROPERTIES.values())
    if major != 1 or minor > modelled:
        raise SystemExit(
            f"{stem}: Vulkan {major}.{minor}'s device-property structures are not modelled "
            f"(CORE_PROPERTIES reaches 1.{modelled})")
    features_stated = stated_members(capabilities, "features")
    properties_stated = stated_members(capabilities, "properties")
    features = {}
    for version in range(minor + 1):
        # vk.xml may name a requirement in the core structure and in the extension structure
        # it absorbed: stated once, in the core one (the `VkPhysicalDeviceVulkan1x` or 1.0
        # structure), as profile documents do.
        pairs = sorted(registry.required_by_version.get(f"VK_VERSION_1_{version}", []),
                       key=lambda pair: not pair[0].startswith(("VkPhysicalDeviceVulkan1",
                                                                "VkPhysicalDeviceFeatures")))
        for struct, member in pairs:
            if member not in features_stated:
                features.setdefault(struct, {})[member] = True
                features_stated.add(member)
    features_true = {member for block in list(capabilities.values()) + [{"features": features}]
                     for members in block.get("features", {}).values()
                     for member, value in members.items() if value is True}
    properties = {}
    for struct, (core, members) in CORE_PROPERTIES.items():
        if core > minor:
            continue
        nested = members.get("limits") if struct == "VkPhysicalDeviceProperties" else None
        for member, value in (nested if nested is not None else members).items():
            if member not in properties_stated:
                exact = FEATURE_EXACT_PROPERTIES.get((struct, member))
                if exact is not None and exact[0] in features_true:
                    value = exact[1]
                target = properties.setdefault(struct, {})
                if nested is not None:
                    target = target.setdefault("limits", {})
                target[member] = copy.deepcopy(value)
    extensions = sorted({name for block in capabilities.values()
                         for name in block.get("extensions", {})})
    for extension in extensions:
        for struct in registry.extension_structs(extension, "VkPhysicalDeviceProperties2"):
            resolved = registry.resolve(struct)
            if resolved in EXTENSION_PROPERTIES:
                for member, value in EXTENSION_PROPERTIES[resolved].items():
                    if member not in properties_stated:
                        properties.setdefault(struct, {})[member] = copy.deepcopy(value)
            elif resolved in UNMODELLED_PROPERTY_STRUCTS:
                omitted = sorted(member.name for member in registry.struct(struct)["members"]
                                 if member.name not in ("sType", "pNext")
                                 and member.name not in properties_stated)
                if omitted:
                    raise SystemExit(
                        f"{stem}: {extension}'s {struct} is read by the product, and the profile "
                        f"omits {omitted}: state the values its reader takes in "
                        "EXTENSION_PROPERTIES")
    block = {}
    if features:
        block["features"] = features
    if properties:
        block["properties"] = properties
    formats = presented_formats(capabilities, stem)
    if formats:
        block["formats"] = formats

    def names(prefix, members):
        for member, value in members.items():
            if member == "limits" and isinstance(value, dict):
                yield from names(f"{prefix}.limits", value)
            else:
                yield f"{prefix}.{member}"

    named = sorted([name for section in ("features", "properties")
                    for struct, members in block.get(section, {}).items()
                    for name in names(struct, members)]
                   + [f"formats.{name}" for name in formats])
    return block, named


def extract(data, source, formats, registry=None):
    """One profile plus the capability blocks it references, verbatim. A profile whose blocks
    state no formats gets the Required Format Support blocks (`required_format_blocks`); a source
    with `layer_presentation` gets the `LAYER_PRESENTATION_BLOCK` (`layer_presentation`, which
    reads the pinned `registry`)."""
    doc = json.loads(data)
    profile = copy.deepcopy(doc["profiles"][source["profile"]])
    if profile.get("profiles"):
        raise SystemExit(f"{source['profile']} requires other profiles; not supported")
    names = referenced(profile.get("capabilities", []) + profile.get("optionals", []))
    capabilities = {name: doc["capabilities"][name] for name in names}
    provenance = {
        "source": "khronos",
        "url": source["url"],
        "sha256": source["sha256"],
        "source_profile": source["profile"],
        "source_profile_version": profile["version"],
    }
    if not any("formats" in capabilities[name] for name in names):
        required_format_blocks(capabilities, profile["capabilities"], provenance, formats)
    if source.get("layer_presentation"):
        # What the profile states is its capabilities' (the reader and the layer read no
        # optionals).
        required = {name: capabilities[name] for name in referenced(profile["capabilities"])}
        block, named = layer_presentation(registry, required, profile["api-version"],
                                          source["name"])
        if block:
            capabilities[LAYER_PRESENTATION_BLOCK] = block
            profile["capabilities"].append(LAYER_PRESENTATION_BLOCK)
        provenance["layer_presentation"] = {
            "registry": REGISTRY,
            "rule": (f"the {LAYER_PRESENTATION_BLOCK} block states each feature the profile's "
                     "core version requires without a condition, and each member the product "
                     "reads of the core device-property structures of its version and of the "
                     "device-property structures of the extensions it lists, that the profile "
                     "omits, at the value the product reads, "
                     + ("and each format several of its blocks state at the union of their "
                        "flags, which the product reads (the layer keeps the last block's entry "
                        "of a format), " if "formats" in block else "")
                     + "so that VK_LAYER_KHRONOS_profiles reports the profile as the product "
                     "reads it"),
            "stated": named,
        }
    out = {
        "$schema": SCHEMA,
        "provenance": provenance,
        "capabilities": capabilities,
        "profiles": {profile_name(source["name"]): profile},
    }
    profile["description"] = (f"{source['profile']} extracted verbatim from "
                              f"{source['url']}. {profile['description']}")
    return out


def derive(registry, data, source, formats):
    doc = json.loads(data)
    base = doc["profiles"][MINIMUM_REQUIREMENTS["profile"]]
    names = referenced(base["capabilities"])
    capabilities = {}
    dropped = []
    corrected = []
    for name in names:
        block = copy.deepcopy(doc["capabilities"][name])
        for struct, members in block.get("features", {}).items():
            for member in list(members):
                if member not in registry.required_features:
                    dropped.append(f"{struct}.{member}")
                    del members[member]
        block["features"] = {s: m for s, m in block.get("features", {}).items() if m}
        if not block["features"]:
            del block["features"]
        for struct, members in block.get("properties", {}).items():
            nested = members.get("limits") if struct == "VkPhysicalDeviceProperties" else None
            target = nested if nested is not None else members
            for member in list(target):
                correction = SPEC_CORRECTIONS.get((struct, member))
                if correction is None:
                    continue
                stated, specified = correction
                if target[member] != stated:
                    raise SystemExit(f"{source['name']}: the base profile states {struct}."
                                     f"{member} = {json.dumps(target[member])}, not "
                                     f"{json.dumps(stated)}: review SPEC_CORRECTIONS")
                target[member] = copy.deepcopy(specified)
                corrected.append(f"{struct}.{member}: {json.dumps(stated)} -> "
                                 f"{json.dumps(specified)}")
        capabilities[name] = block
    missing = sorted(registry.required_features - {
        member for block in capabilities.values()
        for members in block.get("features", {}).values() for member in members})
    if missing:
        raise SystemExit(f"{source['name']}: base profile lacks required features {missing}")
    uncorrected = sorted(f"{struct}.{member}" for struct, member in SPEC_CORRECTIONS
                         if not any(f"{struct}.{member}:" in line for line in corrected))
    if uncorrected:
        raise SystemExit(f"{source['name']}: the base profile no longer states {uncorrected}: "
                         "review SPEC_CORRECTIONS")
    required = list(names)
    provenance = {
        "source": "derived",
        "base_url": MINIMUM_REQUIREMENTS["url"],
        "base_sha256": MINIMUM_REQUIREMENTS["sha256"],
        "base_profile": MINIMUM_REQUIREMENTS["profile"],
        "registry": REGISTRY,
        "rule": ("features reduced to the unconditional VK_VERSION_1_0..1_2 feature "
                 "requirements of vk.xml; limits and properties are the base profile's "
                 "minimums, but where the Vulkan 1.2 specification states another "
                 "(spec_corrections)"),
        "dropped_features": sorted(dropped),
        "spec_corrections": sorted(corrected),
    }
    def flattened(struct, members):
        nested = members.get("limits") if struct == "VkPhysicalDeviceProperties" else None
        return nested if nested is not None else members

    stated = {(struct, member) for block in capabilities.values()
              for struct, members in block.get("properties", {}).items()
              for member in flattened(struct, members)}
    restated = sorted(f"{struct}.{member}" for struct, members in UNREQUIRED_PROPERTIES.items()
                      for member in flattened(struct, members) if (struct, member) in stated)
    if restated:
        raise SystemExit(f"{source['name']}: the base profile now states {restated}")
    capabilities[UNREQUIRED_PROPERTIES_BLOCK] = {
        "properties": copy.deepcopy(UNREQUIRED_PROPERTIES),
    }
    required.append(UNREQUIRED_PROPERTIES_BLOCK)
    provenance["rule"] += (f"; plus the {UNREQUIRED_PROPERTIES_BLOCK} block: the weakest value "
                           "the specification permits for each property the product reads, and "
                           "the profiles layer sets, that the base profile leaves without a "
                           "requirement")
    required_format_blocks(capabilities, required, provenance, formats)
    return document(source["name"], base["api-version"], source["label"],
                    f"Derived by {CONVERTER} from {MINIMUM_REQUIREMENTS['profile']}; "
                    "see provenance.", capabilities, required, [], provenance)


def import_capture(data, source, supplement=None, registry=None):
    """The profile `source` names in the `vulkaninfo --json` capture `data`, with its capability
    blocks; with `supplement` (the device's `corpus-vulkan --capture-formats` document, parsed)
    and the pinned `registry`, every format also states its `VkFormatProperties3`, and each color
    attachment format its sample counts under `SAMPLE_COUNTS_KEY` ([`merge_formats`])."""
    doc = json.loads(data)
    profile = copy.deepcopy(doc["profiles"][source["profile"]])
    names = referenced(profile["capabilities"] + profile.get("optionals", []))
    capabilities = {name: doc["capabilities"][name] for name in names}
    device = capabilities["device"]["properties"]
    driver = (device.get("VkPhysicalDeviceVulkan12Properties")
              or device.get("VkPhysicalDeviceDriverProperties") or {})
    core = device["VkPhysicalDeviceProperties"]
    transforms = []
    # The profiles schema requires unique queue-family entries; MoltenVK reports several
    # identical families. Keep the first of each and say so.
    for block in capabilities.values():
        families = block.get("queueFamiliesProperties")
        if families:
            unique = []
            for family in families:
                if family not in unique:
                    unique.append(family)
            if len(unique) != len(families):
                transforms.append(f"queueFamiliesProperties: dropped {len(families) - len(unique)} "
                                  "entries identical to an earlier family (schema uniqueItems)")
                block["queueFamiliesProperties"] = unique
    provenance = {
        "source": "vulkaninfo",
        "command": source["command"],
        "sha256": source["sha256"],
        "source_profile": source["profile"],
        "device_name": core["deviceName"],
        "vendor_id": core["vendorID"],
        "device_id": core["deviceID"],
        "driver_name": driver.get("driverName"),
        "driver_info": driver.get("driverInfo"),
        "driver_version": str(core["driverVersion"]),
        "api_version": profile["api-version"],
        "host": source["host"],
        "captured": profile["history"][0]["date"],
        "transforms": transforms,
    }
    if registry is not None:
        unmapped = conform_capture(capabilities, registry)
        if unmapped:
            provenance["unmapped"] = unmapped
    if source.get("formats") is not None:
        if supplement is None or registry is None:
            raise SystemExit(f"{source['name']}: the source declares a formats supplement; pass "
                             f"--formats {source['name']}=PATH")
        provenance["formats"], counts = merge_formats(capabilities, core, driver, supplement,
                                                      registry, source)
    else:
        counts = {}
    profile["description"] = (f"Local capture of {source['profile']}; see provenance. "
                              f"{profile['description']}")
    doc = {
        "$schema": SCHEMA,
        "provenance": provenance,
        "capabilities": capabilities,
        "profiles": {profile_name(source["name"]): profile},
    }
    if counts:
        doc[SAMPLE_COUNTS_KEY] = {profile_name(source["name"]): counts}
    return doc


def conform_capture(capabilities, registry):
    """Removes what a `vulkaninfo --json` capture states that the profiles schema of the pinned
    registry cannot hold, and returns it as `provenance.unmapped` entries: an extension the
    registry does not define (a driver's unpublished one), and a pointer member vulkaninfo did not
    follow, which it writes as the string `NULL` where the schema types the array it points to.
    Every other member is kept verbatim, a pointer member written as its array included."""
    unmapped = []
    for block_name, block in sorted(capabilities.items()):
        extensions = block.get("extensions") or {}
        for name in sorted(extensions):
            if name not in registry.extension_types:
                del extensions[name]
                unmapped.append(f"{block_name}.extensions.{name} (not in the pinned registry)")
        for section in ("features", "properties"):
            for struct, members in sorted((block.get(section) or {}).items()):
                if not isinstance(members, dict) or registry.resolve(struct) not in registry.structs:
                    continue
                for name in sorted(members):
                    member = registry.member(registry.resolve(struct), name)
                    if member is not None and member.pointer and members[name] == "NULL":
                        del members[name]
                        unmapped.append(f"{block_name}.{section}.{struct}.{name} "
                                        "(pointer member written as NULL)")
    return unmapped


FORMAT_MEMBERS = ("linearTilingFeatures", "optimalTilingFeatures", "bufferFeatures")


def enum_value(registry, type_name, name):
    """The value `type_name`'s enumerant `name` has in the pinned registry."""
    for value, known in registry.enum_values.get(registry.resolve(type_name), {}).items():
        if known == name:
            return value
    raise SystemExit(f"{type_name} has no enumerant {name} in the pinned registry")


def flag_bits(registry, type_name, names):
    """The mask the flag `names` of bitmask `type_name` spell in the pinned registry."""
    bits = registry.bitmask_bits[registry.resolve(type_name)]
    mask = 0
    for name in names:
        mask |= enum_value(registry, bits, name)
    return mask


def merge_formats(capabilities, core, driver, supplement, registry, source):
    """Writes `VkFormatProperties3` beside the `VkFormatProperties` of every format the capture's
    capability blocks state, from `supplement`, the same device's `corpus-vulkan
    --capture-formats` document, and returns its provenance record with the color
    attachment sample counts of each stated format that has them, by name. Fails
    unless both describe one device (name, vendor, device, driver ID, driver version, driver info,
    API version, device type), each stated format is in the supplement with the capture's low 32
    bits and its color attachment sample counts, those counts belong to a format whose optimal
    tiling has `COLOR_ATTACHMENT`, and every format the supplement reports with a feature is one
    the capture states."""
    formats = source["formats"]
    device = supplement.get("device") or {}
    expected = {
        "device": core["deviceName"],
        "vendor_id": core["vendorID"],
        "device_id": core["deviceID"],
        "device_type": enum_value(registry, "VkPhysicalDeviceType", core["deviceType"]),
        "api_version": core["apiVersion"],
        "driver_id": enum_value(registry, "VkDriverId", driver["driverID"]),
        "driver_version": core["driverVersion"],
        "driver_info": driver["driverInfo"],
    }
    differ = [f"{key} {device.get(key)!r} (capture {value!r})" for key, value in expected.items()
              if device.get(key) != value]
    if differ:
        raise SystemExit(f"{source['name']}: the formats supplement is of another device: "
                         + "; ".join(differ))
    reported = supplement.get("formats") or {}
    # `VkFormatFeatureFlags2` shares its low bits with `VkFormatFeatureFlags`, but 32-bit
    # `VkFormatProperties` can hold only the bits `VkFormatFeatureFlagBits` defines (bit 31, the
    # storage read without a format, is 64-bit only).
    legacy_bits = 0
    for value in registry.enum_values.get(registry.bitmask_bits["VkFormatFeatureFlags"], {}):
        legacy_bits |= value
    color_attachment = flag_bits(registry, "VkFormatFeatureFlags2",
                                 ["VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT"])
    stated = set()
    counts = {}
    for block in capabilities.values():
        for name, entry in (block.get("formats") or {}).items():
            if name not in reported:
                raise SystemExit(f"{source['name']}: the formats supplement lacks {name}, "
                                 "which the capture states")
            masks = reported[name]
            legacy = entry.get("VkFormatProperties") or {}
            for member in FORMAT_MEMBERS:
                low = flag_bits(registry, "VkFormatFeatureFlags", legacy.get(member, []))
                if masks[member] & legacy_bits != low:
                    raise SystemExit(f"{source['name']}: {name}.{member}: the supplement's "
                                     f"32-bit features {masks[member] & legacy_bits:#x} are not "
                                     f"the capture's VkFormatProperties {low:#x}")
            converter = Converter(registry)
            try:
                entry["VkFormatProperties3"] = {
                    member: converter.flag_names("VkFormatFeatureFlags2", masks[member])
                    for member in FORMAT_MEMBERS}
            except Unmapped as error:
                raise SystemExit(f"{source['name']}: {name}: {error} is outside the pinned "
                                 "registry") from None
            samples = masks.get(SAMPLE_COUNTS_MEMBER)
            if samples is None:
                raise SystemExit(f"{source['name']}: the formats supplement states no "
                                 f"{SAMPLE_COUNTS_MEMBER} for {name}: recapture it")
            if samples and not masks["optimalTilingFeatures"] & color_attachment:
                raise SystemExit(f"{source['name']}: {name}: the supplement's color attachment "
                                 f"sample counts {samples:#x} are of a format whose optimal "
                                 "tiling has no COLOR_ATTACHMENT")
            if samples:
                try:
                    counts[name] = converter.flag_names("VkSampleCountFlags", samples)
                except Unmapped as error:
                    raise SystemExit(f"{source['name']}: {name}: {error} is outside the pinned "
                                     "registry") from None
            stated.add(name)
    unstated = sorted(name for name, masks in reported.items()
                      if name not in stated and any(masks[m] for m in FORMAT_MEMBERS))
    if unstated:
        raise SystemExit(f"{source['name']}: the formats supplement reports features of "
                         f"{', '.join(unstated)}, which the capture does not state")
    return {
        "command": formats["command"],
        "sha256": formats["sha256"],
        "queried": len(reported),
        "stated": len(stated),
        "color_attachment_sample_counts": len(counts),
    }, dict(sorted(counts.items()))


def unlisted_identifiers(registry):
    """The registry's opaque identifier members that `HOST_IDENTITY` does not withhold."""
    return sorted(registry.identifier_members() - HOST_IDENTITY)


def withhold_host_identity(doc):
    """The last step of every conversion: removes each `HOST_IDENTITY` member from the document's
    property structs (a struct or section left empty goes too) and lists what it removed, as
    `Struct.member`, in `provenance.withheld`. Returns `doc`, changed in place."""
    withheld = set()
    for block in doc["capabilities"].values():
        properties = block.get("properties")
        if properties is None:
            continue
        for struct in list(properties):
            members = properties[struct]
            for member in [name for name in members if name in HOST_IDENTITY]:
                del members[member]
                withheld.add(f"{struct}.{member}")
            if not members:
                del properties[struct]
        if not properties:
            del block["properties"]
    if withheld:
        doc["provenance"]["withheld"] = sorted(withheld)
    return doc


def render(doc):
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--offline", action="store_true", help="use only the download cache")
    parser.add_argument("--only", action="append", default=[], help="regenerate one profile")
    parser.add_argument("--capture", action="append", default=[], metavar="NAME=PATH",
                        help="a vulkaninfo --json capture for a vulkaninfo-kind profile")
    parser.add_argument("--formats", action="append", default=[], metavar="NAME=PATH",
                        help="a corpus-vulkan --capture-formats supplement for that capture")
    args = parser.parse_args()
    captures = dict(item.split("=", 1) for item in args.capture)
    supplements = dict(item.split("=", 1) for item in args.formats)
    registry = None
    formats = None
    args.out.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        if args.only and source["name"] not in args.only:
            continue
        target = args.out / f"{source['name']}.json"
        kind = source["kind"]
        recaptured = kind == "vulkaninfo" and source["name"] in captures
        presented = source.get("layer_presentation", False)
        if (kind in ("gpuinfo", "derived") or recaptured or presented) and registry is None:
            registry = Registry(fetch(REGISTRY["url"], REGISTRY["sha256"], args.offline))
            unlisted = unlisted_identifiers(registry)
            if unlisted:
                raise SystemExit(f"vk.xml declares device identifiers HOST_IDENTITY does not "
                                 f"withhold: {', '.join(unlisted)}")
        if kind in ("khronos", "derived") and formats is None:
            spec = required_formats.SPEC
            text = fetch(spec["url"], spec["sha256"], args.offline).decode("utf-8")
            formats = required_formats.required_formats(text, required_formats.VERSION_1_2)
        if kind == "gpuinfo":
            data = fetch(GPUINFO_REPORT.format(source["report"]), source["sha256"], args.offline)
            doc = convert_gpuinfo(registry, json.loads(data), source)
        elif kind == "khronos":
            doc = extract(fetch(source["url"], source["sha256"], args.offline), source, formats,
                          registry)
        elif kind == "derived":
            data = fetch(MINIMUM_REQUIREMENTS["url"], MINIMUM_REQUIREMENTS["sha256"], args.offline)
            doc = derive(registry, data, source, formats)
        elif kind == "vulkaninfo":
            path = captures.get(source["name"])
            if path is None:
                # The committed conversion stands.
                print(f"kept the capture in {target.name}: no --capture {source['name']}=PATH "
                      "(needs a device)")
                doc = json.loads(target.read_text())
            else:
                data = Path(path).read_bytes()
                if sha256(data) != source["sha256"]:
                    raise SystemExit(f"{path}: sha256 {sha256(data)} does not match pin")
                supplement = None
                if source.get("formats") is not None:
                    formats_path = supplements.get(source["name"])
                    if formats_path is None:
                        raise SystemExit(f"{source['name']}: pass --formats "
                                         f"{source['name']}=PATH with the capture")
                    formats_data = Path(formats_path).read_bytes()
                    if sha256(formats_data) != source["formats"]["sha256"]:
                        raise SystemExit(f"{formats_path}: sha256 {sha256(formats_data)} does "
                                         "not match pin")
                    supplement = json.loads(formats_data)
                doc = import_capture(data, source, supplement, registry)
        else:
            raise SystemExit(f"unknown source kind {kind}")
        text = render(withhold_host_identity(doc))
        if not target.exists() or target.read_text() != text:
            target.write_text(text)
            print(f"wrote {target.relative_to(REPO) if target.is_relative_to(REPO) else target}")
        else:
            print(f"unchanged {target.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
