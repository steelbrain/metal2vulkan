import copy
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import check_device_profiles as check
import regen_device_profiles as regen

REGISTRY = b"""<registry>
<types>
  <type category="enum" name="VkPointClippingBehavior"/>
  <type category="bitmask" name="VkSampleCountFlags" requires="VkSampleCountFlagBits"/>
  <type category="struct" name="VkExtent2D">
    <member><type>uint32_t</type> <name>width</name></member>
    <member><type>uint32_t</type> <name>height</name></member>
  </type>
  <type category="struct" name="VkPhysicalDeviceToyProperties" structextends="VkPhysicalDeviceProperties2">
    <member><type>VkStructureType</type> <name>sType</name></member>
    <member optional="true"><type>void</type>* <name>pNext</name></member>
    <member><type>VkBool32</type> <name>flag</name></member>
    <member><type>uint64_t</type> <name>big</name></member>
    <member><type>VkSampleCountFlags</type> <name>counts</name></member>
    <member><type>VkPointClippingBehavior</type> <name>clipping</name></member>
    <member><type>VkExtent2D</type> <name>extent</name></member>
    <member><type>uint8_t</type> <name>uuid</name>[<enum>VK_UUID_SIZE</enum>]</member>
    <member optional="true" len="count"><type>uint32_t</type>* <name>pValues</name></member>
  </type>
  <type category="struct" name="VkPhysicalDeviceToyPropertiesEXT" alias="VkPhysicalDeviceToyProperties"/>
</types>
<enums name="API Constants"><enum value="16" name="VK_UUID_SIZE"/></enums>
<enums name="VkPointClippingBehavior" type="enum">
  <enum value="0" name="VK_POINT_CLIPPING_BEHAVIOR_ALL_CLIP_PLANES"/>
</enums>
<enums name="VkSampleCountFlagBits" type="bitmask">
  <enum bitpos="0" name="VK_SAMPLE_COUNT_1_BIT"/>
  <enum bitpos="2" name="VK_SAMPLE_COUNT_4_BIT"/>
</enums>
<feature api="vulkan" name="VK_VERSION_1_0">
  <require><feature name="robustBufferAccess" struct="VkPhysicalDeviceFeatures"/></require>
  <require depends="VK_EXT_x"><feature name="conditional" struct="VkPhysicalDeviceFeatures"/></require>
</feature>
<extensions>
  <extension name="VK_EXT_toy" number="7" supported="vulkan">
    <require><type name="VkPhysicalDeviceToyPropertiesEXT"/></require>
  </extension>
</extensions>
</registry>"""


class ConverterTest(unittest.TestCase):
    def setUp(self):
        self.registry = regen.Registry(REGISTRY)
        self.converter = regen.Converter(self.registry)

    def test_members_convert_by_registry_type(self):
        raw = {"flag": "true", "big": "18446744073709551615", "counts": 5, "clipping": 0,
               "extent": [8, 4], "uuid": list(range(16))}
        value, unmapped = self.converter.fill("VkPhysicalDeviceToyPropertiesEXT", raw, "toy")
        self.assertEqual(unmapped, [])
        self.assertEqual(value, {
            "flag": True,
            "big": 18446744073709551615,
            "counts": ["VK_SAMPLE_COUNT_1_BIT", "VK_SAMPLE_COUNT_4_BIT"],
            "clipping": "VK_POINT_CLIPPING_BEHAVIOR_ALL_CLIP_PLANES",
            "extent": {"width": 8, "height": 4},
            "uuid": list(range(16)),
        })

    def test_unrepresentable_fields_are_reported_not_guessed(self):
        raw = {"counts": 2, "flag": 3, "pValues": [1], "stranger": 1}
        value, unmapped = self.converter.fill("VkPhysicalDeviceToyProperties", raw, "toy")
        self.assertEqual(value, {})
        self.assertEqual(sorted(unmapped), [
            "toy.counts (VkSampleCountFlags bit 1)",
            "toy.flag (VkBool32 value 3)",
            "toy.pValues (pointer member)",
            "toy.stranger (not a member)",
        ])

    def test_extension_rows_fold_into_the_extension_struct(self):
        output, unmapped = {}, []
        rows = [{"extension": "VK_EXT_toy", "name": "big", "value": "7"},
                {"extension": "VK_EXT_toy", "name": "nope", "value": "1"}]
        regen.group_extended(self.registry, rows, "VkPhysicalDeviceProperties2", output,
                             unmapped, self.converter)
        self.assertEqual(output, {"VkPhysicalDeviceToyPropertiesEXT": {"big": 7}})
        self.assertEqual(unmapped,
                         ["VK_EXT_toy.nope (no VkPhysicalDeviceProperties2 struct declares it)"])

    def test_required_features_ignore_conditional_requirements(self):
        self.assertEqual(self.registry.required_features, {"robustBufferAccess"})

    def test_identifier_members_are_the_uuid_and_luid_arrays_of_device_properties(self):
        self.assertEqual(self.registry.identifier_members(), {"uuid"})
        # A registry identifier the withheld set does not name stops the regeneration.
        self.assertEqual(regen.unlisted_identifiers(self.registry), ["uuid"])


class WithholdTest(unittest.TestCase):
    def test_host_identity_is_withheld_and_named_in_provenance(self):
        doc = {
            "provenance": {"source": "gpuinfo"},
            "capabilities": {
                "device": {
                    "extensions": {"VK_EXT_pci_bus_info": 2},
                    "properties": {
                        "VkPhysicalDeviceProperties": {"deviceName": "toy",
                                                       "pipelineCacheUUID": [1] * 16},
                        "VkPhysicalDevicePCIBusInfoPropertiesEXT": {
                            "pciDomain": 0, "pciBus": 1, "pciDevice": 0, "pciFunction": 0},
                        "VkPhysicalDeviceVulkan11Properties": {
                            "deviceUUID": [2] * 16, "deviceLUID": [3] * 8,
                            "deviceLUIDValid": False, "subgroupSize": 32},
                    },
                },
                "drm": {"properties": {"VkPhysicalDeviceDrmPropertiesEXT": {"renderMinor": 128}}},
                "floor": {"features": {"VkPhysicalDeviceFeatures": {"shaderInt16": True}}},
            },
        }
        regen.withhold_host_identity(doc)
        self.assertEqual(doc["capabilities"], {
            "device": {
                "extensions": {"VK_EXT_pci_bus_info": 2},
                "properties": {
                    "VkPhysicalDeviceProperties": {"deviceName": "toy"},
                    "VkPhysicalDeviceVulkan11Properties": {"deviceLUIDValid": False,
                                                           "subgroupSize": 32},
                },
            },
            "drm": {},
            "floor": {"features": {"VkPhysicalDeviceFeatures": {"shaderInt16": True}}},
        })
        self.assertEqual(doc["provenance"]["withheld"], [
            "VkPhysicalDeviceDrmPropertiesEXT.renderMinor",
            "VkPhysicalDevicePCIBusInfoPropertiesEXT.pciBus",
            "VkPhysicalDevicePCIBusInfoPropertiesEXT.pciDevice",
            "VkPhysicalDevicePCIBusInfoPropertiesEXT.pciDomain",
            "VkPhysicalDevicePCIBusInfoPropertiesEXT.pciFunction",
            "VkPhysicalDeviceProperties.pipelineCacheUUID",
            "VkPhysicalDeviceVulkan11Properties.deviceLUID",
            "VkPhysicalDeviceVulkan11Properties.deviceUUID",
        ])
        # Idempotent: withholding a withheld document changes nothing.
        again = json.loads(json.dumps(doc))
        self.assertEqual(regen.withhold_host_identity(again), doc)

    def test_a_document_without_host_identity_gains_no_provenance(self):
        doc = {"provenance": {"source": "khronos"},
               "capabilities": {"block": {"properties": {
                   "VkPhysicalDeviceProperties": {"limits": {"maxImageDimension1D": 4096}}}}}}
        expected = json.loads(json.dumps(doc))
        self.assertEqual(regen.withhold_host_identity(doc), expected)


class CheckTest(unittest.TestCase):
    def write(self, directory, stem, doc):
        path = Path(directory) / f"{stem}.json"
        path.write_text(regen.render(doc))
        return path

    def minimal(self, stem):
        return {
            "$schema": regen.SCHEMA,
            "provenance": {"source": "khronos", "url": "u", "sha256": "0" * 64,
                           "source_profile": "VP_X_y", "source_profile_version": 1},
            "capabilities": {"block": {"features": {}}},
            "profiles": {regen.profile_name(stem): {
                "version": 1, "api-version": "1.2.0", "label": "l", "description": "d",
                "capabilities": ["block"]}},
        }

    def test_well_formed_file_has_no_findings(self):
        with tempfile.TemporaryDirectory() as directory:
            findings = []
            check.check_file(self.write(directory, "toy", self.minimal("toy")), None, findings)
            self.assertEqual(findings, [])

    def test_missing_provenance_and_dangling_blocks_are_findings(self):
        doc = self.minimal("toy")
        del doc["provenance"]
        doc["capabilities"]["orphan"] = {}
        doc["profiles"]["VP_M2V_toy"]["capabilities"].append("absent")
        with tempfile.TemporaryDirectory() as directory:
            findings = []
            check.check_file(self.write(directory, "toy", doc), None, findings)
        self.assertEqual(findings, [
            "toy.json: capability block absent is referenced but missing",
            "toy.json: capability block orphan is never referenced",
            "toy.json: missing provenance object",
        ])

    def test_host_identity_is_a_finding(self):
        doc = self.minimal("toy")
        doc["capabilities"]["block"]["properties"] = {
            "VkPhysicalDeviceVulkan11Properties": {"deviceUUID": [0] * 16, "subgroupSize": 32}}
        doc["provenance"]["withheld"] = ["VkPhysicalDeviceVulkan11Properties.subgroupSize"]
        with tempfile.TemporaryDirectory() as directory:
            findings = []
            check.check_file(self.write(directory, "toy", doc), None, findings)
        self.assertEqual(findings, [
            "toy.json: capabilities.block.properties.VkPhysicalDeviceVulkan11Properties.deviceUUID "
            "identifies the host (regenerate; the script withholds it)",
            "toy.json: provenance.withheld names VkPhysicalDeviceVulkan11Properties.subgroupSize, "
            "not a host identity member",
        ])

    def test_hand_edits_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, "toy", self.minimal("toy"))
            path.write_text(json.dumps(json.loads(path.read_text())))
            findings = []
            check.check_file(path, None, findings)
        self.assertEqual(findings,
                         ["toy.json: not in canonical form (regenerate instead of hand editing)"])


class DeriveTest(unittest.TestCase):
    """A derived profile states the properties the product reads, and the profiles layer sets,
    that its base leaves without a requirement."""

    FORMATS = {"unconditional": {}, "conditional": [], "tables": [], "not_represented": []}

    def base(self, properties):
        profile = regen.MINIMUM_REQUIREMENTS["profile"]
        return json.dumps({
            "capabilities": {"base": {
                "features": {"VkPhysicalDeviceFeatures": {"robustBufferAccess": True}},
                "properties": properties,
            }},
            "profiles": {profile: {"api-version": "1.2.131", "capabilities": ["base"]}},
        })

    def derive(self, properties, corrections=None):
        source = {"name": "toy-floor", "label": "toy"}
        with mock.patch.dict(regen.SPEC_CORRECTIONS, corrections or {}, clear=True):
            return regen.derive(regen.Registry(REGISTRY), self.base(properties), source,
                                self.FORMATS)

    # A base value the specification contradicts, in place of SPEC_CORRECTIONS.
    CORRECTIONS = {
        ("VkPhysicalDeviceVulkan12Properties", "supportedDepthResolveModes"):
            ([], ["VK_RESOLVE_MODE_SAMPLE_ZERO_BIT"]),
        ("VkPhysicalDeviceProperties", "maxDrawIndexedIndexValue"): (16777216, 16777215),
    }

    def test_a_base_value_the_specification_contradicts_is_corrected(self):
        doc = self.derive({
            "VkPhysicalDeviceProperties": {"limits": {"maxDrawIndexedIndexValue": 16777216}},
            "VkPhysicalDeviceVulkan12Properties": {"supportedDepthResolveModes": []},
        }, corrections=self.CORRECTIONS)
        properties = doc["capabilities"]["base"]["properties"]
        self.assertEqual(properties["VkPhysicalDeviceProperties"]["limits"],
                         {"maxDrawIndexedIndexValue": 16777215})
        self.assertEqual(properties["VkPhysicalDeviceVulkan12Properties"],
                         {"supportedDepthResolveModes": ["VK_RESOLVE_MODE_SAMPLE_ZERO_BIT"]})
        self.assertEqual(doc["provenance"]["spec_corrections"], [
            "VkPhysicalDeviceProperties.maxDrawIndexedIndexValue: 16777216 -> 16777215",
            "VkPhysicalDeviceVulkan12Properties.supportedDepthResolveModes: [] -> "
            "[\"VK_RESOLVE_MODE_SAMPLE_ZERO_BIT\"]",
        ])

    def test_a_base_that_states_another_value_or_none_fails_regeneration(self):
        with self.assertRaises(SystemExit):
            self.derive({
                "VkPhysicalDeviceProperties": {"limits": {"maxDrawIndexedIndexValue": 16777215}},
                "VkPhysicalDeviceVulkan12Properties": {"supportedDepthResolveModes": []},
            }, corrections=self.CORRECTIONS)
        with self.assertRaises(SystemExit):
            self.derive({
                "VkPhysicalDeviceProperties": {"limits": {"maxDrawIndexedIndexValue": 16777216}},
            }, corrections=self.CORRECTIONS)

    def test_unrequired_properties_are_stated_at_their_weakest(self):
        doc = self.derive({"VkPhysicalDeviceVulkan11Properties": {"subgroupSize": 1}})
        block = doc["capabilities"][regen.UNREQUIRED_PROPERTIES_BLOCK]
        self.assertEqual(block, {"properties": {
            "VkPhysicalDeviceProperties": {"limits": {"strictLines": False}},
            "VkPhysicalDeviceVulkan11Properties": {"subgroupQuadOperationsInAllStages": False}}})
        [profile] = doc["profiles"].values()
        self.assertIn(regen.UNREQUIRED_PROPERTIES_BLOCK, profile["capabilities"])
        self.assertIn(regen.UNREQUIRED_PROPERTIES_BLOCK, doc["provenance"]["rule"])

    def test_a_base_that_states_one_fails_regeneration(self):
        stated = {"VkPhysicalDeviceVulkan11Properties": {"subgroupQuadOperationsInAllStages": True}}
        with self.assertRaises(SystemExit):
            self.derive(stated)
        with self.assertRaises(SystemExit):
            self.derive({"VkPhysicalDeviceProperties": {"limits": {"strictLines": True}}})

    def test_a_limit_is_told_apart_from_the_limits_a_base_states(self):
        doc = self.derive({"VkPhysicalDeviceProperties": {"limits": {"maxImageDimension1D": 4096}}})
        block = doc["capabilities"][regen.UNREQUIRED_PROPERTIES_BLOCK]
        self.assertEqual(block["properties"]["VkPhysicalDeviceProperties"],
                         {"limits": {"strictLines": False}})


PRESENTATION_REGISTRY = b"""<registry>
<types>
  <type category="struct" name="VkPhysicalDeviceSubgroupSizeControlProperties" structextends="VkPhysicalDeviceProperties2">
    <member><type>VkStructureType</type> <name>sType</name></member>
    <member optional="true"><type>void</type>* <name>pNext</name></member>
    <member><type>uint32_t</type> <name>minSubgroupSize</name></member>
    <member><type>uint32_t</type> <name>maxSubgroupSize</name></member>
    <member><type>uint32_t</type> <name>maxComputeWorkgroupSubgroups</name></member>
    <member><type>VkShaderStageFlags</type> <name>requiredSubgroupSizeStages</name></member>
  </type>
  <type category="struct" name="VkPhysicalDeviceSubgroupSizeControlPropertiesEXT" alias="VkPhysicalDeviceSubgroupSizeControlProperties"/>
  <type category="struct" name="VkPhysicalDeviceMaintenance4Properties" structextends="VkPhysicalDeviceProperties2">
    <member><type>VkStructureType</type> <name>sType</name></member>
    <member optional="true"><type>void</type>* <name>pNext</name></member>
    <member><type>VkDeviceSize</type> <name>maxBufferSize</name></member>
  </type>
  <type category="struct" name="VkPhysicalDeviceMaintenance4PropertiesKHR" alias="VkPhysicalDeviceMaintenance4Properties"/>
</types>
<feature api="vulkan" name="VK_VERSION_1_0">
  <require><feature name="robustBufferAccess" struct="VkPhysicalDeviceFeatures"/></require>
</feature>
<feature api="vulkan" name="VK_VERSION_1_2">
  <require>
    <feature name="shaderSubgroupExtendedTypes" struct="VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures"/>
    <feature name="shaderSubgroupExtendedTypes" struct="VkPhysicalDeviceVulkan12Features"/>
  </require>
  <require depends="VK_KHR_x"><feature name="conditional" struct="VkPhysicalDeviceVulkan12Features"/></require>
</feature>
<feature api="vulkan" name="VK_VERSION_1_3">
  <require><feature name="robustImageAccess" struct="VkPhysicalDeviceVulkan13Features"/></require>
</feature>
<extensions>
  <extension name="VK_EXT_subgroup_size_control" number="226" supported="vulkan">
    <require><type name="VkPhysicalDeviceSubgroupSizeControlPropertiesEXT"/></require>
  </extension>
  <extension name="VK_KHR_maintenance4" number="414" supported="vulkan">
    <require><type name="VkPhysicalDeviceMaintenance4PropertiesKHR"/></require>
  </extension>
</extensions>
</registry>"""


class LayerPresentationTest(unittest.TestCase):
    """A requirements document is completed with what the product reads that it omits, at the
    reader's value, so the profiles layer reports it as the product reads it."""

    # A few core members the product reads, in place of CORE_PROPERTIES.
    CORE = {
        "VkPhysicalDeviceProperties": (0, {"limits": {"maxImageDimension3D": 256,
                                                      "minTexelOffset": -8}}),
        "VkPhysicalDeviceVulkan12Properties": (2, {"filterMinmaxSingleComponentFormats": False}),
        "VkPhysicalDeviceVulkan13Properties": (3, {"maxBufferSize": 1073741824}),
    }
    STATED_CORE = {
        "VkPhysicalDeviceProperties": {"limits": {"maxImageDimension3D": 2048,
                                                  "minTexelOffset": -8}},
        "VkPhysicalDeviceVulkan12Properties": {"filterMinmaxSingleComponentFormats": True},
    }

    def setUp(self):
        self.registry = regen.Registry(PRESENTATION_REGISTRY)

    def present(self, capabilities, api_version="1.2.148"):
        with mock.patch.dict(regen.CORE_PROPERTIES, self.CORE, clear=True):
            return regen.layer_presentation(self.registry, capabilities, api_version, "toy")

    def test_omitted_required_features_and_read_properties_are_stated(self):
        capabilities = {"base": {
            "extensions": {"VK_EXT_subgroup_size_control": 1},
            "features": {"VkPhysicalDeviceFeatures": {"robustBufferAccess": True}},
            "properties": {
                "VkPhysicalDeviceProperties": {"limits": {"maxImageDimension3D": 2048}},
                "VkPhysicalDeviceSubgroupSizeControlPropertiesEXT": {"maxSubgroupSize": 64},
            },
        }}
        block, named = self.present(capabilities)
        # Features stated once, in the core structure; nothing conditional, nothing of a later
        # version, nothing already stated; and likewise the core properties.
        self.assertEqual(block, {
            "features": {"VkPhysicalDeviceVulkan12Features": {
                "shaderSubgroupExtendedTypes": True}},
            "properties": {
                "VkPhysicalDeviceProperties": {"limits": {"minTexelOffset": -8}},
                "VkPhysicalDeviceVulkan12Properties": {
                    "filterMinmaxSingleComponentFormats": False},
                "VkPhysicalDeviceSubgroupSizeControlPropertiesEXT": {
                    "minSubgroupSize": 0,
                    "maxComputeWorkgroupSubgroups": 0,
                    "requiredSubgroupSizeStages": [],
                },
            },
        })
        self.assertEqual(named, [
            "VkPhysicalDeviceProperties.limits.minTexelOffset",
            "VkPhysicalDeviceSubgroupSizeControlPropertiesEXT.maxComputeWorkgroupSubgroups",
            "VkPhysicalDeviceSubgroupSizeControlPropertiesEXT.minSubgroupSize",
            "VkPhysicalDeviceSubgroupSizeControlPropertiesEXT.requiredSubgroupSizeStages",
            "VkPhysicalDeviceVulkan12Features.shaderSubgroupExtendedTypes",
            "VkPhysicalDeviceVulkan12Properties.filterMinmaxSingleComponentFormats",
        ])

    def test_a_later_version_adds_its_requirements_and_structures(self):
        capabilities = {"base": {
            "features": {"VkPhysicalDeviceFeatures": {"robustBufferAccess": True},
                         "VkPhysicalDeviceVulkan12Features": {
                             "shaderSubgroupExtendedTypes": True}},
            "properties": self.STATED_CORE,
        }}
        self.assertEqual(self.present(capabilities), ({}, []))
        block, named = self.present(capabilities, "1.3.204")
        self.assertEqual(block, {
            "features": {"VkPhysicalDeviceVulkan13Features": {"robustImageAccess": True}},
            "properties": {"VkPhysicalDeviceVulkan13Properties": {"maxBufferSize": 1073741824}},
        })
        self.assertEqual(named, ["VkPhysicalDeviceVulkan13Features.robustImageAccess",
                                 "VkPhysicalDeviceVulkan13Properties.maxBufferSize"])

    def test_a_format_several_blocks_state_is_stated_at_their_union(self):
        # The profiles layer keeps the last block's entry of a format; the product's reader ORs
        # every block's flags. A format stated once, or whose last entry is the union, needs
        # nothing.
        sampled = {"VkFormatProperties": {"optimalTilingFeatures": [
            "VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT"], "bufferFeatures": [
            "VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT"]}}
        storage = {"VkFormatProperties": {"optimalTilingFeatures": [
            "VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT"]}}
        capabilities = {
            "base": {"features": {"VkPhysicalDeviceFeatures": {"robustBufferAccess": True},
                                  "VkPhysicalDeviceVulkan12Features": {
                                      "shaderSubgroupExtendedTypes": True}},
                     "properties": self.STATED_CORE},
            "required": {"formats": {"VK_FORMAT_R8_UNORM": sampled,
                                     "VK_FORMAT_R8_UINT": sampled,
                                     "VK_FORMAT_R16_UNORM": storage}},
            "conditional": {"formats": {"VK_FORMAT_R8_UNORM": storage,
                                        "VK_FORMAT_R8_UINT": sampled}},
        }
        block, named = self.present(capabilities)
        self.assertEqual(block, {"formats": {"VK_FORMAT_R8_UNORM": {"VkFormatProperties": {
            "optimalTilingFeatures": ["VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT",
                                      "VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT"],
            "bufferFeatures": ["VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT"]}}}})
        self.assertEqual(named, ["formats.VK_FORMAT_R8_UNORM"])
        capabilities["conditional"]["formats"]["VK_FORMAT_R8_UNORM"] = {
            "VkFormatProperties2": {"formatProperties": {}}}
        with self.assertRaises(SystemExit):
            self.present(capabilities)

    def test_a_member_a_stated_feature_makes_exact_is_stated_at_that_value(self):
        core = dict(self.CORE)
        core["VkPhysicalDeviceProperties"] = (0, {"limits": {"maxDrawIndexedIndexValue": 16777215}})
        features = {"VkPhysicalDeviceFeatures": {"robustBufferAccess": True},
                    "VkPhysicalDeviceVulkan12Features": {"shaderSubgroupExtendedTypes": True}}
        with mock.patch.dict(regen.CORE_PROPERTIES, core, clear=True):
            block, _ = regen.layer_presentation(
                self.registry, {"base": {"features": features}}, "1.2.148", "toy")
            self.assertEqual(block["properties"]["VkPhysicalDeviceProperties"],
                             {"limits": {"maxDrawIndexedIndexValue": 16777215}})
            features["VkPhysicalDeviceFeatures"]["fullDrawIndexUint32"] = True
            block, _ = regen.layer_presentation(
                self.registry, {"base": {"features": features}}, "1.2.148", "toy")
            self.assertEqual(block["properties"]["VkPhysicalDeviceProperties"],
                             {"limits": {"maxDrawIndexedIndexValue": 4294967295}})

    def test_a_version_whose_structures_are_not_modelled_fails_regeneration(self):
        with self.assertRaises(SystemExit):
            self.present({"base": {}}, "1.4.309")

    def test_an_omitted_member_of_an_unmodelled_structure_fails_regeneration(self):
        capabilities = {"base": {"extensions": {"VK_KHR_maintenance4": 1}}}
        with self.assertRaises(SystemExit):
            self.present(capabilities)

FORMAT_REGISTRY = b"""<registry>
<types>
  <type category="enum" name="VkPhysicalDeviceType"/>
  <type category="enum" name="VkDriverId"/>
  <type category="bitmask" name="VkFormatFeatureFlags" requires="VkFormatFeatureFlagBits"/>
  <type category="bitmask" name="VkFormatFeatureFlags2" bitvalues="VkFormatFeatureFlagBits2"/>
  <type category="bitmask" name="VkSampleCountFlags" requires="VkSampleCountFlagBits"/>
</types>
<enums name="VkPhysicalDeviceType" type="enum">
  <enum value="1" name="VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU"/>
</enums>
<enums name="VkDriverId" type="enum">
  <enum value="3" name="VK_DRIVER_ID_MESA_RADV"/>
  <enum value="14" name="VK_DRIVER_ID_MOLTENVK"/>
</enums>
<enums name="VkFormatFeatureFlagBits" type="bitmask">
  <enum bitpos="0" name="VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT"/>
  <enum bitpos="1" name="VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT"/>
  <enum bitpos="7" name="VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT"/>
</enums>
<enums name="VkFormatFeatureFlagBits2" type="bitmask" bitwidth="64">
  <enum bitpos="0" name="VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT"/>
  <enum bitpos="1" name="VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT"/>
  <enum bitpos="7" name="VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT"/>
  <enum bitpos="31" name="VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT"/>
  <enum bitpos="32" name="VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT"/>
</enums>
<enums name="VkSampleCountFlagBits" type="bitmask">
  <enum bitpos="0" name="VK_SAMPLE_COUNT_1_BIT"/>
  <enum bitpos="1" name="VK_SAMPLE_COUNT_2_BIT"/>
  <enum bitpos="2" name="VK_SAMPLE_COUNT_4_BIT"/>
</enums>
<extensions/>
</registry>"""


class FormatsSupplementTest(unittest.TestCase):
    """A vulkaninfo capture's formats gain `VkFormatProperties3`, and its color attachment formats
    their sample counts, from the same device's `corpus-vulkan --capture-formats` supplement,
    which must describe that device and agree with the capture's 32-bit bits; the lint holds a
    1.3+ vulkaninfo profile to it."""

    SOURCE = {"name": "toy", "kind": "vulkaninfo", "sha256": "1" * 64,
              "formats": {"command": "corpus-vulkan --capture-formats", "sha256": "2" * 64}}
    CORE = {"deviceName": "Toy GPU", "vendorID": 4098, "deviceID": 5567,
            "deviceType": "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU", "apiVersion": 4210999,
            "driverVersion": 109060100}
    DRIVER = {"driverID": "VK_DRIVER_ID_MESA_RADV", "driverInfo": "Mesa 26.2.4"}

    def setUp(self):
        self.registry = regen.Registry(FORMAT_REGISTRY)

    def capabilities(self):
        return {"device": {"formats": {"VK_FORMAT_R8_UNORM": {"VkFormatProperties": {
            "linearTilingFeatures": [],
            "optimalTilingFeatures": ["VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT",
                                      "VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT"],
            "bufferFeatures": []}}}}}

    def supplement(self):
        return {
            "device": {"device": "Toy GPU", "vendor_id": 4098, "device_id": 5567,
                       "device_type": 1, "api_version": 4210999, "driver_id": 3,
                       "driver_version": 109060100, "driver_info": "Mesa 26.2.4"},
            "formats": {
                "VK_FORMAT_R8_UNORM": {"linearTilingFeatures": 0,
                                       "optimalTilingFeatures": 3 | (1 << 31) | (1 << 32),
                                       "bufferFeatures": 0, "colorAttachmentSampleCounts": 0},
                "VK_FORMAT_R8_SNORM": {"linearTilingFeatures": 0, "optimalTilingFeatures": 0,
                                       "bufferFeatures": 0, "colorAttachmentSampleCounts": 0},
            },
        }

    def merge(self, capabilities, supplement):
        return regen.merge_formats(capabilities, self.CORE, self.DRIVER, supplement,
                                   self.registry, self.SOURCE)

    def test_the_64_bit_masks_are_written_beside_the_capture_bits(self):
        capabilities = self.capabilities()
        provenance, counts = self.merge(capabilities, self.supplement())
        entry = capabilities["device"]["formats"]["VK_FORMAT_R8_UNORM"]
        self.assertEqual(entry["VkFormatProperties3"], {
            "linearTilingFeatures": [],
            "optimalTilingFeatures": ["VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT",
                                      "VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT",
                                      "VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT",
                                      "VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT"],
            "bufferFeatures": []})
        self.assertEqual(provenance, {"command": "corpus-vulkan --capture-formats",
                                      "sha256": "2" * 64, "queried": 2, "stated": 1,
                                      "color_attachment_sample_counts": 0})
        self.assertEqual(counts, {})

    def attachment(self):
        """The capture and supplement with R8_UNORM a color attachment of 1 and 4 samples."""
        capabilities = self.capabilities()
        legacy = capabilities["device"]["formats"]["VK_FORMAT_R8_UNORM"]["VkFormatProperties"]
        legacy["optimalTilingFeatures"].append("VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT")
        supplement = self.supplement()
        masks = supplement["formats"]["VK_FORMAT_R8_UNORM"]
        masks["optimalTilingFeatures"] |= 1 << 7
        masks["colorAttachmentSampleCounts"] = 0b101
        return capabilities, supplement

    def test_color_attachment_sample_counts_are_returned_by_format(self):
        capabilities, supplement = self.attachment()
        provenance, counts = self.merge(capabilities, supplement)
        self.assertEqual(counts, {"VK_FORMAT_R8_UNORM": ["VK_SAMPLE_COUNT_1_BIT",
                                                         "VK_SAMPLE_COUNT_4_BIT"]})
        self.assertEqual(provenance["color_attachment_sample_counts"], 1)

    def test_a_supplement_without_sample_counts_or_with_them_on_no_attachment_fails(self):
        capabilities, supplement = self.attachment()
        del supplement["formats"]["VK_FORMAT_R8_UNORM"]["colorAttachmentSampleCounts"]
        with self.assertRaises(SystemExit) as raised:
            self.merge(capabilities, supplement)
        self.assertIn("recapture it", str(raised.exception))
        supplement = self.supplement()
        supplement["formats"]["VK_FORMAT_R8_UNORM"]["colorAttachmentSampleCounts"] = 1
        with self.assertRaises(SystemExit) as raised:
            self.merge(self.capabilities(), supplement)
        self.assertIn("no COLOR_ATTACHMENT", str(raised.exception))

    def test_a_supplement_of_another_device_fails(self):
        supplement = self.supplement()
        supplement["device"]["driver_info"] = "Mesa 26.2.5"
        with self.assertRaises(SystemExit) as raised:
            self.merge(self.capabilities(), supplement)
        self.assertIn("another device", str(raised.exception))

    def linear_attachment(self):
        """The capture with R8_UNORM's linear tiling a color attachment, the supplement without."""
        capabilities = self.capabilities()
        legacy = capabilities["device"]["formats"]["VK_FORMAT_R8_UNORM"]["VkFormatProperties"]
        legacy["linearTilingFeatures"].append("VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT")
        return capabilities, self.supplement()

    def test_a_linear_attachment_feature_must_agree(self):
        capabilities, supplement = self.linear_attachment()
        with self.assertRaises(SystemExit) as raised:
            self.merge(capabilities, supplement)
        self.assertIn("32-bit features", str(raised.exception))

    def test_low_bits_that_disagree_with_the_capture_fail(self):
        supplement = self.supplement()
        supplement["formats"]["VK_FORMAT_R8_UNORM"]["optimalTilingFeatures"] = 1
        with self.assertRaises(SystemExit) as raised:
            self.merge(self.capabilities(), supplement)
        self.assertIn("32-bit features", str(raised.exception))

    def test_every_stated_format_and_every_supported_one_must_be_in_both(self):
        supplement = self.supplement()
        del supplement["formats"]["VK_FORMAT_R8_UNORM"]
        with self.assertRaises(SystemExit) as raised:
            self.merge(self.capabilities(), supplement)
        self.assertIn("lacks VK_FORMAT_R8_UNORM", str(raised.exception))
        supplement = self.supplement()
        supplement["formats"]["VK_FORMAT_R8_SNORM"]["bufferFeatures"] = 1
        with self.assertRaises(SystemExit) as raised:
            self.merge(self.capabilities(), supplement)
        self.assertIn("VK_FORMAT_R8_SNORM, which the capture does not state",
                      str(raised.exception))

    def test_the_lint_refuses_a_1_3_capture_without_the_64_bit_masks(self):
        doc = CheckTest.minimal(CheckTest(), "toy")
        doc["provenance"] = {"source": "vulkaninfo", "command": "vulkaninfo --json",
                             "sha256": "1" * 64, "source_profile": "VP_VULKANINFO_toy",
                             "device_name": "Toy GPU", "driver_version": "1", "api_version": "1.3.0",
                             "host": "h", "captured": "c", "transforms": []}
        doc["capabilities"]["block"] = self.capabilities()["device"]
        doc["profiles"]["VP_M2V_toy"]["api-version"] = "1.3.250"
        with tempfile.TemporaryDirectory() as directory:
            findings = []
            check.check_file(CheckTest().write(directory, "toy", doc), self.SOURCE, findings)
            self.assertEqual(findings, [
                "toy.json: VP_M2V_toy is a Vulkan 1.3 device's vulkaninfo capture without "
                "VkFormatProperties3 for 1 formats (first VK_FORMAT_R8_UNORM): regenerate it "
                "with its corpus-vulkan --capture-formats supplement",
                "toy.json: provenance.formats disagrees with the script's formats pin",
            ])
            self.merge(doc["capabilities"], self.supplement())
            doc["provenance"]["formats"] = {"sha256": "2" * 64}
            findings = []
            check.check_file(CheckTest().write(directory, "toy", doc), self.SOURCE, findings)
            self.assertEqual(findings, [])
            doc["profiles"]["VP_M2V_toy"]["api-version"] = "1.2.198"
            del doc["capabilities"]["block"]["formats"]["VK_FORMAT_R8_UNORM"]["VkFormatProperties3"]
            findings = []
            check.check_file(CheckTest().write(directory, "toy", doc), self.SOURCE, findings)
            self.assertEqual(findings, [])


    def test_the_lint_holds_the_sample_counts_to_the_record_and_the_stated_attachments(self):
        doc = CheckTest.minimal(CheckTest(), "toy")
        doc["provenance"] = {"source": "vulkaninfo", "command": "vulkaninfo --json",
                             "sha256": "1" * 64, "source_profile": "VP_VULKANINFO_toy",
                             "device_name": "Toy GPU", "driver_version": "1", "api_version": "1.2.0",
                             "host": "h", "captured": "c", "transforms": []}
        capabilities, supplement = self.attachment()
        record, counts = self.merge(capabilities, supplement)
        doc["capabilities"]["block"] = capabilities["device"]
        doc["provenance"]["formats"] = record
        doc[regen.SAMPLE_COUNTS_KEY] = {"VP_M2V_toy": counts}

        def lint(doc):
            with tempfile.TemporaryDirectory() as directory:
                findings = []
                check.check_file(CheckTest().write(directory, "toy", doc), self.SOURCE, findings)
                return findings

        self.assertEqual(lint(doc), [])
        unstated = copy.deepcopy(doc)
        unstated[regen.SAMPLE_COUNTS_KEY]["VP_M2V_toy"]["VK_FORMAT_R8_SNORM"] = [
            "VK_SAMPLE_COUNT_1_BIT"]
        self.assertEqual(lint(unstated), [
            "toy.json: m2v_color_attachment_sample_counts holds 2 formats, provenance.formats "
            "records 1",
            "toy.json: m2v_color_attachment_sample_counts names VK_FORMAT_R8_SNORM, which the "
            "profile does not state as a color attachment",
        ])
        unordered = copy.deepcopy(doc)
        unordered[regen.SAMPLE_COUNTS_KEY]["VP_M2V_toy"]["VK_FORMAT_R8_UNORM"].reverse()
        self.assertEqual(len(lint(unordered)), 1)
        other = copy.deepcopy(doc)
        other[regen.SAMPLE_COUNTS_KEY] = {"VP_M2V_other": counts}
        self.assertEqual(lint(other), [
            "toy.json: m2v_color_attachment_sample_counts must hold exactly VP_M2V_toy"])
        missing = copy.deepcopy(doc)
        del missing[regen.SAMPLE_COUNTS_KEY]
        self.assertEqual(lint(missing), [
            "toy.json: m2v_color_attachment_sample_counts must hold exactly VP_M2V_toy"])


class ConformCaptureTest(unittest.TestCase):
    """What a vulkaninfo capture states that the pinned registry's profiles schema cannot hold is
    removed and named under `provenance.unmapped`; everything else stays verbatim."""

    def test_unpublished_extensions_and_unfollowed_pointers_are_named_not_kept(self):
        registry = regen.Registry(REGISTRY)
        capabilities = {"device": {
            "extensions": {"VK_EXT_toy": 1, "VK_NV_internal_thing": 1},
            "properties": {"VkPhysicalDeviceToyPropertiesEXT": {"flag": True, "pValues": "NULL"}},
        }}
        unmapped = regen.conform_capture(capabilities, registry)
        self.assertEqual(capabilities, {"device": {
            "extensions": {"VK_EXT_toy": 1},
            "properties": {"VkPhysicalDeviceToyPropertiesEXT": {"flag": True}},
        }})
        self.assertEqual(unmapped, [
            "device.extensions.VK_NV_internal_thing (not in the pinned registry)",
            "device.properties.VkPhysicalDeviceToyPropertiesEXT.pValues "
            "(pointer member written as NULL)",
        ])

    def test_a_pointer_member_written_as_its_array_is_kept(self):
        registry = regen.Registry(REGISTRY)
        capabilities = {"device": {
            "properties": {"VkPhysicalDeviceToyProperties": {"pValues": [1, 2]}}}}
        self.assertEqual(regen.conform_capture(capabilities, registry), [])
        self.assertEqual(capabilities["device"]["properties"]["VkPhysicalDeviceToyProperties"],
                         {"pValues": [1, 2]})

    def test_the_lint_accepts_the_schemas_video_profiles_section(self):
        doc = CheckTest.minimal(CheckTest(), "toy")
        doc["capabilities"]["block"]["videoProfiles"] = []
        with tempfile.TemporaryDirectory() as directory:
            findings = []
            check.check_file(CheckTest().write(directory, "toy", doc), None, findings)
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
