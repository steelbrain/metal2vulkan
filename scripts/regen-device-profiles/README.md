# regen-device-profiles

Regenerates the committed device profiles in `crates/metal2vulkan-validation/profiles/*.json` from pinned public
sources, and lints them. Each file is a Khronos profiles-schema (0.8) document with exactly one
profile, `VP_M2V_<file stem>`, plus a top-level `provenance` object. Nothing is invented: a field
the source does not report is absent, and a source field the converter cannot represent is listed
in `provenance.unmapped` rather than guessed.

```sh
python3 scripts/regen-device-profiles/regen_device_profiles.py            # fetch, convert, write
python3 scripts/regen-device-profiles/regen_device_profiles.py --offline  # cache only
python3 scripts/regen-device-profiles/check_device_profiles.py            # offline lint
python3 scripts/regen-device-profiles/check_device_profiles.py --schema profiles-0.8-latest.json
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/regen-device-profiles
```

The profiles are data for the translator's `device` module and validation's
per-profile checks; nothing in the product reads them at runtime.

## Sources

| Kind | Input | Transformation |
|---|---|---|
| `gpuinfo` | raw vulkan.gpuinfo.org report, `api/v2/getreport.php?id=N`, pinned by SHA-256 | typed conversion through the pinned Vulkan-Headers `vk.xml` |
| `khronos` | a pinned KhronosGroup/Vulkan-Profiles file | one profile and its capability blocks, verbatim; profile renamed; a profile that states no formats gets the Required Format Support tables (below); a source marked `layer_presentation` gets the block `m2v_layer_presentation`: each feature its core version requires that it omits, and each member the product reads of the core property structures of its version (`CORE_PROPERTIES`, or the value a feature it states makes exact, `FEATURE_EXACT_PROPERTIES`) and of the property structures of the extensions it lists (`EXTENSION_PROPERTIES`) that it omits, at the value the product's reader takes, and each format several of its blocks state, at the union of their flags (the layer keeps the last block's entry of a format; the reader ORs them), so that `VK_LAYER_KHRONOS_profiles` reports the profile as the product reads it (the regeneration fails on a version or an extension property structure the product reads whose values it does not model) |
| `derived` | `VP_LUNARG_minimum_requirements_1_2` | features reduced to `vk.xml`'s unconditional 1.0–1.2 requirements; a base value the Vulkan 1.2 specification contradicts replaced by the specification's (`SPEC_CORRECTIONS`, named in `provenance.spec_corrections`; the regeneration fails when the base states another value or none); plus the Vulkan 1.2 Required Format Support tables (below), and the block `m2v_unrequired_properties`: the weakest value the specification permits for each property the product reads, and `VK_LAYER_KHRONOS_profiles` sets, that the base leaves without a requirement (strict lines, quad operations in all stages), which the layer would otherwise take from the device |
| `vulkaninfo` | a local `vulkaninfo --json` capture, and for a Vulkan 1.3+ device its `corpus-vulkan --capture-formats` supplement | the capture's profile (a portability-subset one where the device has the subset); identical queue families collapsed (schema `uniqueItems`); an extension the pinned registry does not define and a pointer member written as `NULL` removed and named in `provenance.unmapped`; each format's `VkFormatProperties3` from the supplement |

gpuinfo's own profile export (`api/v3/getprofile.php`) returned HTTP 500 for every report when
these files were produced, so reports are converted here. The converter was cross-checked against
gpuinfo's export of report 49700 committed in KhronosGroup/Vulkan-Profiles
(`VP_LUNARG_desktop_baseline_2026/vp_gpuinfo_intel_r__iris_r__xe_graphics__tgl_gt2__26_1_3_void_unknown.json`):
every shared value agrees except where the export drops information (format feature bit 16,
subgroup rotate bits, `maxTimelineSemaphoreValueDifference` clamped to `INT64_MAX`), spells an
enum through an alias, or rounds a float.

### Required Format Support

`required_formats.py` reads the Vulkan 1.2 specification's "Required Format Support" tables from
its AsciiDoc source (Vulkan-Docs `v1.2.203`, the last 1.2 revision, `chapters/formats.txt`,
pinned by SHA-256): `vk.xml` describes formats but not the features they must support, and
Vulkan-Profiles encodes no core format table. It evaluates the section's conditionals for core
1.2 (`VK_VERSION_1_0..1_2`, no extension), lays every `formats-mandatory-features-*` table out as
a grid, and reads `{sym1}` as unconditional and every other symbol through its table's footnote,
which must match one of three forms exactly (a feature condition, a choice between two formats, a
choice between whole tables); anything else fails the regeneration. Formats required to support
`SAMPLED_IMAGE` also get `TRANSFER_SRC`/`TRANSFER_DST` (the section's own sentence).

A requirements profile that states no formats (`derived`, and `khronos` without a `formats`
section) gets the block `m2v_vulkan12_required_formats` (every unconditional entry), one block
`m2v_vulkan12_required_formats_if_<feature>` per feature it requires that makes conditional
entries unconditional, and `provenance.required_formats`: the pin, the rule, and every
conditional requirement (`any_of` tables, at least one of which holds in full; `if_feature` when
it binds only a device with that feature). A choice between formats or tables has no single-device
form, so no profile states it; the Rust test `device::tests::required_formats` checks every
committed profile against both parts.

The `vulkaninfo` capture opens a device, so the script never takes it. Capture it by hand on the
host whose profile it records. `vulkaninfo --json` never writes `VkFormatProperties3`, so a source
that declares `formats` also takes the same device's supplement from `corpus-vulkan
--capture-formats`, which selects the device as a venue run does and writes the device's identity
and the three 64-bit `VkFormatFeatureFlags2` masks of every format the pinned registry names that
the device exposes, with the sample counts of a color attachment of each format whose optimal
tiling has `COLOR_ATTACHMENT`:

```sh
vulkaninfo --json
corpus-vulkan --capture-formats formats.json --device-type gpu
python3 scripts/regen-device-profiles/regen_device_profiles.py --only moltenvk-m3 \
    --capture moltenvk-m3=VP_VULKANINFO_Apple_M3_Max_0_2_2210.json \
    --formats moltenvk-m3=formats.json
```

The conversion fails unless both files describe one device (name, vendor and device IDs, driver
ID, driver version, driver info, API version, device type), every format the capture states is in
the supplement with the capture's 32-bit features, and every format the supplement reports with a
feature is one the capture states; it then writes each format's `VkFormatProperties3` beside its
`VkFormatProperties`, the color attachment formats' sample counts under the top-level object
`m2v_color_attachment_sample_counts` (keyed by profile name; the schema has no structure for them),
and `provenance.formats` names the supplement's pin and how many formats have counts. A
supplement without the counts, or with counts for a format whose optimal tiling has no
`COLOR_ATTACHMENT`, fails the conversion. Without `--capture` the
committed conversion is kept. A new capture or
supplement changes its pinned SHA-256 in `SOURCES`, so bump it deliberately. The lint refuses a
`vulkaninfo` profile of a Vulkan 1.3 or later device that lacks `VkFormatProperties3` for a format
it states, or whose `provenance.formats` is not the supplement pinned in `SOURCES`.

A venue's profile (`crates/metal2vulkan-validation/profiles/README.md`) is recaptured, with its supplement, from the
venue's own device whenever its driver changes, in the commit that declares the venue's new
identity (`docs/VALIDATION.md`).

Downloads are cached by SHA-256 under `.cache/device-profiles/` (gitignored). Rerunning the script
over unchanged sources is byte-identical; bump `GENERATED` when regenerated content changes.

### Deviations

A document states no deviation from the specifications. Where a driver's answer to what the
specifications leave open is Metal's, the product takes it from the release-keyed table of what
each driver release is measured with (`crates/metal2vulkan/src/device/driver_facts.rs`), keyed on the document's
`driverID`, `vendorID` and `driverVersion`, and refuses a document that still holds the former
top-level `m2v_deviations` object.

### Host identity

The last step of every conversion withholds the device-property members that identify the host or
driver build a source came from rather than describe the device (`HOST_IDENTITY`): every
`VK_UUID_SIZE`/`VK_LUID_SIZE` byte array of a device property struct (`pipelineCacheUUID`,
`deviceUUID`, `driverUUID`, `deviceLUID`, `optimalTilingLayoutUUID`, `shaderBinaryUUID`,
`shaderModuleIdentifierAlgorithmUUID`), the PCI address, and the DRM node numbers. A struct left
empty is dropped, and `provenance.withheld` lists what was removed. Regeneration fails when the
pinned `vk.xml` declares a UUID or LUID property the set does not name; the lint and the Rust test
`device::tests::host_identity` reject any committed file that carries one. A `vulkaninfo`
recapture goes through the same step.
