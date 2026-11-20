#!/usr/bin/env python3
"""Offline structure and provenance lint for `crates/metal2vulkan-validation/profiles/*.json`.

Checks, for every committed profile file:

* canonical formatting (what `regen_device_profiles.py` writes), so a hand edit is visible;
* the Khronos profiles-0.8 schema id, exactly one profile named `VP_M2V_<file stem>`, the
  profile's required keys, and a closed capability set (every referenced block exists, every
  block is referenced);
* capability blocks use only the schema's section names;
* a `provenance` object whose required fields match its source kind (report id, driver version,
  pinned SHA-256, ...), and a `required_formats` record, where present, pinned to the source
  `required_formats.py` declares;
* no member that identifies the reporting host or driver build (`regen.HOST_IDENTITY`) anywhere in
  the document, and a `provenance.withheld` list, where present, naming only such members;
* a `vulkaninfo` profile of a Vulkan 1.3 or later device states `VkFormatProperties3` for every
  format, and its `provenance.formats` names the supplement pinned in the script, since
  `vulkaninfo --json` never writes that structure;
* a top-level `m2v_color_attachment_sample_counts` object, where present or recorded in
  `provenance.formats`, holds only this file's profile, as many formats as the record counts, each
  a format the profile states as a color attachment, with sample count bits only;
* the committed set is exactly the set the regeneration script declares.

With `--schema PATH` and the `jsonschema` package installed it also validates every file against
the Khronos schema; without them that step is skipped with a visible message. Exit status is
non-zero on any finding.
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import regen_device_profiles as regen  # noqa: E402
import required_formats  # noqa: E402

SECTIONS = {"extensions", "features", "properties", "formats", "queueFamiliesProperties",
            "videoProfiles"}
PROFILE_KEYS = {"version", "api-version", "label", "description", "capabilities"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED = {
    "gpuinfo": {"report_id", "url", "raw_url", "sha256", "device_name", "driver_version",
                "api_version", "os", "converter", "registry", "unmapped"},
    "khronos": {"url", "sha256", "source_profile", "source_profile_version"},
    "derived": {"base_url", "base_sha256", "base_profile", "registry", "rule",
                "dropped_features", "required_formats"},
    "vulkaninfo": {"command", "sha256", "source_profile", "device_name", "driver_version",
                   "api_version", "host", "captured", "transforms"},
}


def referenced(entries):
    names = set()
    for entry in entries:
        names.update(entry if isinstance(entry, list) else [entry])
    return names


def identity_keys(value, where):
    """Paths of every object key in `value` that `regen.HOST_IDENTITY` withholds."""
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{where}.{key}" if where else key
            if key in regen.HOST_IDENTITY:
                yield path
            yield from identity_keys(item, path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from identity_keys(item, f"{where}[{index}]")


def check_provenance(name, provenance, source, findings):
    kind = provenance.get("source")
    if kind not in REQUIRED:
        findings.append(f"{name}: provenance.source {kind!r} is not a known kind")
        return
    missing = sorted(REQUIRED[kind] - set(provenance))
    if missing:
        findings.append(f"{name}: provenance lacks {', '.join(missing)}")
    if source is not None and source["kind"] != kind:
        findings.append(f"{name}: provenance kind {kind} but the script declares {source['kind']}")
    for key in ("sha256", "base_sha256"):
        if key in provenance and not SHA256.match(str(provenance[key])):
            findings.append(f"{name}: provenance.{key} is not a SHA-256")
    formats = provenance.get("required_formats")
    if formats is not None and (formats.get("url") != required_formats.SPEC["url"]
                                or formats.get("sha256") != required_formats.SPEC["sha256"]):
        findings.append(f"{name}: provenance.required_formats disagrees with the script's pin")
    for entry in provenance.get("withheld", []):
        if str(entry).rpartition(".")[2] not in regen.HOST_IDENTITY:
            findings.append(f"{name}: provenance.withheld names {entry}, "
                            "not a host identity member")
    if kind == "gpuinfo":
        report = provenance.get("report_id")
        if not isinstance(report, int) or report <= 0:
            findings.append(f"{name}: provenance.report_id must be a positive integer")
        elif not str(provenance.get("url", "")).endswith(f"id={report}"):
            findings.append(f"{name}: provenance.url does not name report {report}")
        if source is not None and (source.get("report") != report
                                   or source.get("sha256") != provenance.get("sha256")):
            findings.append(f"{name}: provenance disagrees with the script's pin")
    elif kind in ("khronos", "vulkaninfo") and source is not None:
        if source.get("sha256") != provenance.get("sha256"):
            findings.append(f"{name}: provenance sha256 disagrees with the script's pin")
    if kind == "vulkaninfo" and source is not None:
        declared = (source.get("formats") or {}).get("sha256")
        recorded = (provenance.get("formats") or {}).get("sha256")
        if declared != recorded:
            findings.append(f"{name}: provenance.formats disagrees with the script's formats pin")


def check_format_flags2(name, doc, findings):
    """A `vulkaninfo` profile of a Vulkan 1.3 or later device states `VkFormatProperties3` for every
    format it states: `vulkaninfo --json` writes only `VkFormatProperties`, whose 32 bits omit the
    storage-without-format and depth-comparison bits, so without the supplement the profile would
    state them clear on a device that has them."""
    if (doc.get("provenance") or {}).get("source") != "vulkaninfo":
        return
    for profile_id, profile in (doc.get("profiles") or {}).items():
        version = tuple(int(part) for part in re.findall(r"\d+", str(profile.get("api-version")))[:2])
        if version < (1, 3):
            continue
        blocks = referenced(profile.get("capabilities", []) + profile.get("optionals", []))
        missing = sorted(format_name for block in blocks
                         for format_name, entry in
                         ((doc.get("capabilities") or {}).get(block, {}).get("formats") or {}).items()
                         if "VkFormatProperties3" not in entry)
        if missing:
            findings.append(f"{name}: {profile_id} is a Vulkan {version[0]}.{version[1]} device's "
                            f"vulkaninfo capture without VkFormatProperties3 for {len(missing)} "
                            f"formats (first {missing[0]}): regenerate it with its "
                            "corpus-vulkan --capture-formats supplement")


SAMPLE_COUNT_BITS = [f"VK_SAMPLE_COUNT_{n}_BIT" for n in (1, 2, 4, 8, 16, 32, 64)]
COLOR_ATTACHMENT_BITS = {"VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT",
                         "VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT"}


def check_sample_counts(name, stem, doc, findings):
    """The document's `regen.SAMPLE_COUNTS_KEY` object, where present or recorded in
    `provenance.formats`, holds exactly this file's profile and as many formats as the record
    counts, each a format the profile states with `COLOR_ATTACHMENT` in its optimal tiling, with a
    non-empty list of sample count bits in bit order."""
    block = doc.get(regen.SAMPLE_COUNTS_KEY)
    recorded = ((doc.get("provenance") or {}).get("formats") or {}).get(
        "color_attachment_sample_counts")
    if block is None and not recorded:
        return
    profile = regen.profile_name(stem)
    if not isinstance(block, dict) or list(block) != [profile] or not isinstance(block[profile],
                                                                                  dict):
        findings.append(f"{name}: {regen.SAMPLE_COUNTS_KEY} must hold exactly {profile}")
        return
    counts = block[profile]
    if recorded != len(counts):
        findings.append(f"{name}: {regen.SAMPLE_COUNTS_KEY} holds {len(counts)} formats, "
                        f"provenance.formats records {recorded}")
    attachments = set()
    for entry in (doc.get("profiles") or {}).values():
        for block_name in referenced(entry.get("capabilities", []) + entry.get("optionals", [])):
            formats = ((doc.get("capabilities") or {}).get(block_name) or {}).get("formats") or {}
            for format_name, properties in formats.items():
                for structure in ("VkFormatProperties", "VkFormatProperties3"):
                    optimal = (properties.get(structure) or {}).get("optimalTilingFeatures") or []
                    if COLOR_ATTACHMENT_BITS & set(optimal):
                        attachments.add(format_name)
    for format_name, bits in counts.items():
        if format_name not in attachments:
            findings.append(f"{name}: {regen.SAMPLE_COUNTS_KEY} names {format_name}, which the "
                            "profile does not state as a color attachment")
        if (not isinstance(bits, list) or not bits or any(bit not in SAMPLE_COUNT_BITS
                                                          for bit in bits)
                or bits != sorted(bits, key=SAMPLE_COUNT_BITS.index)):
            findings.append(f"{name}: {regen.SAMPLE_COUNTS_KEY} states {format_name}'s counts "
                            f"{bits!r}, not sample count bits in bit order")


def check_file(path, source, findings):
    name = path.name
    stem = path.stem
    text = path.read_text()
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as error:
        findings.append(f"{name}: not JSON: {error}")
        return None
    if text != regen.render(doc):
        findings.append(f"{name}: not in canonical form (regenerate instead of hand editing)")
    if doc.get("$schema") != regen.SCHEMA:
        findings.append(f"{name}: $schema must be {regen.SCHEMA}")
    profiles = doc.get("profiles") or {}
    expected = regen.profile_name(path.stem)
    if list(profiles) != [expected]:
        findings.append(f"{name}: must define exactly one profile, {expected}")
    capabilities = doc.get("capabilities") or {}
    used = set()
    for profile_id, profile in profiles.items():
        missing = sorted(PROFILE_KEYS - set(profile))
        if missing:
            findings.append(f"{name}: {profile_id} lacks {', '.join(missing)}")
        if not re.match(r"^\d+\.\d+\.\d+$", str(profile.get("api-version", ""))):
            findings.append(f"{name}: {profile_id} api-version is not major.minor.patch")
        if profile.get("profiles"):
            findings.append(f"{name}: {profile_id} requires other profiles; files are standalone")
        used |= referenced(profile.get("capabilities", []) + profile.get("optionals", []))
    for block in sorted(used - set(capabilities)):
        findings.append(f"{name}: capability block {block} is referenced but missing")
    for block in sorted(set(capabilities) - used):
        findings.append(f"{name}: capability block {block} is never referenced")
    for block, sections in capabilities.items():
        unknown = sorted(set(sections) - SECTIONS)
        if unknown:
            findings.append(f"{name}: block {block} has unknown sections {', '.join(unknown)}")
    for path in identity_keys(doc, ""):
        findings.append(f"{name}: {path} identifies the host (regenerate; the script withholds it)")
    check_format_flags2(name, doc, findings)
    check_sample_counts(name, stem, doc, findings)
    provenance = doc.get("provenance")
    if not isinstance(provenance, dict):
        findings.append(f"{name}: missing provenance object")
    else:
        check_provenance(name, provenance, source, findings)
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=regen.OUT_DIR)
    parser.add_argument("--schema", type=Path, help="Khronos profiles schema JSON to validate against")
    args = parser.parse_args()
    findings = []
    sources = {source["name"]: source for source in regen.SOURCES}
    files = sorted(args.dir.glob("*.json")) + sorted((args.dir / regen.BELOW_FLOOR_DIR).glob("*.json"))
    names = {path.stem for path in files}
    for missing in sorted(set(sources) - names):
        findings.append(f"{missing}.json: declared by the regeneration script but not committed")
    for extra in sorted(names - set(sources)):
        findings.append(f"{extra}.json: not declared by the regeneration script")
    docs = {}
    for path in files:
        source = sources.get(path.stem)
        if source is not None and path != regen.target_of(args.dir, source):
            findings.append(f"{path.stem}.json: belongs at {regen.target_of(args.dir, source)}")
        doc = check_file(path, source, findings)
        if doc is not None:
            docs[path.name] = doc
    if args.schema:
        try:
            import jsonschema
        except ImportError:
            print("skipped schema validation: the jsonschema package is not installed")
        else:
            validator = jsonschema.Draft7Validator(json.loads(args.schema.read_text()))
            for name, doc in docs.items():
                for error in validator.iter_errors(doc):
                    path = "/".join(str(p) for p in error.absolute_path)
                    findings.append(f"{name}: schema: {path}: {error.message[:200]}")
    else:
        print("skipped schema validation: pass --schema PATH (needs the jsonschema package)")
    for finding in findings:
        print(finding)
    print(f"checked {len(files)} profiles: {len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
