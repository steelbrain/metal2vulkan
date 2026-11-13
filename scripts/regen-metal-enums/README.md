# regen-metal-enums

Generates `crates/metal2vulkan/src/metal/generated.rs`, the Metal enums and option sets of the translator's `metal`
module, and `crates/metal2vulkan-air/src/metal/generated.rs`, the ones the AIR front
end reads (`AIR_ENUMS`: `MTLDataType`, which `metal` re-exports), from the
macOS SDK's `Metal.framework` headers.

Only enumerator names and numeric values are extracted: they are the Metal API's wire contract
(guest command streams and Metal-shaped descriptors carry them as raw ordinals). No header text is
copied. Each listed `NS_ENUM` becomes a `metal_enum!` invocation (a closed Rust enum whose
`TryFrom<u32>` rejects unknown ordinals) and each listed `NS_OPTIONS` a `metal_flags!` invocation
(a bit set whose `TryFrom<u32>` rejects unknown bits); the macros live in `crates/metal2vulkan/src/metal/mod.rs` and,
for the front end's enums, `crates/metal2vulkan-air/src/metal/mod.rs`.
Deprecated spellings with the same value are recorded as aliases. Value expressions other than
integer literals, shifts, ors, and references to earlier enumerators fail generation.

Values the headers do not declare but Metal takes are the script's `PRIVATE` input, one line per
value after the type's header enumerators, each with a comment giving the evidence that Metal takes
it (the device, the macOS release, what Metal did with it); generation fails once the headers
declare one. `PRIVATE_DOCS` gives such a value a doc comment on its variant, where a consumer needs
to know what Metal does with it.

It also generates `scripts/conformance-coverage/metal_enumerators.tsv` from the `TABLE_ONLY` list:
the SDK's enumerators and ordinals for the Metal enums and option sets the tree declares **by
hand** rather than from these headers, because the compile-and-plan layer never reads them. No Rust
is generated for those, and none should be; the table is the conformance suite's option-space
denominator for them, so that a hand-written variant list is measured against the SDK rather than
against itself. One row per name, with `role` saying whether it is the value or a deprecated
spelling of one.

`MTLResourceOptions` is in no list: the SDK writes its values in terms of `MTLCPUCacheMode`'s and
`MTLStorageMode`'s enumerators shifted by `#define`d amounts, which `evaluate` fails on rather than
guessing.

The output records the SDK version it came from, so `--check` passes only against that SDK. To add
a type, append its Metal name to `ENUMS`, `AIR_ENUMS`, `OPTIONS` or `TABLE_ONLY` and regenerate.
Never hand-edit the output.

```sh
# regenerate from the active Xcode SDK (xcrun), or pass --headers DIR --sdk-version V
python3 scripts/regen-metal-enums/regen_metal_enums.py
# idempotence: fail if a committed file differs from a fresh generation
python3 scripts/regen-metal-enums/regen_metal_enums.py --check
# generator unit tests (no SDK needed)
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/regen-metal-enums
```

Requires macOS with Xcode (or the Command Line Tools SDK) and Python 3.
