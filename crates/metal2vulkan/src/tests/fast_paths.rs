//! docs/FAST-PATHS.md, the ledger of the defaults that trade exactness against Metal for making
//! time, stays accurate by construction: it names every accuracy choice of `DevicePolicy` with each
//! of its values, as `field = Type::Value`, and every precision note a fast default, a
//! not-yet-built accurate form, an opt-in or a listed device gap records, with its degrade code;
//! and it names no policy value that no longer exists. The lists below are exhaustive matches: a
//! field, a policy value or a precision note added to the crate fails to compile here until it is
//! classed, and so reaches the ledger.

use crate::device::{
    BatchOrderPolicy, CompletionPolicy, DecodedTexelsPolicy, DevicePolicy, FloatMathsPolicy,
    HalfBlendPolicy, HalfRoundingPolicy, HalfTexelBitsPolicy, HeapCompressionPolicy,
    MultisampleDepthTwinPolicy, NanMinMaxPolicy, RenderLoweringPolicy, ResolvePolicy,
    RoundingOrderPolicy, SamplingPolicy, TexelConversionPolicy, TileStorePolicy,
    Unorm8LinearFilterPolicy, YCbCrFilterPolicy,
};
use crate::pipeline::PrecisionNote;
use std::collections::BTreeSet;

/// The file's text with every run of whitespace one space, so a name may wrap across lines.
fn text(relative: &str) -> String {
    let path = format!("{}/{relative}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// One accuracy choice of `DevicePolicy`: the field, its value under `DevicePolicy::UNTRUSTED`
/// (the default), and every value of its type, each as `Type::Value`.
struct Choice {
    field: &'static str,
    default: &'static str,
    values: Vec<&'static str>,
}

/// The [`Choice`] of `policy.field`, whose type is `ty` with the values listed. The match is
/// exhaustive, so a value added to the type fails to compile until it is listed.
macro_rules! choice {
    ($policy:ident . $field:ident : $ty:ident { $($v:ident),+ $(,)? }) => {{
        fn name(v: $ty) -> &'static str {
            match v {
                $($ty::$v => concat!(stringify!($ty), "::", stringify!($v)),)+
            }
        }
        Choice {
            field: stringify!($field),
            default: name($policy.$field),
            values: vec![$(name($ty::$v)),+],
        }
    }};
}

/// Every accuracy choice of `DevicePolicy`.
fn choices() -> Vec<Choice> {
    let p = DevicePolicy::UNTRUSTED;
    // Every field is classed: one added to `DevicePolicy` fails to compile here until it is
    // listed below as an accuracy choice or here as none.
    let DevicePolicy {
        // How pointers and handles resolve, and whether accesses and guests are contained: no
        // answer of Metal's arithmetic or sampling.
        pointer_domain: _,
        handle_domain: _,
        containment: _,
        isolation: _,
        // The accuracy choices, below.
        unorm8_linear_filter: _,
        sampling: _,
        ycbcr_filter: _,
        float_maths: _,
        texel_conversion: _,
        half_rounding: _,
        nan_and_min_max: _,
        rounding_order: _,
        half_blend: _,
        half_texel_bits: _,
        // The runtime's: where a multisampled `Depth32Float` texture's texels are held, the
        // texels of a carried texture's decoded twin, a heap's compressed
        // texture's image, when the host sees a command buffer complete, a loaded attachment's
        // tile store, a multisampled color attachment's resolve, how a batch begins after the
        // earlier ones of its queue, and where a render encoder's calls are lowered.
        multisample_depth_twin: _,
        decoded_texels: _,
        heap_compression: _,
        completion: _,
        tile_store: _,
        resolve: _,
        batch_order: _,
        render_lowering: _,
    } = p;
    vec![
        choice!(p.unorm8_linear_filter: Unorm8LinearFilterPolicy { Exact, DeviceWithinOneCode }),
        choice!(p.sampling: SamplingPolicy { NativeDevice, Exact }),
        choice!(p.ycbcr_filter: YCbCrFilterPolicy { Device, Exact }),
        choice!(p.float_maths: FloatMathsPolicy { Native, Exact }),
        choice!(p.texel_conversion: TexelConversionPolicy { Device, Exact }),
        choice!(p.half_rounding: HalfRoundingPolicy { Native, Exact }),
        choice!(p.nan_and_min_max: NanMinMaxPolicy { Native, Exact }),
        choice!(p.rounding_order: RoundingOrderPolicy { Native, Exact }),
        choice!(p.half_blend: HalfBlendPolicy { Device, Exact }),
        choice!(p.half_texel_bits: HalfTexelBitsPolicy { Device, Exact }),
        choice!(p.multisample_depth_twin: MultisampleDepthTwinPolicy { Device, Exact }),
        choice!(p.decoded_texels: DecodedTexelsPolicy { Compact, Exact }),
        choice!(p.heap_compression: HeapCompressionPolicy { Compressed, Uncompressed }),
        choice!(p.completion: CompletionPolicy { Grouped, Exact }),
        choice!(p.tile_store: TileStorePolicy { Device, Exact }),
        choice!(p.resolve: ResolvePolicy { Device, Exact }),
        choice!(p.batch_order: BatchOrderPolicy { Ends, Exact }),
        choice!(p.render_lowering: RenderLoweringPolicy { AtCall, Worker }),
    ]
}

/// Where a precision note stands in the ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ledger {
    /// Recorded by a fast default whose accurate form is a switch: a row of the first table.
    FastDefault,
    /// The default's form where the accurate one is not built yet: a row of the first table.
    NotBuilt,
    /// Recorded only under a non-default opt-in: the list of switches whose default is Metal's.
    OptIn,
    /// A device gap, no speed choice: a row of the second table.
    DeviceGap,
    /// None of these: docs/API.md section 9 states it.
    Elsewhere,
}

/// Every precision note with its place in the ledger. The match is exhaustive, so a note added
/// to `PrecisionNote` fails to compile until it is classed, and no note is classed twice.
macro_rules! notes {
    ($($place:ident: [$($note:ident),+ $(,)?]),+ $(,)?) => {{
        fn place(n: PrecisionNote) -> Ledger {
            match n {
                $($(PrecisionNote::$note => Ledger::$place,)+)+
            }
        }
        vec![$($((stringify!($note), place(PrecisionNote::$note)),)+)+]
    }};
}

fn notes() -> Vec<(&'static str, Ledger)> {
    notes! {
        FastDefault: [
            DeviceSampling,
            DeviceTexelConversion,
            HalfTexelConversion,
            NativeTranscendentals,
            NativeDivision,
            NativeFma,
            NativeNanEncoding,
            NativeMinMax,
            NativeDot,
            NativeMix,
            NativeHalfConversion,
            NativeHalfQuantize,
            HalfOutputPrecision,
        ],
        NotBuilt: [
            ExtendedRangeBlendSource,
            ExtendedRangeFloatBlendSource,
            BicubicFilterArithmetic,
        ],
        OptIn: [DeviceUnorm8LinearFilter],
        DeviceGap: [
            DeviceDenormals,
            DeviceTexelDenormals,
            RestrictedDepthRange,
            DeviceSignedZeroInfNan,
            DeviceInterpolation,
            HalfOutputFastMathBlend,
            UnmeasuredFetchBlend,
            ReductionOrder,
            FunctionStorageAssumed,
            ColorAttachmentGroups,
        ],
        Elsewhere: [
            F16FmaDoubleRounding,
            F16IntConvDoubleRounding,
            GradFromBias,
            DeviceTranscendentals,
            PixelSamplerScaling,
            DeviceArithmetic,
            DeviceRayIntersection,
            FastMathRewrites,
            TessellationWinding,
            TessellationFactorEdgeCases,
            TessellationFactorScale,
            TessellationFactorInstance,
            Fp8Value,
            MultisamplePosition,
            RasterizationRateMap,
            SparseResidency,
            DeviceNanEncoding,
            CubeGradientFootprint,
            HalfSampleFilter,
            SrgbTexelDecode,
            FloatSampleFilter,
            DeviceGatherFootprint,
            CubeFaceReciprocal,
            QuadsCountedBeforeDiscard,
            DeviceDepthComparison,
            BeyondMeasuredDeviation,
            ViewBaseTexelOffset,
            DeviceColorConversion,
            QuadsSplitByClipping,
            VertexSimdgroupGrouping,
            FragmentSimdgroupGrouping,
            BlendedWriteRounding,
            CoverageReadsRoundedAlpha,
            RotatedBlendRounding,
            LinkedSamplerNumbering,
            ComputedSampleDepth,
            ArgumentPixelSamplerScaling,
            DiscardDropsSimdgroupStores,
            HelperBranchDropsSimdgroupStores,
            AnisotropicFilter,
            DeviceAnisotropicFilter,
            DevicePointSize,
            ClippedCoverageSplit,
            LoopedSampleOutputs,
            TessellationTriangleOrder,
            UnmeasuredGroupSpeculation,
            NotMipmappedAnisotropicFilter,
            DeviceSampleLevel,
        ],
    }
}

/// A note's degrade code, the runtime's kind code without its prefix: the note's name in
/// lowercase words joined by hyphens (`NativeHalfConversion`: `native-half-conversion`).
fn code(note: &str) -> String {
    let mut code = String::new();
    let mut previous: Option<char> = None;
    for c in note.chars() {
        if c.is_ascii_uppercase()
            && previous.is_some_and(|p| p.is_ascii_lowercase() || p.is_ascii_digit())
        {
            code.push('-');
        }
        code.push(c.to_ascii_lowercase());
        previous = Some(c);
    }
    code
}

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// The identifier ending at byte `end` of `text`.
fn ident_before(text: &str, end: usize) -> &str {
    let head = &text[..end];
    let start = head
        .char_indices()
        .rev()
        .take_while(|&(_, c)| is_ident(c))
        .last()
        .map_or(end, |(i, _)| i);
    &head[start..]
}

/// The identifier starting at byte `start` of `text`.
fn ident_after(text: &str, start: usize) -> &str {
    let tail = &text[start..];
    let end = tail.find(|c: char| !is_ident(c)).unwrap_or(tail.len());
    &tail[..end]
}

/// Every `Type::Value` the text names whose type is a policy's (`...Policy`, but
/// `DevicePolicy`).
fn policy_values(text: &str) -> BTreeSet<String> {
    text.match_indices("::")
        .filter_map(|(i, _)| {
            let ty = ident_before(text, i);
            let value = ident_after(text, i + 2);
            (ty.ends_with("Policy") && ty != "DevicePolicy" && !value.is_empty())
                .then(|| format!("{ty}::{value}"))
        })
        .collect()
}

/// Every `field = Type::Value` the text names, as its field and its `Type::Value`.
fn assignments(text: &str) -> BTreeSet<(String, String)> {
    text.match_indices(" = ")
        .filter_map(|(i, _)| {
            let field = ident_before(text, i);
            let ty = ident_after(text, i + 3);
            let rest = &text[i + 3 + ty.len()..];
            let value = ident_after(rest.strip_prefix("::")?, 0);
            (!field.is_empty() && ty.ends_with("Policy") && !value.is_empty())
                .then(|| (field.to_owned(), format!("{ty}::{value}")))
        })
        .collect()
}

/// The ledger names every accuracy choice of `DevicePolicy` with each of its values, as
/// `field = Type::Value`, the default's among them.
#[test]
fn the_ledger_names_every_accuracy_choice_and_value() {
    let ledger = text("../../docs/FAST-PATHS.md");
    for choice in choices() {
        assert!(
            choice.values.contains(&choice.default),
            "{}: the default {} is not among its values",
            choice.field,
            choice.default
        );
        for value in &choice.values {
            let named = format!("{} = {value}", choice.field);
            assert!(
                ledger.contains(&named),
                "docs/FAST-PATHS.md does not name `{named}`: give the choice its row (AGENTS.md, \
                 Translation honesty)"
            );
        }
    }
}

/// The ledger names no policy value, and no `field = Type::Value`, that the crate no longer has:
/// a removed or renamed choice leaves its row.
#[test]
fn the_ledger_names_only_choices_the_policy_has() {
    let ledger = text("../../docs/FAST-PATHS.md");
    let choices = choices();
    let values: BTreeSet<&str> = choices
        .iter()
        .flat_map(|c| c.values.iter().copied())
        .collect();
    for value in policy_values(&ledger) {
        assert!(
            values.contains(value.as_str()),
            "docs/FAST-PATHS.md names `{value}`, which is no policy value"
        );
    }
    let assignments = assignments(&ledger);
    assert!(
        !assignments.is_empty(),
        "the ledger's switches are not found"
    );
    for (field, value) in assignments {
        assert!(
            choices
                .iter()
                .any(|c| c.field == field && c.values.contains(&value.as_str())),
            "docs/FAST-PATHS.md names `{field} = {value}`, which is no choice of DevicePolicy"
        );
    }
}

/// The ledger names every precision note a fast default, a not-yet-built accurate form, an
/// opt-in or a listed device gap records, each with its degrade code, which docs/API.md's
/// catalogue lists.
#[test]
fn the_ledger_names_every_note_of_its_rows_with_its_code() {
    let ledger = text("../../docs/FAST-PATHS.md");
    let catalogue = text("../../docs/API.md");
    let words: BTreeSet<&str> = ledger
        .split(|c: char| !is_ident(c))
        .filter(|w| !w.is_empty())
        .collect();
    let mut named = 0;
    for (note, place) in notes() {
        if place == Ledger::Elsewhere {
            continue;
        }
        named += 1;
        assert!(
            words.contains(note),
            "docs/FAST-PATHS.md does not name `{note}` ({place:?})"
        );
        let code = code(note);
        assert!(
            ledger.contains(&format!("`{code}`")),
            "docs/FAST-PATHS.md does not name `{note}`'s code `{code}`"
        );
        assert!(
            catalogue.contains(&format!("`metal2vulkan-degrade-{code}`")),
            "docs/API.md's catalogue does not list `metal2vulkan-degrade-{code}`, `{note}`'s code"
        );
    }
    assert!(named > 0);
}

/// The helpers read what the ledger writes.
#[test]
fn the_ledger_readers_read_names_and_switches() {
    assert_eq!(code("NativeHalfConversion"), "native-half-conversion");
    assert_eq!(
        code("DeviceUnorm8LinearFilter"),
        "device-unorm8-linear-filter"
    );
    assert_eq!(code("DeviceSignedZeroInfNan"), "device-signed-zero-inf-nan");
    let text = "`sampling = SamplingPolicy::Exact` and `DevicePolicy::exact_answers()`, \
                `HalfRoundingPolicy::Exact`.";
    assert_eq!(
        policy_values(text),
        BTreeSet::from([
            "SamplingPolicy::Exact".to_owned(),
            "HalfRoundingPolicy::Exact".to_owned()
        ])
    );
    assert_eq!(
        assignments(text),
        BTreeSet::from([("sampling".to_owned(), "SamplingPolicy::Exact".to_owned())])
    );
}

/// The runtime's choice of a heap's compressed texture's image (`heap_compression`) moves no
/// committed profile's fingerprint: no profile carries it, so no translation follows it.
#[test]
fn the_runtime_s_heap_compression_choice_keys_no_profile() {
    use crate::spirv::testing::committed_profiles_with;
    let compressed = committed_profiles_with(DevicePolicy::UNTRUSTED).expect("profiles");
    let uncompressed = committed_profiles_with(DevicePolicy {
        heap_compression: HeapCompressionPolicy::Uncompressed,
        ..DevicePolicy::UNTRUSTED
    })
    .expect("profiles");
    assert_eq!(compressed.len(), uncompressed.len());
    for ((stem, c), (other, u)) in compressed.iter().zip(&uncompressed) {
        assert_eq!(stem, other);
        assert_eq!(c.fingerprint(), u.fingerprint(), "{stem}");
    }
}

/// The runtime's choice of when the host sees a command buffer complete (`completion`) moves no
/// committed profile's fingerprint: no profile carries it, so no translation follows it.
#[test]
fn the_runtime_s_completion_choice_keys_no_profile() {
    use crate::spirv::testing::committed_profiles_with;
    let grouped = committed_profiles_with(DevicePolicy::UNTRUSTED).expect("profiles");
    let exact = committed_profiles_with(DevicePolicy {
        completion: CompletionPolicy::Exact,
        ..DevicePolicy::UNTRUSTED
    })
    .expect("profiles");
    assert_eq!(grouped.len(), exact.len());
    for ((stem, g), (other, e)) in grouped.iter().zip(&exact) {
        assert_eq!(stem, other);
        assert_eq!(g.fingerprint(), e.fingerprint(), "{stem}");
    }
}

/// The runtime's choice of a loaded attachment's tile store (`tile_store`) moves no committed
/// profile's fingerprint: no profile carries it, so no translation follows it.
#[test]
fn the_runtime_s_tile_store_choice_keys_no_profile() {
    use crate::spirv::testing::committed_profiles_with;
    let device = committed_profiles_with(DevicePolicy::UNTRUSTED).expect("profiles");
    let exact = committed_profiles_with(DevicePolicy {
        tile_store: TileStorePolicy::Exact,
        ..DevicePolicy::UNTRUSTED
    })
    .expect("profiles");
    assert_eq!(device.len(), exact.len());
    for ((stem, d), (other, e)) in device.iter().zip(&exact) {
        assert_eq!(stem, other);
        assert_eq!(d.fingerprint(), e.fingerprint(), "{stem}");
    }
}

/// The runtime's choice of a multisampled color attachment's resolve (`resolve`) moves no
/// committed profile's fingerprint: no profile carries it, so no translation follows it.
#[test]
fn the_runtime_s_resolve_choice_keys_no_profile() {
    use crate::spirv::testing::committed_profiles_with;
    let device = committed_profiles_with(DevicePolicy::UNTRUSTED).expect("profiles");
    let exact = committed_profiles_with(DevicePolicy {
        resolve: ResolvePolicy::Exact,
        ..DevicePolicy::UNTRUSTED
    })
    .expect("profiles");
    assert_eq!(device.len(), exact.len());
    for ((stem, d), (other, e)) in device.iter().zip(&exact) {
        assert_eq!(stem, other);
        assert_eq!(d.fingerprint(), e.fingerprint(), "{stem}");
    }
}

/// The runtime's choice of how a batch begins after the earlier ones of its queue
/// (`batch_order`) moves no committed profile's fingerprint: no profile carries it, so no
/// translation follows it.
#[test]
fn the_runtime_s_batch_order_choice_keys_no_profile() {
    use crate::spirv::testing::committed_profiles_with;
    let ends = committed_profiles_with(DevicePolicy::UNTRUSTED).expect("profiles");
    let exact = committed_profiles_with(DevicePolicy {
        batch_order: BatchOrderPolicy::Exact,
        ..DevicePolicy::UNTRUSTED
    })
    .expect("profiles");
    assert_eq!(ends.len(), exact.len());
    for ((stem, e), (other, x)) in ends.iter().zip(&exact) {
        assert_eq!(stem, other);
        assert_eq!(e.fingerprint(), x.fingerprint(), "{stem}");
    }
}

/// The runtime's choice of where a render encoder's calls are lowered (`render_lowering`) moves
/// no committed profile's fingerprint: no profile carries it, so no translation follows it.
#[test]
fn the_runtime_s_render_lowering_choice_keys_no_profile() {
    use crate::spirv::testing::committed_profiles_with;
    let at_call = committed_profiles_with(DevicePolicy::UNTRUSTED).expect("profiles");
    let worker = committed_profiles_with(DevicePolicy {
        render_lowering: RenderLoweringPolicy::Worker,
        ..DevicePolicy::UNTRUSTED
    })
    .expect("profiles");
    assert_eq!(worker.len(), at_call.len());
    for ((stem, w), (other, a)) in worker.iter().zip(&at_call) {
        assert_eq!(stem, other);
        assert_eq!(w.fingerprint(), a.fingerprint(), "{stem}");
    }
}

/// The runtime's choice of where a multisampled `Depth32Float` texture's texels are held
/// (`multisample_depth_twin`) moves no committed profile's fingerprint: no profile carries it, so
/// no translation follows it.
#[test]
fn the_runtime_s_multisample_depth_twin_choice_keys_no_profile() {
    use crate::spirv::testing::committed_profiles_with;
    let device = committed_profiles_with(DevicePolicy::UNTRUSTED).expect("profiles");
    let exact = committed_profiles_with(DevicePolicy {
        multisample_depth_twin: MultisampleDepthTwinPolicy::Exact,
        ..DevicePolicy::UNTRUSTED
    })
    .expect("profiles");
    assert_eq!(device.len(), exact.len());
    for ((stem, d), (other, e)) in device.iter().zip(&exact) {
        assert_eq!(stem, other);
        assert_eq!(d.fingerprint(), e.fingerprint(), "{stem}");
    }
}

/// The runtime's choice of a carried texture's decoded twin (`decoded_texels`) moves no committed
/// profile's fingerprint: no profile carries it, so no translation follows it.
#[test]
fn the_runtime_s_decoded_texels_choice_keys_no_profile() {
    use crate::spirv::testing::committed_profiles_with;
    let codes = committed_profiles_with(DevicePolicy::UNTRUSTED).expect("profiles");
    let exact = committed_profiles_with(DevicePolicy {
        decoded_texels: DecodedTexelsPolicy::Exact,
        ..DevicePolicy::UNTRUSTED
    })
    .expect("profiles");
    assert_eq!(codes.len(), exact.len());
    for ((stem, a), (other, e)) in codes.iter().zip(&exact) {
        assert_eq!(stem, other);
        assert_eq!(a.fingerprint(), e.fingerprint(), "{stem}");
    }
}
