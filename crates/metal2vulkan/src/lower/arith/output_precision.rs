//! The notes of a fragment function's color outputs that Metal may store otherwise than the
//! function computes them. Nothing here changes what the function computes: each output is its own
//! arithmetic, rounded as the AIR says.
//!
//! **A `half` color output** ([`note_half_output`]). Metal's compiler may carry an output's last
//! operation at `float` precision into the attachment, the blend and the format's conversion, where
//! the function rounds it to `half` first (an M4 under macOS 27.0.1 and the M3 Max under macOS
//! 26.7: under fast math a product, a sum or difference, or a fused multiply-add of `half`s with no
//! other use, into a 32-bit float attachment). The note follows the module's math mode: a module
//! compiled with fast math records `PrecisionNote::HalfOutputFastMathBlend`, where Metal's
//! compiler, folding the blend into the function, reads the output's `float` or its `half` and fuses
//! or regroups the blend per pipeline, so no fixed form is Metal's; any other module (without fast
//! math, or naming no mode) records `PrecisionNote::HalfOutputPrecision`.
//!
//! A module without fast math (`air.compile.fast_math_disable`) records no note itself: Metal
//! stores the `half` it returns, so only a blend can store another value. It lists the attachment
//! (`StageIo::half_blend_colors`), and the pipeline's stage records
//! `PrecisionNote::HalfOutputPrecision` where that attachment blends
//! (`color_writes::blends_output`). A module with fast math lists an attachment it reaches as a
//! color attachment too (`StageIo::half_fast_colors`), and the pipeline's stage records
//! `PrecisionNote::HalfOutputFastMathBlend` where that attachment stores some channel
//! (`color_writes::stores_output`): one whose write mask stores none holds nothing the output could
//! differ by. A module naming no math mode keeps the note at any attachment: what Metal's compiler
//! does there was not measured.
//!
//! **A blended extended-range output** ([`note_extended_range_sources`]): the function blends it by
//! the device's arithmetic, where Metal's compiler folds the blend into the function.

use crate::air::ir::Type;
use crate::error::Error;
use crate::lower::select::SelectCx;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::sym::Stage;

/// The note a `half` color output records where Metal may hand it to the attachment otherwise
/// than the function stores it (the module docs): `PrecisionNote::HalfOutputFastMathBlend` in a
/// module compiled with fast math (`air.compile.fast_math_enable`), whose blend Metal's compiler
/// fuses and regroups per pipeline; `PrecisionNote::HalfOutputPrecision` in any other, a module
/// that names no math mode too (no module is taken into the fast-math class unless it states
/// fast math).
pub(crate) fn half_output_note(cx: &SelectCx<'_>) -> PrecisionNote {
    match cx.entry.compile.fast_math {
        Some(true) => PrecisionNote::HalfOutputFastMathBlend,
        Some(false) | None => PrecisionNote::HalfOutputPrecision,
    }
}

/// Record that a `half` color output reaches attachment `color` where Metal may store it otherwise
/// than the function: in a module compiled without fast math (`air.compile.fast_math_disable`)
/// the attachment itself (`StageIo::half_blend_colors`), whose note the pipeline decides by
/// whether the attachment blends, since Metal stores the `half` the function returns wherever it
/// does not (the sweep: the `half4` output and a `float4` of its `half` store alike
/// under every blend and format); in a module compiled with fast math the attachment too
/// (`StageIo::half_fast_colors`), whose note the pipeline decides by whether the attachment stores
/// some channel, since one that stores none (its write mask none, or absent from the pipeline)
/// holds nothing the output could differ by, but [`half_output_note`] itself for an attachment
/// the function writes in the storage form, which the pipeline's color attachments do not show;
/// in a module naming no math mode [`half_output_note`] itself. Nothing where the function
/// computes Metal's 16-bit float blend there itself (a fetched attachment's blend), which stores
/// Metal's value, nor where it writes the attachment in the
/// storage form under a write mask that names none of the format's channels, which stores nothing.
pub(crate) fn note_half_output(cx: &mut SelectCx<'_>, color: u32) {
    // Metal's 16-bit float blend, computed by the function: the attachment stores Metal's value.
    // A storage write of no channel stores nothing.
    if crate::lower::fragment::blends_half_in_storage(cx, color)
        || crate::lower::fragment::stores_nothing_in_storage(cx, color)
    {
        return;
    }
    let math = cx.entry.compile.fast_math;
    let in_storage = crate::lower::fragment::written_in_storage(cx, color);
    if math.is_none() || (math == Some(true) && in_storage) {
        let note = half_output_note(cx);
        cx.precision.insert(note);
        return;
    }
    let io = &mut cx.families.stage_io.interface;
    let colors = match math {
        Some(false) => &mut io.half_blend_colors,
        _ => &mut io.half_fast_colors,
    };
    if let Err(at) = colors.binary_search(&color) {
        colors.insert(at, color);
    }
}

/// Record `PrecisionNote::ExtendedRangeBlendSource` (a `half` output) or
/// `PrecisionNote::ExtendedRangeFloatBlendSource` (a `float` one) for each color output the
/// function blends into an extended-range attachment, where the write stores some channel: the
/// function computes that blend by the device's arithmetic (`extended_range::blend::native`),
/// where Metal's compiler folds the blend into the function in forms of its own (each sum and
/// product rounded to `half`, lerps, common factors and fused pairs chosen by the factors and the
/// function's own arithmetic), so a blended channel may store a value apart from Metal's.
pub fn note_extended_range_sources(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    use crate::air::abi::{AbiInt, OutputRole};
    if cx.stage != Stage::Fragment {
        return Ok(());
    }
    let (_, tys) = crate::lower::stage_io::output_types(cx)?;
    let types = cx.entry.module.types();
    let mut notes = Vec::new();
    for (o, ty) in cx.entry.ep.outputs.iter().zip(tys) {
        let OutputRole::RenderTarget {
            attachment: AbiInt::Const(color),
            index: 0,
            ..
        } = o.role
        else {
            continue;
        };
        let blended = cx
            .inputs
            .extended_range_colors
            .get(&color)
            .is_some_and(|e| e.blended);
        // A write of none of the attachment's channels blends nothing.
        let stores = crate::lower::fragment::shader_blend(cx, color)
            .is_none_or(|(eq, _, nothing)| eq.is_some() || !nothing);
        if !blended || !stores {
            continue;
        }
        let elem = match types.get(ty) {
            Some(Type::Vector { elem, .. }) => types.get(*elem),
            other => other,
        };
        notes.push(match elem {
            Some(Type::Float) => PrecisionNote::ExtendedRangeFloatBlendSource,
            _ => PrecisionNote::ExtendedRangeBlendSource,
        });
    }
    cx.precision.extend(notes);
    Ok(())
}
