//! The source ranges Metal's blend into an extended-range attachment clamps its source, its second
//! source and the blend color to (`super::blend::native`): the format's range. Nothing here builds
//! SPIR-V.

/// The range Metal clamps a blend's source color to before blending into an attachment of a
/// linear form, `BGR10_XR_sRGB` or the two-plane `_sRGB` form: the linear values of codes 0 and
/// 1022, `-384 / 510` and `638 / 510`, rounded to `half`, nearest even (every one of the probe's
/// 8,192 sources a NaN or beyond the range lands there, for each format): `-0.7529296875` and
/// `1.2509765625`. The blend clamps the blend color's color channels to it too.
pub(crate) const SOURCE_RANGE: (f32, f32) = (-1542.0 / 2048.0, 1281.0 / 1024.0);

/// [`SOURCE_RANGE`] for `BGRA10_XR_sRGB`: the tile values of codes 0 and 1022
/// (`super::tile_value`, the table's steps 384 and 638 over 8192 rounded to `half`, ties away from
/// zero): `-0.52734375` and `1.6689453125`. A blend into it clamps the blend color's color channels
/// to it too.
pub(crate) const SRGB_SOURCE_RANGE: (f32, f32) = (-1080.0 / 2048.0, 1709.0 / 1024.0);
