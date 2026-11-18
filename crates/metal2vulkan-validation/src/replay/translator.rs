//! What a call describes in the translator's types, which every backend takes: a sampler
//! descriptor and function constant values.

use super::args::{metal, refuse, Fields};
use super::player::PlayError;
use super::stream::Call;
use super::value::Value;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};

/// `newSamplerStateWithDescriptor:`'s descriptor, `[map]`, as the translator's: every key but
/// `borderColor`, `supportArgumentBuffers` and `resourceIndex`, which the backend reads from the
/// fields it returns.
pub(super) fn sampler(call: &Call) -> Result<(SamplerDescriptor, Fields<'_>), PlayError> {
    let known = [
        "minFilter",
        "magFilter",
        "lodAverage",
        "compareFunction",
        "sAddressMode",
        "tAddressMode",
        "rAddressMode",
        "mipFilter",
        "maxAnisotropy",
        "normalizedCoordinates",
        "supportArgumentBuffers",
        "reductionMode",
        "lodMinClamp",
        "lodMaxClamp",
        "lodBias",
        "borderColor",
        "resourceIndex",
    ];
    let f = Fields::arg(call, 0, &known)?;
    let mut s = SamplerDescriptor::new();
    if let Some(filter) = f.metal("minFilter")? {
        s.min_filter = filter;
    }
    if let Some(filter) = f.metal("magFilter")? {
        s.mag_filter = filter;
    }
    if let Some(average) = f.bool("lodAverage")? {
        s.lod_average = average;
    }
    if let Some(compare) = f.metal("compareFunction")? {
        s.compare_function = compare;
    }
    if let Some(mode) = f.metal("sAddressMode")? {
        s.s_address_mode = mode;
    }
    if let Some(mode) = f.metal("tAddressMode")? {
        s.t_address_mode = mode;
    }
    if let Some(mode) = f.metal("rAddressMode")? {
        s.r_address_mode = mode;
    }
    if let Some(filter) = f.metal("mipFilter")? {
        s.mip_filter = filter;
    }
    if let Some(anisotropy) = f.count("maxAnisotropy")? {
        s.max_anisotropy = anisotropy;
    }
    if let Some(normalized) = f.bool("normalizedCoordinates")? {
        s.normalized_coordinates = normalized;
    }
    if let Some(mode) = f.metal("reductionMode")? {
        s.reduction_mode = mode;
    }
    let lods: [(&str, &mut f32); 3] = [
        ("lodMinClamp", &mut s.lod_min_clamp),
        ("lodMaxClamp", &mut s.lod_max_clamp),
        ("lodBias", &mut s.lod_bias),
    ];
    for (key, lod) in lods {
        if let Some(value) = f.float(key)? {
            *lod = value as f32;
        }
    }
    Ok((s, f))
}

/// The packed bytes of a function constant of `MTLDataType` `raw`: its lanes' size times their
/// count (a `float3` is 12 bytes); none for a type no function constant takes.
fn constant_bytes(raw: u64) -> Option<usize> {
    // Each family's scalar, then its two-, three- and four-lane vectors, numbered in a row.
    let (lane, scalar) = match raw {
        3..=6 => (4, 3),
        16..=19 => (2, 16),
        29..=32 => (4, 29),
        33..=36 => (4, 33),
        37..=40 => (2, 37),
        41..=44 => (2, 41),
        45..=48 => (1, 45),
        49..=52 => (1, 49),
        53..=56 => (1, 53),
        81..=84 => (8, 81),
        85..=88 => (8, 85),
        _ => return None,
    };
    Some(lane * usize::try_from(raw - scalar + 1).ok()?)
}

/// The function constants of `list`, each `{index, dataType, value}`, the value its bytes packed
/// from the first (the rest zero).
pub(super) fn constants(call: &Call, list: &[Value]) -> Result<FunctionConstantValues, PlayError> {
    let mut values = FunctionConstantValues::defaults();
    for constant in list {
        let f = Fields::of(call, constant, &["index", "dataType", "value"])?;
        let index = f.required("index", Fields::count)?;
        let raw = f.required("dataType", Fields::uint)?;
        let ty = metal(call, raw)?;
        let bytes = f.required("value", Fields::list)?;
        let bytes: Option<Vec<u8>> = bytes
            .iter()
            .map(|byte| byte.as_uint().and_then(|byte| u8::try_from(byte).ok()))
            .collect();
        let bytes = bytes.ok_or_else(|| refuse(call, format!("constant {index}'s value")))?;
        let size = constant_bytes(raw)
            .filter(|&size| size <= bytes.len())
            .ok_or_else(|| refuse(call, format!("constant {index} of data type {raw}")))?;
        let value = FunctionConstantValue::new(ty, &bytes[..size]);
        values = values.with_index(index, value);
    }
    Ok(values)
}
