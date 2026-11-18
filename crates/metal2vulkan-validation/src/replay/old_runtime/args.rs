//! A call's arguments as the old runtime's values (`docs/REPLAY-STREAM.md`, section 2): a value
//! not of the form its selector takes is refused, naming it.

pub(super) use super::super::args::{
    arg, data, index, list, metal, narrow, option_sets, range, refuse, uint, variant, Fields,
};
use super::super::player::PlayError;
use super::super::stream::Call;
use super::super::value::Value;
use metal2vulkan_runtime::{Error, Origin, Region, ResourceOptions, Size};

pub(super) fn failed<'c>(call: &'c Call) -> impl Fn(Error) -> PlayError + 'c {
    move |error| PlayError::Runtime {
        selector: call.selector.to_string(),
        error: error.to_string(),
    }
}

pub(super) fn options(call: &Call, raw: u64) -> Result<ResourceOptions, PlayError> {
    ResourceOptions::from_raw(raw).map_err(failed(call))
}

/// `MTLSize`: `{width, height, depth}`.
pub(super) fn size(call: &Call, value: &Value) -> Result<Size, PlayError> {
    let f = Fields::of(call, value, &["width", "height", "depth"])?;
    let uint = Fields::uint;
    Ok(Size::new(
        f.required("width", uint)?,
        f.required("height", uint)?,
        f.required("depth", uint)?,
    ))
}

/// `MTLOrigin`: `{x, y, z}`.
pub(super) fn origin(call: &Call, value: &Value) -> Result<Origin, PlayError> {
    let f = Fields::of(call, value, &["x", "y", "z"])?;
    let uint = Fields::uint;
    Ok(Origin::new(
        f.required("x", uint)?,
        f.required("y", uint)?,
        f.required("z", uint)?,
    ))
}

/// `MTLRegion`: `{origin, size}`.
pub(super) fn region(call: &Call, value: &Value) -> Result<Region, PlayError> {
    let f = Fields::of(call, value, &["origin", "size"])?;
    let part = |key: &str| f.get(key).ok_or_else(|| refuse(call, format!("no {key}")));
    Ok(Region {
        origin: origin(call, part("origin")?)?,
        size: size(call, part("size")?)?,
    })
}
