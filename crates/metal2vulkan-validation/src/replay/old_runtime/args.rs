//! A call's arguments as the old runtime's values (`docs/REPLAY-STREAM.md`, section 2): a value
//! not of the form its selector takes is refused, naming it.

use super::super::player::PlayError;
use super::super::stream::Call;
use super::super::value::{Name, Value};
use metal2vulkan_runtime::{Error, Origin, Region, ResourceOptions, Size};
use std::fmt::Display;
use std::ops::Range;

pub(super) fn refuse(call: &Call, why: impl Into<String>) -> PlayError {
    PlayError::Refused {
        selector: call.selector.to_string(),
        why: why.into(),
    }
}

pub(super) fn failed<'c>(call: &'c Call) -> impl Fn(Error) -> PlayError + 'c {
    move |error| PlayError::Runtime {
        selector: call.selector.to_string(),
        error: error.to_string(),
    }
}

pub(super) fn arg(call: &Call, at: usize) -> Result<&Value, PlayError> {
    call.args
        .get(at)
        .ok_or_else(|| refuse(call, format!("no argument {at}")))
}

pub(super) fn uint(call: &Call, at: usize) -> Result<u64, PlayError> {
    let value = arg(call, at)?;
    value
        .as_uint()
        .ok_or_else(|| refuse(call, format!("argument {at} is {value}, not a uint")))
}

/// `raw`, which the runtime takes as a `u32`.
pub(super) fn narrow(call: &Call, raw: u64, what: &str) -> Result<u32, PlayError> {
    u32::try_from(raw).map_err(|_| refuse(call, format!("{what} {raw}, past u32")))
}

pub(super) fn index(call: &Call, at: usize) -> Result<u32, PlayError> {
    narrow(call, uint(call, at)?, &format!("argument {at}"))
}

pub(super) fn data(call: &Call, at: usize) -> Result<&[u8], PlayError> {
    let value = arg(call, at)?;
    value
        .as_bytes()
        .ok_or_else(|| refuse(call, format!("argument {at} is {value}, not data")))
}

pub(super) fn list(call: &Call, at: usize) -> Result<&[Value], PlayError> {
    let value = arg(call, at)?;
    value
        .as_list()
        .ok_or_else(|| refuse(call, format!("argument {at} is {value}, not a list")))
}

/// A Metal enum or option set of raw value `raw`.
pub(super) fn metal<T: TryFrom<u32>>(call: &Call, raw: u64) -> Result<T, PlayError>
where
    T::Error: Display,
{
    let narrow = u32::try_from(raw).map_err(|_| refuse(call, format!("raw value {raw}")))?;
    T::try_from(narrow).map_err(|error| refuse(call, format!("raw value {raw}: {error}")))
}

pub(super) fn options(call: &Call, raw: u64) -> Result<ResourceOptions, PlayError> {
    ResourceOptions::from_raw(raw).map_err(failed(call))
}

/// A map's fields (a recorder struct's, keyed by its Rust field names in camelCase): each key one
/// `known` names, and nil where the struct leaves the property unset.
#[derive(Clone, Copy)]
pub(super) struct Fields<'c> {
    call: &'c Call,
    map: &'c [(Name, Value)],
}

impl<'c> Fields<'c> {
    pub(super) fn of(
        call: &'c Call,
        value: &'c Value,
        known: &[&str],
    ) -> Result<Fields<'c>, PlayError> {
        let map = value
            .as_map()
            .ok_or_else(|| refuse(call, format!("{value} is no map")))?;
        if let Some((key, _)) = map.iter().find(|(key, _)| !known.contains(&&**key)) {
            return Err(refuse(call, format!("the key {key}")));
        }
        Ok(Fields { call, map })
    }

    /// The fields of argument `at`.
    pub(super) fn arg(call: &'c Call, at: usize, known: &[&str]) -> Result<Fields<'c>, PlayError> {
        Fields::of(call, arg(call, at)?, known)
    }

    /// The value of `key`: none where it is nil or absent.
    pub(super) fn get(&self, key: &str) -> Option<&'c Value> {
        let value = self.map.iter().find(|(k, _)| &**k == key).map(|(_, v)| v);
        value.filter(|value| !matches!(value, Value::Nil))
    }

    fn typed<T>(
        &self,
        key: &str,
        what: &str,
        read: impl Fn(&'c Value) -> Option<T>,
    ) -> Result<Option<T>, PlayError> {
        match self.get(key) {
            None => Ok(None),
            Some(value) => read(value)
                .map(Some)
                .ok_or_else(|| refuse(self.call, format!("{key} is {value}, not {what}"))),
        }
    }

    pub(super) fn uint(&self, key: &str) -> Result<Option<u64>, PlayError> {
        self.typed(key, "a uint", Value::as_uint)
    }

    pub(super) fn bool(&self, key: &str) -> Result<Option<bool>, PlayError> {
        self.typed(key, "a bool", Value::as_bool)
    }

    pub(super) fn float(&self, key: &str) -> Result<Option<f64>, PlayError> {
        self.typed(key, "a number", Value::as_f64)
    }

    pub(super) fn object(&self, key: &str) -> Result<Option<u64>, PlayError> {
        self.typed(key, "an object", Value::as_object)
    }

    /// An object property a setter record sets: none where it is left as it was (nil), `Some(None)`
    /// where it is set to nil (object 0), else the object.
    pub(super) fn set_object(&self, key: &str) -> Result<Option<Option<u64>>, PlayError> {
        let set = self.object(key)?;
        Ok(set.map(|id| Some(id).filter(|&id| id != 0)))
    }

    pub(super) fn str(&self, key: &str) -> Result<Option<&'c str>, PlayError> {
        self.typed(key, "a string", Value::as_str)
    }

    pub(super) fn list(&self, key: &str) -> Result<Option<&'c [Value]>, PlayError> {
        self.typed(key, "a list", Value::as_list)
    }

    pub(super) fn fields(
        &self,
        key: &str,
        known: &[&str],
    ) -> Result<Option<Fields<'c>>, PlayError> {
        match self.get(key) {
            None => Ok(None),
            Some(value) => Fields::of(self.call, value, known).map(Some),
        }
    }

    /// `key`, read by `read`, which the struct always sets.
    pub(super) fn required<T>(
        &self,
        key: &str,
        read: impl Fn(&Self, &str) -> Result<Option<T>, PlayError>,
    ) -> Result<T, PlayError> {
        read(self, key)?.ok_or_else(|| refuse(self.call, format!("no {key}")))
    }

    /// A raw value of `key`, which the runtime takes as a `u32`.
    pub(super) fn count(&self, key: &str) -> Result<Option<u32>, PlayError> {
        let raw = self.uint(key)?;
        raw.map(|raw| narrow(self.call, raw, key)).transpose()
    }

    /// The Metal enum or option set of `key`'s raw value.
    pub(super) fn metal<T: TryFrom<u32>>(&self, key: &str) -> Result<Option<T>, PlayError>
    where
        T::Error: Display,
    {
        let raw = self.uint(key)?;
        raw.map(|raw| metal(self.call, raw)).transpose()
    }

    /// Refused where `key` is set to anything but `default`: a property the player does not
    /// carry.
    pub(super) fn only<T: PartialEq + Display>(
        &self,
        key: &str,
        read: impl Fn(&Self, &str) -> Result<Option<T>, PlayError>,
        default: T,
    ) -> Result<(), PlayError> {
        match read(self, key)? {
            Some(value) if value != default => Err(refuse(
                self.call,
                format!("{key} {value}, which is not played"),
            )),
            _ => Ok(()),
        }
    }
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

/// `NSRange`: `{location, length}`, as the bytes or indices it covers.
pub(super) fn range(call: &Call, value: &Value) -> Result<Range<u64>, PlayError> {
    let f = Fields::of(call, value, &["location", "length"])?;
    let start = f.required("location", Fields::uint)?;
    let length = f.required("length", Fields::uint)?;
    let end = start
        .checked_add(length)
        .ok_or_else(|| refuse(call, "a range past 2^64"))?;
    Ok(start..end)
}

/// A tuple variant's value, `{Name: value}`: its name and its value.
pub(super) fn variant<'c>(
    call: &Call,
    value: &'c Value,
) -> Result<(&'c str, &'c Value), PlayError> {
    match value.as_map() {
        Some([(name, value)]) => Ok((&**name, value)),
        _ => Err(refuse(call, format!("{value} is no {{Variant: value}}"))),
    }
}
