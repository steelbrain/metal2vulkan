//! Plain-data Vulkan values: what artifacts, plans, and device descriptions carry.
//!
//! Every Vulkan enum, flag, and single-bit type here is a transparent newtype over the raw
//! registry value, generated from a pinned `vk.xml`, so the API depends on no Vulkan binding:
//! hand `as_raw` values to yours, or enable feature `ash` for `From` conversions to and from
//! `ash::vk` (module `ash_interop`). Single-bit types ([`ShaderStage`]) are closed: their values
//! are exactly the named single bits of one bitmask.
//!
//! Raw values that this registry revision does not name are representable on purpose (a newer
//! driver reports values the pinned registry lacks); [`Format::is_known`] and its siblings say
//! whether the registry names a value. Values the translator emits are always named constants.
//! [`Format::block`] is the registry's texel block of each format, with each aspect's numeric
//! format, and [`Format::numeric_type`] reads it; [`Format::exposure`] says which core version or
//! extensions define each format value.
//!
//! The hand-written structs mirror Vulkan's: [`SpecializationMapEntry`], [`ComponentMapping`],
//! and [`SamplerDesc`] (the translation-relevant part of `VkSamplerCreateInfo`), with
//! [`FormatBlock`] and [`NumericFormat`] for the registry's format facts.

// `generated.rs` is written by `scripts/regen-vk-types/` (never hand-edited); the hand-written
// structs live in `structs.rs`.

// The generated file invokes the macros below, so it is declared after them (textual scoping).
mod structs;

pub use generated::*;
pub use structs::{
    ComponentMapping, FormatBlock, FormatExposure, NumericFormat, NumericType, SamplerDesc,
    SpecializationMapEntry,
};

// The Vulkan API version vocabulary type, shared with `device` (`major.minor`; the floor is 1.2).
pub use crate::device::requirements::ApiVersion;

impl ApiVersion {
    /// Decodes a packed `VK_MAKE_API_VERSION` value (as in `VkPhysicalDeviceProperties::apiVersion`),
    /// dropping the variant and patch fields, which translation never depends on.
    pub const fn from_packed(packed: u32) -> ApiVersion {
        ApiVersion {
            major: (packed >> 22) & 0x7f,
            minor: (packed >> 12) & 0x3ff,
        }
    }

    /// Encodes this version as `VK_MAKE_API_VERSION(0, major, minor, 0)`.
    pub const fn packed(self) -> u32 {
        ((self.major & 0x7f) << 22) | ((self.minor & 0x3ff) << 12)
    }
}

/// Defines one registry `enum` type as a transparent raw-value newtype.
macro_rules! vk_enum {
    (
        $(#[$meta:meta])*
        $name:ident($raw:ty) = $vk:literal {
            $($konst:ident = $value:literal, $vkname:literal;)*
        }
        aliases {
            $($alias:literal => $target:ident;)*
        }
    ) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(feature = "serde", serde(transparent))]
        pub struct $name($raw);

        impl $name {
            $(
                #[doc = concat!("`", $vkname, "`.")]
                pub const $konst: $name = $name($value);
            )*

            /// The registry type name.
            pub const VK_NAME: &'static str = $vk;

            /// Every value the pinned registry names, in registry order (aliases excluded).
            pub const KNOWN: &'static [$name] = &[$($name::$konst),*];

            const NAMES: &'static [($name, &'static str)] = &[$(($name::$konst, $vkname)),*];

            const ALIASES: &'static [(&'static str, $name)] = &[$(($alias, $name::$target)),*];

            /// The value with this raw registry number, named or not.
            pub const fn from_raw(raw: $raw) -> $name {
                $name(raw)
            }

            /// The raw registry number.
            pub const fn as_raw(self) -> $raw {
                self.0
            }

            /// The registry name of this value, if the pinned registry names it.
            pub fn name(self) -> Option<&'static str> {
                Self::NAMES.iter().find(|(v, _)| *v == self).map(|(_, n)| *n)
            }

            /// Whether the pinned registry names this value.
            pub fn is_known(self) -> bool {
                self.name().is_some()
            }

            /// The value with this registry name (canonical or alias).
            pub fn from_name(name: &str) -> Option<$name> {
                Self::NAMES
                    .iter()
                    .find(|(_, n)| *n == name)
                    .map(|(v, _)| *v)
                    .or_else(|| Self::ALIASES.iter().find(|(n, _)| *n == name).map(|(_, v)| *v))
            }
        }

        impl ::std::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                match self.name() {
                    Some(name) => f.write_str(name),
                    None => write!(f, "{}({})", $vk, self.0),
                }
            }
        }
    };
}

/// Defines one registry bitmask type as a transparent raw-value newtype.
macro_rules! vk_flags {
    (
        $(#[$meta:meta])*
        $name:ident($raw:ty) = $vk:literal {
            $($konst:ident = $value:literal, $vkname:literal;)*
        }
        aliases {
            $($alias:literal => $target:ident;)*
        }
    ) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(feature = "serde", serde(transparent))]
        pub struct $name($raw);

        impl $name {
            $(
                #[doc = concat!("`", $vkname, "`.")]
                pub const $konst: $name = $name($value);
            )*

            /// The registry type name.
            pub const VK_NAME: &'static str = $vk;

            /// Every named value (single bits and named combinations), in registry order.
            pub const KNOWN: &'static [($name, &'static str)] = &[$(($name::$konst, $vkname)),*];

            const ALIASES: &'static [(&'static str, $name)] = &[$(($alias, $name::$target)),*];

            /// No bits.
            pub const fn empty() -> $name {
                $name(0)
            }

            /// The flags with these raw bits, named or not.
            pub const fn from_raw(raw: $raw) -> $name {
                $name(raw)
            }

            /// The raw bits.
            pub const fn as_raw(self) -> $raw {
                self.0
            }

            /// Whether no bit is set.
            pub const fn is_empty(self) -> bool {
                self.0 == 0
            }

            /// Whether every bit of `other` is set in `self`.
            pub const fn contains(self, other: $name) -> bool {
                self.0 & other.0 == other.0
            }

            /// Whether `self` and `other` share a bit.
            pub const fn intersects(self, other: $name) -> bool {
                self.0 & other.0 != 0
            }

            /// The bits of either.
            pub const fn union(self, other: $name) -> $name {
                $name(self.0 | other.0)
            }

            /// The bits of both.
            pub const fn intersection(self, other: $name) -> $name {
                $name(self.0 & other.0)
            }

            /// The bits of `self` that are not in `other`.
            pub const fn difference(self, other: $name) -> $name {
                $name(self.0 & !other.0)
            }

            /// The union of every single bit the pinned registry names.
            pub const fn known_bits() -> $name {
                let mut bits: $raw = 0;
                let mut i = 0;
                while i < Self::KNOWN.len() {
                    let v = Self::KNOWN[i].0 .0;
                    if v.count_ones() == 1 {
                        bits |= v;
                    }
                    i += 1;
                }
                $name(bits)
            }

            /// The registry name of this exact value, if it is a named bit or combination.
            pub fn name(self) -> Option<&'static str> {
                Self::KNOWN.iter().find(|(v, _)| *v == self).map(|(_, n)| *n)
            }

            /// The value with this registry name (canonical or alias).
            pub fn from_name(name: &str) -> Option<$name> {
                Self::KNOWN
                    .iter()
                    .find(|(_, n)| *n == name)
                    .map(|(v, _)| *v)
                    .or_else(|| Self::ALIASES.iter().find(|(n, _)| *n == name).map(|(_, v)| *v))
            }
        }

        impl ::std::ops::BitOr for $name {
            type Output = $name;
            fn bitor(self, rhs: $name) -> $name {
                self.union(rhs)
            }
        }

        impl ::std::ops::BitOrAssign for $name {
            fn bitor_assign(&mut self, rhs: $name) {
                *self = self.union(rhs);
            }
        }

        impl ::std::ops::BitAnd for $name {
            type Output = $name;
            fn bitand(self, rhs: $name) -> $name {
                self.intersection(rhs)
            }
        }

        impl ::std::ops::BitAndAssign for $name {
            fn bitand_assign(&mut self, rhs: $name) {
                *self = self.intersection(rhs);
            }
        }

        impl ::std::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                if let Some(name) = self.name() {
                    return f.write_str(name);
                }
                let mut rest = self.0;
                let mut first = true;
                for (v, n) in Self::KNOWN {
                    if v.0.count_ones() == 1 && rest & v.0 != 0 {
                        if !first {
                            f.write_str(" | ")?;
                        }
                        f.write_str(n)?;
                        rest &= !v.0;
                        first = false;
                    }
                }
                if rest != 0 || first {
                    if !first {
                        f.write_str(" | ")?;
                    }
                    write!(f, "{:#x}", rest)?;
                }
                Ok(())
            }
        }
    };
}

/// Defines one closed single-bit type: exactly one named bit of a registry bitmask.
///
/// Unlike [`vk_flags!`] types, a value is always one bit the pinned registry names; construction
/// from raw bits is checked (`from_raw`, `TryFrom`), and named combinations (`..._ALL`) are not
/// values of the type.
macro_rules! vk_bit {
    (
        $(#[$meta:meta])*
        $name:ident($raw:ty) = $vk:literal in $flags:ident {
            $($konst:ident = $value:literal, $vkname:literal;)*
        }
        aliases {
            $($alias:literal => $target:ident;)*
        }
    ) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(feature = "serde", serde(try_from = "u32", into = "u32"))]
        pub struct $name($raw);

        impl $name {
            $(
                #[doc = concat!("`", $vkname, "`.")]
                pub const $konst: $name = $name($value);
            )*

            /// The registry type name of the bits.
            pub const VK_NAME: &'static str = $vk;

            /// Every single bit the pinned registry names, in registry order (aliases excluded).
            pub const KNOWN: &'static [$name] = &[$($name::$konst),*];

            const NAMES: &'static [($name, &'static str)] = &[$(($name::$konst, $vkname)),*];

            const ALIASES: &'static [(&'static str, $name)] = &[$(($alias, $name::$target)),*];

            /// The bit with this raw value, if it is exactly one named bit.
            pub const fn from_raw(raw: $raw) -> Option<$name> {
                let mut i = 0;
                while i < Self::KNOWN.len() {
                    if Self::KNOWN[i].0 == raw {
                        return Some(Self::KNOWN[i]);
                    }
                    i += 1;
                }
                None
            }

            /// The bit, if `flags` is exactly one named bit.
            pub const fn from_flags(flags: $flags) -> Option<$name> {
                Self::from_raw(flags.as_raw())
            }

            /// The raw bit.
            pub const fn as_raw(self) -> $raw {
                self.0
            }

            /// This bit as a flag set.
            pub const fn flags(self) -> $flags {
                $flags::from_raw(self.0)
            }

            /// The registry name of this bit.
            pub fn name(self) -> &'static str {
                Self::NAMES
                    .iter()
                    .find(|(v, _)| *v == self)
                    .map(|(_, n)| *n)
                    .expect("a single-bit value is always a named bit")
            }

            /// The bit with this registry name (canonical or alias).
            pub fn from_name(name: &str) -> Option<$name> {
                Self::NAMES
                    .iter()
                    .find(|(_, n)| *n == name)
                    .map(|(v, _)| *v)
                    .or_else(|| Self::ALIASES.iter().find(|(n, _)| *n == name).map(|(_, v)| *v))
            }
        }

        impl ::std::convert::TryFrom<$raw> for $name {
            /// The rejected raw value.
            type Error = $raw;
            fn try_from(raw: $raw) -> Result<$name, $raw> {
                $name::from_raw(raw).ok_or(raw)
            }
        }

        impl From<$name> for $raw {
            fn from(bit: $name) -> $raw {
                bit.0
            }
        }

        impl From<$name> for $flags {
            fn from(bit: $name) -> $flags {
                bit.flags()
            }
        }

        impl ::std::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(self.name())
            }
        }
    };
}

/// Defines [`Format::block`] and [`Format::exposure`] from the registry's `<formats>`: per
/// format, `blockSize`, `texelsPerBlock`, its compatibility `class`, the `packed` width when
/// present, the bit width every component shares when there is one, its color components, the
/// `numericFormat` of each aspect it has, and the core version and extensions that define the enum
/// value.
macro_rules! vk_format_blocks {
    (@opt) => { None };
    (@opt $v:literal) => { Some($v) };
    (@components) => { $crate::vk::ColorComponentFlags::empty() };
    (@components $($c:ident)*) => {
        $crate::vk::ColorComponentFlags::empty()$(.union($crate::vk::ColorComponentFlags::$c))*
    };
    (@numeric) => { None };
    (@numeric $v:ident) => { Some($crate::vk::NumericFormat::$v) };
    (@core) => { None };
    (@core $major:literal $minor:literal) => {
        Some($crate::vk::ApiVersion { major: $major, minor: $minor })
    };
    (
        $($konst:ident = $size:literal, $texels:literal, class $class:literal
            $(, packed $packed:literal)? $(, bits $bits:literal)?
            $(, components [$($component:ident)*])?
            $(, color $color:ident)? $(, depth $depth:ident)? $(, stencil $stencil:ident)?
            $(, core $major:literal $minor:literal)? $(, ext $ext:literal)*;)*
    ) => {
        impl Format {
            /// The texel block of this format as the registry describes it (the Vulkan spec's
            /// "Compatible Formats" and "Packed Formats" tables), or `None` for `UNDEFINED` and
            /// values the pinned registry does not name.
            pub const fn block(self) -> Option<$crate::vk::FormatBlock> {
                match self {
                    $(Format::$konst => Some($crate::vk::FormatBlock {
                        size: $size,
                        texels: $texels,
                        class: $class,
                        packed_bits: vk_format_blocks!(@opt $($packed)?),
                        component_bits: vk_format_blocks!(@opt $($bits)?),
                        color_components: vk_format_blocks!(@components $($($component)*)?),
                        color: vk_format_blocks!(@numeric $($color)?),
                        depth: vk_format_blocks!(@numeric $($depth)?),
                        stencil: vk_format_blocks!(@numeric $($stencil)?),
                    }),)*
                    _ => None,
                }
            }

            /// Where the pinned registry defines this format's enum value: the lowest core
            /// version, and every extension that defines it or an alias of it. `None` for
            /// `UNDEFINED` and values the pinned registry does not name. A device may use the
            /// format (query it, create images or views of it) only where it exposes the value:
            /// at that core version, or with one of those extensions.
            pub const fn exposure(self) -> Option<$crate::vk::FormatExposure> {
                match self {
                    $(Format::$konst => Some($crate::vk::FormatExposure {
                        core: vk_format_blocks!(@core $($major $minor)?),
                        extensions: &[$($ext),*],
                    }),)*
                    _ => None,
                }
            }
        }
    };
}

mod generated;

#[cfg(test)]
mod tests;
