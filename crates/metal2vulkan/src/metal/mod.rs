//! Metal's enums and option sets, and their mapping to Vulkan.
//!
//! The enums and option sets mirror the Metal API's numeric values, which Metal-shaped descriptors
//! and guest command streams carry as raw `u32` ordinals; they are generated from the macOS SDK
//! headers (names and values only). Each enum is a closed Rust enum whose `TryFrom<u32>` rejects
//! an ordinal outside the known set with
//! [`Error::InvalidDescriptor`]`(`[`DescriptorError::UnknownEnumValue`]`)`: an unknown value is
//! never guessed. Each option set is a bit-set newtype whose `TryFrom<u32>` rejects unknown bits
//! the same way, and keeps the bits Metal's descriptor takes that its object's answers drop
//! (`DROPPED`: `MTLTextureUsage`'s 0x8).
//!
//! The submodules map Metal state to the Vulkan values a consumer creates its objects with, for a
//! device's profile: [`format`](mod@format) (pixel formats, storage-image choices, texture view
//! types), [`vertex`] (vertex formats and step functions), [`sampler`] (sampler descriptors), and
//! [`state`] (blend, depth/stencil, topology, and index types). The hidden `bicubic` module states
//! how Metal's bicubic filter treats a bound format's texels, and the hidden `texel_format` module
//! how Metal converts a bound format's texels to a `half` result. Pipelines and binding plans use
//! the same mappings.

// `generated.rs` is produced from the macOS SDK headers by `scripts/regen-metal-enums/` (names
// and values only; never hand-edited).

use crate::error::{DescriptorError, Error};

/// The error for a raw ordinal (or bit set) outside the known values of Metal type `ty`.
pub(crate) fn unknown(ty: &'static str, value: u32) -> Error {
    Error::InvalidDescriptor(DescriptorError::UnknownEnumValue { ty, value })
}

/// Defines one Metal `NS_ENUM` as a closed Rust enum with a checked `TryFrom<u32>`. The generated
/// invocation states `#[non_exhaustive]` itself, except for the enums a consumer maps completely.
macro_rules! metal_enum {
    (
        $(#[$meta:meta])*
        $name:ident = $metal:literal {
            $($(#[$vmeta:meta])* $variant:ident = $value:literal, $mname:literal;)*
        }
        aliases {
            $($alias:literal => $target:ident;)*
        }
    ) => {
        $(#[$meta])*
        #[repr(u32)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        pub enum $name {
            $(
                #[doc = concat!("`", $mname, "`.")]
                $(#[$vmeta])*
                $variant = $value,
            )*
        }

        impl $name {
            /// The Metal type name.
            pub const METAL_NAME: &'static str = $metal;

            /// Every value, in declaration order.
            pub const ALL: &'static [$name] = &[$($name::$variant),*];

            const ALIASES: &'static [(&'static str, $name)] = &[$(($alias, $name::$target)),*];

            /// The raw Metal ordinal.
            pub const fn as_raw(self) -> u32 {
                self as u32
            }

            /// The Metal enumerator name.
            pub const fn name(self) -> &'static str {
                match self {
                    $($name::$variant => $mname,)*
                }
            }

            /// The value with this Metal enumerator name (current or deprecated spelling).
            pub fn from_name(name: &str) -> Option<$name> {
                Self::ALL
                    .iter()
                    .copied()
                    .find(|v| v.name() == name)
                    .or_else(|| Self::ALIASES.iter().find(|(n, _)| *n == name).map(|(_, v)| *v))
            }
        }

        impl From<$name> for u32 {
            fn from(value: $name) -> u32 {
                value.as_raw()
            }
        }

        impl TryFrom<u32> for $name {
            type Error = $crate::error::Error;

            fn try_from(value: u32) -> Result<$name, $crate::error::Error> {
                match value {
                    $($value => Ok($name::$variant),)*
                    _ => Err($crate::metal::unknown($metal, value)),
                }
            }
        }
    };
}

/// Defines one Metal `NS_OPTIONS` type as a bit-set newtype with a checked `TryFrom<u32>`.
macro_rules! metal_flags {
    (
        $(#[$meta:meta])*
        $name:ident = $metal:literal {
            $($konst:ident = $value:literal, $mname:literal;)*
        }
        aliases {
            $($alias:literal => $target:ident;)*
        }
        dropped {
            $($dropped:literal;)*
        }
    ) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(feature = "serde", serde(try_from = "u32", into = "u32"))]
        pub struct $name(u32);

        impl $name {
            $(
                #[doc = concat!("`", $mname, "`.")]
                pub const $konst: $name = $name($value);
            )*

            /// The Metal type name.
            pub const METAL_NAME: &'static str = $metal;

            /// Every named value (single bits and named combinations), in declaration order.
            pub const KNOWN: &'static [($name, &'static str)] = &[$(($name::$konst, $mname)),*];

            const ALIASES: &'static [(&'static str, $name)] = &[$(($alias, $name::$target)),*];

            /// The union of every named value; `TryFrom<u32>` accepts exactly its subsets, beside
            /// [`Self::DROPPED`].
            pub const ALL_KNOWN: $name = $name(0 $(| $value)*);

            /// Bits no name declares that Metal's descriptor takes and its object's answers drop
            /// (`MTLTextureUsage`'s 0x8: a texture made with it answers its usage without it, while
            /// the heap size query of a descriptor of it alone answers as for a usage, not as for
            /// `Unknown`). `TryFrom<u32>` keeps them, so a descriptor holds them as Metal's does;
            /// [`Self::without_dropped`] is the value an object made of it answers.
            pub const DROPPED: u32 = 0 $(| $dropped)*;

            /// No bits.
            pub const fn empty() -> $name {
                $name(0)
            }

            /// The raw Metal bits.
            pub const fn as_raw(self) -> u32 {
                self.0
            }

            /// Whether no bit is set.
            pub const fn is_empty(self) -> bool {
                self.0 == 0
            }

            /// The value without [`Self::DROPPED`]'s bits: what an object Metal makes of a
            /// descriptor of `self` answers.
            pub const fn without_dropped(self) -> $name {
                $name(self.0 & !Self::DROPPED)
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

            /// The value with this Metal enumerator name (current or deprecated spelling).
            pub fn from_name(name: &str) -> Option<$name> {
                Self::KNOWN
                    .iter()
                    .find(|(_, n)| *n == name)
                    .map(|(v, _)| *v)
                    .or_else(|| Self::ALIASES.iter().find(|(n, _)| *n == name).map(|(_, v)| *v))
            }
        }

        impl From<$name> for u32 {
            fn from(value: $name) -> u32 {
                value.as_raw()
            }
        }

        impl TryFrom<u32> for $name {
            type Error = $crate::error::Error;

            fn try_from(value: u32) -> Result<$name, $crate::error::Error> {
                if value & !($name::ALL_KNOWN.0 | $name::DROPPED) == 0 {
                    Ok($name(value))
                } else {
                    Err($crate::metal::unknown($metal, value))
                }
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
                if let Some((_, name)) = Self::KNOWN.iter().find(|(v, _)| v == self) {
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

// The generated file invokes the macros above, so it is declared after them (textual scoping).
mod generated;

pub use generated::*;
// `MTLDataType`, the type of a function constant, is the AIR front end's: the ABI and a
// specialization request name it.
pub use crate::air::metal::DataType;

/// `MTLTextureSwizzleChannels`: the source of each channel a read or sample of a texture returns,
/// border texels included. A value whose four fields are its identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TextureSwizzleChannels {
    pub red: TextureSwizzle,
    pub green: TextureSwizzle,
    pub blue: TextureSwizzle,
    pub alpha: TextureSwizzle,
}

impl TextureSwizzleChannels {
    /// `MTLTextureSwizzleChannelsDefault`: each channel from itself.
    pub const DEFAULT: TextureSwizzleChannels = TextureSwizzleChannels {
        red: TextureSwizzle::Red,
        green: TextureSwizzle::Green,
        blue: TextureSwizzle::Blue,
        alpha: TextureSwizzle::Alpha,
    };

    /// `MTLTextureSwizzleChannelsMake(r, g, b, a)`.
    pub fn new(
        red: TextureSwizzle,
        green: TextureSwizzle,
        blue: TextureSwizzle,
        alpha: TextureSwizzle,
    ) -> TextureSwizzleChannels {
        TextureSwizzleChannels {
            red,
            green,
            blue,
            alpha,
        }
    }

    /// Whether every channel comes from itself.
    pub fn is_default(self) -> bool {
        self == TextureSwizzleChannels::DEFAULT
    }

    /// [`Self::word`] of [`Self::DEFAULT`].
    pub(crate) const DEFAULT_WORD: u32 = TextureSwizzleChannels::DEFAULT.word();

    /// The swizzle as one word, a texture buffer slot's `SpecKey::TextureSwizzle`: each channel's
    /// `MTLTextureSwizzle` raw value in three bits, red's lowest.
    pub(crate) const fn word(self) -> u32 {
        self.red.as_raw()
            | self.green.as_raw() << 3
            | self.blue.as_raw() << 6
            | self.alpha.as_raw() << 9
    }

    /// The component mapping that gives this swizzle of the channels `format` presents (a view
    /// format's mapping, `FormatMapping::components`): each channel a swizzle names is the one
    /// `format` gives there, and `Zero` and `One` are constants. A channel from itself is
    /// `IDENTITY`, so the composition of the default swizzle is `format` in that canonical form.
    /// Metal applies a swizzle to the channels the texture's format gives, border texels included
    /// (measured on an Apple M3 Max and M4).
    pub fn compose(self, format: crate::vk::ComponentMapping) -> crate::vk::ComponentMapping {
        use crate::vk::ComponentSwizzle as S;
        let own = [S::R, S::G, S::B, S::A];
        let given = [format.r, format.g, format.b, format.a];
        let source = |k: usize| {
            if given[k] == S::IDENTITY {
                own[k]
            } else {
                given[k]
            }
        };
        let channel = |c: usize, s: TextureSwizzle| {
            let out = match s {
                TextureSwizzle::Zero => S::ZERO,
                TextureSwizzle::One => S::ONE,
                TextureSwizzle::Red => source(0),
                TextureSwizzle::Green => source(1),
                TextureSwizzle::Blue => source(2),
                TextureSwizzle::Alpha => source(3),
            };
            if out == own[c] {
                S::IDENTITY
            } else {
                out
            }
        };
        crate::vk::ComponentMapping {
            r: channel(0, self.red),
            g: channel(1, self.green),
            b: channel(2, self.blue),
            a: channel(3, self.alpha),
        }
    }

    /// This swizzle of the texel `(0, 0, 0, a)` an `A8Unorm` texture buffer reads, as a swizzle
    /// of the texel `(a, 0, 0, 1)` its bytes' `R8_UNORM` view reads
    /// (`metal::format::texel_buffer_alpha_in_red`): each channel taken from red, green or blue is
    /// zero, one taken from alpha the view's red, and `Zero` and `One` stay.
    pub(crate) fn alpha_in_red(self) -> TextureSwizzleChannels {
        let held = |s: TextureSwizzle| match s {
            TextureSwizzle::Red | TextureSwizzle::Green | TextureSwizzle::Blue => {
                TextureSwizzle::Zero
            }
            TextureSwizzle::Alpha => TextureSwizzle::Red,
            constant => constant,
        };
        TextureSwizzleChannels {
            red: held(self.red),
            green: held(self.green),
            blue: held(self.blue),
            alpha: held(self.alpha),
        }
    }
}

impl Default for TextureSwizzleChannels {
    fn default() -> TextureSwizzleChannels {
        TextureSwizzleChannels::DEFAULT
    }
}

/// `textureWriteRoundingMode`, Apple's SPI on the pipeline descriptors: how a render pipeline
/// state's `float` writes to half-float channels round (on an M4 under macOS 27.0.1,
/// and the consumer's measurement on an M3 Max under macOS 26.7). Metal's raw 0 (the getter's
/// default) and 1 write as a state made without it: toward zero; 2 rounds a render state's
/// half-float color attachments, blended or not, and the half-float textures its fragment function
/// writes, to nearest even, from 65520 up to infinity. A kernel's, a tile function's and a vertex
/// function's writes, and depth, are the same at every mode. `supportsTextureWriteRoundingMode:`
/// answers yes for 0 to 2, and a pipeline descriptor of 3 or more stops at Metal's descriptor
/// validation ([`TextureWriteRoundingMode::from_raw`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextureWriteRoundingMode {
    /// Metal's raw 0 and 1, and a descriptor's default: a half-float channel rounds toward zero.
    #[default]
    TowardZero,
    /// Metal's raw 2: a render state's half-float writes round to nearest even.
    NearestEven,
}

impl TextureWriteRoundingMode {
    /// The mode of Metal's raw value `raw`, where `supportsTextureWriteRoundingMode:` answers
    /// yes (0 to 2); a larger one is `InvalidDescriptor(UnknownEnumValue)`, where Metal's pipeline
    /// creation stops at its descriptor validation ("device does not support Unknown", with or
    /// without the debug layer).
    pub fn from_raw(raw: u64) -> Result<TextureWriteRoundingMode, crate::error::Error> {
        match raw {
            0 | 1 => Ok(TextureWriteRoundingMode::TowardZero),
            2 => Ok(TextureWriteRoundingMode::NearestEven),
            _ => Err(crate::error::Error::InvalidDescriptor(
                crate::error::DescriptorError::UnknownEnumValue {
                    ty: "MTLTextureWriteRoundingMode",
                    value: u32::try_from(raw).unwrap_or(u32::MAX),
                },
            )),
        }
    }
}

pub(crate) mod bicubic;
pub mod format;
pub mod sampler;
pub mod state;
pub(crate) mod texel_format;
pub mod vertex;

#[cfg(test)]
mod tests;
