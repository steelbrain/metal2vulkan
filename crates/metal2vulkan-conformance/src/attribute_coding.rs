//! What one `MTLVertexFormat` or `MTLAttributeFormat` is, and the codes that separate the decodings.
//!
//! One vocabulary for the two Metal types, because they **are** one: the generated tables give
//! `MTLAttributeFormat` and `MTLVertexFormat` the same 54 enumerator names with the same ordinals,
//! and the same decoding behind each. Two sweeps with two shape parsers and two code tables would
//! be two answers to one question, so the shape and the codes live here and each family states
//! only what its own path can reach.
//!
//! The shape is derived from the SDK enumerator's **own name**, which states it:
//! `MTLVertexFormatUChar4Normalized_BGRA` is four unsigned bytes, normalized, in BGRA order. So a
//! format the SDK adds either parses here or is reported unclassified by the family's own test
//! rather than dropped from a sweep in silence, and the denominator stays the generated table's.
//!
//! The twelve code lists are each encoding's documented edges; `cases::vertex_formats`'s module
//! docs hold the table of what each list holds and why those codes separate the decodings.

/// Bytes each format's slot takes in a record: the widest format the SDK table holds is 16 bytes,
/// and 16 is a multiple of every component width, so one constant keeps every attribute's offset
/// aligned whatever slot a format sits in.
pub const SLOT: u32 = 16;

/// The byte a record holds where no format's component lies, so a fetch that read past its own
/// components read a value no code list holds.
pub const FILLER: u8 = 0xEE;

/// Rows a sweep compares: one fetch of every format per row.
pub const ROWS: usize = 12;

/// How a format's components are coded, as the SDK enumerator's name states it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coding {
    /// Components of `bytes` bytes, read as the integer they hold.
    Integer { bytes: usize, signed: bool },
    /// Components of `bytes` bytes, scaled into `[0, 1]` or `[-1, 1]`.
    Normalized { bytes: usize, signed: bool },
    /// IEEE binary16 components.
    Half,
    /// IEEE binary32 components.
    Single,
    /// One 32-bit word holding four normalized fields of 2, 10, 10 and 10 bits.
    Packed1010102 { signed: bool },
    /// One 32-bit word holding three unsigned floats of 11, 11 and 10 bits.
    PackedRG11B10,
    /// One 32-bit word holding three 9-bit mantissas and a shared 5-bit exponent.
    PackedRGB9E5,
}

impl Coding {
    /// Bytes one component of the coding takes; for a packed coding, the whole word.
    pub fn width(self) -> usize {
        match self {
            Coding::Integer { bytes, .. } | Coding::Normalized { bytes, .. } => bytes,
            Coding::Half => 2,
            Coding::Single
            | Coding::Packed1010102 { .. }
            | Coding::PackedRG11B10
            | Coding::PackedRGB9E5 => 4,
        }
    }

    /// Whether the coding puts every component in one word, so the row's code is the word and not
    /// one component of it.
    pub fn packed(self) -> bool {
        matches!(
            self,
            Coding::Packed1010102 { .. } | Coding::PackedRG11B10 | Coding::PackedRGB9E5
        )
    }

    /// Whether a float-declared attribute can read the coding.
    ///
    /// Every coding but a 32-bit integer one: Metal converts an 8- or 16-bit integer format to
    /// float in a float attribute (`integer-vertex-attributes` records it doing so), while a
    /// 32-bit integer format reaches the shader as an integer and needs an integer-declared
    /// attribute.
    pub fn float_readable(self) -> bool {
        !matches!(self, Coding::Integer { bytes: 4, .. })
    }
}

/// What one `MTLVertexFormat` is, derived from the SDK enumerator's own name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// Components the shader sees.
    pub components: usize,
    /// Bytes the format occupies in one element.
    pub size: usize,
    pub coding: Coding,
    /// Whether the format presents its first and third components swapped (`_BGRA`).
    pub bgra: bool,
}

/// The shape the SDK enumerator `name` states, or `None` where the name states none.
///
/// The name is the SDK's own statement of the shape — `MTLVertexFormatUChar4Normalized_BGRA` is
/// four unsigned bytes, normalized, in BGRA order — so reading it keeps this family's denominator
/// the generated table's rather than a list written here. A name the parse does not cover returns
/// `None`, which `tests::every_vertex_format_of_the_sdk_table_is_classified` turns into a
/// failure naming it.
pub fn shape(name: &str) -> Option<Shape> {
    let tail = name
        .strip_prefix("MTLVertexFormat")
        .or_else(|| name.strip_prefix("MTLAttributeFormat"))?;
    let (tail, bgra) = match tail.strip_suffix("_BGRA") {
        Some(head) => (head, true),
        None => (tail, false),
    };
    let (tail, normalized) = match tail.strip_suffix("Normalized") {
        Some(head) => (head, true),
        None => (tail, false),
    };
    // The packed formats name their field widths, so each is matched whole before a trailing digit
    // could be read as a component count: taking one off `Int1010102` leaves `Int101010`.
    let packed = match tail {
        "Int1010102" => Some((4, Coding::Packed1010102 { signed: true })),
        "UInt1010102" => Some((4, Coding::Packed1010102 { signed: false })),
        "FloatRG11B10" => Some((3, Coding::PackedRG11B10)),
        "FloatRGB9E5" => Some((3, Coding::PackedRGB9E5)),
        _ => None,
    };
    if let Some((components, coding)) = packed {
        return Some(Shape {
            components,
            size: coding.width(),
            coding,
            bgra,
        });
    }
    let (base, components) = match tail.as_bytes().last() {
        Some(digit @ b'2'..=b'4') => (&tail[..tail.len() - 1], usize::from(digit - b'0')),
        _ => (tail, 1),
    };
    let coding = match (base, normalized) {
        ("Half", false) => Coding::Half,
        ("Float", false) => Coding::Single,
        ("UChar", true) => Coding::Normalized {
            bytes: 1,
            signed: false,
        },
        ("Char", true) => Coding::Normalized {
            bytes: 1,
            signed: true,
        },
        ("UShort", true) => Coding::Normalized {
            bytes: 2,
            signed: false,
        },
        ("Short", true) => Coding::Normalized {
            bytes: 2,
            signed: true,
        },
        ("UChar", false) => Coding::Integer {
            bytes: 1,
            signed: false,
        },
        ("Char", false) => Coding::Integer {
            bytes: 1,
            signed: true,
        },
        ("UShort", false) => Coding::Integer {
            bytes: 2,
            signed: false,
        },
        ("Short", false) => Coding::Integer {
            bytes: 2,
            signed: true,
        },
        ("UInt", false) => Coding::Integer {
            bytes: 4,
            signed: false,
        },
        ("Int", false) => Coding::Integer {
            bytes: 4,
            signed: true,
        },
        _ => return None,
    };
    Some(Shape {
        components,
        size: coding.width() * components,
        coding,
        bgra,
    })
}

// ---- The codes ---------------------------------------------------------------------------------

/// The twelve codes one component of `coding` takes, each as the little-endian integer of the
/// component's own width (or of the whole word, for a packed coding).
///
/// The module's prose says what each list holds and why; the one row of it the rest of the family
/// leans on is that index 3 and index 4 of a signed list are the minimum code and the code above
/// it, both of which a signed normalized decode maps to -1.0.
pub fn codes(coding: Coding) -> [u64; ROWS] {
    match coding {
        Coding::Integer {
            bytes: 1,
            signed: false,
        }
        | Coding::Normalized {
            bytes: 1,
            signed: false,
        } => [
            0, 1, 2, 0x7F, 0x80, 0x81, 0xC8, 0xFD, 0xFE, 0xFF, 0x55, 0xAA,
        ],
        Coding::Integer {
            bytes: 1,
            signed: true,
        }
        | Coding::Normalized {
            bytes: 1,
            signed: true,
        } => [
            0, 1, 0x80, 0x81, 0x7F, 0xFF, 0x02, 0xFE, 0x64, 0x9C, 0x40, 0xC0,
        ],
        Coding::Integer {
            bytes: 2,
            signed: false,
        }
        | Coding::Normalized {
            bytes: 2,
            signed: false,
        } => [
            0, 1, 2, 0x7FFF, 0x8000, 0x8001, 0xFFFD, 0xFFFE, 0xFFFF, 0x5555, 0xAAAA, 0x03E8,
        ],
        Coding::Integer {
            bytes: 2,
            signed: true,
        }
        | Coding::Normalized {
            bytes: 2,
            signed: true,
        } => [
            0, 1, 0x8000, 0x8001, 0x7FFF, 0xFFFF, 2, 0xFFFE, 0x03E8, 0xFC18, 0x4000, 0xC000,
        ],
        Coding::Integer { signed: false, .. } | Coding::Normalized { signed: false, .. } => [
            0,
            1,
            2,
            0x7FFF_FFFF,
            0x8000_0000,
            0x8000_0001,
            0xFFFF_FFFD,
            0xFFFF_FFFE,
            0xFFFF_FFFF,
            0x5555_5555,
            0xAAAA_AAAA,
            0x3E8,
        ],
        Coding::Integer { signed: true, .. } | Coding::Normalized { signed: true, .. } => [
            0,
            1,
            0x8000_0000,
            0x8000_0001,
            0x7FFF_FFFF,
            0xFFFF_FFFF,
            2,
            0xFFFF_FFFE,
            0x3E8,
            0xFFFF_FC18,
            0x4000_0000,
            0xC000_0000,
        ],
        // +0, -0, the smallest subnormal, the largest subnormal, the smallest normal, 1.0, the
        // largest finite, +inf, -inf, a NaN payload and its negative, and 1/3 rounded.
        Coding::Half => [
            0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x7BFF, 0x7C00, 0xFC00, 0x7E01, 0xFE01,
            0x3555,
        ],
        Coding::Single => [
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x007F_FFFF,
            0x0080_0000,
            0x3F80_0000,
            0x7F7F_FFFF,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC0_0001,
            0xFFC0_0001,
            0x3EAA_AAAB,
        ],
        // Zero; each field alone at its maximum (x in bits 0..10, y in 10..20, z in 20..30, w in
        // 30..32); every field at its maximum; the 10-bit field at 512, 513 and 511, the first two
        // of which a signed normalized decode maps to -1.0; the 2-bit field at its two negative
        // signed values; and a mixed word.
        Coding::Packed1010102 { .. } => [
            0x0000_0000,
            0x0000_03FF,
            0x000F_FC00,
            0x3FF0_0000,
            0xC000_0000,
            0xFFFF_FFFF,
            0x0000_0200,
            0x0000_0201,
            0x0000_01FF,
            0x8000_0000,
            0x4000_0000,
            0x2AAA_AAAA,
        ],
        // Zero; the lowest set bit; the 11-bit field at its maximum and one past it; the second
        // field's boundary on both sides; the top field alone; every bit set, which puts an
        // all-ones exponent in every field; and three mixed words.
        Coding::PackedRG11B10 | Coding::PackedRGB9E5 => [
            0x0000_0000,
            0x0000_0001,
            0x0000_07FF,
            0x0000_0800,
            0x003F_F800,
            0x0040_0000,
            0xFFC0_0000,
            0xFFFF_FFFF,
            0x7C1F_07C1,
            0x0000_FFFF,
            0xFFFF_0000,
            0x3C00_0000,
        ],
    }
}

/// The bytes format `format`'s slot holds in row `row`.
///
/// A packed format's row is one word. Every other format's component `c` takes the code at index
/// `(row + 3 c) % ROWS`, so one fetch's components differ from each other and a decoder that
/// crossed two of them is caught by the values.
pub fn slot(name: &str, row: usize) -> Vec<u8> {
    let held = shape(name).expect("a swept format has a shape");
    let list = codes(held.coding);
    let width = held.coding.width();
    if held.coding.packed() {
        return list[row % ROWS].to_le_bytes()[..width].to_vec();
    }
    let mut out = Vec::with_capacity(held.size);
    for component in 0..held.components {
        let code = list[(row + 3 * component) % ROWS];
        out.extend_from_slice(&code.to_le_bytes()[..width]);
    }
    out
}

/// Bytes one record takes: one [`SLOT`] per swept format.
pub fn stride(names: &[&str]) -> u32 {
    SLOT * names.len() as u32
}

/// The records the vertex buffer holds: row `r`'s slot for each format at byte `SLOT i`, and
/// [`FILLER`] everywhere else.
pub fn records(names: &[&str]) -> Vec<u8> {
    let stride = stride(names) as usize;
    let mut out = vec![FILLER; stride * ROWS];
    for row in 0..ROWS {
        for (at, name) in names.iter().enumerate() {
            let bytes = slot(name, row);
            let first = row * stride + SLOT as usize * at;
            out[first..first + bytes.len()].copy_from_slice(&bytes);
        }
    }
    out
}
