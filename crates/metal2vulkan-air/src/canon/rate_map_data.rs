//! Rasterization-rate map parameter data: Apple's layout, the bytes
//! `MTLRasterizationRateMap.copyParameterDataToBuffer:offset:` writes, and the decoders that
//! `rasterization_rate_map_decoder` and the map's coordinate queries compile to (Apple's layout and
//! arithmetic measured).
//!
//! AIR passes a `constant rasterization_rate_map_data &` buffer to the opaque
//! `air.map_screen_to_physical_coordinates` and `air.map_physical_to_screen_coordinates`, which
//! Apple's GPU compiler lowers against the tables below. Canonicalization compiles each call to
//! this module's shader decoder over the same bytes (a format canonicalization reads lives with
//! it). The consumer, which implements `MTLRasterizationRateMap`, builds the map and its bytes with
//! the translator's `bind::RasterizationRateMap`, whose CPU answers are this module's map decoders:
//! so a buffer holds Metal's bytes whichever side wrote them, and every answer is Metal's
//! arithmetic over them.
//!
//! **Measured** (Apple M4, macOS 27.0.1; 54 maps of 1 to 16384 pixels a side, zero to two layers,
//! uniform, two-sample, many-sample and two-layer qualities, nine of them with offsets at or near a
//! half step; every byte of each map's data; 610,387 CPU answers and 609,587 GPU answers under fast
//! and as many under safe math, along each axis and at signed zeros, denormals, `NaN`s, infinities
//! and far values; 225,672 more of a shader's, layers 0 to 5 over the data followed by a pattern):
//! this module's encoder writes each map's 24,624 bytes, and its decoders give every answer, bit
//! for bit, the sign of a zero included, but a shader's read past the buffer (Metal reads what
//! follows it). An M3 Max on macOS 26.7 gave the first two probes' maps the same bytes and every
//! answer alike.
//!
//! **The map.** Per layer and axis the screen extent `[0, W)` is cut into tiles. Tile `k` covers
//! screen `[S_k, S_k + s_k)` and physical `[P_k, P_k + p_k)`, every tile but the last
//! [`GRANULARITY`] physical pixels wide and at least as many screen pixels wide; `W` is at most
//! 16384. Its line is Apple's fixed point (`tile_line`): the scale `K / 2^24`, for
//! `K = round(p_k · 2^24 / s_k)` (never a tie: `s_k` is below `2^25`), its `f32` reciprocal, and the
//! offset `O / 256`, for `O = round(256 · (P_k - S_k · p_k / s_k))`, the tile's exact line at
//! screen 0 (not the fixed-point scale's), so physical `≈ s · scale + offset`. `O` is never a tie
//! on a tile Apple's compile makes (`offset_tie`).
//!
//! **Layout**: 24,624 bytes ([`BYTES`]), six tables of 513 entries ([`ENTRIES`]) of two
//! little-endian 32-bit words each, per 32-pixel granule ([`GRANULARITY`]):
//!
//! | words | table |
//! |---|---|
//! | `[0, 1026)` | layer 0, horizontal, physical to screen ([`HORIZONTAL_TO_SCREEN`]) |
//! | `[1026, 2052)` | layer 1, horizontal, physical to screen |
//! | `[2052, 3078)` | layer 0, horizontal, screen to physical ([`HORIZONTAL_TO_PHYSICAL`]) |
//! | `[3078, 4104)` | layer 1, horizontal, screen to physical |
//! | `[4104, 5130)` | vertical (every layer's), physical to screen ([`VERTICAL_TO_SCREEN`]) |
//! | `[5130, 6156)` | vertical, screen to physical ([`VERTICAL_TO_PHYSICAL`]) |
//!
//! - A physical-to-screen entry `k` is the line of the tile that holds physical granule `k`
//!   (tile `k`, every tile but the last being one granule): the `f32` reciprocal, then the `f32`
//!   offset. There are `min(⌈P / 32⌉ + 1, 513)` for a physical extent `P`, the last tile's line
//!   repeated after it, and zero words after them.
//! - A screen-to-physical entry `j` is the line of the tile that holds screen `32·j`: the `f32`
//!   scale, then the offset `O` as a signed 24-bit integer shifted left 8, over a boundary byte:
//!   `2·(S_{k+1} - 32·j)` (half pixels) where the next tile starts inside the granule, else
//!   [`NO_BOUNDARY`]. There are `min(⌈W / 32⌉ + 2, 513)`, the last tile's after it, and zero words
//!   after them.
//! - A layer the map lacks (a map of one layer's layer 1, every horizontal table of a map of no
//!   layers) is zero words. A map of no layers has a vertical axis all the same: the axis of no
//!   quality function, every sample at the least quality.
//! - An axis of no screen pixels has one tile of no screen and no physical pixels, whose line
//!   Apple writes as the least step the fixed point holds ([`LEAST_SCALE_STEPS`]): a scale of
//!   `2^-24`, a reciprocal of `2^24` and an offset of 0, in two entries of each table, whatever
//!   the axis's quality (measured on an Apple M3 Max under macOS 27.0.1: every word of maps of
//!   0 by 64 at qualities 1, 0.5 and 0.01, of 64 by 0, of 0 by 0, of 0 by 256, and of 0 by 64
//!   over two layers, where layer 1's horizontal tables hold the same). A map of no layers keeps
//!   no horizontal table at all, so a zero screen width leaves those words zero.
//!
//! **Decoding** (per axis):
//!
//! - screen to physical, as the map answers it (Apple's CPU, `physicalCoordinates`, the layer
//!   clamped to the map's by the caller): `j` the integer part of `s / 32` clamped to the table's
//!   last written entry (`NaN` reading 0); where the entry's boundary byte `b` is not
//!   [`NO_BOUNDARY`] and `2·(s - 32·j) ≥ b`, `b` read as its low seven bits
//!   ([`BOUNDARY_MASK`]), entry `j + 1` instead; the result is
//!   `fma(s, scale, O / 256)`, one rounding. Past either end of the screen the first or last
//!   tile's line extends without limit;
//! - screen to physical, as a shader answers it (Apple's GPU, `map_screen_to_physical_coordinates`):
//!   `j` the integer part of `(s + 1/2) / 32`, saturating (`NaN` and below 0 reading 0, from
//!   `2^32` on `2^32 - 1`) and not clamped, the same boundary test on `s`, and
//!   `s · scale + O / 256`, two roundings. The entry's words are `table + 2·j` and on, in 32-bit
//!   arithmetic that wraps: past the last written entry a shader reads the words that follow (the
//!   zero entries, the next table, past the data the buffer's next bytes), so past
//!   `⌈E/32⌉·32 + 32` physical to screen it answers 0 where the data's entries end;
//! - physical to screen: `k` the integer part of `p / 32`, the map's clamped to the table's last
//!   written entry, a shader's saturating and unclamped as above, and `(p - offset) · reciprocal`,
//!   two roundings;
//! - a shader's `uint2` overloads (`.v2i32`): screen to physical answers the `float2` decoder's
//!   answer at the pixel's centre, `c + 1/2`, converted to `uint` (its integer part, saturating);
//!   physical to screen reads the entry of granule `c >> 5` as the `float2` decoder does and
//!   answers the bits of the `float` `(0 - offset) · reciprocal`, two roundings: the coordinate's
//!   value is not read, so every coordinate of a granule answers alike and the answer is a
//!   `float`'s bits, not a screen coordinate. Measured on an Apple M3 Max under macOS 27.0.1 (our
//!   probe `rate_map_integer_overloads`): every coordinate along each axis of four maps (64 by 64
//!   of quality 1; 256 by 128 falling to 0.25; 200 by 96 of two layers; 3648 by 1734 falling to
//!   0.3), from 0 to 48 past the larger of the axis's screen and physical extents, at each map's
//!   layers and two past them, 20,114 coordinates in all, both overloads, every answer.
//!
//! The shader's operations read Metal's `float` denormals: each operand and result that is
//! denormal is the zero of its sign, and a `NaN` result is Metal's `0x7fc00000` (the translated
//! module's `fadd`, `fsub`, `fmul`, and comparisons leave both to the device, which answers so
//! where it is measured to read denormals as Metal does). Its horizontal table is the layer's, unclamped: layer `l` reads the words `l ·
//! 1026` past layer 0's, as Apple's GPU does (a map of one layer's layer 1 reads zeros, layer 2 the
//! screen-to-physical tables as physical-to-screen ones); its vertical table is every layer's. The
//! map's operations are IEEE `f32`, denormals kept, a `NaN` the input's. A read past the words
//! reads 0 here; in a module it is the buffer's contained read, so any bytes give some value, never
//! a fault (Metal reads what follows the buffer).

/// The parameter data's size in bytes, for every map (`parameterBufferSizeAndAlign.size`).
pub const BYTES: u32 = 24_624;
/// The parameter data's 32-bit words.
pub const WORDS: u32 = BYTES / 4;
/// The entries of one table.
pub const ENTRIES: u32 = 513;
/// The last entry: a granule index is clamped to it.
pub const LAST_ENTRY: u32 = ENTRIES - 1;
/// The words of one entry.
pub const ENTRY_WORDS: u32 = 2;
/// The words of one table.
pub const TABLE_WORDS: u32 = ENTRIES * ENTRY_WORDS;
/// The layers the data holds a horizontal axis for.
pub const LAYERS: u32 = 2;
/// The first word of layer 0's horizontal physical-to-screen table; layer `l`'s starts
/// `l · TABLE_WORDS` on.
pub const HORIZONTAL_TO_SCREEN: u32 = 0;
/// The first word of layer 0's horizontal screen-to-physical table; layer `l`'s starts
/// `l · TABLE_WORDS` on.
pub const HORIZONTAL_TO_PHYSICAL: u32 = LAYERS * TABLE_WORDS;
/// The first word of the vertical physical-to-screen table, every layer's.
pub const VERTICAL_TO_SCREEN: u32 = 2 * LAYERS * TABLE_WORDS;
/// The first word of the vertical screen-to-physical table, every layer's.
pub const VERTICAL_TO_PHYSICAL: u32 = VERTICAL_TO_SCREEN + TABLE_WORDS;
/// The pixels of one granule: every tile but an axis's last is this many physical pixels wide.
pub const GRANULARITY: u32 = 32;
/// `1 / GRANULARITY`, exact.
pub const INV_GRANULARITY: f32 = 1.0 / 32.0;
/// `GRANULARITY`'s power of two: a shader's `uint` physical coordinate's granule is the
/// coordinate shifted right this many bits.
pub const GRANULE_SHIFT: u32 = GRANULARITY.trailing_zeros();
/// The largest screen extent the tables hold (512 granules; entry 512 lies past it).
pub const MAX_EXTENT: u32 = LAST_ENTRY * GRANULARITY;
/// A screen-to-physical entry's boundary byte where no tile starts inside its granule.
pub const NO_BOUNDARY: u32 = 0xff;
/// The bits of a boundary byte the decoders compare: its low seven. A boundary is at most 64 half
/// pixels; a byte of `0x80`, which a shader reads only in another table's words (a layer past the
/// map's), compares as 0 on Apple's GPU (measured on an Apple M3 Max, macOS 27.0.1).
pub const BOUNDARY_MASK: u32 = 0x7f;
/// A screen-to-physical entry's second word holds the boundary byte below this many bits.
pub const OFFSET_SHIFT: u32 = 8;
/// The unit of a screen-to-physical entry's offset: `1/256` pixel.
pub const OFFSET_UNIT: f32 = 1.0 / 256.0;
/// The fixed-point bits of a scale: a tile's scale is `K / 2^24`.
pub const SCALE_BITS: u32 = 24;
/// The `K` of the one tile of an axis of no screen pixels: the least step the fixed point holds,
/// so the scale is `2^-24` and its reciprocal `2^24`. Apple's, measured on an Apple M3 Max under
/// macOS 27.0.1 (the module docs).
pub const LEAST_SCALE_STEPS: u32 = 1;
/// The largest value the map's index converts (`2^24`): an `f32` at most this converts to an
/// integer exactly.
pub const INDEX_CAP: f32 = 16_777_216.0;
/// Where a shader's index saturates (`2^32`): from it on, the index is `u32::MAX`.
pub const SHADER_INDEX_SATURATION: f32 = 4_294_967_296.0;
/// What a shader's screen-to-physical decoder adds to the coordinate before it looks up the
/// coordinate's granule: half a pixel.
pub const SHADER_GRANULE_OFFSET: f32 = 0.5;

/// A tile's line in Apple's fixed point (the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileLine {
    /// The screen-to-physical scale, `K / 2^24`.
    pub scale: f32,
    /// The physical-to-screen scale: the `f32` reciprocal of `scale`.
    pub reciprocal: f32,
    /// The offset in units of [`OFFSET_UNIT`].
    pub offset: i32,
}

/// The offset of a tile's line in units of [`OFFSET_UNIT`] times its screen extent:
/// `256 · (P · s - S · p)`, exact.
fn offset_numerator(
    screen_start: u32,
    physical_start: u32,
    screen_extent: u32,
    physical_extent: u32,
) -> i64 {
    let (s0, p0) = (i64::from(screen_start), i64::from(physical_start));
    let (se, pe) = (i64::from(screen_extent), i64::from(physical_extent));
    (p0 * se - s0 * pe) << OFFSET_SHIFT
}

/// Whether the offset of a tile's line (the module docs) lies exactly halfway between two
/// steps of [`OFFSET_UNIT`], where how Apple rounds it is not measured. A tie needs the screen
/// extent `s` to be `2^9` times the power of two in `P · s - S · p`. Apple's compile never makes
/// such a tile: every `P` is a multiple of 32, and every `p` is 32 but on an axis's last tile,
/// where an `s` that is a multiple of `2^9` maps to 8, 16, 24 or 32 physical pixels (the sixteenths
/// rule), so `s` would have to be a multiple of `2^12`, past the 3200 screen pixels a tile spans at
/// most.
pub fn offset_tie(
    screen_start: u32,
    physical_start: u32,
    screen_extent: u32,
    physical_extent: u32,
) -> bool {
    let se = i64::from(screen_extent.max(1));
    let num = offset_numerator(screen_start, physical_start, screen_extent, physical_extent);
    (2 * num).rem_euclid(2 * se) == se
}

/// The line of a tile from screen `screen_start` and physical `physical_start`, of
/// `screen_extent` screen pixels and `physical_extent` physical ones. An offset
/// tie ([`offset_tie`], never Apple's) rounds to the even step. A tile of no physical pixels —
/// only the one tile of an axis of no screen pixels ([`LEAST_SCALE_STEPS`]) — has the least
/// scale the fixed point holds, Apple's.
pub fn tile_line(
    screen_start: u32,
    physical_start: u32,
    screen_extent: u32,
    physical_extent: u32,
) -> TileLine {
    let se = i64::from(screen_extent.max(1));
    // `round(p · 2^24 / s)`: the quotient's half is never a whole number, as `s < 2^25`.
    let k = ((i64::from(physical_extent) << (SCALE_BITS + 1)) + se) / (2 * se);
    // `K` is 0 only where the tile has no physical pixel, which only an axis of no screen pixels
    // has; Apple writes the least step there, measured (the module docs).
    let k = k.max(i64::from(LEAST_SCALE_STEPS));
    // `K` is at most `2^24` for a tile of quality at most 1, so the scale is exact.
    let scale = k as f32 / (1u64 << SCALE_BITS) as f32;
    let num = offset_numerator(screen_start, physical_start, screen_extent, physical_extent);
    let (floor, rest) = (num.div_euclid(se), num.rem_euclid(se));
    let up = 2 * rest > se || (2 * rest == se && floor % 2 != 0);
    TileLine {
        scale,
        reciprocal: 1.0 / scale,
        offset: (floor + i64::from(up)) as i32,
    }
}

/// Write one axis's tables: tiles `(screen extent, physical extent)` from screen 0, every one but
/// the last [`GRANULARITY`] physical pixels wide and at least as many screen pixels, into
/// `to_screen` and `to_physical` (each a table's [`TABLE_WORDS`] words). It writes both words of
/// each entry the axis has and no other word, so a table that held another axis keeps that axis's
/// entries past this one's, as Metal's reset of a map leaves them. Entries past a table's words
/// are not written.
pub fn encode_axis(tiles: &[(u32, u32)], to_screen: &mut [u32], to_physical: &mut [u32]) {
    let mut starts = Vec::with_capacity(tiles.len());
    let (mut s, mut p) = (0u32, 0u32);
    for &(se, pe) in tiles {
        starts.push((s, tile_line(s, p, se, pe)));
        s = s.saturating_add(se);
        p = p.saturating_add(pe);
    }
    let Some(&(_, last)) = starts.last() else {
        return;
    };
    let put = |table: &mut [u32], entry: u32, words: [u32; 2]| {
        let at = (entry * ENTRY_WORDS) as usize;
        if let Some(slot) = table.get_mut(at..at + 2) {
            slot.copy_from_slice(&words);
        }
    };
    let screen = s;
    // One entry per tile and one more, the last tile's line repeated. Every tile but the last is
    // a whole granule, so the tiles are `⌈P / 32⌉` but on an axis of no screen pixels, whose one
    // tile has no physical pixel and whose table Apple writes two entries of all the same.
    for k in 0..(tiles.len() as u32 + 1).min(ENTRIES) {
        let line = starts.get(k as usize).map_or(last, |&(_, line)| line);
        let offset = line.offset as f32 * OFFSET_UNIT;
        put(to_screen, k, [line.reciprocal.to_bits(), offset.to_bits()]);
    }
    let mut tile = 0usize;
    for j in 0..(screen.div_ceil(GRANULARITY) + 2).min(ENTRIES) {
        let x = j * GRANULARITY;
        while starts.get(tile + 1).is_some_and(|&(start, _)| start <= x) {
            tile += 1;
        }
        let line = starts.get(tile).map_or(last, |&(_, line)| line);
        let boundary = match starts.get(tile + 1) {
            Some(&(next, _)) if next > x && next < x + GRANULARITY => 2 * (next - x),
            _ => NO_BOUNDARY,
        };
        let packed = ((line.offset as u32) << OFFSET_SHIFT) | boundary;
        put(to_physical, j, [line.scale.to_bits(), packed]);
    }
}

/// Word `i` of `words`, or 0 past them.
fn word(words: &[u32], i: u32) -> u32 {
    words.get(i as usize).copied().unwrap_or(0)
}

/// Entry `j` of the table at word `table`: its two words.
fn entry(words: &[u32], table: u32, j: u32) -> (u32, u32) {
    let at = table.wrapping_add(j.wrapping_mul(ENTRY_WORDS));
    (word(words, at), word(words, at.wrapping_add(1)))
}

/// The map's index: the integer part of `t` clamped to `[0, last]`; `NaN` is 0.
pub fn index(t: f32, last: u32) -> u32 {
    let a = if t >= 0.0 { t } else { 0.0 };
    let a = if a < INDEX_CAP { a } else { INDEX_CAP };
    // `a` is in `[0, 2^24]`: the conversion truncates exactly.
    let i = a as u32;
    if i < last {
        i
    } else {
        last
    }
}

/// A shader's index: the integer part of `t`, saturating (`NaN` and below 0 are 0, from
/// [`SHADER_INDEX_SATURATION`] on `u32::MAX`), unclamped.
pub fn shader_index(t: f32) -> u32 {
    // Rust's conversion saturates so, `NaN` to 0.
    t as u32
}

/// The last entry the encoder writes in a screen-to-physical table of screen extent `extent`
/// (the map's index is clamped to it).
pub fn last_screen_entry(extent: u32) -> u32 {
    (extent.div_ceil(GRANULARITY) + 1).min(LAST_ENTRY)
}

/// The last entry the encoder writes in a physical-to-screen table of physical extent `extent`
/// (the map's index is clamped to it): one per tile, the tiles being `⌈P / 32⌉`, or the one tile
/// of an axis of no screen pixels, whose table holds two entries of all the same.
pub fn last_physical_entry(extent: u32) -> u32 {
    extent.div_ceil(GRANULARITY).clamp(1, LAST_ENTRY)
}

/// The first word of the table of `axis` (0 horizontal, 1 vertical) in `direction` for layer
/// `layer`, as a shader addresses it: a horizontal table `layer · TABLE_WORDS` past layer 0's,
/// unclamped and wrapping; the vertical one every layer's.
pub fn table(layer: u32, axis: u32, to_physical: bool) -> u32 {
    match (axis, to_physical) {
        (0, false) => HORIZONTAL_TO_SCREEN.wrapping_add(layer.wrapping_mul(TABLE_WORDS)),
        (0, true) => HORIZONTAL_TO_PHYSICAL.wrapping_add(layer.wrapping_mul(TABLE_WORDS)),
        (_, false) => VERTICAL_TO_SCREEN,
        (_, true) => VERTICAL_TO_PHYSICAL,
    }
}

/// Metal's reading of a `float` denormal in a shader: the zero of its sign.
fn flush(x: f32) -> f32 {
    if x.is_subnormal() {
        f32::from_bits(x.to_bits() & 0x8000_0000)
    } else {
        x
    }
}

/// Metal's `NaN` result in a shader: `0x7fc00000`.
fn canonical(x: f32) -> f32 {
    if x.is_nan() {
        f32::from_bits(0x7fc0_0000)
    } else {
        x
    }
}

/// Whose arithmetic a decoder follows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Decoder {
    /// The map's (Apple's CPU): IEEE `f32`, a fused multiply-add screen to physical, the index
    /// clamped to this last entry.
    Map { last: u32 },
    /// A shader's (Apple's GPU): denormals flushed, separate roundings, the index unclamped.
    Shader,
}

impl Decoder {
    /// An operand or result as the decoder reads it.
    fn read(self, x: f32) -> f32 {
        match self {
            Decoder::Map { .. } => x,
            Decoder::Shader => flush(x),
        }
    }

    /// The entry of the integer part of `t`.
    fn index(self, t: f32) -> u32 {
        match self {
            Decoder::Map { last } => index(t, last),
            Decoder::Shader => shader_index(t),
        }
    }
}

fn axis_to_physical(words: &[u32], table: u32, s: f32, decoder: Decoder) -> f32 {
    let r = |x| decoder.read(x);
    let s = r(s);
    let t = match decoder {
        Decoder::Map { .. } => s * INV_GRANULARITY,
        Decoder::Shader => r(r(s + SHADER_GRANULE_OFFSET) * INV_GRANULARITY),
    };
    let j = decoder.index(t);
    let (_, packed) = entry(words, table, j);
    let (byte, boundary) = (packed & NO_BOUNDARY, packed & BOUNDARY_MASK);
    let start = r(j as f32 * GRANULARITY as f32);
    let into = r(r(s - start) * 2.0);
    let j = if byte != NO_BOUNDARY && into >= boundary as f32 {
        j.wrapping_add(1)
    } else {
        j
    };
    let (scale, packed) = entry(words, table, j);
    let scale = r(f32::from_bits(scale));
    let offset = ((packed as i32) >> OFFSET_SHIFT) as f32 * OFFSET_UNIT;
    match decoder {
        Decoder::Map { .. } => s.mul_add(scale, offset),
        Decoder::Shader => canonical(r(r(s * scale) + offset)),
    }
}

fn axis_to_screen(words: &[u32], table: u32, p: f32, decoder: Decoder) -> f32 {
    let r = |x| decoder.read(x);
    let p = r(p);
    let k = decoder.index(r(p * INV_GRANULARITY));
    let (reciprocal, offset) = entry(words, table, k);
    let (reciprocal, offset) = (r(f32::from_bits(reciprocal)), r(f32::from_bits(offset)));
    let q = r(r(p - offset) * reciprocal);
    match decoder {
        Decoder::Map { .. } => q,
        Decoder::Shader => canonical(q),
    }
}

/// Apple's CPU map (`physicalCoordinates(screenCoordinates:layer:)`) of `c` over parameter data
/// `words`, reading layer `layer`'s tables (the caller clamps it to the map's layers), for a
/// map of screen extent `screen`.
pub fn screen_to_physical(words: &[u32], layer: u32, c: [f32; 2], screen: [u32; 2]) -> [f32; 2] {
    let [x, y] = c;
    let map = |e| Decoder::Map {
        last: last_screen_entry(e),
    };
    [
        axis_to_physical(words, table(layer, 0, true), x, map(screen[0])),
        axis_to_physical(words, table(layer, 1, true), y, map(screen[1])),
    ]
}

/// A shader's `map_screen_to_physical_coordinates(c, layer)` over parameter data `words`, as
/// Apple's GPU computes it, and the translated decoder on a device that reads `float` denormals as
/// Metal does.
pub fn shader_screen_to_physical(words: &[u32], layer: u32, c: [f32; 2]) -> [f32; 2] {
    let [x, y] = c;
    [
        axis_to_physical(words, table(layer, 0, true), x, Decoder::Shader),
        axis_to_physical(words, table(layer, 1, true), y, Decoder::Shader),
    ]
}

/// Apple's CPU map (`screenCoordinates(physicalCoordinates:layer:)`) of `c` over parameter data
/// `words`, reading layer `layer`'s tables (the caller clamps it to the map's layers), for a
/// layer of physical extent `physical`.
pub fn physical_to_screen(words: &[u32], layer: u32, c: [f32; 2], physical: [u32; 2]) -> [f32; 2] {
    let [x, y] = c;
    let map = |e| Decoder::Map {
        last: last_physical_entry(e),
    };
    [
        axis_to_screen(words, table(layer, 0, false), x, map(physical[0])),
        axis_to_screen(words, table(layer, 1, false), y, map(physical[1])),
    ]
}

/// A shader's `map_physical_to_screen_coordinates(c, layer)` over parameter data `words`, as
/// Apple's GPU computes it, and the translated decoder on a device that reads `float` denormals as
/// Metal does.
pub fn shader_physical_to_screen(words: &[u32], layer: u32, c: [f32; 2]) -> [f32; 2] {
    let [x, y] = c;
    [
        axis_to_screen(words, table(layer, 0, false), x, Decoder::Shader),
        axis_to_screen(words, table(layer, 1, false), y, Decoder::Shader),
    ]
}

/// A shader's `uint2` `map_screen_to_physical_coordinates(c, layer)` over parameter data
/// `words`, as Apple's GPU answers it (the module docs): per axis, the `float2` decoder's answer
/// at the pixel's centre, `c + 1/2`, converted to `uint` as a shader converts, saturating.
pub fn shader_screen_to_physical_integer(words: &[u32], layer: u32, c: [u32; 2]) -> [u32; 2] {
    let centre = c.map(|v| v as f32 + 0.5);
    shader_screen_to_physical(words, layer, centre).map(shader_index)
}

/// A shader's `uint2` `map_physical_to_screen_coordinates(c, layer)` over parameter data
/// `words`, as Apple's GPU answers it (the module docs): per axis, the bits of the `float`
/// `(0 - offset) · reciprocal` of the entry of granule `c >> 5`, the coordinate's own value not
/// read.
pub fn shader_physical_to_screen_integer(words: &[u32], layer: u32, c: [u32; 2]) -> [u32; 2] {
    let r = |x| Decoder::Shader.read(x);
    let axis = |axis: u32, p: u32| {
        let (reciprocal, offset) = entry(words, table(layer, axis, false), p >> GRANULE_SHIFT);
        let (reciprocal, offset) = (r(f32::from_bits(reciprocal)), r(f32::from_bits(offset)));
        canonical(r(r(0.0 - offset) * reciprocal)).to_bits()
    };
    [axis(0, c[0]), axis(1, c[1])]
}

#[cfg(test)]
mod tests;
