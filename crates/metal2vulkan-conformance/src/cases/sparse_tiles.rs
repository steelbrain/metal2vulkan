//! `MTLDevice`'s sparse tile getters: the tile a sparse texture maps in, and that tile's bytes.
//!
//! Four Metal selectors, all pure getters:
//!
//! - `sparseTileSizeWithTextureType:pixelFormat:sampleCount:`
//! - `sparseTileSizeWithTextureType:pixelFormat:sampleCount:sparsePageSize:`
//! - `sparseTileSizeInBytes`
//! - `sparseTileSizeInBytesForSparsePageSize:`
//!
//! # Why this family comes first of the sparse ones
//!
//! It is the one sparse family that needs **no sparse-capable device**. The getters compile
//! nothing, allocate nothing and reach no GPU, and neither side asks the device whether it binds
//! sparse memory: the runtime answers from a model of Metal's rule (`sparse::tile_texels`) and
//! Metal answers from its own. So the whole family compares on a venue that has no sparse binding
//! at all, and a lavapipe leg here reports real answers rather than a hole.
//!
//! # What the sweep checks
//!
//! The rule has two claims a sweep can check:
//!
//! - **the answer does not depend on the texture type**, which [`BY_TYPE`] asks of every texture
//!   type a sparse heap makes a texture of;
//! - **the tile halves a side as the block's bytes double**, keeping the tile's bytes at the
//!   page's, which [`BY_FORMAT`] asks of formats from 1 to 16 bytes a block and of the
//!   block-compressed families, in all three pages.
//!
//! # Why every case here is safe
//!
//! No case reaches the GPU, allocates or compiles. The one hazard a getter of Metal's can carry is
//! an argument its own API validation stops, which ends Metal's process, and the arguments here are
//! kept to the ones Metal describes a sparse texture of:
//!
//! - the **texture types** are the seven a sparse heap makes a texture of. `Type1D`, `Type1DArray`
//!   and `TextureBuffer`, for which `Heap::new_texture` answers nil, are left out: the model would
//!   answer for them as for any other type, and asking Metal is the only way to find out, but
//!   nothing in the tree says Metal's validation takes them. That is the next query to add, and it
//!   needs its own evidence first.
//! - the **sample counts** are 1 for a single-sampled type, and 2 and 4 for a multisample type,
//!   which are the counts measured on this oracle. A count a device does not hold, and a count on a
//!   single-sampled type, are left out for the same reason.
//! - the **formats** are colour formats and the four depth and stencil formats this chip has.
//!   `Depth24Unorm_Stencil8`, which Apple silicon does not support, is left out.
//!
//! Two guards hold that argument, run on a device: the arguments of every query are inside the
//! shapes above, and the runtime's model answers a real tile for each of them — a query the model
//! has no rule for is one whose Metal answer nothing in this tree can predict. Both run before
//! anybody records, which is what the suite's order of work asks for.

use crate::case::{Abort, Case, Input, MetalSource, Side};
use crate::sparse_query::{page_word, page_words, QueryPage, SparseTileQuery as Q};
use crate::transcript::{Sites, Value};
use crate::values::SparsePageSize;
use metal2vulkan::metal::{PixelFormat, TextureType};

/// The case names, as the report and the recordings spell them.
pub const BY_FORMAT: &str = "sparse-tile-size-by-format";
pub const BY_TYPE: &str = "sparse-tile-size-by-texture-type";
pub const MULTISAMPLE: &str = "sparse-tile-size-multisample";
pub const TILE_BYTES: &str = "sparse-tile-size-in-bytes";

/// The pages every case sweeps: the getter without a page, then each of Metal's three.
///
/// The pageless getter is swept beside `Size16` rather than assumed to equal it, because they are
/// two selectors and this family's business is what each selector answers.
const PAGES: [QueryPage; 4] = [
    None,
    Some(SparsePageSize::Size16),
    Some(SparsePageSize::Size64),
    Some(SparsePageSize::Size256),
];

/// The single-sampled texture types a sparse heap makes a texture of.
const SINGLE_SAMPLED_TYPES: [TextureType; 5] = [
    TextureType::Type2D,
    TextureType::Type2DArray,
    TextureType::Cube,
    TextureType::CubeArray,
    TextureType::Type3D,
];

/// The multisample texture types a sparse heap makes a texture of.
const MULTISAMPLE_TYPES: [TextureType; 2] = [
    TextureType::Type2DMultisample,
    TextureType::Type2DMultisampleArray,
];

/// The formats [`BY_FORMAT`] sweeps: the rule's whole range of block bytes, and both block
/// extents.
///
/// One format of each size would show the halving; two of a size show that the answer follows the
/// **bytes** and not the channel count or the sRGB bit. The compressed families are here because
/// the rule multiplies the tile by the block's texels, which is the part of it no uncompressed
/// format can show: a 4 by 4 block of 8 bytes (BC1, EAC/ETC2's one-bit-alpha pair) and of 16 bytes
/// (BC7, ASTC 4x4, EAC RGBA8).
const COLOUR_FORMATS: [PixelFormat; 18] = [
    // One byte a texel.
    PixelFormat::R8Unorm,
    PixelFormat::R8UnormSRGB,
    // Two.
    PixelFormat::R16Float,
    PixelFormat::RG8Unorm,
    // Four.
    PixelFormat::R32Float,
    PixelFormat::RGBA8Unorm,
    PixelFormat::RGBA8UnormSRGB,
    PixelFormat::BGR10A2Unorm,
    PixelFormat::RGB9E5Float,
    // Eight.
    PixelFormat::RG32Float,
    PixelFormat::RGBA16Float,
    PixelFormat::RGBA16Unorm,
    // Sixteen.
    PixelFormat::RGBA32Float,
    // Block-compressed: 4 by 4 blocks of 8 and of 16 bytes.
    PixelFormat::BC1RGBA,
    PixelFormat::ETC2RGB8A1,
    PixelFormat::BC7RGBAUnorm,
    PixelFormat::EACRGBA8,
    PixelFormat::ASTC4x4LDR,
];

/// The depth and stencil formats [`BY_FORMAT`] sweeps: the four this chip has.
///
/// They are in the family because the model answers them from a **separate** rule — a depth or
/// stencil format's tile counts its depth plane's bytes, 4 for `Depth32Float_Stencil8`, and its
/// block is one texel — and that rule's comment says "measured" with no measurement anywhere in the
/// tree. `Depth24Unorm_Stencil8` is left out: Apple silicon does not support it.
const DEPTH_FORMATS: [PixelFormat; 4] = [
    PixelFormat::Depth16Unorm,
    PixelFormat::Depth32Float,
    PixelFormat::Stencil8,
    PixelFormat::Depth32FloatStencil8,
];

/// The format [`BY_TYPE`] asks every texture type about.
const BY_TYPE_FORMAT: PixelFormat = PixelFormat::R32Float;

/// The formats [`MULTISAMPLE`] sweeps, whose block bytes the sample count multiplies differently.
const MULTISAMPLE_FORMATS: [PixelFormat; 4] = [
    PixelFormat::RGBA8Unorm,
    PixelFormat::R8Unorm,
    PixelFormat::R32Float,
    PixelFormat::RGBA32Float,
];

/// The sample counts [`MULTISAMPLE`] asks: the two measured on this oracle.
const MULTISAMPLE_COUNTS: [u32; 2] = [2, 4];

/// A case over one table of tile queries, and optionally over the tile's bytes.
pub struct SparseTileCase {
    name: &'static str,
    queries: Vec<Q>,
    /// Whether the case also asks `sparseTileSizeInBytes` for each page of [`PAGES`].
    ///
    /// One case asks it, because the answer depends on the page alone: asking it beside every
    /// query would record the same four numbers a few hundred times.
    asks_bytes: bool,
}

/// Every case of the family.
pub fn all() -> Vec<SparseTileCase> {
    vec![by_format(), by_texture_type(), multisample(), tile_bytes()]
}

/// Every format of [`COLOUR_FORMATS`] and [`DEPTH_FORMATS`], as a single-sampled 2D texture, in
/// each page.
fn by_format() -> SparseTileCase {
    let mut queries = Vec::new();
    for format in COLOUR_FORMATS.into_iter().chain(DEPTH_FORMATS) {
        for page in PAGES {
            queries.push(Q::new(TextureType::Type2D, format, 1, page));
        }
    }
    SparseTileCase {
        name: BY_FORMAT,
        queries,
        asks_bytes: false,
    }
}

/// Every texture type a sparse heap makes a texture of, at the sample count that type takes, over
/// three formats of different block bytes, in each page.
///
/// This is the case that checks the model's type-independence claim against Metal rather than
/// against the model's own comment. Three formats and not one, so a type that answered differently
/// for a wide block than for a narrow one would show it.
fn by_texture_type() -> SparseTileCase {
    let mut queries = Vec::new();
    for texture_type in SINGLE_SAMPLED_TYPES {
        for format in [
            BY_TYPE_FORMAT,
            PixelFormat::R8Unorm,
            PixelFormat::ASTC4x4LDR,
        ] {
            for page in PAGES {
                queries.push(Q::new(texture_type, format, 1, page));
            }
        }
    }
    SparseTileCase {
        name: BY_TYPE,
        queries,
        asks_bytes: false,
    }
}

/// The multisample types at 2 and 4 samples, over four formats, in each page.
fn multisample() -> SparseTileCase {
    let mut queries = Vec::new();
    for texture_type in MULTISAMPLE_TYPES {
        for samples in MULTISAMPLE_COUNTS {
            for format in MULTISAMPLE_FORMATS {
                for page in PAGES {
                    queries.push(Q::new(texture_type, format, samples, page));
                }
            }
        }
    }
    SparseTileCase {
        name: MULTISAMPLE,
        queries,
        asks_bytes: false,
    }
}

/// `sparseTileSizeInBytes` and its per-page form, beside the tile of one format in each page, so
/// the recording holds both halves of the rule's invariant — the tile's texels and the tile's bytes
/// — and a reader can check that one tile is one page.
fn tile_bytes() -> SparseTileCase {
    let mut queries = Vec::new();
    for page in PAGES {
        queries.push(Q::new(TextureType::Type2D, BY_TYPE_FORMAT, 1, page));
    }
    SparseTileCase {
        name: TILE_BYTES,
        queries,
        asks_bytes: true,
    }
}

/// The site a page's tile bytes sit at.
fn bytes_site(page: QueryPage) -> String {
    format!("tile-bytes/{}", page_word(page))
}

impl Case for SparseTileCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // The queries are the whole of what the case feeds, and the order it asks them in: a table
        // that gains, loses or re-points a row is a change to the case's inputs, so a recording
        // cannot decay into the answers of queries nobody asked about.
        let mut words = Vec::new();
        for query in &self.queries {
            words.extend(query.words());
        }
        let mut asks = Vec::new();
        if self.asks_bytes {
            for page in PAGES {
                asks.extend(page_words(page));
            }
        }
        Ok(vec![
            Input::bytes("queries", &words),
            Input::bytes("asks-tile-bytes", &asks),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for query in &self.queries {
            // Each query's answer is a site of its own, and a call that did not answer is recorded
            // at its site and the sweep goes on: one query the runtime refuses compares as a
            // refusal at that one site, and the other answers stay readable instead of being lost
            // behind an abort.
            crate::case::probe(sites, &query.site(), side.sparse_tile_size(query), |tile| {
                tile.value()
            });
        }
        if self.asks_bytes {
            for page in PAGES {
                crate::case::probe(
                    sites,
                    &bytes_site(page),
                    side.sparse_tile_size_in_bytes(page),
                    |bytes| Value::Uint(*bytes),
                );
            }
        }
        Ok(())
    }
}

/// The guard this family's safety argument rests on: record mode hands every query of these
/// tables to Metal, and an argument Metal's own API validation stops **ends Metal's process**,
/// so the arguments stay inside the shapes the module's prose allows.
#[cfg(test)]
mod tests {
    use super::*;

    /// The family's tables hold no query whose arguments the module's own safety prose excludes.
    ///
    /// The prose is the family's safety argument, so it needs a check rather than a reader: a
    /// texture type a sparse heap makes no texture of, a sample count other than 1 on a
    /// single-sampled type, a multisample type at one sample, and `Depth24Unorm_Stencil8`, which
    /// Apple silicon does not support.
    ///
    /// Verified to fail: a `TextureBuffer` query and a `Depth24Unorm_Stencil8` one, each added to a
    /// table, fail here naming the query and which exclusion it breaks.
    #[test]
    fn no_query_of_the_family_is_one_the_safety_argument_excludes() {
        let mut bad = Vec::new();
        for case in all() {
            for query in &case.queries {
                let multisample = MULTISAMPLE_TYPES.contains(&query.texture_type);
                let single = SINGLE_SAMPLED_TYPES.contains(&query.texture_type);
                if !multisample && !single {
                    bad.push(format!(
                        "{}: {} is not a sparse texture type",
                        case.name,
                        query.site()
                    ));
                }
                if single && query.samples != 1 {
                    bad.push(format!(
                        "{}: {} is a single-sampled type at more than one sample",
                        case.name,
                        query.site()
                    ));
                }
                if multisample && !MULTISAMPLE_COUNTS.contains(&query.samples) {
                    bad.push(format!(
                        "{}: {} is a sample count the oracle was not measured at",
                        case.name,
                        query.site()
                    ));
                }
                if query.format == PixelFormat::Depth24UnormStencil8 {
                    bad.push(format!(
                        "{}: {} names a format Apple silicon does not support",
                        case.name,
                        query.site()
                    ));
                }
            }
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }
}
