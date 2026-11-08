//! `MTLDevice`'s pure queries, the ones whose answer is a **model of Metal's rule** rather than a
//! reading of the device.
//!
//! [`crate::cases::sparse_tiles`] states the test this family is built on, and states it exactly:
//! the runtime answers a sparse tile size "from a model of Metal's arithmetic that is the same on
//! every device, so the numbers are a contract with Metal and with nothing else, and the only thing
//! that can check them is Metal". Every query here passes that test, each one checked against the
//! runtime's own source:
//!
//! - `heapBufferSizeAndAlignWithLength:options:` answers `length` bytes at a multiple of 256 "for
//!   every length and options, as Metal answers (measured)" — a constant with the word *measured*
//!   beside it and, until now, no run behind it;
//! - `supportsVertexAmplificationCount:` answers yes up to a constant 8, which the runtime's own
//!   documentation attributes to "a Metal probe: the M3 Max with macOS 26.7 and an M4 with macOS
//!   27.0.1, alike" — a probe nobody kept;
//! - `supportsRasterizationRateMapWithLayerCount:` answers yes for 1 and 2, "the layer counts
//!   Apple's map compile was measured with and the compile layer reproduces on every device";
//! - `minimumLinearTextureAlignmentForPixelFormat:` answers at least 16 "for every format a buffer
//!   can back, as Metal answers on an M3 Max and an M4", and more only where the device's linear
//!   images need more: a reading of the device, which on the oracle's MoltenVK answers 16, so the
//!   family still holds it to Metal's;
//! - `convertSparsePixelRegions:…` and `convertSparseTileRegions:…` are arithmetic over the
//!   caller's regions and a tile size the caller states.
//!
//! # What this family deliberately leaves out, and why
//!
//! **Every `MTLDevice` query the runtime answers from the Vulkan device is absent.** `name`,
//! `maxBufferLength`, `maxThreadsPerThreadgroup`, `maxThreadgroupMemoryLength`,
//! `hasUnifiedMemory`, `argumentBuffersSupport`, `readWriteTextureSupport`,
//! `supports32BitFloatFiltering`, `areRasterOrderGroupsSupported`,
//! `areProgrammableSamplePositionsSupported`, `isDepth24Stencil8PixelFormatSupported`,
//! `supportsTextureSampleCount:` and `supportsFamily:` each answer from a property, a limit, a
//! feature or a format of the device the runtime opened. The reason they are out is the one
//! [`crate::pipeline_facts`] already gives for a kernel's `maxTotalThreadsPerThreadgroup`: a venue
//! that answers differently "is not diverging from Metal, it is a venue with a smaller device, and
//! the suite has no way to say that of a number". They are a named hole rather than a comparison
//! that would fail honestly on another venue, and closing it needs a vocabulary for "the device's
//! own answer" that the transcript does not have.
//!
//! # Why every case of this family is safe
//!
//! No query here compiles, allocates or reaches the GPU: each is CPU arithmetic over the case's own
//! numbers. The hazard a pure getter of Metal's carries is an argument its own validation stops,
//! which ends Metal's process, and each sweep is kept inside arguments something already measured:
//!
//! - the **pixel formats** are colour formats a buffer can back. Metal's alignment query
//!   *asserts* on a compressed, depth or stencil format, and on a format the host's Metal
//!   describes none of — which is exactly the set the runtime refuses `Invalid` for, so the
//!   order of work [`crate::cases::heap_metal_sizes`] states holds here: the runtime's leg runs
//!   first and a table that answers `Invalid` anywhere is never handed to Metal;
//! - the **lengths** are 1 and above. A length of 0 is left out: nothing in the tree says Metal's
//!   suballocator takes it, and the answer our model would give for it is not evidence that it
//!   does;
//! - the **layer counts** are 1 to 4. A count of 0 is left out for the same reason, although the
//!   runtime answers no for it;
//! - the **amplification counts** run 0 to 16, which are the counts the runtime's own
//!   documentation says were probed on this chip and on an M4;
//! - the **tile sizes** include one with a zero axis, which the runtime's documentation says was
//!   measured on an M3 Max and an M4, both alignment modes, as writing nothing. Slices of
//!   different lengths, which Metal's single `numRegions` cannot state, are never asked: the case
//!   hands one length.

use crate::texture_query::SizeAndAlign;
use crate::transcript::Value;
use metal2vulkan::metal::PixelFormat;

/// One `heapBufferSizeAndAlignWithLength:options:` query: a length, and Metal's own
/// `MTLResourceOptions` bits.
///
/// The options are the **raw bits** because the answer is claimed to be the same for all of them:
/// the case states Metal's own value rather than a vocabulary of modes that would have to grow
/// before a case could state a combination nobody modelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapBufferQuery {
    pub length: u64,
    pub options: u64,
}

impl HeapBufferQuery {
    /// The site this query's answer sits at.
    pub fn site(&self) -> String {
        format!("length-{}-options-{:#x}", self.length, self.options)
    }

    /// The words a case's input fingerprint covers.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.length.to_le_bytes().to_vec();
        out.extend(self.options.to_le_bytes());
        out
    }
}

/// A box of texels or of tiles, as both sides state and answer one: the origin's three axes then
/// the size's three.
///
/// A plain array and not either API's region type: the two name different types for the same six
/// numbers, and a case states the numbers.
pub type Box6 = [u64; 6];

/// One region conversion a case asks for: the regions, the tile size, and — for the pixels-to-tiles
/// direction — which alignment mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionQuery {
    /// What the case hands in: texel boxes for the pixels-to-tiles direction, tile boxes for the
    /// other.
    pub regions: &'static [Box6],
    /// The tile size, in texels.
    pub tile: [u64; 3],
    /// `MTLSparseTextureRegionAlignmentModeInward` rather than `Outward`. Read only by the
    /// pixels-to-tiles direction, which is the only call that takes a mode.
    pub inward: bool,
}

impl RegionQuery {
    /// The query with `regions` over a tile of `tile`, aligned outward.
    pub const fn new(regions: &'static [Box6], tile: [u64; 3]) -> Self {
        Self {
            regions,
            tile,
            inward: false,
        }
    }

    /// The query aligned inward.
    #[must_use]
    pub const fn inward(mut self) -> Self {
        self.inward = true;
        self
    }

    /// The site this query's answer sits at.
    pub fn site(&self, direction: &str) -> String {
        format!(
            "{direction}-tile-{}x{}x{}-{}",
            self.tile[0],
            self.tile[1],
            self.tile[2],
            if self.inward { "inward" } else { "outward" }
        )
    }

    /// The words a case's input fingerprint covers: every region, the tile and the mode.
    pub fn words(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for region in self.regions {
            for axis in region {
                out.extend(axis.to_le_bytes());
            }
        }
        for axis in self.tile {
            out.extend(axis.to_le_bytes());
        }
        out.push(u8::from(self.inward));
        out
    }
}

/// A box as a site records it.
pub fn box_value(held: &Box6) -> Value {
    Value::fields([
        (
            "origin",
            Value::fields([
                ("x", Value::Uint(held[0])),
                ("y", Value::Uint(held[1])),
                ("z", Value::Uint(held[2])),
            ]),
        ),
        (
            "size",
            Value::fields([
                ("width", Value::Uint(held[3])),
                ("height", Value::Uint(held[4])),
                ("depth", Value::Uint(held[5])),
            ]),
        ),
    ])
}

/// A list of boxes as a site records it.
pub fn boxes_value(held: &[Box6]) -> Value {
    Value::List(held.iter().map(box_value).collect())
}

/// A `size` and `align` pair as a site records it, in the shape
/// [`crate::texture_query`]'s heap query uses.
pub fn size_and_align(held: &SizeAndAlign) -> Value {
    Value::fields([
        ("size", Value::Uint(held.size)),
        ("align", Value::Uint(held.align)),
    ])
}

/// The site a pixel format's alignment answer sits at: the format's own Metal name, so a diff names
/// the format rather than an ordinal.
pub fn format_site(format: PixelFormat) -> String {
    format.name().to_owned()
}
