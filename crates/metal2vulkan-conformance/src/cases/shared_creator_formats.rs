//! What `newSharedTextureWithDescriptor:` makes, crossed with the **storage mode** it is asked
//! in: the four padded 10-bit formats, the one extended-range format it answers nil for, and two
//! plain formats as controls.
//!
//! The suite held the shared creator and these formats on two axes that never met.
//! [`crate::cases::texture_views`]' `texture-shared-handles` asks the shared creator for one
//! format, `RGBA8Unorm`, and [`crate::cases::textures`] round-trips 576 to 579 from the
//! **device's** creator. So no case crossed the creator with the format, and no case crossed
//! either with the storage mode at all.
//!
//! # The bug the missing cell would have caught
//!
//! A rule in the runtime claimed `newSharedTextureWithDescriptor:` answers nil for each of
//! `R10Unorm_X6` (576), `R10Unorm_X6_sRGB` (577), `RG10Unorm_X12` (578) and `RG10Unorm_X12_sRGB`
//! (579), where plain Metal makes a shareable texture of each. The probe behind it **ran**: it set
//! `storageMode = MTLStorageModeShared` and recorded the nil that came back, which is the storage
//! mode's nil and is answered for every format alike, while the comment beside it wrote the
//! storage down as `Private`. One attribute of one descriptor, recorded wrongly, held a nil over
//! four formats.
//!
//! So the cell this family adds is not "the shared creator with a padded 10-bit format". It is
//! **the shared creator with a padded 10-bit format at each of Metal's four storage modes**, with
//! the storage mode read back off the texture Metal made. A case that asked only at `Private`
//! would pass today and would still not catch the inverse mistake — a rule that answered nil at
//! `Private` and a texture at `Shared`.
//!
//! # The two cases
//!
//! - **`shared-creator-padded-format-storage`**: six formats by four storage modes, 24 cells. The
//!   four padded 10-bit formats, and `RGBA8Unorm` and `R16Uint` as the **controls**. On an Apple
//!   M3 Max under macOS 27.0.1, plain Metal, the four and the two controls answer alike at every
//!   mode — nil at `Shared`, `Managed` and `Memoryless`, a texture at `Private`, 8 of 8 per cell —
//!   and that is the whole finding: the creator's nil belongs to the mode and to no format.
//!   Without the two controls the 18 nils would read as "these formats are refused somewhere",
//!   which is the reading the removed rule was built on. The runtime's own side of it is
//!   `tests::texture_descriptor::the_shared_creator_makes_a_padded_ten_bit_texture`; this case is
//!   the half that holds it against Metal.
//! - **`shared-creator-extended-range-samples`**: `RGBA16Float_XR` (556) at 1, 4 and 8 samples in
//!   `Private` storage, which is the shared creator's **other** nil. It had the same storage-mode
//!   exposure as the four and was read at `Private` on the same chip and release, where it stands:
//!   Metal makes no texture of it alone, from this creator, at any of the three counts (nil 8 of 8
//!   at each). The controls are the same two formats at one sample, made in the same process, so
//!   the row's nil is the format's and not the case's.
//!
//! Together the two say which of the five formats is genuinely nil at `Private` and which four are
//! not, which is exactly the distinction the misread probe lost.
//!
//! # The two axes, by value
//!
//! The storage axis is swept **exhaustively**: `MTLStorageModeShared`, `MTLStorageModeManaged`,
//! `MTLStorageModePrivate` and `MTLStorageModeMemoryless` are every value `MTLStorageMode` has,
//! and each is asked of every format of the first table. The format axis is the five
//! `MTLPixelFormat` values the shared creator's two nils were ever claimed of —
//! `MTLPixelFormatR10Unorm_X6`, `MTLPixelFormatR10Unorm_X6_sRGB`, `MTLPixelFormatRG10Unorm_X12`,
//! `MTLPixelFormatRG10Unorm_X12_sRGB` and `MTLPixelFormatRGBA16Float_XR` — with
//! `MTLPixelFormatRGBA8Unorm` and `MTLPixelFormatR16Uint` beside them as controls. So the first
//! table is the **cross** of the two: 24 cells, and neither axis read alone would settle which of
//! them a nil belongs to.
//!
//! # What each cell records, and why the storage mode is read off the object
//!
//! - **`made`** — whether the creator answered a texture. Metal's nil and the runtime's typed
//!   `Nil` are the same fact and both record `false` (`made_or_nil`), as
//!   [`Side::makes_texture`] already collapses them for the device's creator. Anything else is our
//!   own refusal and is recorded as one, so a typed error cannot hide inside a `false`.
//! - **`resource`** — every `MTLResource` getter of the made texture, **`storageMode` among
//!   them**. This is the readback the removed rule's probe did not have: the mode the case asked
//!   in is stated in the site's name, derived from the same `HeapTextureQuery` field the input
//!   fingerprint covers ([`HeapTextureQuery::words`]), and the mode Metal gave the object is read
//!   back off the object. A recording whose site says `private` and whose `storage_mode` reads 0
//!   would be loud, where a comment saying `Private` beside a call that asked in `Shared` was
//!   silent.
//! - **`resource-after-handle`** — the same getters of the same object, read a second time once
//!   the handle has been made and a texture opened from it. Reading a getter twice on one object
//!   is only worth a site where something happened between the two readings, and here something
//!   did: whether `newSharedTextureHandle` and `newSharedTextureWithHandle:` leave the texture
//!   answering what it answered is Metal's to settle, and nothing in this tree had asked. The two
//!   sites compare independently, so a release where the second moves fails the case.
//! - **`texture`** — `pixelFormat`, `textureType`, `mipmapLevelCount`, `arrayLength` and
//!   `isShareable` of the made texture, so a cell says Metal made a shareable texture **of the
//!   format asked** rather than of some other.
//! - **`handle`** — `newSharedTextureHandle`, whether its `device` is the device that made the
//!   texture, and the texture `newSharedTextureWithHandle:` opens from it with its extent and
//!   level count. Asked only of a texture the creator answered, which is shareable by
//!   construction: what Metal does for one that is not is unmeasured, and asking would end its
//!   process.
//!
//! # The positive control, and why these tables need one
//!
//! Eighteen of the 24 cells of the first case record `made` false. A side that answered nil to
//! every shared-creator call would agree with all eighteen, so the tables are built so it cannot:
//!
//! - **six cells record `made` true**, every format's `Private` row, each with a texture's getters
//!   and a handle's under it. A side that refused the creator outright differs at six cells and at
//!   every leaf beneath them;
//! - **the two controls run at all four modes**, so "nil at this mode" and "nil for this format"
//!   are two different shapes in one recording rather than one claim;
//! - **this module's own tests**, with no device, hold that both tables name both controls, that
//!   every format of the first table is asked at all four modes, and that the `Private` rows exist
//!   — so a table somebody narrows by mistake fails here rather than passing by testing less.
//!
//! # Why every cell is safe
//!
//! A descriptor is CPU-side bookkeeping: nothing here compiles, dispatches, draws, blits or
//! commits, and the largest texture is 8 by 8 of one level. At most eight textures and eight
//! opened ones exist across both cases.
//!
//! The hazard the shared creator carries is a descriptor Metal's **validation** stops, which ends
//! its process rather than answering nil. Two guards:
//!
//! - [`HeapTextureQuery::forbidden`] decides the descriptor's own stopped shapes from its fields
//!   alone, and this module's tests hold every cell of both tables to it with no device;
//! - the **sample counts** are the rest, and they are why the second case holds one format and not
//!   three. Plain Metal stops a multisample texture of a count the device does not take
//!   ("sampleCount (8) is not supported by device" — 3, 8 and 16 on this chip) from every creator,
//!   **except** that the shared creator answers nil first for a format of which Metal makes no
//!   texture alone. `RGBA16Float_XR` is such a format and was measured nil 8 of 8 at 4 and at 8
//!   samples from this creator; `RGBA8Unorm` and `R16Uint` are not, so a multisample row of either
//!   is a shape nobody has measured from this creator and is left out. `samples_are_measured`
//!   holds that, with no device.

use crate::case::{Abort, Answer, Case, Failure, Handle, Input, MetalSource, Side};
use crate::heap_query::QueriedResource;
use crate::texture_query::{HeapTextureQuery, Storage};
use crate::transcript::{Node, Observed, Sites, Value};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};

/// The case names, as the report and the recordings spell them.
pub const PADDED_STORAGE: &str = "shared-creator-padded-format-storage";
pub const EXTENDED_RANGE_SAMPLES: &str = "shared-creator-extended-range-samples";

/// Every case of the family.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![
        Box::new(SharedCreatorCase::padded_storage()),
        Box::new(SharedCreatorCase::extended_range_samples()),
    ]
}

/// The four padded 10-bit formats, in their raw order: 576, 577, 578 and 579.
const PADDED: [PixelFormat; 4] = [
    PixelFormat::R10UnormX6,
    PixelFormat::R10UnormX6SRGB,
    PixelFormat::RG10UnormX12,
    PixelFormat::RG10UnormX12SRGB,
];

/// The controls, and what each one controls for.
///
/// `RGBA8Unorm` is the format the suite's other shared-creator case already uses, so a cell of it
/// here is comparable with one there; `R16Uint` is a single-channel integer format, as unlike a
/// padded 10-bit `Unorm` as the uncompressed colour formats get. Both are formats of which Metal
/// makes a texture from every creator, so their rows at a mode that answers nil can only be the
/// **mode's** nil — which is the whole attribution this family rests on.
const CONTROLS: [PixelFormat; 2] = [PixelFormat::RGBA8Unorm, PixelFormat::R16Uint];

/// The extended-range format the shared creator answers nil for at `Private`, of which Metal makes
/// no texture alone.
const EXTENDED_RANGE: PixelFormat = PixelFormat::RGBA16FloatXR;

/// Metal's four storage modes, which is the axis the removed rule's probe got wrong:
/// `MTLStorageModeShared`, `MTLStorageModeManaged`, `MTLStorageModePrivate` and
/// `MTLStorageModeMemoryless`, every value `MTLStorageMode` has.
///
/// All four, swept against every format of the storage table, because the whole finding is which
/// of the two axes a nil belongs to and a sweep of three could not say.
const MODES: [Storage; 4] = [
    Storage::Shared,
    Storage::Managed,
    Storage::Private,
    Storage::Memoryless,
];

/// The extent every descriptor names: the smallest shape the re-measurement used, so a cell holds
/// a texture of 64 texels at most.
const SIDE: u32 = 8;

/// The sample counts the extended-range case asks at: one, one the device takes, and one it does
/// not.
///
/// All three from the shared creator alone, and of one format. See the module's safety argument
/// and `samples_are_measured`.
const SAMPLES: [u32; 3] = [1, 4, 8];

/// The site each of a cell's four answers sits at.
mod site {
    /// Whether `newSharedTextureWithDescriptor:` answered a texture.
    pub const MADE: &str = "made";
    /// Every `MTLResource` getter of the made texture, `storageMode` among them, read before the
    /// handle is made.
    pub const RESOURCE: &str = "resource";
    /// The same getters of the same object, read again **after** `newSharedTextureHandle`.
    pub const RESOURCE_AGAIN: &str = "resource-after-handle";
    /// What the made texture answers of itself: its format, type, levels, slices and `shareable`.
    pub const TEXTURE: &str = "texture";
    /// `newSharedTextureHandle` and what it opens.
    pub const HANDLE: &str = "handle";
}

/// The variant name the **Metal** side records a nil answer under
/// ([`crate::case::Failure::named`] in its `shared_texture`).
const METAL_NIL: &str = "Nil";

/// The variant name **our** side records a typed `Error::Nil` under
/// ([`crate::case::Failure::of_rt`]).
const OUR_NIL: &str = "Error::Nil";

/// One case of the family: a table of descriptors, each asked of the shared creator.
///
/// Both cases are the same walk over a different table, because what they differ in is the axis
/// they sweep and not what they record of a cell.
pub struct SharedCreatorCase {
    name: &'static str,
    cells: Vec<HeapTextureQuery>,
}

impl SharedCreatorCase {
    /// The four padded 10-bit formats and the two controls, each at all four storage modes.
    fn padded_storage() -> Self {
        let mut cells = Vec::new();
        for format in PADDED.into_iter().chain(CONTROLS) {
            for storage in MODES {
                cells.push(descriptor(format, storage));
            }
        }
        Self {
            name: PADDED_STORAGE,
            cells,
        }
    }

    /// `RGBA16Float_XR` at each of [`SAMPLES`] in `Private`, and the two controls at one sample.
    fn extended_range_samples() -> Self {
        let mut cells = Vec::new();
        for samples in SAMPLES {
            cells.push(sampled(EXTENDED_RANGE, samples));
        }
        for format in CONTROLS {
            cells.push(sampled(format, 1));
        }
        Self {
            name: EXTENDED_RANGE_SAMPLES,
            cells,
        }
    }
}

impl Case for SharedCreatorCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut words = Vec::new();
        for cell in &self.cells {
            words.extend(cell.words());
        }
        Ok(vec![Input::bytes("descriptors", &words)])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for cell in &self.cells {
            let mut node = Node::branch(cell.site(), Vec::new());
            // Each cell's answer is a site of its own and a cell that did not answer is recorded
            // where it sits while the sweep goes on: one format our side refuses compares as a
            // refusal at that one cell, and the other 23 stay readable instead of being lost
            // behind an abort.
            let made = match made_or_nil(side.shared_texture(cell)) {
                Ok(made) => made,
                Err(failure) => {
                    sites.push(
                        node.with(Node::leaf(site::MADE, Observed::value(failure.recorded()))),
                    );
                    continue;
                }
            };
            node = node.with(Node::leaf(
                site::MADE,
                Observed::value(Value::ok(Value::Bool(made.is_some()))),
            ));
            if let Some(texture) = made {
                node = self.answers_of(side, texture, node);
            }
            sites.push(node);
        }
        Ok(())
    }
}

impl SharedCreatorCase {
    /// The three sites a texture the creator answered carries: its `MTLResource` getters, what it
    /// says of itself, and its handle.
    ///
    /// The handle is asked of this texture alone, which the creator answered and which is
    /// therefore shareable: what Metal does with `newSharedTextureHandle` on one that is not is
    /// unmeasured, and a case that asked and was wrong would end Metal's process.
    fn answers_of(&self, side: &mut dyn Side, texture: Handle, node: Node) -> Node {
        let resource = resource_facts(side, texture, site::RESOURCE);
        let own = match side.texture_view_facts(texture, None) {
            Ok(facts) => facts.node(site::TEXTURE),
            Err(failure) => Node::leaf(site::TEXTURE, Observed::value(failure.recorded())),
        };
        let handle = match side.shared_texture_handle(texture) {
            Ok(facts) => facts.node(site::HANDLE),
            Err(failure) => Node::leaf(site::HANDLE, Observed::value(failure.recorded())),
        };
        // The same getters of the same object again, after the handle: the one place a case can
        // read a getter twice, and the second reading is a fact rather than a mirror because
        // `newSharedTextureHandle` and the texture it opens sit between the two.
        let again = resource_facts(side, texture, site::RESOURCE_AGAIN);
        node.with(resource).with(own).with(handle).with(again)
    }
}

/// Every `MTLResource` getter of the texture `texture` names, at `at`.
fn resource_facts(side: &mut dyn Side, texture: Handle, at: &str) -> Node {
    match side.resource_facts(QueriedResource::Texture(texture), None) {
        Ok(facts) => facts.node(at.to_owned()),
        Err(failure) => Node::leaf(at.to_owned(), Observed::value(failure.recorded())),
    }
}

/// The descriptor a cell of the storage sweep names: 2D, `SIDE` by `SIDE`, one level, one sample,
/// read by shaders, of `storage`.
fn descriptor(format: PixelFormat, storage: Storage) -> HeapTextureQuery {
    HeapTextureQuery::texture_2d(format, [SIDE, SIDE], storage)
        .with_usage(TextureUsage::SHADER_READ)
}

/// The descriptor a cell of the sample sweep names: the same shape in `Private`, at `samples`
/// samples, of the texture type that count takes.
fn sampled(format: PixelFormat, samples: u32) -> HeapTextureQuery {
    let query = descriptor(format, Storage::Private).with_samples(samples);
    match samples {
        1 => query,
        _ => query.of_type(TextureType::Type2DMultisample, 1, 1),
    }
}

/// What the creator answered, with Metal's nil and our typed `Nil` read as the one fact they are.
///
/// `Ok(Some(_))` is a texture, `Ok(None)` is nil, and `Err` is a refusal of ours — a typed error
/// that is not `Nil`, or one the venue itself cannot answer. The collapse is
/// [`Side::makes_texture`]'s, which the device's creator already has for exactly this reason: the
/// two sides word a nil differently, and a case that compared the wording would read "both sides
/// answered nil" as a difference. What it must never do is read a refusal as a nil, so only the
/// `Nil` variants collapse, and a failure that names a gap of the venue's never does.
fn made_or_nil(answer: Answer<Handle>) -> Result<Option<Handle>, Failure> {
    match answer {
        Ok(texture) => Ok(Some(texture)),
        Err(failure) if is_nil(&failure) => Ok(None),
        Err(failure) => Err(failure),
    }
}

/// Whether `failure` is a side saying Metal answered nil, rather than refusing the call.
///
/// Read off the recorded value's variant, which is the shape the comparison itself reads a refusal
/// from, so this cannot drift from what a recording holds. A failure that names what the venue
/// lacks is never a nil: that is the venue's own gap and has its own verdict.
fn is_nil(failure: &Failure) -> bool {
    if failure.lacks.is_some() {
        return false;
    }
    match failure.recorded() {
        Value::Variant { payload, .. } => matches!(
            &*payload,
            Value::Variant { name, .. } if name == METAL_NIL || name == OUR_NIL
        ),
        _ => false,
    }
}

/// The guards this family's safety argument and its claim to measure anything rest on, all with no
/// device at all — which is what a guard against ending Metal's process has to do.
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Both cases, for a test that walks the family.
    fn family() -> Vec<SharedCreatorCase> {
        vec![
            SharedCreatorCase::padded_storage(),
            SharedCreatorCase::extended_range_samples(),
        ]
    }

    /// No cell of either table is a descriptor Metal's own validation stops, which would end its
    /// process in record mode.
    ///
    /// Verified to fail: giving [`SharedCreatorCase::padded_storage`]'s descriptor
    /// `.with_levels(5)` fails here naming every cell, because an 8 by 8 descriptor's full chain
    /// is four levels.
    #[test]
    fn no_cell_is_a_descriptor_metals_validation_stops() {
        let mut stopped = Vec::new();
        for case in family() {
            for cell in &case.cells {
                if let Some(why) = cell.forbidden() {
                    stopped.push(format!("{}: {}: {why}", case.name, cell.site()));
                }
            }
        }
        assert!(
            stopped.is_empty(),
            "these descriptors are ones Metal's validation stops, so record mode would end \
             Metal's process; take them out of the table:\n{}",
            stopped.join("\n")
        );
    }

    /// The only format either table asks at more than one sample is the one measured there, and it
    /// is the one the shared creator answers nil for before any sample-count rule.
    ///
    /// This is the second half of the safety argument and the one no descriptor field can carry:
    /// plain Metal **stops** a multisample texture of a count the device does not take, from every
    /// creator, for every format of which it makes a texture at all. The shared creator's
    /// nil-first rule for a format of which Metal makes no texture alone is what makes
    /// `RGBA16Float_XR` safe at 8 samples, and nothing makes `RGBA8Unorm` safe there.
    ///
    /// Verified to fail: adding `sampled(PixelFormat::RGBA8Unorm, 8)` to the sample table fails
    /// here naming the format and the count.
    #[test]
    fn samples_are_measured() {
        let mut unmeasured = Vec::new();
        for case in family() {
            for cell in &case.cells {
                if cell.samples > 1 && cell.format != EXTENDED_RANGE {
                    unmeasured.push(format!(
                        "{}: {} at {} samples",
                        case.name,
                        cell.format.name(),
                        cell.samples
                    ));
                }
            }
        }
        assert!(
            unmeasured.is_empty(),
            "a multisample descriptor of a format of which Metal makes a texture is handed to the \
             shared creator, and a count this device does not take ends Metal's process there; \
             only {}'s nil comes first:\n{}",
            EXTENDED_RANGE.name(),
            unmeasured.join("\n")
        );
    }

    /// Every format of the storage table is asked at all four of Metal's storage modes.
    ///
    /// Without this the table could narrow to the modes somebody last looked at, and the one axis
    /// the removed rule's probe got wrong would stop being swept — which is the whole reason this
    /// family exists.
    ///
    /// Verified to fail: dropping `Storage::Memoryless` from [`MODES`] fails here naming all six
    /// formats and the mode.
    #[test]
    fn every_format_of_the_storage_table_is_asked_at_every_storage_mode() {
        let case = SharedCreatorCase::padded_storage();
        let mut missing = Vec::new();
        for format in PADDED.into_iter().chain(CONTROLS) {
            for storage in [
                Storage::Shared,
                Storage::Managed,
                Storage::Private,
                Storage::Memoryless,
            ] {
                let asked = case
                    .cells
                    .iter()
                    .any(|cell| cell.format == format && cell.storage == storage);
                if !asked {
                    missing.push(format!("{} at {}", format.name(), storage.word()));
                }
            }
        }
        assert!(
            missing.is_empty(),
            "the storage mode is the axis this family exists to sweep, and these cells are not in \
             the table:\n{}",
            missing.join("\n")
        );
    }

    /// Both tables hold both controls, and the storage table holds each of them at every mode.
    ///
    /// A column without a control is not a measurement: 18 of the storage table's 24 cells record
    /// a nil, and what says those nils belong to the **mode** rather than to the four formats is
    /// that two formats Metal makes a texture of answer the same way at the same modes.
    ///
    /// The two controls are named here **literally** rather than read out of [`CONTROLS`]: a test
    /// that swept whatever that list held would pass when the list shrank, which is the one way
    /// this guard could be removed without anything turning red.
    ///
    /// Verified to fail: taking `PixelFormat::R16Uint` out of [`CONTROLS`] fails here naming both
    /// cases and the format.
    #[test]
    fn both_tables_name_both_controls() {
        for case in family() {
            let formats: BTreeSet<u32> =
                case.cells.iter().map(|cell| cell.format.as_raw()).collect();
            for control in [PixelFormat::RGBA8Unorm, PixelFormat::R16Uint] {
                assert!(
                    formats.contains(&control.as_raw()),
                    "{}: no cell of {}, so the cells that record a nil have nothing to be read \
                     against and the nil could be read as the format's",
                    case.name,
                    control.name()
                );
            }
        }
    }

    /// Every format of both tables is asked at `Private`, which is the one mode where the two
    /// cases' answers differ from each other.
    ///
    /// This is the positive control: six of the storage table's cells and three of the sample
    /// table's record a texture, with its getters and its handle under them, so a side that
    /// refused the shared creator outright differs at nine cells rather than agreeing with a table
    /// of nils.
    ///
    /// Verified to fail: taking `Storage::Private` out of [`MODES`] fails here naming all six
    /// formats of the storage table.
    #[test]
    fn every_format_is_asked_in_private_storage() {
        let mut missing = Vec::new();
        for case in family() {
            let formats: BTreeSet<u32> =
                case.cells.iter().map(|cell| cell.format.as_raw()).collect();
            for format in formats {
                let asked = case
                    .cells
                    .iter()
                    .any(|cell| cell.format.as_raw() == format && cell.storage == Storage::Private);
                if !asked {
                    missing.push(format!("{}: format {format}", case.name));
                }
            }
        }
        assert!(
            missing.is_empty(),
            "`Private` is the mode the creator makes a texture in, so a table without it records \
             nothing but nils and a side that refused the call would pass it:\n{}",
            missing.join("\n")
        );
    }

    /// No two cells of one table share a site, so every cell's answer has a name of its own.
    ///
    /// Verified to fail: asking the sample table's three `RGBA16Float_XR` cells at one sample each
    /// fails here, because the three then sit at one site.
    #[test]
    fn no_two_cells_of_a_table_share_a_site() {
        for case in family() {
            let mut seen = BTreeSet::new();
            for cell in &case.cells {
                assert!(
                    seen.insert(cell.site()),
                    "{}: two cells sit at {}, so one of the two answers is compared against \
                     nothing",
                    case.name,
                    cell.site()
                );
            }
        }
    }

    /// A cell's site names the storage mode it asked in, and the four modes give four names.
    ///
    /// The misread probe's whole failure was that the storage mode it asked in and the storage
    /// mode written beside the answer were different. A site derived from the descriptor's own
    /// field cannot be, and the input fingerprint covers the same field
    /// ([`HeapTextureQuery::words`]), so a recording's row says which mode it was taken at and a
    /// change of mode makes the case unrecorded rather than silently re-pointing the answer.
    #[test]
    fn a_cells_site_and_fingerprint_both_name_its_storage_mode() {
        let mut sites = BTreeSet::new();
        let mut words = BTreeSet::new();
        for storage in MODES {
            let cell = descriptor(PixelFormat::R10UnormX6, storage);
            assert!(
                cell.site().contains(storage.word()),
                "{}: the site does not name the storage mode",
                cell.site()
            );
            assert!(sites.insert(cell.site()), "two modes share one site name");
            assert!(
                words.insert(cell.words()),
                "two modes fingerprint alike, so a recording taken at one would compare against a \
                 run at the other"
            );
        }
        assert_eq!(sites.len(), MODES.len());
        assert_eq!(words.len(), MODES.len());
    }

    /// Both cases state their inputs, and the two fingerprints differ.
    #[test]
    fn both_cases_state_their_own_inputs() {
        let mut digests: Vec<crate::transcript::Digest> = Vec::new();
        for case in all() {
            let inputs = case.inputs().expect("a table states its own inputs");
            assert_eq!(inputs.len(), 1, "{}: one input", case.name());
            assert!(
                !digests.contains(&inputs[0].digest),
                "{}: two cases fingerprint alike",
                case.name()
            );
            digests.push(inputs[0].digest.clone());
            assert!(matches!(case.metal_source(), MetalSource::Run));
        }
    }
}
