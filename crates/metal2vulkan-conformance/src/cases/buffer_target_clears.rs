//! A render pass that only clears a target lying over a buffer: what it wrote, and what it left in
//! the bytes between the rows.
//!
//! A render target over a buffer (`newTextureWithDescriptor:offset:bytesPerRow:` on an
//! `MTLBuffer`, [`crate::render::TargetSpec::over_buffer`]) has its texels in memory the host can
//! read whole. So the family's question is one a texture-shaped read cannot ask: a pass clears the
//! target, and **the buffer comes back whole** — the bytes before it, the bytes between its rows,
//! and the bytes after it, beside the texels themselves. A `getBytes:` of the target would walk
//! only the texels and pass over the padding in silence, which is where the answer lives.
//!
//! **Pre-written, because the answer is about bytes nothing wrote.** The buffer holds a pattern of
//! this family's own before the pass (`pattern`), so an untouched byte is a byte this program put
//! there rather than whatever the allocator left. Without that the `DontCare` rows would be
//! unfalsifiable: a store that wrongly wrote zeros and a store that wrote nothing read alike
//! against a buffer nobody filled.
//!
//! Six cases, because the family has two axes and each crossing is its own question: three row
//! pitches — 32, 256 and 16 bytes behind a row of 16 — each with `MTLStoreActionStore` and with
//! `MTLStoreActionDontCare`. Nothing else varies: one 4 x 4 `RGBA8Unorm` target at one offset in
//! one buffer, one `Clear` load of one colour, and a pass with **no pipeline, no draw and no
//! shader fixture at all**, which is what makes these the rows that need no fixture while still
//! exercising the whole widening end to end.
//!
//! # The three pitches, and why each is a different question
//!
//! - **32**, twice the row: every row of texels is followed by 16 bytes no texel covers, inside
//!   the target's own span. This is where a store that wrote a whole pitch rather than a row shows.
//! - **256**, sixteen times the row: the rows lie far apart, so the target spans 784 bytes of which
//!   64 are texels. A store that cleared its image's whole extent would reach hundreds of bytes
//!   here.
//! - **16**, exactly one row: the rows are contiguous and there is no padding between them at all.
//!   The question is then whether the pass stays inside the target's 64 bytes.
//!
//! # The pitch-16 rows went through our runtime before they went to Metal
//!
//! This order is deliberate, and the next author should keep it. The guard that keeps this suite
//! off Metal's fatal shapes ([`crate::render::OverBuffer::unsafe_stride`]) refuses a pitch **below**
//! the row, because that one fails AGX's assertion `bytes_per_row >= used_bytes_per_row` and ends
//! Metal's process. A pitch **equal to** the row passes that guard — and passing the guard is not
//! the same as being safe to ask. While the buffer-backed texture family was being written, the
//! runtime answered `Error::Invalid` for an `R8Unorm` 8 x 8 texture at a pitch of exactly its row
//! of 8, and by the runtime's own documented model an `Error::Invalid` means *Metal stops this
//! call* (as against `Error::Nil`, *plain Metal answers nil and the process goes on*). Handing such
//! a shape to the oracle risks exactly what the sub-row pitch does.
//!
//! So the table went through our own runtime first, where an `Error::Invalid` is a recorded value
//! and nothing worse, and only a table with no `Invalid` anywhere was handed to Metal.
//! The module's own device test (`no_row_of_the_table_is_one_metal_stops`) is that check, kept
//! so it runs again whenever anybody widens the table. It carries a control shape in the same run,
//! a pitch of 24, so that it is a check that can fail rather than a walk that always passes.
//!
//! **What it answered**, measured on this M3 Max through MoltenVK before any Metal leg ran: the
//! runtime makes all three targets, and no row of the table is one it answers `Invalid` for, so the
//! pitch-16 rows reached Metal only after our own side had taken them.
//!
//! **And it corrects the reason, which is the more useful half.** "A pitch equal to the row is the
//! refused shape" was the reading that made this ordering feel necessary, and it is **wrong**. The
//! rule the `R8Unorm` row broke is that a 2D texture over a buffer takes a `bytesPerRow` that is a
//! **multiple of 16** (`metal2vulkan-runtime`'s `linear_plan`, `LINEAR_ALIGNMENT`), which is the
//! alignment Metal's own `minimumLinearTextureAlignmentForPixelFormat:` answers for every format a
//! buffer can back. 8 is not a multiple of 16; that row's pitch being its row as well was a
//! coincidence of an 8 x 8 `R8Unorm` texture. 16, 32 and 256 all satisfy the rule, and a pitch
//! equal to the row is nothing the runtime refuses on its own.
//!
//! The ordering stands all the same, for the reason it always had: the runtime's model is the only
//! thing that tells a shape Metal *stops* from one it answers nil for, so our side is where a new
//! pitch gets asked first. What changes is that the rule is now checkable rather than folklore —
//! the control row is a pitch of **24**, which is at or above the row and so passes the sub-row
//! guard, and breaks the multiple-of-16 rule, and the device test asserts the runtime refuses it
//! `Invalid`. If that rule ever moves, that assertion is what says so.
//!
//! # What a clearing pass did to the bytes between the rows
//!
//! Measured on this M3 Max (macOS 27.0.1), recorded from plain Metal over three runs a case, every
//! byte of the 2,048-byte buffer compared against the fill it held before the pass. **A clearing
//! pass writes the texels and nothing else.**
//!
//! - **Stored, at every pitch: exactly 64 of the 2,048 bytes moved** — the 16 texels, four bytes
//!   each — and they are the only bytes that moved. Where they lie follows the pitch exactly: at a
//!   pitch of 32, four 16-byte runs at bytes 256, 288, 320 and 352, so the **16 bytes after each
//!   row kept the fill**; at a pitch of 256, four 16-byte runs at 256, 512, 768 and 1,024, so the
//!   **240 bytes after each row kept it**; at a pitch of 16, one contiguous run of 64 bytes,
//!   256 to 319. The 256 bytes before the target and every byte past it kept the fill in all three.
//!   Each cleared texel holds `20 60 9F DF`, the four clear channels as `RGBA8Unorm` codes 32, 96,
//!   159 and 223.
//! - **`DontCare`, at every pitch: not one of the 2,048 bytes moved** — the texels included. The
//!   clear value never reaches the buffer at all. That is the answer
//!   `metal2vulkan-runtime`'s own `kept_bytes` states from its own `DontCare`-store probe, and it
//!   is now a recorded Metal fact of the suite's rather than a module's prose.
//!
//! So a clearing pass treats the inter-row bytes exactly as a blit does, which is what the landed
//! [`super::buffer_texture_blits`] family measured: the padding keeps its original fill throughout.
//! The clear adds one fact a blit cannot give — a store action that keeps nothing leaves the
//! **texels** at their original fill too, so at a pitch of 16, where there is no padding to speak
//! of, the `DontCare` row is the only thing distinguishing a pass that wrote nothing from one that
//! wrote the clear.
//!
//! # The two venues run different code of ours, and neither sees both halves
//!
//! Metal's answer above is one answer. **Our own side reaches it two different ways**, and which
//! way depends on the device, so the second venue here is not breadth over the first — it is cover
//! for a path the first never takes.
//!
//! `metal2vulkan-runtime` renders a target whose rows the device's own linear image cannot lay out
//! as Metal does into a **shadow** image and copies back (`shadowed-texture`). Where the image
//! really does lie over the buffer's bytes and the pass stores nothing of it, it renders into a
//! **kept image** instead so the buffer keeps its bytes (`kept_bytes`, `dont-care-stand-in`). The
//! two are exclusive: `kept_bytes` skips an attachment that already has a shadow, and it is reached
//! only by an attachment the pass does not store.
//!
//! Measured on both:
//!
//! - **MoltenVK on this M3 Max**: all six rows shadow, so all six announce `shadowed-texture` and
//!   **no row reaches the kept-image path at all**.
//! - **lavapipe**: the tight pitch of 16 aliases the buffer rather than shadowing, so
//!   `shadowed-texture` fires for five of the six and not for `pitch-16-stored`, and
//!   `pitch-16-dont-care` takes the kept-image path and announces `dont-care-stand-in` — a code
//!   MoltenVK never announces for any row of this family.
//!
//! So a one-venue run of this family leaves one of our own two implementations of the answer
//! unexercised, whichever venue it is. That is the reason to run both, and it is why the
//! declarations name two codes where no single run announces more than one of them.
//!
//! Nothing here faults a GPU, hangs one or trips a watchdog: one 4 x 4 target a case, one clear and
//! no draw, over two kilobytes of shared memory, committed once and read back.

use crate::case::{call, made, Abort, Case, Handle, Input, MetalSource, Side};
use crate::render::{OverBuffer, PassColor, RenderPass, RenderWork, TargetSpec};
use crate::transcript::{ByteLayout, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::StoreAction;

/// The buffer's bytes. Two kilobytes: enough for the widest pitch's rows, bytes before the target
/// and a long tail after it that no pass should reach, and an exact multiple of all three pitches,
/// so the read-back's rows divide the buffer evenly whichever pitch a case states.
const LEN: u64 = 2048;

/// Where the target's first row's first byte lies in the buffer.
///
/// 256 is a multiple of 16, which a 2D texture over a buffer needs, and a multiple of all three
/// pitches, so the target's first byte starts one of the read-back's rows in every case and a
/// difference names the row and the byte inside it.
const OFFSET: u64 = 256;

/// The target's extent in texels.
const SIDE: u32 = 4;

/// One row of the target in bytes, **stated** rather than derived from the pixel format, for the
/// reason the buffer-backed texture cases state theirs: `MTLTexture` answers no bytes-per-texel and
/// a format table of ours would be a second spelling of Metal's to keep in step. It is what
/// [`OverBuffer::unsafe_stride`] holds a case's pitch against.
const ROW_BYTES: u64 = 16;

/// The format every target here is of: four `RGBA8Unorm` texels to a 16-byte row.
const FORMAT: PixelFormat = PixelFormat::RGBA8Unorm;

/// The clear colour, as `MTLRenderPassColorAttachmentDescriptor` takes it: four doubles.
///
/// Four exact binary fractions, none of them a tie. `MTLClearColor` is a double and the store
/// rounds it to a code, so a channel at half a code would put a rounding rule in the middle of a
/// case about byte placement: 0.5 x 255 is 127.5, which two honest implementations may round two
/// ways. These four land at 31.875, 95.625, 159.375 and 223.125, where round-to-nearest is
/// unambiguous, and the four codes differ from each other so a channel in the wrong place is
/// visible.
const CLEAR: [f64; 4] = [0.125, 0.375, 0.625, 0.875];

/// The site the whole buffer's bytes after the pass sit at.
///
/// Its own `const` because the one thing it must not be is the name of another site of the same
/// case: see where it is pushed.
const BUFFER_SITE: &str = "buffer-after-the-pass";

/// The buffer's bytes before anything runs: byte `i` holds `1 + 11i mod 251`.
///
/// **Never zero**, deliberately. A `DontCare` store's answer is that the buffer keeps its bytes, and
/// against a buffer of zeros a store that wrote zeros and a store that wrote nothing would read
/// alike. A pattern with no zero byte in it makes that distinction, and 251 being prime keeps the
/// pattern from repeating along any of the three pitches.
fn pattern(n: usize) -> Vec<u8> {
    (0..n).map(|i| (1 + (i * 11) % 251) as u8).collect()
}

/// One case: a pitch, a store action, and the name both of them give it.
struct Row {
    /// The case name, as the report and the recording spell it. A literal rather than something
    /// built from the fields, so a recording's file name is a string a reader can grep for here.
    case: &'static str,
    /// `bytesPerRow`: how far apart the target's rows lie in the buffer.
    pitch: u64,
    store: StoreAction,
    /// Whether a venue has been **measured** announcing the kept-image notice
    /// (`dont-care-stand-in`) for this row, which is what the row must declare.
    ///
    /// Measured rather than reasoned, and narrow on purpose. The path is reachable only by a row
    /// the pass does not store, so a `Store` row can never need it -- but of the three `DontCare`
    /// rows only the tight pitch has been seen to take it, on lavapipe, where its target aliases
    /// the buffer instead of being shadowed. On MoltenVK no row of this family takes it, and on
    /// lavapipe the two padded pitches are shadowed and do not either.
    ///
    /// So the padded rows declare nothing here. A declaration no run of the suite can check is the
    /// noise the unfired tally exists to show, and the failure direction is the better one: a
    /// device that does announce it for a padded row fails loudly as an undeclared notice, and
    /// whoever sees that adds the declaration with the measurement that earned it.
    kept_image: bool,
}

/// The six rows: three pitches, each stored and each thrown away.
const ROWS: [Row; 6] = [
    Row {
        case: "buffer-target-clears-pitch-32-stored",
        pitch: 32,
        store: StoreAction::Store,
        kept_image: false,
    },
    Row {
        case: "buffer-target-clears-pitch-32-dont-care",
        pitch: 32,
        store: StoreAction::DontCare,
        kept_image: false,
    },
    Row {
        case: "buffer-target-clears-pitch-256-stored",
        pitch: 256,
        store: StoreAction::Store,
        kept_image: false,
    },
    Row {
        case: "buffer-target-clears-pitch-256-dont-care",
        pitch: 256,
        store: StoreAction::DontCare,
        kept_image: false,
    },
    Row {
        case: "buffer-target-clears-pitch-16-stored",
        pitch: 16,
        store: StoreAction::Store,
        kept_image: false,
    },
    Row {
        case: "buffer-target-clears-pitch-16-dont-care",
        pitch: 16,
        store: StoreAction::DontCare,
        kept_image: true,
    },
];

/// Every case of the family.
pub fn all() -> Vec<Box<dyn Case>> {
    ROWS.iter()
        .map(|row| Box::new(BufferTargetClear { row }) as _)
        .collect()
}

/// A case: one pass that clears one target over a buffer, at one pitch and one store action.
pub struct BufferTargetClear {
    row: &'static Row,
}

impl Row {
    /// Where the target's texels lie in the buffer, over `buffer`.
    const fn backing(&self, buffer: Handle) -> OverBuffer {
        OverBuffer::new(buffer, OFFSET, self.pitch, ROW_BYTES)
    }

    /// The target: `MTLTextureUsageRenderTarget` alone, which is what a pass that only clears needs
    /// and what the tile-store and visibility families make theirs with. Its storage is the
    /// buffer's, which [`TargetSpec::over_buffer`] states for it.
    fn target(&self, buffer: Handle) -> TargetSpec {
        TargetSpec::plain(FORMAT, [SIDE, SIDE]).over_buffer(self.backing(buffer))
    }

    /// The pass: one color attachment, cleared, stored as the row says, and **no step at all**.
    ///
    /// The empty step list is the case rather than an oversight: a pass with no draw call is one of
    /// the facts Metal's clear route reads, and it is what lets this family be asked without a
    /// vertex-and-fragment pair.
    fn pass(&self, target: Handle) -> RenderPass {
        RenderPass::clearing(vec![PassColor::cleared(target, CLEAR).stored(self.store)])
    }

    /// The whole buffer as a site's layout, divided into rows of the case's own pitch.
    ///
    /// **Every byte is an element**, so the bytes the target does not cover compare like any other:
    /// the padding after each row at a pitch of 32 or 256, the 256 bytes before the target, and the
    /// tail after it. That is the whole point of reading the buffer rather than the target — a byte
    /// the pass wrote outside the texels is visible here and nowhere else.
    fn layout(&self) -> ByteLayout {
        let pitch = u32::try_from(self.pitch).expect("a pitch in a u32");
        ByteLayout {
            region: Region::Buffer {
                offset: 0,
                length: LEN,
            },
            element: Element::U8,
            row_offset: 0,
            row_elements: pitch,
            row_pitch: pitch,
            rows: u32::try_from(LEN / self.pitch).expect("the rows in a u32"),
            image_pitch: u32::try_from(LEN).expect("the length in a u32"),
            images: 1,
            // A row's elements are its whole pitch here: this view divides the buffer up, it does
            // not lay the target over it, so nothing lies between these rows.
            gaps: Gaps::Ignored,
        }
    }
}

impl Case for BufferTargetClear {
    fn name(&self) -> &'static str {
        self.row.case
    }

    fn metal_source(&self) -> MetalSource {
        // Every call is public: `newBufferWithBytes:length:options:`,
        // `newTextureWithDescriptor:offset:bytesPerRow:`, `renderCommandEncoderWithDescriptor:`
        // and the buffer's own `contents`. The one shape of this call that ends Metal's process is
        // a pitch below the target's row, which `OverBuffer::unsafe_stride` refuses before any
        // object is made and no row of this table asks for.
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<crate::allowance::Allowance> {
        // Not a byte of the buffer is loosened: the texels and the padding compare exactly,
        // Metal's against ours. What differs is the **announcing**. Our runtime keeps a target
        // whose rows the device's own linear image does not lay out as Metal does as a shadow
        // image the pass renders into and the encoders copy back, and it announces that once per
        // device (`shadowed-texture`); Metal announces nothing, ever, because nothing of the kind
        // is a thing Metal has to do.
        //
        // **One declaration per code the case announces, measured on both venues.** A code a case
        // announces and declares nowhere is a divergence at the notices site
        // (`DivergenceKind::UndeclaredNotice`), so this list is a measured set and not a guess.
        // On this M3 Max through MoltenVK all six rows announce `shadowed-texture` and nothing
        // else. On lavapipe the tight pitch of 16 announces it for no row at all and the
        // `DontCare` one announces `dont-care-stand-in` instead, so the union is two codes and
        // neither venue sees both.
        //
        // `dont-care-stand-in` is declared on the one row a venue has been **measured**
        // announcing it for (`Row::kept_image`), not on every row that could in principle take the
        // path. The first shape of this declared it on all three `DontCare` rows, and the M3 Max
        // replay reported three declarations fired on no venue of which three "name nothing this
        // run justifies either, so no run of this suite has checked them" -- two of them true of
        // lavapipe as well, since the padded pitches are shadowed there too.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for all 6 rows of the family and each
        // was divergent on that field and nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        let mut named = vec![
            crate::replay::shared_memory_notice(),
            crate::replay::notice_allowance(
                metal2vulkan_runtime::Degradation::ShadowedTexture.code(),
            ),
        ];
        if self.row.kept_image {
            named.push(crate::replay::notice_allowance(
                metal2vulkan_runtime::Degradation::DontCareStandIn.code(),
            ));
        }
        named
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // The bytes the buffer starts with, and the whole of what the case states about the target
        // and the pass: the call's three numbers and the stated row, the attachment's load and
        // store actions and the clear colour's bits. A handle's number is a side's own and is not
        // among them, so a placeholder stands in for the buffer here.
        let placeholder = Handle(0);
        Ok(vec![
            Input::bytes("buffer-bytes", &pattern(LEN as usize)),
            Input::bytes("target", &self.row.target(placeholder).words()),
            Input::bytes("pass", &self.row.pass(placeholder).words()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let buffer = made(sites, "buffer", side.buffer(&pattern(LEN as usize)))?;
        let target = made(sites, "target", side.target(&self.row.target(buffer)))?;
        let work = vec![RenderWork::Pass(self.row.pass(target))];
        let outcome = call(sites, "render", side.render(&work), |_| Value::Absent)?;
        super::extended_range_targets::program(sites, "program", &outcome, &work);
        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(buffer, 0, LEN as usize),
            |_| Value::Absent,
        )?;
        // `buffer-after-the-pass`, and **not** `buffer`, which the call that made it already
        // holds. Two sites of one name at one path is a site the comparison cannot settle: the
        // first shape of this case named both of them `buffer`, and the whole-case replay paired
        // Metal's bytes against our `Ok` of the make and reported six cases divergent at
        // `case/buffer` without once comparing a byte of the buffer.
        sites.push(Node::leaf(
            BUFFER_SITE,
            Observed::bytes(bytes, self.row.layout()),
        ));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The code an `RGBA8Unorm` store rounds `channel` to, by the rule both sides state.
    ///
    /// It is here to check that the **pattern** cannot be mistaken for a cleared texel, and for
    /// nothing else. No test compares it with a recording: what the clear stored is Metal's answer,
    /// and a case that stated its own would be exercising our arithmetic on both sides of the
    /// equals sign.
    fn code(channel: f64) -> u8 {
        (channel.clamp(0.0, 1.0) * 255.0).round() as u8
    }

    /// The cleared texel's four bytes.
    fn cleared() -> [u8; 4] {
        [
            code(CLEAR[0]),
            code(CLEAR[1]),
            code(CLEAR[2]),
            code(CLEAR[3]),
        ]
    }

    /// Every case has a name of its own, so no case's answers are read as another's.
    #[test]
    fn every_case_has_a_name_of_its_own() {
        let mut seen = std::collections::BTreeSet::new();
        for row in &ROWS {
            assert!(seen.insert(row.case), "two cases are named {}", row.case);
        }
        assert_eq!(seen.len(), ROWS.len());
    }

    /// Both axes are crossed: each pitch is asked with each store action, and no pair is repeated.
    ///
    /// Without this a pitch could quietly lose one of its two store actions and the family would
    /// still look complete, which is the shape the `DontCare` answer is least able to survive: the
    /// stored rows alone cannot say whether the clear value reaches the buffer.
    #[test]
    fn every_pitch_is_asked_with_both_store_actions() {
        let mut pairs: Vec<(u64, u64)> = ROWS
            .iter()
            .map(|row| (row.pitch, row.store.raw()))
            .collect();
        pairs.sort();
        let total = pairs.len();
        pairs.dedup();
        assert_eq!(pairs.len(), total, "two rows ask the same pitch and store");
        for pitch in [32u64, 256, 16] {
            for store in [StoreAction::Store, StoreAction::DontCare] {
                assert!(
                    pairs.contains(&(pitch, store.raw())),
                    "no row asks a pitch of {pitch} stored {store:?}"
                );
            }
        }
    }

    /// No row of the table asks for the one shape that ends Metal's process, and the guard that
    /// decides it still refuses the shape below the row.
    ///
    /// Its failure mode was exercised while writing it: a row at a pitch of 8 — below the target's
    /// row of 16 — fails the first half, naming the case and the guard's words.
    #[test]
    fn no_row_would_end_metals_process() {
        let buffer = Handle(0);
        for row in &ROWS {
            assert_eq!(
                row.backing(buffer).unsafe_stride(),
                None,
                "{}: its pitch is one that ends Metal's process",
                row.case
            );
        }
        let below = OverBuffer::new(buffer, OFFSET, ROW_BYTES - 1, ROW_BYTES);
        assert!(
            below.unsafe_stride().is_some(),
            "the guard no longer refuses a pitch below the row, so a row of this table could \
             reach Metal's assertion"
        );
    }

    /// Every row's target lies inside the buffer, and its first byte starts one of the read-back's
    /// rows.
    ///
    /// A target running past the buffer is a call Metal refuses, so such a row would record a
    /// refusal rather than where a clear puts its bytes. Its failure mode was exercised while
    /// writing it: a pitch of 1024 fails here, naming the byte it reaches.
    #[test]
    fn every_targets_texels_lie_inside_the_buffer_at_a_row_of_its_own() {
        for row in &ROWS {
            let reaches = OFFSET + (u64::from(SIDE) - 1) * row.pitch + ROW_BYTES;
            assert!(
                reaches <= LEN,
                "{}: its target reaches byte {reaches} of a buffer of {LEN}",
                row.case
            );
            assert_eq!(
                OFFSET % row.pitch,
                0,
                "{}: its target starts inside one of the read-back's rows rather than at the \
                 start of one",
                row.case
            );
            assert_eq!(
                LEN % row.pitch,
                0,
                "{}: the buffer is not a whole number of its rows, so the read-back's layout \
                 would not cover every byte",
                row.case
            );
            assert_eq!(
                row.layout().extent(),
                LEN as usize,
                "{}: its layout and the bytes it is laid over are different sizes",
                row.case
            );
        }
    }

    /// The buffer's pattern holds no zero byte and never holds the cleared texel's bytes where a
    /// texel belongs, in any of the three pitches.
    ///
    /// Both halves are what make the family's answers falsifiable. A zero in the pattern would let
    /// a store that wrote zeros read as a store that wrote nothing; a pattern that happened to hold
    /// the clear colour at a texel's place would let a `DontCare` that stored the clear read as one
    /// that kept the buffer's bytes. Its failure mode was exercised while writing it: a pattern of
    /// `i as u8` fails the first half at byte 0.
    #[test]
    fn the_pattern_is_not_mistakable_for_a_clear() {
        let bytes = pattern(LEN as usize);
        assert!(
            !bytes.contains(&0),
            "the pattern holds a zero byte, so a store that wrote zeros would read as one that \
             wrote nothing"
        );
        let texel = cleared();
        assert_eq!(
            std::collections::BTreeSet::from(texel).len(),
            4,
            "two channels of the clear colour round to the same code, so a channel in the wrong \
             place would be invisible"
        );
        for row in &ROWS {
            for y in 0..u64::from(SIDE) {
                for x in 0..u64::from(SIDE) {
                    let at = (OFFSET + y * row.pitch + x * 4) as usize;
                    // Past the buffer is a broken table, which the test above owns and names
                    // better than a slice index would.
                    let Some(found) = bytes.get(at..at + 4) else {
                        continue;
                    };
                    assert_ne!(
                        found, texel,
                        "{}: the pattern holds the cleared texel's bytes where texel {x},{y} \
                         belongs",
                        row.case
                    );
                }
            }
        }
    }
}

/// The safety guard this family rests on, run on a device: **the pitch-16 rows go through our
/// runtime before they go to Metal.**
///
/// Record mode hands every row of the table to Metal, and the runtime's own model is what tells a
/// shape Metal *stops* from one it answers nil for: `Error::Invalid` is "Metal stops this call",
/// `Error::Nil` is "plain Metal answers nil and the process goes on". So a row that makes the
/// runtime answer `Invalid` is a row that must never be recorded, and this is the check that says
/// so before anybody runs `conformance record`.
///
/// A pitch equal to the target's own row is exactly the shape that needs it. The sub-row guard
/// ([`OverBuffer::unsafe_stride`]) lets it through, and while the buffer-backed texture family was
/// being written the runtime answered `Invalid` for an `R8Unorm` 8 x 8 texture at a pitch of its
/// own row of 8. That one broke the **multiple-of-16** rule a 2D texture over a buffer is held to,
/// not a pitch-equals-row rule, so 16 here is fine — but the way to know that is to ask our own
/// side, where an `Invalid` costs nothing, and not the oracle.
///
/// **It carries its own control**, because a sweep that only ever walks shapes the runtime answers
/// is a sweep that would pass with the check removed. A pitch of 24 is at or above the target's row
/// of 16, so the sub-row guard lets it through, and it breaks the rule the `R8Unorm` row broke —
/// `bytesPerRow` must be a multiple of 16 — so the runtime must answer `Invalid` for it. The second
/// half asserts that, in the same run on the same device: where it holds, the first half is a check
/// that can fail rather than a walk that always passes.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod device_tests {
    use super::*;
    use crate::device::{self, Notices, ScratchCache, Venues};
    use crate::runtime_side::RuntimeSide;

    /// The venue the guard opens: a GPU where the tests are built for one, else a CPU ICD.
    const VENUE: Venues = if cfg!(m2v_gpu) {
        Venues::Gpu
    } else {
        Venues::Cpu
    };

    /// The variant a failure names, where it names one.
    fn variant(value: &Value) -> String {
        match value {
            Value::Variant { name, payload } if name == "Err" => match payload.as_ref() {
                Value::Variant { name, .. } => name.clone(),
                other => format!("{other:?}"),
            },
            other => format!("{other:?}"),
        }
    }

    #[test]
    fn no_row_of_the_table_is_one_metal_stops() {
        let mut notices = Notices::default();
        let cache = ScratchCache::new().unwrap();
        let runtime = device::open(VENUE, &notices, &cache).unwrap();
        let mut side = RuntimeSide::new(
            &runtime,
            notices.open("no_row_of_the_table_is_one_metal_stops"),
        );
        let mut stopped = Vec::new();
        for row in &ROWS {
            let buffer = match side.buffer(&pattern(LEN as usize)) {
                Ok(buffer) => buffer,
                Err(failure) => {
                    stopped.push(format!(
                        "{}: its buffer: {}",
                        row.case,
                        variant(&failure.recorded())
                    ));
                    continue;
                }
            };
            if let Err(failure) = side.target(&row.target(buffer)) {
                let named = variant(&failure.recorded());
                if named.contains("Invalid") {
                    stopped.push(format!("{}: {named}", row.case));
                }
            }
        }
        assert!(
            stopped.is_empty(),
            "these rows are ones the runtime says Metal stops, so record mode would end Metal's \
             process; take them out of the table:\n{}",
            stopped.join("\n")
        );
        // The control: a pitch of 24 passes the sub-row guard and breaks the 16-byte pitch rule,
        // so the runtime says Metal stops it. Without this the walk above would read the same with
        // the `Invalid` test deleted.
        let buffer = side
            .buffer(&pattern(LEN as usize))
            .expect("a buffer for the control shape");
        let control = TargetSpec::plain(FORMAT, [SIDE, SIDE])
            .over_buffer(OverBuffer::new(buffer, OFFSET, 24, ROW_BYTES));
        let answered = side
            .target(&control)
            .map(|_| ())
            .map_err(|failure| variant(&failure.recorded()));
        assert!(
            matches!(&answered, Err(named) if named.contains("Invalid")),
            "a target at a pitch of 24 is not one the runtime says Metal stops ({answered:?}), so \
             the sweep above is not a check: either the rule moved, in which case say so here, or \
             the test no longer reads a refusal"
        );
    }
}
