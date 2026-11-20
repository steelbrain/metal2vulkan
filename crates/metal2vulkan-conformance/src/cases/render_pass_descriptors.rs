//! What a render pass descriptor says about itself, swept over its option space.
//!
//! `MTLRenderPassDescriptor` was exercised at ten of its seventeen items and
//! `MTLRenderPassAttachmentDescriptor` at six of twelve, every one of them by a render family
//! **handing the descriptor to an encoder**. Nothing read one back, so the array length, the
//! imageblock sample length, the threadgroup memory length, the rate map, the sample-position
//! getter, the six index fields of every attachment and the colour-array setter were reached by
//! nothing at all — and the fields that *were* reached were reached at one value each.
//!
//! [`crate::pass_descriptors`] holds the vocabulary, the sweeps and the argument about what each
//! shape can catch. This module is the cases.
//!
//! # The cases, and the depth of each
//!
//! A ratio below is **values compared over values the type has**, on this family's oracle.
//!
//! - `pass-descriptor-defaults`: a fresh descriptor, read whole — all eight colour attachments,
//!   both aspects, the eight numbers, the rate map, the visibility buffer and the sample
//!   positions. The one shape that says what a program gets for free, and (by the measurement in
//!   [`crate::pass_descriptors`]) the one shape that cannot catch a crossed getter.
//! - `pass-descriptor-set`: every field of all ten attachments and all eight numbers stated, no
//!   two alike, read whole. This is the shape that catches a crossing: 60 index fields holding 60
//!   different numbers, ten different (load, store) pairs, three clear colours apart in all twelve
//!   channels, and the depth and stencil clears and filters away from their defaults.
//! - `pass-descriptor-assigned`: attachment descriptors built apart from the pass, stated, assigned
//!   into it and then mutated through the object that was assigned. What the pass answers says
//!   whether `setObject:atIndexedSubscript:`, `setDepthAttachment:` and `setStencilAttachment:`
//!   **copy** the descriptor or keep the caller's — the one way the array's setter can be compared
//!   at all, since the runtime's counterpart is an assignment into a value.
//! - `pass-descriptor-cleared`: the same, then every attachment set to nil, read whole: what a pass
//!   answers for an attachment a program has cleared.
//! - `pass-attachment-actions`, 18 rows: the load action, the store action and the store action
//!   options, **36 of 36 triples** — every `MTLLoadAction` against every `MTLStoreAction` against
//!   both options, exhaustive, because each row states three attachments at three places of the
//!   space ([`crate::pass_descriptors::action_rows`]). Load 3/3, store 6/6, options 2/2 of the
//!   values Metal defines — the mask's undefined bits are not swept, for the reason
//!   [`crate::pass_descriptors::Actions`] gives.
//! - `pass-attachment-filters`, 6 rows: 6 of 6 pairs. Depth filter 3/3, stencil filter 2/2.
//! - `pass-attachment-indices`, 8 rows: the six index fields at 0, 1, 2, 7, 8, the last 32-bit
//!   index, the first value past 32 bits and the last `NSUInteger`. A 64-bit space, so the ratio is
//!   the eight values named rather than a share, and the two above 32 bits are what a side holding
//!   an index in a `u32` fails.
//! - `pass-clear-colors`, 6 rows of four channels: 24 doubles including a NaN with a payload, both
//!   infinities, both zeros, a denormal and the largest and smallest finite doubles.
//! - `pass-clear-depths`, 13 rows: both ends of the documented `[0, 1]`, the middle, a value either
//!   side of each end, both zeros, a NaN of each sign, both infinities, a denormal and the double
//!   just below 1.
//! - `pass-clear-stencils`, 262 rows: **every one of the 256 bytes**, exhaustive, and six words
//!   above them. 256/256 of the byte space.
//! - `pass-numbers`, 36 rows: each of the pass's eight numbers at 0, 1, its documented limit and
//!   the value either side of it ([`crate::pass_descriptors::NUMBER_SWEEPS`] says which limit is
//!   which).
//! - `pass-sample-positions`, 4 rows: no positions at all, four exact sixteenths, two positions,
//!   and four coordinates that are not sixteenths. 2 of the sample counts an Apple GPU takes custom
//!   positions at, and the fourth row is what says whether the getter answers the value a program
//!   gave or the one the rasterizer would use.
//!
//! # What is swept and what is not, with the reason
//!
//! - **The textures are not.** An attachment's `texture` and `resolveTexture` are read as whether
//!   the descriptor holds one, and no shape states one: what Metal renders through a texture at a
//!   level and a slice is every other render family's subject, and Metal checks none of these
//!   fields until an encoder opens.
//! - **The rate map and the visibility buffer are not**, for the same reason from the other end:
//!   each is an object whose identity nothing here compares. Both getters are read.
//! - **Sample positions outside `[0, 1)` are not**, nor more positions than a sample count: Metal
//!   documents and asserts on both, and [`crate::render::forbidden_positions`] is the suite's one
//!   guard for it, which this family's Metal side calls before the setter.
//! - **`sampleBufferAttachments` is not**: counter sample buffers are a runtime gap
//!   (`RuntimeGap::CounterSampleBuffers`), so the coverage map does not hold the item and there is
//!   nothing of ours to compare.
//!
//! # What the oracle answered
//!
//! Measured on an Apple M3 Max under macOS 27.0.1 (26A434), which is every recording here, and
//! each of these was an open question in the tree before:
//!
//! - **The three attachment setters copy.** A colour attachment assigned into slot 3 with
//!   `setObject:atIndexedSubscript:`, and a depth and a stencil attachment assigned with their own
//!   setters, keep the values the object held **at assignment**: mutating that object afterwards
//!   does not reach the pass. The runtime's counterpart is an assignment into a value, which copies
//!   by construction, so the two agree — but that they agree is now a measurement rather than an
//!   assumption about `NSCopying`.
//! - **An attachment set to nil answers a fresh descriptor**, not nil: every field of all eight
//!   colour attachments and of both aspects is back at Metal's default, and the pass's own eight
//!   numbers and its sample positions are untouched. This is also what makes the shape safe at all,
//!   since `objc2-metal` 0.3.2 would panic on a nil here.
//! - **Nothing in the descriptor is normalized.** All 36 action triples, both resolve filters, all
//!   262 stencil clear words, all 13 clear depths (both NaNs, both infinities, a denormal and the
//!   values outside `[0, 1]` among them), all 24 clear-colour channels and all 36 pass numbers came
//!   back exactly as they were set, the values past each documented limit included. A descriptor
//!   stores; the checking happens when an encoder opens.
//! - **An index field is a full 64-bit `NSUInteger`.** `level`, `slice`, `depthPlane` and the three
//!   resolve indices keep `0xffff_ffff`, `0x1_0000_0000` and `u64::MAX`, so a side holding one in a
//!   `u32` answers a different number.
//! - **`getSamplePositions:count:` answers the positions verbatim**, including coordinates that are
//!   not sixteenths (0.1 comes back as 0.1): the quantization the rasterizer applies is not applied
//!   to the descriptor. A count of 0 answers how many the pass holds, and setting an empty array
//!   takes the custom positions back off (a count of 0 afterwards).
//!
//! # Exactness
//!
//! Every value is a number, a bool, an ordinal or a float by its bits, so the family declares no
//! tolerance and announces nothing. The clears and the sample positions are compared **by their
//! bits**, which is why a NaN's payload and a signed zero are part of the sweep rather than a
//! rounding question.
//!
//! # Why this family is safe
//!
//! No case makes a library, a texture, a pipeline, a command buffer or an encoder. Each makes
//! descriptor objects, reads them and drops them, so nothing can fault the device, trip its
//! watchdog or reach a queue. The two hazards in reach are named in
//! [`crate::pass_descriptors`]: the sample-position assert, which the guard stops, and
//! `objc2-metal` 0.3.2's non-null attachment accessors.

use crate::case::{Abort, Case, Input, MetalSource, Side};
use crate::pass_descriptors::{
    action_rows, clear_stencils, filter_rows, PassSpec, Reads, Shape, CLEAR_COLORS, CLEAR_DEPTHS,
    INDEX_VALUES, NUMBER_SWEEPS, POSITION_ROWS, STATED_NUMBERS, STATED_POSITIONS,
};
use crate::transcript::{Node, Observed, Sites};
use crate::values::{
    LoadAction, MultisampleDepthResolveFilter, MultisampleStencilResolveFilter, StoreAction,
};

/// The site a whole-descriptor shape's answers sit at.
const FACTS: &str = "facts";

// ---- The swept values' own names ---------------------------------------------------------------
//
// The action and filter sweeps state their values as Metal's **raw ordinals**, for the reason
// `crate::pass_descriptors::Actions` gives: the whole of what an action means is its ordinal, and a
// third vocabulary beside the two the sides already have would be a third thing to keep in step.
// That is right for the value a row feeds, and it leaves the row's **site name** saying
// `load-0-store-4`, which names no Metal value a reader or a census could recognise. These four
// functions close that: each is a match over the runtime's own enum — one of the two vocabularies,
// not a third — from the ordinal the row holds to the SDK enumerator the ordinal **is**, and the
// site is named after it. So a diff names the value Metal was asked for rather than a number, and
// the enumerator is in the compared text, which is the only place the option-space census can read
// it from. A value whose ordinal no sweep reaches has no name here to go unused: the sweeps are
// exhaustive over all four types, which `tests::every_swept_action_and_filter_is_named` holds.

/// The SDK enumerator of `MTLLoadAction`'s raw value `raw`.
fn load_action_name(raw: u64) -> Result<&'static str, String> {
    Ok(match LoadAction::from_raw(raw)? {
        LoadAction::DontCare => "MTLLoadActionDontCare",
        LoadAction::Load => "MTLLoadActionLoad",
        LoadAction::Clear => "MTLLoadActionClear",
    })
}

/// The SDK enumerator of `MTLStoreAction`'s raw value `raw`.
fn store_action_name(raw: u64) -> Result<&'static str, String> {
    Ok(match StoreAction::from_raw(raw)? {
        StoreAction::DontCare => "MTLStoreActionDontCare",
        StoreAction::Store => "MTLStoreActionStore",
        StoreAction::MultisampleResolve => "MTLStoreActionMultisampleResolve",
        StoreAction::StoreAndMultisampleResolve => "MTLStoreActionStoreAndMultisampleResolve",
        StoreAction::Unknown => "MTLStoreActionUnknown",
        StoreAction::CustomSampleDepthStore => "MTLStoreActionCustomSampleDepthStore",
    })
}

/// The SDK enumerator of `MTLMultisampleDepthResolveFilter`'s raw value `raw`.
fn depth_filter_name(raw: u64) -> Result<&'static str, String> {
    Ok(match MultisampleDepthResolveFilter::from_raw(raw)? {
        MultisampleDepthResolveFilter::Sample0 => "MTLMultisampleDepthResolveFilterSample0",
        MultisampleDepthResolveFilter::Min => "MTLMultisampleDepthResolveFilterMin",
        MultisampleDepthResolveFilter::Max => "MTLMultisampleDepthResolveFilterMax",
    })
}

/// The SDK enumerator of `MTLMultisampleStencilResolveFilter`'s raw value `raw`.
fn stencil_filter_name(raw: u64) -> Result<&'static str, String> {
    Ok(match MultisampleStencilResolveFilter::from_raw(raw)? {
        MultisampleStencilResolveFilter::Sample0 => "MTLMultisampleStencilResolveFilterSample0",
        MultisampleStencilResolveFilter::DepthResolvedSample => {
            "MTLMultisampleStencilResolveFilterDepthResolvedSample"
        }
    })
}

/// The site a row of the action sweep is emitted at: the two SDK enumerators **colour
/// attachment 0** carries, which is the row's own unrotated triple (`Row::rotated`: the swept
/// colour takes the row, the depth attachment `rotated(1)` and the stencil attachment
/// `rotated(2)`). Naming the attachment the row is *about* is what keeps the site the selector it
/// reads; the other two attachments' values are in the same row's compared facts.
fn action_site(load: u64, store: u64) -> String {
    match (load_action_name(load), store_action_name(store)) {
        (Ok(load), Ok(store)) => format!("{load}/{store}"),
        _ => format!("load-{load}-store-{store}"),
    }
}

/// The site a row of the filter sweep is emitted at: the SDK enumerator the **depth** attachment
/// carries and the one the **stencil** attachment carries.
///
/// A resolve filter is only ever on those two attachments, and they read `rotated(1)` and
/// `rotated(2)` of the row rather than the row itself, so the caller passes the rotated values. The
/// row's own unrotated depth filter is on no attachment at all, and a site named after it would
/// name a value nothing in the row reads.
fn filter_site(depth: u64, stencil: u64) -> String {
    match (depth_filter_name(depth), stencil_filter_name(stencil)) {
        (Ok(depth), Ok(stencil)) => format!("{depth}/{stencil}"),
        _ => format!("depth-{depth}-stencil-{stencil}"),
    }
}

/// What one case sweeps, and so which rows its `run` states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sweep {
    /// One descriptor of the named shape, read whole.
    Whole(Shape),
    /// Every row of [`action_rows`].
    Actions,
    /// Every row of [`filter_rows`].
    Filters,
    /// Every value of [`INDEX_VALUES`], on all six index fields at once.
    Indices,
    /// Every row of [`CLEAR_COLORS`].
    ClearColors,
    /// Every value of [`CLEAR_DEPTHS`].
    ClearDepths,
    /// Every value of [`clear_stencils`].
    ClearStencils,
    /// Every value of [`NUMBER_SWEEPS`], one field of the pass at a time.
    Numbers,
    /// Every row of [`POSITION_ROWS`].
    Positions,
}

/// A case over one render pass descriptor's own facts, or over one sweep of them.
pub struct PassDescriptorCase {
    name: &'static str,
    sweep: Sweep,
}

/// Every case of the family.
pub fn all() -> Vec<PassDescriptorCase> {
    vec![
        PassDescriptorCase {
            name: "pass-descriptor-defaults",
            sweep: Sweep::Whole(Shape::Defaults),
        },
        PassDescriptorCase {
            name: "pass-descriptor-set",
            sweep: Sweep::Whole(Shape::Set),
        },
        PassDescriptorCase {
            name: "pass-descriptor-assigned",
            sweep: Sweep::Whole(Shape::Assigned),
        },
        PassDescriptorCase {
            name: "pass-descriptor-cleared",
            sweep: Sweep::Whole(Shape::Cleared),
        },
        PassDescriptorCase {
            name: "pass-attachment-actions",
            sweep: Sweep::Actions,
        },
        PassDescriptorCase {
            name: "pass-attachment-filters",
            sweep: Sweep::Filters,
        },
        PassDescriptorCase {
            name: "pass-attachment-indices",
            sweep: Sweep::Indices,
        },
        PassDescriptorCase {
            name: "pass-clear-colors",
            sweep: Sweep::ClearColors,
        },
        PassDescriptorCase {
            name: "pass-clear-depths",
            sweep: Sweep::ClearDepths,
        },
        PassDescriptorCase {
            name: "pass-clear-stencils",
            sweep: Sweep::ClearStencils,
        },
        PassDescriptorCase {
            name: "pass-numbers",
            sweep: Sweep::Numbers,
        },
        PassDescriptorCase {
            name: "pass-sample-positions",
            sweep: Sweep::Positions,
        },
    ]
}

impl Sweep {
    /// Every row the sweep states, each at the site it is recorded under.
    ///
    /// The one place a case's rows are decided, so the fingerprint a case feeds and the rows its
    /// `run` states are the same list by construction.
    fn rows(self) -> Vec<(String, PassSpec)> {
        let base = crate::pass_descriptors::BASE;
        match self {
            Sweep::Whole(shape) => {
                let mut spec = PassSpec::new(shape, Reads::Whole);
                if shape.sets() || shape == Shape::Assigned {
                    spec = spec
                        .with_numbers(STATED_NUMBERS)
                        .with_positions(&STATED_POSITIONS);
                }
                vec![(FACTS.to_owned(), spec)]
            }
            Sweep::Actions => action_rows()
                .into_iter()
                .map(|row| {
                    (
                        action_site(row.actions.load, row.actions.store),
                        PassSpec::new(Shape::Defaults, Reads::Attachments).at(row),
                    )
                })
                .collect(),
            Sweep::Filters => filter_rows()
                .into_iter()
                .map(|row| {
                    (
                        filter_site(row.rotated(1).filters.depth, row.rotated(2).filters.stencil),
                        PassSpec::new(Shape::Defaults, Reads::Attachments).at(row),
                    )
                })
                .collect(),
            Sweep::Indices => INDEX_VALUES
                .iter()
                .map(|value| {
                    (
                        format!("index-{value:#x}"),
                        PassSpec::new(Shape::Defaults, Reads::Attachments)
                            .at(base.with_indices(*value)),
                    )
                })
                .collect(),
            Sweep::ClearColors => CLEAR_COLORS
                .iter()
                .enumerate()
                .map(|(at, clear)| {
                    (
                        format!("color-{at}"),
                        PassSpec::new(Shape::Defaults, Reads::ClearColor)
                            .at(base.with_clear_color(*clear)),
                    )
                })
                .collect(),
            Sweep::ClearDepths => CLEAR_DEPTHS
                .iter()
                .map(|clear| {
                    (
                        // The bits and not the value: a NaN and a negative NaN print alike, and
                        // two sites of one name would be two rows the comparison cannot tell
                        // apart.
                        format!("depth-{:#018x}", clear.to_bits()),
                        PassSpec::new(Shape::Defaults, Reads::ClearDepth)
                            .at(base.with_clear_depth(*clear)),
                    )
                })
                .collect(),
            Sweep::ClearStencils => clear_stencils()
                .into_iter()
                .map(|clear| {
                    (
                        format!("stencil-{clear}"),
                        PassSpec::new(Shape::Defaults, Reads::ClearStencil)
                            .at(base.with_clear_stencil(clear)),
                    )
                })
                .collect(),
            Sweep::Numbers => NUMBER_SWEEPS
                .iter()
                .flat_map(|(field, values)| {
                    values.iter().map(move |value| {
                        // Every other number stays at the stated table's, so each row holds one
                        // number away from eight others and a side answering one of them out of
                        // another's field reads wrong.
                        let mut numbers = STATED_NUMBERS;
                        numbers.0[*field] = *value;
                        (
                            format!("number-{field}-{value}"),
                            PassSpec::new(Shape::Defaults, Reads::Numbers).with_numbers(numbers),
                        )
                    })
                })
                .collect(),
            Sweep::Positions => POSITION_ROWS
                .iter()
                .enumerate()
                .map(|(at, positions)| {
                    (
                        format!("positions-{at}"),
                        PassSpec::new(Shape::Defaults, Reads::Positions).with_positions(positions),
                    )
                })
                .collect(),
        }
    }
}

impl Case for PassDescriptorCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // Every row the case states, in order, each by its site and its whole stated table: a row
        // that moves, is added or is dropped is a different observation.
        let mut words = Vec::new();
        for (at, spec) in self.sweep.rows() {
            words.extend(at.as_bytes());
            words.push(0);
            words.extend(spec.words());
        }
        Ok(vec![Input::bytes("rows", &words)])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for (at, spec) in self.sweep.rows() {
            match side.pass_descriptor_facts(&spec) {
                Ok(facts) => sites.push(facts.node(at)),
                // A row the side refused is recorded at its own site and the sweep goes on: the
                // other rows' answers stay readable, and a refusal where Metal answered compares
                // as a refusal at that one row.
                Err(failure) => sites.push(Node::leaf(at, Observed::value(failure.recorded()))),
            }
        }
        Ok(())
    }
}

/// The properties the sweeps' arguments rest on: that the stated table is apart in every field, that
/// the defaults shape provably cannot catch a crossing, and that each sweep covers the space its
/// prose claims.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass_descriptors::{
        stated_actions, stated_clear_color, stated_index, DEPTH_FILTERS, INDEX_FIELDS,
        LOAD_ACTIONS, STATED_ATTACHMENTS, STENCIL_FILTERS, STORE_ACTIONS, STORE_OPTIONS,
    };
    use std::collections::BTreeSet;

    /// Every one of the sixty index fields the set shape states holds a different number, so a side
    /// answering one of them out of another field — of its own attachment or of another — reads
    /// wrong.
    #[test]
    fn the_stated_index_fields_are_all_different() {
        let mut seen: BTreeSet<u64> = BTreeSet::new();
        let mut count = 0;
        for at in 0..STATED_ATTACHMENTS {
            for field in 0..INDEX_FIELDS {
                seen.insert(stated_index(at, field));
                count += 1;
            }
        }
        assert_eq!(count, 60, "ten attachments of six index fields");
        assert_eq!(seen.len(), count, "two stated index fields hold one number");
    }

    /// No two attachments the set shape states hold one (load, store) pair, so an attachment
    /// answered out of another attachment's fields reads wrong.
    #[test]
    fn no_two_stated_attachments_share_an_action_pair() {
        let pairs: BTreeSet<(u64, u64)> = (0..STATED_ATTACHMENTS)
            .map(|at| {
                let actions = stated_actions(at);
                (actions.load, actions.store)
            })
            .collect();
        assert_eq!(pairs.len(), STATED_ATTACHMENTS as usize);
    }

    /// Every channel of every clear colour the set shape states is a different number.
    #[test]
    fn the_stated_clear_colors_are_all_different() {
        let channels: BTreeSet<u64> = (0..STATED_ATTACHMENTS)
            .flat_map(stated_clear_color)
            .map(f64::to_bits)
            .collect();
        assert_eq!(channels.len(), STATED_ATTACHMENTS as usize * 4);
    }

    /// The defaults shape **cannot** catch a crossing among the index fields, and this is what says
    /// so: Metal's default for all six is one value, so the shape that measures the defaults is the
    /// wrong shape to measure field identity, and the set shape above is the right one.
    #[test]
    fn the_defaults_shape_cannot_catch_a_crossed_index_getter() {
        let base = crate::pass_descriptors::BASE;
        let fields: BTreeSet<u64> = base.indices.0.into_iter().collect();
        assert_eq!(
            fields.len(),
            1,
            "a default attachment's six index fields differ, so the claim in the module's prose \
             is wrong"
        );
    }

    /// The action sweep compares the **whole** 3 x 6 x 2 product: each row states colour 0 at its
    /// own triple and the depth attachment one place on in all three spaces, so the eighteen rows
    /// hold all thirty-six.
    #[test]
    fn the_action_sweep_is_exhaustive() {
        let mut seen: BTreeSet<(u64, u64, u64)> = BTreeSet::new();
        for row in action_rows() {
            for shift in [0, 1, 2] {
                let actions = row.rotated(shift).actions;
                seen.insert((actions.load, actions.store, actions.options));
            }
        }
        let whole = (LOAD_ACTIONS * STORE_ACTIONS * STORE_OPTIONS) as usize;
        assert_eq!(seen.len(), whole, "the action sweep misses a triple");
    }

    /// No row of the action sweep gives two attachments one value of any of the three axes, so a
    /// crossing between attachment kinds fails every row.
    #[test]
    fn no_action_row_repeats_a_value_across_the_attachments() {
        for row in action_rows() {
            let stated: Vec<_> = [0, 1, 2]
                .into_iter()
                .map(|shift| row.rotated(shift).actions)
                .collect();
            for (one, two) in [(0, 1), (0, 2), (1, 2)] {
                assert_ne!(stated[one].load, stated[two].load);
                assert_ne!(stated[one].store, stated[two].store);
            }
        }
    }

    /// Which rows of the action sweep cannot catch a load action answered out of the store action's
    /// field: exactly the three whose two ordinals are equal. Named rather than left implicit,
    /// because a row that cannot catch a crossing is a row whose pass means less.
    #[test]
    fn three_action_rows_cannot_catch_a_load_crossed_with_a_store() {
        let blind = action_rows()
            .into_iter()
            .filter(|row| row.actions.load == row.actions.store)
            .count();
        assert_eq!(blind, LOAD_ACTIONS as usize);
    }

    /// The filter sweep compares every pair of the two resolve filters.
    #[test]
    fn the_filter_sweep_is_exhaustive() {
        let depth: BTreeSet<u64> = filter_rows()
            .iter()
            .map(|row| row.rotated(1).filters.depth)
            .collect();
        let stencil: BTreeSet<u64> = filter_rows()
            .iter()
            .map(|row| row.rotated(2).filters.stencil)
            .collect();
        assert_eq!(depth.len(), DEPTH_FILTERS as usize);
        assert_eq!(stencil.len(), STENCIL_FILTERS as usize);
        assert_eq!(
            filter_rows().len(),
            (DEPTH_FILTERS * STENCIL_FILTERS) as usize
        );
    }

    /// Every value the action and filter sweeps put to Metal has an SDK enumerator name here, and
    /// every name here is a value one of them reaches.
    ///
    /// The depth claim of these two sweeps, held by execution. They state their rows as raw
    /// ordinals on purpose, so the enumerator reaches the compared text only through the site name:
    /// a value whose ordinal a row feeds but whose name no function above returns would be swept
    /// and invisible, and a name no row reaches would be a value this module claims and never asks
    /// for. Both directions fail here.
    #[test]
    fn every_swept_action_and_filter_is_named() {
        let mut loads: BTreeSet<&str> = BTreeSet::new();
        let mut stores: BTreeSet<&str> = BTreeSet::new();
        for row in action_rows() {
            for shift in [0, 1, 2] {
                let actions = row.rotated(shift).actions;
                loads.insert(load_action_name(actions.load).expect("a swept load action is named"));
                stores.insert(
                    store_action_name(actions.store).expect("a swept store action is named"),
                );
            }
        }
        assert_eq!(loads.len(), LOAD_ACTIONS as usize, "{loads:?}");
        assert_eq!(stores.len(), STORE_ACTIONS as usize, "{stores:?}");
        // The two the suite reached nowhere else, said by name so this test is what keeps them.
        assert!(stores.contains("MTLStoreActionUnknown"));
        assert!(stores.contains("MTLStoreActionCustomSampleDepthStore"));

        let mut depths: BTreeSet<&str> = BTreeSet::new();
        let mut stencils: BTreeSet<&str> = BTreeSet::new();
        for row in filter_rows() {
            depths.insert(
                depth_filter_name(row.rotated(1).filters.depth).expect("a swept filter is named"),
            );
            stencils.insert(
                stencil_filter_name(row.rotated(2).filters.stencil)
                    .expect("a swept filter is named"),
            );
        }
        assert_eq!(depths.len(), DEPTH_FILTERS as usize, "{depths:?}");
        assert_eq!(stencils.len(), STENCIL_FILTERS as usize, "{stencils:?}");
        assert!(depths.contains("MTLMultisampleDepthResolveFilterMin"));
        assert!(depths.contains("MTLMultisampleDepthResolveFilterMax"));
        assert!(stencils.contains("MTLMultisampleStencilResolveFilterDepthResolvedSample"));

        // No name is unreachable: a raw value past each type's count has no name, so the functions
        // are exactly the swept spaces and not a wider list written once and left.
        assert!(load_action_name(LOAD_ACTIONS).is_err());
        assert!(store_action_name(STORE_ACTIONS).is_err());
        assert!(depth_filter_name(DEPTH_FILTERS).is_err());
        assert!(stencil_filter_name(STENCIL_FILTERS).is_err());
    }

    /// No two rows of either sweep name one site.
    ///
    /// A transcript holding two sites of one name silently loses one, so a site name that stopped
    /// being unique would quietly halve a sweep.
    #[test]
    fn no_two_rows_of_the_action_or_filter_sweeps_name_one_site() {
        let actions: BTreeSet<String> = action_rows()
            .into_iter()
            .map(|row| action_site(row.actions.load, row.actions.store))
            .collect();
        assert_eq!(actions.len(), action_rows().len(), "{actions:?}");
        let filters: BTreeSet<String> = filter_rows()
            .into_iter()
            .map(|row| filter_site(row.rotated(1).filters.depth, row.rotated(2).filters.stencil))
            .collect();
        assert_eq!(filters.len(), filter_rows().len(), "{filters:?}");
        // Every site carries both enumerators rather than falling back to the ordinals.
        for site in actions.iter().chain(&filters) {
            assert!(site.starts_with("MTL"), "{site} is not an enumerator pair");
        }
    }

    /// The stencil clear sweep is exhaustive over the byte, which is the claim its prose makes.
    #[test]
    fn the_stencil_clear_sweep_covers_every_byte() {
        let swept: BTreeSet<u32> = clear_stencils().into_iter().collect();
        for byte in 0..=255u32 {
            assert!(
                swept.contains(&byte),
                "stencil clear {byte} is never stated"
            );
        }
        assert_eq!(swept.len(), clear_stencils().len(), "a row is stated twice");
    }

    /// Every clear the sweeps state is a different bit pattern, so no two rows of one sweep are one
    /// observation under two names.
    #[test]
    fn the_swept_clears_are_all_distinct() {
        let depths: BTreeSet<u64> = CLEAR_DEPTHS.iter().map(|clear| clear.to_bits()).collect();
        assert_eq!(depths.len(), CLEAR_DEPTHS.len());
        let colors: BTreeSet<[u64; 4]> = CLEAR_COLORS
            .iter()
            .map(|clear| std::array::from_fn(|at| clear[at].to_bits()))
            .collect();
        assert_eq!(colors.len(), CLEAR_COLORS.len());
    }

    /// Which rows of the clear-colour sweep cannot catch a channel answered out of another
    /// channel's field: exactly the three whose channels are not all different, which are Metal's
    /// own default and the two unit corners. They are in the sweep because they are the values a
    /// program is likeliest to state, and the other three rows are what catches a crossed channel.
    #[test]
    fn three_clear_color_rows_cannot_catch_a_crossed_channel() {
        let blind = CLEAR_COLORS
            .iter()
            .filter(|clear| {
                clear
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<BTreeSet<u64>>()
                    .len()
                    < 4
            })
            .count();
        assert_eq!(blind, 3);
        assert_eq!(CLEAR_COLORS.len() - blind, 3, "no row catches a crossing");
    }

    /// No value the number sweep states for a field is that field's own number in the stated
    /// table, so every row of the sweep is its own observation.
    #[test]
    fn no_swept_number_is_its_fields_stated_number() {
        for (field, values) in NUMBER_SWEEPS {
            assert!(
                !values.contains(&STATED_NUMBERS.0[field]),
                "number {field} is swept at the value the stated table already holds"
            );
        }
    }

    /// Every row of every sweep sits at its own site, so no recording holds one name twice.
    #[test]
    fn every_row_has_its_own_site() {
        for case in all() {
            let rows = case.sweep.rows();
            let sites: BTreeSet<String> = rows.iter().map(|(at, _)| at.clone()).collect();
            assert_eq!(sites.len(), rows.len(), "{} repeats a site", case.name);
        }
    }

    /// Every row of every sweep states a different table, so two sites of one case are two
    /// observations rather than one asked twice.
    #[test]
    fn every_row_states_its_own_table() {
        for case in all() {
            let rows = case.sweep.rows();
            let tables: BTreeSet<Vec<u8>> = rows.iter().map(|(_, spec)| spec.words()).collect();
            assert_eq!(
                tables.len(),
                rows.len(),
                "{} states a table twice",
                case.name
            );
        }
    }

    /// Each case's rows are the list its fingerprint covers, so a sweep cannot grow a row the
    /// inputs do not name.
    #[test]
    fn the_numbers_sweep_states_every_field() {
        let fields: BTreeSet<usize> = NUMBER_SWEEPS.iter().map(|(field, _)| *field).collect();
        assert_eq!(fields.len(), STATED_NUMBERS.0.len());
        for (field, values) in NUMBER_SWEEPS {
            assert!(values.contains(&0), "field {field} is never stated at 0");
            assert!(
                values.len() >= 4,
                "field {field} is swept at fewer than four values"
            );
        }
    }

    /// The position sweep states no row Metal asserts on: every coordinate inside `[0, 1)`, and no
    /// row longer than the sample counts an Apple GPU takes custom positions at.
    #[test]
    fn no_position_row_is_one_metal_asserts_on() {
        for positions in POSITION_ROWS {
            assert!(
                matches!(positions.len(), 0 | 2 | 4),
                "a row of {} positions is not a sample count Metal takes",
                positions.len()
            );
            for (x, y) in positions {
                assert!(
                    (0.0..1.0).contains(x),
                    "sample position x {x} is outside [0, 1)"
                );
                assert!(
                    (0.0..1.0).contains(y),
                    "sample position y {y} is outside [0, 1)"
                );
            }
        }
    }
}
