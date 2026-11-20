//! What a view of a texture answers about its parent, what each of `MTLTexture`'s four view
//! creators makes, and what a shareable texture's handle opens.
//!
//! The suite already made views, through one selector and always of the parent's whole level and
//! slice ranges ([`crate::texture::Over::View`]) — so `parentTexture`, `parentRelativeLevel`,
//! `parentRelativeSlice` and `swizzle` were never read of anything, and the `newTextureViewWith`
//! family's other three selectors were never sent. This family states the **ranges**, states
//! which selector makes the view, and reads the four getters of every view it makes.
//!
//! # The cases, and what each one would catch
//!
//! - **`texture-view-parents`**: a base texture, a view of levels 1 to 2, a view of slices 1 to 2
//!   of an array, and — the row that matters — a **view of a view**. What
//!   `parentRelativeLevel` counts from there is a real question with two defensible answers: the
//!   immediate parent, or the root of the chain. The runtime models the immediate parent
//!   (`Texture::parent_relative_level` subtracts the parent's own base level), and nothing in this
//!   tree had asked Metal. The base texture's own row is the baseline: a texture that is not a view
//!   answers no parent and 0 for both relative getters. The case also sweeps **every**
//!   `MTLTextureSwizzle` value as a parent's swizzle — each of the six in all four channels, and
//!   each of the six in one channel over the default, twelve parents in all — and reads each back
//!   off a view made by a selector that states no swizzle. That reading is Metal's choice between
//!   the parent's swizzle and the default, so it is the one place in the suite where the swizzle
//!   enum is compared value by value rather than at one representative.
//! - **`texture-view-selectors`**: the same whole-texture view through all four selectors, and the
//!   same sub-range view through the three that can state one. Two facts: whether
//!   `newTextureViewWithPixelFormat:` and the `textureType:levels:slices:` form agree over the
//!   whole texture, and what `swizzle` a view made by a selector that states none answers — the
//!   parent's, or `MTLTextureSwizzleChannelsDefault`. The case makes the question answerable by
//!   giving the **parent** a non-default swizzle, so "the parent's" and "the default" are two
//!   different recorded values rather than one.
//! - **`texture-view-descriptors`**: `newTextureViewWithDescriptor:` beside the five-argument
//!   selector over the same five values, at three range and swizzle combinations. This is the only
//!   way the suite reaches `MTLTextureViewDescriptor` at all, and what it records is that the two
//!   paths answer alike — a release where they part fails the case rather than passing under a rule
//!   nobody re-measured.
//! - **`texture-shared-handles`**: `shareable` of an ordinary texture, of a view of one, of a
//!   texture of `newSharedTextureWithDescriptor:` and of a view of **that**; then that shared
//!   texture's `newSharedTextureHandle`, the handle's own `device`, and the texture
//!   `newSharedTextureWithHandle:` opens from it. The runtime models `isShareable` as false for
//!   every view, whatever its parent, which is a claim about Metal with no run behind it.
//!
//! # The positive control, and why this family needs one
//!
//! A view's getters are the shape that could read as a pass while nothing was viewed: a side that
//! answered 0, 0 and no parent for everything would agree with a side that made no views at all.
//! So every table is built so the recorded values are ones nothing else leaves:
//!
//! - **the relative getters are recorded non-zero somewhere.** This module's own tests assert, with no device,
//!   that every table holds a view whose first level or first slice is above 0, and that
//!   `texture-view-parents` holds a view **of a view**. Without those rows the case would record
//!   0 at every site, which is also what a side that ignored the ranges would answer;
//! - **the parent carries a non-default swizzle**, so the two possible answers for a view made
//!   without one are two different values;
//! - **the swizzle sweep moves the view's recorded answer at every row**: measured on this chip
//!   the view inherits its parent's swizzle exactly, at all twelve, so a side answering the
//!   default for an inheriting view differs at eleven of them. The sweep states each value in one
//!   channel as well as in four, which is what tells a per-channel reading from a side that read
//!   one channel and broadcast it;
//! - **`shareable` is recorded both ways**, because the shared-handle table holds an ordinary
//!   texture and a shared one.
//!
//! # The swizzle space, and what of it is a mirror
//!
//! `MTLTextureSwizzle` has six values and all six are swept, but only as the **parent's**: a
//! view's own stated swizzle is read straight back from what the case wrote, so sweeping that is
//! indistinguishable by construction and `texture-view-descriptors` states three of them for the
//! different reason that the descriptor path and the selector path must agree on each. The
//! `MTLTextureUsage` bitfield is the other small space this family touches and does **not** sweep:
//! `usage` reads back what the descriptor said, and the usage bits that decide anything decide it
//! by Metal's validation refusing a call — which ends Metal's process and so cannot be a case.
//!
//! # Why every case here is safe
//!
//! A view holds no memory: the call is Metal's bookkeeping over a texture that already exists, it
//! compiles nothing, allocates no texels and reaches no queue. The textures are 64 by 64
//! `RGBA8Unorm` of four levels, and the array ones four slices of that; nothing here dispatches,
//! draws, blits or commits.
//!
//! The one hazard a view creator carries is a range or a format Metal's validation stops, which
//! ends its process. [`crate::texture_views::ViewRequest::is_well_formed`] decides that from the
//! request's own shape and the parent's, this module's own tests hold every row of every table to it with no
//! device, and the Metal side calls it in the one place that makes a view — so a row somebody adds
//! to a recorded table by mistake is refused by the side rather than by a convention.
//!
//! `newSharedTextureHandle` is asked **only of a shareable texture**. What Metal does for one that
//! is not is unmeasured: the runtime models nil, and a case that asked and was wrong would end
//! Metal's process. It is named here as an open hole rather than probed.

use crate::case::{Abort, Case, Handle, Input, MetalSource, Side};
use crate::texture::TextureRequest;
use crate::texture_query::{HeapTextureQuery, Storage};
use crate::texture_views::{swizzle_word, ParentShape, ViewCall, ViewRequest};
use crate::transcript::{Node, Observed, Sites};
use metal2vulkan::metal::{
    PixelFormat, TextureSwizzle, TextureSwizzleChannels, TextureType, TextureUsage,
};

/// The case names, as the report and the recordings spell them.
pub const PARENTS: &str = "texture-view-parents";
pub const SELECTORS: &str = "texture-view-selectors";
pub const DESCRIPTORS: &str = "texture-view-descriptors";
pub const SHARED_HANDLES: &str = "texture-shared-handles";

/// The format every texture of the family is: a plain uncompressed colour format, so a view of it
/// is a view and not a reinterpretation.
const FORMAT: PixelFormat = PixelFormat::RGBA8Unorm;

/// Every texture's side, and an array's slices and a 2D texture's levels.
const SIDE: u32 = 64;
const LEVELS: u64 = 4;
const SLICES: u64 = 4;

/// The swizzle the **parent** textures carry, and the one some views state.
///
/// A non-default swizzle on the parent is the positive control for the selectors case: a view made
/// by a selector that states none answers either the parent's or Metal's default, and those are
/// only two different recorded values if the parent's is not the default.
const PARENT_SWIZZLE: TextureSwizzleChannels = TextureSwizzleChannels {
    red: TextureSwizzle::Blue,
    green: TextureSwizzle::Green,
    blue: TextureSwizzle::Red,
    alpha: TextureSwizzle::One,
};

/// The swizzle the views that state one carry: a third value, different from both the parent's and
/// Metal's default, so a view that took its own and a view that took its parent's are told apart.
const VIEW_SWIZZLE: TextureSwizzleChannels = TextureSwizzleChannels {
    red: TextureSwizzle::Alpha,
    green: TextureSwizzle::Zero,
    blue: TextureSwizzle::Green,
    alpha: TextureSwizzle::Red,
};

/// Every value `MTLTextureSwizzle` has, in Metal's own order.
///
/// Six, which is all of them. They are swept as the **parent's** swizzle, through a selector that
/// states none, because that is the one reading Metal decides: a view's own stated swizzle is read
/// back from what the case wrote and is indistinguishable by construction. What the sweep asks is
/// whether every one of the six survives into a view that never mentioned it.
const EVERY_SWIZZLE: [TextureSwizzle; 6] = [
    TextureSwizzle::Zero,
    TextureSwizzle::One,
    TextureSwizzle::Red,
    TextureSwizzle::Green,
    TextureSwizzle::Blue,
    TextureSwizzle::Alpha,
];

/// The parent swizzles the inheritance sweep states: Metal's default, each of the six values in
/// **all four** channels, and each of the six in **one** channel over the default.
///
/// The second half is what makes the sweep a sweep of a swizzle and not of a value: a side that
/// read one channel and filled the other three from it would answer every all-four row correctly
/// and differ at all six of the others.
fn parent_swizzles() -> Vec<TextureSwizzleChannels> {
    let mut swizzles = vec![TextureSwizzleChannels::DEFAULT];
    for channel in EVERY_SWIZZLE {
        swizzles.push(TextureSwizzleChannels {
            red: channel,
            green: channel,
            blue: channel,
            alpha: channel,
        });
    }
    for channel in EVERY_SWIZZLE {
        let one_channel = TextureSwizzleChannels {
            green: channel,
            ..TextureSwizzleChannels::DEFAULT
        };
        // `green: Green` **is** the default, so the one-channel loop reaches it twice. One site a
        // swizzle: two nodes of one name would leave a diff unable to say which row moved.
        if !swizzles.contains(&one_channel) {
            swizzles.push(one_channel);
        }
    }
    swizzles
}

/// The shape of the 2D parent, as a view request reads it.
const FLAT: ParentShape = ParentShape::texture_2d(FORMAT, LEVELS);

/// The shape of the array parent.
const ARRAY: ParentShape = ParentShape::texture_2d(FORMAT, LEVELS).array(SLICES);

/// One row of a table: a view of the parent, or a view of **that** view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row {
    /// The view made of the case's parent texture.
    pub view: ViewRequest,
    /// A view made of `view`, where the row makes one: the chain this family is for.
    pub of_the_view: Option<ViewRequest>,
}

impl Row {
    /// One view of the parent and nothing beyond it.
    pub const fn one(view: ViewRequest) -> Self {
        Self {
            view,
            of_the_view: None,
        }
    }

    /// A view of the parent, and then `of_the_view` made of that view.
    pub const fn then(view: ViewRequest, of_the_view: ViewRequest) -> Self {
        Self {
            view,
            of_the_view: Some(of_the_view),
        }
    }

    /// The site this row's answers sit under.
    pub fn site(&self) -> String {
        match self.of_the_view {
            None => self.view.site(),
            Some(inner) => format!("{}/then/{}", self.view.site(), inner.site()),
        }
    }

    /// The words the row fingerprints as.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.view.words();
        if let Some(inner) = self.of_the_view {
            out.extend(b"then");
            out.extend(inner.words());
        }
        out
    }
}

/// A case over one table of views of one parent shape.
pub struct TextureViewCase {
    name: &'static str,
    /// The parent every row views: flat or array.
    parent: ParentShape,
    rows: Vec<Row>,
    /// Whether the case also asks `newSharedTextureWithDescriptor:` and the handle calls, which
    /// `texture-shared-handles` alone does.
    asks_sharing: bool,
    /// Whether the case also sweeps every `MTLTextureSwizzle` as a parent's swizzle and reads it
    /// back off a view that states none, which `texture-view-parents` alone does.
    asks_inherited_swizzles: bool,
}

/// Every case of the family.
pub fn all() -> Vec<TextureViewCase> {
    vec![parents(), selectors(), descriptors(), shared_handles()]
}

/// A view of levels, a view of slices, and a view of a view.
///
/// The array parent, because a slice range is half of what the relative getters answer and a flat
/// texture has only one slice to give. The chain row is the case's subject: a view of levels 1 to 2
/// of a view of levels 1 to 3 is at level 2 of the **root** and level 1 of its parent, so the two
/// candidate rules give two different numbers.
fn parents() -> TextureViewCase {
    let whole = |call: ViewCall| ViewRequest::whole(call, &ARRAY, FORMAT);
    let rows = vec![
        // The whole texture through the four-argument selector: the baseline, where both relative
        // getters are 0 and the view is the parent's whole shape.
        Row::one(whole(ViewCall::LevelsSlices)),
        // Levels alone, slices alone, and both at once.
        Row::one(whole(ViewCall::LevelsSlices).levels(1, 2)),
        Row::one(whole(ViewCall::LevelsSlices).slices(TextureType::Type2DArray, 1, 2)),
        Row::one(
            whole(ViewCall::LevelsSlices)
                .levels(2, 2)
                .slices(TextureType::Type2DArray, 2, 2),
        ),
        // A single-slice view of an array, as a plain 2D texture: the shape that asks what a view
        // whose type is not its parent's answers for `arrayLength`.
        Row::one(
            whole(ViewCall::LevelsSlices)
                .levels(1, 1)
                .slices(TextureType::Type2D, 3, 1),
        ),
        // The chain. The inner request's ranges are counted from the **outer view's** own first
        // level and slice, which is what both candidate rules make different numbers of.
        Row::then(
            whole(ViewCall::LevelsSlices).levels(1, 3),
            ViewRequest::whole(
                ViewCall::LevelsSlices,
                &ParentShape::viewed(FORMAT, TextureType::Type2DArray, 3, SLICES),
                FORMAT,
            )
            .levels(1, 2),
        ),
        Row::then(
            whole(ViewCall::LevelsSlices).slices(TextureType::Type2DArray, 1, 3),
            ViewRequest::whole(
                ViewCall::LevelsSlices,
                &ParentShape::viewed(FORMAT, TextureType::Type2DArray, LEVELS, 3),
                FORMAT,
            )
            .slices(TextureType::Type2DArray, 1, 2),
        ),
    ];
    TextureViewCase {
        name: PARENTS,
        parent: ARRAY,
        rows,
        asks_sharing: false,
        asks_inherited_swizzles: true,
    }
}

/// The same view through every selector that can state it.
///
/// The whole texture through all four, then a sub-range through the three that take ranges: what
/// the recording says is whether the four agree, and what `swizzle` the two that state none answer.
fn selectors() -> TextureViewCase {
    let mut rows = Vec::new();
    for call in [
        ViewCall::PixelFormat,
        ViewCall::LevelsSlices,
        ViewCall::Swizzled,
        ViewCall::Descriptor,
    ] {
        let whole = ViewRequest::whole(call, &FLAT, FORMAT);
        rows.push(Row::one(if call.states_a_swizzle() {
            whole.with_swizzle(VIEW_SWIZZLE)
        } else {
            whole
        }));
    }
    for call in [
        ViewCall::LevelsSlices,
        ViewCall::Swizzled,
        ViewCall::Descriptor,
    ] {
        let view = ViewRequest::whole(call, &FLAT, FORMAT).levels(1, 2);
        rows.push(Row::one(if call.states_a_swizzle() {
            view.with_swizzle(VIEW_SWIZZLE)
        } else {
            view
        }));
    }
    // One swizzled view that states Metal's **default** swizzle: without it a reader cannot tell
    // "the selector kept the parent's" from "the selector took what it was given" where what it was
    // given happens to be the parent's.
    rows.push(Row::one(
        ViewRequest::whole(ViewCall::Swizzled, &FLAT, FORMAT)
            .levels(1, 2)
            .with_swizzle(TextureSwizzleChannels::DEFAULT),
    ));
    TextureViewCase {
        name: SELECTORS,
        parent: FLAT,
        rows,
        asks_sharing: false,
        asks_inherited_swizzles: false,
    }
}

/// `newTextureViewWithDescriptor:` beside the five-argument selector over the same five values.
fn descriptors() -> TextureViewCase {
    let mut rows = Vec::new();
    for (first_level, level_count, first_slice, slice_count) in
        [(0, LEVELS, 0, SLICES), (1, 2, 1, 2), (3, 1, 3, 1)]
    {
        for swizzle in [
            TextureSwizzleChannels::DEFAULT,
            PARENT_SWIZZLE,
            VIEW_SWIZZLE,
        ] {
            for call in [ViewCall::Descriptor, ViewCall::Swizzled] {
                rows.push(Row::one(
                    ViewRequest::whole(call, &ARRAY, FORMAT)
                        .levels(first_level, level_count)
                        .slices(TextureType::Type2DArray, first_slice, slice_count)
                        .with_swizzle(swizzle),
                ));
            }
        }
    }
    TextureViewCase {
        name: DESCRIPTORS,
        parent: ARRAY,
        rows,
        asks_sharing: true,
        asks_inherited_swizzles: false,
    }
}

/// `shareable` of an ordinary texture, of views of it, and of a shared texture and views of that;
/// then the shared texture's handle and what it opens.
fn shared_handles() -> TextureViewCase {
    TextureViewCase {
        name: SHARED_HANDLES,
        parent: FLAT,
        rows: vec![
            Row::one(ViewRequest::whole(ViewCall::PixelFormat, &FLAT, FORMAT)),
            Row::one(ViewRequest::whole(ViewCall::LevelsSlices, &FLAT, FORMAT).levels(1, 2)),
            Row::then(
                ViewRequest::whole(ViewCall::LevelsSlices, &FLAT, FORMAT).levels(1, 3),
                ViewRequest::whole(
                    ViewCall::LevelsSlices,
                    &ParentShape::viewed(FORMAT, TextureType::Type2D, 3, 1),
                    FORMAT,
                )
                .levels(1, 2),
            ),
        ],
        asks_sharing: true,
        asks_inherited_swizzles: false,
    }
}

/// The sites a case emits.
mod site {
    /// The parent texture's own answers, before any view is made of it.
    pub const PARENT: &str = "parent";
    /// A view's answers.
    pub const VIEW: &str = "view";
    /// A view of a view's answers.
    pub const INNER: &str = "view-of-the-view";
    /// A shared texture's answers.
    pub const SHARED: &str = "shared-texture";
    /// A view of a shared texture's answers.
    pub const SHARED_VIEW: &str = "view-of-the-shared-texture";
    /// What the shared texture's handle answers and opens.
    pub const HANDLE: &str = "shared-handle";
}

impl Case for TextureViewCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<crate::recording::ShaderRef>, String> {
        // Nothing here compiles: every answer is a getter of a texture or of a view of one.
        Ok(Vec::new())
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut words = parent_texture(self.parent, PARENT_SWIZZLE).words();
        for row in &self.rows {
            words.extend(row.words());
        }
        if self.asks_sharing {
            words.extend(shared_descriptor().words());
        }
        if self.asks_inherited_swizzles {
            for swizzle in parent_swizzles() {
                words.extend(parent_texture(self.parent, swizzle).words());
            }
        }
        Ok(vec![
            Input::bytes("views", &words),
            Input::bytes(
                "asks",
                if self.asks_sharing {
                    b"newSharedTextureWithDescriptor,newSharedTextureHandle".as_slice()
                } else {
                    b"".as_slice()
                },
            ),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        // The parent is the one call of the family that can stop the run: every row views it, so a
        // venue that cannot make it has nothing to say about any row.
        let parent = crate::case::made(
            sites,
            "parent-texture",
            side.texture(&parent_texture(self.parent, PARENT_SWIZZLE)),
        )?;
        sites.push(view_facts(side, parent, None, site::PARENT));
        for row in &self.rows {
            let mut node = Node::branch(row.site(), Vec::new());
            // A row the venue cannot make is recorded at its own site and the sweep goes on.
            let view = match side.texture_view(parent, &row.view) {
                Ok(view) => view,
                Err(failure) => {
                    sites.push(
                        node.with(Node::leaf(site::VIEW, Observed::value(failure.recorded()))),
                    );
                    continue;
                }
            };
            node = node.with(view_facts(side, view, Some(parent), site::VIEW));
            if let Some(inner) = row.of_the_view {
                node = node.with(match side.texture_view(view, &inner) {
                    Ok(made) => view_facts(side, made, Some(view), site::INNER),
                    Err(failure) => Node::leaf(site::INNER, Observed::value(failure.recorded())),
                });
            }
            sites.push(node);
        }
        if self.asks_sharing {
            self.sharing(side, sites);
        }
        if self.asks_inherited_swizzles {
            self.inherited_swizzles(side, sites);
        }
        Ok(())
    }
}

impl TextureViewCase {
    /// Every `MTLTextureSwizzle` as a parent's swizzle, each read back off a view made by a
    /// selector that states none.
    ///
    /// One parent and one view a swizzle, each at its own site, so the recording holds the
    /// parent's reading beside the view's: the parent's is the case's own value and the view's is
    /// Metal's choice between it and the default. The thirteen swizzles are
    /// [`parent_swizzles`] — the default, each of the six in all four channels, and each of the
    /// six in one channel — which is every value of the enum twice over in two positions.
    ///
    /// Nothing is encoded or dispatched: thirteen 64 by 64 four-level textures are made and their
    /// getters read.
    fn inherited_swizzles(&self, side: &mut dyn Side, sites: &mut Sites) {
        for swizzle in parent_swizzles() {
            let at = format!("inherited-{}", swizzle_word(swizzle));
            let mut node = Node::branch(at.clone(), Vec::new());
            let parent = match side.texture(&parent_texture(self.parent, swizzle)) {
                Ok(parent) => parent,
                Err(failure) => {
                    sites.push(node.with(Node::leaf(
                        site::PARENT,
                        Observed::value(failure.recorded()),
                    )));
                    continue;
                }
            };
            node = node.with(view_facts(side, parent, None, site::PARENT));
            // The four-argument selector, which states no swizzle at all: what the view answers is
            // the parent's swizzle or Metal's default, and never a value this call carried.
            let request = ViewRequest::whole(ViewCall::LevelsSlices, &self.parent, FORMAT);
            node = node.with(match side.texture_view(parent, &request) {
                Ok(view) => view_facts(side, view, Some(parent), site::VIEW),
                Err(failure) => Node::leaf(site::VIEW, Observed::value(failure.recorded())),
            });
            sites.push(node);
        }
    }

    /// `newSharedTextureWithDescriptor:`, the shared texture's answers, a view of it, and the
    /// handle calls.
    fn sharing(&self, side: &mut dyn Side, sites: &mut Sites) {
        let shared = match side.shared_texture(&shared_descriptor()) {
            Ok(shared) => shared,
            Err(failure) => {
                sites.push(Node::leaf(
                    site::SHARED,
                    Observed::value(failure.recorded()),
                ));
                return;
            }
        };
        sites.push(view_facts(side, shared, None, site::SHARED));
        // A view of a shared texture: the runtime models `isShareable` as false for every view
        // whatever its parent, which is exactly the claim this row holds against Metal.
        let of_the_shared =
            ViewRequest::whole(ViewCall::LevelsSlices, &SHARED_SHAPE, FORMAT).levels(1, 2);
        match side.texture_view(shared, &of_the_shared) {
            Ok(view) => sites.push(view_facts(side, view, Some(shared), site::SHARED_VIEW)),
            Err(failure) => sites.push(Node::leaf(
                site::SHARED_VIEW,
                Observed::value(failure.recorded()),
            )),
        }
        // Asked of the shareable texture alone: what Metal does for one that is not shareable is
        // unmeasured, and a case that asked and was wrong would end Metal's process.
        sites.push(match side.shared_texture_handle(shared) {
            Ok(facts) => facts.node(site::HANDLE),
            Err(failure) => Node::leaf(site::HANDLE, Observed::value(failure.recorded())),
        });
    }
}

/// The parent texture a case makes: `shape`'s own type, levels and slices, carrying `swizzle`.
fn parent_texture(shape: ParentShape, swizzle: TextureSwizzleChannels) -> TextureRequest<'static> {
    let mut request = TextureRequest::read_2d(FORMAT.as_raw(), SIDE, SIDE)
        .levels(shape.levels as u32)
        .swizzled(swizzle);
    if matches!(shape.texture_type, TextureType::Type2DArray) {
        request.texture_type = TextureType::Type2DArray;
        request.slices = shape.slices as u32;
    }
    request
}

/// The shape of the shared texture, as a view request reads it.
const SHARED_SHAPE: ParentShape = ParentShape::texture_2d(FORMAT, LEVELS);

/// The descriptor `newSharedTextureWithDescriptor:` is given.
///
/// `Private` storage, which is the only storage Metal makes a shareable texture in: plain Metal
/// answers nil for any other, and its validation stops it.
fn shared_descriptor() -> HeapTextureQuery {
    HeapTextureQuery::texture_2d(FORMAT, [SIDE, SIDE], Storage::Private)
        .with_levels(LEVELS as u32)
        .with_usage(TextureUsage::SHADER_READ)
}

/// What one texture answers about its place in a view chain, at `at`.
fn view_facts(side: &mut dyn Side, texture: Handle, parent: Option<Handle>, at: &str) -> Node {
    match side.texture_view_facts(texture, parent) {
        Ok(facts) => facts.node(at.to_owned()),
        Err(failure) => Node::leaf(at.to_owned(), Observed::value(failure.recorded())),
    }
}

/// The guards this family's safety argument and its claim to measure anything rest on, all with no
/// device at all — which is what a guard against ending Metal's process has to do.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture_views::swizzle_channels;
    use std::collections::BTreeSet;

    /// No view of the family is one Metal's validation stops, against the shape of the texture it
    /// is made of.
    ///
    /// Verified to fail: making [`parents`]' `levels(1, 2)` row ask for `levels(3, 2)` of the
    /// four-level parent fails here, naming the range and the parent's level count.
    #[test]
    fn no_view_of_the_family_is_one_metals_validation_stops() {
        let mut stopped = Vec::new();
        for case in all() {
            for row in &case.rows {
                if let Some(why) = row.view.is_well_formed(&case.parent) {
                    stopped.push(format!("{}: {}: {why}", case.name, row.view.site()));
                }
                if let Some(inner) = row.of_the_view {
                    // The inner view's parent is the outer view: its shape is the outer request's
                    // own ranges and type, which is what a view of a view is counted against.
                    let shape = ParentShape::viewed(
                        row.view.format,
                        row.view.texture_type,
                        row.view.level_count,
                        row.view.slice_count,
                    );
                    if let Some(why) = inner.is_well_formed(&shape) {
                        stopped.push(format!(
                            "{}: {} of {}: {why}",
                            case.name,
                            inner.site(),
                            row.view.site()
                        ));
                    }
                }
            }
            if case.asks_sharing {
                let of_the_shared =
                    ViewRequest::whole(ViewCall::LevelsSlices, &SHARED_SHAPE, FORMAT).levels(1, 2);
                if let Some(why) = of_the_shared.is_well_formed(&SHARED_SHAPE) {
                    stopped.push(format!("{}: the shared texture's view: {why}", case.name));
                }
            }
        }
        assert!(
            stopped.is_empty(),
            "these views are ones Metal's validation stops, so record mode would end Metal's \
             process; take them out of the table:\n{}",
            stopped.join("\n")
        );
    }

    /// Every table holds a view whose first level or first slice is above 0, so the two relative
    /// getters are recorded non-zero somewhere.
    ///
    /// This is the positive control: a table of whole-texture views alone would record 0 at every
    /// site, which is also what a side that ignored the ranges entirely would answer.
    ///
    /// Verified to fail: making [`selectors`]' sub-range views whole **and** dropping its last
    /// row — the swizzled one that also starts at level 1 — fails here naming the case. Either
    /// alone leaves a row that starts above the parent's first, which is the guard working.
    #[test]
    fn every_table_holds_a_view_that_does_not_start_at_its_parents_first() {
        for case in all() {
            let moved = case
                .rows
                .iter()
                .any(|row| row.view.first_level > 0 || row.view.first_slice > 0);
            assert!(
                moved,
                "{}: every view starts at its parent's own first level and slice, so \
                 `parentRelativeLevel` and `parentRelativeSlice` are recorded as 0 everywhere and \
                 a side that ignored the ranges would pass",
                case.name
            );
        }
    }

    /// The parents case holds a view **of a view** whose inner range starts above 0, which is the
    /// one shape where the two candidate rules for `parentRelativeLevel` give different numbers.
    ///
    /// Verified to fail: making **both** chain rows of [`parents`] start at their parent's own
    /// first — `levels(0, 3)` and `slices(.., 0, 3)` — fails here. One alone leaves the other
    /// chain, which is the guard working rather than a hole.
    #[test]
    fn the_parents_case_holds_a_chain_whose_inner_view_starts_above_its_parents_first() {
        let case = parents();
        let chained: Vec<&Row> = case
            .rows
            .iter()
            .filter(|row| {
                row.of_the_view.is_some_and(|inner| {
                    (inner.first_level > 0 && row.view.first_level > 0)
                        || (inner.first_slice > 0 && row.view.first_slice > 0)
                })
            })
            .collect();
        assert!(
            !chained.is_empty(),
            "no view of a view whose own first level or slice and whose parent's are both above 0, \
             so `parentRelativeLevel` counted from the immediate parent and counted from the root \
             of the chain would record the same number and the case would settle nothing"
        );
    }

    /// The parent's swizzle is not Metal's default, and the swizzle some views state is neither
    /// the parent's nor the default: three distinct values, which is what makes "the view took its
    /// parent's" and "the view took the default" two different recorded answers.
    ///
    /// Verified to fail: setting `PARENT_SWIZZLE` to `TextureSwizzleChannels::DEFAULT` fails here.
    #[test]
    fn three_swizzles_tell_the_two_candidate_answers_apart() {
        let words: BTreeSet<String> = [
            TextureSwizzleChannels::DEFAULT,
            PARENT_SWIZZLE,
            VIEW_SWIZZLE,
        ]
        .into_iter()
        .map(swizzle_word)
        .collect();
        assert_eq!(
            words.len(),
            3,
            "the default, the parent's and the view's swizzle are not three distinct values, so a \
             view made by a selector that states none records one answer whichever rule Metal \
             follows: {words:?}"
        );
    }

    /// The inheritance sweep states **every** `MTLTextureSwizzle` value, and states each of them
    /// in a channel of its own as well as in all four at once.
    ///
    /// This is the depth claim for `MTLTextureSwizzleChannels`: six values, all six compared.
    /// A sweep of all-four-channel swizzles alone would pass a side that read one channel and
    /// filled the rest from it, which is why the one-channel half is asserted too.
    ///
    /// Verified to fail: emptying [`parent_swizzles`]' second loop, so every row states a swizzle
    /// whose four channels agree, fails the per-channel half here naming all six values.
    #[test]
    fn the_inheritance_sweep_states_every_swizzle_value() {
        let swizzles = parent_swizzles();
        let mut missing = Vec::new();
        let mut not_alone = Vec::new();
        for channel in EVERY_SWIZZLE {
            let appears = swizzles
                .iter()
                .any(|swizzle| swizzle_channels(*swizzle).contains(&channel));
            if !appears {
                missing.push(channel.as_raw());
            }
            // A swizzle where this value is in one channel and the others are not it: the shape
            // that tells a per-channel reading from a broadcast one.
            let alone = swizzles.iter().any(|swizzle| {
                let channels = swizzle_channels(*swizzle);
                channels.iter().filter(|held| **held == channel).count() == 1
            });
            if !alone {
                not_alone.push(channel.as_raw());
            }
        }
        assert!(
            missing.is_empty(),
            "these `MTLTextureSwizzle` values are compared against Metal nowhere in the suite: \
             {missing:?}"
        );
        assert!(
            not_alone.is_empty(),
            "these values appear only with all four channels the same, so a side that broadcast \
             one channel over the other three would pass every row: {not_alone:?}"
        );
        assert!(
            parents().asks_inherited_swizzles,
            "the sweep is not switched on in the case that is supposed to run it"
        );
        let distinct: BTreeSet<String> = swizzles.iter().copied().map(swizzle_word).collect();
        assert_eq!(
            distinct.len(),
            swizzles.len(),
            "two rows of the sweep are the same swizzle, so one site name is emitted twice and a \
             diff cannot say which row moved"
        );
    }

    /// The selectors case sends all four, and the descriptors case sends the two that take a
    /// swizzle over the same values.
    #[test]
    fn the_tables_send_the_selectors_they_are_for() {
        let sent: BTreeSet<&'static str> = selectors()
            .rows
            .iter()
            .map(|row| row.view.call.selector())
            .collect();
        assert_eq!(sent.len(), 4, "not every view selector is sent: {sent:?}");
        let paired: BTreeSet<&'static str> = descriptors()
            .rows
            .iter()
            .map(|row| row.view.call.selector())
            .collect();
        assert_eq!(
            paired.len(),
            2,
            "the descriptors case does not send both the descriptor form and the five-argument \
             selector, so nothing says they agree: {paired:?}"
        );
    }

    /// The shared-handle table makes a texture that is shareable and views that are not, so
    /// `shareable` is recorded both ways.
    #[test]
    fn the_shared_handle_case_records_shareable_both_ways() {
        let case = shared_handles();
        assert!(case.asks_sharing, "no shared texture is made");
        assert!(
            !case.rows.is_empty(),
            "no view of the ordinary texture, so `shareable` is recorded for one object only"
        );
        assert!(
            matches!(shared_descriptor().storage, Storage::Private),
            "a shareable texture is private storage in Metal; any other is one it answers nil for"
        );
    }

    /// No two rows of a case share a site, and each case fingerprints its own table.
    #[test]
    fn every_row_names_its_own_site_and_every_case_its_own_inputs() {
        for case in all() {
            let sites: BTreeSet<String> = case.rows.iter().map(Row::site).collect();
            assert_eq!(
                sites.len(),
                case.rows.len(),
                "{}: two rows share a site",
                case.name
            );
        }
        let fingerprints: BTreeSet<Vec<String>> = all()
            .iter()
            .map(|case| {
                case.inputs()
                    .expect("the inputs are the case's own words")
                    .into_iter()
                    .map(|input| input.digest.hex)
                    .collect()
            })
            .collect();
        assert_eq!(
            fingerprints.len(),
            all().len(),
            "two cases fingerprint alike"
        );
    }
}
