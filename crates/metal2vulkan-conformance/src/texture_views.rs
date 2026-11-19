//! A view of a texture, as both sides ask for one, and what the view answers about its parent:
//! `MTLTexture`'s four view creators, `parentTexture`, `parentRelativeLevel`,
//! `parentRelativeSlice`, `swizzle`, `shareable` and `newSharedTextureHandle`.
//!
//! [`crate::texture::Over::View`] already makes a view, but only ever of `0..levels` and
//! `0..slices` through one of the four selectors — so every view the suite had made began at its
//! parent's own first level and slice, and the two relative getters could only ever answer 0. This
//! type states the **ranges**, and states which of the four calls makes the view, which is what
//! lets a case ask the questions Metal actually settles:
//!
//! - what `parentRelativeLevel` of a view **of a view** counts from: the immediate parent, or the
//!   root of the chain. The runtime models the immediate parent, and nothing until now compared
//!   that against Metal;
//! - what `swizzle` a view made by a call that takes none answers — the parent's, or the default;
//! - whether the one-argument `newTextureViewWithPixelFormat:` and the
//!   `textureType:levels:slices:` form agree on the whole texture;
//! - whether `newTextureViewWithDescriptor:` answers exactly what the five-argument selector does
//!   for the same ranges, which is the only way the suite reaches `MTLTextureViewDescriptor` at
//!   all.
//!
//! # Why every case over this type is safe
//!
//! A view holds no memory: the call is Metal's own bookkeeping over a texture that already exists,
//! it compiles nothing, allocates no texels and reaches no queue. The textures the views are made
//! of are small and uncompressed, and nothing here dispatches or draws.
//!
//! The hazard a view creator carries is a range outside the parent's, or a format of another size
//! class, which Metal's validation stops — ending its process. So [`ViewRequest::is_well_formed`]
//! decides that from the request's own shape and the parent's, a case's guard reaches it with no
//! device at all, and the Metal side calls it before it makes the call.

use crate::case::Answer;
use crate::transcript::{Node, NodeName, Observed, Value};
use metal2vulkan::metal::{PixelFormat, TextureSwizzle, TextureSwizzleChannels, TextureType};

/// Which of `MTLTexture`'s four view creators a request is made by.
///
/// Four variants and not one call with optional arguments, because the suite's business is what
/// each **selector** answers: a release where the one-argument form stops agreeing with the
/// four-argument one over the whole texture fails a case rather than passing under a rule nobody
/// re-measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewCall {
    /// `newTextureViewWithPixelFormat:`: the whole texture, every level and slice, as another
    /// format. It takes no ranges, so a request that names it must name the parent's own whole
    /// ranges ([`ViewRequest::is_well_formed`]).
    PixelFormat,
    /// `newTextureViewWithPixelFormat:textureType:levels:slices:`.
    LevelsSlices,
    /// `newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:`.
    Swizzled,
    /// `newTextureViewWithDescriptor:`, through an `MTLTextureViewDescriptor` holding exactly the
    /// same five values.
    Descriptor,
}

impl ViewCall {
    /// The selector this call sends, as a diff and a fingerprint name it.
    pub const fn selector(self) -> &'static str {
        match self {
            ViewCall::PixelFormat => "newTextureViewWithPixelFormat:",
            ViewCall::LevelsSlices => "newTextureViewWithPixelFormat:textureType:levels:slices:",
            ViewCall::Swizzled => {
                "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:"
            }
            ViewCall::Descriptor => "newTextureViewWithDescriptor:",
        }
    }

    /// The word the site's name carries.
    pub const fn word(self) -> &'static str {
        match self {
            ViewCall::PixelFormat => "format-only",
            ViewCall::LevelsSlices => "levels-slices",
            ViewCall::Swizzled => "swizzled",
            ViewCall::Descriptor => "descriptor",
        }
    }

    /// Whether this call states a swizzle of its own. The two that do not leave the view's
    /// `swizzle` to Metal, which is one of the facts a case records.
    pub const fn states_a_swizzle(self) -> bool {
        matches!(self, ViewCall::Swizzled | ViewCall::Descriptor)
    }
}

/// The shape of the texture a view is made of, as far as a view's rules read it: what
/// [`ViewRequest::is_well_formed`] needs to know about the parent.
///
/// Stated by the case rather than read off either side, because the check must run with no device
/// at all — that is the only proof a guard against ending Metal's process can have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParentShape {
    pub texture_type: TextureType,
    pub format: PixelFormat,
    /// `mipmapLevelCount`.
    pub levels: u64,
    /// The slices the parent holds: an array's `arrayLength`, a cube's six faces times its array
    /// length, 1 otherwise — as Metal counts a view's `sliceRange`.
    pub slices: u64,
    /// Whether the parent is itself a view, and so whether a request of it makes a view of a view.
    pub is_a_view: bool,
}

impl ParentShape {
    /// A 2D texture of `format`, `levels` levels, one slice.
    pub const fn texture_2d(format: PixelFormat, levels: u64) -> Self {
        Self {
            texture_type: TextureType::Type2D,
            format,
            levels,
            slices: 1,
            is_a_view: false,
        }
    }

    /// The same shape as a 2D array of `slices` slices.
    #[must_use]
    pub const fn array(mut self, slices: u64) -> Self {
        self.texture_type = TextureType::Type2DArray;
        self.slices = slices;
        self
    }

    /// The shape a view of this shape has: `levels` and `slices` long, of `texture_type` and
    /// `format`, and itself a view.
    pub const fn viewed(
        format: PixelFormat,
        texture_type: TextureType,
        levels: u64,
        slices: u64,
    ) -> Self {
        Self {
            texture_type,
            format,
            levels,
            slices,
            is_a_view: true,
        }
    }
}

/// One view a case asks for: the format, the texture type, the level and slice ranges counted from
/// the parent's own first, the swizzle where the call takes one, and which call makes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewRequest {
    pub call: ViewCall,
    pub format: PixelFormat,
    pub texture_type: TextureType,
    /// `levels.location` and `levels.length`, counted from the parent's own first level.
    pub first_level: u64,
    pub level_count: u64,
    /// `slices.location` and `slices.length`, counted from the parent's own first slice.
    pub first_slice: u64,
    pub slice_count: u64,
    /// The swizzle the two calls that take one state. Read only by those two; a request of the
    /// other two carries the default and the side never hands it over.
    pub swizzle: TextureSwizzleChannels,
}

impl ViewRequest {
    /// The whole of a parent of `shape` as `format`, through `call`.
    pub const fn whole(call: ViewCall, shape: &ParentShape, format: PixelFormat) -> Self {
        Self {
            call,
            format,
            texture_type: shape.texture_type,
            first_level: 0,
            level_count: shape.levels,
            first_slice: 0,
            slice_count: shape.slices,
            swizzle: TextureSwizzleChannels::DEFAULT,
        }
    }

    /// The same view over `first_level..first_level + level_count` of the parent's levels.
    #[must_use]
    pub const fn levels(mut self, first_level: u64, level_count: u64) -> Self {
        self.first_level = first_level;
        self.level_count = level_count;
        self
    }

    /// The same view over `first_slice..first_slice + slice_count` of the parent's slices, as
    /// `texture_type`.
    #[must_use]
    pub const fn slices(mut self, texture_type: TextureType, first: u64, count: u64) -> Self {
        self.texture_type = texture_type;
        self.first_slice = first;
        self.slice_count = count;
        self
    }

    /// The same view through `swizzle`.
    #[must_use]
    pub const fn with_swizzle(mut self, swizzle: TextureSwizzleChannels) -> Self {
        self.swizzle = swizzle;
        self
    }

    /// Why this request must never be handed to Metal, where it must not.
    ///
    /// One list, decided from the request's own shape and `parent`'s, and the only such list: the
    /// Metal side calls it in the one place that makes a view, so a row of a recorded table cannot
    /// reach Metal around it. A test reaches it with no device at all, which is the only proof such
    /// a guard can have.
    ///
    /// The shapes, each a failure of Metal's own validation and so a call that ends its process:
    ///
    /// - an **empty** level or slice range, or one that runs **past** the parent's;
    /// - a format that is not the parent's and not of the parent's own size class. A view's format
    ///   must be reinterpretable, and the only reinterpretation every case here needs is between
    ///   a format and itself, so the check is the strict one: the parent's format, or nothing;
    /// - a **texture type** the ranges do not fit: a non-array type asked for more than one slice,
    ///   and an array type where the suite has nothing measured about a one-slice array view;
    /// - a request through [`ViewCall::PixelFormat`] that is not the parent's whole ranges and
    ///   type, which that selector cannot state and so would silently be a different view from the
    ///   one the case wrote.
    pub fn is_well_formed(&self, parent: &ParentShape) -> Option<String> {
        if self.level_count == 0 || self.slice_count == 0 {
            return Some("an empty level or slice range is one Metal's validation stops".into());
        }
        if self.first_level + self.level_count > parent.levels {
            return Some(format!(
                "levels {}..{} run past the parent's {} levels",
                self.first_level,
                self.first_level + self.level_count,
                parent.levels
            ));
        }
        if self.first_slice + self.slice_count > parent.slices {
            return Some(format!(
                "slices {}..{} run past the parent's {} slices",
                self.first_slice,
                self.first_slice + self.slice_count,
                parent.slices
            ));
        }
        if self.format != parent.format {
            return Some(format!(
                "{} is not the parent's {}, and this family states no reinterpretation",
                self.format.name(),
                parent.format.name()
            ));
        }
        // Apple's `makeTextureView(pixelFormat:textureType:levels:slices:)`: "The length value of
        // the sliceRange parameter needs to be 6 if the new texture type value is
        // MTLTextureType.typeCube, or a multiple of 6 if the new texture type value is
        // MTLTextureType.typeCubeArray"; a type that is not an array holds one.
        let fits = match self.texture_type {
            TextureType::Type2DArray | TextureType::Type1DArray => true,
            TextureType::Cube => self.slice_count == 6,
            TextureType::CubeArray => self.slice_count.is_multiple_of(6),
            _ => self.slice_count == 1,
        };
        if !fits {
            return Some(format!(
                "{:?} holds no view of {} slices",
                self.texture_type, self.slice_count
            ));
        }
        if self.call == ViewCall::PixelFormat
            && (self.first_level != 0
                || self.level_count != parent.levels
                || self.first_slice != 0
                || self.slice_count != parent.slices
                || self.texture_type != parent.texture_type)
        {
            return Some(
                "newTextureViewWithPixelFormat: states no ranges and no type, so a request \
                 through it that is not the parent's whole shape would not be the view the case \
                 wrote"
                    .into(),
            );
        }
        None
    }

    /// The site this view's answers sit under: the call and every value it hands, so no two
    /// requests of a table share a site.
    pub fn site(&self) -> String {
        format!(
            "{}/{}/{:?}/levels-{}+{}/slices-{}+{}/{}",
            self.call.word(),
            self.format.name(),
            self.texture_type,
            self.first_level,
            self.level_count,
            self.first_slice,
            self.slice_count,
            swizzle_word(self.swizzle),
        )
    }

    /// The words the request fingerprints as: the selector and every value it hands that selector.
    ///
    /// The swizzle is written **only where the call states one**, so a request through a call that
    /// takes none fingerprints without it and a table cannot read as changed by a field its call
    /// never sends.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.call.selector().as_bytes().to_vec();
        out.extend(self.format.as_raw().to_le_bytes());
        out.extend(self.texture_type.as_raw().to_le_bytes());
        for word in [
            self.first_level,
            self.level_count,
            self.first_slice,
            self.slice_count,
        ] {
            out.extend(word.to_le_bytes());
        }
        if self.call.states_a_swizzle() {
            for channel in swizzle_channels(self.swizzle) {
                out.extend(channel.as_raw().to_le_bytes());
            }
        }
        out
    }
}

/// A swizzle's four channels in Metal's order.
pub fn swizzle_channels(swizzle: TextureSwizzleChannels) -> [TextureSwizzle; 4] {
    [swizzle.red, swizzle.green, swizzle.blue, swizzle.alpha]
}

/// The word a swizzle carries in a site's name: the four channels' raw ordinals, so a reader of a
/// diff sees which swizzle the row is of without decoding a number.
pub fn swizzle_word(swizzle: TextureSwizzleChannels) -> String {
    let channels = swizzle_channels(swizzle);
    format!(
        "swizzle-{}{}{}{}",
        channels[0].as_raw(),
        channels[1].as_raw(),
        channels[2].as_raw(),
        channels[3].as_raw(),
    )
}

/// The four raw `MTLTextureSwizzle` ordinals a case's swizzle names, in Metal's order.
pub fn swizzle_raw(swizzle: TextureSwizzleChannels) -> [u32; 4] {
    swizzle_channels(swizzle).map(TextureSwizzle::as_raw)
}

/// A swizzle as a site records it: the four channels by their raw `MTLTextureSwizzle`, so a diff
/// names which channel moved.
pub fn swizzle_value(swizzle: [u32; 4]) -> Value {
    Value::fields([
        ("red", Value::Uint(u64::from(swizzle[0]))),
        ("green", Value::Uint(u64::from(swizzle[1]))),
        ("blue", Value::Uint(u64::from(swizzle[2]))),
        ("alpha", Value::Uint(u64::from(swizzle[3]))),
    ])
}

/// What a texture or a view answers about its place in a view chain, and about sharing: the
/// parent's facts, which a runtime may leave to its consumer, and the view's own shape and
/// swizzle, each recorded at a site of its own so that either can be compared without the other.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewFacts {
    /// `parentTexture`, `parentRelativeLevel`, `parentRelativeSlice` and `shareable`, or why the
    /// side has none of them: a runtime whose consumer keeps a view's parent and what it shared
    /// answers the consumer's own failure at each of their four sites.
    pub parent: Answer<ViewParent>,
    /// `swizzle`, by its four raw `MTLTextureSwizzle` ordinals in Metal's order.
    ///
    /// Raw and not the translator's own enumeration, deliberately: that type is
    /// `#[non_exhaustive]` and has no constructor from a raw value, so a side mapping Metal's
    /// answer into it would have to **coerce** a channel Metal has a name for and the enumeration
    /// does not. The ordinals are Metal's own, which is the suite's convention everywhere else.
    pub swizzle: [u32; 4],
    /// `textureType`, `pixelFormat`, `mipmapLevelCount` and `arrayLength` of the view, so a
    /// recording states the shape the relative getters are relative to rather than leaving a
    /// reader to find this module.
    pub texture_type: u64,
    pub pixel_format: u64,
    pub mipmap_level_count: u64,
    pub array_length: u64,
}

/// What a texture or a view answers about its parent and about sharing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewParent {
    /// Whether `parentTexture` answered a texture at all. A texture that is not a view answers
    /// nil, which is the baseline every other row is read against.
    pub has_parent: bool,
    /// Whether the texture `parentTexture` answered is the one the case made the view of.
    ///
    /// A Bool and not the object: a recorded message holds no address, and the fact under test is
    /// the relation. `false` for a texture with no parent, so the two fields read together.
    pub expected_parent: bool,
    /// `parentRelativeLevel`.
    pub parent_relative_level: u64,
    /// `parentRelativeSlice`.
    pub parent_relative_slice: u64,
    /// `shareable`.
    pub shareable: bool,
}

impl ViewFacts {
    /// The facts as the sites a case emits under one node, in the order the recordings hold
    /// them; each of the parent's four getters (five sites) holds the side's failure where it
    /// answered none.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let number = |value: u64| Observed::value(Value::Uint(value));
        let parent = |of: fn(&ViewParent) -> Observed| match &self.parent {
            Ok(parent) => of(parent),
            Err(failure) => Observed::value(failure.recorded()),
        };
        Node::branch(name, Vec::new())
            .with(Node::leaf(
                "has_parent",
                parent(|p| Observed::value(Value::Bool(p.has_parent))),
            ))
            .with(Node::leaf(
                "expected_parent",
                parent(|p| Observed::value(Value::Bool(p.expected_parent))),
            ))
            .with(Node::leaf(
                "parent_relative_level",
                parent(|p| Observed::value(Value::Uint(p.parent_relative_level))),
            ))
            .with(Node::leaf(
                "parent_relative_slice",
                parent(|p| Observed::value(Value::Uint(p.parent_relative_slice))),
            ))
            .with(Node::leaf(
                "swizzle",
                Observed::value(swizzle_value(self.swizzle)),
            ))
            .with(Node::leaf(
                "shareable",
                parent(|p| Observed::value(Value::Bool(p.shareable))),
            ))
            .with(Node::leaf("texture_type", number(self.texture_type)))
            .with(Node::leaf("pixel_format", number(self.pixel_format)))
            .with(Node::leaf(
                "mipmap_level_count",
                number(self.mipmap_level_count),
            ))
            .with(Node::leaf("array_length", number(self.array_length)))
    }
}

/// What a shared texture's handle answers: `MTLSharedTextureHandle`'s own surface, and the texture
/// Metal opens from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedHandleFacts {
    /// Whether `device` of the handle has the `registryID` of the device that made the texture
    /// ([`crate::heap_query::HeapOwnFacts::own_device`] says why it is a Bool and why it is the
    /// registry ID).
    pub own_device: bool,
    /// Whether `newSharedTextureWithHandle:` answered a texture.
    pub opens: bool,
    /// The opened texture's `shareable`: whether a texture opened from a handle is itself
    /// shareable, which is Metal's to settle.
    pub opened_shareable: bool,
    /// The opened texture's `width`, `height` and `mipmapLevelCount`, so a recording says the
    /// handle opened the texture it was made of rather than some other.
    pub opened_width: u64,
    pub opened_height: u64,
    pub opened_levels: u64,
}

impl SharedHandleFacts {
    /// The facts as the sites a case emits under one node.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let number = |value: u64| Observed::value(Value::Uint(value));
        let flag = |held: bool| Observed::value(Value::Bool(held));
        Node::branch(name, Vec::new())
            .with(Node::leaf("own_device", flag(self.own_device)))
            .with(Node::leaf("opens", flag(self.opens)))
            .with(Node::leaf("opened_shareable", flag(self.opened_shareable)))
            .with(Node::leaf("opened_width", number(self.opened_width)))
            .with(Node::leaf("opened_height", number(self.opened_height)))
            .with(Node::leaf("opened_levels", number(self.opened_levels)))
    }
}
