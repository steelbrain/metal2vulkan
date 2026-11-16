//! What a view of a texture in a pixel format holds its texels in, answered before it is made
//! ([`Texture::view_placement`](crate::Texture::view_placement)): by the decision its making takes
//! (`Texture::view_hold`, `texture/view_shadow.rs`), the same function, so the answer and the
//! making never disagree.

/// What a view of a texture in a pixel format would be if made now, as
/// [`Texture::view_placement`](crate::Texture::view_placement) answers it. A runtime extension:
/// Metal has no such query, as Metal makes every view it admits of the texture's own memory; this
/// says how the runtime would make it (`texture/view_shadow.rs`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ViewPlacement {
    /// A Vulkan view of the texture's image (a texture buffer's: a texel buffer view of its
    /// bytes): the view shares the texture's storage, so its copies, bindings, attachments and
    /// CPU access are the image's, and its uses are ordered with every other view's of the image
    /// (the image's own uses). Of a texture that shadows a buffer's bytes, a view of that shadow.
    Image,
    /// The texture's image in the image's own format, presented through a swizzle over its
    /// channels (a view in other channels of the image's bits: an `A8Unorm` view of an `R8Unorm`
    /// texture; `WorkCounts::views_through_swizzle`): its bindings, copies and CPU access are the
    /// image's, as [`Image`](Self::Image)'s are, but a render pass renders into it through a view
    /// shadow of its own format, made at the first such pass and copied in before each command
    /// that uses it and back after one that writes it (announced, `shadowed-texture-view`).
    ImageThroughSwizzle,
    /// A view shadow: an image of the view's format in memory of its own, whose texels the
    /// encoders copy in from the texture's image before a command uses the view and back after
    /// one writes through it (announced once per device, `shadowed-texture-view`;
    /// `WorkCounts::view_shadows`). Why the image takes no Vulkan view of the format.
    Shadow(ViewShadowReason),
    /// Storage of its own over the bytes the texture is over, in the view's format, as a texture
    /// of that format over them is made (a texture over a buffer's bytes whose image takes no
    /// view of the format: an image aliasing the bytes, or a shadow of them; a two-plane YCbCr
    /// texture's `_sRGB` view, a converted image over the same planes; a rotated texture's view,
    /// the stored frame in the view's format). Refused, when it is made, as such a texture is.
    OverBytes,
}

/// Why a texture's image takes no Vulkan view of a format, so that the view is a shadow of it
/// ([`ViewPlacement::Shadow`]), as the view's notice (`shadowed-texture-view`) names it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ViewShadowReason {
    /// The device's Vulkan views reinterpret no image's components, and the view's format holds
    /// the image's in other components: a portability subset without
    /// `imageViewFormatReinterpretation`, or an adopted device whose consumer did not say it
    /// enabled it (`RawDevice::view_format_reinterpretation`).
    Reinterpretation,
    /// The view's Vulkan format and the image's are of different compatibility classes
    /// (`R10G10B10A2` of an `RGBA8Unorm` texture is of its class, `A8Unorm` of an `R8Unorm` one
    /// is not), which no view of the image takes.
    CompatibilityClass,
    /// The image lists no format of the view's: the texture was made without `PixelFormatView`
    /// usage, so its image takes views of its own format, its sRGB twin's and, where it renders,
    /// its channel-order sibling's and that sibling's twin's alone (kept so for the driver's
    /// compression).
    NotListed,
}
