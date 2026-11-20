//! Where a rotated framebuffer-only texture's texels lie: the layout a case states, and the two
//! rules that decide where a texel lands and which strides are safe to ask for.
//!
//! # Why this is its own module, and holds no IOSurface
//!
//! Everything here is **portable**, and deliberately: a case states its plane, a recording holds
//! those words, and a Metal-less venue replays it. The surface itself, its CoreFoundation and
//! IOSurface entry points and the handle that owns one all live in `crate::metal_surface` (private
//! to the suite, so not linked here), which is the only module that makes an `IOSurfaceRef` — see
//! its header for why the YCbCr plane textures and the rotated targets share one mechanism there.
//!
//! The runtime models a rotated texture over a **buffer standing for** an IOSurface plane
//! (`Buffer::new_texture_with_iosurface_plane`), because a Metal-less host has no IOSurface. Metal
//! has no such stand-in: `framebufferOnly` over anything but an IOSurface plane makes it assert, so
//! the Metal leg of a rotated case needs an actual surface, which is why the two sides of this
//! family read the same [`PlaneSpec`] and build different things from it.
//!
//! # The private setters
//!
//! `MTLTextureDescriptor`'s `setRotation:` and `setFramebufferOnly:` are Apple SPI, reached by
//! selector and **only after `respondsToSelector:`** (`MetalSide::require_selector`), so a release
//! that stops answering them gives a typed refusal naming the selector rather than an
//! unrecognised-selector crash. Both were verified answered on an Apple M3 Max under macOS 27.0.1
//! (26A434), where the descriptor also reads the rotation back.
//!
//! # The stride rule, which is a safety rule
//!
//! **Metal asserts, ending the process, on an IOSurface texture whose `bytesPerRow` is below the
//! texture's `width` times its bytes per element** — not the *surface's* width, the texture's,
//! whatever the rotation. Measured here the hard way, by a probe that died:
//!
//! ```text
//! _mtlValidateStrideTextureParameters:1938: failed assertion `Texture Descriptor Validation
//! IOSurface texture: bytesPerRow (16) must be greater or equal to (32) bytes'
//! ```
//!
//! for a 4x8 surface at its own tight stride of 16 behind an 8x4 texture. So [`PlaneSpec::unsafe_stride`]
//! is checked before any descriptor reaches Metal, and a case whose plane would breach it is
//! refused rather than run. This is the suite's no-ending-the-process rule made mechanical for this
//! family.

/// Where a rotated framebuffer-only texture's texels live: the plane's extent, its stride, the
/// rotation the descriptor's private setter takes, and the byte the plane starts as.
///
/// Nothing here names a device or a vendor. A side that cannot make such a texture answers its own
/// typed refusal, which is how the suite learns the venue's honest gap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneSpec {
    /// `setRotation:` (Apple SPI): 0, or 1 to 4. A pass draws the frame unturned and stores it
    /// turned; 2 is a half turn, 1 and 4 one quarter turn and 3 the other.
    pub rotation: u32,
    /// The plane's extent in texels. For a quarter turn that is the frame's extent turned around,
    /// which is where the stored frame lands.
    pub plane: [u32; 2],
    /// The plane's `bytesPerRow`.
    pub row_pitch: u32,
    /// Bytes a texel of the texture's format, which is also the surface's
    /// `kIOSurfaceBytesPerElement`. The case states it, as it states the stride, because both are
    /// part of what the case means; this family's formats are all four bytes.
    pub texel: u32,
    /// The byte every plane byte holds before the case writes anything, so a byte the pass did not
    /// touch is visibly untouched rather than zero.
    pub pad: u8,
}

impl PlaneSpec {
    /// A plane of `plane` texels of `texel` bytes at `row_pitch` bytes a row, for a texture of
    /// `rotation`, padded with `pad`.
    pub const fn new(rotation: u32, plane: [u32; 2], row_pitch: u32, texel: u32, pad: u8) -> Self {
        Self {
            rotation,
            plane,
            row_pitch,
            texel,
            pad,
        }
    }

    /// The plane's whole length in bytes: one stride a row.
    pub const fn length(&self) -> usize {
        (self.row_pitch as usize) * (self.plane[1] as usize)
    }

    /// **Whether handing this plane to Metal would end the process**: the stride is below the
    /// *texture's* own row, which Metal's descriptor validation asserts on
    /// ([the module's header](self)).
    ///
    /// `frame` is the texture's extent. A case checks this before it ever asks Metal, and **both
    /// sides check it again at the call**, because the rule is a safety rule and one guard in a
    /// case body is one `if` away from being lost.
    pub const fn unsafe_stride(&self, frame: [u32; 2]) -> bool {
        self.row_pitch < frame[0] * self.texel
    }

    /// The words a case's input fingerprint covers: every field, so a plane that moves is a change
    /// to the case's inputs rather than a quiet pass against another plane's answer.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = self.rotation.to_le_bytes().to_vec();
        for axis in self.plane {
            bytes.extend(axis.to_le_bytes());
        }
        bytes.extend(self.row_pitch.to_le_bytes());
        bytes.extend(self.texel.to_le_bytes());
        bytes.push(self.pad);
        bytes
    }
}

/// Where the store of a `frame`-sized pass lands in the plane of a texture of `rotation`: the
/// plane's own extent.
///
/// **Measured on the oracle**, not carried: an Apple M3 Max under macOS 27.0.1 stores a quarter
/// turn's frame into a plane of the frame's height by its width
/// (`probes/rotated_surface_planes.m`).
pub const fn plane_extent(rotation: u32, frame: [u32; 2]) -> [u32; 2] {
    match rotation {
        1 | 3 | 4 => [frame[1], frame[0]],
        _ => frame,
    }
}
