//! A picture's Direct3D 11 textures as a surface ([`Device::import_d3d11_surface`]), a Windows
//! host's: the planes another API decodes into, shared by their handles. Refused for now.

use crate::device::Device;
use crate::error::{Error, Gap};
use crate::surface::Surface;
use crate::texture::TextureDescriptor;
use crate::PixelFormat;
use std::os::windows::io::BorrowedHandle;

/// No Metal counterpart: one plane of a picture another API made as Direct3D 11 textures
/// ([`Device::import_d3d11_surface`]): the shared texture's handle and its own format, and the
/// texture rt makes over it. Metal's counterpart of the whole is an IOSurface's plane under a
/// texture (`newTextureWithDescriptor:iosurface:plane:`).
#[derive(Clone, Copy, Debug)]
pub struct D3d11Plane<'a> {
    /// The shared texture's NT handle, as Vulkan imports a Direct3D 11 texture
    /// (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT`). The consumer keeps it.
    pub handle: BorrowedHandle<'a>,
    /// The Direct3D 11 texture's own format: `R8Unorm` or `R16Unorm` for a luma plane,
    /// `R16Uint` or `R32Uint` for a chroma plane, which the texture views as two channels.
    pub image_format: PixelFormat,
    /// The texture rt makes over the plane, as [`PlaneSpec::descriptor`](crate::PlaneSpec):
    /// its pixel format is the view's (`R8Unorm`, `RG8Unorm`, `R16Unorm`, `RG16Unorm`).
    pub descriptor: TextureDescriptor,
    /// The texture's index, as a [`PlaneSpec`](crate::PlaneSpec)'s: with none, the plane is
    /// reached through its views alone.
    pub index: Option<u32>,
}

impl Device {
    /// No Metal counterpart of the call: a picture's planes, Direct3D 11 textures another API
    /// made (a video decoder's), as a surface whose planes are textures rt's shaders read, as
    /// Metal reads an IOSurface's plane through a texture over it
    /// (`newTextureWithDescriptor:iosurface:plane:`). Refused for now: the import is not built,
    /// and every call answers [`Gap::D3d11Surface`].
    ///
    /// Err: [`Gap::D3d11Surface`].
    pub fn import_d3d11_surface(&self, _planes: &[D3d11Plane<'_>]) -> Result<Surface, Error> {
        Err(Error::Unsupported(Gap::D3d11Surface))
    }
}
