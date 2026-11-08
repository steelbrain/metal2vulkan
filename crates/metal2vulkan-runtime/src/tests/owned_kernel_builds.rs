//! Where the build of one of the runtime's own kernels falls, counted.
//!
//! A format the runtime emulates needs kernels of its own, and a kernel is built by the Metal
//! call that first needs it and then kept for the device (`owned_kernel.rs`). So the cost of the
//! build lands inside one call and inside no other, and nothing about that call says it: Metal
//! builds no such kernel, and a program that times its calls sees one of them cost far more than
//! Metal's for reasons the call itself cannot explain.
//!
//! `WorkCounts::owned_kernels_made` and `owned_kernel_spirv_bytes` name it. Here two two-plane
//! YCbCr textures over a surface's planes, whose first conversion needs the conversion's kernels
//! (`ycbcr.rs`, `texture/ycbcr_image.rs`), are each read: the making builds none, the first
//! texture's first conversion builds them, and no later one builds any. The counter is what any
//! call increments where it is the call that first needs a kernel, so the sites can be told apart
//! in one recording.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch};
use super::ycbcr_in_place_loop::{read_with, reader};
use crate::{
    Device, Error, IoSurfacePlane, IoSurfaceYCbCr, ResourceOptions, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The surface's size, as a stream of such a texture holds it.
const W: u64 = 3840;
const H: u64 = 2160;

/// The luma plane's offset and rows, and the chroma plane's: 10-bit codes, 2 bytes each, the
/// chroma plane after the luma plane and half its height.
const LUMA: (u64, u64) = (0, 2 * W);
const CHROMA: (u64, u64) = (2 * W * H, 2 * W);

/// The surface's bytes.
const LENGTH: usize = (CHROMA.0 + CHROMA.1 * H / 2) as usize;

/// The texels a conversion's read reads, from the origin: its command converts the whole image.
const GRID: [u64; 2] = [8, 8];

/// The kernels built and the SPIR-V bytes they compiled while `run` ran, and the conversion
/// scratch buffers it took and their bytes.
fn built(device: &Device, run: impl FnOnce()) -> [u64; 4] {
    let before = device.work_counts();
    run();
    let after = device.work_counts();
    [
        after.owned_kernels_made - before.owned_kernels_made,
        after.owned_kernel_spirv_bytes - before.owned_kernel_spirv_bytes,
        after.ycbcr_conversion_scratch_buffers - before.ycbcr_conversion_scratch_buffers,
        after.ycbcr_conversion_scratch_bytes - before.ycbcr_conversion_scratch_bytes,
    ]
}

/// A two-plane YCbCr texture over the `x420` surface's planes, as the suite's cases make one.
fn plane_texture(buffer: &crate::Buffer) -> Result<Texture, Error> {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::YCBCR104202P, W, H, false);
    descriptor.resource_options = ResourceOptions::shared();
    descriptor.usage = TextureUsage::SHADER_READ;
    let facts = IoSurfaceYCbCr::new(u32::from_be_bytes(*b"x420"), 0, [W, H], CHROMA.0, CHROMA.1);
    let plane = IoSurfacePlane::new(LUMA.0, LUMA.1, true).with_ycbcr(facts);
    buffer.new_texture_with_iosurface_plane(&descriptor, plane)
}

/// The making of such a texture builds no kernel and takes no scratch: its conversion is made at
/// its first conversion (`ycbcr.rs`), which a `read()` of it takes (it reads the converted image).
/// That conversion builds the kernels and takes the texture's scratch, and the next builds and
/// takes none; a second texture over the same surface builds none at its first conversion and
/// takes scratch of its own. So the count says which call paid for them.
#[test]
fn an_emulated_formats_kernels_are_built_by_the_call_that_first_needs_them() {
    let scratch = Scratch::new("owned-kernel-builds");
    let device = device(&scratch.0);
    let bytes = vec![0x40; LENGTH];
    let Ok(buffer) = device.new_buffer_with_bytes(&bytes, ResourceOptions::shared()) else {
        eprintln!("{}: no buffer of the surface's bytes", device.name());
        return;
    };
    let reader = reader(&device);
    let mut first = None;
    let making = built(&device, || first = Some(plane_texture(&buffer)));
    let texture = match first {
        Some(Ok(texture)) => texture,
        made => {
            eprintln!("{}: no `x420` plane texture: {made:?}", device.name());
            return;
        }
    };
    let converted = built(&device, || read_with(&device, &reader, &texture, GRID));
    let again = built(&device, || read_with(&device, &reader, &texture, GRID));
    let Ok(other) = plane_texture(&buffer) else {
        panic!("a second `x420` plane texture over the surface was refused");
    };
    let other = built(&device, || read_with(&device, &reader, &other, GRID));
    eprintln!(
        "{}: [kernels, SPIR-V bytes, scratch buffers, scratch bytes]: the making {making:?}, \
         the first conversion {converted:?}, the next {again:?}, a second texture's first {other:?}",
        device.name()
    );
    assert_eq!(
        making, [0; 4],
        "the making of the texture built kernels or took scratch, which its first conversion makes"
    );
    let [kernels, words, scratch, scratch_bytes] = converted;
    assert!(
        kernels > 0 && words > 0,
        "the first conversion of a texture of an emulated format built no kernel of the runtime's \
         own, so the counter no longer names where the build falls"
    );
    assert!(
        scratch > 0 && scratch_bytes >= LENGTH as u64,
        "the first conversion took no scratch of the planes' byte range"
    );
    assert_eq!(
        again, [0; 4],
        "the texture's next conversion built kernels or took scratch again"
    );
    // The kernels are the device's, the scratch is the texture's: the second takes its own.
    assert_eq!(
        other,
        [0, 0, scratch, scratch_bytes],
        "a second texture's first conversion built kernels again, so they are not kept for the \
         device, or took other scratch, so the count is not per texture"
    );
}
